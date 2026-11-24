"""Screener tone readout: news screener pass rates by headline type, per
screener prompt version (audit 2026-11-22 02:56 #3). Read-only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/screener_tone_readout.py [--since 2026-11-01] [--env live] \
        [--log logs/celery_screen_worker.log.1 --log logs/celery_screen_worker.log] [--out FILE]

It re-measures two things on the prompt in force:
  - upgrade/raise vs downgrade/cut headlines (A1853-1's "good news passes 4x
    more"; audit 02:56 found the reverse);
  - executive-departure and offering/dilution headlines (02:56: 21% of 102,
    and 1 of 22).
The four categories use the audit's own title lexicon (scr.py), printed
in the output with its sha. A title can match more than one category.

Rows: the screener's SCREENED_IN / SCREENED_OUT signals for news items
(source "news"), grouped by provenance.prompt_template_sha (the screener
prompt's hash, on every row since 10-28). "Passed" is SCREENED_IN; "7B passed"
leaves out passes made by the held-negative override (provenance.override).
"Title names ticker" is the row's recorded provenance.title_names_ticker
(the news fetcher's mentions_company test, as of the screen).

The DB rows carry the title from 2026-11-10 only, and SCREENED_IN rows from
11-05. With --log, ET days whose DB rows lack titles are read from the screen
worker log instead (the audit's source): those titles are cut to 60
characters, the prompt version is the DB rows' sha for that day, and
title_names_ticker is computed now, with the current watchlist. Log and DB
rows are reported apart, never added together. Without --log those days are
listed as not measured.

Counts and rates only: no prices or account data. Which rate counts as
asymmetric, and whether departures or offerings join the screener's
relevance list, are not this script's call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")
STATEMENT_TIMEOUT_MS = 60000
LOG_TITLE_CHARS = 60
TOP_TICKERS = 3   # tickers with the most named rows per category, so one story shows

# The audit's lexicon (audit 2026-11-22 02:56, scr.py), unchanged so the
# figures compare with its 13% / 4% / 21% / 1-of-22.
CATEGORIES = [
    ("upgrade_raise", re.compile(
        r"\b(upgrade[sd]?|raises? (price )?target|to (buy|outperform|overweight))\b", re.I)),
    ("downgrade_cut", re.compile(
        r"\b(downgrade[sd]?|cuts? (price )?target|lowers? (price )?target|"
        r"to (sell|underperform|underweight))\b", re.I)),
    ("exec_departure", re.compile(
        r"\b(CEO|CFO|COO|CTO|chief executive|chief financial|president)\b.*\b(exit|exits|resign|"
        r"resigns|steps down|stepping down|depart|departs|departure|ousted|leaves|to leave|retire|"
        r"retires|shake-?up)|\b(exit|exits|resigns|steps down|departure|shake-?up)\b.*\b(CEO|CFO|chief)",
        re.I)),
    ("offering_dilution", re.compile(
        r"\b(offering|stock sale|share sale|dilution|dilutive|convertible|ATM program|"
        r"at-the-market|raises \$[\d.]+ ?(billion|million) in)\b", re.I)),
]
LEXICON_SHA = hashlib.sha256(json.dumps(
    [[n, p.pattern, int(p.flags)] for n, p in CATEGORIES]).encode()).hexdigest()[:12]

_SQL_ROWS = (
    "SELECT timestamp, symbol, signal, provenance->>'prompt_template_sha', "
    "provenance->>'override', provenance->>'title_names_ticker', decision->>'title', "
    "decision->>'item_id' FROM signals WHERE trading_env = %(env)s "
    "AND signal IN ('SCREENED_IN', 'SCREENED_OUT') AND timestamp >= %(since)s "
    "AND coalesce(provenance->>'source', decision->>'source') = 'news'")
# Every screener row, any source, per ET day: the prompt sha(s) in force and
# whether the day's rows carry titles (which decides DB vs log for that day).
_SQL_DAYS = (
    "SELECT (timestamp AT TIME ZONE 'America/New_York')::date, "
    "array_agg(DISTINCT provenance->>'prompt_template_sha'), count(*), "
    "count(*) FILTER (WHERE decision->>'title' IS NOT NULL), "
    "count(*) FILTER (WHERE signal = 'SCREENED_IN') FROM signals "
    "WHERE trading_env = %(env)s AND signal IN ('SCREENED_IN', 'SCREENED_OUT') "
    "AND timestamp >= %(since)s GROUP BY 1 ORDER BY 1")


def categories(title: str | None) -> list[str]:
    return [name for name, rx in CATEGORIES if rx.search(title or "")]


def _bool(v):
    return True if v in (True, "true") else False if v in (False, "false") else None


def day_table(day_rows) -> dict:
    """{ET date: {"shas": [...], "rows", "titled", "passed"}} from _SQL_DAYS."""
    out = {}
    for d, shas, n, titled, passed in day_rows:
        out[d] = {"shas": sorted(s for s in (shas or []) if s), "rows": n,
                  "titled": titled, "passed": passed}
    return out


def db_days(days: dict) -> set:
    """ET days measured from the DB: every screener row that day carries its
    title, so the lexicon has full titles to read."""
    return {d for d, v in days.items() if v["rows"] and v["titled"] == v["rows"]}


def from_db(rows, measured_days: set) -> list[dict]:
    out = []
    for ts, ticker, signal, sha, override, names, title, item_id in rows:
        d = ts.astimezone(ET).date()
        if d not in measured_days:
            continue
        out.append({"day": d, "sha": sha or "unknown", "ticker": ticker, "title": title,
                    "item": item_id or title, "passed": signal == "SCREENED_IN",
                    "override": bool(override), "names_ticker": _bool(names), "origin": "db"})
    return out


_LOG_EVENT = re.compile(r"^\[(\d{4}-\d\d-\d\d) [\d:,]+: INFO/(ForkPoolWorker-\d+)\] "
                        r"\[([A-Z.]+)\] Screening event: (\w+) — (.*)$")
_LOG_OVERRIDE = re.compile(r"^\[[\d\- :,]+: INFO/(ForkPoolWorker-\d+)\] \[([A-Z.]+)\] Screener override:")
_LOG_RESULT = re.compile(r"^\[(\d{4}-\d\d-\d\d) [\d:,]+: INFO/(ForkPoolWorker-\d+)\] "
                         r"Screened \[(\w+)\] ([A-Z.]+): relevant=(\w+)")


def parse_log(lines, since: date, keep_days, names_ticker=None) -> list[dict]:
    """News screens from screen-worker log lines on keep_days (ET dates; the
    log's clock is ET). A worker screens one event at a time, so each result
    line pairs with that worker's last "Screening event" line for the same
    ticker. names_ticker(ticker, title) gives title_names_ticker (computed
    now); the default is agents.screener._title_names_ticker."""
    if names_ticker is None:
        from agents.screener import _title_names_ticker as names_ticker
    pending, out = {}, []
    for line in lines:
        m = _LOG_EVENT.match(line)
        if m:
            d, worker, ticker, source, title = m.groups()
            pending[worker] = {"ticker": ticker, "source": source, "title": title.rstrip(),
                               "override": False}
            continue
        m = _LOG_OVERRIDE.match(line)
        if m:
            p = pending.get(m.group(1))
            if p and p["ticker"] == m.group(2):
                p["override"] = True
            continue
        m = _LOG_RESULT.match(line)
        if not m:
            continue
        d, worker, source, ticker, relevant = m.groups()
        p = pending.pop(worker, None)
        day = date.fromisoformat(d)
        if not p or p["ticker"] != ticker or source != "news" or p["source"] != "news":
            continue
        if day < since or day not in keep_days:
            continue
        out.append({"day": day, "sha": None, "ticker": ticker, "title": p["title"],
                    "item": p["title"], "passed": relevant == "True",
                    "override": p["override"], "names_ticker": names_ticker(ticker, p["title"]),
                    "origin": "log"})
    return out


def assign_log_shas(rows, days: dict) -> None:
    """A log row's prompt version is the DB rows' sha that ET day ("mixed"
    when the day had two, "unknown" when it had none)."""
    for r in rows:
        shas = (days.get(r["day"]) or {}).get("shas") or []
        r["sha"] = shas[0] if len(shas) == 1 else ("mixed" if shas else "unknown")


def tally(rows) -> dict:
    """{(origin, sha): {"window", "all", "named", "unknown_named", "dupes",
    "cats": {cat: {"all": Counter, "named": Counter, "tickers": Counter}}}}.
    "all"/"named" hold n, passed and passed_7b; "tickers" counts named rows
    by ticker."""
    groups = {}
    for r in rows:
        g = groups.setdefault((r["origin"], r["sha"]), {
            "days": set(), "all": Counter(), "named": Counter(), "unknown_named": 0,
            "items": Counter(),
            "cats": defaultdict(lambda: {"all": Counter(), "named": Counter(), "tickers": Counter()})})
        g["days"].add(r["day"])
        c = Counter(n=1, passed=int(r["passed"]), passed_7b=int(r["passed"] and not r["override"]))
        g["all"].update(c)
        named = r["names_ticker"] is True
        if r["names_ticker"] is None:
            g["unknown_named"] += 1
        if named:
            g["named"].update(c)
            g["items"][(r["ticker"], r["item"])] += 1
        for cat in categories(r["title"]):
            g["cats"][cat]["all"].update(c)
            if named:
                g["cats"][cat]["named"].update(c)
                g["cats"][cat]["tickers"][r["ticker"]] += 1
    for g in groups.values():
        g["dupes"] = sum(n - 1 for n in g["items"].values() if n > 1)
        g["window"] = (min(g["days"]), max(g["days"])) if g["days"] else (None, None)
    return groups


def _rate(c: Counter, key: str) -> str:
    return f"{c[key]} ({c[key] / c['n']:.0%})" if c["n"] else "0 (-)"


def render(groups: dict, days: dict, measured_db: set, log_days: set, env: str, since: date,
           head: str, log_paths) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Screener tone readout ({env}, news, since {since})", "",
             f"Generated {now} at code {head}. Counts only. Lexicon sha {LEXICON_SHA} "
             "(the audit 2026-11-22 02:56 scr.py patterns).", "",
             "Passed = SCREENED_IN. 7B passed = passed, less the held-negative override. "
             "Title names ticker = provenance.title_names_ticker (DB rows, as recorded) or "
             f"computed now on the {LOG_TITLE_CHARS}-char log title (log rows).", ""]
    lines += ["## Days and sources", "", "| ET day | prompt sha | screener rows | with title | "
              "SCREENED_IN rows | measured from |", "|---|---|---|---|---|---|"]
    for d in sorted(days):
        v = days[d]
        src = "DB" if d in measured_db else ("log" if d in log_days else "not measured")
        lines.append(f"| {d} | {', '.join(v['shas']) or '-'} | {v['rows']} | {v['titled']} | "
                     f"{v['passed']} | {src} |")
    if log_paths:
        lines += ["", "Log files: " + ", ".join(log_paths)]
    for (origin, sha), g in sorted(groups.items(), key=lambda kv: (kv[1]["window"][0], kv[0])):
        lo, hi = g["window"]
        lines += ["", f"## Prompt {sha} ({origin.upper()} rows, {lo} to {hi}, {len(g['days'])} days)", "",
                  f"All news screens: n={g['all']['n']}, passed {_rate(g['all'], 'passed')}, "
                  f"7B passed {_rate(g['all'], 'passed_7b')}.",
                  f"Title names ticker: n={g['named']['n']}, passed {_rate(g['named'], 'passed')}, "
                  f"7B passed {_rate(g['named'], 'passed_7b')}; names_ticker unknown on "
                  f"{g['unknown_named']} rows; {g['dupes']} repeat screens of a (ticker, item) "
                  "among the named rows (counted as rows).", "",
                  "| category | title names ticker: n | passed | 7B passed | all titles: n | passed "
                  "| 7B passed | most named rows |", "|---|---|---|---|---|---|---|---|"]
        for cat, _ in CATEGORIES:
            c = g["cats"].get(cat) or {"all": Counter(), "named": Counter(), "tickers": Counter()}
            top = ", ".join(f"{t} {n}" for t, n in c["tickers"].most_common(TOP_TICKERS)) or "-"
            lines.append(f"| {cat} | {c['named']['n']} | {_rate(c['named'], 'passed')} | "
                         f"{_rate(c['named'], 'passed_7b')} | {c['all']['n']} | "
                         f"{_rate(c['all'], 'passed')} | {_rate(c['all'], 'passed_7b')} | {top} |")
    lines += ["", "## Lexicon", ""]
    lines += [f"- `{name}`: `{rx.pattern}` (case-insensitive)" for name, rx in CATEGORIES]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-01", help="ET date, inclusive")
    ap.add_argument("--env", default="live")
    ap.add_argument("--log", action="append", default=[],
                    help="screen worker log file(s), oldest first, for days without DB titles")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since = date.fromisoformat(args.since)
    since_ts = datetime(since.year, since.month, since.day, tzinfo=ET)

    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
            params = {"env": args.env, "since": since_ts}
            cur.execute(_SQL_DAYS, params)
            days = day_table(cur.fetchall())
            cur.execute(_SQL_ROWS, params)
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    measured_db = db_days(days)
    rows = from_db(raw, measured_db)
    log_days = set()
    if args.log:
        log_days = {d for d in days if d not in measured_db}
        log_rows = []
        for path in args.log:
            with open(path, errors="replace") as f:
                log_rows += parse_log(f, since, log_days)
        assign_log_shas(log_rows, days)
        rows += log_rows
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = render(tally(rows), days, measured_db, log_days, args.env, since, head or "unknown",
                  args.log)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
