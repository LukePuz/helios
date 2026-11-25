"""Analyst verdict flips readout, per ticker per New York day.

Engineering diagnostic (pilot phase), not a study analysis. A verdict is a
signals row that carries the analyst's decision record (decision has
final_action: HOLD, WATCH, ADD, OPEN, PASS, ...); routing and shadow rows
(ENTRY_GATE_BLOCKED, *_SHADOW, SCREENED_*, ...) and degraded rows written
without a decision are not verdicts. Within one ticker and one New York
calendar day, verdicts are ordered by (timestamp, id) and each pair of
consecutive verdicts whose stored signal differs is a flip. Pairs that span
two days are not compared.

For each flip:
  - the gap in minutes between the two rows' timestamps;
  - conviction (decision.final_conviction, else confidence * 10) and UDR
    (decision.udr_check.model) of both calls, and whether each changed
    ("?" when either call lacks it);
  - whether both calls came from the same event: provenance.trigger's
    story_dedup.story_key when both have one, else the trigger url, else
    the trigger title; "?" when either call has none (scheduled reviews
    and screener paths carry no event);
  - the two trigger paths (event, scheduled_review, ...).
Per day: verdicts, flips, flips with a gap of at most 5 and at most 30
minutes, and how many flips kept conviction, kept UDR, or shared an event.
With --by-path: verdicts per trigger path (provenance.trigger.path: event,
scheduled_review, daily_scan for the screener, earnings_review; "none" for
rows written before trigger provenance existed) and flips per pair of paths
(from the earlier call's path to the later one's), with the same tallies.
Counts only: no thresholds, no prices, nothing recommended.

Rows are filtered on trading_env, so rows from before 2026-10-28, whose
trading_env is NULL, are not read (see audit-method-notes.md).

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/verdict_flip_readout.py [--since YYYY-MM-DD] [--until YYYY-MM-DD] \
        [--ticker NVDA] [--no-detail] [--by-path] [--out FILE]

Read-only: one transaction opened READ ONLY, then rolled back.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

NY = ZoneInfo("America/New_York")
GAPS_MIN = (5, 30)


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _aware(t):
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def verdict(row: dict) -> dict:
    """One signals row (id, symbol, timestamp, signal, confidence, decision,
    trigger) normalised to {id, ticker, at, day, signal, model_action,
    conviction, udr, path, event_key, event_basis}."""
    dec = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    trig = row.get("trigger") if isinstance(row.get("trigger"), dict) else {}
    conv = _num(dec.get("final_conviction"))
    if conv is None:
        c = _num(row.get("confidence"))
        conv = round(c * 10, 1) if c is not None else None
    udr_check = dec.get("udr_check") if isinstance(dec.get("udr_check"), dict) else {}
    dedup = trig.get("story_dedup") if isinstance(trig.get("story_dedup"), dict) else {}
    key, basis = None, None
    for b, v in (("story_key", dedup.get("story_key")), ("url", trig.get("url")),
                 ("title", trig.get("title"))):
        if v:
            key, basis = str(v), b
            break
    at = _aware(row["timestamp"])
    return {"id": row["id"], "ticker": row["symbol"], "at": at, "day": at.astimezone(NY).date(),
            "signal": row["signal"], "model_action": dec.get("model_action"),
            "conviction": conv, "udr": _num(udr_check.get("model")),
            "path": trig.get("path"), "event_key": key, "event_basis": basis}


def _changed(a, b):
    return None if a is None or b is None else a != b


def _same_event(a: dict, b: dict):
    """True/False when both calls name an event on a shared basis, else None."""
    if not a["event_key"] or not b["event_key"]:
        return None
    if a["event_basis"] == b["event_basis"]:
        return a["event_key"] == b["event_key"]
    return None


def group(verdicts: list[dict]) -> dict:
    """{(day, ticker): [verdicts ordered by (at, id)]}."""
    out: dict = {}
    for v in verdicts:
        out.setdefault((v["day"], v["ticker"]), []).append(v)
    for rows in out.values():
        rows.sort(key=lambda v: (v["at"], v["id"]))
    return out


def flips(rows: list[dict]) -> list[dict]:
    """Consecutive pairs of one ticker-day whose stored signal differs."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["signal"] == b["signal"]:
            continue
        out.append({"a": a, "b": b,
                    "gap_min": round((b["at"] - a["at"]).total_seconds() / 60, 1),
                    "conv_changed": _changed(a["conviction"], b["conviction"]),
                    "udr_changed": _changed(a["udr"], b["udr"]),
                    "same_event": _same_event(a, b)})
    return out


def _yn(v) -> str:
    return "?" if v is None else ("yes" if v else "no")


def _fmt(v) -> str:
    if v is None:
        return "?"
    return f"{v:g}"


NO_PATH = "none"


def _path_of(v: dict) -> str:
    return v["path"] or NO_PATH


def _tally(fl: list[dict]) -> list[str]:
    """Cells: gap <= 5 min, gap <= 30 min, conviction unchanged, UDR
    unchanged, same event ("k of n known", or "0" when there are no flips)."""
    def of(pred):
        known = [f for f in fl if pred(f) is not None]
        return f"{sum(1 for f in known if pred(f))} of {len(known)}" if fl else "0"
    return ([str(sum(1 for f in fl if f["gap_min"] <= g)) for g in GAPS_MIN]
            + [of(lambda f: None if f["conv_changed"] is None else not f["conv_changed"]),
               of(lambda f: None if f["udr_changed"] is None else not f["udr_changed"]),
               of(lambda f: f["same_event"])])


def by_path(verdicts: list[dict], per_day: dict, days: list) -> list[str]:
    """Section lines: verdicts per day and path, flips per day and path pair
    (plus an all-days row per pair)."""
    lines = ["", "## 4. By trigger path", "",
             "Path = provenance.trigger.path; none = not recorded (rows before trigger "
             "provenance). A flip's pair is the earlier call's path -> the later call's.", "",
             "| day | path | verdicts |", "|---|---|---|"]
    counts: dict = {}
    for v in verdicts:
        k = (v["day"], _path_of(v))
        counts[k] = counts.get(k, 0) + 1
    for (d, p), n in sorted(counts.items()):
        lines.append(f"| {d} | {p} | {n} |")
    if not counts:
        lines.append("| - | - | 0 |")
    lines += ["", "| day | from path | to path | flips | gap <= 5 min | gap <= 30 min "
              "| conviction unchanged | UDR unchanged | same event |",
              "|---|---|---|---|---|---|---|---|---|"]
    pairs_all: dict = {}
    for day in days:
        pairs: dict = {}
        for f in per_day[day]:
            pairs.setdefault((_path_of(f["a"]), _path_of(f["b"])), []).append(f)
        for (pa, pb), fl in sorted(pairs.items()):
            pairs_all.setdefault((pa, pb), []).extend(fl)
            lines.append(f"| {day} | {pa} | {pb} | {len(fl)} | " + " | ".join(_tally(fl)) + " |")
    for (pa, pb), fl in sorted(pairs_all.items()):
        lines.append(f"| all | {pa} | {pb} | {len(fl)} | " + " | ".join(_tally(fl)) + " |")
    if not pairs_all:
        lines.append("| - | - | - | 0 | 0 | 0 | 0 | 0 | 0 |")
    return lines


def readout(verdicts: list[dict], since: date, until: date, env: str, head: str, now: datetime,
            detail: bool = True, ticker: str | None = None, by_path_section: bool = False) -> str:
    groups = group(verdicts)
    days = sorted({d for d, _ in groups})
    scope = f", ticker {ticker}" if ticker else ""
    lines = [f"# Analyst verdict flips ({env}, {since} to {until} New York days{scope})", "",
             f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} at code {head}. Pilot-phase "
             "engineering diagnostic, not a study result. A flip is two consecutive verdicts "
             "of one ticker on one New York day whose stored signal differs; pairs across two "
             "days are not compared. Gap = minutes between the two rows' timestamps. "
             "Conviction = final_conviction; UDR = udr_check.model; same event = same "
             "trigger story_key (else url, else title); ? = not recorded on one of the calls. "
             "Counts only; no thresholds chosen.", "",
             "## 1. Per day", "",
             "| day | verdicts | tickers | flips | tickers with a flip | gap <= 5 min "
             "| gap <= 30 min | conviction unchanged | UDR unchanged | same event |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    per_day: dict = {}
    for day in days:
        keys = [k for k in groups if k[0] == day]
        fl = [f for k in keys for f in flips(groups[k])]
        per_day[day] = fl
        lines.append(
            f"| {day} | {sum(len(groups[k]) for k in keys)} | {len(keys)} | {len(fl)} "
            f"| {len({f['a']['ticker'] for f in fl})} | " + " | ".join(_tally(fl)) + " |")
    if not days:
        lines.append("| - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
    tot = [f for fl in per_day.values() for f in fl]
    lines += ["", f"Total: {len(verdicts)} verdicts, {len(tot)} flips, "
              + ", ".join(f"{sum(1 for f in tot if f['gap_min'] <= g)} within {g} min"
                          for g in GAPS_MIN) + ".",
              "", "## 2. Per ticker and day (ticker-days with a flip)", "",
              "| day | ticker | verdicts | flips | gap <= 5 min | gap <= 30 min | pairs |",
              "|---|---|---|---|---|---|---|"]
    any_row = False
    for day in days:
        rows = []
        for (d, t), vs in groups.items():
            if d != day:
                continue
            fl = flips(vs)
            if not fl:
                continue
            pairs: dict = {}
            for f in fl:
                p = f"{f['a']['signal']}->{f['b']['signal']}"
                pairs[p] = pairs.get(p, 0) + 1
            rows.append((-len(fl), t, len(vs), len(fl),
                         sum(1 for f in fl if f["gap_min"] <= GAPS_MIN[0]),
                         sum(1 for f in fl if f["gap_min"] <= GAPS_MIN[1]),
                         ", ".join(f"{p} {c}" for p, c in sorted(pairs.items()))))
        for _, t, nv, nf, g5, g30, pr in sorted(rows):
            lines.append(f"| {day} | {t} | {nv} | {nf} | {g5} | {g30} | {pr} |")
            any_row = True
    if not any_row:
        lines.append("| - | - | 0 | 0 | 0 | 0 | - |")
    if detail:
        lines += ["", "## 3. Each flip", "",
                  "Times New York. Signal ids are the two signals rows.", "",
                  "| day | ticker | from | to | gap min | conviction | UDR | conviction changed "
                  "| UDR changed | same event | paths | signal ids |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for day in days:
            for f in sorted(per_day[day], key=lambda f: (f["a"]["ticker"], f["b"]["at"])):
                a, b = f["a"], f["b"]
                lines.append(
                    f"| {day} | {a['ticker']} | {a['signal']} {a['at'].astimezone(NY):%H:%M:%S} "
                    f"| {b['signal']} {b['at'].astimezone(NY):%H:%M:%S} | {f['gap_min']:g} "
                    f"| {_fmt(a['conviction'])} -> {_fmt(b['conviction'])} "
                    f"| {_fmt(a['udr'])} -> {_fmt(b['udr'])} | {_yn(f['conv_changed'])} "
                    f"| {_yn(f['udr_changed'])} | {_yn(f['same_event'])} "
                    f"| {_path_of(a)} -> {_path_of(b)} | {a['id']}, {b['id']} |")
        if not tot:
            lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")
    if by_path_section:
        lines += by_path(verdicts, per_day, days)
    return "\n".join(lines) + "\n"


def _fetch(since: date, until: date, env: str, ticker: str | None):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    end = datetime.combine(until + timedelta(days=1), time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            sql = ("SELECT id, symbol, timestamp, signal, confidence, "
                   "jsonb_build_object('final_conviction', decision->'final_conviction', "
                   "'model_action', decision->'model_action', 'udr_check', decision->'udr_check'), "
                   "provenance->'trigger' FROM signals "
                   "WHERE trading_env = %s AND decision ? 'final_action' "
                   "AND timestamp >= %s AND timestamp < %s")
            args = [env, start, end]
            if ticker:
                sql += " AND symbol = %s"
                args.append(ticker)
            cur.execute(sql, args)
            cols = ("id", "symbol", "timestamp", "signal", "confidence", "decision", "trigger")
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return [verdict(r) for r in rows]


def main() -> None:
    today = datetime.now(NY).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=str(today - timedelta(days=6)), help="YYYY-MM-DD, New York")
    ap.add_argument("--until", default=str(today), help="YYYY-MM-DD, New York, inclusive")
    ap.add_argument("--ticker", help="one ticker only")
    ap.add_argument("--no-detail", action="store_true", help="leave out the per-flip table")
    ap.add_argument("--by-path", action="store_true",
                    help="add verdicts per trigger path and flips per path pair")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)
    ticker = args.ticker.upper() if args.ticker else None
    from config import TRADING_ENV
    verdicts = _fetch(since, until, TRADING_ENV, ticker)
    try:
        head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    text = readout(verdicts, since, until, TRADING_ENV, head, datetime.now(timezone.utc),
                   detail=not args.no_detail, ticker=ticker,
                   by_path_section=args.by_path)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
