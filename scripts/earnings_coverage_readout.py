"""Earnings coverage readout: for each watchlist or held ticker, what
earnings_calendar holds and whether the brief can show it (audit
2026-11-27-1053 #3, roadmap 04fz). Record-only engineering readout; tickers,
dates, days, counts and ids only: no price, dollar figure or share count is
printed, and no plan or calendar text.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/earnings_coverage_readout.py [--env live] [--out FILE] [--json FILE]

Today is the New York date of the run. Per ticker (the watchlist, which is
knowledge/watchlist.md parsed as text with knowledge_loader.load_watchlist's
row pattern, plus any held name not on it):
  - next row: the earliest earnings_calendar date on or after today
    (db.get_next_earnings_row's rule, which the brief's own-earnings line
    uses), its confirmed flag, its days ahead, the New York date it was last
    refreshed, and how many rows are on or after today;
  - inside window: whether the next row's days ahead is within the brief's
    own-earnings cutoff (context_engine._own_earnings_line hides a row whose
    days ahead is greater than the cutoff), "no row" without a next row,
    "unknown" when the cutoff can't be read;
  - last past row: the latest date before today, and its days ago;
  - held names: the active trade plan's id and catalyst_date, the
    catalyst's days ahead, and the gap in days, catalyst_date minus the next
    row's date ("no row" without one). plans._catalyst_date stores a
    month-only catalyst as the month's last day, so a gap can be a
    month-end artefact; the readout doesn't tell them apart.
Held is the keys of the latest portfolio_snapshots row in --env (the
routing's _held_tickers rule); only the keys are selected. The cutoff and
the catalyst section's days_ahead are read from the working directory's
context_engine.py as text (located by function name, not imported), so a
run in the live clone reads the live brief's window.

Summaries: held names without a next row; held names whose next row is
outside the window; next rows by confirmed flag (earnings_calendar has no
source column; its one writer is ingestion/earnings_calendar.py's yfinance
refresh, so confirmed / estimated is the only split stored); tickers with
no row of any kind; calendar tickers that are neither watchlist nor held
(a count). Postgres CURRENT_DATE (db.get_next_earnings_row's today) is
printed beside the New York date, as the two can differ in the evening.

Filling rows or changing the window is analyst input and Luke's; nothing
here is a rule or a threshold. SUPABASE_PORT must be 6543 (the transaction
pooler); one read-only transaction with a statement timeout, rolled back.
It does not read between 04:50 and 05:12 ET (the stack's start). db is
imported from the working directory when it holds a db.py (the live clone,
with its .env), else from beside this script; with no SUPABASE_HOST after
the import the readout stops before connecting. Imports: db and the
standard library. Not loaded by weekly_readouts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
QUIET_START, QUIET_END = time(4, 50), time(5, 12)   # no DB reads at the stack's start
# knowledge_loader.load_watchlist's table-row pattern (a test pins it).
WATCHLIST_ROW = re.compile(
    r"^\|\s*([A-Z]{1,6})\s*\|\s*(.+?)\s*\|\s*(Compounder|Accelerator|Moonshot)\s*\|"
    r"\s*(Core|Watch|Speculative)\s*\|")
TICKERISH_ROW = re.compile(r"^\|\s*([A-Z]{1,6})\s*\|")
_TOP_LEVEL = re.compile(r"^(?:def |class |@|[A-Za-z_])", re.M)
NOTE = ("Record-only: what the calendar holds and whether the brief's window shows it. "
        "Filling rows or changing the window is analyst input and Luke's.")

_TODAY_SQL = "SELECT CURRENT_DATE"
_HELD_SQL = """
SELECT timestamp,
       CASE WHEN jsonb_typeof(positions_json) = 'object'
            THEN ARRAY(SELECT jsonb_object_keys(positions_json)) END
FROM portfolio_snapshots
WHERE trading_env = %(env)s
ORDER BY timestamp DESC LIMIT 1
"""
_CAL_SQL = """
SELECT upper(ticker), earnings_date, confirmed, refreshed_at
FROM earnings_calendar
WHERE ticker IS NOT NULL AND earnings_date IS NOT NULL
ORDER BY 1, 2
"""
_PLANS_SQL = """
SELECT id, upper(ticker), catalyst_date, created_at
FROM trade_plans
WHERE status = 'active' AND (trading_env = %(env)s OR trading_env IS NULL)
ORDER BY 2, created_at DESC, id DESC
"""


def db_root(cwd: str | None = None) -> str:
    """The directory db is imported from: cwd when it holds a db.py, else
    this script's checkout."""
    cwd = os.getcwd() if cwd is None else cwd
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    """True between 04:50 and 05:12 ET, when the readout doesn't read."""
    t = now.astimezone(NY).time()
    return QUIET_START <= t < QUIET_END


def parse_watchlist(text: str) -> tuple[dict, list[str]]:
    """({ticker: {"etf": bool}}, ticker-like rows that don't parse), with
    load_watchlist's row pattern; "etf" is get_etf_tickers' rule (the
    company name says ETF)."""
    out, unparsed = {}, []
    for line in text.splitlines():
        m = WATCHLIST_ROW.match(line)
        if m:
            out[m.group(1)] = {"etf": bool(re.search(r"\bETF\b", m.group(2)))}
            continue
        t = TICKERISH_ROW.match(line)
        if t:
            unparsed.append(t.group(1))
    return out, unparsed


def _function_src(src: str, name: str) -> tuple[str | None, int | None]:
    """The source of top-level function `name` and its 1-based line."""
    m = re.search(rf"^def {re.escape(name)}\(", src, re.M)
    if not m:
        return None, None
    end = _TOP_LEVEL.search(src, m.end())     # with pos, ^ is a real line start only
    body = src[m.start(): end.start()] if end else src[m.start():]
    return body, src.count("\n", 0, m.start()) + 1


def read_window(src: str) -> dict:
    """The brief's earnings windows, read from context_engine.py's text:
    the own-earnings cutoff (`.days > N` in _own_earnings_line) and the
    catalyst section's days_ahead (in _catalyst_calendar_section). None
    when the pattern isn't found."""
    out = {"own_cutoff_days": None, "own_line": None,
           "calendar_days_ahead": None, "calendar_line": None}
    body, line = _function_src(src, "_own_earnings_line")
    m = re.search(r"\.days\s*>\s*(\d+)", body or "")
    if m:
        out["own_cutoff_days"], out["own_line"] = int(m.group(1)), line
    body, line = _function_src(src, "_catalyst_calendar_section")
    m = re.search(r"days_ahead\s*=\s*(\d+)", body or "")
    if m:
        out["calendar_days_ahead"], out["calendar_line"] = int(m.group(1)), line
    return out


def _d(v) -> date | None:
    if isinstance(v, datetime):
        return v.astimezone(NY).date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def _iso(v) -> str:
    return v.isoformat() if isinstance(v, (date, datetime)) else "none"


def build(watch: dict, held: set[str], cal_rows, plan_rows, today: date,
          cutoff: int | None) -> tuple[list[dict], dict]:
    """One record per watchlist-or-held ticker, and the summaries."""
    cal: dict[str, list] = {}
    for tk, d, confirmed, refreshed in cal_rows:
        d = _d(d)
        if tk and d:
            cal.setdefault(tk, []).append((d, confirmed, _d(refreshed)))
    plans: dict[str, list] = {}
    for pid, tk, cat, _created in plan_rows:
        plans.setdefault(tk, []).append((pid, _d(cat)))   # newest first
    recs = []
    for tk in sorted(set(watch) | held):
        rows = sorted(cal.get(tk, []), key=lambda r: r[0])
        future = [r for r in rows if r[0] >= today]
        past = [r for r in rows if r[0] < today]
        nxt = future[0] if future else None
        rec = {"ticker": tk, "watchlist": tk in watch, "held": tk in held,
               "etf": bool(watch.get(tk, {}).get("etf")), "rows": len(rows),
               "future_rows": len(future),
               "next_date": _iso(nxt[0]) if nxt else "none",
               "next_days_ahead": (nxt[0] - today).days if nxt else None,
               "next_confirmed": (None if nxt is None or nxt[1] is None else bool(nxt[1])),
               "next_refreshed": _iso(nxt[2]) if nxt else "none",
               "last_past_date": _iso(past[-1][0]) if past else "none",
               "last_past_days_ago": (today - past[-1][0]).days if past else None}
        if nxt is None:
            rec["inside_window"] = "no row"
        elif cutoff is None:
            rec["inside_window"] = "unknown"
        else:
            rec["inside_window"] = "yes" if rec["next_days_ahead"] <= cutoff else "no"
        if tk in held:
            ps = plans.get(tk, [])
            pid, cat = ps[0] if ps else (None, None)
            rec.update({"active_plans": len(ps), "plan_id": pid,
                        "plan_catalyst_date": _iso(cat),
                        "plan_catalyst_days_ahead": (cat - today).days if cat else None,
                        "catalyst_minus_next_days": ((cat - nxt[0]).days if cat and nxt
                                                     else None)})
        recs.append(rec)
    known = set(watch) | held
    summ = {
        "tickers": len(recs), "watchlist": len(watch), "held": len(held),
        "held_not_on_watchlist": sorted(held - set(watch)),
        "with_next_row": sum(r["future_rows"] > 0 for r in recs),
        "without_next_row": sorted(r["ticker"] for r in recs if r["future_rows"] == 0),
        "no_row_at_all": sorted(r["ticker"] for r in recs if r["rows"] == 0),
        "several_future_rows": sorted(r["ticker"] for r in recs if r["future_rows"] > 1),
        "inside_window": dict(Counter(r["inside_window"] for r in recs)),
        "held_without_next_row": sorted(r["ticker"] for r in recs
                                        if r["held"] and r["future_rows"] == 0),
        "held_next_outside_window": sorted(r["ticker"] for r in recs
                                           if r["held"] and r["inside_window"] == "no"),
        "held_without_active_plan": sorted(r["ticker"] for r in recs
                                           if r["held"] and not r.get("active_plans")),
        "next_rows_by_confirmed": dict(Counter(
            {True: "confirmed", False: "estimated", None: "null"}[r["next_confirmed"]]
            for r in recs if r["future_rows"] > 0)),
        "calendar_tickers_not_watchlist_or_held": len(set(cal) - known),
    }
    return recs, summ


def _v(x) -> str:
    return "none" if x is None else str(x)


def render(recs, summ, window: dict, env: str, today: date, db_today, held_at, head: str,
           unparsed: list[str]) -> str:
    cut = window["own_cutoff_days"]
    lines = ["# Earnings coverage readout", "",
             f"Env {env}; today {today.isoformat()} (New York), Postgres CURRENT_DATE "
             f"{_iso(_d(db_today)) if db_today else 'none'}; code {head}. Tickers, dates, days, "
             "counts and ids only.", "", NOTE, "",
             "Brief window (context_engine.py, read as text): own-earnings line shown when days "
             f"ahead <= {_v(cut)} (_own_earnings_line, line {_v(window['own_line'])}); catalyst "
             f"section days_ahead {_v(window['calendar_days_ahead'])} "
             f"(_catalyst_calendar_section, line {_v(window['calendar_line'])}).",
             f"Held: keys of the latest {env} portfolio snapshot, "
             f"{_iso(held_at) if held_at else 'none (no snapshot)'}.",
             f"Watchlist rows parsed: {summ['watchlist']}; ticker-like rows not parsed: "
             f"{len(unparsed)}{' (' + ', '.join(unparsed) + ')' if unparsed else ''}.", "",
             "## Summaries", "",
             f"- Tickers: {summ['tickers']} (watchlist {summ['watchlist']}, held {summ['held']}, "
             f"held not on the watchlist: {', '.join(summ['held_not_on_watchlist']) or 'none'}).",
             f"- Held without a next row: {len(summ['held_without_next_row'])} "
             f"({', '.join(summ['held_without_next_row']) or 'none'}).",
             f"- Held whose next row is outside the window: "
             f"{len(summ['held_next_outside_window'])} "
             f"({', '.join(summ['held_next_outside_window']) or 'none'}).",
             f"- Held without an active plan: {len(summ['held_without_active_plan'])} "
             f"({', '.join(summ['held_without_active_plan']) or 'none'}).",
             f"- With a next row: {summ['with_next_row']} of {summ['tickers']}; without: "
             f"{len(summ['without_next_row'])} ({', '.join(summ['without_next_row']) or 'none'}).",
             f"- No row of any kind: {len(summ['no_row_at_all'])} "
             f"({', '.join(summ['no_row_at_all']) or 'none'}).",
             f"- More than one row on or after today: {len(summ['several_future_rows'])} "
             f"({', '.join(summ['several_future_rows']) or 'none'}).",
             "- Inside window: " + ", ".join(f"{k} {summ['inside_window'].get(k, 0)}"
                                             for k in ("yes", "no", "no row", "unknown")) + ".",
             "- Next rows by confirmed flag (earnings_calendar has no source column; its one "
             "writer is the yfinance refresh): "
             + ", ".join(f"{k} {summ['next_rows_by_confirmed'].get(k, 0)}"
                         for k in ("confirmed", "estimated", "null")) + ".",
             f"- Calendar tickers neither on the watchlist nor held: "
             f"{summ['calendar_tickers_not_watchlist_or_held']}.", ""]
    held = [r for r in recs if r["held"]]
    lines += ["## Held names", "",
              "| ticker | next row | days ahead | inside window | last past row | days ago | "
              "plan id | active plans | plan catalyst_date | catalyst days ahead | "
              "catalyst minus next row (days) |",
              "|---|---|---:|---|---|---:|---:|---:|---|---:|---:|"]
    for r in held:
        gap = r["catalyst_minus_next_days"]
        if gap is None:
            gap = "no row" if r["future_rows"] == 0 else "none"
        lines.append(f"| {r['ticker']} | {r['next_date']} | {_v(r['next_days_ahead'])} | "
                     f"{r['inside_window']} | {r['last_past_date']} | "
                     f"{_v(r['last_past_days_ago'])} | {_v(r['plan_id'])} | "
                     f"{r['active_plans']} | {r['plan_catalyst_date']} | "
                     f"{_v(r['plan_catalyst_days_ahead'])} | {gap} |")
    if not held:
        lines.append("| none | | | | | | | | | | |")
    lines += ["", "## All tickers", "",
              "| ticker | watchlist | held | etf | rows | rows on/after today | next row | "
              "days ahead | confirmed | refreshed | inside window | last past row | days ago |",
              "|---|---|---|---|---:|---:|---|---:|---|---|---|---|---:|"]
    for r in recs:
        conf = {True: "yes", False: "no", None: "none"}[r["next_confirmed"]]
        lines.append(f"| {r['ticker']} | {'yes' if r['watchlist'] else 'no'} | "
                     f"{'yes' if r['held'] else 'no'} | {'yes' if r['etf'] else 'no'} | "
                     f"{r['rows']} | {r['future_rows']} | {r['next_date']} | "
                     f"{_v(r['next_days_ahead'])} | {conf} | {r['next_refreshed']} | "
                     f"{r['inside_window']} | {r['last_past_date']} | "
                     f"{_v(r['last_past_days_ago'])} |")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the records here")
    args = ap.parse_args(argv)
    now = now or datetime.now(timezone.utc)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    if in_quiet_window(now):
        print("04:50-05:12 ET is the stack's start; not reading.", file=sys.stderr)
        return 2
    base = db_root()
    try:
        with open(os.path.join(base, "knowledge", "watchlist.md"), encoding="utf-8") as f:
            watch, unparsed = parse_watchlist(f.read())
        with open(os.path.join(base, "context_engine.py"), encoding="utf-8") as f:
            window = read_window(f.read())
    except OSError as e:
        print(f"Can't read the watchlist or context_engine.py under {base}: {e}", file=sys.stderr)
        return 2
    today = now.astimezone(NY).date()
    params = {"env": args.env}

    sys.path.insert(0, base)
    import db
    if not os.getenv("SUPABASE_HOST"):
        print(f"No SUPABASE_HOST after importing {getattr(db, '__file__', 'db')} (no .env beside "
              "it?); run with env -C ~/Desktop/trading_bot. Not connecting.", file=sys.stderr)
        return 2
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
            cur.execute(_TODAY_SQL)
            db_today = (cur.fetchone() or (None,))[0]
            cur.execute(_HELD_SQL, params)
            snap = cur.fetchone()
            cur.execute(_CAL_SQL)
            cal_rows = cur.fetchall()
            cur.execute(_PLANS_SQL, params)
            plan_rows = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    held_at = snap[0] if snap else None
    held = {str(k).upper() for k in ((snap[1] if snap else None) or [])}
    recs, summ = build(watch, held, cal_rows, plan_rows, today, window["own_cutoff_days"])
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(recs, summ, window, args.env, today, db_today, held_at, head or "unknown",
                  unparsed)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head or "unknown",
                       "env": args.env, "today": today.isoformat(),
                       "db_current_date": _iso(_d(db_today)) if db_today else None,
                       "held_snapshot_at": _iso(held_at) if held_at else None,
                       "window": window, "note": NOTE, "summary": summ, "tickers": recs,
                       "watchlist_unparsed": unparsed}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
