"""earnings_calendar_check: the earnings calendar refresh, checked per ticker
(batch 1134 rows 2 and 5; audit 2026-11-26-0056 #4). Engineering readout,
record only: counts and missing rows. held_side_readout shows the held names'
entries day by day; this covers the whole row.

From the stage's activity rows (ingestion/earnings_calendar.py:
_record_earnings_checks; weekdays 08:00 ET):
  - per refresh row: checked, counts (future_date / no_future_date / error),
    held_read, held_no_future, warned, and from row 5 catalyst_write_counts
    and cadence_estimated;
  - over the window's rows, per ticker check: result, error_at,
    catalyst_write and cadence_none_reason counts; the tickers whose latest
    check has no future date, the tickers with any catalyst_write error;
  - missing rows: weekdays in the window with no row (holidays are not
    excluded: the refresh is scheduled on weekdays), days with more than
    one row, and tickers in an earlier row that the latest row lacks.

A key a row doesn't carry reads "not recorded yet" (row 5's keys load with
row 2, in the same batch). When the stage has no row at all every count
prints "not loaded".

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/earnings_check_readout.py [--env live] [--since YYYY-MM-DD] \\
        [--until YYYY-MM-DD] [--json] [--out FILE]

--since defaults to 14 days before today (ET), --until to today. Read-only:
SUPABASE_PORT must be 6543 (the transaction pooler); one transaction, SET
TRANSACTION READ ONLY with a statement timeout, rolled back. Counts, dates
and tickers only. Descriptive: whether a missing date should gate or label
anything is Luke's. Not part of weekly_readouts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 30000
DEFAULT_DAYS = 14
NOT_LOADED = "not loaded"
NOT_RECORDED = "not recorded yet"

ROWS_SQL = ("SELECT ts, extra_json FROM activity_feed WHERE stage = 'earnings_calendar_check' "
            "AND ts >= %(since)s AND ts < %(until)s "
            "AND COALESCE(extra_json->>'env', %(env)s) = %(env)s ORDER BY ts")
ROWS_EVER_SQL = "SELECT count(*) FROM activity_feed WHERE stage = 'earnings_calendar_check'"

RUN_FIELDS = ("checked", "counts", "held_read", "held_no_future", "warned",
              "catalyst_write_counts", "cadence_estimated")
TOTAL_KEYS = ("rows", "ticker_checks", "result", "error_at", "catalyst_write", "cadence_none_reason",
              "latest_no_future_date", "catalyst_write_error_tickers", "weekdays_without_row",
              "days_with_several_rows", "dropped_from_latest")


def _et_day(ts) -> str | None:
    if isinstance(ts, datetime):
        return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(NY).date().isoformat()
    try:
        return _et_day(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def _pick(d: dict, k):
    return d[k] if k in d else NOT_RECORDED


def _tally(values) -> dict | str:
    """Counts of the values; NOT_RECORDED when no check carries the key."""
    vals = list(values)
    if vals and all(v is NOT_RECORDED for v in vals):
        return NOT_RECORDED
    return dict(sorted(Counter("n/a" if v is None else str(v)
                               for v in vals if v is not NOT_RECORDED).items()))


def build(rows: list[dict], rows_ever: int, since: str, until: str) -> dict:
    """rows: activity rows {ts, extra_json}, oldest first; rows_ever: rows of
    the stage at any time. Returns {"loaded", "runs", "totals"}."""
    if not rows_ever:
        return {"loaded": False, "runs": [], "totals": dict.fromkeys(TOTAL_KEYS, NOT_LOADED)}
    runs, per_day, checks = [], Counter(), []
    for r in rows:
        ex = r.get("extra_json") or {}
        day = str(ex.get("day") or "")[:10] or _et_day(r.get("ts"))
        per_day[day] += 1
        run = {"day": day}
        for k in RUN_FIELDS:
            run[k] = _pick(ex, k)
        runs.append(run)
        for t, c in sorted((ex.get("tickers") or {}).items()):
            if isinstance(c, dict):
                checks.append((day, str(t).upper(), c))
    latest_ex = (rows[-1].get("extra_json") or {}) if rows else {}
    latest = {str(t).upper(): c for t, c in (latest_ex.get("tickers") or {}).items() if isinstance(c, dict)}
    earlier = {t for _d, t, _c in checks} - set(latest)
    weekdays, d = [], date.fromisoformat(since)
    while d <= date.fromisoformat(until):
        if d.weekday() < 5 and d.isoformat() not in per_day:
            weekdays.append(d.isoformat())
        d += timedelta(days=1)
    totals = {
        "rows": len(rows), "ticker_checks": len(checks),
        "result": _tally(_pick(c, "result") for _d, _t, c in checks),
        "error_at": _tally(c.get("error_at") for _d, _t, c in checks if c.get("result") == "error"),
        "catalyst_write": _tally(_pick(c, "catalyst_write") for _d, _t, c in checks),
        "cadence_none_reason": _tally(_pick(c, "cadence_none_reason") for _d, _t, c in checks),
        "latest_no_future_date": sorted(t for t, c in latest.items() if c.get("result") == "no_future_date"),
        "catalyst_write_error_tickers": sorted({t for _d, t, c in checks
                                                if c.get("catalyst_write") == "error"}),
        "weekdays_without_row": weekdays,
        "days_with_several_rows": sorted(k for k, n in per_day.items() if n > 1),
        "dropped_from_latest": sorted(earlier),
    }
    return {"loaded": True, "runs": runs, "totals": totals}


def _v(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k} {v[k]}" for k in v) or "none"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) or "none"
    return "n/a" if v is None else str(v)


def render(res: dict, env: str, since: str, until: str) -> str:
    L = [f"# earnings_calendar_check ({env}, {since} to {until} ET)", ""]
    if not res["loaded"]:
        L += [f"earnings_calendar_check: {NOT_LOADED} (no activity row of the stage exists).", ""]
    L += ["| count | value |", "|---|---|"]
    L += [f"| {k} | {_v(res['totals'][k])} |" for k in TOTAL_KEYS]
    if res["runs"]:
        L += ["", "## Refresh rows", "", "| day | " + " | ".join(RUN_FIELDS) + " |",
              "|" + "---|" * (len(RUN_FIELDS) + 1)]
        L += ["| " + " | ".join([r["day"]] + [_v(r[k]) for k in RUN_FIELDS]) + " |" for r in res["runs"]]
    return "\n".join(L) + "\n"


def _bounds(since: str, until: str) -> tuple[datetime, datetime]:
    """UTC bounds for ET days since..until, both inclusive."""
    lo = datetime.combine(date.fromisoformat(since), time.min, tzinfo=NY)
    hi = datetime.combine(date.fromisoformat(until) + timedelta(days=1), time.min, tzinfo=NY)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def fetch(conn, env: str, since: str, until: str) -> tuple[list, int]:
    """The reads in one read-only transaction, rolled back."""
    lo, hi = _bounds(since, until)
    params = {"env": env, "since": lo, "until": hi}
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(ROWS_SQL, params)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.execute(ROWS_EVER_SQL, params)
            r = cur.fetchone()
            ever = int(r[0]) if r and r[0] is not None else 0
    finally:
        conn.rollback()
    return rows, ever


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="earnings_calendar_check: counts and missing rows")
    ap.add_argument("--env", help="trading env (default config.TRADING_ENV)")
    ap.add_argument("--since", help="first ET day, YYYY-MM-DD (default 14 days ago)")
    ap.add_argument("--until", help="last ET day, YYYY-MM-DD (default today)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    today = datetime.now(NY).date()
    until = args.until or today.isoformat()
    since = args.since or (today - timedelta(days=DEFAULT_DAYS)).isoformat()
    import config
    import db
    env = args.env or config.TRADING_ENV
    conn = db._get_conn()
    try:
        rows, ever = fetch(conn, env, since, until)
    finally:
        db._put_conn(conn)
    res = build(rows, ever, since, until)
    text = (json.dumps({"env": env, "since": since, "until": until, **res}, default=str, indent=1) + "\n"
            if args.json else render(res, env, since, until))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
