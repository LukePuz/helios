"""unrouted_verdict_forward_shadow: forward closes after non-routed verdicts
(batch 1128; audit 2026-11-26-0056 #4). Engineering readout, record only:
counts and missing rows. No move is summarized: which verdicts and horizon an
evaluation counts, and against what, are Luke's.

Two sources:
  - the book, system_state unrouted_verdict_forward_shadow_<env>
    (crew._unrouted_verdict_forward_record): records waiting for their
    forward rows, by action and verdict ET date, with or without a base bar,
    and the events each has logged. A record still there more than
    STALE_DAYS (the writer's 2 x 20 + 10) calendar days after its verdict
    should have been closed as expired: listed as overdue;
  - the activity rows, stage unrouted_verdict_forward_shadow
    (crew._check_unrouted_verdict_forwards; nightly): rows by event
    (forward_5, forward_10, forward_20, expired) and by action, base bars
    that don't match the base session, expired rows with
    window_complete false, and the forward events an expired record never
    logged (from its forward_logged), per horizon. A record is ticker +
    verdict ET date (the shadow key's identity).

When neither source exists (no book key, and no activity row of the stage at
any time) every count prints "not loaded".

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/unrouted_verdict_forward_readout.py [--env live] [--since YYYY-MM-DD] \\
        [--until YYYY-MM-DD] [--json] [--out FILE]

The window (ET days of the row's ts; --since defaults to 30 days before
today, --until to today) applies to the activity rows; the book is read as it
is now. Read-only: SUPABASE_PORT must be 6543 (the transaction pooler); one
transaction, SET TRANSACTION READ ONLY with a statement timeout, rolled back.
Counts, dates and actions only: no prices or moves. Not part of
weekly_readouts.
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
DEFAULT_DAYS = 30
NOT_LOADED = "not loaded"
STAGE = "unrouted_verdict_forward_shadow"
BOOK_KEY = "unrouted_verdict_forward_shadow_{env}"
HORIZONS = (5, 10, 20)                       # crew._UVF_FORWARD_SESSIONS
STALE_DAYS = 2 * max(HORIZONS) + 10          # crew's expiry for a record whose bars never arrive

ROWS_SQL = ("SELECT ts, ticker, extra_json FROM activity_feed WHERE stage = %(stage)s "
            "AND ts >= %(since)s AND ts < %(until)s "
            "AND COALESCE(extra_json->>'env', %(env)s) = %(env)s ORDER BY ts")
ROWS_EVER_SQL = "SELECT count(*) FROM activity_feed WHERE stage = %(stage)s"
BOOK_SQL = "SELECT value FROM system_state WHERE key = %(key)s"

BOOK_COUNTS = ("records", "by_action", "with_base_bar", "waiting_for_base_bar", "logged_events",
               "overdue")
ROW_COUNTS = ("rows", "by_event", "by_action_event", "records_seen", "base_bar_mismatch",
              "expired_window_incomplete", "expired_missing_forward")


def _book(raw) -> dict | None:
    """The book as a dict; None for no key, {"_unreadable": True} for a
    value that isn't a JSON object."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        b = json.loads(raw)
    except (TypeError, ValueError):
        return {"_unreadable": True}
    return b if isinstance(b, dict) else {"_unreadable": True}


def _days_since(d, today: date):
    try:
        return (today - date.fromisoformat(str(d)[:10])).days
    except (TypeError, ValueError):
        return None


def build(rows: list[dict], book_raw, rows_ever: int, today: date) -> dict:
    """rows: activity rows {ts, ticker, extra_json}, oldest first; book_raw:
    the system_state value (None when the key is absent); rows_ever: activity
    rows of the stage at any time. Returns {"loaded", "book", "rows",
    "overdue", "missing"}."""
    book = _book(book_raw)
    loaded = book is not None or bool(rows_ever)
    if not loaded:
        return {"loaded": False, "book": dict.fromkeys(BOOK_COUNTS, NOT_LOADED),
                "rows": dict.fromkeys(ROW_COUNTS, NOT_LOADED), "overdue": [], "missing": [],
                "book_read": NOT_LOADED}
    recs = [r for r in (book or {}).values() if isinstance(r, dict)]
    overdue = []
    for r in recs:
        age = _days_since(r.get("et_date"), today)
        if age is not None and age > STALE_DAYS:
            overdue.append({"ticker": r.get("ticker"), "et_date": r.get("et_date"),
                            "action": r.get("action"), "days": age,
                            "logged": sorted((r.get("logged") or {}).keys())})
    logged = Counter(k for r in recs for k in (r.get("logged") or {}))
    if book is None:
        book_read, bk = "no key", dict.fromkeys(BOOK_COUNTS, "no key")
    elif book.get("_unreadable"):
        book_read, bk = "unreadable", dict.fromkeys(BOOK_COUNTS, "unreadable")
    else:
        book_read = "ok"
        bk = {"records": len(recs),
              "by_action": dict(sorted(Counter(r.get("action") or "n/a" for r in recs).items())),
              "with_base_bar": sum(1 for r in recs if r.get("base_bar_date")),
              "waiting_for_base_bar": sum(1 for r in recs if not r.get("base_bar_date")),
              "logged_events": dict(sorted(logged.items())),
              "overdue": len(overdue)}
    by_event, by_ae, ids, mismatch, incomplete = Counter(), Counter(), set(), 0, 0
    missing_by_h, missing = Counter(), []
    for y in rows:
        ex = y.get("extra_json") or {}
        ev, act = ex.get("event") or "n/a", ex.get("action") or "n/a"
        by_event[ev] += 1
        by_ae[f"{act} {ev}"] += 1
        ids.add((str(y.get("ticker") or "").upper(), ex.get("verdict_et_date")))
        if ex.get("base_bar_matches") is False:
            mismatch += 1
        if ev == "expired":
            if ex.get("window_complete") is False:
                incomplete += 1
            got = set(ex.get("forward_logged") or [])
            lacks = [f"forward_{h}" for h in HORIZONS if f"forward_{h}" not in got]
            for k in lacks:
                missing_by_h[k] += 1
            if lacks:
                missing.append({"ticker": y.get("ticker"), "verdict_et_date": ex.get("verdict_et_date"),
                                "action": act, "sessions": ex.get("sessions"),
                                "window_complete": ex.get("window_complete"), "missing": lacks})
    rw = {"rows": len(rows), "by_event": dict(sorted(by_event.items())),
          "by_action_event": dict(sorted(by_ae.items())), "records_seen": len(ids),
          "base_bar_mismatch": mismatch, "expired_window_incomplete": incomplete,
          "expired_missing_forward": {f"forward_{h}": missing_by_h.get(f"forward_{h}", 0)
                                      for h in HORIZONS}}
    return {"loaded": True, "book": bk, "rows": rw, "overdue": overdue, "missing": missing,
            "book_read": book_read}


def _v(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k} {v[k]}" for k in v) or "none"
    return str(v)


def render(res: dict, env: str, since: str, until: str) -> str:
    L = [f"# unrouted_verdict_forward_shadow ({env}; rows {since} to {until} ET)", ""]
    if not res["loaded"]:
        L += [f"unrouted_verdict_forward_shadow: {NOT_LOADED} (no book key, and no activity row "
              f"of the stage exists).", ""]
    L += [f"## Book (now; read {res['book_read']})", "", "| count | value |", "|---|---|"]
    L += [f"| {k} | {_v(res['book'][k])} |" for k in BOOK_COUNTS]
    L += ["", "## Activity rows in the window", "", "| count | value |", "|---|---|"]
    L += [f"| {k} | {_v(res['rows'][k])} |" for k in ROW_COUNTS]
    if res["overdue"]:
        L += ["", f"## Overdue: in the book more than {STALE_DAYS} days after the verdict", "",
              "| ticker | verdict ET date | action | days | logged |", "|---|---|---|---|---|"]
        L += [f"| {o['ticker']} | {o['et_date']} | {o['action']} | {o['days']} | "
              f"{', '.join(o['logged']) or 'none'} |" for o in res["overdue"]]
    if res["missing"]:
        L += ["", "## Expired records with forward events never logged", "",
              "| ticker | verdict ET date | action | sessions | window complete | missing |",
              "|---|---|---|---|---|---|"]
        L += [f"| {m['ticker']} | {m['verdict_et_date']} | {m['action']} | {m['sessions']} | "
              f"{m['window_complete']} | {', '.join(m['missing'])} |" for m in res["missing"]]
    return "\n".join(L) + "\n"


def _bounds(since: str, until: str) -> tuple[datetime, datetime]:
    """UTC bounds for ET days since..until, both inclusive."""
    lo = datetime.combine(date.fromisoformat(since), time.min, tzinfo=NY)
    hi = datetime.combine(date.fromisoformat(until) + timedelta(days=1), time.min, tzinfo=NY)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def fetch(conn, env: str, since: str, until: str) -> tuple[list, object, int]:
    """The reads in one read-only transaction, rolled back. The book value
    is None when the key is absent."""
    lo, hi = _bounds(since, until)
    params = {"env": env, "since": lo, "until": hi, "stage": STAGE, "key": BOOK_KEY.format(env=env)}
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
            cur.execute(BOOK_SQL, params)
            b = cur.fetchone()
            book = b[0] if b else None
    finally:
        conn.rollback()
    return rows, book, ever


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="unrouted_verdict_forward_shadow: counts and missing rows")
    ap.add_argument("--env", help="trading env (default config.TRADING_ENV)")
    ap.add_argument("--since", help="first ET day of rows, YYYY-MM-DD (default 30 days ago)")
    ap.add_argument("--until", help="last ET day of rows, YYYY-MM-DD (default today)")
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
        rows, book, ever = fetch(conn, env, since, until)
    finally:
        db._put_conn(conn)
    res = build(rows, book, ever, today)
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
