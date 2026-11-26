"""brief_price_yahoo: Yahoo's 1-minute bar beside a stale daily-bar brief
price (batch 1133 row 12; audit 2026-11-26-0056 #4). Engineering readout,
record only: counts and missing rows.

One row per ticker + brief_built_at (the join key), from two sources:
  - the verdict's provenance.trigger: brief_price_source,
    brief_price_quote_source, brief_price_is_stale, brief_price_age_s and the
    brief_price_yahoo marker ({status queued / not_queued, queued_at, error},
    null when the writer judged the brief not due, the key absent before the
    writer loaded). A routing row can carry the same trigger; the verdict is
    the key's first signals row outside db._NON_ANALYST_SIGNALS
    (signal_rows: how many);
  - the brief_price_yahoo activity row (tasks.brief_price_yahoo_shadow):
    status, error, bar_age_s, bar_in_rth, yahoo_vs_brief_pct, queue_delay_s,
    elapsed_s. The row carries no env key, so a row with no marker in this
    env is listed as "no marker" (it may be another env's).

Due by the writer's rule (crew._brief_yahoo_due), as far as the trigger shows
it: brief_price_source latest_quote, quote source daily_fallback, stale True,
built on a weekday 09:30-16:00 ET. Holidays and early closes are not checked
here (the writer checks them with market_data.is_market_hours), so a due key
with a null marker on such a day is the writer's own skip.

Missing rows: due keys with a null marker, queued markers with no activity
row, not_queued markers (the publish raised), and activity rows with no
marker. When the writer has not loaded (no verdict in the window carries the
marker key, and no activity row exists at all) every count prints
"not loaded". The trigger's quote-source keys load in the same batch (none of
17755 live verdicts in the 14 days to 2026-11-26 carried
brief_price_quote_source), so "due" can't be counted before the load either.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/brief_price_yahoo_readout.py [--env live] [--since YYYY-MM-DD] \\
        [--until YYYY-MM-DD] [--json] [--out FILE]

--since defaults to 14 days before today (ET), --until to today. Read-only:
SUPABASE_PORT must be 6543 (the transaction pooler); one transaction, SET
TRANSACTION READ ONLY with a statement timeout, rolled back. Percentages,
ages and counts only: no prices. Descriptive: no thresholds; what any of it
should change is a study choice. Not part of weekly_readouts.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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

ROWS_SQL = ("SELECT ts, ticker, extra_json FROM activity_feed WHERE stage = 'brief_price_yahoo' "
            "AND ts >= %(since)s AND ts < %(until)s "
            "AND COALESCE(extra_json->>'env', %(env)s) = %(env)s ORDER BY ts")
ROWS_EVER_SQL = "SELECT count(*) FROM activity_feed WHERE stage = 'brief_price_yahoo'"
MARKER_KEY_SQL = ("SELECT count(*) FROM signals WHERE trading_env = %(env)s "
                  "AND timestamp >= %(since)s AND timestamp < %(until)s "
                  "AND provenance->'trigger' ? 'brief_price_yahoo'")
VERDICT_SQL = (
    "SELECT symbol, timestamp, signal, provenance->'trigger'->>'brief_built_at' AS brief_built_at, "
    "provenance->'trigger'->'brief_price_yahoo' AS marker, "
    "provenance->'trigger' ? 'brief_price_yahoo' AS has_marker_key, "
    "provenance->'trigger'->>'brief_price_source' AS price_source, "
    "provenance->'trigger'->>'brief_price_quote_source' AS quote_source, "
    "provenance->'trigger'->>'brief_price_is_stale' AS is_stale, "
    "provenance->'trigger'->>'brief_price_age_s' AS brief_price_age_s FROM signals "
    "WHERE trading_env = %(env)s AND timestamp >= %(since)s AND timestamp < %(until)s "
    "AND (provenance->'trigger'->>'brief_price_quote_source' = 'daily_fallback' "
    "OR jsonb_typeof(provenance->'trigger'->'brief_price_yahoo') = 'object') ORDER BY timestamp")

_DIST = ("brief_price_age_s", "bar_age_s", "yahoo_vs_brief_pct", "queue_delay_s", "elapsed_s")
COUNT_KEYS = ("keys", "due", "marker_queued", "marker_not_queued", "due_marker_null",
              "due_before_load", "queued_with_row", "queued_no_row", "row_no_marker",
              "rows_in_window", "rows_by_status")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _when(v) -> datetime | None:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def due_by_rule(v: dict) -> bool:
    """The trigger's side of crew._brief_yahoo_due: latest_quote, its
    daily_fallback, stale, built on a weekday 09:30-16:00 ET (no holiday
    or early-close check)."""
    if not (v.get("price_source") == "latest_quote" and v.get("quote_source") == "daily_fallback"
            and str(v.get("is_stale")).lower() == "true"):
        return False
    b = _when(v.get("brief_built_at"))
    if b is None:
        return False
    et = b.astimezone(NY)
    return et.weekday() < 5 and time(9, 30) <= et.time() < time(16, 0)


def _key(ticker, built_at) -> tuple:
    return (str(ticker or "").upper(), str(built_at) if built_at else None)


def build(rows: list[dict], verdicts: list[dict], non_analyst=(), marker_key_rows: int = 0,
          rows_ever: int = 0) -> dict:
    """rows: brief_price_yahoo activity rows {ts, ticker, extra_json};
    verdicts: VERDICT_SQL rows; each oldest first. marker_key_rows: verdicts
    in the window with the marker key; rows_ever: activity rows of the stage
    at any time. Returns {"loaded", "counts", "keys"}."""
    loaded = bool(marker_key_rows) or bool(rows_ever)
    skip = set(non_analyst)
    vs: dict = {}
    for v in verdicts:
        vs.setdefault(_key(v.get("symbol"), v.get("brief_built_at")), []).append(v)
    ys: dict = {}
    for y in rows:
        ex = y.get("extra_json") or {}
        ys.setdefault(_key(y.get("ticker"), ex.get("brief_built_at")), []).append(y)
    keys = []
    for k in sorted(set(vs) | set(ys), key=lambda k: (k[1] or "", k[0])):
        vlist, ylist = vs.get(k, []), ys.get(k, [])
        v = next((x for x in vlist if x.get("signal") not in skip), vlist[0] if vlist else None)
        m = (v or {}).get("marker")
        if v is None:
            marker = "no marker"
        elif not (v.get("has_marker_key") in (True, "true", "t")):
            marker = "before load"
        elif isinstance(m, dict):
            marker = m.get("status") or "n/a"
        else:
            marker = "null"
        y = (ylist[0].get("extra_json") or {}) if ylist else {}
        keys.append({
            "ticker": k[0], "brief_built_at": k[1], "signal": (v or {}).get("signal"),
            "signal_rows": len(vlist), "due": due_by_rule(v) if v else None,
            "marker": marker, "marker_error": m.get("error") if isinstance(m, dict) else None,
            "quote_source": (v or {}).get("quote_source"), "is_stale": (v or {}).get("is_stale"),
            "brief_price_age_s": _num((v or {}).get("brief_price_age_s")),
            "yahoo_rows": len(ylist), "yahoo_status": y.get("status") if ylist else "no row",
            "yahoo_error": y.get("error"), "bar_age_s": _num(y.get("bar_age_s")),
            "bar_in_rth": y.get("bar_in_rth"), "yahoo_vs_brief_pct": _num(y.get("yahoo_vs_brief_pct")),
            "queue_delay_s": _num(y.get("queue_delay_s")), "elapsed_s": _num(y.get("elapsed_s")),
        })
    if not loaded:
        counts = dict.fromkeys(COUNT_KEYS, NOT_LOADED)
    else:
        counts = {
            "keys": len(keys),
            "due": sum(1 for r in keys if r["due"]),
            "marker_queued": sum(1 for r in keys if r["marker"] == "queued"),
            "marker_not_queued": sum(1 for r in keys if r["marker"] == "not_queued"),
            "due_marker_null": sum(1 for r in keys if r["due"] and r["marker"] == "null"),
            "due_before_load": sum(1 for r in keys if r["due"] and r["marker"] == "before load"),
            "queued_with_row": sum(1 for r in keys if r["marker"] == "queued" and r["yahoo_rows"]),
            "queued_no_row": sum(1 for r in keys if r["marker"] == "queued" and not r["yahoo_rows"]),
            "row_no_marker": sum(1 for r in keys if r["marker"] == "no marker"),
            "rows_in_window": len(rows),
            "rows_by_status": dict(sorted(Counter(
                (y.get("extra_json") or {}).get("status") or "n/a" for y in rows).items())),
        }
    return {"loaded": loaded, "counts": counts, "keys": keys,
            "marker_key_rows": marker_key_rows, "rows_ever": rows_ever}


def _fmt(v, pct=True):
    if v is None:
        return "n/a"
    return f"{v:+.2f}%" if pct else f"{v:.1f}"


def render(res: dict, env: str, since: str, until: str) -> str:
    L = [f"# brief_price_yahoo ({env}, {since} to {until} ET)", ""]
    if not res["loaded"]:
        L += [f"brief_price_yahoo: {NOT_LOADED} (no verdict in the window carries the marker key; "
              f"no activity row of the stage exists).", ""]
    L += ["| count | value |", "|---|---|"]
    for k in COUNT_KEYS:
        L.append(f"| {k} | {res['counts'][k]} |")
    if res["loaded"]:
        L += ["", "| field | n | min | median | max |", "|---|---|---|---|---|"]
        for k in _DIST:
            vals = [r[k] for r in res["keys"] if r[k] is not None]
            pct = k.endswith("_pct")
            if vals:
                L.append(f"| {k} | {len(vals)} | {_fmt(min(vals), pct)} | "
                         f"{_fmt(statistics.median(vals), pct)} | {_fmt(max(vals), pct)} |")
            else:
                L.append(f"| {k} | 0 | n/a | n/a | n/a |")
    L += ["", "| brief built (UTC) | ticker | signal | signal rows | due | marker | quote source | "
          "stale | brief price age s | Yahoo | bar age s | bar in RTH | Yahoo vs brief | "
          "queue delay s |", "|" + "---|" * 14]
    for r in res["keys"]:
        status = r["yahoo_status"] + (f" ({r['yahoo_error']})" if r["yahoo_error"] else "")
        marker = r["marker"] + (f" ({r['marker_error']})" if r["marker_error"] else "")
        due = "n/a" if r["due"] is None else ("yes" if r["due"] else "no")
        L.append(f"| {(r['brief_built_at'] or 'n/a')[:19]} | {r['ticker']} | {r['signal'] or 'n/a'} | "
                 f"{r['signal_rows']} | {due} | {marker} | {r['quote_source'] or 'n/a'} | "
                 f"{r['is_stale'] if r['is_stale'] is not None else 'n/a'} | "
                 f"{_fmt(r['brief_price_age_s'], False)} | {status} | {_fmt(r['bar_age_s'], False)} | "
                 f"{r['bar_in_rth'] if r['bar_in_rth'] is not None else 'n/a'} | "
                 f"{_fmt(r['yahoo_vs_brief_pct'])} | {_fmt(r['queue_delay_s'], False)} |")
    return "\n".join(L) + "\n"


def _bounds(since: str, until: str) -> tuple[datetime, datetime]:
    """UTC bounds for ET days since..until, both inclusive."""
    lo = datetime.combine(date.fromisoformat(since), time.min, tzinfo=NY)
    hi = datetime.combine(date.fromisoformat(until) + timedelta(days=1), time.min, tzinfo=NY)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def fetch(conn, env: str, since: str, until: str) -> tuple[list, list, int, int]:
    """The reads in one read-only transaction, rolled back."""
    lo, hi = _bounds(since, until)
    params = {"env": env, "since": lo, "until": hi}

    def rows(cur, sql):
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def one(cur, sql):
        cur.execute(sql, params)
        r = cur.fetchone()
        return int(r[0]) if r and r[0] is not None else 0
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            act = rows(cur, ROWS_SQL)
            verdicts = rows(cur, VERDICT_SQL)
            marker_key_rows = one(cur, MARKER_KEY_SQL)
            rows_ever = one(cur, ROWS_EVER_SQL)
    finally:
        conn.rollback()
    return act, verdicts, marker_key_rows, rows_ever


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="brief_price_yahoo: counts and missing rows")
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
        act, verdicts, mk, ever = fetch(conn, env, since, until)
    finally:
        db._put_conn(conn)
    res = build(act, verdicts, getattr(db, "_NON_ANALYST_SIGNALS", ()), mk, ever)
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
