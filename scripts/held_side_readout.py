"""Held side, one row per held position per day (audit 2026-11-26-0056 #4,
roadmap 04ff; engineering readout, record only).

Joins, by ET day and ticker, the held-side records that have no other reader:
  - position_exit_frame (activity_feed, weekdays 17:30 ET; the day's latest
    row per ticker):
      - the frame: gain_pct, above_stop_pct, close_is_today, broker_held;
      - exit_frame_plan (batch 1134 row 6): plan_read, plan_id,
        plan_revision, invalidation_vs_close_pct, target_vs_close_pct,
        t1_vs_close_pct, horizon_days_left, next_print_days and
        next_print_read;
      - open_risk_to_stop (batch 1134 row 3): open_risk_to_stop_frac (the
        capped risk to the stop), its read, open_risk_gap_adjusted's
        gap_p5_frac / gap_p1_frac and uncapped_weight_frac, shown as % of
        NAV; and the run's open_risk_book once per day;
  - held_udr (signals.provenance.held_udr, batch 1133 row 5, widened by 1134
    row 7; the day's latest verdict per ticker, with the day's count):
    status, udr_now and its components, the first lot's entry UDR (stated
    and udr_at_sizing), the changes since the first and last lot, and
    code_udr_now (read, udr, level_source) with its plan_id / plan_revision;
  - earnings_calendar_check (activity_feed, weekdays 08:00 ET; batch 1134
    rows 2 and 5; the day's latest row, its tickers entry): result,
    next_date, is_estimate, has_future_row, days_since_last_print,
    est_next_print_from_cadence (with cadence_est_days, that date less the
    day), cadence_median_days, cadence_none_reason and catalyst_write.

A ticker is on a day's list when any of the three says it is held there: an
exit frame row (one per position_stops row), a held_udr record other than
no_open_lot, or an earnings check entry with held True.

Keys that are not live yet (they arrive with batches 1133 and 1134) read
"not recorded yet"; a key that is there with no value reads "n/a"; a source
with no row for that day and ticker (for the earnings check, no entry for the
ticker in that day's row) reads "no row". The coverage table counts each; a
group recorded on no row in the window reads "not loaded" there, not 0. A
plan read that raised writes plan_read alone, so its other plan keys read
"not recorded yet" beside plan_read "error:<class>".

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/held_side_readout.py [--env live] [--since YYYY-MM-DD] \\
        [--until YYYY-MM-DD] [--json] [--out FILE]

--since defaults to 14 days before today (ET), --until to today. Read-only:
SUPABASE_PORT must be 6543 (the transaction pooler); one transaction, SET
TRANSACTION READ ONLY with a statement timeout, rolled back. Percentages,
ratios, days, ids and counts only: no prices, share counts or account values.
Descriptive: no thresholds; what any of it should change is a study choice.
Not part of weekly_readouts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 30000
DEFAULT_DAYS = 14

NOT_RECORDED = "not recorded yet"
NO_ROW = "no row"
NOT_LOADED = "not loaded"

FRAME_SQL = ("SELECT ts, ticker, extra_json FROM activity_feed WHERE stage = 'position_exit_frame' "
             "AND ts >= %(since)s AND ts < %(until)s "
             "AND COALESCE(extra_json->>'env', %(env)s) = %(env)s ORDER BY ts")
UDR_SQL = ("SELECT symbol, timestamp, signal, provenance->'held_udr' AS held_udr FROM signals "
           "WHERE trading_env = %(env)s AND timestamp >= %(since)s AND timestamp < %(until)s "
           "AND provenance ? 'held_udr' ORDER BY timestamp")
EARN_SQL = ("SELECT ts, extra_json FROM activity_feed WHERE stage = 'earnings_calendar_check' "
            "AND ts >= %(since)s AND ts < %(until)s "
            "AND COALESCE(extra_json->>'env', %(env)s) = %(env)s ORDER BY ts")

# (column, path into the source record). A path step that is missing reads
# NOT_RECORDED; a present None reads None ("n/a").
FRAME_FIELDS = (
    ("gain_pct", ("gain_pct",)),
    ("above_stop_pct", ("above_stop_pct",)),
    ("close_is_today", ("close_is_today",)),
    ("broker_held", ("broker_held",)),
)
PLAN_FIELDS = (
    ("plan_read", ("plan_read",)),
    ("plan_id", ("plan_id",)),
    ("plan_revision", ("plan_revision",)),
    ("invalidation_vs_close_pct", ("invalidation_vs_close_pct",)),
    ("target_vs_close_pct", ("target_vs_close_pct",)),
    ("t1_vs_close_pct", ("t1_vs_close_pct",)),
    ("horizon_days_left", ("horizon_days_left",)),
    ("next_print_days", ("next_print_days",)),
    ("next_print_read", ("next_print_read",)),
)
RISK_FIELDS = (
    ("open_risk_to_stop_frac", ("open_risk_to_stop_frac",)),
    ("open_risk_read", ("open_risk_to_stop", "read")),
    ("stop_at_or_above_price", ("open_risk_to_stop", "stop_at_or_above_price")),
    ("gap_p5_frac", ("open_risk_gap_adjusted", "gap_p5_frac")),
    ("gap_p1_frac", ("open_risk_gap_adjusted", "gap_p1_frac")),
    ("gap_read", ("open_risk_gap_adjusted", "read")),
    ("uncapped_weight_frac", ("uncapped_weight_frac",)),
)
UDR_FIELDS = (
    ("udr_status", ("status",)),
    ("udr_now", ("udr_now",)),
    ("expected_return_pct_now", ("components_now", "expected_return_pct")),
    ("expected_downside_pct_now", ("components_now", "expected_downside_pct")),
    ("first_udr", ("first", "udr")),
    ("first_udr_at_sizing", ("first", "udr_at_sizing")),
    ("change_since_first", ("change_since_first",)),
    ("change_since_last", ("change_since_last",)),
    ("open_lots", ("open_lots",)),
    ("code_udr_read", ("code_udr_now", "read")),
    ("code_udr_now", ("code_udr_now", "udr")),
    ("code_udr_level_source", ("code_udr_now", "level_source")),
    ("udr_plan_id", ("plan_id",)),
    ("udr_plan_revision", ("plan_revision",)),
)
EARN_FIELDS = (
    ("earn_result", ("result",)),
    ("earn_next_date", ("next_date",)),
    ("earn_is_estimate", ("is_estimate",)),
    ("has_future_row", ("has_future_row",)),
    ("days_since_last_print", ("days_since_last_print",)),
    ("est_next_print_from_cadence", ("est_next_print_from_cadence",)),
    ("cadence_median_days", ("cadence_median_days",)),
    ("cadence_none_reason", ("cadence_none_reason",)),
    ("catalyst_write", ("catalyst_write",)),
)
BOOK_FIELDS = (
    ("book_open_risk_to_stop_frac", ("book_open_risk_to_stop_frac",)),
    ("book_uncapped_weight_frac", ("book_uncapped_weight_frac",)),
    ("book_open_risk_gap_p5_frac", ("book_open_risk_gap_p5_frac",)),
    ("book_open_risk_gap_p1_frac", ("book_open_risk_gap_p1_frac",)),
    ("n_held_without_stop_row", ("n_held_without_stop_row",)),
    ("snapshot_read", ("snapshot_read",)),
    ("n_by_read", ("n_by_read",)),
)
# Columns that hold a fraction of NAV, shown as a percentage.
_FRAC = {"open_risk_to_stop_frac", "gap_p5_frac", "gap_p1_frac", "uncapped_weight_frac",
         "book_open_risk_to_stop_frac", "book_uncapped_weight_frac",
         "book_open_risk_gap_p5_frac", "book_open_risk_gap_p1_frac"}
# Columns already in percent.
_PCT = {"gain_pct", "above_stop_pct", "invalidation_vs_close_pct", "target_vs_close_pct",
        "t1_vs_close_pct", "expected_return_pct_now", "expected_downside_pct_now"}
# Written only on some rows even when live (uncapped_weight_frac: a held row
# whose stop row has no usable stop price), so not counted in coverage.
_CONDITIONAL = {"uncapped_weight_frac"}
_GROUPS = (("frame", FRAME_FIELDS), ("plan", PLAN_FIELDS), ("open_risk", RISK_FIELDS),
           ("held_udr", UDR_FIELDS), ("earnings", EARN_FIELDS))


def _pick(rec, path):
    """The value at path in rec; NO_ROW when rec is None, NOT_RECORDED when
    a step is missing (or a step above it is not a dict)."""
    if rec is None:
        return NO_ROW
    cur = rec
    for step in path:
        if not isinstance(cur, dict) or step not in cur:
            return NOT_RECORDED
        cur = cur[step]
    return cur


def _et_day(ts) -> str | None:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(NY).date().isoformat()
    try:
        return _et_day(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def _as_dict(v) -> dict | None:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None
        return v if isinstance(v, dict) else None
    return None


def _days_between(later, day: str):
    try:
        return (date.fromisoformat(str(later)[:10]) - date.fromisoformat(day)).days
    except (TypeError, ValueError):
        return None


def build_rows(frames: list[dict], udrs: list[dict], earnings: list[dict]) -> tuple[list[dict], list[dict]]:
    """frames: activity rows {ts, ticker, extra_json} of position_exit_frame;
    udrs: signals rows {symbol, timestamp, signal, held_udr}; earnings:
    activity rows {ts, extra_json} of earnings_calendar_check. Each oldest
    first. Returns (rows, book): one row per (ET day, ticker) held by any
    source, and one book row per day with exit frame rows."""
    frame_by, book_by = {}, {}
    for f in frames:
        ex = _as_dict(f.get("extra_json")) or {}
        day, t = _et_day(f.get("ts")), str(f.get("ticker") or "").upper()
        if day is None or not t:
            continue
        frame_by[(day, t)] = ex                      # oldest first: the latest wins
        book_by[day] = ex.get("open_risk_book", NOT_RECORDED)
    udr_by, udr_n = {}, {}
    for s in udrs:
        rec = _as_dict(s.get("held_udr"))
        day, t = _et_day(s.get("timestamp")), str(s.get("symbol") or "").upper()
        if day is None or not t or rec is None:
            continue
        udr_by[(day, t)] = {**rec, "_signal": s.get("signal")}
        udr_n[(day, t)] = udr_n.get((day, t), 0) + 1
    earn_by = {}
    for e in earnings:
        ex = _as_dict(e.get("extra_json")) or {}
        day = str(ex.get("day") or "")[:10] or _et_day(e.get("ts"))
        if day:
            earn_by[day] = ex
    keys = set(frame_by)
    keys |= {k for k, r in udr_by.items() if r.get("status") != "no_open_lot"}
    for day, ex in earn_by.items():
        for t, r in (ex.get("tickers") or {}).items():
            if isinstance(r, dict) and r.get("held") is True:
                keys.add((day, str(t).upper()))
    rows = []
    for day, t in sorted(keys):
        fr = frame_by.get((day, t))
        ud = udr_by.get((day, t))
        ex = earn_by.get(day)
        tick = (ex or {}).get("tickers")
        er = tick.get(t) if isinstance(tick, dict) and isinstance(tick.get(t), dict) else None
        row = {"day": day, "ticker": t,
               "sources": "/".join(n for n, v in (("frame", fr), ("held_udr", ud), ("earnings", er))
                                   if v is not None),
               "udr_verdicts": udr_n.get((day, t), 0),
               "udr_signal": ud.get("_signal") if ud else None}
        for col, path in FRAME_FIELDS + PLAN_FIELDS + RISK_FIELDS:
            row[col] = _pick(fr, path)
        for col, path in UDR_FIELDS:
            row[col] = _pick(ud, path)
        for col, path in EARN_FIELDS:
            row[col] = _pick(er, path)
        est = row["est_next_print_from_cadence"]
        row["cadence_est_days"] = (est if est in (NO_ROW, NOT_RECORDED)
                                   else _days_between(est, day) if isinstance(est, str) else None)
        rows.append(row)
    book = []
    for day in sorted(book_by):
        b = book_by[day]
        brow = {"day": day}
        for col, path in BOOK_FIELDS:
            brow[col] = _pick(b, path) if isinstance(b, dict) else NOT_RECORDED
        book.append(brow)
    return rows, book


def _fmt(col, v) -> str:
    if v in (NO_ROW, NOT_RECORDED):
        return v
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, dict):
        return ", ".join(f"{k} {v[k]}" for k in sorted(v)) or "{}"
    if col in _FRAC and isinstance(v, (int, float)):
        return f"{v * 100:.2f}% NAV"
    if col in _PCT and isinstance(v, (int, float)):
        return f"{v:+.2f}%"
    return str(v)


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(_fmt(c, r.get(c)) for c in cols) + " |")
    return L


def coverage(rows: list[dict]) -> list[dict]:
    """Per source group: rows with the group recorded (any value), with the
    keys not recorded yet (keys written only on some rows, _CONDITIONAL,
    are left out), and with no source row. A group recorded on no row in
    the window reads NOT_LOADED in every count, not 0."""
    out = []
    for name, fields in _GROUPS:
        cols = [c for c, _ in fields if c not in _CONDITIONAL]
        n_row = sum(1 for r in rows if all(r[c] == NO_ROW for c in cols))
        n_nr = sum(1 for r in rows if any(r[c] == NOT_RECORDED for c in cols)
                   and not all(r[c] == NO_ROW for c in cols))
        rec = len(rows) - n_row - n_nr
        if rec:
            out.append({"group": name, "rows": len(rows), "no_row": n_row,
                        "not_recorded_yet": n_nr, "recorded": rec})
        else:
            out.append({"group": name, "rows": len(rows), "no_row": NOT_LOADED,
                        "not_recorded_yet": NOT_LOADED, "recorded": NOT_LOADED})
    return out


def render(rows: list[dict], book: list[dict], env: str, since: str, until: str) -> str:
    L = [f"# Held side by day ({env}, {since} to {until} ET)", "",
         f"{len(rows)} position-days, {len({r['ticker'] for r in rows})} tickers, "
         f"{len({r['day'] for r in rows})} days.", "",
         "Coverage (a group counts as not recorded yet when any of its keys is missing; "
         "a group recorded on no row in the window reads not loaded):", ""]
    L += _table(coverage(rows), ["group", "rows", "recorded", "not_recorded_yet", "no_row"])
    sections = (
        ("Plan and distance to its levels (exit frame)",
         ["sources", "gain_pct", "above_stop_pct", "close_is_today", "broker_held"]
         + [c for c, _ in PLAN_FIELDS]),
        ("Open risk (exit frame)", [c for c, _ in RISK_FIELDS]),
        ("held_udr (the day's latest verdict)",
         ["udr_verdicts", "udr_signal"] + [c for c, _ in UDR_FIELDS]),
        ("Next print (earnings calendar check)",
         ["next_print_days"] + [c for c, _ in EARN_FIELDS] + ["cadence_est_days"]),
    )
    for title, cols in sections:
        L += ["", f"## {title}", ""]
        L += _table(rows, ["day", "ticker"] + cols)
    L += ["", "## Book open risk (open_risk_book, one per day)", ""]
    L += _table(book, ["day"] + [c for c, _ in BOOK_FIELDS])
    return "\n".join(L) + "\n"


def _bounds(since: str, until: str) -> tuple[datetime, datetime]:
    """UTC bounds for ET days since..until, both inclusive."""
    lo = datetime.combine(date.fromisoformat(since), time.min, tzinfo=NY)
    hi = datetime.combine(date.fromisoformat(until) + timedelta(days=1), time.min, tzinfo=NY)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def fetch(conn, env: str, since: str, until: str) -> tuple[list, list, list]:
    """The three reads in one read-only transaction, rolled back."""
    lo, hi = _bounds(since, until)
    params = {"env": env, "since": lo, "until": hi}

    def rows(cur, sql):
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            frames = rows(cur, FRAME_SQL)
            udrs = rows(cur, UDR_SQL)
            earn = rows(cur, EARN_SQL)
    finally:
        conn.rollback()
    return frames, udrs, earn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Held side, one row per held position per day")
    ap.add_argument("--env", help="trading env (default config.TRADING_ENV)")
    ap.add_argument("--since", help="first ET day, YYYY-MM-DD (default 14 days ago)")
    ap.add_argument("--until", help="last ET day, YYYY-MM-DD (default today)")
    ap.add_argument("--json", action="store_true", help="rows and book as JSON")
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
        frames, udrs, earn = fetch(conn, env, since, until)
    finally:
        db._put_conn(conn)
    rows, book = build_rows(frames, udrs, earn)
    if args.json:
        text = json.dumps({"env": env, "since": since, "until": until, "rows": rows,
                           "book": book, "coverage": coverage(rows)}, default=str, indent=1) + "\n"
    else:
        text = render(rows, book, env, since, until)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
