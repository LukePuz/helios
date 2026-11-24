"""Execution quality (implementation shortfall), an engineering readout, not a
study analysis. Basis points of price, signed so positive = worse for us.

Entries and bot sells (orders rows, filled, of the env since --since):
  vs decision  fill vs orders.decision_price (the price at sizing, the route
               price): BUY (fill / decision - 1) * 1e4, SELL
               (1 - fill / decision) * 1e4.
  vs submit    the same against orders.submit_price (the quote read just
               before the submit: the last 1-minute bar's close, the same read
               as the decision price on an immediate entry).
  vs brief     the same against risk_json.arrival.brief_price, the price when
               the analyst's brief was built (approvals since 11-10).
  split by     order type; submit time (ET): open_15 (09:30-09:45 on a
               session), later_rth (09:45-16:00), outside_rth; path:
               deferred (submitted more than DEFERRED_MIN_MINUTES after its
               approval was written: a deferred entry's retry) or immediate;
               submit-to-fill lag (<= 5 s, 5-30 s, > 30 s).
Stop fills (activity rows stop_slippage, env from the row): fill vs the
  recorded stop, (1 - fill / stop) * 1e4, and vs the broker's own trigger
  where read. gap_through: filled at the open (09:30-09:35, the row's
  at_open) or outside regular hours; normal otherwise. Broker sells (trades
  rows SELL with no order row) that have no stop_slippage row within 2 h are
  counted, not priced.
Partial stop fills (activity rows stop_slippage_partial, env from the row):
  one booking of a still-held long's partial exit, priced at the booked
  order's day VWAP at booking (not per execution), measured the same way, in
  a section of its own. Never pooled with the stop fills: a same-day
  stop_slippage row's price already includes these executions. The count of
  broker sells with no stop_slippage row is unchanged; how many of them have
  a partial row within 2 h is reported beside it.
Missing or NULL fields are counted per field and never imputed: a row
without the price a measure needs is left out of that measure only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/execution_quality.py [--since 2026-10-28] [--env live] [--out FILE]

Read-only: one DB transaction opened READ ONLY with a short statement
timeout, then rolled back. Output is basis points and counts only: no
prices, quantities or account values.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from market_calendar import NYSE_HOLIDAYS  # noqa: E402  (no broker imports)

NY = ZoneInfo("America/New_York")
DEFAULT_SINCE = date(2026, 10, 28)
DEFERRED_MIN_MINUTES = 10
LAG_BUCKETS = ((5, "<=5s"), (30, "5-30s"))       # above the last: ">30s"
UNMATCHED_WINDOW = timedelta(hours=2)
SUSPECT_LAG_S = 3600     # a submit-to-fill lag above this is listed, not summarised
STATEMENT_TIMEOUT_MS = 10000


def _aware(ts):
    if ts is None:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _pos(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 and f == f else None


def bp_buy(fill, ref) -> float | None:
    f, r = _pos(fill), _pos(ref)
    return (f / r - 1) * 1e4 if f and r else None


def bp_sell(fill, ref) -> float | None:
    f, r = _pos(fill), _pos(ref)
    return (1 - f / r) * 1e4 if f and r else None


def open_window(ts) -> str | None:
    ts = _aware(ts)
    if ts is None:
        return None
    et = ts.astimezone(NY)
    t = et.time()
    if et.weekday() >= 5 or et.date() in NYSE_HOLIDAYS or not time(9, 30) <= t < time(16, 0):
        return "outside_rth"
    return "open_15" if t < time(9, 45) else "later_rth"


def lag_bucket(seconds) -> str | None:
    if seconds is None:
        return None
    for limit, name in LAG_BUCKETS:
        if seconds <= limit:
            return name
    return f">{LAG_BUCKETS[-1][0]}s"


def entry_row(o: dict) -> dict:
    """One filled orders row as its measures; None where an input is missing."""
    side = o.get("side")
    bp = bp_buy if side == "BUY" else bp_sell
    sub, fil, appr = _aware(o.get("submitted_at")), _aware(o.get("filled_at")), _aware(o.get("approval_at"))
    lag = (fil - sub).total_seconds() if sub and fil else None
    path = None
    if sub and appr:
        path = "deferred" if (sub - appr) > timedelta(minutes=DEFERRED_MIN_MINUTES) else "immediate"
    return {"side": side, "intent": o.get("intent") or "none",
            "order_type": o.get("order_type") or "none",
            "bp_decision": bp(o.get("fill_price"), o.get("decision_price")),
            "bp_submit": bp(o.get("fill_price"), o.get("submit_price")),
            "bp_brief": bp(o.get("fill_price"), o.get("brief_price")),
            "decision_is_submit": (_pos(o.get("decision_price")) is not None
                                   and _pos(o.get("decision_price")) == _pos(o.get("submit_price"))),
            "window": open_window(sub), "path": path,
            "lag_s": lag if lag is None or lag <= SUSPECT_LAG_S else None,
            "lag_suspect": lag is not None and lag > SUSPECT_LAG_S,
            "lag": "suspect" if lag is not None and lag > SUSPECT_LAG_S else lag_bucket(lag)}


def stop_row(extra: dict) -> dict:
    e = extra if isinstance(extra, dict) else {}
    slip = e.get("slippage_pct")
    trig = e.get("slippage_vs_broker_trigger_pct")
    try:
        bp_stop = -float(slip) * 100 if slip is not None else None
    except (TypeError, ValueError):
        bp_stop = None
    try:
        bp_trig = -float(trig) * 100 if trig is not None else None
    except (TypeError, ValueError):
        bp_trig = None
    at_open, outside = e.get("at_open"), e.get("outside_rth")
    kind = (None if at_open is None and outside is None
            else "gap_through" if (at_open is True or outside is True) else "normal")
    return {"bp_stop": bp_stop, "bp_trigger": bp_trig, "kind": kind,
            "order_type": e.get("broker_order_type") or "none",
            "trailing": e.get("trailing_active")}


def broker_sells_unmatched(sells, stop_rows) -> int:
    """Broker sells (dicts {ticker, ts}) with no stop_slippage row (dicts
    {ticker, ts}) on the same ticker within UNMATCHED_WINDOW."""
    n = 0
    for s in sells:
        ts = _aware(s.get("ts"))
        if not any(r.get("ticker") == s.get("ticker") and ts and _aware(r.get("ts"))
                   and abs(_aware(r["ts"]) - ts) <= UNMATCHED_WINDOW for r in stop_rows):
            n += 1
    return n


def _stats(vals) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "n 0"
    return (f"n {len(v)}, median {statistics.median(v):+.1f}, "
            f"range {min(v):+.1f} to {max(v):+.1f}")


def _group(rows, key, measure) -> list[str]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(r.get(key) or "missing", []).append(r.get(measure))
    return [f"| {g} | {_stats(v)} |" for g, v in sorted(groups.items())]


def _missing(orders) -> Counter:
    c = Counter()
    for o in orders:
        for f in ("fill_price", "decision_price", "submit_price", "submitted_at", "filled_at",
                  "approval_at", "brief_price"):
            if o.get(f) is None:
                c[f] += 1
        if o.get("filled_at") is not None and o.get("filled_at_source") is None:
            c["filled_at_source"] += 1
    return c


def build(orders, stop_extras, broker_sells, stop_rows_meta, since, env, head,
          partial_rows=None) -> str:
    """Markdown. orders: filled orders rows (dicts, approval_at joined);
    stop_extras: stop_slippage extra_json dicts of this env; broker_sells and
    stop_rows_meta: dicts {ticker, ts}; partial_rows: stop_slippage_partial
    rows of this env (dicts {ts, ticker, extra}), reported on their own."""
    buys = [o for o in orders if o.get("side") == "BUY"]
    sells = [o for o in orders if o.get("side") == "SELL"]
    er = [entry_row(o) for o in buys]
    sr = [entry_row(o) for o in sells]
    st = [stop_row(e) for e in stop_extras]
    lines = [f"# Execution quality ({env}) since {since}", "",
             f"Code {head}. Basis points of price, positive = worse for us. Missing fields "
             "are counted, never imputed.", "",
             "## Entries (filled BUY orders)", "",
             f"Filled BUYs: {len(buys)}.", "",
             "| measure | value |", "|---|---|",
             f"| vs decision price | {_stats(r['bp_decision'] for r in er)} |",
             f"| vs quote at submit | {_stats(r['bp_submit'] for r in er)} |",
             f"| vs price when the brief was built | {_stats(r['bp_brief'] for r in er)} |",
             f"| submit-to-fill lag, seconds | {_stats(r['lag_s'] for r in er)} |",
             f"| lag over {SUSPECT_LAG_S} s (listed, not summarised) "
             f"| {sum(r['lag_suspect'] for r in er)} |",
             f"| decision price equal to the quote at submit "
             f"| {sum(r['decision_is_submit'] for r in er)} of {len(er)} |"]
    for key, title in (("order_type", "order type"), ("window", "submit time (ET)"),
                       ("path", "path (deferred = submitted > "
                                f"{DEFERRED_MIN_MINUTES} min after the approval)"),
                       ("lag", "submit-to-fill lag"), ("intent", "intent")):
        lines += ["", f"### vs decision price, by {title}", "", "| group | bp |", "|---|---|"]
        lines += _group(er, key, "bp_decision")
        lines += ["", f"### vs quote at submit, by {title}", "", "| group | bp |", "|---|---|"]
        lines += _group(er, key, "bp_submit")

    lines += ["", "## Bot sells (filled SELL orders: profit targets, TRIM, EXIT)", "",
              f"Filled SELLs: {len(sells)}."]
    if sr:
        lines += ["", "| intent | vs decision price, bp | vs quote at submit, bp |", "|---|---|---|"]
        by: dict = {}
        for r in sr:
            by.setdefault(r["intent"], []).append(r)
        for k, rows in sorted(by.items()):
            lines.append(f"| {k} | {_stats(r['bp_decision'] for r in rows)} "
                         f"| {_stats(r['bp_submit'] for r in rows)} |")

    lines += ["", "## Stop fills (stop_slippage rows)", "",
              f"Rows: {len(st)}. Broker sells with no stop_slippage row within "
              f"{int(UNMATCHED_WINDOW.total_seconds() // 3600)} h: "
              f"{broker_sells_unmatched(broker_sells, stop_rows_meta)} of {len(broker_sells)}.", "",
              "| group | vs recorded stop, bp | vs broker trigger, bp |", "|---|---|---|",
              f"| all | {_stats(r['bp_stop'] for r in st)} | {_stats(r['bp_trigger'] for r in st)} |"]
    by = {}
    for r in st:
        by.setdefault(r["kind"] or "missing", []).append(r)
    for k, rows in sorted(by.items()):
        lines.append(f"| {k} | {_stats(r['bp_stop'] for r in rows)} "
                     f"| {_stats(r['bp_trigger'] for r in rows)} |")

    partial_rows = list(partial_rows or [])
    pt = [stop_row(r.get("extra")) for r in partial_rows]
    pmeta = [{"ticker": r.get("ticker"), "ts": r.get("ts")} for r in partial_rows]
    covered = (broker_sells_unmatched(broker_sells, stop_rows_meta)
               - broker_sells_unmatched(broker_sells, list(stop_rows_meta) + pmeta))
    lines += ["", "## Partial stop fills (stop_slippage_partial rows; order's day VWAP at booking)", "",
              f"Rows: {len(pt)}. Not pooled with the stop fills above: a same-day "
              "stop_slippage row's price already includes these executions. Broker sells "
              f"counted above with no stop_slippage row that have a partial row within "
              f"{int(UNMATCHED_WINDOW.total_seconds() // 3600)} h: {covered}.", "",
              "| group | vs recorded stop, bp | vs broker trigger, bp |", "|---|---|---|",
              f"| all | {_stats(r['bp_stop'] for r in pt)} | {_stats(r['bp_trigger'] for r in pt)} |"]

    miss = _missing(orders)
    st_miss = Counter()
    for e in stop_extras:
        for f in ("slippage_pct", "slippage_vs_broker_trigger_pct", "at_open", "outside_rth"):
            if (e or {}).get(f) is None:
                st_miss[f] += 1
    lines += ["", "## Missing fields", "",
              "orders: " + (", ".join(f"{k} {v}" for k, v in sorted(miss.items())) or "none")
              + f" (of {len(orders)}).",
              "stop_slippage: " + (", ".join(f"{k} {v}" for k, v in sorted(st_miss.items()))
                                   or "none") + f" (of {len(stop_extras)})."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def _fetch(since: date, env: str):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY).astimezone(timezone.utc)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT o.side, o.order_type, o.intent, o.decision_price, o.submit_price, "
                        "o.fill_price, o.submitted_at, o.filled_at, o.filled_at_source, a.timestamp, "
                        "a.risk_json->'arrival'->>'brief_price' "
                        "FROM orders o LEFT JOIN pending_approvals a ON a.id = o.approval_id "
                        "WHERE o.trading_env = %s AND o.created_at >= %s "
                        "AND o.status = 'filled' AND coalesce(o.filled_qty, 0) > 0",
                        (env, start))
            cols = ("side", "order_type", "intent", "decision_price", "submit_price", "fill_price",
                    "submitted_at", "filled_at", "filled_at_source", "approval_at", "brief_price")
            orders = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.execute("SELECT ts, ticker, extra_json FROM activity_feed "
                        "WHERE stage = 'stop_slippage' AND ts >= %s", (start,))
            stop_rows = [{"ts": t, "ticker": k, "extra": e} for t, k, e in cur.fetchall()
                         if (e or {}).get("env") == env]
            cur.execute("SELECT ts, ticker, extra_json FROM activity_feed "
                        "WHERE stage = 'stop_slippage_partial' AND ts >= %s", (start,))
            partial_rows = [{"ts": t, "ticker": k, "extra": e} for t, k, e in cur.fetchall()
                            if (e or {}).get("env") == env]
            cur.execute("SELECT symbol, timestamp FROM trades WHERE trading_env = %s "
                        "AND side = 'SELL' AND order_row_id IS NULL AND timestamp >= %s",
                        (env, start))
            broker_sells = [{"ticker": s, "ts": t} for s, t in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return orders, stop_rows, broker_sells, partial_rows


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=DEFAULT_SINCE.isoformat())
    ap.add_argument("--env", default="live")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    since = date.fromisoformat(a.since)
    orders, stop_rows, broker_sells, partial_rows = _fetch(since, a.env)
    text = build(orders, [r["extra"] for r in stop_rows], broker_sells,
                 [{"ticker": r["ticker"], "ts": r["ts"]} for r in stop_rows], since, a.env,
                 _git_head(), partial_rows=partial_rows)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
