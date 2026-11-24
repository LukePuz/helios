"""Fill-lag readout: how long after a broker exit the bot knew of it.

Engineering diagnostic, not a study analysis. For each exit the bot did not
place (trade_outcomes exit_source 'broker_exit', grouped per ticker and fill
time, since --since) it reports:
  - fill: exit_fill_at, the broker's execution time;
  - seen: the broker_sell_seen activity row (tasks.watch_broker_sells; none
    before that task existed) of the ticker whose first execution time is
    closest to the exit's fill, within SEEN_MATCH_TOLERANCE;
  - booked: the reconcile's SELL trade row (the earliest at or after the fill);
  - lags fill -> seen and fill -> booked, in minutes;
  - decisions on the ticker inside [fill, booked): signals rows by kind,
    pending_approvals created or decided, and deferred-retry rows.
Times and counts only: no prices, quantities or account values.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/fill_lag_readout.py [--since 2026-08-16] [--out FILE]

Read-only: one transaction opened READ ONLY, then rolled back.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY = ZoneInfo("America/New_York")
DECISION_KINDS_SKIPPED = ("SCREENED_OUT",)      # screener noise; counted separately
SEEN_MATCH_TOLERANCE = timedelta(minutes=30)    # seen row's first execution vs the exit's fill


def _aware(t):
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def exits(outcome_rows) -> list[dict]:
    """outcome_rows: (ticker, exit_fill_at, exit_date). One exit per (ticker,
    fill time); several lots closed by one fill share it."""
    seen, out = set(), []
    for ticker, fill_at, exit_date in outcome_rows:
        fill = _aware(fill_at)
        key = (ticker, fill or _aware(exit_date))
        if key in seen:
            continue
        seen.add(key)
        out.append({"ticker": ticker, "fill": fill, "exit_date": _aware(exit_date)})
    return sorted(out, key=lambda e: (e["fill"] or e["exit_date"]))


def attach(ex: list[dict], trade_rows, seen_rows, signal_rows, approval_rows) -> list[dict]:
    """trade_rows: (ticker, ts) SELLs. seen_rows: (ticker, ts, first_exec_at).
    signal_rows: (ticker, ts, kind). approval_rows: (ticker, created, decided)."""
    for e in ex:
        t0 = e["fill"]
        e["booked"] = min((_aware(ts) for tk, ts in trade_rows
                           if tk == e["ticker"] and t0 and _aware(ts) >= t0
                           and _aware(ts) <= t0 + timedelta(days=1)), default=None)
        e["seen"] = None
        if t0:
            best = None
            for tk, ts, first in seen_rows:
                try:
                    f = _aware(first) if first else None
                except (TypeError, ValueError):
                    f = None
                if tk != e["ticker"] or f is None or abs(f - t0) > SEEN_MATCH_TOLERANCE:
                    continue
                if best is None or abs(f - t0) < best[0]:
                    best = (abs(f - t0), _aware(ts))
            e["seen"] = best[1] if best else None
        e["lag_booked_min"] = ((e["booked"] - t0).total_seconds() / 60
                               if t0 and e["booked"] else None)
        e["lag_seen_min"] = ((e["seen"] - t0).total_seconds() / 60 if t0 and e["seen"] else None)
        kinds: dict = {}
        approvals = 0
        if t0 and e["booked"]:
            for tk, ts, kind in signal_rows:
                if tk == e["ticker"] and t0 <= _aware(ts) < e["booked"]:
                    kinds[kind] = kinds.get(kind, 0) + 1
            for tk, created, decided in approval_rows:
                if tk != e["ticker"]:
                    continue
                if any(x is not None and t0 <= _aware(x) < e["booked"] for x in (created, decided)):
                    approvals += 1
        e["signals_in_window"] = kinds
        e["approvals_in_window"] = approvals
    return ex


def _fmt(t):
    return t.astimezone(NY).strftime("%Y-%m-%d %H:%M ET") if t else "-"


def readout(ex: list[dict], since: date, env: str, head: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Fill-lag readout ({env}, exits since {since})", "",
             f"Generated {now} at code {head}. Engineering diagnostic. Times and counts only.", "",
             "| ticker | fill | seen | booked | fill->seen min | fill->booked min | decisions in window | approvals |",
             "|---|---|---|---|---|---|---|---|"]
    for e in ex:
        dec = ", ".join(f"{k} {v}" for k, v in sorted(e["signals_in_window"].items())
                        if k not in DECISION_KINDS_SKIPPED) or "none"
        skipped = sum(v for k, v in e["signals_in_window"].items() if k in DECISION_KINDS_SKIPPED)
        if skipped:
            dec += f" (+{skipped} screened out)"
        ls, lb = e["lag_seen_min"], e["lag_booked_min"]
        lines.append(f"| {e['ticker']} | {_fmt(e['fill'])} | {_fmt(e['seen'])} | {_fmt(e['booked'])} | "
                     f"{'-' if ls is None else f'{ls:.1f}'} | {'-' if lb is None else f'{lb:.1f}'} | "
                     f"{dec} | {e['approvals_in_window']} |")
    for name, key in (("fill -> booked", "lag_booked_min"), ("fill -> seen", "lag_seen_min")):
        vals = [e[key] for e in ex if e[key] is not None]
        if vals:
            lines += ["", f"{name}: n {len(vals)}, median {statistics.median(vals):.1f} min, "
                          f"min {min(vals):.1f}, max {max(vals):.1f}."]
    no_fill = sum(1 for e in ex if e["fill"] is None)
    if no_fill:
        lines += ["", f"{no_fill} exit(s) have no broker fill time (exit_fill_at), so no lag."]
    return "\n".join(lines) + "\n"


def _fetch(since: date, env: str):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT ticker, exit_fill_at, exit_date FROM trade_outcomes "
                        "WHERE trading_env=%s AND exit_source='broker_exit' "
                        "AND COALESCE(exit_fill_at, exit_date) >= %s", (env, start))
            outcomes = cur.fetchall()
            tickers = sorted({r[0] for r in outcomes})
            cur.execute("SELECT symbol, timestamp FROM trades WHERE side='SELL' AND trading_env=%s "
                        "AND timestamp >= %s AND symbol = ANY(%s)", (env, start, tickers))
            trades = cur.fetchall()
            cur.execute("SELECT ticker, ts, extra_json->>'first_exec_at' FROM activity_feed "
                        "WHERE stage='broker_sell_seen' AND ts >= %s AND ticker = ANY(%s) "
                        "AND coalesce(extra_json->>'env', %s) = %s", (start, tickers, env, env))
            seen = cur.fetchall()
            cur.execute("SELECT symbol, timestamp, signal FROM signals WHERE timestamp >= %s "
                        "AND symbol = ANY(%s) AND (trading_env = %s OR trading_env IS NULL)",
                        (start, tickers, env))
            signals = cur.fetchall()
            cur.execute("SELECT ticker, timestamp, decided_at FROM pending_approvals "
                        "WHERE trading_env=%s AND timestamp >= %s - interval '2 days' "
                        "AND ticker = ANY(%s)", (env, start, tickers))
            approvals = cur.fetchall()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return outcomes, trades, seen, signals, approvals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-16", help="YYYY-MM-DD, New York date")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since = date.fromisoformat(args.since)
    from config import TRADING_ENV
    outcomes, trades, seen, signals, approvals = _fetch(since, TRADING_ENV)
    ex = attach(exits(outcomes), trades, seen, signals, approvals)
    try:
        import subprocess
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = readout(ex, since, TRADING_ENV, head or "unknown")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
