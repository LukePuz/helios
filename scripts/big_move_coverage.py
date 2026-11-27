"""Big-move coverage readout: for a date range, the watchlist ticker-days
whose close-to-close move was at least k ATR, with what the screener and the
analyst did with the ticker that day.

Engineering readout, not a study analysis. k is an argument (default 2, the
DIP_REVIEW_MIN_ATR default), stored in the output; what size of move should
get a review is Luke's. Percentages and ATR multiples only: no prices.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/big_move_coverage.py --start 2026-10-28 --end 2026-11-20 [--k 2] [--json] [--out FILE]

Definitions (also in the output's "definitions"):
  bars        price_history daily bars (the table the book gap-risk record
              reads), every watchlist ticker as knowledge_loader loads it now.
  move_pct    the session's close over the previous session's close, less 1, x 100.
  atr_mult    move_pct over ATR(14) in percent of the previous close (compared
              with k unrounded; shown to 2 decimals): a simple
              14-session mean of the true range over the 15 sessions before
              the day (tasks._atr14, as DIP_REVIEW computes it).
  big         |atr_mult| >= k.
  window      the session's close-to-close window: after 16:00 ET on the
              previous session's date, up to 16:00 ET on the session's date
              (the window the move happened in; a half-day's 13:00 close is
              still read as 16:00).
  items       the screener's SCREENED_IN / SCREENED_OUT signals rows for the
              ticker in the window.
  verdict     an analyst verdict (decision ? model_output, not a routing
              signal) on the ticker in the window.
  all_screened_out  items > 0 and none screened in. The summary also splits
              the big days into a partition: verdict / no verdict and every
              item screened out / no verdict and no items / no verdict but
              screened in.
  own_ticker_rejected  SCREENED_OUT rows that day with
              provenance.own_ticker_rejected true (agents/own_ticker.py);
              rows before that record shipped carry no key (n_own_ticker_recorded).
  ordinary    the sessions under k ATR, and how many had a verdict.

Read-only: one transaction opened READ ONLY with a statement timeout, then
rolled back. Refuses to run unless SUPABASE_PORT is 6543 (the transaction
pooler), so the bot's session pool is never used.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
DEFAULT_K = 2.0
READ_PORT = 6543
STATEMENT_TIMEOUT_MS = 30000
ATR_SESSIONS = 14
LOOKBACK_DAYS = 40          # calendar days of bars read before --start (15 sessions needed)
SESSION_CLOSE = time(16, 0)
DEFINITIONS = {
    "bars": "price_history daily bars; watchlist as knowledge_loader loads it now",
    "move_pct": "(close / previous close - 1) x 100",
    "atr_mult": "move_pct / (ATR(14) / previous close x 100); ATR(14) = mean true range of the "
                "14 sessions before the day (tasks._atr14)",
    "big": "|atr_mult| >= k",
    "window": "after 16:00 ET on the previous session's date, up to 16:00 ET on the session's date",
    "items": "SCREENED_IN and SCREENED_OUT signals rows on the ticker in the window",
    "verdict": "analyst verdict (decision ? model_output, not a routing signal) in the window",
    "all_screened_out": "items > 0 and n_screened_in == 0 (a verdict from another path, such "
                        "as the daily scan, can still exist)",
    "own_ticker_rejected": "SCREENED_OUT rows with provenance.own_ticker_rejected true",
}


def _atr14(bars: list[dict]) -> float | None:
    """tasks._atr14: simple 14-session mean true range from daily bars,
    oldest first; None with fewer than 15."""
    if len(bars) < ATR_SESSIONS + 1:
        return None
    trs = [max(b["high"] - b["low"], abs(b["high"] - p["close"]), abs(b["low"] - p["close"]))
           for p, b in zip(bars[-(ATR_SESSIONS + 1):], bars[-ATR_SESSIONS:])]
    return sum(trs) / len(trs)


def day_moves(bars_by_ticker: dict, start: date, end: date) -> list[dict]:
    """One row per ticker and session in [start, end] that has a previous
    close: {ticker, date, prev_date, move_pct, atr_pct, atr_mult,
    atr_mult_raw}; atr_mult None
    without 15 earlier sessions. Bars: {ticker: [{date, high, low, close}]}."""
    out = []
    for t, bars in sorted(bars_by_ticker.items()):
        bars = sorted((b for b in bars if None not in (b.get("high"), b.get("low"), b.get("close"))),
                      key=lambda b: b["date"])
        for i, b in enumerate(bars):
            if not (start <= b["date"] <= end) or i == 0:
                continue
            prev = bars[i - 1]["close"]
            if not prev:
                continue
            move = (b["close"] / prev - 1) * 100
            atr = _atr14(bars[:i])
            atr_pct = atr / prev * 100 if atr else None
            # atr_mult_raw (unrounded) is what is compared with k.
            out.append({"ticker": t, "date": b["date"], "prev_date": bars[i - 1]["date"],
                        "move_pct": round(move, 2),
                        "atr_pct": round(atr_pct, 2) if atr_pct else None,
                        "atr_mult": round(move / atr_pct, 2) if atr_pct else None,
                        "atr_mult_raw": move / atr_pct if atr_pct else None})
    return out


def coverage(moves: list[dict], signals: list[dict], k: float = DEFAULT_K) -> dict:
    """The readout from day_moves rows and signals rows
    {ticker, ts (tz-aware), kind ("in", "out" or "verdict"),
    own_ticker_rejected} (own_ticker_rejected: True/False/None, "absent"
    when the row has no key). Each signal counts toward the session whose
    window holds it; one outside every session's window is left out."""
    closes = {}
    for m in moves:
        closes.setdefault(m["ticker"], []).append(
            (datetime.combine(m["date"], SESSION_CLOSE, NY),
             datetime.combine(m["prev_date"], SESSION_CLOSE, NY), m["date"]))
    keys = {}
    for t, v in closes.items():
        v.sort()
        keys[t] = [c for c, _, _ in v]
    by = {}
    for s in signals:
        sess = closes.get(s["ticker"]) or []
        i = bisect.bisect_left(keys.get(s["ticker"]) or [], s["ts"])
        if i == len(sess) or not s["ts"] > sess[i][1]:
            continue
        d = by.setdefault((s["ticker"], sess[i][2]),
                          {"in": 0, "out": 0, "verdict": 0, "otr": 0, "otr_recorded": 0})
        d[s["kind"]] += 1
        if s["kind"] == "out" and s.get("own_ticker_rejected") != "absent":
            d["otr_recorded"] += 1
            d["otr"] += s.get("own_ticker_rejected") is True
    rows, ordinary, ordinary_v, no_atr = [], 0, 0, 0
    for m in moves:
        if m["atr_mult"] is None:
            no_atr += 1
            continue
        d = by.get((m["ticker"], m["date"]), {"in": 0, "out": 0, "verdict": 0, "otr": 0,
                                              "otr_recorded": 0})
        if abs(m.get("atr_mult_raw", m["atr_mult"])) < k:
            ordinary += 1
            ordinary_v += d["verdict"] > 0
            continue
        n_items = d["in"] + d["out"]
        rows.append({"ticker": m["ticker"], "date": m["date"].isoformat(),
                     "move_pct": m["move_pct"], "atr_mult": m["atr_mult"],
                     "n_items": n_items, "n_screened_in": d["in"], "n_screened_out": d["out"],
                     "verdict": d["verdict"] > 0, "n_verdicts": d["verdict"],
                     "all_screened_out": n_items > 0 and d["in"] == 0,
                     "n_own_ticker_rejected": d["otr"], "n_own_ticker_recorded": d["otr_recorded"]})
    rows.sort(key=lambda r: (r["date"], -abs(r["atr_mult"]), r["ticker"]))
    n_big = len(rows)
    pct = lambda a, b: round(a / b * 100, 1) if b else None
    n_v = sum(r["verdict"] for r in rows)
    return {"k": k, "n_ticker_days": len(moves), "n_ticker_days_no_atr": no_atr,
            "n_big": n_big, "n_big_verdict": n_v, "big_verdict_pct": pct(n_v, n_big),
            "n_big_all_screened_out": sum(r["all_screened_out"] for r in rows),
            "n_big_all_screened_out_with_verdict": sum(r["all_screened_out"] and r["verdict"]
                                                       for r in rows),
            "n_big_no_items": sum(r["n_items"] == 0 for r in rows),
            # A partition of the big days: a verdict; else every item screened
            # out; else none at all; else screened in with no verdict.
            "n_big_no_verdict_all_screened_out": sum(r["all_screened_out"] and not r["verdict"]
                                                     for r in rows),
            "n_big_no_verdict_no_items": sum(r["n_items"] == 0 and not r["verdict"] for r in rows),
            "n_big_no_verdict_screened_in": sum(r["n_screened_in"] > 0 and not r["verdict"]
                                                for r in rows),
            "n_big_own_ticker_rejected": sum(r["n_own_ticker_rejected"] > 0 for r in rows),
            "n_ordinary": ordinary, "n_ordinary_verdict": ordinary_v,
            "ordinary_verdict_pct": pct(ordinary_v, ordinary),
            "definitions": DEFINITIONS, "rows": rows}


def _et_bounds(start: date, end: date):
    """Signals read: from 16:00 ET a week before start (covers the first
    session's previous close across a holiday weekend) to 16:00 ET on end."""
    return (datetime.combine(start - timedelta(days=7), SESSION_CLOSE, NY),
            datetime.combine(end, SESSION_CLOSE, NY))


def fetch(conn, env: str, tickers: list[str], start: date, end: date) -> tuple[dict, list[dict]]:
    """(bars_by_ticker, signals) in one READ ONLY transaction, rolled back."""
    import db
    non_analyst = [s for s in getattr(db, "_NON_ANALYST_SIGNALS", ()) if s != "SCREENED_IN"]
    lo, hi = _et_bounds(start, end)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute("SELECT ticker, date, high, low, close FROM price_history "
                        "WHERE ticker = ANY(%s) AND date >= %s AND date <= %s",
                        (tickers, start - timedelta(days=LOOKBACK_DAYS), end))
            bars = {}
            for t, d, h, l, c in cur.fetchall():
                bars.setdefault(t, []).append({"date": d, "high": float(h) if h is not None else None,
                                               "low": float(l) if l is not None else None,
                                               "close": float(c) if c is not None else None})
            cur.execute(
                "SELECT symbol, timestamp, signal, "
                "CASE WHEN provenance ? 'own_ticker_rejected' "
                "THEN provenance->>'own_ticker_rejected' ELSE 'absent' END, "
                "decision ? 'model_output' FROM signals "
                "WHERE trading_env = %s AND symbol = ANY(%s) AND timestamp > %s AND timestamp <= %s "
                "AND (signal IN ('SCREENED_IN', 'SCREENED_OUT') "
                "     OR (decision ? 'model_output' AND NOT (signal = ANY(%s))))",
                (env, tickers, lo, hi, non_analyst))
            sigs = []
            for sym, ts, sig, otr, mo in cur.fetchall():
                kind = "in" if sig == "SCREENED_IN" else "out" if sig == "SCREENED_OUT" else "verdict"
                sigs.append({"ticker": sym, "ts": ts, "kind": kind,
                             "own_ticker_rejected": {"true": True, "false": False}.get(otr, None)
                             if otr != "absent" else "absent"})
        return bars, sigs
    finally:
        try:
            conn.rollback()
        except Exception:
            pass


def render(res: dict, start: date, end: date, n_tickers: int) -> str:
    lines = [f"# Big-move coverage, {start} to {end} (k = {res['k']:g} ATR)", "",
             f"- Watchlist names: {n_tickers}; ticker-days with an ATR: "
             f"{res['n_ticker_days'] - res['n_ticker_days_no_atr']} "
             f"(without: {res['n_ticker_days_no_atr']})",
             f"- Days with |move| >= {res['k']:g} ATR: {res['n_big']}; with an analyst verdict: "
             f"{res['n_big_verdict']} ({res['big_verdict_pct']}%); ordinary days with a verdict: "
             f"{res['n_ordinary_verdict']} of {res['n_ordinary']} ({res['ordinary_verdict_pct']}%)",
             f"- Without a verdict: every item screened out {res['n_big_no_verdict_all_screened_out']}, "
             f"no items {res['n_big_no_verdict_no_items']}, screened in "
             f"{res['n_big_no_verdict_screened_in']}",
             f"- Every item screened out, verdict or not: {res['n_big_all_screened_out']} "
             f"(with a verdict from another path: {res['n_big_all_screened_out_with_verdict']}); "
             f"with an own_ticker_rejected row: {res['n_big_own_ticker_rejected']}", "",
             "| date | ticker | move % | ATR x | items | in | out | verdicts | all out | own-ticker rejected (recorded) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['date']} | {r['ticker']} | {r['move_pct']:+.1f} | {r['atr_mult']:+.1f} | "
                     f"{r['n_items']} | {r['n_screened_in']} | {r['n_screened_out']} | {r['n_verdicts']} | "
                     f"{'yes' if r['all_screened_out'] else 'no'} | "
                     f"{r['n_own_ticker_rejected']} ({r['n_own_ticker_recorded']}) |")
    lines += ["", "Definitions:"] + [f"- {k}: {v}" for k, v in res["definitions"].items()]
    return "\n".join(lines) + "\n"


def _bar_qa():
    """scripts/_bar_qa.py (the non-finite bar flag) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bar_qa.py")
    spec = importlib.util.spec_from_file_location("_bar_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first session, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last session, YYYY-MM-DD")
    ap.add_argument("--k", type=float, default=DEFAULT_K, help="ATR multiple (default 2)")
    ap.add_argument("--env", default=None, help="trading_env (default: config.TRADING_ENV)")
    ap.add_argument("--json", action="store_true", help="JSON instead of markdown")
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT") != str(READ_PORT):
        print(f"big_move_coverage: run with SUPABASE_PORT={READ_PORT} (the transaction pooler)",
              file=sys.stderr)
        return 2
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    import db
    from config import TRADING_ENV
    from knowledge_loader import load_watchlist
    tickers = sorted(load_watchlist())
    conn = db._get_conn()
    try:
        bars, sigs = fetch(conn, args.env or TRADING_ENV, tickers, start, end)
    finally:
        db._put_conn(conn)
    res = coverage(day_moves(bars, start, end), sigs, args.k)
    res.update(start=start.isoformat(), end=end.isoformat(), n_watchlist=len(tickers),
               n_watchlist_with_bars=sum(1 for t in tickers if bars.get(t)))
    qa = _bar_qa()   # flag only (scripts/_bar_qa.py): no bar is dropped or changed
    if args.json:
        res["nonfinite_bars"] = qa.flag_record(bars)
        text = json.dumps(res, indent=1, default=str) + "\n"
    else:
        text = render(res, start, end, len(tickers)) + "\n" + qa.flag_line(bars) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
