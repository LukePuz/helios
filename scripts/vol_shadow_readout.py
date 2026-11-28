"""Volatility shadow readout: what size_position's record-only
vol_target_shadow and stop_vol would have said on past buy fills.

Engineering diagnostic, not a study analysis. For each BUY fill in trades it
rebuilds the inputs the live shadow sees at sizing time and runs the live
helper itself (risk_manager._vol_shadow_record) on them, so the numbers come
from the same code:
  - bars: completed price_history bars dated before the fill's New York date;
  - weight: fill quantity times fill price over the newest portfolio
    snapshot at or before the fill (same env), plus the name's position
    before the fill from the trades ledger (buys minus sells), valued at the
    fill price;
  - stop: the approval's stop_loss_price, via orders.approval_id, else the
    newest APPROVED approval of the name in the same trading_env in the 2
    days before the fill;
  - ATR: the mean true range of the 19 cached bars before the fill (the
    simulation path of RiskManager.get_atr); the ATR multiple is the bucket's
    ATR_STOP_MULTIPLIER midpoint when the approval's stop_method was an ATR
    stop, else none.
It also gives each fill's 20-session return minus SPY from the fill-day
close, for context only. When either close of that return (the name's or
SPY's) is NaN or infinite, x20 is "non-finite" with the ticker named, and
the row is kept out of the x20 mean; no bar is dropped or changed (whether
to is Luke's; scripts/_bar_qa.py flags the bars). Output is percentages and ratios: no prices, share
counts, dollar amounts or account values. Read-only: one transaction opened
READ ONLY, then rolled back.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/vol_shadow_readout.py [--env live|paper|all] [--out FILE]
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import math
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY = ZoneInfo("America/New_York")
ATR_BARS = 19          # _cached_bars_for_atr(days=14) reads days + 5 rows
NONFINITE = "non-finite"   # _x20 when a close it reads is NaN or infinite


def nonfinite(v) -> bool:
    """True for a NaN or infinite number; None and non-numbers are not."""
    if v is None or isinstance(v, bool):
        return False
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):
        return False


def _ny_date(ts: datetime) -> date:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(NY).date()


def _bars_before(bars: list[dict], d: date) -> list[dict]:
    """Bars dated strictly before d (bars sorted by ISO date string)."""
    keys = [b["date"] for b in bars]
    return bars[:bisect.bisect_left(keys, d.isoformat())]


def _atr(bars: list[dict]) -> float | None:
    import risk_manager
    tail = bars[-ATR_BARS:]
    return risk_manager._compute_atr([{"h": b["high"], "l": b["low"], "c": b["close"]} for b in tail])


def x20_detail(bars: list[dict], spy: list[dict], d: date) -> tuple[float | None, str | None]:
    """(20-session return minus SPY from the close of session d, or the next
    cached session of the name; which side was non-finite). The second item
    is "name" or "SPY" when a close the return reads is NaN or infinite (the
    first is then None), else None."""
    keys = [b["date"] for b in bars]
    i = bisect.bisect_left(keys, d.isoformat())
    if i + 20 >= len(bars):
        return None, None
    c0, c1 = bars[i]["close"], bars[i + 20]["close"]
    if nonfinite(c0) or nonfinite(c1):
        return None, "name"
    r, span = c1 / c0 - 1, (keys[i], keys[i + 20])
    sk = {b["date"]: b["close"] for b in spy}
    if span[0] not in sk or span[1] not in sk:
        return None, None
    s0, s1 = sk[span[0]], sk[span[1]]
    if nonfinite(s0) or nonfinite(s1):
        return None, "SPY"
    return r - (s1 / s0 - 1), None


def _x20(bars: list[dict], spy: list[dict], d: date) -> float | str | None:
    """20-session return minus SPY from the close of session d (or the next
    cached session of the name); NONFINITE when a close it reads is NaN or
    infinite."""
    v, bad = x20_detail(bars, spy, d)
    return NONFINITE if bad else v


def fill_rows(fills, bars_by_ticker, snapshots, approvals) -> list[dict]:
    """fills: dicts symbol, qty, price, ts, env, approval_id, sells_before_qty,
    buys_before_qty. snapshots: env -> sorted [(ts, total_value)].
    approvals: dicts id, ticker, ts, status, stop, bucket, stop_method.
    Returns one record per fill with the live shadow's two dicts."""
    import risk_manager
    from config import ATR_STOP_MULTIPLIER
    rows = []
    for f in fills:
        d = _ny_date(f["ts"])
        bars = _bars_before(bars_by_ticker.get(f["symbol"], []), d)
        snaps = snapshots.get(f["env"], [])
        i = bisect.bisect_right([s[0] for s in snaps], f["ts"])
        pv = snaps[i - 1][1] if i else None
        appr = None
        if f.get("approval_id") is not None:
            appr = next((a for a in approvals if a["id"] == f["approval_id"]), None)
        if appr is None:
            cands = [a for a in approvals if a["ticker"] == f["symbol"] and a["status"] == "APPROVED"
                     and a.get("env") == f["env"]
                     and f["ts"] - timedelta(days=2) <= a["ts"] <= f["ts"]]
            appr = max(cands, key=lambda a: a["ts"]) if cands else None
        bucket = appr["bucket"] if appr else None
        method = (appr or {}).get("stop_method") or ""
        mult = None
        if "ATR" in method and "allback" not in method and bucket in ATR_STOP_MULTIPLIER:
            lo, hi = ATR_STOP_MULTIPLIER[bucket]
            mult = (lo + hi) / 2 if lo else None
        held_qty = max((f.get("buys_before_qty") or 0) - (f.get("sells_before_qty") or 0), 0)
        saved = risk_manager._vol_shadow_bars
        try:
            risk_manager._vol_shadow_bars = lambda ticker, window, _b=bars: _b
            rec = risk_manager._vol_shadow_record(
                f["symbol"], f["price"], f["qty"], held_qty * f["price"],
                appr["stop"] if appr else None, _atr(bars), mult, pv)
        finally:
            risk_manager._vol_shadow_bars = saved
        rec["vol_target_shadow"]["weights_at"] = "fill"     # the fill quantity is the final order
        x20, bad = x20_detail(bars_by_ticker.get(f["symbol"], []), bars_by_ticker.get("SPY", []), d)
        rows.append({"date": d, "symbol": f["symbol"], "env": f["env"], "bucket": bucket,
                     "linked": appr is not None, "pv_known": pv is not None,
                     "x20": x20,
                     "x20_nonfinite": [] if bad is None else [f["symbol"] if bad == "name" else "SPY"],
                     **rec})
    return rows


def _pct(x, nd=1):
    return f"{x * 100:.{nd}f}" if isinstance(x, (int, float)) else "-"


def _num(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def _med(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def _nf_cell(rs) -> str:
    """'0', or 'N (TICKERS)' for rows whose x20 read a non-finite close."""
    bad = [r for r in rs if r.get("x20_nonfinite")]
    tick = sorted({t for r in bad for t in r["x20_nonfinite"]})
    return f"{len(bad)} ({', '.join(tick)})" if bad else "0"


def readout(rows: list[dict], env: str, head: str, now: datetime | None = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    from config import VOL_TARGET_CONTRIBUTION
    ok = [r for r in rows if r["vol_target_shadow"]["read"] == "ok"]
    lines = [f"# Volatility shadow readout ({env} fills)", "",
             f"Generated {now} at code {head}. Engineering diagnostic, not a study result. "
             f"Target contribution {VOL_TARGET_CONTRIBUTION:.3f} (weight x rvol20). "
             "Weights are % of the portfolio. Ratio = vol weight / position weight after the "
             "fill; under 1 means the vol budget would have cut the buy. Percentages and "
             "ratios only.", "",
             f"Fills: {len(rows)}; bars ok: {len(ok)}; stop linked: {sum(r['linked'] for r in rows)}; "
             f"portfolio snapshot found: {sum(r['pv_known'] for r in rows)}; x20 non-finite "
             f"(a NaN or infinite close of the name or SPY; out of the x20 mean, no bar "
             f"changed): {_nf_cell(rows)}.", ""]
    cut = [r for r in ok if r["vol_target_shadow"]["would_cut"] is True]
    keep = [r for r in ok if r["vol_target_shadow"]["would_cut"] is False]
    lines += ["## Summary", "",
              "| group | fills | median rvol20 % | median ratio | median stop sigma20 | median stop ATR | mean x20 % "
              "| x20 non-finite |",
              "|---|---|---|---|---|---|---|---|"]

    def summ(name, rs):
        xs = [r["x20"] for r in rs if r["x20"] is not None]
        lines.append(f"| {name} | {len(rs)} | {_pct(_med(r['vol_target_shadow']['rvol20'] for r in rs), 0)} | "
                     f"{_num(_med(r['vol_target_shadow']['weight_ratio'] for r in rs))} | "
                     f"{_num(_med(r['stop_vol']['stop_sigma20'] for r in rs))} | "
                     f"{_num(_med(r['stop_vol']['stop_distance_atr'] for r in rs))} | "
                     f"{_pct(statistics.mean(xs)) if xs else '-'} ({len(xs)}) | {_nf_cell(rs)} |")
    summ("all with bars", ok)
    summ("vol budget would cut", cut)
    summ("vol budget would not cut", keep)
    for b in sorted({r["bucket"] for r in ok if r["bucket"]}):
        summ(f"bucket {b}", [r for r in ok if r["bucket"] == b])
    lines += ["", "## Fills", "",
              "| date | name | env | bucket | rvol20 % | weight after % | vol weight % | ratio | would cut | "
              "stop % | stop ATR | ATR mult | stop sigma20 | x20 % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["date"], r["symbol"])):
        vt, sv = r["vol_target_shadow"], r["stop_vol"]
        wc = vt["would_cut"]
        lines.append(f"| {r['date']} | {r['symbol']} | {r['env']} | {r['bucket'] or '-'} | "
                     f"{_pct(vt['rvol20'], 0)} | {_pct(vt['position_weight_after'])} | "
                     f"{_pct(vt['vol_weight'])} | {_num(vt['weight_ratio'])} | "
                     f"{'-' if wc is None else ('yes' if wc else 'no')} | {_pct(sv['stop_pct'])} | "
                     f"{_num(sv['stop_distance_atr'])} | {_num(sv['atr_multiple'])} | "
                     f"{_num(sv['stop_sigma20'])} | "
                     + (f"non-finite ({', '.join(r['x20_nonfinite'])})" if r.get("x20_nonfinite")
                        else _pct(r['x20'])) + " |")
    lines += ["", "x20 is context, not a test: fills of one day move together, and the vol "
              "budget's target was not tuned on these outcomes."]
    return "\n".join(lines) + "\n"


def _fetch(env: str):
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            env_sql, env_args = ("", ()) if env == "all" else (" AND t.trading_env = %s", (env,))
            cur.execute("SELECT t.symbol, t.side, t.quantity, t.price, t.timestamp, t.trading_env, "
                        "o.approval_id FROM trades t LEFT JOIN orders o ON o.id = t.order_row_id "
                        "WHERE t.symbol <> 'TESTX'" + env_sql + " ORDER BY t.timestamp", env_args)
            trades = cur.fetchall()
            cur.execute("SELECT id, ticker, timestamp, status, stop_loss_price, thesis_bucket, "
                        "risk_json->>'stop_method', trading_env FROM pending_approvals")
            approvals = [{"id": a[0], "ticker": a[1], "ts": a[2], "status": a[3],
                          "stop": float(a[4]) if a[4] is not None else None, "bucket": a[5],
                          "stop_method": a[6], "env": a[7]} for a in cur.fetchall()]
            cur.execute("SELECT trading_env, timestamp, total_value FROM portfolio_snapshots "
                        "WHERE total_value > 0 ORDER BY timestamp")
            snapshots: dict[str, list] = {}
            for e, ts, v in cur.fetchall():
                snapshots.setdefault(e, []).append((ts, float(v)))
            tickers = sorted({t[0] for t in trades if t[1] == "BUY"} | {"SPY"})
            first = min((t[4] for t in trades), default=datetime.now(timezone.utc))
            cur.execute("SELECT ticker, date, high, low, close FROM price_history "
                        "WHERE ticker = ANY(%s) AND date >= %s ORDER BY ticker, date",
                        (tickers, (first - timedelta(days=60)).date()))
            bars: dict[str, list] = {}
            for t, d, h, l, c in cur.fetchall():
                bars.setdefault(t, []).append({"date": str(d), "high": float(h), "low": float(l),
                                               "close": float(c)})
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    fills, pos = [], {}
    for sym, side, qty, price, ts, e, appr_id in trades:
        key = (sym, e)
        b, s = pos.get(key, (0.0, 0.0))
        if side == "BUY":
            fills.append({"symbol": sym, "qty": float(qty), "price": float(price), "ts": ts, "env": e,
                          "approval_id": appr_id, "buys_before_qty": b, "sells_before_qty": s})
            pos[key] = (b + float(qty), s)
        elif side == "SELL":
            pos[key] = (b, s + float(qty))
    return fills, bars, snapshots, approvals


def _bar_qa():
    """scripts/_bar_qa.py (the non-finite bar flag) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bar_qa.py")
    spec = importlib.util.spec_from_file_location("_bar_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="live, paper or all (default: config TRADING_ENV)")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    from config import TRADING_ENV
    env = args.env or TRADING_ENV
    fills, bars, snapshots, approvals = _fetch(env)
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = readout(fill_rows(fills, bars, snapshots, approvals), env, head or "unknown")
    text += "\n" + _bar_qa().flag_line(bars) + "\n"   # flag only (scripts/_bar_qa.py)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
