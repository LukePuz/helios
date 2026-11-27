"""Stop cushion readout: for each held name, how far the last close sits
above its stop, and how often over the last year the closes fell that far
within 1, 5 and 10 sessions (audit 2026-11-26 22:53 #3; batch-1135 row 21
records the distance, this reads the history). Read-only engineering
readout; counts and percentages only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/stop_cushion_readout.py [--env live] [--out FILE] [--export FILE.json]

Names: every position_stops row in --env (a held name has a stop row). Bars:
the newest BARS_READ price_history rows per name (position_exit_frame's
read), oldest first.

Per name, with row 21's definitions (tasks._near_stop_record):
  - price: the newest bar with a finite close (_frame_last_close);
  - stop_drop_pct = (price - stop) / price x 100, the drop from the price to
    the stop in percent of the price;
  - ATR: the mean true range over the newest ATR_ROWS rows
    (risk_manager._compute_atr; get_atr's price_history window), None with
    atr_read "nonfinite" when any high, low or close in that window is not
    finite; shown as a percent of the price (atr_pct) and as the cushion in
    ATRs, (price - stop) / ATR. Not the ATR the stop was set from. The same
    two figures over the newest ATR14_ROWS rows (14 true ranges, the audit's
    "ATR14") are shown beside them; neither window is chosen here.
  - windows: each of the newest LOOKBACK rows that has k rows after it
    starts a window (k = 1, 5, 10); a window counts when the lowest close of
    the next k rows is at or below the start close x (1 - stop_drop_pct /
    100). n windows, the count that fell that far, and their percent. A
    window with a non-finite close in it has no defined lowest close: it is
    left out of n and counted as skipped. Today's distance is applied to
    every past window. Not computed when the price is at or below the stop.

Closes only: intraday lows reach a stop more often, so the percent is a
lower bound. No threshold, label or recommendation is given; a pre-stop
review is Luke's. The non-finite bar line is scripts/_bar_qa.py's.

No price, stop or level is printed or exported: percents, ATR multiples,
counts and dates only. SUPABASE_PORT must be 6543 (the transaction pooler);
not run 04:50-05:12 ET; one read-only transaction with a statement timeout,
rolled back. db is imported from the working directory when it holds a
db.py (the live clone, with its .env), else from beside this script; with no
SUPABASE_HOST after the import the readout stops before connecting.
Imports: db, scripts/_bar_qa.py and the standard library only. Not loaded
by weekly_readouts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 30000
NY = ZoneInfo("America/New_York")
QUIET = (time(4, 50), time(5, 12))     # no DB reads in this New York window
BARS_READ = 400                        # position_exit_frame's get_price_history(days=400)
ATR_ROWS = 19                          # tasks.NEAR_STOP_ATR_ROWS (get_atr's 14 + 5)
ATR14_ROWS = 15                        # a 14-true-range ATR (the audit's "ATR14"), shown beside it
LOOKBACK = 252                         # about one year of sessions
HORIZONS = (1, 5, 10)
CAVEAT = "Closes only: intraday lows hit stops more often; this is a lower bound."
NO_RULE = ("No threshold, label or recommendation is given; a pre-stop review is Luke's.")

_STOPS_SQL = """
SELECT ticker, stop_price FROM position_stops WHERE trading_env = %(env)s ORDER BY ticker
"""
_BARS_SQL = """
WITH r AS (
    SELECT ticker, date, high, low, close,
           row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
    FROM price_history WHERE ticker = ANY(%(tickers)s))
SELECT ticker, date, high, low, close FROM r WHERE rn <= %(rows)s ORDER BY ticker, date
"""


def db_root(cwd: str | None = None) -> str:
    """The directory db is imported from: cwd when it holds a db.py, else
    this script's checkout."""
    cwd = os.getcwd() if cwd is None else cwd
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def _bar_qa():
    """scripts/_bar_qa.py (the non-finite bar flag) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bar_qa.py")
    spec = importlib.util.spec_from_file_location("_bar_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET[0] <= t < QUIET[1]


def _num(v) -> float | None:
    """float, or None for a missing value; NaN and inf stay as they are."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


def last_close(bars: list[dict]) -> tuple[float | None, str | None]:
    """(close, date) of the newest bar with a finite close."""
    for b in reversed(bars):
        c = _num(b.get("close"))
        if _finite(c):
            return c, str(b.get("date"))[:10]
    return None, None


def atr(bars: list[dict], rows_n: int = ATR_ROWS) -> tuple[float | None, str]:
    """(ATR, atr_read) over the newest rows_n bars: risk_manager._compute_atr's
    mean true range. atr_read "ok", "too_few_bars", "nonfinite" or "zero"."""
    rows = bars[-rows_n:]
    hlc = [(_num(b.get("high")), _num(b.get("low")), _num(b.get("close"))) for b in rows]
    if len(hlc) < 2:
        return None, "too_few_bars"
    if not all(_finite(v) for t in hlc for v in t):
        return None, "nonfinite"
    trs = [max(h - l, abs(h - pc), abs(l - pc)) for (h, l, _), (_, _, pc) in zip(hlc[1:], hlc)]
    a = sum(trs) / len(trs)
    if not math.isfinite(a):
        return None, "nonfinite"
    if a <= 0:
        return None, "zero"
    return a, "ok"


def windows(closes: list[float | None], drop_pct: float, k: int) -> dict:
    """Over the newest LOOKBACK closes: n windows of k sessions with a
    defined lowest close, how many fell at least drop_pct below their start
    close, and how many were skipped for a non-finite close. Pure."""
    c = closes[-LOOKBACK:]
    n = hits = skipped = 0
    for t in range(len(c) - k):
        seg = c[t:t + k + 1]
        if not all(_finite(x) for x in seg):
            skipped += 1
            continue
        n += 1
        if min(seg[1:]) <= seg[0] * (1 - drop_pct / 100):
            hits += 1
    return {"k": k, "n": n, "fell": hits,
            "pct": round(hits / n * 100, 1) if n else None, "skipped": skipped}


def name_record(ticker: str, stop_price, bars: list[dict]) -> dict:
    """The percent-only record of one held name. Pure; no price is kept."""
    p, pdate = last_close(bars)
    s = _num(stop_price)
    a, read = atr(bars)
    a14, read14 = atr(bars, ATR14_ROWS)
    diff = p - s if _finite(p) and p > 0 and _finite(s) and s > 0 else None
    drop = round(diff / p * 100, 2) if diff is not None else None
    rec = {"ticker": ticker, "bars": len(bars), "last_close_date": pdate,
           "stop_drop_pct": drop,
           "atr_pct": round(a / p * 100, 2) if a and _finite(p) and p > 0 else None,
           "stop_dist_atr": round(diff / a, 2) if diff is not None and a else None,
           "atr_read": read,
           "atr14_pct": round(a14 / p * 100, 2) if a14 and _finite(p) and p > 0 else None,
           "stop_dist_atr14": round(diff / a14, 2) if diff is not None and a14 else None,
           "atr14_read": read14, "windows": None,
           "note": None}
    if drop is None:
        rec["note"] = "no price or stop"
    elif drop <= 0:
        rec["note"] = "price at or below the stop"
    else:
        closes = [_num(b.get("close")) for b in bars]
        rec["windows"] = [windows(closes, drop, k) for k in HORIZONS]
    return rec


def build(stops: list[tuple], bars: dict) -> list[dict]:
    return [name_record(t, sp, bars.get(t, [])) for t, sp in stops]


def _c(v) -> str:
    return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def render(recs: list[dict], env: str, head: str, now: datetime, flag: str) -> str:
    lines = ["# Stop cushion readout", "",
             f"Env {env}; held names (position_stops rows) at "
             f"{now.astimezone(NY):%Y-%m-%d %H:%M} ET; code {head}. Counts and percentages only; "
             "no price, stop or level is printed.", "",
             CAVEAT, NO_RULE, "",
             f"stop_drop_pct = (last close - stop) / last close x 100 (batch-1135 row 21's "
             f"definition). ATR: mean true range over the newest {ATR_ROWS} price_history rows "
             f"(get_atr's window, as row 21), and over the newest {ATR14_ROWS} (ATR14); neither is "
             "the ATR the stop was set from, and no window is chosen here. Windows: each of the newest "
             f"{LOOKBACK} rows with k rows after it; counted when the lowest close of the next k "
             "rows is at least stop_drop_pct below the start close. Today's distance is applied to "
             "every past window; windows with a non-finite close are skipped and counted.", "",
             flag, "",
             "| name | last close date | bars | stop_drop_pct | ATR % of price | cushion (ATRs) "
             "| ATR14 % of price | cushion (ATR14s) | ATR read | "
             + " | ".join(f"{k}-session: fell / n (%)" for k in HORIZONS)
             + " | skipped (non-finite) | note |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|" + "---:|" * len(HORIZONS) + "---:|---|"]
    for r in recs:
        if r["windows"]:
            cells = [f"{w['fell']} / {w['n']} ({w['pct']:.1f}%)" if w["pct"] is not None
                     else f"{w['fell']} / {w['n']}" for w in r["windows"]]
            skipped = "/".join(str(w["skipped"]) for w in r["windows"])
        else:
            cells, skipped = ["-"] * len(HORIZONS), "-"
        lines.append(f"| {r['ticker']} | {r['last_close_date'] or '-'} | {r['bars']} | "
                     f"{_c(r['stop_drop_pct'])} | {_c(r['atr_pct'])} | {_c(r['stop_dist_atr'])} | "
                     f"{_c(r['atr14_pct'])} | {_c(r['stop_dist_atr14'])} | "
                     f"{r['atr_read'] if r['atr_read'] == r['atr14_read'] else r['atr_read'] + ' / ' + r['atr14_read']} | " + " | ".join(cells) + f" | {skipped} | {r['note'] or ''} |")
    if not recs:
        lines.append("| (no position_stops rows) |" + " |" * (10 + len(HORIZONS)))
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--export", help="write the per-name records (JSON, percent only) here")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    if in_quiet_window(now):
        print("04:50-05:12 ET: no DB reads in this window; not reading.", file=sys.stderr)
        return 2

    sys.path.insert(0, db_root())
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
            cur.execute(_STOPS_SQL, {"env": args.env})
            stops = [(t, sp) for t, sp in cur.fetchall()]
            bars: dict = {}
            if stops:
                cur.execute(_BARS_SQL, {"tickers": sorted({t for t, _ in stops}), "rows": BARS_READ})
                for t, d, h, l, c in cur.fetchall():
                    bars.setdefault(t, []).append({"date": d, "high": h, "low": l, "close": c})
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    recs = build(stops, bars)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(recs, args.env, head or "unknown", now, _bar_qa().flag_line(bars))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    if args.export:
        with open(args.export, "w") as f:
            json.dump({"env": args.env, "caveat": CAVEAT, "atr_rows": ATR_ROWS,
                       "lookback": LOOKBACK, "horizons": list(HORIZONS),
                       "nonfinite": _bar_qa().flag_record(bars), "names": recs},
                      f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
