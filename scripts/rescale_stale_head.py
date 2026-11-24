"""Stale cached rows on an old split basis: find them, and rescale them on request.

The nightly refresh (ingestion/price_history.py) fetches one split-adjusted
year and upserts it, so after a split the rows older than that year keep the
old scale: a one-day step in the stored closes about one year back (SNDK
2025-11-21, x0.1277 on its 8-for-1 split; MNST and APH 2025-09-12, about x0.5).

Dry run (default, read-only: SET TRANSACTION READ ONLY, then rollback): for
each ticker (all cached tickers, or --ticker), every one-day close ratio near
a split factor (within 3% of 2, 3, 4, 5, 8, 10, 3/2 or their inverses) or at
or beyond 1/3 or 3, with the step date (first day on the new basis), the
ratio, the 20-bar median volume after over before (from the step day), the
largest of the first 3 volumes over the after-median, and the stored rows
dated before the step (the rows a rescale would change).

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python scripts/rescale_stale_head.py [--ticker SNDK ...]

Apply (writes price_history; one transaction):

    ... scripts/rescale_stale_head.py --apply --ticker SNDK --factor 1/8 [--before 2025-11-21]

multiplies open/high/low/close of the ticker's rows dated before the step by
the factor and divides their volume by it. --before defaults to the ticker's
only candidate step; with several it is required. Refused when the stored
step is no longer within 10% of the factor (already applied, or the wrong
factor). Prints ratios and counts only.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FACTORS = (2, 3, 4, 5, 8, 10, 1.5)
FACTOR_TOL = 0.03
BREAK = 3.0
VOL_BARS = 20
APPLY_TOL = 0.10


def parse_factor(s: str) -> float:
    f = float(Fraction(s.strip()))
    if not f > 0:
        raise ValueError(f"factor must be positive: {s}")
    return f


def _median_pos(xs):
    xs = [float(x) for x in xs if x is not None and float(x) > 0]
    return statistics.median(xs) if len(xs) >= 5 else None


def candidates(rows: list[dict]) -> list[dict]:
    """Candidate steps in one ticker's stored rows (oldest first)."""
    out = []
    for i in range(1, len(rows)):
        a, b = float(rows[i - 1]["close"]), float(rows[i]["close"])
        if a <= 0 or b <= 0:
            continue
        r = b / a
        label = next((lab for f in FACTORS for t, lab in ((1 / f, f"1/{f:g}"), (f, f"{f:g}"))
                      if abs(r / t - 1) <= FACTOR_TOL), None)
        if label is None and not (r <= 1 / BREAK or r >= BREAK):
            continue
        before = _median_pos([x.get("volume") for x in rows[max(0, i - VOL_BARS):i]])
        after = _median_pos([x.get("volume") for x in rows[i:i + VOL_BARS]])
        first3 = [float(x.get("volume") or 0) for x in rows[i:i + 3]]
        out.append({"step_date": str(rows[i]["date"])[:10], "ratio": round(r, 4),
                    "nearest": label or "scale_break",
                    "volume_ratio": round(after / before, 2) if before and after else None,
                    "spike_ratio": round(max(first3) / after, 2) if after and first3 else None,
                    "rows_before": i})
    return out


def _rows(cur, ticker):
    cur.execute("SELECT date, close, volume FROM price_history WHERE ticker=%s ORDER BY date", (ticker,))
    return [{"date": d, "close": c, "volume": v} for d, c, v in cur.fetchall()]


def dry_run(conn, tickers: list[str] | None) -> list[tuple[str, dict]]:
    found = []
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        if not tickers:
            cur.execute("SELECT DISTINCT ticker FROM price_history ORDER BY ticker")
            tickers = [r[0] for r in cur.fetchall()]
        for t in tickers:
            for c in candidates(_rows(cur, t)):
                found.append((t, c))
    conn.rollback()
    return found


def apply(conn, ticker: str, factor: float, before: str | None) -> dict:
    """Rescale ticker's rows dated before the step by factor, in one
    transaction. Raises ValueError (nothing written) when the step can't be
    identified or no longer matches the factor."""
    with conn.cursor() as cur:
        rows = _rows(cur, ticker)
        cands = candidates(rows)
        if before is None:
            if len(cands) != 1:
                conn.rollback()
                raise ValueError(f"{ticker}: {len(cands)} candidate steps; pass --before")
            before = cands[0]["step_date"]
        step = next((c for c in cands if c["step_date"] == before), None)
        if step is None or abs(step["ratio"] / factor - 1) > APPLY_TOL:
            conn.rollback()
            raise ValueError(f"{ticker}: no stored step at {before} within {APPLY_TOL:.0%} of "
                             f"factor {factor:g} (already applied, or the wrong factor)")
        cur.execute("UPDATE price_history SET open=open*%s, high=high*%s, low=low*%s, close=close*%s, "
                    "volume=ROUND(volume/%s) WHERE ticker=%s AND date < %s",
                    (factor, factor, factor, factor, factor, ticker, before))
        n = cur.rowcount
    conn.commit()
    return {"ticker": ticker, "before": before, "factor": factor, "rows": n,
            "step_ratio_was": step["ratio"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", action="append", help="limit to these tickers (repeatable)")
    ap.add_argument("--apply", action="store_true", help="write: rescale one ticker's stale rows")
    ap.add_argument("--factor", help="multiplier for the old rows, e.g. 1/8 for an 8-for-1 split")
    ap.add_argument("--before", help="step date (first day on the new basis); default: the only candidate")
    args = ap.parse_args(argv)
    import db
    conn = db._get_conn()
    try:
        if args.apply:
            if not args.ticker or len(args.ticker) != 1 or not args.factor:
                print("--apply needs exactly one --ticker and --factor")
                return 2
            try:
                res = apply(conn, args.ticker[0].upper(), parse_factor(args.factor), args.before)
            except ValueError as e:
                print(f"not applied: {e}")
                return 1
            print(f"applied: {res['ticker']} rows before {res['before']} x {res['factor']:g} "
                  f"({res['rows']} rows; stored step was x{res['step_ratio_was']})")
            return 0
        found = dry_run(conn, [t.upper() for t in args.ticker] if args.ticker else None)
        print("DRY RUN (read-only). ticker | step date | ratio | nearest | volume after/before "
              "| first-3 max / after-median | rows that would change")
        for t, c in found:
            print(f"{t} | {c['step_date']} | x{c['ratio']} | {c['nearest']} | {c['volume_ratio']} | "
                  f"{c['spike_ratio']} | {c['rows_before']}")
        print(f"{len(found)} candidate step(s).")
        return 0
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)


if __name__ == "__main__":
    sys.exit(main())
