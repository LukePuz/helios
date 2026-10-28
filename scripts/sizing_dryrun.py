"""Sizing dry run: could Helios buy anything right now, and if not, which
step returns 0 shares? (limitations audit 2026-10-28 05:00, finding 5)

For every watchlist ticker it runs the live entry sizing chain from
agents/crew.py (size_position, then the concentration cap and its int()
scaling) as an OPEN at a given conviction, against the latest portfolio
snapshot in the database. Read-only: no IBKR connection (prices and ATR come
from yfinance), and db.set_state / alerts are stubbed so nothing is written
or sent.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/sizing_dryrun.py [--conviction 7] [--regime Neutral]

Prints one line per ticker and a count of outcomes. Shows percentages and
share counts only, no account values.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _snapshot_client(row):
    total, cash, positions = row
    positions = positions if isinstance(positions, dict) else json.loads(positions or "{}")
    acct = SimpleNamespace(portfolio_value=float(total), cash=float(cash), buying_power=float(cash))
    held = [SimpleNamespace(symbol=t, qty=float(p["qty"]), market_value=float(p["market_value"]),
                            avg_entry_price=float(p["avg_entry_price"]),
                            current_price=float(p["market_value"]) / float(p["qty"]) if float(p["qty"]) else 0.0,
                            unrealized_pl=float(p.get("unrealized_pl") or 0))
            for t, p in positions.items()]

    class _Client:
        def get_account(self):
            return acct

        def get_all_positions(self):
            return held

        def get_open_position(self, symbol):
            for p in held:
                if p.symbol == symbol:
                    return p
            raise Exception(f"No open position in {symbol}")
    return _Client()


def _yf_bars(tickers):
    import yfinance as yf
    data = yf.download(tickers, period="2mo", interval="1d", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t].dropna()
            out[t] = [{"h": float(r["High"]), "l": float(r["Low"]), "c": float(r["Close"])}
                      for _, r in df.iterrows()]
        except Exception:
            out[t] = []
    return out


def size_like_crew(rm, ticker, conviction, bucket, regime, price):
    """Mirror of the sizing steps in crew._route_to_approval for an OPEN."""
    sizing = rm.size_position(ticker, conviction, bucket, regime, price)
    if sizing.get("shares_initial", 0) <= 0:
        return 0, "size_position: " + "; ".join(sizing.get("flags", []))[:160]
    conc = rm.check_concentration(ticker, sizing["position_size_pct"], bucket=bucket)
    if not conc["passed"]:
        caps = [c for c in (conc.get("ai_infra_size_cap_pct"), conc.get("theme_size_cap_pct"),
                            conc.get("moonshot_size_cap_pct")) if c is not None]
        cap = min(caps) if caps else None
        if cap is not None and cap < sizing["position_size_pct"]:
            scale = cap / sizing["position_size_pct"] if sizing["position_size_pct"] > 0 else 0.0
            shares = int(sizing["shares_initial"] * scale)
            if shares <= 0:
                return 0, f"concentration cap {cap:.1%} x int() -> 0 ({'; '.join(conc['flags'])[:120]})"
            return shares, f"capped by concentration to {cap:.1%}"
    return sizing["shares_initial"], f"ok, target {sizing['position_size_pct']:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conviction", type=int, default=7)
    ap.add_argument("--regime", default=None, help="default: latest regime in the DB")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    import db
    import risk_manager
    from knowledge_loader import load_watchlist

    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT total_value, cash, positions_json FROM portfolio_snapshots "
                        "WHERE trading_env=%s ORDER BY id DESC LIMIT 1", (db.TRADING_ENV,))
            row = cur.fetchone()
    finally:
        db._put_conn(conn)
    if not row:
        sys.exit("no portfolio snapshot")
    regime = args.regime or (db.get_latest_regime() or {}).get("regime", "Neutral")
    watchlist = load_watchlist()
    tickers = sorted(watchlist)
    bars = _yf_bars(tickers)

    def _no_ib(*a, **k):
        raise RuntimeError("dry run: no IBKR")

    client = _snapshot_client(row)
    outcomes = Counter()
    with patch("db.set_state"), patch("alerts._send"), \
         patch("ibkr.get_client", _no_ib), patch("ibkr.get_historical_bars", _no_ib), \
         patch.object(risk_manager, "_fetch_bars_for_atr", lambda t, days=20: bars.get(t, [])[-(days + 5):]):
        rm = risk_manager.RiskManager(client)
        print(f"regime={regime} conviction={args.conviction} positions={len(client.get_all_positions())}")
        for t in tickers:
            if not bars.get(t):
                print(f"{t:6} no price data"); outcomes["no price"] += 1
                continue
            price = bars[t][-1]["c"]
            bucket = watchlist[t].get("role", "Accelerator")
            try:
                shares, why = size_like_crew(rm, t, args.conviction, bucket, regime, price)
            except Exception as e:
                shares, why = 0, f"error: {e}"
            # Flags quote allocations in dollars, which give away the account
            # size; keep only percentages and share counts.
            why = re.sub(r"\$[\d,]+(\.\d+)?", "$…", why)
            key = "buyable" if shares > 0 else why.split(":")[0]
            outcomes[key] += 1
            print(f"{t:6} {bucket:11} ${price:>8.2f}  shares={shares:<3} {why}")
    print("\nOutcomes:", dict(outcomes))


if __name__ == "__main__":
    main()
