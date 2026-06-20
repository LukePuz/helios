"""
MANGOS paper trading agent — momentum strategy.

Entry:  5-day momentum > +2%  AND  news sentiment >= -0.3
Exit:   5-day momentum < -2%  OR   position up >= 8% from entry
        (blocked if news sentiment > +0.3 on a SELL signal)

Risk rules enforced by RiskManager:
  - Max 2% of portfolio per trade
  - Max 20% of portfolio per position
  - Full halt if portfolio drops 10% from start
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import alerts
import db
import news_agent
from market_data import get_latest_quotes
from risk_manager import RiskManager

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise EnvironmentError("ALPACA_API_KEY or ALPACA_API_SECRET not found in .env")

SYMBOLS = ["META", "AAPL", "NVDA", "GOOGL", "ORCL"]
POLL_INTERVAL_SECONDS = 300

MOMENTUM_BUY_THRESHOLD = 0.02
MOMENTUM_SELL_THRESHOLD = -0.02
TAKE_PROFIT_PCT = 0.08
SENTIMENT_BLOCK_BUY = -0.3   # block BUY if sentiment is below this
SENTIMENT_BLOCK_SELL = 0.3   # block SELL if sentiment is above this

ET = ZoneInfo("America/New_York")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
risk_manager = RiskManager(trading_client)

_daily_summary_sent: bool = False


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def _get_momentum(symbols: list[str]) -> dict[str, float]:
    """Returns 5-trading-day momentum per symbol. 0.0 if insufficient data."""
    start = datetime.now(timezone.utc) - timedelta(days=10)
    try:
        resp = data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start,
            )
        )
        bars_data: dict = resp.data if hasattr(resp, "data") else {}
    except Exception as e:
        logging.error(f"Momentum fetch failed: {e}")
        return {s: 0.0 for s in symbols}

    result: dict[str, float] = {}
    for symbol in symbols:
        bars = bars_data.get(symbol, [])
        if len(bars) >= 5:
            old = float(bars[-5].close)
            new = float(bars[-1].close)
            result[symbol] = (new - old) / old
        else:
            logging.warning(f"{symbol}: only {len(bars)} bars — momentum defaulting to 0.0")
            result[symbol] = 0.0
    return result


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def _execute_order(symbol: str, side: OrderSide, qty: int, price: float, reason: str) -> None:
    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )
    except Exception as e:
        logging.error(f"Order submission failed: {symbol} {side} ×{qty} — {e}")
        return

    side_str = "BUY" if side == OrderSide.BUY else "SELL"
    logging.info(f"ORDER  {side_str} {qty} {symbol} @ ~${price:.2f}  |  {reason}")
    db.log_trade(symbol=symbol, side=side_str, quantity=qty, price=price, reason=reason)
    alerts.send_trade_alert(symbol, side_str, qty, price, reason)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _take_snapshot() -> None:
    try:
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        db.log_snapshot(
            total_value=float(account.portfolio_value),
            cash=float(account.cash),
            positions={
                p.symbol: {
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry_price": float(p.avg_entry_price),
                    "unrealized_pl": float(p.unrealized_pl),
                }
                for p in positions
            },
        )
    except Exception as e:
        logging.error(f"Snapshot failed: {e}")


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------

def on_tick(quotes: dict[str, dict]) -> None:
    global _daily_summary_sent

    now_et = datetime.now(ET)
    logging.info(f"Tick at {now_et.strftime('%H:%M:%S ET')}")

    # Daily summary at 4:30 PM ET after market close
    if now_et.hour == 16 and now_et.minute >= 30 and not _daily_summary_sent:
        try:
            account = trading_client.get_account()
            pv = float(account.portfolio_value)
            alerts.send_daily_summary(pv, pv - risk_manager.start_value)
            _daily_summary_sent = True
        except Exception as e:
            logging.error(f"Daily summary failed: {e}")
    elif now_et.hour < 16:
        _daily_summary_sent = False

    # Hard halt — no trades, still snapshot so we can see the drawdown in the dashboard
    if risk_manager.is_halted():
        logging.warning("HALT: drawdown limit reached — no trades this tick.")
        alerts.send_halt_alert(risk_manager.portfolio_value(), risk_manager.start_value)
        _take_snapshot()
        return

    # Data fetch
    sentiments = news_agent.get_sentiment(SYMBOLS)
    momentum = _get_momentum(SYMBOLS)

    try:
        open_positions = {p.symbol: p for p in trading_client.get_all_positions()}
    except Exception as e:
        logging.error(f"Failed to fetch positions: {e}")
        open_positions = {}

    for symbol in SYMBOLS:
        price: float = quotes[symbol]["mid"]
        mom: float = momentum.get(symbol, 0.0)
        sentiment: float = sentiments.get(symbol, 0.0)
        in_position: bool = symbol in open_positions

        # Determine raw signal from momentum
        if mom > MOMENTUM_BUY_THRESHOLD:
            raw_signal = "BUY"
        elif mom < MOMENTUM_SELL_THRESHOLD:
            raw_signal = "SELL"
        else:
            raw_signal = "HOLD"

        # ---- SELL logic (exit existing positions) ----
        if in_position:
            pos = open_positions[symbol]
            entry = float(pos.avg_entry_price)
            pnl_pct = (price - entry) / entry

            sell_reason: str | None = None
            if mom < MOMENTUM_SELL_THRESHOLD:
                sell_reason = f"Negative momentum {mom:.1%}"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                sell_reason = f"Take profit +{pnl_pct:.1%} from entry ${entry:.2f}"

            if sell_reason:
                if sentiment > SENTIMENT_BLOCK_SELL:
                    final_signal = "SELL_BLOCKED"
                    logging.info(f"{symbol}: SELL blocked — sentiment {sentiment:+.3f} > {SENTIMENT_BLOCK_SELL}")
                else:
                    qty = int(float(pos.qty))
                    if risk_manager.check_sell_allowed(symbol, qty):
                        _execute_order(symbol, OrderSide.SELL, qty, price, sell_reason)
                        final_signal = "SELL"
                    else:
                        final_signal = "SELL_REJECTED"
            else:
                final_signal = raw_signal  # HOLD or BUY (already in position, no action)

        # ---- BUY logic (enter new position) ----
        else:
            if raw_signal == "BUY":
                if sentiment < SENTIMENT_BLOCK_BUY:
                    final_signal = "BUY_BLOCKED"
                    logging.info(f"{symbol}: BUY blocked — sentiment {sentiment:+.3f} < {SENTIMENT_BLOCK_BUY}")
                else:
                    qty = risk_manager.shares_affordable(symbol, price)
                    if qty > 0:
                        _execute_order(
                            symbol, OrderSide.BUY, qty, price,
                            f"Momentum {mom:.1%}, sentiment {sentiment:+.3f}",
                        )
                        final_signal = "BUY"
                    else:
                        final_signal = "BUY_CAPPED"
            else:
                final_signal = raw_signal  # HOLD or SELL (no position to sell)

        db.log_signal(
            symbol=symbol,
            signal=final_signal,
            confidence=abs(mom),
            reasoning=f"momentum={mom:.2%}, sentiment={sentiment:+.3f}, price=${price:.2f}",
        )

    _take_snapshot()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    return trading_client.get_clock().is_open


def run() -> None:
    logging.info(f"Agent starting | symbols={SYMBOLS} | start_value=${risk_manager.start_value:,.2f}")

    while True:
        if not is_market_open():
            logging.debug("Market closed — sleeping 60s.")
            time.sleep(60)
            continue

        try:
            quotes = get_latest_quotes(SYMBOLS)
            on_tick(quotes)
        except Exception as e:
            logging.error(f"Tick error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    run()
