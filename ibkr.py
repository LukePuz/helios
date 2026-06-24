"""
Interactive Brokers connection module via ib_insync.
Replaces Alpaca for order execution, account data, and market data.

Requirements:
  pip install ib_insync nest_asyncio

TWS must be running with API enabled:
  - Paper trading port: 7497  (IB_PORT=7497)
  - Live trading port:  7496  (IB_PORT=7496)

Set in .env:
  IB_HOST=127.0.0.1
  IB_PORT=7497
  IB_CLIENT_ID=1
"""
import logging
import os
from pathlib import Path
from typing import Any

import nest_asyncio
nest_asyncio.apply()

import ib_insync as ibi
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

IB_HOST      = os.getenv("IB_HOST",      "127.0.0.1")
IB_PORT      = int(os.getenv("IB_PORT",      "7497"))   # 7497=paper, 7496=live
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

_ib: ibi.IB | None = None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_ib() -> ibi.IB:
    global _ib
    if _ib is None or not _ib.isConnected():
        _ib = ibi.IB()
        _ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=False)
        log.info(f"Connected to IBKR TWS at {IB_HOST}:{IB_PORT} (clientId={IB_CLIENT_ID})")
    return _ib


def disconnect() -> None:
    global _ib
    if _ib and _ib.isConnected():
        _ib.disconnect()
        _ib = None
        log.info("Disconnected from IBKR TWS.")


def _stock(ticker: str) -> ibi.Stock:
    return ibi.Stock(ticker, "SMART", "USD")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_latest_price(ticker: str) -> float | None:
    """
    Returns the latest price via historical minute bars.
    Uses zero market data lines — no streaming subscription opened.
    """
    try:
        # Try last minute bar first (most recent trade during/after market hours)
        bars = get_historical_bars(ticker, duration="1 D", bar_size="1 min")
        if bars:
            return float(bars[-1]["close"])
        # Fall back to daily bar (last close)
        bars = get_historical_bars(ticker, duration="2 D", bar_size="1 day")
        if bars:
            return float(bars[-1]["close"])
    except Exception as e:
        log.warning(f"get_latest_price({ticker}) failed: {e}")
    return None


def get_historical_bars(
    ticker: str,
    duration: str = "20 D",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> list[dict]:
    """
    Returns OHLCV bars from IBKR historical data.
    duration examples: '10 D', '1 M', '3 M'
    bar_size examples: '1 day', '1 hour', '5 mins'
    """
    try:
        ib = get_ib()
        contract = _stock(ticker)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )
        return [
            {
                "time":   str(b.date),
                "open":   float(b.open),
                "high":   float(b.high),
                "low":    float(b.low),
                "close":  float(b.close),
                "volume": int(b.volume),
            }
            for b in bars
        ]
    except Exception as e:
        log.warning(f"get_historical_bars({ticker}) failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def place_market_order(ticker: str, side: str, qty: float) -> dict:
    """
    Place a market order. side = 'BUY' or 'SELL'.
    Returns dict with order_id, status.
    """
    ib = get_ib()
    contract = _stock(ticker)

    # Subscribe to delayed data briefly to satisfy TWS precautionary check
    ib.reqMarketDataType(3)
    td = ib.reqMktData(contract, "", False, False)
    ib.sleep(1)

    order = ibi.MarketOrder(side.upper(), qty)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)

    ib.cancelMktData(contract)

    log.info(f"Order placed: {side.upper()} {qty} {ticker} (orderId={trade.order.orderId})")
    return {
        "order_id": trade.order.orderId,
        "ticker":   ticker,
        "side":     side.upper(),
        "qty":      qty,
        "status":   trade.orderStatus.status,
    }


def place_stop_order(ticker: str, qty: float, stop_price: float) -> dict:
    """Place a sell stop-loss order."""
    ib = get_ib()
    contract = _stock(ticker)
    order    = ibi.StopOrder("SELL", qty, stop_price, tif="GTC")
    trade    = ib.placeOrder(contract, order)
    ib.sleep(1)
    log.info(f"Stop order placed: SELL {qty} {ticker} @ ${stop_price:.2f}")
    return {"order_id": trade.order.orderId, "stop_price": stop_price}


# ---------------------------------------------------------------------------
# Alpaca-compatible adapter (used by RiskManager and tasks)
# ---------------------------------------------------------------------------

class _IBKRAccount:
    """Mirrors Alpaca's TradeAccount interface."""
    def __init__(self, portfolio_value: float, buying_power: float, cash: float):
        self.portfolio_value = portfolio_value
        self.buying_power    = buying_power
        self.cash            = cash


class _IBKRPosition:
    """Mirrors Alpaca's Position interface."""
    def __init__(
        self,
        symbol: str,
        qty: float,
        market_value: float,
        avg_entry_price: float,
        current_price: float,
        unrealized_pl: float,
    ):
        self.symbol          = symbol
        self.qty             = qty
        self.market_value    = market_value
        self.avg_entry_price = avg_entry_price
        self.current_price   = current_price
        self.unrealized_pl   = unrealized_pl


class IBKRClient:
    """
    Drop-in adapter for Alpaca's TradingClient.
    Exposes get_account(), get_all_positions(), get_open_position()
    so RiskManager and tasks work without further changes.
    """

    def get_account(self) -> _IBKRAccount:
        ib   = get_ib()
        vals = {}
        for v in ib.accountValues():
            if v.currency == "USD":
                vals[v.tag] = v.value
        return _IBKRAccount(
            portfolio_value = float(vals.get("NetLiquidation", 0)),
            buying_power    = float(vals.get("BuyingPower",    0)),
            cash            = float(vals.get("CashBalance",    0)),
        )

    def get_all_positions(self) -> list[_IBKRPosition]:
        ib = get_ib()
        return [
            _IBKRPosition(
                symbol          = item.contract.symbol,
                qty             = float(item.position),
                market_value    = float(item.marketValue),
                avg_entry_price = float(item.averageCost),
                current_price   = float(item.marketPrice),
                unrealized_pl   = float(item.unrealizedPNL),
            )
            for item in ib.portfolio()
            if item.contract.secType == "STK" and float(item.position) != 0
        ]

    def get_open_position(self, symbol: str) -> _IBKRPosition:
        """Raises Exception if no position exists — matches Alpaca behaviour."""
        for pos in self.get_all_positions():
            if pos.symbol == symbol:
                return pos
        raise Exception(f"No open position in {symbol}")


# Module-level singleton — reuse across tasks
_client: IBKRClient | None = None


def get_client() -> IBKRClient:
    global _client
    if _client is None:
        _client = IBKRClient()
    return _client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_client()
    acct   = client.get_account()
    print(f"Portfolio value: ${acct.portfolio_value:,.2f}")
    print(f"Cash:            ${acct.cash:,.2f}")
    print(f"Buying power:    ${acct.buying_power:,.2f}")
    print()
    positions = client.get_all_positions()
    if positions:
        print(f"{'Symbol':<8} {'Qty':>6} {'Mkt Value':>12} {'Unreal P&L':>12}")
        for p in positions:
            print(f"{p.symbol:<8} {p.qty:>6.0f} ${p.market_value:>11,.2f} ${p.unrealized_pl:>11,.2f}")
    else:
        print("No open positions.")
