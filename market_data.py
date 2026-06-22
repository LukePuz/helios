"""
Market data access via Interactive Brokers (ib_insync).
TWS must be running on port 7497 (paper) or 7496 (live).
"""
import logging
from datetime import datetime, timezone

from ibkr import get_latest_price, get_historical_bars
from config import SYMBOLS

log = logging.getLogger(__name__)


def get_latest_quotes(symbols: list[str]) -> dict[str, dict]:
    """Returns latest price for each symbol."""
    result = {}
    for ticker in symbols:
        price = get_latest_price(ticker)
        if price:
            result[ticker] = {
                "bid":       price,
                "ask":       price,
                "mid":       price,
                "timestamp": datetime.now(timezone.utc),
            }
    return result


def get_bars(
    ticker: str,
    days: int = 20,
    bar_size: str = "1 day",
) -> list[dict]:
    """Returns OHLCV bars. Used by scanner and risk manager."""
    duration = f"{days} D"
    return get_historical_bars(ticker, duration=duration, bar_size=bar_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Fetching quotes via IBKR at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    quotes = get_latest_quotes(SYMBOLS[:5])
    print(f"{'Symbol':<8} {'Price':>10}")
    print("-" * 20)
    for symbol, q in quotes.items():
        print(f"{symbol:<8} ${q['mid']:>9.2f}")
