import os
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise EnvironmentError("ALPACA_API_KEY or ALPACA_API_SECRET not found in .env")

SYMBOLS = ["META", "AAPL", "NVDA", "GOOGL", "ORCL"]

client = StockHistoricalDataClient(API_KEY, API_SECRET)


def get_latest_quotes(symbols: list[str]) -> dict[str, dict]:
    request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
    quotes = client.get_stock_latest_quote(request)
    result = {}
    for symbol, quote in quotes.items():
        result[symbol] = {
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
            "mid": round((float(quote.bid_price) + float(quote.ask_price)) / 2, 4),
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "timestamp": quote.timestamp,
        }
    return result


if __name__ == "__main__":
    print(f"Fetching quotes at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    quotes = get_latest_quotes(SYMBOLS)
    print(f"{'Symbol':<8} {'Bid':>10} {'Ask':>10} {'Mid':>10}")
    print("-" * 42)
    for symbol, q in quotes.items():
        print(f"{symbol:<8} ${q['bid']:>9.2f} ${q['ask']:>9.2f} ${q['mid']:>9.2f}")
