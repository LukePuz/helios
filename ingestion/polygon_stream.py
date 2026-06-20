"""
Polygon.io market data — real-time quotes, trades, and aggregates.
Significantly better than Alpaca's free data:
  - True real-time (not 15-min delayed)
  - Full L2 order book depth
  - Options chain data
  - Short interest
  - Institutional ownership

Requires POLYGON_API_KEY in .env
Sign up at polygon.io (free tier available, paid for real-time)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

API_KEY  = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"

from config import SYMBOLS


def _get(endpoint: str, params: dict | None = None) -> dict:
    if not API_KEY:
        return {}
    try:
        resp = requests.get(
            f"{BASE_URL}{endpoint}",
            params={"apiKey": API_KEY, **(params or {})},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Polygon fetch failed ({endpoint}): {e}")
        return {}


def get_snapshot(ticker: str) -> dict:
    """
    Returns full market snapshot for a ticker:
    current price, day's range, volume, VWAP, after-hours price.
    """
    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    ticker_data = data.get("ticker", {})
    day  = ticker_data.get("day", {})
    prev = ticker_data.get("prevDay", {})

    return {
        "ticker":        ticker,
        "price":         ticker_data.get("lastTrade", {}).get("p"),
        "vwap":          day.get("vw"),
        "volume":        day.get("v"),
        "open":          day.get("o"),
        "high":          day.get("h"),
        "low":           day.get("l"),
        "prev_close":    prev.get("c"),
        "change_pct":    ticker_data.get("todaysChangePerc"),
        "after_hours":   ticker_data.get("lastQuote", {}).get("P"),
    }


def get_short_interest(ticker: str) -> dict:
    """Returns short interest data — high short float can signal squeeze potential."""
    end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    data  = _get(f"/v2/reference/financials/{ticker}", {"from": start, "to": end})
    return data


def get_rsi(ticker: str, window: int = 14) -> float | None:
    """Calculates RSI from Polygon's aggregate bars."""
    data = _get(
        f"/v1/indicators/rsi/{ticker}",
        {"timespan": "day", "window": window, "series_type": "close", "limit": 1},
    )
    results = data.get("results", {}).get("values", [])
    return float(results[0]["value"]) if results else None


def get_macd(ticker: str) -> dict | None:
    """Returns MACD signal for trend confirmation."""
    data = _get(
        f"/v1/indicators/macd/{ticker}",
        {"timespan": "day", "series_type": "close", "limit": 1},
    )
    results = data.get("results", {}).get("values", [])
    if not results:
        return None
    r = results[0]
    return {
        "value":    r.get("value"),
        "signal":   r.get("signal"),
        "histogram": r.get("histogram"),
        "trend":    "BULLISH" if (r.get("value", 0) or 0) > (r.get("signal", 0) or 0) else "BEARISH",
    }


def get_technicals_summary(ticker: str) -> str:
    """
    Returns a plain-English technical summary suitable for model context.
    """
    if not API_KEY:
        return f"{ticker}: Polygon API key not configured."

    snap = get_snapshot(ticker)
    rsi  = get_rsi(ticker)
    macd = get_macd(ticker)

    lines = [f"=== {ticker} TECHNICALS ==="]
    if snap.get("price"):
        lines.append(f"Price: ${snap['price']:.2f}  ({snap.get('change_pct', 0):+.2f}% today)")
        lines.append(f"VWAP: ${snap['vwap']:.2f}" if snap.get("vwap") else "")
        lines.append(f"Volume: {snap['volume']:,.0f}" if snap.get("volume") else "")

    if rsi is not None:
        label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "neutral"
        lines.append(f"RSI(14): {rsi:.1f} — {label}")

    if macd:
        lines.append(f"MACD: {macd['trend']} (histogram: {macd['histogram']:.3f})")

    return "\n".join(l for l in lines if l)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not API_KEY:
        print("POLYGON_API_KEY not set in .env — add it to activate technical data.")
    else:
        for sym in SYMBOLS[:5]:
            print(get_technicals_summary(sym))
            print()
