"""SPY/QQQ price-history fetch retries and falls back to Alpaca (roadmap 3)."""
from unittest.mock import patch

from ingestion import price_history

_BAR = {"date": "2026-10-27", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}


def test_retries_then_succeeds():
    with patch("ingestion.price_history._fetch_bars_yf", side_effect=[[], [_BAR]]), \
         patch("ingestion.price_history.time.sleep"):
        assert price_history._fetch_bars("SPY") == [_BAR]


def test_benchmark_falls_back_to_alpaca():
    with patch("ingestion.price_history._fetch_bars_yf", return_value=[]), \
         patch("ingestion.price_history.time.sleep"), \
         patch("agents.market_regime._fetch_bars", return_value=[_BAR]) as alpaca:
        assert price_history._fetch_bars("SPY") == [_BAR]
    alpaca.assert_called_once()


def test_watchlist_ticker_has_no_alpaca_fallback():
    with patch("ingestion.price_history._fetch_bars_yf", return_value=[]), \
         patch("ingestion.price_history.time.sleep"), \
         patch("agents.market_regime._fetch_bars") as alpaca:
        assert price_history._fetch_bars("NVDA") == []
    alpaca.assert_not_called()
