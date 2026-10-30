"""Luke D6 (10-29): outside regular hours, price and label come from the
last extended-hours trade, not the prior regular-session close."""
from datetime import datetime, timezone
from unittest.mock import patch

import market_data

PRE = datetime(2026, 10, 29, 11, 52, tzinfo=timezone.utc)      # 07:52 ET
RTH = datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc)       # 11:00 ET


def _bars(t):
    return [{"time": t, "open": 1, "high": 1, "low": 1, "close": 123.0, "volume": 10}]


def test_premarket_quote_and_label():
    with patch("market_data.get_historical_bars", return_value=_bars("2026-10-29 07:51:00-04:00")), \
         patch("market_data.get_priced_quote", return_value={"price": 150.0}):
        q = market_data.get_session_quote("NOW", now_utc=PRE)
        assert q["session"] == "pre-market" and q["price"] == 123.0 and q["prior_close"] == 150.0
        with patch("market_data.datetime") as dt:
            dt.now.return_value = PRE
            dt.fromisoformat = datetime.fromisoformat
            label = market_data.session_price_label("NOW", 123.0)
    assert "pre-market, last trade 07:51 ET; prior close $150.00, -18.0%" in label


def test_regular_hours_returns_nothing():
    assert market_data.get_session_quote("NOW", now_utc=RTH) is None


def test_last_bar_inside_rth_means_no_newer_trade():
    with patch("market_data.get_historical_bars", return_value=_bars("2026-10-28 15:59:00-04:00")), \
         patch("market_data.get_priced_quote", return_value={"price": 150.0}):
        assert market_data.get_session_quote("NOW", now_utc=PRE) is None
