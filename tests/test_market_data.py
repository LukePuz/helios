"""
Tests for market_data.get_priced_quote's bar-time parsing and staleness
computation — regression coverage for the 2026-07-28/29 incident where
ib_insync's intraday bar timestamps changed format to a real UTC-offset
suffix ('2026-07-29 13:04:00-04:00'). The old parser only handled a
' US/Eastern'-suffixed string and a handful of offset-less formats, so it
silently failed on every intraday quote, leaving age_minutes=None — which
market_data.is_stale's "age_min is None -> stale" fallback then turned into
every single quote being reported stale, even fresh ones seconds old. That
false staleness fired a Telegram alert on every trade via execution.py's
drift check.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import market_data


def _bars(time_str: str, price: float = 100.0):
    return [{"time": time_str, "open": price, "high": price, "low": price,
              "close": price, "volume": 1000}]


class TestBarTimeParsing:
    def test_current_tz_offset_format_is_fresh_not_stale(self):
        """Current ib_insync format: 'YYYY-MM-DD HH:MM:SS-04:00'. A bar from
        one minute ago must be treated as fresh, not stale."""
        now = datetime.now(timezone.utc)
        bar_time_et = (now - timedelta(minutes=1)).astimezone(
            timezone(timedelta(hours=-4))
        )
        raw = bar_time_et.strftime("%Y-%m-%d %H:%M:%S%z")
        raw = raw[:-2] + ":" + raw[-2:]  # -0400 -> -04:00
        with patch("market_data.get_historical_bars", return_value=_bars(raw)), \
             patch("market_data._is_market_hours", return_value=True):
            q = market_data.get_priced_quote("AAPL")
        assert q is not None
        assert q["age_minutes"] is not None
        assert q["age_minutes"] < 5
        assert q["is_stale"] is False

    def test_old_us_eastern_suffix_format_still_parses(self):
        """Legacy format some IBKR paths may still return: date-time with a
        trailing ' US/Eastern' label instead of a numeric offset."""
        now = datetime.now(timezone.utc)
        raw = (now - timedelta(minutes=2)).strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
        with patch("market_data.get_historical_bars", return_value=_bars(raw)), \
             patch("market_data._is_market_hours", return_value=True):
            q = market_data.get_priced_quote("AAPL")
        assert q is not None
        assert q["age_minutes"] is not None
        assert q["is_stale"] is False

    def test_daily_bar_date_only_format_still_parses(self):
        """Daily bars ('2 D' fallback) carry a date with no time component."""
        raw = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with patch("market_data.get_historical_bars", return_value=_bars(raw)), \
             patch("market_data._is_market_hours", return_value=True):
            q = market_data.get_priced_quote("AAPL")
        assert q is not None
        # Same-day date-only bar parses to midnight -- may or may not be
        # "stale" depending on time of day, but must not be age_minutes=None.
        assert q["age_minutes"] is not None

    def test_truly_unparseable_time_falls_back_to_stale(self):
        """An unrecognized format should still fail safe (stale=True), not
        raise -- this is the deliberate fallback, not the bug."""
        with patch("market_data.get_historical_bars", return_value=_bars("garbage-not-a-date")), \
             patch("market_data._is_market_hours", return_value=True):
            q = market_data.get_priced_quote("AAPL")
        assert q is not None
        assert q["age_minutes"] is None
        assert q["is_stale"] is True

    def test_stale_bar_far_in_past_is_stale(self):
        """A genuinely old bar (hours ago) must still be reported stale --
        the fix must not make everything look fresh."""
        now = datetime.now(timezone.utc)
        bar_time_et = (now - timedelta(hours=3)).astimezone(timezone(timedelta(hours=-4)))
        raw = bar_time_et.strftime("%Y-%m-%d %H:%M:%S%z")
        raw = raw[:-2] + ":" + raw[-2:]
        with patch("market_data.get_historical_bars", return_value=_bars(raw)), \
             patch("market_data._is_market_hours", return_value=True):
            q = market_data.get_priced_quote("AAPL")
        assert q is not None
        assert q["age_minutes"] > 100
        assert q["is_stale"] is True
