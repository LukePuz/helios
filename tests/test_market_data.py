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

import pytest

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


class TestPriceSource:
    """price_source records which bars the price came from; the stale
    verdict and the age are the same as without it."""

    def _quote(self, minute_bars, daily_bars, market_open=True):
        def bars(ticker, duration, bar_size, **kw):
            return minute_bars if bar_size == "1 min" else daily_bars
        with patch("market_data.get_historical_bars", side_effect=bars) as gh, \
             patch("market_data._is_market_hours", return_value=market_open):
            return market_data.get_priced_quote("AAPL"), gh

    def _fresh_raw(self, minutes=1):
        t = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).astimezone(
            timezone(timedelta(hours=-4)))
        raw = t.strftime("%Y-%m-%d %H:%M:%S%z")
        return raw[:-2] + ":" + raw[-2:]

    def test_one_minute_bars(self):
        q, gh = self._quote(_bars(self._fresh_raw(), 101.0), _bars("2026-07-28", 99.0))
        assert q["price_source"] == "1min" and q["price"] == 101.0
        assert q["is_stale"] is False
        assert gh.call_count == 1

    def test_daily_fallback_when_no_minute_bars(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        q, gh = self._quote([], _bars(today, 99.0))
        assert q["price_source"] == "daily_fallback" and q["price"] == 99.0
        assert gh.call_count == 2
        assert gh.call_args_list[1].kwargs["bar_size"] == "1 day"

    def test_no_bars_at_all_is_still_none(self):
        q, _ = self._quote([], [])
        assert q is None

    @pytest.mark.parametrize("raw_fn, market_open", [
        (lambda s: s._fresh_raw(1), True),
        (lambda s: s._fresh_raw(1), False),
        (lambda s: s._fresh_raw(180), True),
        (lambda s: datetime.now(timezone.utc).strftime("%Y-%m-%d"), True),
        (lambda s: (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"), True),
        (lambda s: "garbage-not-a-date", True),
    ])
    def test_stale_verdict_and_age_do_not_depend_on_the_source(self, raw_fn, market_open):
        """The same bar gives the same is_stale, age and market_open whether
        it came from the 1-min or the daily request."""
        raw = raw_fn(self)
        via_min, _ = self._quote(_bars(raw), [], market_open)
        via_day, _ = self._quote([], _bars(raw), market_open)
        assert (via_min["price_source"], via_day["price_source"]) == ("1min", "daily_fallback")
        for k in ("price", "bar_time", "is_stale", "market_open"):
            assert via_min[k] == via_day[k], k
        if via_min["age_minutes"] is None:
            assert via_day["age_minutes"] is None
        else:
            assert abs(via_min["age_minutes"] - via_day["age_minutes"]) <= 0.2

    def test_date_only_daily_bar_is_stale_during_rth(self):
        """The fallback's date-only bar is aged from midnight UTC: stale
        whenever it is more than STALE_AFTER_MINUTES past midnight UTC.
        Unchanged; recorded here with its source."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        q, _ = self._quote([], _bars(yesterday))
        assert q["price_source"] == "daily_fallback" and q["is_stale"] is True
        assert set(q) == {"price", "bar_time", "age_minutes", "is_stale", "market_open",
                          "price_source"}


def _daily_intraday(prior_close, opens_and_closes):
    """opens_and_closes: list of (open, close) tuples, one per 1-min bar,
    in order since the open -- lets a test describe a price path cheaply."""
    daily = [{"time": "2026-07-28", "open": prior_close, "high": prior_close,
              "low": prior_close, "close": prior_close, "volume": 1_000_000}]
    intraday = [
        {"time": f"2026-07-29 09:{30+i}:00-04:00", "open": o, "high": max(o, c),
         "low": min(o, c), "close": c, "volume": 10_000}
        for i, (o, c) in enumerate(opens_and_closes)
    ]

    def fake(ticker, duration="20 D", bar_size="1 day", **kwargs):
        return daily if bar_size == "1 day" else intraday
    return fake


class TestGapConfirmation:
    """Regression coverage for the 2026-07-29 TER incident: an opening gap
    was treated as a valid, tradeable price the instant the market-hours
    gate cleared, with no check for whether the gap was actually holding.
    TER gapped from a $320.65 close to a $368 open and fully round-tripped
    (low $319.41) within the session -- market_data.check_gap_confirmation
    is the deterministic judge of "holding vs. fading" that execution.py's
    BUY path now consults before firing."""

    def test_small_move_is_no_gap(self):
        """A move below the threshold isn't a "gap" at all -- the ordinary
        drift guard governs, this check doesn't apply."""
        fake = _daily_intraday(prior_close=100.0, opens_and_closes=[(101.0, 101.2)])
        with patch("market_data.get_historical_bars", side_effect=fake):
            result = market_data.check_gap_confirmation("AAPL", today="2026-07-29")
        assert result["status"] == "no_gap"

    def test_gap_holding_after_enough_bars_is_confirmed(self):
        """A real gap (>5%) where price has stayed near the open across
        enough bars to judge -- must not be flagged as fading."""
        path = [(110.0, 110.0)] * 12  # 12 bars, essentially flat at the open
        fake = _daily_intraday(prior_close=100.0, opens_and_closes=path)
        with patch("market_data.get_historical_bars", side_effect=fake):
            result = market_data.check_gap_confirmation("AAPL", today="2026-07-29")
        assert result["status"] == "confirmed"
        assert result["gap_pct"] == pytest.approx(0.10)

    def test_not_enough_bars_yet_is_pending(self):
        """Right after the open there's no way to tell holding from fading
        -- must not guess either way."""
        path = [(110.0, 110.0)] * 3
        fake = _daily_intraday(prior_close=100.0, opens_and_closes=path)
        with patch("market_data.get_historical_bars", side_effect=fake):
            result = market_data.check_gap_confirmation("AAPL", today="2026-07-29")
        assert result["status"] == "pending"

    def test_gap_fully_reversing_is_failed(self):
        """The exact TER shape: gapped up, then gave essentially the whole
        move back -- must be flagged as failed, not confirmed."""
        path = [(110.0, 108.0), (108.0, 104.0), (104.0, 101.0),
                (101.0, 100.5), (100.5, 100.2), (100.2, 100.1),
                (100.1, 100.0), (100.0, 100.0), (100.0, 100.0),
                (100.0, 100.0), (100.0, 100.0)]
        fake = _daily_intraday(prior_close=100.0, opens_and_closes=path)
        with patch("market_data.get_historical_bars", side_effect=fake):
            result = market_data.check_gap_confirmation("AAPL", today="2026-07-29")
        assert result["status"] == "failed"
        assert result["retracement_pct"] >= 0.5

    def test_todays_partial_daily_bar_is_not_the_prior_close(self):
        """Audit A1031-1453-1: in market hours the last daily bar is today's
        partial one; a gap-up that kept rising read as a reversed gap-down."""
        path = [(110.0, 111.0)] * 12 + [(111.0, 116.0)]
        fake_base = _daily_intraday(prior_close=100.0, opens_and_closes=path)
        today_partial = {"time": "2026-07-29", "open": 110.0, "high": 116.0, "low": 110.0,
                         "close": 116.0, "volume": 500_000}

        def fake(ticker, duration="20 D", bar_size="1 day", **kwargs):
            bars = fake_base(ticker, duration, bar_size)
            return bars + [today_partial] if bar_size == "1 day" else bars
        with patch("market_data.get_historical_bars", side_effect=fake):
            result = market_data.check_gap_confirmation("AAPL", today="2026-07-29")
        assert result["prior_close"] == 100.0 and result["gap_pct"] == pytest.approx(0.10)
        assert result["status"] == "confirmed"

    def test_no_prior_session_bar_is_no_gap(self):
        intraday = [{"time": "2026-07-29 09:30:00-04:00", "open": 110.0, "high": 110.0,
                     "low": 110.0, "close": 110.0, "volume": 1}]
        daily = [{"time": "2026-07-29", "open": 110.0, "high": 110.0, "low": 110.0,
                  "close": 110.0, "volume": 1}]
        with patch("market_data.get_historical_bars",
                   side_effect=lambda t, duration="20 D", bar_size="1 day", **k:
                   daily if bar_size == "1 day" else intraday):
            assert market_data.check_gap_confirmation("AAPL", today="2026-07-29")["status"] == "unreadable"

    def test_before_todays_first_bar_is_no_gap(self):
        """Pre-open the 1-min bars are yesterday's session: don't judge yesterday's gap."""
        path = [(110.0, 100.0)] * 12                         # yesterday's faded gap
        fake = _daily_intraday(prior_close=100.0, opens_and_closes=path)
        with patch("market_data.get_historical_bars", side_effect=fake):
            assert market_data.check_gap_confirmation("AAPL", today="2026-07-30")["status"] == "no_gap"

    def test_no_data_available_is_unreadable_not_no_gap(self):
        """Professor M148: missing bars mean "can't tell", which the BUY path
        retries and then skips, not "no gap"."""
        with patch("market_data.get_historical_bars", return_value=[]):
            result = market_data.check_gap_confirmation("AAPL", today="2026-07-29")
        assert result["status"] == "unreadable"
        with patch("market_data.get_historical_bars", side_effect=RuntimeError("pacing")):
            assert market_data.check_gap_confirmation("AAPL", today="2026-07-29")["status"] == "unreadable"
