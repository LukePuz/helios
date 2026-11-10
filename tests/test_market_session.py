"""market_data.market_session labels a time within the NYSE day (record-only;
audit 2026-11-07 05:07 #2)."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from market_data import market_session

_ET = ZoneInfo("America/New_York")


def _at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=_ET)


def test_labels_across_a_trading_day_and_a_weekend():
    assert market_session(_at(2026, 11, 6, 8, 5)) == "pre_open"
    assert market_session(_at(2026, 11, 6, 9, 30)) == "rth"
    assert market_session(_at(2026, 11, 6, 16, 0)) == "rth"
    assert market_session(_at(2026, 11, 6, 16, 47)) == "after_close"
    assert market_session(_at(2026, 11, 7, 10, 15)) == "closed_day"   # Saturday
    assert market_session(_at(2026, 11, 26, 10, 15)) == "closed_day"  # Thanksgiving


def test_review_session_never_raises():
    from agents import analyst
    with patch("market_data.market_session", side_effect=RuntimeError("x")):
        assert analyst._review_session() is None
