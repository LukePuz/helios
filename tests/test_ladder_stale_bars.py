"""The nightly ladder logs tickers whose newest bar predates the latest
session (audit 11-06 17:34 #9): it runs on a clock, not after the price
history refresh. Log-only."""
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks

_ET = ZoneInfo("America/New_York")


def test_friday_evening_wants_fridays_bar():
    now = datetime(2026, 11, 6, 17, 15, tzinfo=_ET)
    with patch("knowledge_loader.get_all_tickers", return_value=["NVDA", "MSFT", "ANET"]), \
         patch("db.get_newest_bar_dates", return_value={"NVDA": date(2026, 11, 6), "MSFT": date(2026, 11, 5)}):
        out = tasks._ladder_stale_bars(now)
    assert out == {"session": "2026-11-06", "tickers": 3, "stale": ["ANET", "MSFT"]}


def test_weekend_wants_the_friday_session():
    now = datetime(2026, 11, 8, 12, 0, tzinfo=_ET)
    with patch("knowledge_loader.get_all_tickers", return_value=["NVDA"]), \
         patch("db.get_newest_bar_dates", return_value={"NVDA": date(2026, 11, 6)}):
        assert tasks._ladder_stale_bars(now)["stale"] == []


def test_never_raises():
    with patch("knowledge_loader.get_all_tickers", side_effect=RuntimeError("x")):
        assert tasks._ladder_stale_bars() == {"error": "RuntimeError"}


def test_an_empty_bar_read_is_reported_as_a_read_failure_not_all_stale():
    with patch("knowledge_loader.get_all_tickers", return_value=["NVDA", "MSFT"]), \
         patch("db.get_newest_bar_dates", return_value={}):
        out = tasks._ladder_stale_bars(datetime(2026, 11, 6, 17, 15, tzinfo=_ET))
    assert out["error"] == "no_bar_dates_read" and "stale" not in out
