"""Luke D3/D9 (10-29): regime and scanner score completed bars only; the
liquidity score can reach Very Bearish."""
from datetime import datetime, timezone
from unittest.mock import patch

from agents import market_regime

BARS = [{"date": "2026-10-28", "close": 1.0}, {"date": "2026-10-29", "close": 2.0}]


def test_unfinished_bar_dropped_before_close():
    morning = datetime(2026, 10, 29, 13, 35, tzinfo=timezone.utc)     # 09:35 ET
    assert market_regime.completed_bars(BARS, now=morning) == BARS[:1]


def test_bar_kept_after_close_and_on_other_days():
    evening = datetime(2026, 10, 29, 20, 5, tzinfo=timezone.utc)      # 16:05 ET
    assert market_regime.completed_bars(BARS, now=evening) == BARS
    next_day = datetime(2026, 10, 30, 13, 35, tzinfo=timezone.utc)
    assert market_regime.completed_bars(BARS, now=next_day) == BARS
    assert market_regime.completed_bars([], now=evening) == []


def test_switch_off_restores_old_reads():
    morning = datetime(2026, 10, 29, 13, 35, tzinfo=timezone.utc)
    with patch("config.REGIME_COMPLETED_BARS", False):
        assert market_regime.completed_bars(BARS, now=morning) == BARS


def test_liquidity_very_bearish_reachable():
    def score(spread):
        return market_regime._score_liquidity({"FEDFUNDS": {"value": 5.6}, "T10Y2Y": {"value": spread}})[0]
    S = market_regime._SCORE
    assert score(-0.6) == S["Very Bearish"]
    assert score(-0.3) == S["Bearish"]
