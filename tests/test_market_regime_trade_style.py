"""
Unit tests for agents.market_regime._score_trade_style — the
Trending/Expanding vs Choppy/Distribution classifier added 2026-08-29 (see
that function's docstring for the motivating Q2->Q3 regime-shift example)
— and get_trade_style_for_ticker, the per-ticker sector-aware resolver
added the same day.

Pure-function tests against synthetic SPY-shaped bars, hand-verified before
asserting rather than trusting the arithmetic blind — same convention used
throughout momentum-scanner's test suite.
"""
from unittest.mock import patch

from agents.market_regime import (
    _score_trade_style,
    TRADE_STYLE_ER_THRESHOLD,
    get_trade_style_for_ticker,
)


def _bars(closes: list[float]) -> list[dict]:
    return [{"close": c} for c in closes]


class TestScoreTradeStyle:
    def test_insufficient_data_defaults_cautious(self):
        label, er, notes = _score_trade_style(_bars([100.0] * 10))
        assert label == "Choppy/Distribution"
        assert er == 0.0
        assert "Insufficient" in notes

    def test_clean_smooth_uptrend_is_trending_expanding(self):
        # Every bar up $1, no chop at all -- net_move == path_length, ER == 1.0
        closes = [100.0 + i for i in range(40)]
        label, er, notes = _score_trade_style(_bars(closes))
        assert er == 1.0
        assert label == "Trending/Expanding"
        assert "rising 20DMA" in notes

    def test_pure_chop_sideways_is_choppy_distribution(self):
        # Oscillates +2/-2 around 100 with zero net progress over the
        # window -- net_move ~= 0, path_length large -> ER near 0.
        closes = []
        for i in range(40):
            closes.append(100.0 + (2 if i % 2 == 0 else -2))
        label, er, notes = _score_trade_style(_bars(closes))
        assert er < TRADE_STYLE_ER_THRESHOLD
        assert label == "Choppy/Distribution"
        assert "no clean net progress" in notes

    def test_smooth_selloff_after_uptrend_is_distribution_not_expansion(self):
        # First 20 bars rise smoothly (builds the 30-bars-ago baseline for
        # the SMA slope check), last 20 fall smoothly and just as cleanly
        # -- high efficiency ratio, but the 20DMA is now falling, so this
        # must NOT be classified as a tradeable uptrend.
        up = [100.0 + i for i in range(20)]
        down = [up[-1] - i for i in range(1, 21)]
        closes = up + down
        label, er, notes = _score_trade_style(_bars(closes))
        assert er >= TRADE_STYLE_ER_THRESHOLD  # the recent 20-bar move IS smooth
        assert label == "Choppy/Distribution"   # but it's a smooth DOWN move
        assert "falling" in notes

    def test_efficiency_ratio_rounds_to_three_decimals(self):
        closes = [100.0 + i for i in range(40)]
        _, er, _ = _score_trade_style(_bars(closes))
        assert er == round(er, 3)


class TestGetTradeStyleForTicker:
    def test_resolves_via_gics_sector_and_etf_map(self):
        with patch("ingestion.sp500_universe.get_sp500_constituents",
                    return_value={"UNH": {"gics_sector": "Health Care"}}), \
             patch("agents.market_regime.get_cached_sector_regime",
                    return_value={"trade_style": "Choppy/Distribution"}) as mock_sector:
            result = get_trade_style_for_ticker("UNH")
        assert result == "Choppy/Distribution"
        mock_sector.assert_called_once_with("XLV", regime_type="gics_sector")

    def test_non_sp500_ticker_returns_none(self):
        with patch("ingestion.sp500_universe.get_sp500_constituents", return_value={}):
            assert get_trade_style_for_ticker("SOME_MICROCAP") is None

    def test_sector_regime_unavailable_returns_none_not_a_crash(self):
        with patch("ingestion.sp500_universe.get_sp500_constituents",
                    return_value={"UNH": {"gics_sector": "Health Care"}}), \
             patch("agents.market_regime.get_cached_sector_regime",
                    side_effect=Exception("regime fetch failed")):
            assert get_trade_style_for_ticker("UNH") is None
