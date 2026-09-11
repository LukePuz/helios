"""
Tests for agents/price_action.compute_trend_template (2026-09-11, user
request): Mark Minervini's 8-point Trend Template gate. Uses synthetic bar
series rather than real market data so the exact criteria can be
controlled and verified directly.
"""
from agents.price_action import compute_trend_template, TREND_TEMPLATE_RS_LOOKBACK


def _bars(closes: list[float]) -> list[dict]:
    return [{"close": c, "open": c, "high": c, "low": c, "volume": 1_000_000} for c in closes]


def _uptrend_bars(n: int, start: float, end: float) -> list[dict]:
    """Monotonic linear rise from start to end over n bars -- guarantees
    50dma > 150dma > 200dma (each shorter window averages more-recent,
    higher prices), a rising 200dma, price at/near the 52-week high, and
    price far above the 52-week low, all at once."""
    step = (end - start) / (n - 1)
    return _bars([start + step * i for i in range(n)])


def _downtrend_bars(n: int, start: float, end: float) -> list[dict]:
    step = (start - end) / (n - 1)
    return _bars([start - step * i for i in range(n)])


_N = TREND_TEMPLATE_RS_LOOKBACK + 20 + 10  # matches compute_trend_template's own min_bars_needed + margin


class TestComputeTrendTemplate:
    def test_insufficient_history_returns_none(self):
        short_bars = _uptrend_bars(100, 50, 100)
        result = compute_trend_template("NVDA", bars=short_bars, spy_bars=_uptrend_bars(100, 50, 60))
        assert result is None

    def test_clean_uptrend_outperforming_spy_passes_all_eight(self):
        ticker_bars = _uptrend_bars(_N, 50.0, 150.0)   # +200% over the year
        spy_bars = _uptrend_bars(_N, 400.0, 480.0)     # +20% -- ticker clearly beats this
        result = compute_trend_template("NVDA", bars=ticker_bars, spy_bars=spy_bars)
        assert result is not None
        assert result["pass"] is True
        assert all(result["criteria"].values())
        assert "All 8 Trend Template criteria met" in result["reasoning"]

    def test_clean_downtrend_fails_most_criteria(self):
        ticker_bars = _downtrend_bars(_N, 150.0, 50.0)  # -66% over the year
        spy_bars = _uptrend_bars(_N, 400.0, 480.0)      # SPY up +20% while this is down hard
        result = compute_trend_template("XYZ", bars=ticker_bars, spy_bars=spy_bars)
        assert result is not None
        assert result["pass"] is False
        assert result["criteria"]["price_above_150dma_and_200dma"] is False
        assert result["criteria"]["relative_strength_proxy"] is False
        assert "Failed" in result["reasoning"]

    def test_missing_spy_data_fails_the_rs_criterion_closed(self):
        # Otherwise-perfect uptrend, but with no SPY data to compare
        # against -- must fail the RS proxy criterion (and therefore the
        # overall gate) rather than silently passing on 7/8.
        ticker_bars = _uptrend_bars(_N, 50.0, 150.0)
        result = compute_trend_template("NVDA", bars=ticker_bars, spy_bars=None)
        assert result is not None
        assert result["criteria"]["relative_strength_proxy"] is False
        assert result["pass"] is False
        assert "RS proxy" in result["reasoning"] or "SPY" in result["reasoning"]

    def test_short_spy_history_also_fails_closed(self):
        ticker_bars = _uptrend_bars(_N, 50.0, 150.0)
        short_spy = _uptrend_bars(50, 400.0, 420.0)  # not enough SPY history
        result = compute_trend_template("NVDA", bars=ticker_bars, spy_bars=short_spy)
        assert result["criteria"]["relative_strength_proxy"] is False
        assert result["pass"] is False

    def test_underperforming_spy_fails_only_the_rs_criterion(self):
        # A real uptrend that just isn't beating the index -- should fail
        # specifically on relative_strength_proxy, with every technical
        # criterion still passing.
        ticker_bars = _uptrend_bars(_N, 100.0, 110.0)   # +10%
        spy_bars = _uptrend_bars(_N, 400.0, 480.0)      # +20% -- SPY wins
        result = compute_trend_template("SLOW", bars=ticker_bars, spy_bars=spy_bars)
        assert result["criteria"]["relative_strength_proxy"] is False
        assert result["criteria"]["price_above_150dma_and_200dma"] is True
        assert result["pass"] is False
