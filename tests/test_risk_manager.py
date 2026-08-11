"""
Pure-logic unit tests for risk_manager.py — no live DB/IBKR/network required,
safe to run in CI. Uses mocks for the IBKR client and db module.
"""
from unittest.mock import MagicMock, patch

import pytest

import risk_manager


def _make_client(portfolio_value=100_000.0, positions=None, cash=None):
    client = MagicMock()
    # cash defaults to full portfolio value (no positions held) so existing
    # sizing tests aren't inadvertently capped to 0 shares by the
    # cash-reserve cap (MAX_TRADE_PCT_OF_CASH_RESERVE) added 2026-07-15 —
    # tests that care about cash-cap behavior specifically pass it in.
    client.get_account.return_value = MagicMock(
        portfolio_value=portfolio_value,
        cash=cash if cash is not None else portfolio_value,
    )
    client.get_all_positions.return_value = positions or []
    return client


class TestSizing:
    def test_shares_initial_is_integer(self):
        client = _make_client()
        with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(120.0, "mocked")), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted": "false",
                                                                    "high_water_mark": "100000"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            sizing = rm.size_position("NVDA", conviction=8, bucket="Compounder",
                                      regime="Risk-On", price=130.0)
        assert isinstance(sizing["shares_initial"], int)
        assert sizing["shares_initial"] > 0

    def test_sub_one_share_allocation_skips_gracefully(self):
        client = _make_client()
        with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(650_000.0, "mocked")), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted": "false",
                                                                    "high_water_mark": "100000"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            sizing = rm.size_position("BRKA", conviction=5, bucket="Moonshot",
                                      regime="Risk-Off", price=700_000.0)
        assert sizing["shares_initial"] == 0
        assert sizing["flags"]


class TestHalt:
    def test_persisted_halt_flag_blocks_trading(self):
        client = _make_client()
        with patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted": "true"}.get(k, d)):
            rm = risk_manager.RiskManager(client)
            assert rm.is_halted() is True

    def test_no_halt_flag_allows_trading(self):
        client = _make_client()
        with patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted": "false",
                                                                    "high_water_mark": "100000"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            assert rm.is_halted() is False


class TestConcentrationFailsClosed:
    def test_errored_check_fails_closed(self):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=100_000.0)
        client.get_all_positions.side_effect = Exception("IBKR down")

        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("NVDA", 0.05)

        assert result["passed"] is False
        assert "RISK CHECK FAILED" in result["flags"][0]

    def test_healthy_check_passes(self):
        client = _make_client(positions=[])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("NVDA", 0.05)
        assert result["passed"] is True


def _position(symbol, pct_of_portfolio, portfolio_value=100_000.0):
    return MagicMock(symbol=symbol, market_value=pct_of_portfolio * portfolio_value)


class TestConcentrationSizeCap:
    """2026-08-11: check_concentration used to be advisory only -- a failed
    check just appended text to a field nobody reviews in real time (auto-
    execution has no human-in-the-loop tap). The portfolio sat over its own
    60% AI-infra limit for a week-plus while every new ADD to an AI-infra
    name kept sailing through anyway. ai_infra_size_cap_pct is the actual
    enforcement mechanism now -- the largest proposed_size_pct for THIS
    ticker that keeps the total at or under the limit."""

    def test_cap_reflects_generous_headroom_when_well_under_the_limit(self):
        """cap is always a number for an AI-infra ticker (its own headroom),
        never None just because the check passed -- None specifically means
        "not an AI-infra ticker," not "under the limit." crew.py only acts
        when cap < proposed size, so a generous cap like this is a no-op."""
        client = _make_client(positions=[_position("NVDA", 0.30)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("MSFT", 0.10)  # 30% + 10% = 40%, well under 60%

        assert result["passed"] is True
        assert result["ai_infra_size_cap_pct"] == pytest.approx(0.30, abs=1e-6)

    def test_cap_is_none_for_a_non_ai_infra_ticker_even_when_total_already_breached(self):
        """A breach caused entirely by other, pre-existing AI-infra
        positions must not cap an unrelated ticker's size -- capping it
        wouldn't address the actual breach."""
        client = _make_client(positions=[_position("NVDA", 0.45), _position("AMD", 0.20)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("JPM", 0.05)  # not in ai_infra_keywords

        assert result["passed"] is False  # still flagged -- the breach is real
        assert result["ai_infra_size_cap_pct"] is None  # but not this ticker's to fix

    def test_cap_reflects_remaining_headroom_for_an_ai_infra_ticker(self):
        """55% already held in AI infra names, proposing another 10% PLTR
        (also AI infra) would hit 65% -- cap should be exactly the 5%
        headroom left under the 60% limit."""
        client = _make_client(positions=[_position("NVDA", 0.35), _position("AMD", 0.20)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("PLTR", 0.10)

        assert result["passed"] is False
        assert result["ai_infra_size_cap_pct"] == pytest.approx(0.05, abs=1e-6)

    def test_cap_is_zero_when_already_at_or_over_the_limit(self):
        client = _make_client(positions=[_position("NVDA", 0.45), _position("AMD", 0.20)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("MSFT", 0.05)  # 65% + 5% = 70%, already over before adding

        assert result["ai_infra_size_cap_pct"] == 0.0


class TestCorrelation:
    def test_price_correlation_perfect_positive(self):
        dates = [f"2026-01-{d:02d}" for d in range(1, 26)]
        returns = [0.01, -0.02, 0.015, 0.005, -0.01] * 4 + [0.02]
        prices_a, prices_b = [100.0], [200.0]
        for r in returns:
            prices_a.append(prices_a[-1] * (1 + r))
            prices_b.append(prices_b[-1] * (1 + r))
        bars_a = [{"date": d, "close": p} for d, p in zip(dates, prices_a)]
        bars_b = [{"date": d, "close": p} for d, p in zip(dates, prices_b)]

        corr = risk_manager._price_correlation(bars_a, bars_b)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_price_correlation_perfect_negative(self):
        dates = [f"2026-01-{d:02d}" for d in range(1, 26)]
        returns = [0.01, -0.02, 0.015, 0.005, -0.01] * 4 + [0.02]
        prices_a, prices_c = [100.0], [50.0]
        for r in returns:
            prices_a.append(prices_a[-1] * (1 + r))
            prices_c.append(prices_c[-1] * (1 - r))
        bars_a = [{"date": d, "close": p} for d, p in zip(dates, prices_a)]
        bars_c = [{"date": d, "close": p} for d, p in zip(dates, prices_c)]

        corr = risk_manager._price_correlation(bars_a, bars_c)
        assert corr == pytest.approx(-1.0, abs=1e-9)

    def test_price_correlation_insufficient_history_returns_none(self):
        bars_a = [{"date": "2026-01-01", "close": 100.0}] * 3
        bars_b = [{"date": "2026-01-01", "close": 200.0}] * 3
        assert risk_manager._price_correlation(bars_a, bars_b) is None

    def test_check_correlation_fails_closed_on_error(self):
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("IBKR down")
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_correlation("NVDA")
        assert result["reduce_by"] == 0.25
        assert "RISK CHECK FAILED" in result["note"]

    def test_check_correlation_falls_back_when_no_price_history(self):
        pos = MagicMock(symbol="AMD")
        client = MagicMock()
        client.get_all_positions.return_value = [pos]
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        with patch("db.get_price_history_bulk", return_value={}):
            result = rm.check_correlation("ZZZNOTCACHED")
        assert "price history not yet cached" in result["note"] or "Sector unknown" in result["note"]
