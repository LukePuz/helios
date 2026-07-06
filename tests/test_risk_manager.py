"""
Pure-logic unit tests for risk_manager.py — no live DB/IBKR/network required,
safe to run in CI. Uses mocks for the IBKR client and db module.
"""
from unittest.mock import MagicMock, patch

import pytest

import risk_manager


def _make_client(portfolio_value=100_000.0, positions=None):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=portfolio_value)
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
