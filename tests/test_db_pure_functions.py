"""
Pure-logic unit tests for db.py's calculation helpers. These do NOT touch
Postgres — only the extracted pure functions (compute_slippage_pct,
compute_outcome_metrics) are exercised, so this suite runs in any CI
environment with no database credentials.
"""
import pytest

import db


class TestSlippage:
    def test_buy_paying_more_is_positive_cost_slippage(self):
        assert db.compute_slippage_pct(100.0, 101.5, "BUY") == pytest.approx(0.015)

    def test_sell_receiving_less_is_positive_cost_slippage(self):
        assert db.compute_slippage_pct(100.0, 98.5, "SELL") == pytest.approx(0.015)

    def test_buy_paying_less_is_negative_slippage(self):
        assert db.compute_slippage_pct(100.0, 99.0, "BUY") == pytest.approx(-0.01)

    def test_missing_decision_price_returns_none(self):
        assert db.compute_slippage_pct(None, 101.5, "BUY") is None

    def test_missing_fill_price_returns_none(self):
        assert db.compute_slippage_pct(100.0, None, "BUY") is None


class TestOutcomeMetrics:
    def test_winning_trade_with_positive_alpha(self):
        m = db.compute_outcome_metrics(
            entry_price=100.0, exit_price=120.0,
            entry_benchmark_price=500.0, exit_benchmark_price=550.0,
        )
        assert m["outcome_pct"] == pytest.approx(0.20)
        assert m["was_correct"] is True
        assert m["benchmark_return_pct"] == pytest.approx(0.10)
        assert m["alpha_pct"] == pytest.approx(0.10)

    def test_losing_trade_marked_incorrect(self):
        m = db.compute_outcome_metrics(
            entry_price=100.0, exit_price=90.0,
            entry_benchmark_price=None, exit_benchmark_price=None,
        )
        assert m["outcome_pct"] == pytest.approx(-0.10)
        assert m["was_correct"] is False
        assert m["benchmark_return_pct"] is None
        assert m["alpha_pct"] is None

    def test_missing_benchmark_prices_skips_alpha_only(self):
        m = db.compute_outcome_metrics(
            entry_price=100.0, exit_price=110.0,
            entry_benchmark_price=None, exit_benchmark_price=550.0,
        )
        assert m["outcome_pct"] == pytest.approx(0.10)
        assert m["benchmark_return_pct"] is None
        assert m["alpha_pct"] is None

    def test_beating_a_falling_benchmark_is_positive_alpha(self):
        # Portfolio flat, benchmark down 10% -> alpha should be +10%
        m = db.compute_outcome_metrics(
            entry_price=100.0, exit_price=100.0,
            entry_benchmark_price=500.0, exit_benchmark_price=450.0,
        )
        assert m["outcome_pct"] == pytest.approx(0.0)
        assert m["benchmark_return_pct"] == pytest.approx(-0.10)
        assert m["alpha_pct"] == pytest.approx(0.10)
