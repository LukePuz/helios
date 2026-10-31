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


def test_log_order_stores_the_permanent_id():
    """Professor M127: orders.perm_id (migration 70); 0 is stored as unknown."""
    from unittest.mock import MagicMock, patch
    import db
    for perm, stored in ((1847293, 1847293), (0, None), (None, None)):
        cur = MagicMock()
        cur.fetchone.return_value = (7,)
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("db._writer_conn", return_value=conn), patch("db._put_conn"):
            assert db.log_order("NVDA", "SELL", "MARKET", 3, broker_order_id=12, perm_id=perm) == 7
        sql, params = cur.execute.call_args.args
        assert "perm_id" in sql and sql.count("%s") == len(params)
        assert params[-1] == stored


class TestRecordedSellOrderIds:
    """Professor m102: a row with a perm_id is matched on it alone, since
    orderIds collide across clientIds; only rows without one fall back to
    their orderId."""

    def _run(self, rows):
        from unittest.mock import MagicMock, patch
        cur = MagicMock()
        if isinstance(rows, Exception):
            cur.execute.side_effect = rows
        else:
            cur.fetchall.return_value = rows
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            return db.recorded_sell_order_ids("DDOG")

    def test_splits_rows_by_whether_they_carry_a_perm_id(self):
        assert self._run([(40, 5001), (41, None), (None, 5002)]) == ({41}, {5001, 5002})

    def test_a_query_error_returns_none(self):
        assert self._run(RuntimeError("down")) is None
