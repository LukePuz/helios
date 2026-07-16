"""
Tests for agents/crew.py's _route_to_approval — the routing bug where
TRIM/EXIT on a held position was silently dropped by buy-sizing logic
(size_position returns 0 shares for an at-target held name), and the
rebuy-cooldown gate that plugs the same-ticker duplicate-buy burst from
2026-07-15. No live DB/IBKR required.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents import crew


def _analyst_result(action="TRIM", ticker="NVDA", conviction=6, bucket="Accelerator"):
    return {
        "ticker": ticker,
        "action": action,
        "conviction": conviction,
        "thesis_bucket": bucket,
        "reasoning": "test",
    }


class TestSellRouting:
    def test_trim_on_held_position_never_calls_size_position(self):
        """TRIM/EXIT must be sized from the held position, never through
        RiskManager.size_position — buy sizing returns 0 shares for an
        at-target held name, which used to silently drop every analyst
        sell recommendation."""
        result = _analyst_result(action="TRIM")
        position = MagicMock(qty=90.0)
        client = MagicMock()
        client.get_open_position.return_value = position

        with patch("db.has_active_approval", return_value=False), \
             patch("db.log_pending_approval", return_value=10), \
             patch("db.claim_approval_for_execution", return_value=True), \
             patch("db.get_approval_by_id", return_value={"id": 10}), \
             patch("execution.execute_approved", return_value=True) as mock_execute, \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls:
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        # size_position must never have been called for a TRIM/EXIT
        mock_rm_cls.return_value.size_position.assert_not_called()
        mock_execute.assert_called_once()

    def test_exit_with_no_position_logs_signal_and_returns(self):
        """EXIT recommended on a ticker we don't actually hold must not
        fall through to buy sizing — just log and stop."""
        result = _analyst_result(action="EXIT")
        client = MagicMock()
        client.get_open_position.side_effect = Exception("No open position in NVDA")

        with patch("db.has_active_approval", return_value=False), \
             patch("db.log_signal") as mock_log_signal, \
             patch("db.log_pending_approval") as mock_log_approval, \
             patch("risk_manager.RiskManager") as mock_rm_cls:
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_log_approval.assert_not_called()
        mock_rm_cls.return_value.size_position.assert_not_called()
        assert mock_log_signal.call_args[0][1] == "SELL_NO_POSITION"


class TestRebuyCooldown:
    def test_equal_conviction_rebuy_within_window_is_skipped(self):
        result = _analyst_result(action="ADD", conviction=6)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval",
                   return_value={"conviction": 6, "decided_at": "2026-07-15T12:00:00Z"}), \
             patch("db.log_signal") as mock_log_signal, \
             patch("risk_manager.RiskManager") as mock_rm_cls:
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_rm_cls.return_value.size_position.assert_not_called()
        assert mock_log_signal.call_args[0][1] == "COOLDOWN_SKIPPED"

    def test_higher_conviction_rebuy_proceeds_despite_cooldown(self):
        result = _analyst_result(action="ADD", conviction=8)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval",
                   return_value={"conviction": 6, "decided_at": "2026-07-15T12:00:00Z"}), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval", return_value=-1):
            mock_rm_cls.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0,
            }
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        # Higher conviction than the cooled-down prior buy must reach
        # sizing (this test doesn't need it to succeed, just to be reached).
        mock_rm_cls.return_value.size_position.assert_called_once()

    def test_no_prior_buy_proceeds_normally(self):
        result = _analyst_result(action="OPEN", conviction=7)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls:
            mock_rm_cls.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": [], "position_size_pct": 0,
            }
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_rm_cls.return_value.size_position.assert_called_once()
