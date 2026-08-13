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


class TestConcentrationCapping:
    """2026-08-11: a failed concentration check used to only append text to
    a flags field nobody reviews in real time -- the portfolio sat over its
    own 60% AI-infra limit for a week-plus while every new ADD kept sailing
    through anyway. check_concentration now returns ai_infra_size_cap_pct;
    _route_to_approval must actually scale sizing down to it instead of
    just logging the breach."""

    def _base_sizing(self):
        return {
            "shares_initial": 10, "initial_dollar": 1000.0, "position_size_pct": 0.10,
            "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0,
        }

    def test_size_is_scaled_down_to_the_cap(self):
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=8)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval", return_value=-1) as mock_log_approval, \
             patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}):
            mock_rm_cls.return_value.size_position.return_value = self._base_sizing()
            mock_rm_cls.return_value.check_concentration.return_value = {
                "passed": False,
                "flags": ["AI infrastructure exposure 65.0% exceeds 60% limit"],
                "ai_infrastructure_total": 0.65,
                "ai_infra_size_cap_pct": 0.05,  # half of the proposed 10%
            }
            mock_rm_cls.return_value.check_correlation.return_value = {"reduce_by": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_log_approval.assert_called_once()
        kwargs = mock_log_approval.call_args.kwargs
        assert kwargs["size_pct"] == pytest.approx(0.05, abs=1e-6)
        assert kwargs["risk_json"]["shares_initial"] == 5  # 10 * (0.05/0.10)

    def test_zero_headroom_skips_the_entry_entirely(self):
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=8)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval") as mock_log_approval, \
             patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}):
            mock_rm_cls.return_value.size_position.return_value = self._base_sizing()
            mock_rm_cls.return_value.check_concentration.return_value = {
                "passed": False,
                "flags": ["AI infrastructure exposure 70.0% exceeds 60% limit"],
                "ai_infrastructure_total": 0.70,
                "ai_infra_size_cap_pct": 0.0,  # already at/over the limit
            }
            mock_rm_cls.return_value.check_correlation.return_value = {"reduce_by": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_log_approval.assert_not_called()

    def test_a_ticker_not_itself_ai_infra_is_never_capped(self):
        """cap=None means this ticker isn't what's driving the breach --
        must proceed at full proposed size, not get scaled to 0 by a stray
        None being treated as "no room"."""
        result = _analyst_result(action="ADD", ticker="JPM", conviction=8)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval", return_value=-1) as mock_log_approval, \
             patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}):
            mock_rm_cls.return_value.size_position.return_value = self._base_sizing()
            mock_rm_cls.return_value.check_concentration.return_value = {
                "passed": False,
                "flags": ["AI infrastructure exposure 65.0% exceeds 60% limit"],
                "ai_infrastructure_total": 0.65,
                "ai_infra_size_cap_pct": None,
            }
            mock_rm_cls.return_value.check_correlation.return_value = {"reduce_by": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_log_approval.assert_called_once()
        kwargs = mock_log_approval.call_args.kwargs
        assert kwargs["size_pct"] == pytest.approx(0.10, abs=1e-6)  # unchanged
        assert kwargs["risk_json"]["shares_initial"] == 10  # unchanged

        mock_rm_cls.return_value.size_position.assert_called_once()


class TestConvictionSwap:
    """2026-08-13: when sizing is blocked purely by a cash shortage (not a
    halt, not "already at target," not low conviction), _route_to_approval
    now checks for a swap candidate before giving up on a high-conviction
    idea. These tests guard the wiring in crew.py specifically — the
    swap-eligibility logic itself is tested in test_risk_manager.py's
    TestFindSwapCandidate."""

    def _cash_blocked_sizing(self):
        return {"shares_initial": 0, "flags": ["Allocation too small"], "cash_blocked": True,
                "position_size_pct": 0}

    def _funded_sizing(self):
        return {
            "shares_initial": 8, "initial_dollar": 800.0, "position_size_pct": 0.08,
            "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0,
        }

    def test_swap_found_resizes_and_proceeds(self):
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=9)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval", return_value=-1) as mock_log_approval, \
             patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
             patch("execution.execute_manual", return_value={"success": True}) as mock_sell, \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.size_position.side_effect = [
                self._cash_blocked_sizing(), self._funded_sizing(),
            ]
            mock_rm_cls.return_value.find_swap_candidate.return_value = {
                "ticker": "SNOW", "conviction": 4, "qty": 10.0,
                "market_value": 1000.0, "unrealized_pl": 150.0,
            }
            mock_rm_cls.return_value.check_concentration.return_value = {
                "passed": True, "flags": [], "ai_infrastructure_total": 0,
                "ai_infra_size_cap_pct": None,
            }
            mock_rm_cls.return_value.check_correlation.return_value = {"reduce_by": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_sell.assert_called_once()
        assert mock_sell.call_args[0][0] == "SNOW"
        assert mock_sell.call_args[0][1] == "SELL"
        mock_alert.assert_called_once()
        assert "SNOW" in mock_alert.call_args[0][0]
        assert "PLTR" in mock_alert.call_args[0][0]
        mock_log_approval.assert_called_once()
        assert mock_log_approval.call_args.kwargs["risk_json"]["shares_initial"] == 8

    def test_no_swap_candidate_falls_through_to_existing_behavior(self):
        """Nothing qualifies -- must behave exactly as before this feature
        existed: log and drop, no sell attempted."""
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=9)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval") as mock_log_approval, \
             patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
             patch("execution.execute_manual") as mock_sell:
            mock_rm_cls.return_value.size_position.return_value = self._cash_blocked_sizing()
            mock_rm_cls.return_value.find_swap_candidate.return_value = None
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_sell.assert_not_called()
        mock_log_approval.assert_not_called()

    def test_swap_lookup_never_attempted_for_non_cash_blocks(self):
        """0 shares for a reason OTHER than cash (e.g. already at target)
        must never trigger a swap lookup at all."""
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=9)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval") as mock_log_approval:
            mock_rm_cls.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["Position already at/near target"],
                "cash_blocked": False, "position_size_pct": 0,
            }
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_rm_cls.return_value.find_swap_candidate.assert_not_called()
        mock_log_approval.assert_not_called()

    def test_swap_sell_failure_falls_through_gracefully(self):
        """The sell didn't actually go through (e.g. broker error) -- must
        not pretend it did or crash; falls back to dropping the new idea,
        same as no candidate being found."""
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=9)
        client = MagicMock()

        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("db.log_pending_approval") as mock_log_approval, \
             patch("execution.execute_manual",
                   return_value={"success": False, "message": "no fill"}), \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.size_position.return_value = self._cash_blocked_sizing()
            mock_rm_cls.return_value.find_swap_candidate.return_value = {
                "ticker": "SNOW", "conviction": 4, "qty": 10.0,
                "market_value": 1000.0, "unrealized_pl": 150.0,
            }
            crew._route_to_approval(result, {"regime": "Neutral"}, client)

        mock_alert.assert_not_called()
        mock_log_approval.assert_not_called()
        # Only the initial sizing call -- must not re-size after a failed sell.
        mock_rm_cls.return_value.size_position.assert_called_once()
