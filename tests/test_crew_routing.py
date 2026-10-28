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


@pytest.fixture(autouse=True)
def _no_earnings_nearby():
    """_route_to_approval checks the earnings calendar before OPEN/ADD
    (2026-10-27 earnings blackout). Default: nothing reporting soon."""
    with patch("db.is_near_earnings", return_value=False), \
         patch("db.get_next_earnings", return_value=None):
        yield


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
             patch("db.update_approval_status"), \
             patch("execution.execute_approved", return_value=True) as mock_execute, \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("config.AUTO_EXECUTE", True):
            # AUTO_EXECUTE explicitly forced True -- this test asserts the
            # auto-execute path actually completes, which must not depend
            # on the ambient .env's TRADING_ENV (e.g. a machine configured
            # for live trading would otherwise silently route this through
            # the approval-request branch instead and mock_execute would
            # never be called, unrelated to what this test is checking).
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

    def test_no_swap_while_market_closed(self):
        result = _analyst_result(action="ADD", ticker="PLTR", conviction=9)
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
             patch("execution.execute_manual") as mock_sell, \
             patch("market_data.is_market_hours", return_value=False), \
             patch("alerts._send"):
            mock_rm_cls.return_value.size_position.return_value = self._cash_blocked_sizing()
            mock_rm_cls.return_value.find_swap_candidate.return_value = {
                "ticker": "SNOW", "conviction": 4, "qty": 10.0,
                "market_value": 1000.0, "unrealized_pl": 150.0,
            }
            crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
        mock_sell.assert_not_called()

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
             patch("market_data.is_market_hours", return_value=True), \
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


class TestReviewAllHeldPositionsPriority:
    """2026-08-29, user request: priority_tickers puts stale-flagged
    positions at the FRONT of the review rotation regardless of when
    they were last reviewed -- fed by risk_manager.flag_stale_positions()
    via tasks.review_positions. These tests guard that ordering directly,
    against review_all_held_positions itself rather than the task
    wrapper (see TestReviewPositionsStalePriority in
    test_tasks_dedup.py for the wiring-through test)."""

    def _client_with(self, tickers):
        client = MagicMock()
        client.get_all_positions.return_value = [
            MagicMock(symbol=t, qty=10.0, avg_entry_price=100.0,
                      current_price=100.0, market_value=1000.0)
            for t in tickers
        ]
        return client

    def test_priority_ticker_reviewed_first_despite_recent_review(self):
        # SNOW was reviewed most recently (should normally be LAST in the
        # rotation) but is flagged stale -- must still go first.
        client = self._client_with(["SNOW", "NVDA", "AMD"])
        last_reviewed = {
            "AMD":  "2026-08-01T00:00:00+00:00",
            "NVDA": "2026-08-15T00:00:00+00:00",
            "SNOW": "2026-08-28T00:00:00+00:00",  # most recent
        }
        reviewed_order = []

        def fake_review(ticker, *a, **kw):
            reviewed_order.append(ticker)
            return {"ticker": ticker}

        with patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("db.get_state", return_value=__import__("json").dumps(last_reviewed)), \
             patch("db.set_state"), \
             patch("agents.crew.review_held_position", side_effect=fake_review):
            crew.review_all_held_positions(max_per_run=10, priority_tickers=["SNOW"])

        assert reviewed_order[0] == "SNOW"
        assert reviewed_order[1:] == ["AMD", "NVDA"]  # rest stay oldest-first

    def test_no_priority_tickers_behaves_exactly_as_before(self):
        client = self._client_with(["AMD", "NVDA"])
        last_reviewed = {"NVDA": "2026-08-01T00:00:00+00:00", "AMD": "2026-08-15T00:00:00+00:00"}
        reviewed_order = []

        def fake_review(ticker, *a, **kw):
            reviewed_order.append(ticker)
            return {"ticker": ticker}

        with patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("db.get_state", return_value=__import__("json").dumps(last_reviewed)), \
             patch("db.set_state"), \
             patch("agents.crew.review_held_position", side_effect=fake_review):
            crew.review_all_held_positions(max_per_run=10)

        assert reviewed_order == ["NVDA", "AMD"]


class TestBackfillCatalystTracking:
    """2026-08-30, user request: one-time gap-closer for positions held
    before catalyst/expected-return tracking existed. These tests guard:
    only missing-catalyst tickers get analyzed (respecting the DB-reported
    filter, not re-deriving it), the approval-routing side effect never
    fires even on a pass_to_risk result (a backfill isn't a new trade
    signal), and a failed analysis is reported as skipped rather than
    silently dropped or crashing the run."""

    def _client_with(self, tickers):
        client = MagicMock()
        client.get_all_positions.return_value = [
            MagicMock(symbol=t, qty=10.0, avg_entry_price=100.0,
                      current_price=105.0, market_value=1050.0)
            for t in tickers
        ]
        return client

    def test_only_missing_tickers_are_analyzed(self):
        client = self._client_with(["AMD", "NVDA", "SNOW"])

        def fake_analyze(ticker, *a, **kw):
            return {"ticker": ticker, "action": "HOLD", "pass_to_risk": False,
                    "catalyst": {"event": f"{ticker} catalyst", "date": "2026-11", "is_priced_in": False},
                    "expected_return_pct": 20.0, "expected_downside_pct": -10.0,
                    "thesis_summary": "test", "sources": [{"type": "news"}], "source_count": 1}

        with patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("db.get_tickers_missing_catalyst_data", return_value=["SNOW"]) as mock_missing, \
             patch("db.get_position_stop", return_value=None), \
             patch("db.backfill_catalyst_outcome") as mock_backfill, \
             patch("agents.crew._analyze_held_position", side_effect=fake_analyze), \
             patch("agents.crew._route_to_approval") as mock_route:
            results = crew.backfill_catalyst_tracking(max_per_run=10)

        mock_missing.assert_called_once()
        assert set(mock_missing.call_args[0][0]) == {"AMD", "NVDA", "SNOW"}
        mock_backfill.assert_called_once()
        assert mock_backfill.call_args.kwargs["ticker"] == "SNOW"
        assert mock_backfill.call_args.kwargs["catalyst_event"] == "SNOW catalyst"
        mock_route.assert_not_called()  # backfill must never trigger a new approval
        assert results == [{"ticker": "SNOW", "action": "BACKFILLED",
                             "catalyst_event": "SNOW catalyst", "source_count": 1}]

    def test_pass_to_risk_result_never_routes_to_approval(self):
        # Even if the fresh analysis WOULD normally trigger a TRIM/EXIT,
        # a backfill pass must not act on it -- it exists to capture a
        # snapshot, not to fire trading actions as a side effect.
        client = self._client_with(["AMD"])

        def fake_analyze(ticker, *a, **kw):
            return {"ticker": ticker, "action": "EXIT", "pass_to_risk": True, "conviction": 8,
                    "catalyst": {"event": "thesis broken", "date": "ongoing", "is_priced_in": True},
                    "expected_return_pct": -30.0, "expected_downside_pct": -30.0,
                    "thesis_summary": "test", "sources": [{"type": "news"}, {"type": "edgar_filing"}],
                    "source_count": 2}

        with patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("db.get_tickers_missing_catalyst_data", return_value=["AMD"]), \
             patch("db.get_position_stop", return_value=None), \
             patch("db.backfill_catalyst_outcome"), \
             patch("agents.crew._analyze_held_position", side_effect=fake_analyze), \
             patch("agents.crew._route_to_approval") as mock_route:
            crew.backfill_catalyst_tracking(max_per_run=10)

        mock_route.assert_not_called()

    def test_analysis_failure_is_reported_as_skipped_not_crashed(self):
        client = self._client_with(["AMD"])

        def fake_analyze(ticker, *a, **kw):
            return {"action": "SKIP", "ticker": ticker, "conviction": 0,
                    "pass_to_risk": False, "reasoning": "Review failed with exception: boom"}

        with patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("db.get_tickers_missing_catalyst_data", return_value=["AMD"]), \
             patch("db.backfill_catalyst_outcome") as mock_backfill, \
             patch("agents.crew._analyze_held_position", side_effect=fake_analyze):
            results = crew.backfill_catalyst_tracking(max_per_run=10)

        mock_backfill.assert_not_called()
        assert results == [{"ticker": "AMD", "action": "SKIPPED",
                             "reason": "Review failed with exception: boom"}]

    def test_uses_position_stop_entry_data_when_available(self):
        from datetime import datetime, timezone
        client = self._client_with(["AMD"])
        opened = datetime(2026, 1, 15, tzinfo=timezone.utc)

        def fake_analyze(ticker, *a, **kw):
            return {"ticker": ticker, "action": "HOLD", "pass_to_risk": False,
                    "catalyst": {}, "expected_return_pct": None, "expected_downside_pct": None,
                    "thesis_summary": None, "sources": [], "source_count": 0}

        with patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("db.get_tickers_missing_catalyst_data", return_value=["AMD"]), \
             patch("db.get_position_stop", return_value={"entry_price": 88.0, "opened_at": opened,
                                                            "thesis_bucket": "Moonshot", "conviction": 7}), \
             patch("db.backfill_catalyst_outcome") as mock_backfill, \
             patch("agents.crew._analyze_held_position", side_effect=fake_analyze):
            crew.backfill_catalyst_tracking(max_per_run=10)

        kwargs = mock_backfill.call_args.kwargs
        assert kwargs["entry_price"] == 88.0
        assert kwargs["entry_date"] == opened
        assert kwargs["bucket"] == "Moonshot"
        assert kwargs["conviction"] == 7


def test_stub_result_is_not_captured_as_a_thesis():
    """Audit 2026-10-27 finding 8: model-outage stubs must not land in
    latest_thesis / thesis_history as if they were real verdicts."""
    from agents import crew
    with patch("db.update_latest_thesis") as update:
        crew._capture_latest_thesis("NVDA", {"model_used": "stub", "action": "WATCH", "conviction": 1})
    update.assert_not_called()


def test_open_is_blocked_inside_the_earnings_blackout():
    result = _analyst_result(action="OPEN", ticker="MSFT", conviction=8)
    with patch("db.is_near_earnings", return_value=True), \
         patch("db.get_next_earnings", return_value="2026-10-28"), \
         patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("db.log_signal") as log_signal, \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("risk_manager.RiskManager") as mock_rm_cls:
        crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
    mock_rm_cls.return_value.size_position.assert_not_called()
    assert log_signal.call_args[0][1] == "EARNINGS_BLOCKED"


def test_open_is_blocked_when_the_earnings_calendar_cannot_be_read():
    result = _analyst_result(action="OPEN", ticker="MSFT", conviction=8)
    with patch("db.is_near_earnings", return_value=None), \
         patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("db.log_signal") as log_signal, \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("risk_manager.RiskManager") as mock_rm_cls:
        crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
    mock_rm_cls.return_value.size_position.assert_not_called()
    assert "unreadable" in log_signal.call_args[0][3]
