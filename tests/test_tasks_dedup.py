"""
Tests for tasks.py's options-flow dispatch dedup and drawdown circuit
breaker plausibility guard — no live Celery/DB/IBKR required. Both cover
2026-07-15 findings: options "unusual" is persistent state that used to
re-dispatch to the full LLM pipeline every 5 minutes forever, and a
transient $0 portfolio-value read used to compute as a 100% drawdown and
latch trading_halted until a manual /resume.
"""
from unittest.mock import MagicMock, patch

import pytest

import tasks


class TestOptionsFlowDedup:
    def test_same_ticker_dispatches_once_per_day(self):
        """A persistently 'unusual' ticker (OI concentration doesn't reset
        every 5 minutes) must only be analyzed once per day, not on every
        poll — this was the single biggest LLM-cost bug found."""
        summaries = {
            "NVDA": {"unusual": True, "summary": "concentrated calls"},
        }
        seen_state = {}

        def fake_is_seen(source, key):
            return (source, key) in seen_state

        def fake_mark_seen(source, key):
            seen_state[(source, key)] = True

        with patch("market_data._is_market_hours", return_value=True), \
             patch.dict("os.environ", {"UNUSUAL_WHALES_API_KEY": "your_key_here"}), \
             patch("ingestion.options_free.get_all_flow_summaries", return_value=summaries), \
             patch("db.is_seen", side_effect=fake_is_seen), \
             patch("db.mark_seen", side_effect=fake_mark_seen), \
             patch("tasks._dispatch_event") as mock_dispatch, \
             patch("tasks._heartbeat_ok"):
            tasks.poll_options_flow.run()
            tasks.poll_options_flow.run()

        # Only the first call should have dispatched anything.
        assert mock_dispatch.call_count == 1

    def test_different_days_dispatch_separately(self):
        """The dedup key includes the date, so the same persistent signal
        legitimately re-analyzes once the next trading day starts."""
        summaries = {"NVDA": {"unusual": True, "summary": "concentrated calls"}}

        with patch("market_data._is_market_hours", return_value=True), \
             patch.dict("os.environ", {"UNUSUAL_WHALES_API_KEY": "your_key_here"}), \
             patch("ingestion.options_free.get_all_flow_summaries", return_value=summaries), \
             patch("db.is_seen", return_value=False), \
             patch("db.mark_seen"), \
             patch("tasks._dispatch_event") as mock_dispatch, \
             patch("tasks._heartbeat_ok"):
            tasks.poll_options_flow.run()

        mock_dispatch.assert_called_once()


class TestDrawdownGuard:
    def _run_with_pv(self, pv, hwm="100000"):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=pv)
        state = {"high_water_mark": hwm, "trading_halted": "false", "drawdown_breach_count": "0"}

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_client", return_value=client), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts.send_halt_alert") as mock_alert, \
             patch("config.START_VALUE", 100_000.0), \
             patch("config.MAX_DRAWDOWN_PCT", 0.15):
            tasks.check_portfolio_drawdown.run()
        return state, mock_alert

    def test_zero_portfolio_value_does_not_halt(self):
        """A transient pv=0 read (clientId collision, cold-morning
        connection race) must not compute as a 100% drawdown."""
        state, mock_alert = self._run_with_pv(0.0)
        assert state["trading_halted"] == "false"
        mock_alert.assert_not_called()

    def test_single_real_breach_does_not_halt_yet(self):
        """First breaching read waits for confirmation — only a sustained
        drawdown (2 consecutive breaching reads) should halt."""
        state, mock_alert = self._run_with_pv(80_000.0)  # 20% down, >15% limit
        assert state["trading_halted"] == "false"
        assert state["drawdown_breach_count"] == "1"
        mock_alert.assert_not_called()

    def test_second_consecutive_breach_halts(self):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=80_000.0)
        state = {"high_water_mark": "100000", "trading_halted": "false",
                 "drawdown_breach_count": "1"}  # already breached once

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_client", return_value=client), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts.send_halt_alert") as mock_alert, \
             patch("config.START_VALUE", 100_000.0), \
             patch("config.MAX_DRAWDOWN_PCT", 0.15):
            tasks.check_portfolio_drawdown.run()

        assert state["trading_halted"] == "true"
        mock_alert.assert_called_once()

    def test_healthy_portfolio_resets_breach_count(self):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=100_000.0)
        state = {"high_water_mark": "100000", "trading_halted": "false",
                 "drawdown_breach_count": "1"}

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_client", return_value=client), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts.send_halt_alert"), \
             patch("config.START_VALUE", 100_000.0), \
             patch("config.MAX_DRAWDOWN_PCT", 0.15):
            tasks.check_portfolio_drawdown.run()

        assert state["drawdown_breach_count"] == "0"


class TestReconciliationAlertThrottle:
    """2026-07-27: reconcile_stop_orders' orphaned-stop-cancelled message
    was deliberately left unthrottled on the assumption a successful
    cancel can't recur for the same order — a cross-clientId order whose
    cancel doesn't actually stick at the broker proved that wrong,
    firing the identical alert ~33x in one trading day for the same
    orphaned SNDK stop. Same fix applied to check_ingestion_health below,
    which had the identical unthrottled-standing-condition pattern."""

    def _run_reconcile(self, state, open_orders, cancel_returns=True):
        client = MagicMock()
        client.get_all_positions.return_value = []  # nothing held -> every stop is orphaned

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=open_orders), \
             patch("ibkr.cancel_order", return_value=cancel_returns), \
             patch("db.get_all_position_stops", return_value=[]), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_stop_orders.run()
        return mock_alert

    def test_repeat_orphan_cancel_for_same_order_does_not_realert(self):
        order = {"order_id": 999, "ticker": "SNDK", "action": "SELL", "order_type": "STP",
                 "qty": 12.0, "stop_price": 1332.93, "status": "PreSubmitted"}
        state = {}
        first  = self._run_reconcile(state, [order])
        second = self._run_reconcile(state, [order])  # same order still "orphaned" next cycle

        assert first.call_count == 1
        assert second.call_count == 0

    def test_different_orphaned_order_still_alerts(self):
        state = {}
        order_a = {"order_id": 1, "ticker": "AAA", "action": "SELL", "order_type": "STP",
                   "qty": 1.0, "stop_price": 10.0, "status": "PreSubmitted"}
        order_b = {"order_id": 2, "ticker": "BBB", "action": "SELL", "order_type": "STP",
                   "qty": 1.0, "stop_price": 20.0, "status": "PreSubmitted"}

        first  = self._run_reconcile(state, [order_a])
        second = self._run_reconcile(state, [order_b])

        assert first.call_count == 1
        assert second.call_count == 1  # a genuinely new/different order still alerts promptly

    def test_ingestion_health_same_stale_sources_do_not_realert(self):
        state = {}

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_stale_ingestion_sources", return_value=["news: never succeeded"]), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts._send") as mock_alert:
            tasks.check_ingestion_health.run()
            tasks.check_ingestion_health.run()

        assert mock_alert.call_count == 1


class TestRetryDeferredEntry:
    """2026-07-28: added alongside the move to a 5:00 AM ET wake. A new
    entry deferred by execution.execute_approved's market-hours gate gets
    exactly one retry scheduled for the next open, reusing
    execute_approved's own safety checks (price drift, drawdown halt)
    rather than re-running the 72B analysis."""

    def test_retries_failed_approval(self):
        approval = {"id": 1, "ticker": "NVDA", "status": "FAILED"}
        with patch("db.get_approval_by_id", return_value=approval), \
             patch("execution.execute_approved", return_value=True) as mock_execute, \
             patch("db.update_approval_status") as mock_update:
            tasks.retry_deferred_entry.run(1)

        mock_execute.assert_called_once_with(approval, trigger="auto_retry")
        mock_update.assert_called_once_with(1, "APPROVED")

    def test_skips_approval_not_in_failed_status(self):
        """Already handled some other way -- e.g. a fresher event
        re-triggered analysis and it's now APPROVED/PROCESSING -- must not
        double-execute."""
        approval = {"id": 2, "ticker": "AMD", "status": "APPROVED"}
        with patch("db.get_approval_by_id", return_value=approval), \
             patch("execution.execute_approved") as mock_execute:
            tasks.retry_deferred_entry.run(2)

        mock_execute.assert_not_called()

    def test_missing_approval_does_not_error(self):
        with patch("db.get_approval_by_id", return_value=None), \
             patch("execution.execute_approved") as mock_execute:
            tasks.retry_deferred_entry.run(999)  # should not raise

        mock_execute.assert_not_called()
