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

import db
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
    """Keys are trading_env-scoped in the real code (high_water_mark_paper
    vs high_water_mark_live, etc. — see tasks.check_portfolio_drawdown's
    2026-08-14 fix, same incident as risk_manager.is_halted's). Every test
    here pins config.TRADING_ENV to "paper" explicitly (tasks.py does a
    fresh `from config import TRADING_ENV` inside the function body, so
    patching config.TRADING_ENV takes effect on the next call) rather than
    relying on whatever this machine's real .env currently says."""

    def _run_with_pv(self, pv, hwm="100000"):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=pv)
        state = {"high_water_mark_paper": hwm, "trading_halted_paper": "false",
                 "drawdown_breach_count_paper": "0"}

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_client", return_value=client), \
             patch("config.TRADING_ENV", "paper"), \
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
        assert state["trading_halted_paper"] == "false"
        mock_alert.assert_not_called()

    def test_single_real_breach_does_not_halt_yet(self):
        """First breaching read waits for confirmation — only a sustained
        drawdown (2 consecutive breaching reads) should halt."""
        state, mock_alert = self._run_with_pv(80_000.0)  # 20% down, >15% limit
        assert state["trading_halted_paper"] == "false"
        assert state["drawdown_breach_count_paper"] == "1"
        mock_alert.assert_not_called()

    def test_second_consecutive_breach_halts(self):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=80_000.0)
        state = {"high_water_mark_paper": "100000", "trading_halted_paper": "false",
                 "drawdown_breach_count_paper": "1"}  # already breached once

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_client", return_value=client), \
             patch("config.TRADING_ENV", "paper"), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts.send_halt_alert") as mock_alert, \
             patch("config.START_VALUE", 100_000.0), \
             patch("config.MAX_DRAWDOWN_PCT", 0.15):
            tasks.check_portfolio_drawdown.run()

        assert state["trading_halted_paper"] == "true"
        mock_alert.assert_called_once()

    def test_healthy_portfolio_resets_breach_count(self):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=100_000.0)
        state = {"high_water_mark_paper": "100000", "trading_halted_paper": "false",
                 "drawdown_breach_count_paper": "1"}

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("tasks._get_client", return_value=client), \
             patch("config.TRADING_ENV", "paper"), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts.send_halt_alert"), \
             patch("config.START_VALUE", 100_000.0), \
             patch("config.MAX_DRAWDOWN_PCT", 0.15):
            tasks.check_portfolio_drawdown.run()

        assert state["drawdown_breach_count_paper"] == "0"


class TestReconciliationAlertThrottle:
    """2026-07-27: reconcile_stop_orders' orphaned-stop-cancelled message
    was deliberately left unthrottled on the assumption a successful
    cancel can't recur for the same order — a cross-clientId order whose
    cancel doesn't actually stick at the broker proved that wrong,
    firing the identical alert ~33x in one trading day for the same
    orphaned SNDK stop. Same fix applied to check_ingestion_health below,
    which had the identical unthrottled-standing-condition pattern.

    2026-07-28: this task no longer sends to Telegram at all (user
    request) — the per-issue throttle below still governs the WARNING-
    level log line, which is what these tests now check via caplog
    instead of an alerts._send mock.

    2026-07-30 (KLAC): the orphan-cancel branch now requires a ticker to
    look positionless on two consecutive cycles before actually
    cancelling anything (see reconcile_stop_orders' docstring) — a
    single missing-this-cycle reading just arms a marker and waits, since
    a real incident showed that single reading can be a momentarily-stale
    portfolio cache rather than a genuinely closed position. Tests below
    call _run_reconcile an extra time to reach the confirmed cycle before
    asserting on the cancel/alert."""

    def _run_reconcile(self, state, open_orders, caplog, cancel_returns=True):
        import logging
        client = MagicMock()
        client.get_all_positions.return_value = []  # nothing held -> every stop is orphaned

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="tasks"), \
             patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=open_orders), \
             patch("ibkr.cancel_order", return_value=cancel_returns), \
             patch("db.get_all_position_stops", return_value=[]), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state):
            tasks.reconcile_stop_orders.run()
        return sum(1 for r in caplog.records if "new/unthrottled" in r.message)

    def test_single_cycle_miss_does_not_cancel_orphaned_stop(self, caplog):
        """The KLAC bug this guards against: one cycle reading a position
        as missing must not be enough to cancel its stop -- only arm the
        pending-confirmation marker."""
        order = {"order_id": 999, "ticker": "SNDK", "action": "SELL", "order_type": "STP",
                 "qty": 12.0, "stop_price": 1332.93, "status": "PreSubmitted"}
        state = {}
        armed_count = self._run_reconcile(state, [order], caplog)

        assert armed_count == 0
        assert state.get("orphan_pending_SNDK")  # marker set, nothing cancelled yet

    def test_repeat_orphan_cancel_for_same_order_does_not_realert(self, caplog):
        order = {"order_id": 999, "ticker": "SNDK", "action": "SELL", "order_type": "STP",
                 "qty": 12.0, "stop_price": 1332.93, "status": "PreSubmitted"}
        state = {}
        self._run_reconcile(state, [order], caplog)                # cycle 1: arms the marker
        first_count  = self._run_reconcile(state, [order], caplog)  # cycle 2: confirmed -> cancels + alerts
        second_count = self._run_reconcile(state, [order], caplog)  # cycle 3: marker cleared post-cancel -> re-arms, no alert

        assert first_count == 1
        assert second_count == 0

    def test_different_orphaned_order_still_alerts(self, caplog):
        state = {}
        order_a = {"order_id": 1, "ticker": "AAA", "action": "SELL", "order_type": "STP",
                   "qty": 1.0, "stop_price": 10.0, "status": "PreSubmitted"}
        order_b = {"order_id": 2, "ticker": "BBB", "action": "SELL", "order_type": "STP",
                   "qty": 1.0, "stop_price": 20.0, "status": "PreSubmitted"}

        self._run_reconcile(state, [order_a], caplog)                # arm A
        first_count  = self._run_reconcile(state, [order_a], caplog)  # confirm + cancel A
        self._run_reconcile(state, [order_b], caplog)                # arm B (independent marker)
        second_count = self._run_reconcile(state, [order_b], caplog)  # confirm + cancel B

        assert first_count == 1
        assert second_count == 1  # a genuinely new/different order still logs promptly, once confirmed

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

        mock_execute.assert_called_once_with(
            approval, trigger="auto_retry", gap_retry_attempt=1, price_retry_attempt=1,
        )
        mock_update.assert_called_once_with(1, "APPROVED", reason=None)

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


class TestBackstopStaleStopCache:
    """2026-07-28 (AMKR): a cached broker stop-order can go stale in
    exactly one dangerous direction -- IBKR only pushes fill/status
    updates for an order to the clientId that placed it, so a different
    long-lived connection (the ops worker, here) can keep reporting an
    already-filled stop as still open indefinitely. A plain ticker-
    membership check ("is AMKR in the set of tickers with an open stop")
    kept concluding the position was covered for hours after the real
    stop had filled and reduced it from 186 to 110 shares, so the
    backstop-sell never fired while the remainder sat unprotected.
    check_trailing_stops now cross-checks the stop's cached quantity
    against the actual held quantity instead of trusting presence alone."""

    def _run(self, held_qty, stop_orders, triggered_action="EXIT"):
        client = MagicMock()
        client.get_open_position.return_value = MagicMock(qty=held_qty)

        with patch("tasks._get_client", return_value=client), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.get_open_orders", return_value=stop_orders), \
             patch("tasks._execute_backstop_sell", return_value=True) as mock_backstop, \
             patch("alerts.send_stop_triggered"), \
             patch("config.AUTO_EXECUTE_PROTECTIVE_SELLS", True):
            mock_rm_cls.return_value.check_trailing_stops.return_value = [{
                "ticker": "AMKR", "action": triggered_action,
                "current_price": 46.0, "stop_price": 55.71,
                "reason": "Stop triggered at $55.71 (current $46.00)",
            }]
            tasks.check_trailing_stops.run()
        return mock_backstop

    def test_stale_cached_stop_for_more_shares_than_held_still_backstops(self):
        """The exact AMKR shape: a cached stop still shows 186 shares open,
        but only 110 are actually held -- the cache is stale, not real
        coverage, so the backstop must still fire."""
        stop_orders = [{"ticker": "AMKR", "action": "SELL", "order_type": "STP",
                         "qty": 186.0, "stop_price": 55.71, "order_id": 70,
                         "status": "PreSubmitted"}]
        mock_backstop = self._run(held_qty=110.0, stop_orders=stop_orders)
        mock_backstop.assert_called_once()

    def test_stop_matching_current_position_is_trusted(self):
        """A stop genuinely covering the current position (quantities
        match) is real protection -- must not double-sell on top of it."""
        stop_orders = [{"ticker": "AMKR", "action": "SELL", "order_type": "STP",
                         "qty": 110.0, "stop_price": 34.66, "order_id": 6,
                         "status": "PreSubmitted"}]
        mock_backstop = self._run(held_qty=110.0, stop_orders=stop_orders)
        mock_backstop.assert_not_called()

    def test_no_stop_at_all_backstops(self):
        mock_backstop = self._run(held_qty=110.0, stop_orders=[])
        mock_backstop.assert_called_once()


class TestReconcileUnconfirmedOrders:
    """2026-07-28 (CRWD): a market order can fill *after*
    ibkr.place_market_order's fill-wait window gives up and the order gets
    logged status='unconfirmed'. Nothing ever went back to check whether it
    filled anyway -- a real 144-share CRWD position sat live at the broker
    all day with no stop and no trade record, invisible to every other
    safety system since they all read the DB, not the broker directly.
    reconcile_unconfirmed_orders closes that gap by checking IBKR's own
    execution record for each 'unconfirmed' order."""

    def _order(self, order_id=44, ticker="CRWD", side="BUY", qty=144.0,
               broker_order_id=764, approval_id=59):
        return {"id": order_id, "ticker": ticker, "side": side, "qty": qty,
                "broker_order_id": broker_order_id, "decision_price": 180.1,
                "approval_id": approval_id}

    def _approval(self, status="FAILED"):
        return {
            "id": 59, "ticker": "CRWD", "status": status,
            "thesis_bucket": "Compounder", "conviction": 7,
            "stop_loss_price": 153.61,
            "risk_json": {"stop_method": "2.5x ATR", "trailing_stop_distance_pct": 0.175},
        }

    def test_late_fill_is_recorded_and_stopped(self):
        """The exact CRWD shape: order was unconfirmed, but IBKR's
        execution record shows it actually filled -- must be logged as a
        real trade, protected with the stop already computed at sizing
        time, and the approval flipped off FAILED."""
        with patch("db.get_unconfirmed_orders", return_value=[self._order()]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
             patch("db.log_trade") as mock_log_trade, \
             patch("db.update_order_fill") as mock_update_fill, \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("ibkr.place_stop_order",
                   return_value={"order_id": 900, "stop_price": 153.61}) as mock_stop, \
             patch("db.upsert_position_stop") as mock_upsert_stop, \
             patch("db.set_stop_broker_order_id") as mock_set_stop_id, \
             patch("db.update_approval_status") as mock_update_approval, \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_unconfirmed_orders.run()

        mock_log_trade.assert_called_once()
        assert mock_log_trade.call_args.kwargs["quantity"] == 144.0
        assert mock_log_trade.call_args.kwargs["price"] == 181.3
        mock_update_fill.assert_called_once_with(44, fill_price=181.3, filled_qty=144.0, status="filled")
        mock_stop.assert_called_once_with("CRWD", 144.0, 153.61)
        mock_upsert_stop.assert_called_once()
        mock_set_stop_id.assert_called_once_with("CRWD", 900)
        mock_update_approval.assert_called_once_with(59, "APPROVED")
        mock_alert.assert_called_once()

    def test_still_unfilled_order_is_left_alone(self):
        """No execution found at the broker -- genuinely never filled (or
        was cancelled/rejected). Must not fabricate anything."""
        with patch("db.get_unconfirmed_orders", return_value=[self._order()]), \
             patch("ibkr.get_fill_for_order", return_value=None), \
             patch("db.log_trade") as mock_log_trade, \
             patch("db.update_approval_status") as mock_update_approval, \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_unconfirmed_orders.run()

        mock_log_trade.assert_not_called()
        mock_update_approval.assert_not_called()
        mock_alert.assert_not_called()

    def test_no_pending_orders_is_a_noop(self):
        with patch("db.get_unconfirmed_orders", return_value=[]), \
             patch("ibkr.get_fill_for_order") as mock_lookup:
            tasks.reconcile_unconfirmed_orders.run()

        mock_lookup.assert_not_called()

    def test_late_fill_without_approval_still_records_trade_but_skips_stop(self):
        """Manual/non-approval orders (no approval_id) have no computed
        stop price to use -- must still record the real fill, just can't
        auto-place a stop blind, and the alert must say so."""
        order = self._order(approval_id=None)
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
             patch("db.log_trade") as mock_log_trade, \
             patch("db.update_order_fill"), \
             patch("ibkr.place_stop_order") as mock_stop, \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_unconfirmed_orders.run()

        mock_log_trade.assert_called_once()
        mock_stop.assert_not_called()
        assert "check TWS" in mock_alert.call_args[0][0]

    def _run_sell(self, positions, stop_rec={"stop_price": 150.0}, **order_extra):
        from unittest.mock import MagicMock
        order = {**self._order(side="SELL", qty=50.0), **order_extra}
        client = MagicMock()
        client.get_all_positions.return_value = positions
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 50.0, "fill_price": 175.0}), \
             patch("db.log_trade"), \
             patch("db.update_order_fill"), \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("ibkr.get_client", return_value=client), \
             patch("db.get_position_stop", return_value=stop_rec), \
             patch("db.set_stop_broker_order_id"), \
             patch("tasks.manage_cancel_stop_orders_for") as cancel, \
             patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
             patch("db.update_approval_status"), \
             patch("db.delete_position_stop") as delete, \
             patch("db.close_trade_outcome") as close, \
             patch("db.mark_profit_target_taken") as taken, \
             patch("alerts._send") as send:
            tasks.reconcile_unconfirmed_orders.run()
        self.extra = {"delete": delete, "close": close, "taken": taken}
        return cancel, place, send

    def test_late_sell_that_closed_the_position_places_no_stop(self):
        cancel, place, _ = self._run_sell([])
        place.assert_not_called()

    def test_lookup_passes_ticker_side_and_placement_margin(self):
        from datetime import datetime, timezone, timedelta
        created = datetime(2026, 10, 28, 14, 0, tzinfo=timezone.utc)
        with patch("db.get_unconfirmed_orders",
                   return_value=[{**self._order(), "created_at": created}]), \
             patch("ibkr.get_fill_for_order", return_value=None) as lookup:
            tasks.reconcile_unconfirmed_orders.run()
        lookup.assert_called_once_with(764, ticker="CRWD", side="BUY",
                                       placed_at=created - timedelta(minutes=5))

    def test_partial_late_fill_is_marked_partial(self):
        """Professor M-4: a fill short of the order used to be marked filled."""
        with patch("db.get_unconfirmed_orders", return_value=[self._order()]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 100.0, "fill_price": 181.0}), \
             patch("db.log_trade") as log_trade, \
             patch("db.update_order_fill") as update_fill, \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.update_approval_status"), patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        update_fill.assert_called_once_with(44, fill_price=181.0, filled_qty=100.0, status="partial")
        assert log_trade.call_args.kwargs["quantity"] == 100.0
        place.assert_called_once_with("CRWD", 100.0, 153.61)

    def test_rest_of_partial_order_logs_only_new_shares(self):
        order = {**self._order(), "filled_qty": 100.0, "fill_price": 181.0}
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 182.0}), \
             patch("db.log_trade") as log_trade, \
             patch("db.update_order_fill") as update_fill, \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.update_approval_status"), patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        kw = log_trade.call_args.kwargs
        assert kw["quantity"] == 44.0
        assert abs(kw["price"] - (144 * 182.0 - 100 * 181.0) / 44) < 1e-9
        update_fill.assert_called_once_with(44, fill_price=182.0, filled_qty=144.0, status="filled")
        place.assert_called_once_with("CRWD", 44.0, 153.61)

    def test_no_new_shares_on_partial_order_does_nothing(self):
        order = {**self._order(), "filled_qty": 100.0, "fill_price": 181.0}
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 100.0, "fill_price": 181.0}), \
             patch("db.log_trade") as log_trade, patch("alerts._send") as send:
            tasks.reconcile_unconfirmed_orders.run()
        log_trade.assert_not_called()
        send.assert_not_called()

    def test_late_exit_closes_stop_record_and_outcome(self):
        """Professor M-3: a late full exit left position_stops behind, so
        stale-stop cleanup logged the SELL a second time."""
        self._run_sell([], intent="EXIT")
        self.extra["delete"].assert_called_once_with("CRWD")
        self.extra["close"].assert_called_once_with("CRWD", exit_price=175.0)

    def test_late_exit_already_cleaned_up_is_not_closed_twice(self):
        self._run_sell([], stop_rec=None, intent="EXIT")
        self.extra["delete"].assert_not_called()
        self.extra["close"].assert_not_called()

    def test_late_profit_target_sale_marks_taken_and_uses_its_stop(self):
        """Professor M-3: the remainder went back on the pre-sale stop and the
        target stayed open for a second sale."""
        from unittest.mock import MagicMock
        _, place, _ = self._run_sell([MagicMock(symbol="CRWD", qty=20)],
                                     intent="PROFIT_TARGET", stop_price=181.0)
        self.extra["taken"].assert_called_once_with("CRWD", 181.0)
        place.assert_called_once_with("CRWD", 20.0, 181.0)

    def test_late_trim_keeps_recorded_stop_and_target_open(self):
        from unittest.mock import MagicMock
        _, place, _ = self._run_sell([MagicMock(symbol="CRWD", qty=20)], intent="TRIM")
        self.extra["taken"].assert_not_called()
        place.assert_called_once_with("CRWD", 20.0, 150.0)

    def test_late_partial_sell_reprotects_the_remainder(self):
        """Professor C12 (2026-10-28): the stop is cancelled before a partial
        sale, so a late fill used to leave the rest with no stop."""
        from unittest.mock import MagicMock
        cancel, place, send = self._run_sell([MagicMock(symbol=self._order()["ticker"], qty=20)])
        cancel.assert_called_once()
        place.assert_called_once_with(self._order()["ticker"], 20.0, 150.0)
        assert "Protective stop placed." in send.call_args.args[0]

    def test_late_partial_sell_without_a_stop_record_warns(self):
        from unittest.mock import MagicMock
        _, place, send = self._run_sell([MagicMock(symbol=self._order()["ticker"], qty=20)],
                                        stop_rec=None)
        place.assert_not_called()
        assert "check TWS" in send.call_args.args[0]


class TestBrokerVisibilityTimeout:
    """Regression guard for the 2026-07-30 incident: tasks.retry_deferred_entry
    schedules etas up to several days out (a Friday deferral retries Monday's
    open), but Celery's Redis transport treats a not-yet-fired eta task as
    "unacked" the whole time it waits -- with the 1-hour default
    visibility_timeout, it got redelivered roughly hourly overnight, so by
    market open several duplicate copies of the same approval fired back to
    back (5+ duplicate "ORDER ABORTED" alerts for one ticker, and a real race
    risk of two copies executing concurrently). Must stay set well past the
    longest eta this app schedules."""

    def test_visibility_timeout_covers_a_weekend_deferral(self):
        timeout = tasks.app.conf.broker_transport_options.get("visibility_timeout")
        assert timeout is not None
        # A Friday-evening deferral to Monday's open is ~65h out -- require
        # comfortable margin beyond that (here, at least 4 days).
        assert timeout >= 4 * 24 * 3600


class TestSystemStateKeyLength:
    """Regression guard for the 2026-07-30 incident: tasks._reconcile_alert_due
    builds throttle keys like "reconcile_alert_ingestion_<all stale source
    names joined>" -- with several sources stale at once this routinely
    exceeded the old VARCHAR(50) limit on system_state.key, so db.set_state
    silently failed every time (caught and logged, not raised) and the
    throttle could never persist. check_ingestion_health's alert then re-fired
    on every hourly check instead of respecting its cooldown."""

    def test_schema_defines_a_wide_enough_key_column(self):
        assert "key         VARCHAR(200) PRIMARY KEY" in db._CREATE_TABLES

    def test_migration_27_widens_the_column(self):
        versions = dict(db._MIGRATIONS)
        assert 27 in versions
        assert "system_state" in versions[27]
        assert "VARCHAR(200)" in versions[27]

    def test_a_realistic_long_throttle_key_fits_in_200_chars(self):
        """The actual key shape that broke at 50 chars -- many stale
        ingestion sources joined together -- must comfortably fit now."""
        stale_sources = ["analyst_ratings", "congressional", "edgar", "insider",
                          "macro", "news", "options_flow", "sp500_history"]
        issue_key = "ingestion_" + "_".join(sorted(stale_sources))
        state_key = f"reconcile_alert_{issue_key}"
        assert len(state_key) <= 200


class TestNoStopTelegramEscalation:
    """2026-07-30 (AMD): the "no protective stop" case was silent (log
    only) since 2026-07-28, alongside the routine orphan-cancel noise it
    was actually meant to quiet. A real 50-share naked AMD position fired
    this check every 15 minutes for ~30 hours with nobody seeing it.
    Escalates to Telegram once a ticker has been continuously unstopped
    for 45+ minutes (not on first sighting -- a TRIM/replace can leave a
    position briefly unstopped for one ordinary cycle), throttled after
    that on the normal RECONCILE_ALERT_THROTTLE_HOURS cadence."""

    def _run(self, state, held_qty=50.0):
        client = MagicMock()
        client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=held_qty)]

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("db.get_all_position_stops", return_value=[]), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_stop_orders.run()
        return mock_alert

    def test_first_sighting_does_not_alert_telegram(self):
        state = {}
        mock_alert = self._run(state)

        mock_alert.assert_not_called()
        assert state.get("no_stop_since_AMD")  # marker armed for next cycle

    def test_sustained_45_minutes_escalates_to_telegram(self):
        from datetime import datetime, timedelta, timezone
        state = {"no_stop_since_AMD": (datetime.now(timezone.utc) - timedelta(minutes=50)).isoformat()}

        mock_alert = self._run(state)

        mock_alert.assert_called_once()
        assert "AMD" in mock_alert.call_args[0][0]

    def test_escalation_is_throttled_on_repeat_cycles(self):
        from datetime import datetime, timedelta, timezone
        state = {"no_stop_since_AMD": (datetime.now(timezone.utc) - timedelta(minutes=50)).isoformat()}
        self._run(state)  # first escalation fires and sets its own throttle key

        mock_alert = self._run(state)  # still unstopped 15 min later, throttle window not elapsed

        mock_alert.assert_not_called()


class TestStaleStopReconciliation:
    """2026-08-06 (DDOG): a position disappeared from the broker (its own
    tracked stop order never filled -- it sat orphaned/uncancellable, a
    different clientId owned it) with no bot-initiated sell anywhere in
    the logs. The stale-DB-stop cleanup used to just delete the
    position_stops record and log it as routine noise -- no trade or
    trade_outcome row was ever written, so the exit and whatever was
    gained or lost on it vanished with nothing pointing back at it. Now
    it searches today's broker execution history by symbol before
    deleting: a match gets written as a real trade and closes the
    trade_outcome; no match escalates to Telegram immediately instead of
    disappearing silently.

    2026-08-17 (QBTS, live's first trade): this branch had no protection
    against a momentarily-stale positions cache -- the same shape as the
    2026-07-30 KLAC incident that gave branch 1 (orphan-cancel, above) its
    two-consecutive-cycles requirement. A fresh fill's position hadn't
    synced into client.get_all_positions() yet when this branch happened
    to run, so it declared the position "disappeared" and deleted the
    (accurate) DB stop record on the very first read. Now requires the
    same two-consecutive-cycles confirmation as branch 1 -- tests below
    call .run() an extra time to reach the confirmed cycle before
    asserting, same pattern as TestReconciliationAlertThrottle above."""

    def _run(self, state, fill_return, alert_due=True):
        client = MagicMock()
        client.get_all_positions.return_value = []  # DDOG no longer held

        stale_rec = {"ticker": "DDOG", "entry_price": 270.79, "stop_price": 228.74,
                     "broker_order_id": 11}

        def fake_get_state(key, default=None):
            return state.get(key, default)

        def fake_set_state(key, value):
            state[key] = value

        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("db.get_all_position_stops", return_value=[stale_rec]), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("ibkr.get_recent_sell_fill", return_value=fill_return) as mock_fill, \
             patch("db.log_trade") as mock_log_trade, \
             patch("db.close_trade_outcome") as mock_close_outcome, \
             patch("db.delete_position_stop") as mock_delete, \
             patch("tasks._reconcile_alert_due", return_value=alert_due), \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_stop_orders.run()
        return mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert

    def test_single_cycle_miss_does_not_reconcile_yet(self):
        """A single missing-this-cycle reading must only arm the pending
        marker, not touch trade/outcome/stop records at all -- this is
        exactly what QBTS needed and didn't have."""
        state = {}
        mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert = \
            self._run(state, fill_return=None)

        mock_fill.assert_not_called()
        mock_log_trade.assert_not_called()
        mock_close_outcome.assert_not_called()
        mock_delete.assert_not_called()
        mock_alert.assert_not_called()
        assert state.get("stale_stop_pending_DDOG")

    def test_found_fill_is_logged_as_a_trade_and_closes_the_outcome(self):
        state = {}
        fill = {"filled_qty": 84.0, "fill_price": 227.80}
        self._run(state, fill_return=fill)  # cycle 1: arms the marker
        mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert = \
            self._run(state, fill_return=fill)  # cycle 2: confirmed -> reconciles

        mock_fill.assert_called_once_with("DDOG")
        mock_log_trade.assert_called_once()
        assert mock_log_trade.call_args.kwargs["symbol"] == "DDOG"
        assert mock_log_trade.call_args.kwargs["side"] == "SELL"
        assert mock_log_trade.call_args.kwargs["quantity"] == 84.0
        assert mock_log_trade.call_args.kwargs["price"] == 227.80
        mock_close_outcome.assert_called_once_with("DDOG", exit_price=227.80)
        mock_delete.assert_called_once_with("DDOG")
        # A reconciled exit is real information, not an emergency -- no Telegram ping.
        mock_alert.assert_not_called()

    def test_no_fill_found_does_not_fabricate_a_trade_and_alerts_instead(self):
        state = {}
        self._run(state, fill_return=None)  # cycle 1: arms the marker
        mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert = \
            self._run(state, fill_return=None)  # cycle 2: confirmed -> reconciles

        mock_log_trade.assert_not_called()
        mock_close_outcome.assert_not_called()
        # The stale record is still cleared (it genuinely doesn't match a
        # held position) but the disappearance itself must not be silent.
        mock_delete.assert_called_once_with("DDOG")
        mock_alert.assert_called_once()
        assert "DDOG" in mock_alert.call_args[0][0]

    def test_no_fill_found_respects_alert_throttle(self):
        state = {}
        self._run(state, fill_return=None, alert_due=False)  # cycle 1: arms
        _, _, _, _, mock_alert = self._run(state, fill_return=None, alert_due=False)  # cycle 2

        mock_alert.assert_not_called()


class TestBackstopSellRetry:
    """2026-07-29 (AMD): a backstop sell that got no fill (submitted
    after-hours -- IBKR queues rather than fills within the 30s wait) used
    to just log an error and give up. The exact position this function
    exists to protect sat naked for ~30 hours because nothing ever
    retried. A zero-fill result now schedules retry_backstop_sell instead
    of dropping the problem."""

    def _alert(self):
        return {"ticker": "AMD", "action": "EXIT", "current_price": 500.0,
                "stop_price": 425.02, "reason": "Stop triggered at $425.02 (current $500.00)"}

    def test_zero_fill_schedules_retry(self):
        alert = self._alert()
        with patch("ibkr.get_client") as mock_get_client, \
             patch("ibkr.cancel_stop_orders_for"), \
             patch("ibkr.place_market_order", return_value={"filled_qty": 0, "order_id": 1122}), \
             patch("market_data.is_market_hours", return_value=False), \
             patch("market_data.next_market_open") as mock_next_open, \
             patch("tasks.retry_backstop_sell") as mock_retry_task:
            mock_get_client.return_value.get_open_position.return_value = MagicMock(qty=50.0)
            result = tasks._execute_backstop_sell("AMD", alert)

        assert result is False
        mock_retry_task.apply_async.assert_called_once()
        _, kwargs = mock_retry_task.apply_async.call_args
        assert kwargs["args"] == ["AMD", alert]
        assert kwargs["eta"] == mock_next_open.return_value

    def test_successful_fill_does_not_schedule_retry(self):
        alert = self._alert()
        with patch("ibkr.get_client") as mock_get_client, \
             patch("ibkr.cancel_stop_orders_for"), \
             patch("ibkr.place_market_order",
                   return_value={"filled_qty": 50.0, "fill_price": 424.0, "order_id": 1122}), \
             patch("db.log_trade"), patch("db.delete_position_stop"), patch("db.close_trade_outcome"), \
             patch("tasks.retry_backstop_sell") as mock_retry_task:
            mock_get_client.return_value.get_open_position.return_value = MagicMock(qty=50.0)
            result = tasks._execute_backstop_sell("AMD", alert)

        assert result is True
        mock_retry_task.apply_async.assert_not_called()


class TestCheckProfitTargetsTask:
    """2026-08-29, user request: the Celery wrapper around
    risk_manager.check_profit_targets() -- this is the task that actually
    makes the profit-target feature place live SELL orders on a schedule
    (see beat_schedule's check-profit-targets entry) rather than the
    feature being present-but-dormant code. The order placement itself
    happens inside risk_manager (unlike check_trailing_stops, there's no
    detect-vs-execute split here), so this task's own job is just
    invocation and alerting -- these tests guard exactly that."""

    def test_sold_positions_get_a_trade_alert(self):
        client = MagicMock()
        with patch("tasks._get_client", return_value=client), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("alerts.send_trade_alert") as mock_alert:
            mock_rm_cls.return_value.check_profit_targets.return_value = [{
                "ticker": "SNOW", "action": "PARTIAL_SELL", "shares_sold": 25.0,
                "new_stop": 100.0, "trade_style": "Trending/Expanding", "gain_pct": 0.30,
                "order_result": {"order_id": 1},
            }]
            tasks.check_profit_targets.run()
        mock_alert.assert_called_once()
        _, kwargs = mock_alert.call_args
        assert kwargs["symbol"] == "SNOW"
        assert kwargs["side"] == "SELL"
        assert kwargs["qty"] == 25

    def test_nothing_sold_sends_no_alert(self):
        client = MagicMock()
        with patch("tasks._get_client", return_value=client), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("alerts.send_trade_alert") as mock_alert:
            mock_rm_cls.return_value.check_profit_targets.return_value = []
            tasks.check_profit_targets.run()
        mock_alert.assert_not_called()

    def test_risk_manager_failure_retries_not_crashes(self):
        with patch("tasks._get_client", side_effect=Exception("IBKR down")):
            with pytest.raises(Exception):
                tasks.check_profit_targets.run()


class TestReviewPositionsStalePriority:
    """2026-08-29, user request: review_positions now asks
    flag_stale_positions() first and passes the result through as
    priority_tickers -- these tests guard that the wiring actually
    happens, and that a flagging failure doesn't block the review itself
    from running unprioritized."""

    def test_stale_tickers_are_passed_through_as_priority(self):
        client = MagicMock()
        with patch("tasks._get_client", return_value=client), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("agents.crew.review_all_held_positions") as mock_review:
            mock_rm_cls.return_value.flag_stale_positions.return_value = [
                {"ticker": "SNOW", "days_held": 20, "pct_move": 0.01},
            ]
            tasks.review_positions.run()
        mock_review.assert_called_once_with(max_per_run=10, priority_tickers=["SNOW"])

    def test_flagging_failure_falls_through_to_unprioritized_review(self):
        with patch("tasks._get_client", side_effect=Exception("IBKR down")), \
             patch("agents.crew.review_all_held_positions") as mock_review:
            tasks.review_positions.run()
        mock_review.assert_called_once_with(max_per_run=10, priority_tickers=[])


class TestBackfillCatalystTrackingTask:
    """2026-08-30, user request: thin Celery wrapper around
    agents.crew.backfill_catalyst_tracking -- on-demand only, not in
    beat_schedule (see the task's own docstring for why)."""

    def test_delegates_to_crew_function_and_returns_results(self):
        with patch("agents.crew.backfill_catalyst_tracking",
                    return_value=[{"ticker": "SNOW", "action": "BACKFILLED"},
                                  {"ticker": "AMD", "action": "SKIPPED"}]) as mock_backfill:
            result = tasks.backfill_catalyst_tracking.run(max_per_run=5)
        mock_backfill.assert_called_once_with(max_per_run=5)
        assert result == [{"ticker": "SNOW", "action": "BACKFILLED"},
                           {"ticker": "AMD", "action": "SKIPPED"}]

    def test_failure_retries_not_crashes(self):
        with patch("agents.crew.backfill_catalyst_tracking", side_effect=Exception("db down")):
            with pytest.raises(Exception):
                tasks.backfill_catalyst_tracking.run()
