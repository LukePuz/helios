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
             patch("tasks._PROCESS_START", 0), \
             patch("db.get_state", side_effect=fake_get_state), \
             patch("db.set_state", side_effect=fake_set_state), \
             patch("alerts._send") as mock_alert:
            tasks.check_ingestion_health.run()
            tasks.check_ingestion_health.run()

        assert mock_alert.call_count == 1


def test_ingestion_health_skipped_right_after_a_start():
    """A0453-7: every poller reads stale after the overnight stop."""
    import time
    with patch("tasks._PROCESS_START", time.time()), \
         patch("tasks._get_stale_ingestion_sources") as stale, \
         patch("alerts._send") as send:
        tasks.check_ingestion_health.run()
    stale.assert_not_called(); send.assert_not_called()


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

    def test_a_stop_sized_net_of_a_working_sale_is_trusted(self):
        """Professor M132: 110 held, a market SELL working on 40, our stop on
        the other 70 (M129) is full cover, not a stale stop."""
        orders = [{"ticker": "AMKR", "action": "SELL", "order_type": "STP", "qty": 70.0,
                   "stop_price": 34.66, "order_id": 6, "status": "PreSubmitted"},
                  {"ticker": "AMKR", "action": "SELL", "order_type": "MKT", "qty": 40.0,
                   "stop_price": None, "order_id": 9, "status": "Submitted", "remaining": 40.0}]
        self._run(held_qty=110.0, stop_orders=orders).assert_not_called()

    def test_a_cancelled_stop_is_not_cover(self):
        orders = [{"ticker": "AMKR", "action": "SELL", "order_type": "STP", "qty": 110.0,
                   "stop_price": 34.66, "order_id": 6, "status": "Cancelled"}]
        self._run(held_qty=110.0, stop_orders=orders).assert_called_once()

    @staticmethod
    def _mkt(qty, status="Submitted", ticker="AMKR"):
        return {"ticker": ticker, "action": "SELL", "order_type": "MKT", "qty": qty, "stop_price": None,
                "order_id": 9, "status": status, "remaining": qty}

    @staticmethod
    def _stp(qty):
        return {"ticker": "AMKR", "action": "SELL", "order_type": "STP", "qty": qty,
                "stop_price": 34.66, "order_id": 6, "status": "PreSubmitted"}

    def test_a_full_stop_plus_a_working_sale_is_over_cover_and_backstops(self):
        """Both would fill and short; the backstop nets the working sale."""
        self._run(held_qty=110.0, stop_orders=[self._stp(110.0), self._mkt(40.0)]).assert_called_once()

    def test_a_working_sale_for_the_whole_position_is_cover(self):
        self._run(held_qty=110.0, stop_orders=[self._mkt(110.0)]).assert_not_called()

    def test_a_dead_market_sell_is_not_cover(self):
        for status in ("Cancelled", "Filled"):
            self._run(held_qty=110.0, stop_orders=[self._stp(70.0), self._mkt(40.0, status)]).assert_called_once()

    def test_another_tickers_sale_is_not_cover(self):
        self._run(held_qty=110.0,
                  stop_orders=[self._stp(70.0), self._mkt(40.0, ticker="NVDA")]).assert_called_once()

    def test_a_part_filled_stop_counts_its_remainder(self):
        """186-share stop with 76 filled leaves 110 working: full cover."""
        orders = [{"ticker": "AMKR", "action": "SELL", "order_type": "STP", "qty": 186.0,
                   "stop_price": 34.66, "order_id": 6, "status": "Submitted", "remaining": 110.0}]
        self._run(held_qty=110.0, stop_orders=orders).assert_not_called()


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

    @staticmethod
    def _client(held):
        from unittest.mock import MagicMock
        c = MagicMock()
        c.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=held)]
        return c

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
             patch("tasks.manage_stop_place",
                   return_value={"order_id": 900, "stop_price": 153.61}) as mock_stop, \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.get_client", return_value=self._client(144)), \
             patch("db.get_position_stop", side_effect=[None, {"stop_price": 153.61}]), \
             patch("tasks._retry_pending_reprotects"), \
             patch("db.upsert_position_stop") as mock_upsert_stop, \
             patch("db.set_stop_broker_order_id") as mock_set_stop_id, \
             patch("db.update_approval_status") as mock_update_approval, \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_unconfirmed_orders.run()

        mock_log_trade.assert_called_once()
        assert mock_log_trade.call_args.kwargs["quantity"] == 144.0
        assert mock_log_trade.call_args.kwargs["price"] == 181.3
        mock_update_fill.assert_called_once_with(44, fill_price=181.3, filled_qty=144.0, status="filled", expect_filled_qty=0.0, filled_at=None)
        mock_stop.assert_called_once_with("CRWD", 144.0, 153.61)
        mock_upsert_stop.assert_called_once()
        mock_set_stop_id.assert_called_once_with("CRWD", 900)
        mock_update_approval.assert_called_once_with(59, "APPROVED")
        mock_alert.assert_called_once()

    def test_late_buy_gets_an_outcome_row_only_when_nothing_was_recorded_at_the_fill(self):
        """Professor M123: a BUY that filled after the wait window had no
        trade_outcomes row. A part fill already logged one at the fill."""
        for prev, expect in ((None, True), (40.0, False)):
            order = {**self._order(), "filled_qty": prev, "fill_price": 180.0 if prev else None}
            with patch("db.get_unconfirmed_orders", return_value=[order]), \
                 patch("ibkr.get_fill_for_order",
                       return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
                 patch("db.log_trade"), patch("db.update_order_fill"), \
                 patch("db.get_approval_by_id", return_value={**self._approval(), "action": "OPEN"}), \
                 patch("execution.log_buy_outcome") as outcome, \
                 patch("db.update_outcome_entry_price") as reavg, \
                 patch("db.get_position_stop", return_value={"stop_price": 160.0}), \
                 patch("tasks._held_before_fill", return_value=True), \
                 patch("tasks._reprotect_after_late_sell", return_value=True), \
                 patch("tasks._retry_pending_reprotects"), \
                 patch("db.update_approval_status"), patch("alerts._send"):
                tasks.reconcile_unconfirmed_orders.run()
            assert outcome.called is expect
            if expect:
                assert outcome.call_args.args[1:] == ("CRWD", "OPEN", 181.3)
            else:
                reavg.assert_called_once_with(59, 181.3)   # entry side of M121

    def test_late_buy_keeps_existing_stop_record_and_queues_on_failure(self):
        """A2253-3: an ADD's late fill must not overwrite (loosen) the
        recorded stop, and a failed placement is retried."""
        with patch("db.get_unconfirmed_orders", return_value=[self._order()]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
             patch("db.log_trade"), patch("db.update_order_fill"), \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("db.get_position_stop", return_value={"stop_price": 160.0}), \
             patch("db.upsert_position_stop") as upsert, \
             patch("tasks._reprotect_after_late_sell", return_value=False), \
             patch("tasks._queue_reprotect") as queued, \
             patch("tasks._retry_pending_reprotects"), \
             patch("db.update_approval_status"), patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        upsert.assert_not_called()
        queued.assert_called_once()

    def test_late_buy_replaces_a_closed_positions_leftover_record(self):
        """A0053-5: nothing held before this fill, so an existing record is a
        closed position's leftover and must not set the new stop."""
        with patch("db.get_unconfirmed_orders", return_value=[self._order()]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
             patch("db.log_trade"), patch("db.update_order_fill"), \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("ibkr.get_client", return_value=self._client(144)), \
             patch("db.get_position_stop", return_value={"stop_price": 200.0}), \
             patch("db.upsert_position_stop") as upsert, \
             patch("tasks._reprotect_after_late_sell", return_value=True), \
             patch("tasks._retry_pending_reprotects"), \
             patch("db.update_approval_status"), patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        upsert.assert_called_once()
        assert upsert.call_args.kwargs["stop_price"] == 153.61

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

    def _run_manual_buy(self, stop_rec, intent="MANUAL_OPEN", quote=None):
        order = {**self._order(approval_id=None), "intent": intent}
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
             patch("db.log_trade"), patch("db.update_order_fill"), \
             patch("ibkr.get_client", return_value=self._client(144)), \
             patch("db.get_position_stop", return_value=stop_rec), \
             patch("db.upsert_position_stop") as upsert, \
             patch("tasks._reprotect_after_late_sell", return_value=True) as reprotect, \
             patch("market_data.get_priced_quote", return_value=quote and {"price": quote}), \
             patch("tasks._retry_pending_reprotects"), patch("alerts._send") as alert:
            tasks.reconcile_unconfirmed_orders.run()
        return upsert, reprotect, alert

    def test_late_manual_add_keeps_the_stop_despite_a_stale_read(self):
        """Review of 9582ede: the read shows only the new shares, but the
        order was sent onto a held position: its stop must not be replaced."""
        upsert, reprotect, _ = self._run_manual_buy({"stop_price": 170.0}, intent="MANUAL_ADD")
        upsert.assert_not_called()
        reprotect.assert_called_once()

    def test_late_manual_buy_stop_stays_under_the_quote(self):
        upsert, _, _ = self._run_manual_buy(None, quote=150.0)     # fell >8% since the fill
        assert upsert.call_args.kwargs["stop_price"] == 138.0

    def test_late_manual_buy_gets_a_fallback_stop(self):
        """Professor M108: a late manual BUY fill is protected 8% below the fill."""
        upsert, reprotect, alert = self._run_manual_buy(None)
        assert upsert.call_args.kwargs["stop_price"] == round(181.3 * 0.92, 2)
        reprotect.assert_called_once()
        assert "Protective stop placed" in alert.call_args[0][0]

    def test_late_manual_buy_closed_positions_leftover_record_is_replaced(self):
        upsert, _, _ = self._run_manual_buy({"stop_price": 200.0})   # 144 held = all from this fill
        assert upsert.call_args.kwargs["stop_price"] == round(181.3 * 0.92, 2)

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
             patch("tasks._queue_reprotect") as queued, patch("time.sleep"), \
             patch("tasks._retry_pending_reprotects"), \
             patch("alerts._send") as send:
            tasks.reconcile_unconfirmed_orders.run()
        self.extra = {"delete": delete, "close": close, "taken": taken, "queued": queued}
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
             patch("tasks.manage_cancel_stop_orders_for"), patch("tasks._retry_pending_reprotects"), \
             patch("db.get_position_stop", return_value={"stop_price": 153.61}), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.get_client", return_value=self._client(100)), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.update_approval_status"), patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        update_fill.assert_called_once_with(44, fill_price=181.0, filled_qty=100.0, status="partial", expect_filled_qty=0.0, filled_at=None)
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
             patch("tasks.manage_cancel_stop_orders_for"), patch("tasks._retry_pending_reprotects"), \
             patch("db.get_position_stop", return_value={"stop_price": 153.61}), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.get_client", return_value=self._client(144)), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.update_approval_status"), patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        kw = log_trade.call_args.kwargs
        assert kw["quantity"] == 44.0
        assert abs(kw["price"] - (144 * 182.0 - 100 * 181.0) / 44) < 1e-9
        update_fill.assert_called_once_with(44, fill_price=182.0, filled_qty=144.0, status="filled", expect_filled_qty=100.0, filled_at=None)
        place.assert_called_once_with("CRWD", 144.0, 153.61)  # whole position (A2253-3)

    def test_no_new_shares_on_partial_order_does_nothing(self):
        order = {**self._order(), "filled_qty": 100.0, "fill_price": 181.0}
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 100.0, "fill_price": 181.0}), \
             patch("db.log_trade") as log_trade, patch("alerts._send") as send:
            tasks.reconcile_unconfirmed_orders.run()
        log_trade.assert_not_called()
        send.assert_not_called()

    def test_failed_reprotect_is_queued_for_retry(self):
        """Professor M50: a busy/failed re-protect after the SELL was booked
        used to be dropped."""
        from unittest.mock import MagicMock
        self._run_sell([MagicMock(symbol="CRWD", qty=20)], stop_rec=None)
        self.extra["queued"].assert_called_once()

    def test_order_not_marked_means_no_trade_logged(self):
        """Professor M51: log_trade ran before update_order_fill, so a failed
        update re-logged the trade every cycle."""
        with patch("db.get_unconfirmed_orders", return_value=[self._order()]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
             patch("db.update_order_fill", return_value=False), \
             patch("db.log_trade") as log_trade, \
             patch("tasks._retry_pending_reprotects"), \
             patch("tasks.manage_stop_place") as place, patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()
        log_trade.assert_not_called()
        place.assert_not_called()

    def test_late_exit_closes_stop_record_and_outcome(self):
        """Professor M-3: a late full exit left position_stops behind, so
        stale-stop cleanup logged the SELL a second time."""
        self._run_sell([], intent="EXIT")
        self.extra["delete"].assert_called_once_with("CRWD")
        self.extra["close"].assert_called_once_with("CRWD", exit_price=175.0, exit_source="late_exit",
                                                         exit_price_source="fill", exit_fill_at=None)

    def test_late_exit_without_a_stop_record_closes_from_the_cumulative_fill(self):
        """Professor M121: a part-filled EXIT deleted the record and left the
        outcome open; the late remainder closes it (a row already closed or
        orphaned is skipped inside close_trade_outcome)."""
        self._run_sell([], stop_rec=None, intent="EXIT")
        self.extra["delete"].assert_not_called()
        self.extra["close"].assert_called_once()

    def test_late_exit_remainder_closes_at_the_cumulative_average_not_the_remainder_price(self):
        """Professor M121: 30 filled @ 170 inside the window, 50 @ 175 in total;
        the remainder alone averaged 182.5."""
        self._run_sell([], stop_rec=None, intent="EXIT", filled_qty=30.0, fill_price=170.0)
        assert self.extra["close"].call_args.kwargs["exit_price"] == 175.0

    def test_late_backstop_remainder_closes_without_a_stop_record(self):
        self._run_sell([], stop_rec=None, intent="BACKSTOP")
        assert self.extra["close"].call_args.kwargs["exit_source"] == "late_backstop"

    def test_late_trim_already_cleaned_up_is_not_closed(self):
        self._run_sell([], stop_rec=None, intent="TRIM")
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

    def _run(self, state, held_qty=50.0, record=None, place=None):
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
             patch("db.get_position_stop", return_value=record), \
             patch("db.set_stop_broker_order_id"), \
             patch("ibkr.get_latest_price", return_value=200.0), \
             patch("tasks.manage_stop_place", place or MagicMock(side_effect=AssertionError)), \
             patch("alerts._send") as mock_alert:
            tasks.reconcile_stop_orders.run()
        return mock_alert

    def test_second_naked_cycle_places_the_recorded_stop(self):
        """A0053-2: confirmed naked with a recorded stop -> stop placed."""
        from datetime import datetime, timedelta, timezone
        state = {"no_stop_since_AMD": (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(),
                 "no_stop_seen_AMD": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()}
        place = MagicMock(return_value={"order_id": 9})
        mock_alert = self._run(state, record={"stop_price": 150.0}, place=place)
        place.assert_called_once_with("AMD", 50.0, 150.0)
        assert "UNPROTECTED" not in mock_alert.call_args[0][0]

    def test_first_sighting_does_not_alert_telegram(self):
        state = {}
        mock_alert = self._run(state)

        mock_alert.assert_not_called()
        assert state.get("no_stop_since_AMD")  # marker armed for next cycle

    def test_stale_marker_from_a_closed_position_restarts_the_count(self):
        """A0253-8: a marker not refreshed within two cycles is not 'the
        previous cycle'."""
        from datetime import datetime, timedelta, timezone
        state = {"no_stop_since_AMD": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
        mock_alert = self._run(state, record={"stop_price": 150.0})  # place would raise
        mock_alert.assert_not_called()
        assert datetime.fromisoformat(state["no_stop_since_AMD"]) > datetime.now(timezone.utc) - timedelta(minutes=1)

    def test_sustained_45_minutes_escalates_to_telegram(self):
        from datetime import datetime, timedelta, timezone
        state = {"no_stop_since_AMD": (datetime.now(timezone.utc) - timedelta(minutes=50)).isoformat(),
                 "no_stop_seen_AMD": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()}

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

    def _run(self, state, fill_return, alert_due=True, unconfirmed=(), recorded=(frozenset({40}), frozenset())):
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
             patch("ibkr.get_recent_sell_fill",
                   **({"side_effect": fill_return} if isinstance(fill_return, Exception)
                      else {"return_value": fill_return})) as mock_fill, \
             patch("db.recorded_sell_order_ids", return_value=recorded), \
             patch("db.get_unconfirmed_orders",
                   **({"side_effect": unconfirmed} if isinstance(unconfirmed, Exception)
                      else {"return_value": list(unconfirmed)})), \
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

    def test_a_failed_exit_lookup_keeps_the_record_and_retries(self):
        """Review of 73494d1: a timed-out execution read is unknown, not "no
        SELL"; it must not label the outcome no_fill or delete the record."""
        state = {}
        self._run(state, fill_return=None)                                  # arms the marker
        with patch("db.flag_orphan_outcomes_for") as flag:
            mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert = \
                self._run(state, fill_return=TimeoutError("no reply"))
        mock_fill.assert_called_once()
        mock_log_trade.assert_not_called(); mock_close_outcome.assert_not_called()
        mock_delete.assert_not_called(); mock_alert.assert_not_called(); flag.assert_not_called()
        assert state.get("stale_stop_pending_DDOG")                          # still armed
        fill = {"filled_qty": 84.0, "fill_price": 227.80}
        _, mock_log_trade, _, _, _ = self._run(state, fill_return=fill)       # next cycle finds it
        mock_log_trade.assert_called_once()

    def test_found_fill_is_logged_as_a_trade_and_closes_the_outcome(self):
        state = {}
        fill = {"filled_qty": 84.0, "fill_price": 227.80}
        self._run(state, fill_return=fill)  # cycle 1: arms the marker
        mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert = \
            self._run(state, fill_return=fill)  # cycle 2: confirmed -> reconciles

        mock_fill.assert_called_once_with("DDOG", {40}, set())   # recorded orders left out (M122)
        mock_log_trade.assert_called_once()
        assert mock_log_trade.call_args.kwargs["symbol"] == "DDOG"
        assert mock_log_trade.call_args.kwargs["side"] == "SELL"
        assert mock_log_trade.call_args.kwargs["quantity"] == 84.0
        assert mock_log_trade.call_args.kwargs["price"] == 227.80
        mock_close_outcome.assert_called_once_with("DDOG", exit_price=227.80, exit_source="broker_exit",
                                                   exit_price_source="fill", exit_fill_at=None)
        mock_delete.assert_called_once_with("DDOG")
        # A reconciled exit is real information, not an emergency -- no Telegram ping.
        mock_alert.assert_not_called()

    def test_waits_while_one_of_our_sells_is_still_unconfirmed(self):
        """Review of cc72fe3: an unconfirmed EXIT keeps its stop record;
        this branch must leave the exit to the late-fill reconcile."""
        from datetime import datetime, timedelta, timezone
        state = {}
        pending = [{"ticker": "DDOG", "side": "SELL", "created_at": datetime.now(timezone.utc)}]
        self._run(state, fill_return=None, unconfirmed=pending)            # arms the marker
        mock_fill, mock_log_trade, _, mock_delete, _ = self._run(state, fill_return=None, unconfirmed=pending)
        mock_fill.assert_not_called()
        mock_log_trade.assert_not_called()
        mock_delete.assert_not_called()
        # An order from an earlier day doesn't hold the cleanup back: the
        # execution lookup only reaches today (review of 5e79eef).
        old = [{**pending[0], "created_at": datetime.now(timezone.utc) - timedelta(days=2)}]
        mock_fill, _, _, _, _ = self._run(state, fill_return=None, unconfirmed=old)
        mock_fill.assert_called_once()

    def test_waits_when_the_recorded_sells_are_unknown(self):
        """Professor, review of M122: a DB error must not fall back to
        blending already-logged sells into the exit."""
        state = {}
        self._run(state, fill_return=None, recorded=None)                  # arms the marker
        mock_fill, mock_log_trade, _, mock_delete, _ = self._run(state, fill_return=None, recorded=None)
        mock_fill.assert_not_called()
        mock_delete.assert_not_called()
        assert state.get("stale_stop_pending_DDOG")                         # marker kept for the retry

    def test_waits_when_the_unconfirmed_orders_are_unknown(self):
        """Professor M126: a DB error read as "none pending" closed a live
        exit as no_fill."""
        state = {}
        err = RuntimeError("pool exhausted")
        self._run(state, fill_return=None, unconfirmed=err)                # arms the marker
        mock_fill, mock_log_trade, mock_close, mock_delete, _ = self._run(state, fill_return=None, unconfirmed=err)
        mock_fill.assert_not_called()
        mock_close.assert_not_called()
        mock_delete.assert_not_called()
        assert state.get("stale_stop_pending_DDOG")

    def test_no_fill_found_does_not_fabricate_a_trade_and_alerts_instead(self):
        state = {}
        self._run(state, fill_return=None)  # cycle 1: arms the marker
        with patch("db.flag_orphan_outcomes_for") as flag:
            mock_fill, mock_log_trade, mock_close_outcome, mock_delete, mock_alert = \
                self._run(state, fill_return=None)  # cycle 2: confirmed -> reconciles
        flag.assert_called_once_with("DDOG", "no_fill")   # professor M120

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
            mock_get_client.return_value.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=50.0)]
            with patch("ibkr.get_open_orders", return_value=[]):
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
            mock_get_client.return_value.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=50.0)]
            with patch("ibkr.get_open_orders", return_value=[]):
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


def test_pending_reprotects_retry_until_done():
    import json
    state = {"reprotect_pending": json.dumps({"AMD": {"intent": "TRIM", "stop_price": None,
                                                     "fill_price": 100.0},
                                             "NVDA": {"intent": None, "stop_price": None,
                                                      "fill_price": 50.0}})}
    results = {"AMD": True, "NVDA": False}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reprotect_after_late_sell", side_effect=lambda t, *a, **k: results[t]):
        tasks._retry_pending_reprotects()
    assert set(json.loads(state["reprotect_pending"])) == {"NVDA"}


def test_entry_queued_during_a_retry_pass_survives():
    """A0653-2: the LLM worker queues a stop while reconcile retries others;
    the write-back must not drop it."""
    import json
    state = {"reprotect_pending": json.dumps({"AMD": {"intent": "TRIM", "stop_price": None,
                                                     "fill_price": 100.0}})}

    def retry(t, *a, **k):
        tasks._queue_reprotect("NVDA", {"intent": "BUY"}, None)  # concurrent writer
        return True
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reprotect_after_late_sell", side_effect=retry):
        tasks._retry_pending_reprotects()
    assert set(json.loads(state["reprotect_pending"])) == {"NVDA"}


def test_queued_reprotect_finding_flat_leaves_close_to_reconcile():
    """A2253-2: a queued retry must not close the outcome at its old price."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.delete_position_stop") as delete, \
         patch("db.close_trade_outcome") as close:
        assert tasks._reprotect_after_late_sell("AMD", {}, 100.0, from_queue=True) is None
    delete.assert_not_called(); close.assert_not_called()


def test_reconcile_skips_while_a_previous_run_holds_the_lock():
    """A2253-5: two overlapping runs would book the same fill."""
    from conftest import _FakeLock
    _FakeLock.held.add("helios:task:reconcile_unconfirmed_orders")
    with patch("db.get_unconfirmed_orders") as load:
        tasks.reconcile_unconfirmed_orders.run()
    load.assert_not_called()


def test_reconcile_skips_when_redis_is_unavailable():
    """A0053-6: no run lock, no run (it used to continue unguarded)."""
    import stop_lock
    with patch.object(stop_lock, "_redis", side_effect=ConnectionError("down")), \
         patch("db.get_unconfirmed_orders") as load:
        tasks.reconcile_unconfirmed_orders.run()
    load.assert_not_called()


def test_conditional_fill_update_reports_a_lost_race():
    from unittest.mock import MagicMock
    import db
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (180.0, "BUY")
    cur.rowcount = 0
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.update_order_fill(44, 181.0, 144.0, "filled", expect_filled_qty=0.0, filled_at=None) is False
    assert "AND status IN ('unconfirmed', 'partial')" in cur.execute.call_args[0][0]


def test_held_before_fill_follows_the_approval_action():
    """A0253-4: the action decides, not a lagging broker cache."""
    with patch("ibkr.get_client", side_effect=AssertionError("must not read")):
        assert tasks._held_before_fill("AMD", 1, "ADD") is True
        assert tasks._held_before_fill("AMD", 1, "OPEN") is False


def test_held_before_fill_reads_flat_as_not_held():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client):
        assert tasks._held_before_fill("AMD", 1) is False


def test_stop_reconcile_skips_while_a_previous_run_holds_the_lock():
    """Professor M71: overlapping stop reconciles could double-book an exit."""
    from conftest import _FakeLock
    _FakeLock.held.add("helios:task:reconcile_stop_orders")
    with patch("ibkr.get_client") as client:
        tasks.reconcile_stop_orders.run()
    client.assert_not_called()


def test_ingestion_staleness_is_calendar_aware():
    """A0653-6: options_flow outside RTH and Monday daily jobs aren't stale."""
    from datetime import datetime, timedelta
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    class _DT(datetime):
        fixed = None

        @classmethod
        def now(cls, tz=None):
            return cls.fixed.astimezone(tz) if tz else cls.fixed

    def stale_at(fixed):
        _DT.fixed = fixed
        rows = [{"source": "options_flow", "last_success_at": fixed - timedelta(hours=14)},
                {"source": "price_history", "last_success_at": fixed - timedelta(hours=63)}]
        with patch("db.get_ingestion_health", return_value=rows), \
             patch("datetime.datetime", _DT):
            return " ".join(tasks._get_stale_ingestion_sources())

    ny = ZoneInfo("America/New_York")
    mon = stale_at(datetime(2026, 11, 2, 6, 0, tzinfo=ny))    # Monday pre-market
    assert "options_flow" not in mon and "price_history" not in mon
    tue = stale_at(datetime(2026, 11, 3, 11, 0, tzinfo=ny))   # Tuesday in RTH
    assert "options_flow" in tue and "price_history" in tue


def test_identical_requeue_during_retry_pass_survives():
    """m36: a re-queue with the same fields while the retry pass works on the
    ticker must not be popped as the entry the pass saw."""
    import json
    import tasks
    state = {}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
        tasks._queue_reprotect("AMD", {"intent": "RESTORE", "stop_price": 90.0}, 100.0)

        def work(ticker, rec, fill, from_queue=False):
            tasks._queue_reprotect("AMD", {"intent": "RESTORE", "stop_price": 90.0}, 100.0)
            return True
        with patch("tasks._reprotect_after_late_sell", side_effect=work):
            tasks._retry_pending_reprotects()
    assert "AMD" in json.loads(state["reprotect_pending"])


class _Pos:
    def __init__(self, symbol, qty):
        self.symbol, self.qty = symbol, qty


def _queue_harness(place_effect):
    """Real queue + retry + _reprotect_after_late_sell; broker and DB faked."""
    import json
    import tasks
    state = {}
    placed = []

    def place(ticker, qty, price):
        if place_effect is not None:
            raise place_effect
        placed.append((ticker, qty, price))
        return {"order_id": 7}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("ibkr.get_client") as client, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_stop_place", side_effect=place):
        client.return_value.get_all_positions.return_value = [_Pos("AMD", 5.0)]
        tasks._queue_reprotect("AMD", {"intent": "BUY"}, None)
        tasks._retry_pending_reprotects()
    return json.loads(state["reprotect_pending"]), placed


def test_queue_retry_places_the_recorded_stop_end_to_end():
    """C23: a queued stop is placed on the next pass for the held shares at the
    recorded price, and leaves the queue."""
    pending, placed = _queue_harness(None)
    assert placed == [("AMD", 5.0, 90.0)]
    assert pending == {}


def test_queue_retry_keeps_the_entry_on_each_broker_failure():
    """C23 per-caller matrix, retry path: Unknown, Expired, LockLost and a
    plain error all keep the entry for the next cycle, with tries counted."""
    import stop_lock
    import tasks
    for err in (tasks.OrdersCallUnknown("x"), tasks.OrdersCallExpired("x"),
                stop_lock.LockLost("AMD"), RuntimeError("boom")):
        pending, placed = _queue_harness(err)
        assert placed == [] and pending["AMD"]["tries"] == 1, err


def test_edgar_dispatches_each_filing_as_found_and_skips_overlap():
    """A1053-4: filings go out during the pass, not after it; a second run
    while one holds the lock does nothing; a failed dispatch doesn't stop it."""
    import tasks
    from ingestion import edgar
    from conftest import _FakeLock
    order = []
    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1", "BBB": "2"}), \
         patch.object(edgar, "FILING_FORMS", ["8-K"]), \
         patch.object(edgar.time, "sleep"), \
         patch.object(edgar, "_get_recent_filings",
                      side_effect=lambda t, c, f: (order.append(("fetch", t)) or [{"ticker": t}])), \
         patch("tasks._dispatch_event",
               side_effect=lambda f, **k: (order.append(("dispatch", f["ticker"])),
                                           f["ticker"] == "AAA" and (_ for _ in ()).throw(RuntimeError("x")))), \
         patch("tasks._heartbeat_ok") as hb:
        tasks.poll_edgar.run()
        assert order == [("fetch", "AAA"), ("dispatch", "AAA"), ("fetch", "BBB"), ("dispatch", "BBB")]
        hb.assert_called_once_with("edgar", 2)
        _FakeLock.held.add("helios:task:poll_edgar")
        order.clear()
        tasks.poll_edgar.run()
        assert order == []


def test_covered_position_is_not_cancelled_by_the_queue():
    """M79: a stop already covering the shares (e.g. placed by hand) is left
    alone; the entry completes."""
    import tasks
    with patch("execution._stops_still_resting", return_value=5.0), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        with patch("db.get_state", return_value=None), patch("db.set_state"), \
             patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
             patch("ibkr.get_client") as client, \
             patch("tasks.manage_stop_place") as place:
            client.return_value.get_all_positions.return_value = [_Pos("AMD", 5.0)]
            assert tasks._reprotect_after_late_sell("AMD", {"intent": "RESTORE"}, None,
                                                    from_queue=True) is True
    cancel.assert_not_called(); place.assert_not_called()


def test_queue_gives_up_after_the_try_cap():
    """M79: retries are capped; the last failure alerts and drops the entry."""
    import json
    import tasks
    state = {"reprotect_pending": json.dumps(
        {"AMD": {"intent": "RESTORE", "tries": tasks._REPROTECT_MAX_TRIES - 1, "nonce": "n"}})}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reprotect_after_late_sell", return_value=False), \
         patch("alerts._send") as send:
        tasks._retry_pending_reprotects()
    assert json.loads(state["reprotect_pending"]) == {}
    assert "STOP NOT RESTORED" in send.call_args[0][0]


def test_restore_does_not_overwrite_a_queued_profit_target():
    """M80: the profit target's raised stop survives a later RESTORE request."""
    import json
    import tasks
    state = {}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        tasks._queue_reprotect("AMD", {"intent": "PROFIT_TARGET", "stop_price": 101.0}, 120.0)
        tasks._queue_reprotect("AMD", {"intent": "RESTORE"}, None)
    rec = json.loads(state["reprotect_pending"])["AMD"]
    assert rec["intent"] == "PROFIT_TARGET" and rec["stop_price"] == 101.0


def test_flat_buy_hands_off_after_about_30_minutes():
    """m55: a queued BUY stop whose position still reads flat alerts and drops
    after 2 cycles, not 6 (~90 min)."""
    import json
    import tasks
    state = {"reprotect_pending": json.dumps({"AMD": {"intent": "BUY", "tries": 1, "nonce": "n"}})}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reprotect_after_late_sell", return_value=None), \
         patch("alerts._send") as send:
        tasks._retry_pending_reprotects()
    assert json.loads(state["reprotect_pending"]) == {}
    assert "STOP NOT PLACED" in send.call_args[0][0]


def test_backstop_does_not_sell_after_a_split():
    """A1453-1: a 4:1 split (broker avg cost a quarter of the recorded entry)
    blocks the market sell and alerts."""
    import tasks
    pos = _Pos("NVDA", 4.0)
    pos.avg_entry_price = 45.0
    with patch("ibkr.get_client") as client, \
         patch("db.get_position_stop", return_value={"entry_price": 180.0, "stop_price": 160.0}), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("ibkr.place_market_order") as sell, \
         patch("alerts._send") as send:
        client.return_value.get_all_positions.return_value = [pos]
        assert tasks._execute_backstop_sell("NVDA", {"stop_price": 160.0}) is False
    cancel.assert_not_called(); sell.assert_not_called()
    assert "POSSIBLE SPLIT" in send.call_args[0][0]
    # No split: entry and avg cost close, so no ratio.
    pos.avg_entry_price = 172.0
    with patch("db.get_position_stop", return_value={"entry_price": 180.0}):
        assert tasks._split_ratio("NVDA", pos) is None
    pos.avg_entry_price = 1800.0  # 1:10 reverse split
    with patch("db.get_position_stop", return_value={"entry_price": 180.0}):
        assert tasks._split_ratio("NVDA", pos) == 0.1


def test_earnings_refresh_replaces_moved_dates_and_flags_estimates():
    """A1453-2: a moved date replaces the old future row; a two-date window is
    stored as an estimate; names with no date are named in the log."""
    from datetime import date, timedelta
    from unittest.mock import MagicMock
    import db
    from ingestion import earnings_calendar as ec
    d1 = date.today() + timedelta(days=5)
    cal = {"AAA": {"Earnings Date": [d1]},
           "BBB": {"Earnings Date": [d1, d1 + timedelta(days=4)]},
           "CCC": {"Earnings Date": []}}
    fake_yf = MagicMock()
    fake_yf.Ticker.side_effect = lambda t: MagicMock(calendar=cal[t])
    with patch.dict("sys.modules", {"yfinance": fake_yf}), \
         patch("knowledge_loader.get_all_tickers", return_value=["AAA", "BBB", "CCC"]), \
         patch("db.upsert_earnings_date") as up, patch.object(ec.time, "sleep"), \
         patch("catalyst_service.storage.upsert_earnings_from_sources", create=True), \
         patch.object(ec.log, "info") as info:
        ec.refresh_all_earnings()
    calls = {c.args[0]: c.kwargs for c in up.call_args_list}
    assert calls["AAA"] == {"confirmed": True, "replace_future": True}
    assert calls["BBB"]["confirmed"] is False
    assert "no upcoming date for CCC" in info.call_args[0][0]
    cur = MagicMock()
    cur.fetchall.return_value = [(d1 + timedelta(days=7),)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.upsert_earnings_date("AAA", d1, confirmed=True, replace_future=True)
    sqls = [c[0][0] for c in cur.execute.call_args_list]
    assert any("INSERT INTO earnings_calendar_history" in q for q in sqls)
    assert any("DELETE FROM earnings_calendar " in q for q in sqls)


def test_manual_actions_clear_when_covered_and_realert_when_not():
    """A1453-4: an open manual action is cleared once stops cover the shares,
    and re-alerted hourly while they don't."""
    import json
    import time
    import tasks
    state = {"manual_actions": json.dumps({
        "AMD": {"what": "place a stop", "first_seen": time.time() - 7200, "last_alert": time.time() - 3700},
        "NVDA": {"what": "place a stop", "first_seen": time.time() - 600, "last_alert": time.time() - 600}})}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("ibkr.get_client") as client, \
         patch("execution._stops_still_resting", side_effect=lambda t: {"AMD": 0.0, "NVDA": 1.0}[t]), \
         patch("alerts._send") as send:
        client.return_value.get_all_positions.return_value = [_Pos("AMD", 5.0), _Pos("NVDA", 1.0)]
        tasks._review_manual_actions()
    items = json.loads(state["manual_actions"])
    assert "NVDA" not in items and "AMD" in items
    assert "STILL OPEN" in send.call_args[0][0] and "AMD" in send.call_args[0][0]


def test_manual_action_needs_two_flat_reads_and_ignores_an_empty_read():
    """C28: one flat read doesn't clear an item; two in a row do; an empty
    position list (reconnect) clears nothing."""
    import json
    import time
    import tasks
    state = {"manual_actions": json.dumps({"AMD": {"what": "x", "first_seen": time.time(),
                                                   "last_alert": time.time()}})}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("ibkr.get_client") as client, \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("alerts._send"):
        client.return_value.get_all_positions.return_value = []
        tasks._review_manual_actions()
        assert "AMD" in json.loads(state["manual_actions"])
        state["manual_actions"] = json.dumps({"AMD": {"what": "x", "first_seen": time.time(),
                                                      "last_alert": time.time()}})
        client.return_value.get_all_positions.return_value = [_Pos("NVDA", 1.0)]
        tasks._review_manual_actions()
        assert json.loads(state["manual_actions"])["AMD"]["flat_reads"] == 1
        tasks._review_manual_actions()
        assert "AMD" not in json.loads(state["manual_actions"])


def test_flat_buy_item_never_clears_on_flat_and_empty_reads_clear_after_four():
    """C28a: a 'position reads flat' item clears only when covered. C28b: four
    empty reads in a row clear an ordinary item."""
    import json
    import time
    import tasks
    now = time.time()
    state = {"manual_actions": json.dumps({
        "AMD": {"what": "x", "first_seen": now, "last_alert": now, "clear_on_flat": False},
        "PLTR": {"what": "y", "first_seen": now, "last_alert": now}})}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("ibkr.get_client") as client, \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("alerts._send"):
        client.return_value.get_all_positions.return_value = []
        for _ in range(4):
            tasks._review_manual_actions()
    items = json.loads(state["manual_actions"])
    assert "AMD" in items and "PLTR" not in items


def test_catalyst_duplicate_earnings_rows_are_logged(tmp_path, monkeypatch):
    """A1653-1: two future earnings rows for a ticker in the SQLite calendar
    are named in a warning (log-only)."""
    import sqlite3
    from ingestion import earnings_calendar as ec
    db_file = tmp_path / "cat.db"
    c = sqlite3.connect(db_file)
    c.execute("CREATE TABLE catalyst_events (ticker TEXT, event_type TEXT, event_date TEXT, superseded_by TEXT)")
    c.executemany("INSERT INTO catalyst_events VALUES (?,?,?,NULL)",
                  [("NET", "earnings", "2099-10-29"), ("NET", "earnings", "2099-11-05"),
                   ("PLTR", "earnings", "2099-11-02")])
    c.commit(); c.close()
    monkeypatch.setattr("catalyst_service.storage.get_conn", lambda: sqlite3.connect(db_file))
    with patch.object(ec.log, "warning") as warn:
        assert ec._warn_catalyst_duplicates() == ["NET (2099-10-29,2099-11-05)"]
    assert "NET" in warn.call_args[0][0] and "PLTR" not in warn.call_args[0][0]


def test_busy_ticker_is_not_counted_as_a_reprotect_try(monkeypatch):
    """Review #5: a busy ticker lock leaves the queued entry untouched."""
    import json
    import stop_lock
    import tasks
    from conftest import _FakeLock
    monkeypatch.setattr(stop_lock, "DEFAULT_WAIT_S", 0)
    _FakeLock.held.add("helios:stop_lock:AMD")
    state = {"reprotect_pending": json.dumps({"AMD": {"intent": "RESTORE", "tries": 3, "nonce": "n"}})}
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("alerts._send") as send:
        tasks._retry_pending_reprotects()
    assert json.loads(state["reprotect_pending"])["AMD"]["tries"] == 3
    send.assert_not_called()


def test_reprotect_does_not_confirm_a_lagging_read_after_a_manual_add():
    """Review of 2f95331: holding 100 with a stop for 100, an ADD of 50, and
    a read still showing 100 must not count the old stop as full cover."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=100)]
    with patch("ibkr.get_client", return_value=client), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("execution._stops_still_resting", return_value=100.0), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_stop_place") as place:
        assert tasks._reprotect_after_late_sell("CRWD", {}, None, from_queue=True, expect_at_least=150) is False
        assert tasks._reprotect_after_late_sell("CRWD", {}, None, from_queue=True) is True
    assert not place.called


def test_orphan_outcome_flag_skips_an_empty_snapshot_and_passes_held_tickers():
    """Professor M111: open outcomes with no position get flagged, never closed."""
    with patch("tasks._held_qty_tickers", return_value=set()), patch("db.flag_orphan_outcomes") as flag:
        assert tasks.flag_orphan_outcomes() is None
    assert not flag.called
    with patch("tasks._held_qty_tickers", return_value={"NVDA", "PLTR"}), \
         patch("db.flag_orphan_outcomes", return_value=(3, 105)) as flag:
        assert tasks.flag_orphan_outcomes() == 105
    assert flag.call_args.args[0] == {"NVDA", "PLTR"}


def test_flag_orphan_outcomes_sql_flags_unheld_and_never_unflags():
    from unittest.mock import MagicMock
    import db
    cur = MagicMock()
    cur.rowcount = 3
    cur.fetchone.return_value = (105,)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.flag_orphan_outcomes({"nvda"}) == (3, 105)
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert "orphaned_at = NOW(), orphan_reason = 'sweep'" in sqls[0] and "<> ALL" in sqls[0]
    assert "NOT EXISTS" in sqls[0] and "position_stops" in sqls[0]     # cache-lag guard
    assert "entry_date < NOW() - INTERVAL '30 minutes'" in sqls[0]
    assert "o.status IN ('unconfirmed', 'partial')" in sqls[0] and "o.side = 'SELL'" in sqls[0]   # m110
    assert not any("orphaned_at = NULL" in q for q in sqls)            # professor M120
    assert cur.execute.call_args_list[0].args[1][1] == ["NVDA"]
    assert "exit_date IS NULL" in sqls[0] and "UPDATE" in sqls[0] and "exit_price" not in sqls[0]


def test_held_tickers_keep_a_position_with_zero_market_value():
    with patch("tasks._fresh_snapshot_positions",
               return_value={"amd": {"qty": 1, "market_value": 0}, "x": {"qty": 0}, "y": "bad"}):
        assert tasks._held_qty_tickers() == {"AMD"}


def test_routing_helper_is_not_shadowed():
    """Review of 352d290: a second _held_tickers replaced the LLM-routing one."""
    import inspect
    assert "max_age_min" not in inspect.signature(tasks._held_tickers).parameters


def test_close_trade_outcome_refuses_an_unusable_price_and_skips_orphaned_rows():
    """Professor M119/M120."""
    from unittest.mock import MagicMock
    import db
    with patch("db._get_conn") as get_conn, patch("db.flag_orphan_outcomes_for", return_value=1) as flag:
        assert db.close_trade_outcome("NVDA", exit_price=0.0) is False
        assert db.close_trade_outcome("NVDA", exit_price=float("nan")) is False
        assert db.close_trade_outcome("NVDA", exit_price=None) is False
    assert not get_conn.called                       # nothing closed
    assert [c.args for c in flag.call_args_list] == [("NVDA", "refused_price")] * 3
    cur = MagicMock()
    cur.fetchall.return_value = []
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.close_trade_outcome("NVDA", exit_price=100.0)
    assert "orphaned_at IS NULL" in cur.execute.call_args_list[0].args[0]



class TestBuyOutcomeRetry:
    """Professor M125: a late BUY's outcome write failed once and was lost;
    the order was already marked filled, so nothing ever retried it."""

    REC = {"kind": "insert", "ticker": "CRWD", "action": "OPEN", "entry_price": 181.3}

    def _state(self):
        state = {}
        return state, patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
            patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v))

    def test_a_failed_write_is_queued_and_a_later_success_clears_it(self):
        import json
        state, gs, ss = self._state()
        with gs, ss, patch("execution.log_buy_outcome", return_value=False):
            tasks._apply_buy_outcome(59, self.REC, {"id": 59})
        assert json.loads(state[tasks._OUTCOME_STATE_KEY])["59"]["tries"] == 0
        with gs, ss, patch("db.get_approval_by_id", return_value={"id": 59}), \
             patch("execution.log_buy_outcome", return_value=True) as write:
            tasks._retry_pending_buy_outcomes()
        write.assert_called_once()
        assert write.call_args.args == ({"id": 59}, "CRWD", "OPEN", 181.3)
        assert json.loads(state[tasks._OUTCOME_STATE_KEY]) == {}

    def test_a_success_queues_nothing(self):
        state, gs, ss = self._state()
        with gs, ss, patch("db.update_outcome_entry_price", return_value=True) as reavg:
            tasks._apply_buy_outcome(59, {**self.REC, "kind": "reaverage"})
        reavg.assert_called_once_with(59, 181.3)
        assert tasks._OUTCOME_STATE_KEY not in state

    def test_gives_up_with_one_alert_after_the_last_try(self):
        import json
        state, gs, ss = self._state()
        state[tasks._OUTCOME_STATE_KEY] = json.dumps(
            {"59": {**self.REC, "tries": tasks._OUTCOME_MAX_TRIES - 1}})
        with gs, ss, patch("db.get_approval_by_id", side_effect=RuntimeError("db down")), \
             patch("alerts._send") as alert:
            tasks._retry_pending_buy_outcomes()
        alert.assert_called_once()
        assert json.loads(state[tasks._OUTCOME_STATE_KEY]) == {}

    def test_reaverage_with_no_open_row_writes_the_row_instead(self):
        """Review of fed009b: the part fill's own row was never written."""
        state, gs, ss = self._state()
        with gs, ss, patch("db.update_outcome_entry_price", return_value=False), \
             patch("execution.log_buy_outcome", return_value=True) as write:
            tasks._apply_buy_outcome(59, {**self.REC, "kind": "reaverage"}, {"id": 59})
        write.assert_called_once()
        assert write.call_args.args == ({"id": 59}, "CRWD", "OPEN", 181.3)
        assert tasks._OUTCOME_STATE_KEY not in state

    def test_a_failed_reaverage_is_retried_not_inserted(self):
        import json
        state, gs, ss = self._state()
        with gs, ss, patch("db.update_outcome_entry_price", return_value=None), \
             patch("execution.log_buy_outcome") as write:
            tasks._apply_buy_outcome(59, {**self.REC, "kind": "reaverage"})
        write.assert_not_called()
        assert json.loads(state[tasks._OUTCOME_STATE_KEY])["59"]["kind"] == "reaverage"

    def test_a_retried_row_keeps_the_first_attempts_time_and_spy_price(self):
        """SEF 10-30: entry_date and the SPY price used to be taken at the retry."""
        import json
        from unittest.mock import MagicMock
        state, gs, ss = self._state()
        spy = MagicMock(); spy.fast_info.last_price = 580.0
        with gs, ss, patch("execution.log_buy_outcome", return_value=False), \
             patch("yfinance.Ticker", return_value=spy):
            tasks._apply_buy_outcome(59, self.REC, {"id": 59})
        first = json.loads(state[tasks._OUTCOME_STATE_KEY])["59"]
        assert first["entry_benchmark_price"] == 580.0 and first["entry_date"]
        spy.fast_info.last_price = 590.0
        with gs, ss, patch("execution.log_buy_outcome", return_value=False), \
             patch("yfinance.Ticker", return_value=spy):
            tasks._apply_buy_outcome(59, {**self.REC, "kind": "reaverage"}, {"id": 59})
        again = json.loads(state[tasks._OUTCOME_STATE_KEY])["59"]
        assert again["entry_benchmark_price"] == 580.0 and again["entry_date"] == first["entry_date"]
        with gs, ss, patch("db.get_approval_by_id", return_value={"id": 59}), \
             patch("db.update_outcome_entry_price", return_value=False), \
             patch("execution.log_buy_outcome", return_value=True) as write:
            tasks._retry_pending_buy_outcomes()
        assert write.call_args.kwargs == {"entry_date": first["entry_date"], "entry_benchmark_price": 580.0}


class TestCancelledUnconfirmedOrder:
    """Professor M127: a cancelled unconfirmed order was never resolved, so the
    stale-stop branch waited on it until midnight."""

    ORDER = {"id": 44, "ticker": "CRWD", "side": "SELL", "qty": 20.0, "broker_order_id": 900,
             "decision_price": 180.0, "approval_id": None, "intent": "EXIT", "stop_price": None,
             "created_at": None, "filled_qty": 0.0, "fill_price": None, "perm_id": 7700}

    def _run(self, completed, fill=None, order=None, positions=(), other_sell=None):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.get_all_positions.return_value = list(positions)
        kw = {"side_effect": completed} if isinstance(completed, Exception) else {"return_value": completed}
        with patch("db.get_unconfirmed_orders", return_value=[order or self.ORDER]), \
             patch("tasks._retry_pending_reprotects"), patch("tasks._retry_pending_buy_outcomes"), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.get_fill_for_order", return_value=fill), \
             patch("ibkr.get_completed_orders", **kw) as read, \
             patch("ibkr.get_client", return_value=client), \
             patch("db.get_position_stop", return_value=None), \
             patch("db.recorded_sell_order_ids", return_value=({900}, set())), \
             patch("ibkr.get_recent_sell_fill", return_value=other_sell), \
             patch("db.mark_order_cancelled", return_value=True) as mark, \
             patch("db.close_trade_outcome") as close, \
             patch("tasks._reconcile_alert_due", return_value=True), patch("alerts._send") as alert, \
             patch("db.log_trade") as trade:
            tasks.reconcile_unconfirmed_orders.run()
        trade.assert_not_called()
        return mark, close, alert, read

    @staticmethod
    def _done(status="Cancelled", filled=0.0, perm=7700, action="SELL", ticker="CRWD"):
        return [{"perm_id": perm, "ticker": ticker, "action": action, "status": status, "filled_qty": filled}]

    def test_a_cancelled_order_with_no_fill_is_marked_cancelled(self):
        mark, close, alert, _ = self._run(self._done())
        mark.assert_called_once_with(44, 0.0)
        close.assert_not_called()
        alert.assert_not_called()

    def test_left_waiting_when_unknown(self):
        for completed in ([], self._done(status="Filled"), self._done(perm=1), self._done(action="BUY"),
                          self._done(filled=None), RuntimeError("gateway")):
            mark, _, alert, _ = self._run(completed)
            mark.assert_not_called()
            alert.assert_not_called()
        mark, _, _, read = self._run(self._done(), order={**self.ORDER, "perm_id": None})
        mark.assert_not_called()
        read.assert_not_called()   # no permId: nothing to match on

    def test_a_larger_broker_fill_is_alerted_not_marked(self):
        mark, close, alert, _ = self._run(self._done(filled=5.0))
        mark.assert_not_called()
        alert.assert_called_once()
        assert "FILL NOT RECORDED" in alert.call_args.args[0]

    def test_a_cancelled_part_filled_exit_is_marked_but_never_closed_here(self):
        """Review of 0a80ce7: flat after a part-filled exit is a mixed close."""
        part = {**self.ORDER, "filled_qty": 8.0, "fill_price": 181.0}
        mark, close, _, _ = self._run(self._done(status="ApiCancelled", filled=8.0),
                                      fill={"filled_qty": 8.0, "fill_price": 181.0}, order=part)
        mark.assert_called_once_with(44, 8.0)
        close.assert_not_called()

    def test_less_filled_at_the_broker_than_recorded_is_left_open(self):
        part = {**self.ORDER, "filled_qty": 8.0, "fill_price": 181.0}
        mark, _, alert, _ = self._run(self._done(filled=5.0), fill={"filled_qty": 8.0, "fill_price": 181.0},
                                      order=part)
        mark.assert_not_called()
        alert.assert_not_called()


def test_perm_id_backfill_needs_one_unambiguous_open_order():
    from unittest.mock import patch as p
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    rows = [{"id": 1, "ticker": "CRWD", "side": "SELL", "qty": 20.0, "broker_order_id": 900, "perm_id": None,
             "created_at": now},
            {"id": 2, "ticker": "AMD", "side": "SELL", "qty": 5.0, "broker_order_id": 901, "perm_id": None,
             "created_at": now},
            {"id": 3, "ticker": "NVDA", "side": "SELL", "qty": 4.0, "broker_order_id": 902, "perm_id": None,
             "created_at": now},
            {"id": 4, "ticker": "TSLA", "side": "SELL", "qty": 2.0, "broker_order_id": 903, "perm_id": None,
             "created_at": now - timedelta(days=3)}]
    mkt = {"order_type": "MKT", "action": "SELL"}
    open_orders = [{**mkt, "order_id": 900, "ticker": "CRWD", "qty": 20.0, "perm_id": 7700},
                   {**mkt, "order_id": 901, "ticker": "AMD", "qty": 5.0, "perm_id": 8800},
                   {**mkt, "order_id": 901, "ticker": "AMD", "qty": 5.0, "perm_id": 8801},   # two clients
                   {"order_id": 902, "ticker": "NVDA", "action": "SELL", "order_type": "STP", "qty": 4.0,
                    "perm_id": 9900},                                                            # a stop
                   {**mkt, "order_id": 903, "ticker": "TSLA", "qty": 2.0, "perm_id": 9910}]   # old row
    with p("ibkr.get_open_orders", return_value=open_orders), \
         p("db.set_order_perm_id", return_value=True) as setp:
        tasks._backfill_perm_ids(rows)
    setp.assert_called_once_with(1, 7700)
    assert rows[0]["perm_id"] == 7700 and rows[1]["perm_id"] is None


class TestRetryHeldSell:
    """m127: one re-execution of a TRIM/EXIT held on unreadable stops, under
    the professor's conditions (once, still held, nothing newer, market hours,
    attempt recorded)."""

    APPROVAL = {"id": 81, "ticker": "AMD", "action": "EXIT", "status": "FAILED", "timestamp": "T0"}

    HELD = "skipped: resting stops unreadable, sell not sent (stop left in place)"

    def _run(self, approval=None, market=True, held=5.0, newer=False, claim=HELD, success=True):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=held)] if held else []
        with patch("db.get_approval_by_id", return_value=approval or self.APPROVAL), \
             patch("market_data.is_market_hours", return_value=market), \
             patch("ibkr.get_client", return_value=client), \
             patch("db.has_newer_ticker_activity", return_value=newer), \
             patch("db.claim_failed_sell_for_retry", return_value=claim) as claim_m, \
             patch("execution.execute_approved", return_value=success) as ex, \
             patch("execution.last_outcome_reason", return_value="skipped: x"), \
             patch("db.update_approval_status") as upd:
            tasks.retry_held_sell.run(81)
        return ex, upd, claim_m

    def test_retries_once_and_records_the_attempt(self):
        ex, upd, _ = self._run()
        ex.assert_called_once()
        assert ex.call_args.kwargs["trigger"] == "held_sell_retry"
        status, reason = upd.call_args.args[1], upd.call_args.kwargs["reason"]
        assert status == "APPROVED" and reason.startswith(self.HELD + " | retried at ") and reason.endswith("executed")

    def test_a_failed_retry_stays_failed_with_its_reason(self):
        _, upd, _ = self._run(success=False)
        assert upd.call_args.args[1] == "FAILED" and upd.call_args.kwargs["reason"].endswith("skipped: x")

    def test_no_retry_outside_market_hours_and_the_row_says_so(self):
        ex, upd, _ = self._run(market=False)
        ex.assert_not_called()
        assert upd.call_args.kwargs["reason"].endswith("not retried: market closed")

    def test_no_retry_when_nothing_is_held(self):
        ex, upd, _ = self._run(held=0.0)
        ex.assert_not_called()
        assert "no position held" in upd.call_args.kwargs["reason"]

    def test_no_retry_when_a_newer_decision_exists(self):
        ex, upd, _ = self._run(newer=True)
        ex.assert_not_called()
        assert "superseded" in upd.call_args.kwargs["reason"]

    def test_no_retry_when_the_activity_check_fails_and_the_row_says_so(self):
        ex, upd, _ = self._run(newer=None)
        ex.assert_not_called()
        assert upd.call_args.kwargs["reason"].endswith("not retried: newer-activity check failed")

    def test_a_failed_position_read_is_noted(self):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.get_all_positions.side_effect = RuntimeError("down")
        with patch("db.get_approval_by_id", return_value=self.APPROVAL), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("ibkr.get_client", return_value=client), \
             patch("db.claim_failed_sell_for_retry", return_value=self.HELD), \
             patch("execution.execute_approved") as ex, \
             patch("db.update_approval_status") as upd:
            tasks.retry_held_sell.run(81)
        ex.assert_not_called()
        assert upd.call_args.kwargs["reason"].endswith("not retried: position read failed")

    def test_a_lost_claim_on_a_skip_writes_nothing(self):
        ex, upd, _ = self._run(market=False, claim=None)
        ex.assert_not_called(); upd.assert_not_called()

    def test_no_retry_when_the_claim_is_lost(self):
        ex, upd, _ = self._run(claim=None)
        ex.assert_not_called(); upd.assert_not_called()

    def test_a_non_failed_approval_is_left_alone(self):
        ex, upd, claim = self._run(approval={**self.APPROVAL, "status": "APPROVED"})
        ex.assert_not_called(); claim.assert_not_called()


def test_claim_failed_sell_for_retry_sql():
    """m127: once only, TRIM/EXIT, the held reason prefix; returns that reason."""
    from unittest.mock import MagicMock
    import db
    cur = MagicMock()
    cur.fetchone.return_value = ("skipped: resting stops unreadable, sell not sent",)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.claim_failed_sell_for_retry(81) == "skipped: resting stops unreadable, sell not sent"
        cur.fetchone.return_value = None
        assert db.claim_failed_sell_for_retry(81) is None
    sql, params = cur.execute.call_args[0]
    assert "status='FAILED'" in sql and "IN ('TRIM','EXIT')" in sql and "NOT LIKE" in sql
    assert "RETURNING status_reason" in sql and "SET status='PROCESSING'" in sql
    assert params == (81, "skipped: resting stops unreadable%", "%retried at%")


def test_has_newer_ticker_activity_checks_approvals_then_verdicts():
    from unittest.mock import MagicMock
    import db
    cur = MagicMock()
    cur.fetchone.side_effect = [None, (1,)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.has_newer_ticker_activity("AMD", 81, "T0") is True
    first, second = cur.execute.call_args_list
    assert "pending_approvals" in first[0][0] and first[0][1][2] == 81
    assert "signals" in second[0][0] and "timestamp > %s" in second[0][0] and second[0][1][-1] == "T0"
    cur.execute.side_effect = RuntimeError("down")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.has_newer_ticker_activity("AMD", 81, "T0") is None
