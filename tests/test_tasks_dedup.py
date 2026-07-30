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

        mock_execute.assert_called_once_with(approval, trigger="auto_retry", gap_retry_attempt=1)
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

    def test_sell_side_late_fill_does_not_attempt_a_stop(self):
        """A late-filled SELL (e.g. a TRIM/EXIT) must never trigger stop
        placement -- stops are a BUY-side concern."""
        order = self._order(side="SELL", qty=50.0)
        with patch("db.get_unconfirmed_orders", return_value=[order]), \
             patch("ibkr.get_fill_for_order",
                   return_value={"filled_qty": 50.0, "fill_price": 175.0}), \
             patch("db.log_trade"), \
             patch("db.update_order_fill"), \
             patch("db.get_approval_by_id", return_value=self._approval()), \
             patch("ibkr.place_stop_order") as mock_stop, \
             patch("db.update_approval_status"), \
             patch("alerts._send"):
            tasks.reconcile_unconfirmed_orders.run()

        mock_stop.assert_not_called()


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
