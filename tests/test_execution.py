"""
Tests for execution.py's sell-path safety properties — no live IBKR/DB
required. Covers the bugs found in the 2026-07-15 audit: TRIM oversell,
drift blocking risk-reducing sells, and phantom fills.
"""
from unittest.mock import MagicMock, patch

import pytest

import execution


def _approval(action="TRIM", shares_initial=100, price_at_sizing=100.0, stop_loss_price=90.0):
    return {
        "id": 1,
        "ticker": "NVDA",
        "action": action,
        "conviction": 7,
        "thesis_bucket": "Accelerator",
        "stop_loss_price": stop_loss_price,
        "risk_json": {
            "shares_initial": shares_initial,
            "price_at_sizing": price_at_sizing,
            "stop_loss_price": stop_loss_price,
        },
    }


class TestSellClampAndDrift:
    def test_trim_clamps_to_owned_shares(self):
        """TRIM sized at 100 shares but only 30 are held must sell 30, not
        100 — selling more than owned would flip the account short."""
        approval = _approval(action="TRIM", shares_initial=100)
        position = MagicMock(qty=30.0)
        ibkr_client = MagicMock()
        ibkr_client.get_open_position.return_value = position

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client", return_value=ibkr_client), \
             patch("ibkr.cancel_stop_orders_for", return_value=0), \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.get_latest_price", return_value=100.0), \
             patch("market_data.get_priced_quote", return_value=None), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.update_order_fill"), patch("db.delete_position_stop"), \
             patch("db.get_position_stop", return_value=None), \
             patch("db.log_trade_outcome"), \
             patch("alerts.send_trade_alert"):
            mock_order.return_value = {"order_id": 1, "fill_price": 100.0, "filled_qty": 30.0}
            execution.execute_approved(approval)

        sold_qty = mock_order.call_args[0][2]
        assert sold_qty == 30.0

    def test_exit_sells_full_held_quantity_ignoring_sizing(self):
        """EXIT must sell the actual held quantity, not whatever
        shares_initial happened to be sized at for a fresh entry."""
        approval = _approval(action="EXIT", shares_initial=999)
        position = MagicMock(qty=42.0)
        ibkr_client = MagicMock()
        ibkr_client.get_open_position.return_value = position

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client", return_value=ibkr_client), \
             patch("ibkr.cancel_stop_orders_for", return_value=0), \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.get_latest_price", return_value=100.0), \
             patch("market_data.get_priced_quote", return_value=None), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.update_order_fill"), patch("db.delete_position_stop"), \
             patch("db.close_trade_outcome"), \
             patch("db.log_trade_outcome"), \
             patch("alerts.send_trade_alert"):
            mock_order.return_value = {"order_id": 1, "fill_price": 100.0, "filled_qty": 42.0}
            execution.execute_approved(approval)

        sold_qty = mock_order.call_args[0][2]
        assert sold_qty == 42.0

    def test_sell_not_aborted_by_large_drift(self):
        """A >8% price move must never block a risk-reducing sell — that's
        backwards (big moves are often exactly why you're exiting), and
        combined with cancelling stops first it used to strand the
        position both unsold and unprotected."""
        approval = _approval(action="EXIT", price_at_sizing=100.0)
        position = MagicMock(qty=10.0)
        ibkr_client = MagicMock()
        ibkr_client.get_open_position.return_value = position

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client", return_value=ibkr_client), \
             patch("ibkr.cancel_stop_orders_for", return_value=1) as mock_cancel, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.get_latest_price", return_value=115.0), \
             patch("market_data.get_priced_quote",
                   return_value={"price": 115.0, "is_stale": False, "age_minutes": 1, "market_open": True}), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.update_order_fill"), patch("db.delete_position_stop"), \
             patch("db.close_trade_outcome"), \
             patch("db.log_trade_outcome"), \
             patch("alerts.send_trade_alert"):
            mock_order.return_value = {"order_id": 1, "fill_price": 115.0, "filled_qty": 10.0}
            result = execution.execute_approved(approval)

        assert result is True
        mock_order.assert_called_once()  # sell actually went through
        mock_cancel.assert_called_once()  # stop still gets cancelled before the sell

    def test_buy_is_aborted_by_large_drift(self):
        """The drift guard should still protect BUYs — only sells are exempt."""
        approval = _approval(action="ADD", price_at_sizing=100.0)
        approval["action"] = "ADD"

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.get_latest_price", return_value=115.0), \
             patch("market_data.get_priced_quote",
                   return_value={"price": 115.0, "is_stale": False, "age_minutes": 1, "market_open": True}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("alerts._send"):
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval)

        assert result is False
        mock_order.assert_not_called()


class TestPhantomFills:
    def test_zero_fill_records_nothing(self):
        """An order that submits but doesn't confirm a fill within the wait
        window must never be recorded as a full-size trade — that
        diverges DB state from the broker (phantom position, phantom
        stop for shares that may not exist)."""
        approval = _approval(action="ADD", shares_initial=50)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.get_latest_price", return_value=100.0), \
             patch("market_data.get_priced_quote", return_value=None), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade") as mock_log_trade, \
             patch("db.log_order") as mock_log_order, \
             patch("db.upsert_position_stop") as mock_upsert_stop, \
             patch("alerts._send"):
            mock_rm_cls.return_value.is_halted.return_value = False
            mock_order.return_value = {"order_id": 55, "fill_price": None, "filled_qty": 0.0}
            result = execution.execute_approved(approval)

        assert result is False
        mock_log_trade.assert_not_called()
        mock_upsert_stop.assert_not_called()
        # The unconfirmed order IS logged (status="unconfirmed") for
        # reconciliation, distinct from a confirmed fill.
        assert mock_log_order.call_args.kwargs.get("status") == "unconfirmed"


class TestMarketHoursGate:
    """Regression coverage for the 2026-07-26 incident: a DAY market BUY
    submitted while the market was closed got queued by IBKR for the next
    session instead of rejected, and the 30s fill-wait then declared the
    trade FAILED without ever cancelling the now-orphaned, still-live
    order. New entries (OPEN/ADD) must never even attempt an order while
    the market is closed; risk-reducing sells (TRIM/EXIT) are deliberately
    exempt."""

    def test_new_entry_blocked_when_market_closed(self):
        """Blocked, and a single retry gets scheduled for the next open —
        added 2026-07-28 alongside the 5:00 AM wake, so pre-market
        analysis that finishes before 9:30 isn't just discarded."""
        import datetime
        approval = _approval(action="ADD")
        fake_eta = datetime.datetime(2026, 7, 28, 13, 30, tzinfo=datetime.timezone.utc)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.is_market_hours", return_value=False), \
             patch("market_data.next_market_open", return_value=fake_eta), \
             patch("tasks.retry_deferred_entry") as mock_retry_task, \
             patch("alerts._send"):
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval)

        assert result is False
        mock_order.assert_not_called()
        mock_retry_task.apply_async.assert_called_once_with(
            args=[approval["id"]], eta=fake_eta, queue="ops",
        )

    def test_sell_not_blocked_when_market_closed(self):
        """TRIM/EXIT must still be allowed to proceed even when closed —
        blocking a risk-reducing sell outright would contradict the "never
        abort a risk-reducing order" principle used for the drift guard."""
        approval = _approval(action="EXIT", shares_initial=100)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client") as mock_client, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.cancel_stop_orders_for", return_value=0), \
             patch("ibkr.get_latest_price", return_value=100.0), \
             patch("market_data.get_priced_quote", return_value=None), \
             patch("market_data.is_market_hours", return_value=False), \
             patch("db.log_trade"), patch("db.log_order"), patch("db.delete_position_stop"), \
             patch("alerts._send"):
            mock_client.return_value.get_open_position.return_value = MagicMock(qty=100)
            mock_order.return_value = {"order_id": 56, "fill_price": 100.0, "filled_qty": 100.0}
            execution.execute_approved(approval)

        mock_order.assert_called_once()


class TestStalePriceBuyGuard:
    """Regression coverage for the 2026-07-29 TER incident: right at market
    open, IBKR's intraday bars aren't populated yet, so get_priced_quote
    falls back to a stale daily close. That stale price used to be fed
    straight into the drift check, which then compared it against a
    price_at_sizing computed from the same pre-gap close -- ~0% apparent
    drift, guard never fires. Six queued retries all bought into a ~13%
    earnings gap this way. A BUY now retries briefly for a fresh quote and
    aborts rather than compare stale-to-stale; SELLs are unaffected."""

    def _stale_quote(self, price=100.0):
        return {"price": price, "is_stale": True, "age_minutes": 2251.0, "market_open": True}

    def _fresh_quote(self, price=100.0):
        return {"price": price, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def test_buy_defers_and_retries_when_price_stays_stale(self):
        """First miss must schedule a retry, not permanently abort.

        Incident 2026-07-30/31: this path used to be a dead end -- no
        retry at all, unlike the market-hours-closed and gap-confirmation
        gates -- and IBKR's HMDS intraday-bar service reliably has no data
        for the first few minutes after the open, so this silently killed
        5 conviction-7/8 approvals on 2026-07-31 alone. Now retries on the
        same bounded shape as GAP_CONFIRMATION_*."""
        approval = _approval(action="ADD", price_at_sizing=100.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._stale_quote()), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.retry_deferred_entry") as mock_retry_task, \
             patch("time.sleep"), \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval, price_retry_attempt=1)

        assert result is False
        mock_order.assert_not_called()
        assert "WAITING FOR FRESH PRICE" in mock_alert.call_args[0][0]
        mock_retry_task.apply_async.assert_called_once()
        _, kwargs = mock_retry_task.apply_async.call_args
        assert kwargs["args"] == [approval["id"], 1, 2]  # gap unchanged, price attempt 1 -> 2

    def test_buy_aborts_when_price_stays_stale_after_max_retries(self):
        """Once STALE_PRICE_MAX_RETRIES is reached, give up for good rather
        than defer forever."""
        from config import STALE_PRICE_MAX_RETRIES
        approval = _approval(action="ADD", price_at_sizing=100.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._stale_quote()), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.retry_deferred_entry") as mock_retry_task, \
             patch("time.sleep"), \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval, price_retry_attempt=STALE_PRICE_MAX_RETRIES)

        assert result is False
        mock_order.assert_not_called()
        assert "Price Unavailable" in mock_alert.call_args[0][0]
        mock_retry_task.apply_async.assert_not_called()

    def test_buy_proceeds_once_price_turns_fresh_within_retries(self):
        """The intraday feed catching up mid-retry (the common case) must
        let the order proceed with the now-fresh price, not abort."""
        approval = _approval(action="ADD", price_at_sizing=100.0)
        quotes = [self._stale_quote(), self._stale_quote(), self._fresh_quote(price=102.0)]

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.get_latest_price", return_value=102.0), \
             patch("market_data.get_priced_quote", side_effect=quotes), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("time.sleep"), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.log_trade_outcome"), patch("alerts._send"), \
             patch("alerts.send_trade_alert"):
            mock_rm_cls.return_value.is_halted.return_value = False
            mock_order.return_value = {"order_id": 1, "fill_price": 102.0, "filled_qty": 100.0}
            result = execution.execute_approved(approval)

        assert result is True
        mock_order.assert_called_once()

    def test_sell_proceeds_despite_stale_price_no_retry(self):
        """A stale price must never block or delay a risk-reducing sell --
        only BUYs get the retry-then-abort treatment."""
        approval = _approval(action="EXIT", shares_initial=100)
        position = MagicMock(qty=100.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client") as mock_client, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.cancel_stop_orders_for", return_value=0), \
             patch("market_data.get_priced_quote", return_value=self._stale_quote(price=100.0)) as mock_quote, \
             patch("db.log_trade"), patch("db.log_order"), patch("db.delete_position_stop"), \
             patch("db.close_trade_outcome"), patch("db.log_trade_outcome"), \
             patch("time.sleep") as mock_sleep, \
             patch("alerts._send"), patch("alerts.send_trade_alert"):
            mock_client.return_value.get_open_position.return_value = position
            mock_order.return_value = {"order_id": 57, "fill_price": 100.0, "filled_qty": 100.0}
            result = execution.execute_approved(approval)

        assert result is True
        mock_order.assert_called_once()
        mock_sleep.assert_not_called()
        assert mock_quote.call_count == 1  # no retry loop for a SELL


class TestGapConfirmationGuard:
    """Regression coverage for the 2026-07-29 TER incident's second half:
    even with a genuinely fresh price, execution.execute_approved had no
    concept of "this gap hasn't been confirmed yet" -- it would fire the
    instant the market-hours gate cleared. market_data.check_gap_confirmation
    now gates every BUY; these tests cover execution.py's handling of its
    three real outcomes plus the "give up" case."""

    def _fresh_quote(self, price=110.0):
        return {"price": price, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def test_buy_proceeds_when_gap_confirmed(self):
        approval = _approval(action="ADD", price_at_sizing=108.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation",
                   return_value={"status": "confirmed", "gap_pct": 0.10, "retracement_pct": 0.1,
                                 "open_price": 110.0, "prior_close": 100.0, "current_price": 109.0}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.log_trade_outcome"), patch("alerts._send"), \
             patch("alerts.send_trade_alert"):
            mock_rm_cls.return_value.is_halted.return_value = False
            mock_order.return_value = {"order_id": 1, "fill_price": 110.0, "filled_qty": 100.0}
            result = execution.execute_approved(approval)

        assert result is True
        mock_order.assert_called_once()

    def test_buy_aborts_when_gap_already_failed(self):
        approval = _approval(action="ADD", price_at_sizing=108.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation",
                   return_value={"status": "failed", "gap_pct": 0.10, "retracement_pct": 0.6,
                                 "open_price": 110.0, "prior_close": 100.0, "current_price": 104.0}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval)

        assert result is False
        mock_order.assert_not_called()
        assert "Gap Reversing" in mock_alert.call_args[0][0]

    def test_buy_defers_and_reschedules_when_gap_pending(self):
        approval = _approval(action="ADD", price_at_sizing=108.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation",
                   return_value={"status": "pending", "gap_pct": 0.10, "retracement_pct": 0.2,
                                 "open_price": 110.0, "prior_close": 100.0, "current_price": 108.0}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.retry_deferred_entry") as mock_retry_task, \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval, gap_retry_attempt=1)

        assert result is False
        mock_order.assert_not_called()
        mock_retry_task.apply_async.assert_called_once()
        assert mock_retry_task.apply_async.call_args.kwargs["args"] == [approval["id"], 2]
        assert "WAITING FOR CONFIRMATION" in mock_alert.call_args[0][0]

    def test_buy_gives_up_when_gap_pending_at_max_retries(self):
        """Bounded retry chain -- must not defer forever."""
        approval = _approval(action="ADD", price_at_sizing=108.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation",
                   return_value={"status": "pending", "gap_pct": 0.10, "retracement_pct": 0.2,
                                 "open_price": 110.0, "prior_close": 100.0, "current_price": 108.0}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.retry_deferred_entry") as mock_retry_task, \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval, gap_retry_attempt=3)

        assert result is False
        mock_order.assert_not_called()
        mock_retry_task.apply_async.assert_not_called()
        assert "Gap Unconfirmed" in mock_alert.call_args[0][0]

    def test_sell_never_calls_gap_confirmation(self):
        approval = _approval(action="EXIT", shares_initial=100)
        position = MagicMock(qty=100.0)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client") as mock_client, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("ibkr.cancel_stop_orders_for", return_value=0), \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote(price=100.0)), \
             patch("market_data.check_gap_confirmation") as mock_gap, \
             patch("db.log_trade"), patch("db.log_order"), patch("db.delete_position_stop"), \
             patch("db.close_trade_outcome"), patch("db.log_trade_outcome"), \
             patch("alerts._send"), patch("alerts.send_trade_alert"):
            mock_client.return_value.get_open_position.return_value = position
            mock_order.return_value = {"order_id": 58, "fill_price": 100.0, "filled_qty": 100.0}
            result = execution.execute_approved(approval)

        assert result is True
        mock_gap.assert_not_called()


class TestCatalystTrackingPersistence:
    """2026-08-30, user request: catalyst description, expected-return
    target, and cited sources used to only live transiently in
    pending_approvals.analyst_json -- these tests guard that
    execute_approved actually promotes them into the log_trade_outcome
    call (which persists to trade_outcomes), not just that they exist
    somewhere in the approval dict."""

    def _fresh_quote(self, price=110.0):
        return {"price": price, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def _approval_with_analyst_json(self, analyst_json):
        return {
            "id": 1, "ticker": "NVDA", "action": "ADD", "conviction": 8,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 100, "price_at_sizing": 108.0, "stop_loss_price": 90.0},
            "analyst_json": analyst_json,
        }

    def test_catalyst_and_expected_return_are_passed_through(self):
        analyst_json = {
            "catalyst": {"event": "Q3 earnings beat + raised guidance",
                          "date": "2026-11", "is_priced_in": False},
            "expected_return_pct": 28.5,
            "expected_downside_pct": -12.0,
            "thesis_summary": "AI datacenter demand still under-modeled by consensus.",
            "sources": [{"type": "news", "detail": "Reuters"}, {"type": "fundamentals_data", "detail": "Q2 beat"}],
            "source_count": 2,
        }
        approval = self._approval_with_analyst_json(analyst_json)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation",
                   return_value={"status": "confirmed", "gap_pct": 0.10, "retracement_pct": 0.1,
                                 "open_price": 110.0, "prior_close": 100.0, "current_price": 109.0}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.log_trade_outcome") as mock_log_outcome, \
             patch("alerts._send"), patch("alerts.send_trade_alert"):
            mock_rm_cls.return_value.is_halted.return_value = False
            mock_order.return_value = {"order_id": 1, "fill_price": 110.0, "filled_qty": 100.0}
            execution.execute_approved(approval)

        _, kwargs = mock_log_outcome.call_args
        assert kwargs["catalyst_event"] == "Q3 earnings beat + raised guidance"
        assert kwargs["catalyst_date"] == "2026-11"
        assert kwargs["catalyst_is_priced_in"] is False
        assert kwargs["expected_return_pct"] == 28.5
        assert kwargs["expected_downside_pct"] == -12.0
        assert kwargs["thesis_summary"] == "AI datacenter demand still under-modeled by consensus."
        assert kwargs["source_count"] == 2
        assert len(kwargs["sources"]) == 2

    def test_missing_catalyst_data_does_not_crash(self):
        # A stub recommendation (Ollama unavailable) has an empty/None
        # analyst_json in some paths -- this must not blow up execution.
        approval = self._approval_with_analyst_json({})

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation",
                   return_value={"status": "confirmed", "gap_pct": 0.10, "retracement_pct": 0.1,
                                 "open_price": 110.0, "prior_close": 100.0, "current_price": 109.0}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.log_trade_outcome") as mock_log_outcome, \
             patch("alerts._send"), patch("alerts.send_trade_alert"):
            mock_rm_cls.return_value.is_halted.return_value = False
            mock_order.return_value = {"order_id": 1, "fill_price": 110.0, "filled_qty": 100.0}
            result = execution.execute_approved(approval)

        assert result is True
        _, kwargs = mock_log_outcome.call_args
        assert kwargs["catalyst_event"] is None
        assert kwargs["expected_return_pct"] is None
        assert kwargs["sources"] is None
