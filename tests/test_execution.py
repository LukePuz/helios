"""
Tests for execution.py's sell-path safety properties — no live IBKR/DB
required. Covers the bugs found in the 2026-07-15 audit: TRIM oversell,
drift blocking risk-reducing sells, and phantom fills.
"""
from unittest.mock import MagicMock, patch

import pytest

import execution


@pytest.fixture(autouse=True)
def _regular_hours():
    """Default to regular hours; tests about closed markets patch it themselves."""
    with patch("market_data.is_market_hours", return_value=True):
        yield


@pytest.fixture(autouse=True)
def _no_prior_stop():
    """BUY fills look up the existing stop to decide whether this is an ADD
    (execution._whole_position_stop). Default: a fresh position."""
    with patch("db.get_position_stop", return_value=None):
        yield


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

    def test_sell_not_sent_when_market_closed_and_stop_kept(self):
        """Limitations audit 20:53 A2053-3: outside regular hours the broker
        call cancels a DAY market SELL anyway, but the stop had already been
        cancelled and was never put back. Now the order isn't sent and the
        stop isn't touched (same outcome for the sale, stop kept)."""
        approval = _approval(action="EXIT", shares_initial=100)

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client") as mock_client, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("tasks.manage_cancel_stop_orders_for") as cancel, \
             patch("market_data.is_market_hours", return_value=False), \
             patch("ibkr.get_latest_price", return_value=101.5), \
             patch("db.log_signal") as log_signal, \
             patch("alerts._send") as send:
            mock_client.return_value.get_open_position.return_value = MagicMock(qty=100)
            assert execution.execute_approved(approval) is False

        mock_order.assert_not_called()
        cancel.assert_not_called()
        assert "market closed" in execution.last_outcome_reason()
        assert "NOT SENT" in send.call_args[0][0]
        # Log-only record of the skipped sell (audit A0903-4)
        assert log_signal.call_args.args[1] == "SELL_SKIPPED_CLOSED"
        assert log_signal.call_args.kwargs["provenance"]["price_at_skip"] == 101.5

    def test_sell_cancelled_unfilled_restores_stop(self):
        approval = _approval(action="TRIM", shares_initial=40)
        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client") as mock_client, \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 9, "filled_qty": 0, "fill_price": None,
                                 "status": "Cancelled"}), \
             patch("tasks.manage_cancel_stop_orders_for", return_value=1), \
             patch("ibkr.get_latest_price", return_value=100.0), \
             patch("market_data.get_priced_quote", return_value=None), \
             patch("execution._restore_stop_after_failed_sell") as restore, \
             patch("db.log_order"), patch("alerts._send"):
            mock_client.return_value.get_open_position.return_value = MagicMock(qty=100)
            assert execution.execute_approved(approval) is False
        restore.assert_called_once_with(approval["ticker"])
        assert "stop restored" in execution.last_outcome_reason()


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

    def _gap_raises(self, attempt):
        approval = _approval(action="ADD", price_at_sizing=108.0)
        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=self._fresh_quote()), \
             patch("market_data.check_gap_confirmation", side_effect=RuntimeError("no bars")), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.retry_deferred_entry") as mock_retry_task, \
             patch("alerts._send") as mock_alert:
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval, gap_retry_attempt=attempt)
        assert result is False
        mock_order.assert_not_called()
        return mock_retry_task, mock_alert

    def test_buy_defers_when_gap_check_errors(self):
        """Audit 2026-10-28 14:53 #8: used to proceed with the BUY unchecked."""
        retry, _ = self._gap_raises(1)
        assert retry.apply_async.call_args.kwargs["args"][1] == 2

    def test_buy_skips_when_gap_check_keeps_erroring(self):
        retry, alert = self._gap_raises(3)
        retry.apply_async.assert_not_called()
        assert "Gap Check Failed" in alert.call_args[0][0]

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


class TestAddStopsWholePosition:
    """2026-10-27 (audit finding 5): an ADD used to overwrite entry_price with
    the add's fill and stop only the added shares."""

    def _run(self, prior, pos_qty=15, avg=105.0, new_stop=95.0, filled=5):
        client = MagicMock()
        client.get_open_position.return_value = MagicMock(qty=pos_qty, avg_entry_price=avg)
        with patch("db.get_position_stop", return_value=prior), \
             patch("ibkr.get_client", return_value=client), \
             patch("tasks.manage_cancel_stop_orders_for") as cancel:
            result = execution._whole_position_stop("NVDA", new_stop, filled)
        return result, cancel

    def test_add_stops_the_whole_position_at_the_tighter_stop(self):
        (stop, add), cancel = self._run({"stop_price": 98.0})
        assert stop == 98.0            # old stop was higher: never loosen
        assert add == (105.0, 15.0)    # avg cost, full quantity
        cancel.assert_called_once_with("NVDA")

    def test_add_raises_the_stop_when_the_new_one_is_higher(self):
        (stop, _), _ = self._run({"stop_price": 90.0}, new_stop=97.0)
        assert stop == 97.0

    def test_fresh_position_is_unchanged(self):
        (stop, add), cancel = self._run(None)
        assert (stop, add) == (95.0, None)
        cancel.assert_not_called()

    def test_broker_lookup_failure_falls_back_to_added_shares_only(self):
        with patch("db.get_position_stop", return_value={"stop_price": 98.0}), \
             patch("ibkr.get_client", side_effect=RuntimeError("down")), \
             patch("tasks.manage_cancel_stop_orders_for") as cancel:
            assert execution._whole_position_stop("NVDA", 95.0, 5) == (95.0, None)
        cancel.assert_not_called()


class TestStopRestoredAfterFailedSell:
    """2026-10-27 (audit finding 8): a TRIM/EXIT whose order throws must not
    leave the position without a broker stop."""

    def test_stop_is_restored_for_the_held_quantity(self):
        client = MagicMock()
        client.get_open_position.return_value = MagicMock(qty=3)
        with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
             patch("db.set_stop_broker_order_id") as set_id, \
             patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
             patch("alerts._send") as send:
            execution._restore_stop_after_failed_sell("NVDA")
        place.assert_called_once_with("NVDA", 3.0, 90.0)
        set_id.assert_called_once_with("NVDA", 5)
        send.assert_not_called()

    def test_alert_when_restore_fails(self):
        with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
             patch("ibkr.get_client", side_effect=RuntimeError("down")), \
             patch("tasks._queue_reprotect") as queued, \
             patch("alerts._send") as send:
            execution._restore_stop_after_failed_sell("NVDA")
        assert "UNPROTECTED" in send.call_args[0][0]
        queued.assert_called_once()  # A0653-5: retried, not left to a manual step


def test_log_trade_outcome_skips_a_second_row_for_the_same_approval():
    import db
    cur = MagicMock()
    cur.fetchone.return_value = (1,)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("yfinance.Ticker"):
        db.log_trade_outcome(approval_id=62, ticker="NVDA", action="OPEN", conviction=7,
                             bucket="Accelerator", entry_price=100.0)
    assert cur.execute.call_count == 1  # only the existence check, no INSERT
    conn.rollback.assert_called_once()


def test_close_trade_outcome_closes_every_open_row_for_the_ticker():
    """Professor C5: OPEN + ADD then EXIT used to close only the newest row."""
    import db
    cur = MagicMock()
    cur.fetchall.return_value = [(11, 100.0, None), (10, 90.0, None)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("yfinance.Ticker"):
        assert db.close_trade_outcome("NVDA", exit_price=110.0) is True
    updated_ids = [c[0][1][-1] for c in cur.execute.call_args_list if c[0][0].startswith("UPDATE")]
    assert updated_ids == [11, 10]


def test_backfill_never_overwrites_an_entry_time_snapshot():
    """Professor C6: the UPDATE only applies to rows with no catalyst snapshot."""
    import db
    cur = MagicMock()
    cur.fetchone.return_value = (5,)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.backfill_catalyst_outcome("NVDA", 7, "Accelerator", 100.0, None, "earnings", "2026-11-19",
                                     False, 10.0, -5.0, "thesis", [], 0)
    sql = cur.execute.call_args_list[-1][0][0]
    assert "catalyst_event IS NULL" in sql and "notes=CONCAT_WS" in sql


def test_log_signal_writes_env_model_and_provenance():
    """Professor M19: signals carries trading_env, model and provenance."""
    import db
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_signal("NVDA", "HOLD", 0.5, "r", model="m", provenance={"code_sha": "x"})
    sql, params = cur.execute.call_args[0]
    assert "trading_env, model, provenance" in sql
    assert params[4] == db.TRADING_ENV and params[5] == "m"


def test_manual_sell_not_sent_when_market_closed():
    with patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order") as order, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("db.get_state", return_value=None), \
         patch("market_data.is_market_hours", return_value=False):
        client.return_value.get_open_position.return_value = MagicMock(qty=5)
        r = execution.execute_manual("NVDA", "SELL", 5, "test")
    assert r["success"] is False and "Market is closed" in r["message"]
    order.assert_not_called(); cancel.assert_not_called()


def test_exit_not_sent_while_a_stop_is_still_resting():
    """Limitations audit 22:53 A2253-1: selling on top of a stop that could
    not be cancelled can short the account after an EXIT."""
    approval = _approval(action="EXIT", shares_initial=100)
    resting = [{"order_id": 4, "ticker": approval["ticker"], "action": "SELL",
                "order_type": "STP", "qty": 100.0, "stop_price": 90.0, "status": "Submitted"}]
    with patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order") as order, \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=RuntimeError("stuck")), \
         patch("ibkr.get_open_orders", return_value=resting), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("alerts._send"):
        client.return_value.get_open_position.return_value = MagicMock(qty=100)
        assert execution.execute_approved(approval) is False
    order.assert_not_called()
    assert "stop still resting" in execution.last_outcome_reason()


def test_restore_covers_only_shares_without_a_resting_stop():
    resting = [{"order_id": 4, "ticker": "AMD", "action": "SELL", "order_type": "STP",
                "qty": 60.0, "stop_price": 90.0, "status": "Submitted"}]
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("ibkr.get_client") as client, \
         patch("ibkr.get_open_orders", return_value=resting), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        client.return_value.get_open_position.return_value = MagicMock(qty=100)
        execution._restore_stop_after_failed_sell("AMD")
    place.assert_called_once_with("AMD", 40.0, 90.0)


def test_restore_places_nothing_when_resting_stops_cannot_be_read():
    """A0053-2: unknown coverage used to count as 0 and stack a full stop."""
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("ibkr.get_client") as client, \
         patch("ibkr.get_open_orders", side_effect=RuntimeError("down")), \
         patch("tasks.manage_stop_place") as place, \
         patch("alerts._send") as send:
        client.return_value.get_open_position.return_value = MagicMock(qty=100)
        execution._restore_stop_after_failed_sell("AMD")
    place.assert_not_called()
    assert "CHECK STOPS" in send.call_args[0][0]


def test_skipped_exit_restores_stops_the_cancel_already_took():
    """A0053-2: two stops, one cancelled and one stuck; the skipped EXIT
    puts the cancelled shares back under a stop."""
    approval = _approval(action="EXIT", shares_initial=100)
    stuck = [{"order_id": 4, "ticker": approval["ticker"], "action": "SELL",
              "order_type": "STP", "qty": 60.0, "stop_price": 90.0, "status": "Submitted"}]
    with patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order") as order, \
         patch("tasks.manage_cancel_stop_orders_for", return_value=1), \
         patch("ibkr.get_open_orders", return_value=stuck), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("alerts._send"):
        client.return_value.get_open_position.return_value = MagicMock(qty=100)
        assert execution.execute_approved(approval) is False
    order.assert_not_called()
    place.assert_called_once_with(approval["ticker"], 40.0, 90.0)


def test_unknown_stop_result_is_not_retried():
    """Professor C24: a lost reply may mean the stop is resting; a retry
    would stack a second one."""
    import db
    import inspect
    import tasks
    approval = _approval(action="OPEN", shares_initial=10)
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 10, "fill_price": 100.0,
                                                        "order_id": 1, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("tasks.manage_stop_place", side_effect=tasks.OrdersCallUnknown("x")) as place, \
         patch("alerts._send") as send:
        client.return_value.get_open_position.side_effect = Exception("No open position")
        client.return_value.get_all_positions.return_value = []
        execution.execute_approved(approval)
    assert place.call_count == 1
    assert any("STOP STATUS UNKNOWN" in c[0][0] for c in send.call_args_list)


def test_expired_stop_request_is_queued_not_retried():
    """A0453-3: a request the busy worker never picked up goes to the
    reprotect queue instead of three blocking retries."""
    import db
    import inspect
    import tasks
    approval = _approval(action="OPEN", shares_initial=10)
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 10, "fill_price": 100.0,
                                                        "order_id": 1, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("tasks.manage_stop_place", side_effect=tasks.OrdersCallExpired("x")) as place, \
         patch("tasks._queue_reprotect") as queued, \
         patch("alerts._send") as send:
        client.return_value.get_open_position.side_effect = Exception("No open position")
        client.return_value.get_all_positions.return_value = []
        execution.execute_approved(approval)
    assert place.call_count == 1
    queued.assert_called_once()
    assert any("STOP DELAYED" in c[0][0] for c in send.call_args_list)


def test_refused_stop_is_not_retried():
    """Professor run 11: a stop the broker layer refuses (short/flat, or
    position lag after the fill) isn't retried inline; it is queued."""
    import db
    import inspect
    import tasks
    approval = _approval(action="OPEN", shares_initial=10)
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 10, "fill_price": 100.0,
                                                        "order_id": 1, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("tasks.manage_stop_place", side_effect=__import__("ibkr").SellExceedsPosition("NVDA: refused")) as place, \
         patch("tasks._queue_reprotect") as queued, \
         patch("alerts._send") as send:
        client.return_value.get_open_position.side_effect = Exception("No open position")
        client.return_value.get_all_positions.return_value = []
        execution.execute_approved(approval)
    assert place.call_count == 1
    queued.assert_called_once()
    assert any("STOP REFUSED" in c[0][0] for c in send.call_args_list)
    assert not any("STOP DELAYED" in c[0][0] for c in send.call_args_list)  # M81: one alert


def test_lock_lost_stop_is_queued_and_failed_stop_queued_too():
    """m39: a lapsed ticker lock goes to the queue at once; a stop that fails
    all three attempts is queued as well as alerted."""
    import db
    import inspect
    import stop_lock
    approval = _approval(action="OPEN", shares_initial=10)
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    for err, calls in ((stop_lock.LockLost("NVDA"), 1), (RuntimeError("boom"), 3)):
        with patch.multiple("db", **db_fns), \
             patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
             patch("ibkr.place_market_order", return_value={"filled_qty": 10, "fill_price": 100.0,
                                                            "order_id": 1, "status": "Filled"}), \
             patch("ibkr.get_latest_price", return_value=100.0), \
             patch("market_data.get_priced_quote", return_value=None), \
             patch("tasks.manage_stop_place", side_effect=err) as place, \
             patch("tasks._queue_reprotect") as queued, \
             patch("time.sleep"), \
             patch("alerts._send"):
            client.return_value.get_open_position.side_effect = Exception("No open position")
            client.return_value.get_all_positions.return_value = []
            execution.execute_approved(approval)
        assert place.call_count == calls
        queued.assert_called_once()


def test_buy_sends_the_share_count_the_room_check_allows():
    """Pre-pull review 10-29 #3: a cut from the execution-time cash check must
    reach the order and the stop, and a skip must stop the order."""
    approval = _approval(action="ADD", shares_initial=5, price_at_sizing=108.0)
    fresh = {"price": 110.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def run(room):
        with patch("ibkr.get_ib"), patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.get_priced_quote", return_value=fresh), \
             patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("config.USE_BRACKET_ORDERS", False), \
             patch("execution._buy_room_at_execution", return_value=room), \
             patch("db.log_trade"), patch("db.log_order", return_value=1), \
             patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
             patch("db.log_trade_outcome"), patch("alerts._send"), \
             patch("alerts.send_trade_alert"):
            mock_rm_cls.return_value.is_halted.return_value = False
            mock_order.return_value = {"order_id": 1, "fill_price": 110.0, "filled_qty": room[0]}
            return execution.execute_approved(approval), mock_order

    result, order = run((2, None))
    assert result is True and order.call_args.args[2] == 2
    result, order = run((0, "not enough cash at execution under the per-trade cash limits"))
    assert result is False and not order.called


def test_buy_reservation_kept_when_the_order_call_raises():
    """Review 4 #6: place_market_order can raise after the order was sent
    (a disconnect while waiting for the fill), so its cash stays reserved."""
    approval = _approval(action="ADD", shares_initial=5, price_at_sizing=108.0)
    fresh = {"price": 110.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}
    with patch("ibkr.get_ib"), patch("ibkr.get_client"), \
         patch("risk_manager.RiskManager") as mock_rm_cls, \
         patch("ibkr.place_market_order", side_effect=ConnectionError("lost")), \
         patch("market_data.get_priced_quote", return_value=fresh), \
         patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("config.USE_BRACKET_ORDERS", False), \
         patch("execution._buy_room_at_execution", return_value=(2, None)), \
         patch("execution._release_buy") as release, \
         patch("db.log_trade"), patch("db.log_order", return_value=1), \
         patch("alerts._send"), patch("alerts.send_trade_alert"):
        mock_rm_cls.return_value.is_halted.return_value = False
        try:
            execution.execute_approved(approval)
        except ConnectionError:
            pass
    release.assert_not_called()


def test_entry_stop_keeps_its_distance_below_a_lower_fill():
    """Review 5 #1: sized at a $106 pre-market print with a $101 stop, filled at
    $99.50: the stop must not sit above the fill."""
    from execution import _stop_anchored_to_fill as anchor
    assert anchor("X", 101.0, 106.0, 99.5) == round(99.5 * 101 / 106, 2) < 99.5
    assert anchor("X", 95.0, 100.0, 100.2) == 95.0        # fill at or above sizing: unchanged
    assert anchor("X", 95.0, 100.0, 97.0) == 95.0         # still safely below the fill: kept
    assert anchor("X", 99.7, 100.0, 99.6) < 99.6          # tight structural stop above the fill
    assert anchor("X", 98.8, 100.0, 99.6) <= 98.8         # never tightened
    assert anchor("X", 95.0, None, 90.0) == 95.0          # unknown sizing price: unchanged


def test_manual_buy_gets_a_fallback_stop_when_atr_fails():
    """Professor M107: a failed calculate_stop left a manual buy with no stop."""
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "fill_price": 100.0, "filled_qty": 2}), \
         patch("db.log_trade"), patch("db.upsert_position_stop") as upsert, \
         patch("db.set_stop_broker_order_id"), patch("tasks.manage_stop_place", return_value={"order_id": 9}), \
         patch("alerts._send") as alert:
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.side_effect = RuntimeError("no bars")
        r = execution.execute_manual("ZZZ", "BUY", 2, "test")
    assert r["success"] and r["stop_price"] == 92.0
    assert upsert.call_args.kwargs["stop_price"] == 92.0 and alert.called


def _manual_buy(fill_price, quote=100.0):
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": quote}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "fill_price": fill_price, "filled_qty": 2}), \
         patch("ibkr.get_latest_price", return_value=None), \
         patch("db.log_trade"), patch("db.upsert_position_stop") as upsert, \
         patch("db.set_stop_broker_order_id"), patch("tasks.manage_stop_place", return_value={"order_id": 9}), \
         patch("tasks.open_manual_action") as action, patch("alerts._send") as alert:
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.side_effect = RuntimeError("no bars")
        try:
            r = execution.execute_manual("ZZZ", "BUY", 2, "test")
        except Exception:
            r = None
    return r, upsert, action, alert


def test_manual_buy_nan_fill_alerts_unprotected_not_nan_stop():
    r, upsert, action, alert = _manual_buy(float("nan"))
    assert not upsert.called and action.called and alert.called


def test_manual_buy_fallback_stop_never_above_the_quote():
    r, upsert, action, alert = _manual_buy(120.0, quote=100.0)    # stale fill 20% above the quote
    assert upsert.call_args.kwargs["stop_price"] == 92.0          # 8% under the quote, not 110.4


def test_manual_orders_get_an_orders_row():
    """Professor M108: an unfilled manual BUY wrote no orders row, so a late
    fill was never reconciled; a part fill stays 'partial' for the rest."""
    def run(result):
        with patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
             patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
             patch("ibkr.place_market_order", return_value=result), \
             patch("db.log_trade"), patch("db.log_order", return_value=5) as log_order, \
             patch("db.update_order_fill") as fill, patch("db.upsert_position_stop"), \
             patch("db.set_stop_broker_order_id"), \
             patch("tasks.manage_stop_place", return_value={"order_id": 9}), patch("alerts._send"):
            rm_cls.return_value.is_halted.return_value = False
            rm_cls.return_value.calculate_stop.return_value = (92.0, "ATR")
            r = execution.execute_manual("ZZZ", "BUY", 3, "test")
        return r, log_order, fill
    r, log_order, fill = run({"order_id": 77, "filled_qty": 0, "status": "Submitted"})
    assert not r["success"] and "reconciled automatically" in r["message"]
    kw = log_order.call_args.kwargs
    assert kw["status"] == "unconfirmed" and kw["intent"] == "MANUAL_ADD" and kw["broker_order_id"] == 77
    assert kw.get("decision_price") is None and not fill.called
    r, log_order, fill = run({"order_id": 78, "fill_price": 100.0, "filled_qty": 2})
    assert r["success"] and log_order.call_args.kwargs["status"] == "submitted"
    fill.assert_called_once_with(5, 100.0, 2.0, "partial")


def test_manual_buy_intent_says_open_when_nothing_held():
    with patch("ibkr.get_client") as client, patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 77, "filled_qty": 0}), \
         patch("db.log_order", return_value=5) as log_order:
        client.return_value.get_open_position.side_effect = Exception("No open position in ZZZ")
        rm_cls.return_value.is_halted.return_value = False
        execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert log_order.call_args.kwargs["intent"] == "MANUAL_OPEN"


def test_log_trade_stores_null_not_zero_and_the_price_source():
    """Professor M109: a 0.0 fallback price reached trades as if real."""
    import db
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._writer_conn", return_value=conn), patch("db._put_conn"):
        db.log_trade("ZZZ", "BUY", 2, 0.0, "test", price_source="quote")
        args = cur.execute.call_args.args[1]
        assert args[4] is None and args[6] == "none"
        db.log_trade("ZZZ", "BUY", 2, 101.5, "test", price_source="fill")
        args = cur.execute.call_args.args[1]
        assert args[4] == 101.5 and args[6] == "fill"


def test_manual_trade_records_quote_as_price_source():
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "filled_qty": 2}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("db.log_trade") as log_trade, patch("db.log_order", return_value=5), \
         patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}), patch("alerts._send"):
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.return_value = (92.0, "ATR")
        execution.execute_manual("ZZZ", "BUY", 2, "test")
    assert log_trade.call_args.kwargs["price_source"] == "quote"


def test_null_trade_price_reads_back_and_prints():
    """Professor M116: a NULL price (73d8150) broke float() in
    get_recent_trades, which blanked every trade on the dashboard and in chat."""
    import db
    import chat_assistant
    cur = MagicMock()
    cur.fetchall.return_value = [("ZZZ", "BUY", 2, None, "2026-10-30", "test", "none")]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        rows = db.get_recent_trades(5)
    assert rows[0]["price"] is None and rows[0]["price_source"] == "none"
    with patch("db.get_recent_trades", return_value=rows):
        assert "price unknown" in chat_assistant._tool_get_trade_history({})


def test_manual_add_keeps_the_stop_and_protects_the_whole_position():
    """Professor M115: a manual ADD overwrote the stop record with a fresh
    stop and placed it for the new shares only."""
    with patch("ibkr.get_client") as client, patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "fill_price": 100.0, "filled_qty": 2}), \
         patch("db.log_trade"), patch("db.log_order", return_value=5), patch("db.update_order_fill"), \
         patch("db.get_position_stop", return_value={"stop_price": 95.0}), \
         patch("db.upsert_position_stop") as upsert, patch("tasks.manage_stop_place") as place, \
         patch("tasks._reprotect_after_late_sell", return_value=True) as reprotect, patch("alerts._send"):
        client.return_value.get_open_position.return_value = MagicMock(qty=5)
        rm_cls.return_value.is_halted.return_value = False
        r = execution.execute_manual("ZZZ", "BUY", 2, "test")
    assert r["success"] and r["stop_price"] == 95.0
    assert not upsert.called and not place.called and reprotect.called


def test_manual_fill_without_a_price_still_marks_the_order_filled():
    """Professor run 17: a fill confirmed without a price left the orders row
    'submitted'."""
    with patch("ibkr.get_client") as client, patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "filled_qty": 2}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("db.log_trade"), patch("db.log_order", return_value=5), \
         patch("db.update_order_fill") as fill, patch("db.upsert_position_stop"), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}), patch("alerts._send"):
        client.return_value.get_open_position.side_effect = Exception("No open position in ZZZ")
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.return_value = (92.0, "ATR")
        execution.execute_manual("ZZZ", "BUY", 2, "test")
    fill.assert_called_once_with(5, None, 2.0, "filled")
