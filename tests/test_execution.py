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
        approval = _approval(action="ADD")

        with patch("ibkr.get_ib"), \
             patch("ibkr.get_client"), \
             patch("risk_manager.RiskManager") as mock_rm_cls, \
             patch("ibkr.place_market_order") as mock_order, \
             patch("market_data.is_market_hours", return_value=False), \
             patch("alerts._send"):
            mock_rm_cls.return_value.is_halted.return_value = False
            result = execution.execute_approved(approval)

        assert result is False
        mock_order.assert_not_called()

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
