"""Profit-target partial sales must keep the broker stop in step (audit 2026-10-27
finding 6). Before the fix the broker STP stayed at the old price for the
original quantity after a partial sale, so it could oversell into a short."""
from unittest.mock import MagicMock, patch

import risk_manager


def _rm():
    return object.__new__(risk_manager.RiskManager)


def _client(remaining):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=remaining)
    return client


def test_stop_is_cancelled_then_replaced_for_remaining_shares_at_breakeven():
    calls = []
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id") as set_id, \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=lambda t: calls.append("cancel")), \
         patch("ibkr.place_market_order",
               side_effect=lambda *a: calls.append("sell") or {"filled_qty": 30, "order_id": 7}), \
         patch("ibkr.get_client", return_value=_client(70)), \
         patch("tasks.manage_stop_place",
               side_effect=lambda *a: calls.append(("stop",) + a) or {"order_id": 8}), \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0)

    assert result == {"filled_qty": 30, "order_id": 7, "new_stop": 100.0}
    assert calls == ["cancel", "sell", ("stop", "NVDA", 70.0, 100.0)]
    set_id.assert_called_once_with("NVDA", 8)
    send.assert_not_called()


def test_failed_sale_restores_the_original_stop_for_the_full_position():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", side_effect=RuntimeError("rejected")), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0)

    assert result is None
    place.assert_called_once_with("NVDA", 100, 90.0)
    send.assert_not_called()


def test_no_sale_when_the_old_stop_cannot_be_cancelled():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=RuntimeError("timeout")), \
         patch("ibkr.place_market_order") as sell:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0)

    assert result is None
    sell.assert_not_called()


def test_alert_when_the_new_stop_cannot_be_placed():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": 30}), \
         patch("ibkr.get_client", return_value=_client(70)), \
         patch("tasks.manage_stop_place", side_effect=RuntimeError("down")), \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0)

    assert result == {"filled_qty": 30, "new_stop": 100.0}
    assert "UNPROTECTED" in send.call_args[0][0]


def test_unconfirmed_fill_is_not_treated_as_a_sale():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", return_value={"order_id": 7, "filled_qty": 0}), \
         patch("tasks.manage_stop_place") as place, \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0)
    assert result is None
    place.assert_not_called()
    assert "UNCONFIRMED" in send.call_args[0][0]


def test_small_position_moves_stop_to_breakeven_without_selling():
    """Audit 2026-10-28 07:00 finding 3: 1 share x 25% rounds to 0, so the
    breakeven move must happen without a sale."""
    calls = []
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("db.mark_profit_target_taken") as taken, \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=lambda t: calls.append("cancel")), \
         patch("tasks.manage_stop_place", side_effect=lambda *a: calls.append(("stop",) + a) or {"order_id": 5}), \
         patch("ibkr.place_market_order") as sell:
        assert _rm()._breakeven_without_sale("NVDA", 1, 100.0) is True
    sell.assert_not_called()
    assert calls == ["cancel", ("stop", "NVDA", 1, 100.0)]
    taken.assert_called_once_with("NVDA", 100.0)


def test_breakeven_move_skipped_when_stop_already_higher():
    with patch("db.get_position_stop", return_value={"stop_price": 105.0}), \
         patch("db.mark_profit_target_taken") as taken, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        assert _rm()._breakeven_without_sale("NVDA", 1, 100.0) is True
    cancel.assert_not_called()
    taken.assert_called_once_with("NVDA", 105.0)


def test_failed_breakeven_stop_restores_old_stop_and_keeps_target_open():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("db.mark_profit_target_taken") as taken, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", side_effect=[RuntimeError("down"), {"order_id": 6}]) as place, \
         patch("alerts._send") as send:
        assert _rm()._breakeven_without_sale("NVDA", 2, 100.0) is False
    assert place.call_args_list[-1][0] == ("NVDA", 2, 90.0)
    taken.assert_not_called()
    send.assert_not_called()
