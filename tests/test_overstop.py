"""Resting stops for more shares than held get rebuilt (professor M54)."""
from unittest.mock import MagicMock, patch

import tasks


def _orders(*qtys):
    return [{"order_id": i, "ticker": "AMD", "action": "SELL", "order_type": "STP",
             "qty": q, "stop_price": 150.0, "status": "Submitted"} for i, q in enumerate(qtys)]


def test_fix_overstop_rebuilds_one_stop_for_the_held_qty():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), []]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    cancel.assert_called_once_with("AMD")
    place.assert_called_once_with("AMD", 10.0, 150.0)
    assert "placed a stop for 10" in msg


def test_fix_overstop_does_nothing_when_cleared_on_recheck():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10)), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    assert "fine on re-check" in msg


def test_fix_overstop_without_recorded_stop_only_alerts():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10, 5)), \
         patch("db.get_position_stop", return_value=None), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    assert "fix the stops in TWS" in msg


def test_stuck_stop_is_reported_not_papered_over():
    """A2253-6: a stop that won't cancel must not get a full-size stop added."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), _orders(10, 10)]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place") as place:
        msg = tasks._fix_overstop("AMD")
    place.assert_not_called()
    assert "won't cancel" in msg


def test_partly_covered_position_gets_the_missing_shares_stopped():
    """A2253-4: a stop for 4 of 10 shares used to pass forever."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(4)), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    place.assert_called_once_with("AMD", 6.0, 150.0)
    assert "only 4 of 10" in msg


def test_naked_position_gets_a_stop_at_the_recorded_level():
    """A0053-2: no stop at all used to be alert-only."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    place.assert_called_once_with("AMD", 10.0, 150.0)
    assert msg.startswith("🔧")


def test_no_stop_placed_at_or_above_the_last_price():
    """A stop above the market would sell at once; alert instead."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    for last in (140.0, None):
        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
             patch("ibkr.get_latest_price", return_value=last), \
             patch("tasks.manage_stop_place") as place:
            msg = tasks._fix_overstop("AMD")
        place.assert_not_called()
        assert "nothing was placed" in msg


class TestBackstopSellReads:
    """Professor M61: read errors, shorts and stuck stops in the backstop."""

    _alert = {"current_price": 140.0, "stop_price": 150.0}

    def _client(self, qty):
        client = MagicMock()
        client.get_all_positions.return_value = [] if qty is None else [MagicMock(symbol="AMD", qty=qty)]
        return client

    def test_read_error_keeps_the_record_and_retries(self):
        with patch("ibkr.get_client", side_effect=RuntimeError("down")), \
             patch("db.delete_position_stop") as delete, \
             patch("tasks._schedule_backstop_retry") as retry, \
             patch("ibkr.place_market_order") as order:
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        delete.assert_not_called(); order.assert_not_called(); retry.assert_called_once()

    def test_confirmed_flat_drops_the_record(self):
        with patch("ibkr.get_client", return_value=self._client(None)), \
             patch("db.delete_position_stop") as delete, \
             patch("ibkr.place_market_order") as order:
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        delete.assert_called_once_with("AMD"); order.assert_not_called()

    def test_short_is_not_sold(self):
        with patch("ibkr.get_client", return_value=self._client(-10)), \
             patch("ibkr.place_market_order") as order, \
             patch("alerts._send") as send:
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        order.assert_not_called()
        assert "SHORT" in send.call_args[0][0]

    def test_stuck_stop_shares_are_not_sold_again(self):
        with patch("ibkr.get_client", return_value=self._client(10)), \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=_orders(4)), \
             patch("ibkr.place_market_order", return_value={"filled_qty": 6, "fill_price": 140.0}) as order, \
             patch("db.log_trade"), patch("db.delete_position_stop") as delete, \
             patch("db.close_trade_outcome") as close, patch("alerts._send"):
            assert tasks._execute_backstop_sell("AMD", self._alert) is True
        order.assert_called_once_with("AMD", "SELL", 6.0)
        delete.assert_not_called(); close.assert_not_called()


def test_backstop_sale_gets_an_orders_row_and_a_part_fill_waits_for_reconcile():
    """Professor (review of c13548c): a part-filled backstop had no orders row,
    so its remainder was never reconciled."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    for filled, status, closes in ((6, "partial", False), (10, "filled", True)):
        with patch("ibkr.get_client", return_value=client), \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 77, "filled_qty": filled, "fill_price": 140.0}), \
             patch("db.log_trade"), patch("db.delete_position_stop"), \
             patch("db.log_order", return_value=5) as log_order, \
             patch("db.update_order_fill") as upd, \
             patch("db.close_trade_outcome") as close, patch("alerts._send"):
            assert tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert) is True
        assert log_order.call_args.kwargs["intent"] == "BACKSTOP"
        assert log_order.call_args.kwargs["broker_order_id"] == 77
        upd.assert_called_once_with(5, 140.0, filled, status)
        assert close.called is closes


def test_backstop_does_not_sell_shares_a_working_sell_already_covers():
    """Review of 16009ad: a second backstop while the first one's remainder
    was still working would short once both filled."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    working = [{"order_id": 77, "ticker": "AMD", "action": "SELL", "order_type": "MKT",
                "qty": 10.0, "stop_price": None, "status": "Submitted", "remaining": 4.0}]
    for open_orders, sold in ((working, 6.0), ([{**working[0], "remaining": 10.0}], None)):
        with patch("ibkr.get_client", return_value=client), \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=open_orders), \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 78, "filled_qty": 6, "fill_price": 140.0}) as order, \
             patch("db.log_trade"), patch("db.delete_position_stop"), patch("db.log_order", return_value=5), \
             patch("db.update_order_fill"), patch("db.close_trade_outcome"), patch("alerts._send"):
            tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert)
        if sold is None:
            order.assert_not_called()
        else:
            order.assert_called_once_with("AMD", "SELL", sold)


def test_cancel_failure_in_overcover_returns_a_line_not_a_crash():
    """A0253-7: label was unbound when the cancel raised."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10, 10)), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=tasks.OrdersCallUnknown("x")):
        msg = tasks._fix_overstop("AMD")
    assert "fix failed" in msg


def test_inactive_stop_is_not_protection_in_reconcile():
    """A0253-6: section 3 used raw status, so an Inactive stop hid a naked
    position."""
    from datetime import datetime, timedelta, timezone
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    inactive = [{**_orders(10)[0], "status": "Inactive"}]
    t = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    state = {"no_stop_since_AMD": t, "no_stop_seen_AMD": t}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=inactive), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._fix_overstop_safe", return_value="🚨 AMD: x") as fix, \
         patch("alerts._send"):
        tasks.reconcile_stop_orders.run()
    fix.assert_called_once_with("AMD")


def test_slow_cancel_gets_a_second_look():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    pending = [{**o, "status": "PendingCancel"} for o in _orders(10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), pending, []]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        tasks._fix_overstop("AMD")
    place.assert_called_once_with("AMD", 10.0, 150.0)


def test_short_position_gets_no_sell_stop():
    """Professor M63: a SELL stop on a short adds to it."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=-10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_stop_place") as place:
        msg = tasks._fix_overstop("AMD")
    place.assert_not_called()
    assert "SHORT" in msg


def test_queued_reprotect_places_no_stop_on_a_short():
    """Professor M63 residual: the queue path used abs(qty)."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=-5)]
    with patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place") as place, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("alerts._send") as send:
        assert tasks._reprotect_after_late_sell("AMD", {"intent": "RESTORE"}, None,
                                                from_queue=True) is None
    place.assert_not_called(); cancel.assert_not_called()
    assert "SHORT" in send.call_args[0][0]
