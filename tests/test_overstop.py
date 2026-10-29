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
