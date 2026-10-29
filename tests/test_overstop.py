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
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    place.assert_called_once_with("AMD", 6.0, 150.0)
    assert "only 4 of 10" in msg
