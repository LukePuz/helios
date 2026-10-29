"""Professor C23 part 2 (2026-10-29): each stop-sending caller under each
broker failure. Unknown must never stack a second stop; Expired, LockLost and
plain errors must end in the re-protect queue or a placed stop, never silence."""
from unittest.mock import MagicMock, patch

import pytest

import execution
import risk_manager
import stop_lock
import tasks


def _errors():
    import ibkr
    return [tasks.OrdersCallUnknown("x"), tasks.OrdersCallExpired("x"),
            stop_lock.LockLost("NVDA"), ibkr.SellExceedsPosition("short"), RuntimeError("boom")]


@pytest.mark.parametrize("err", _errors(), ids=lambda e: type(e).__name__)
def test_restore_after_failed_sell_queues_on_every_failure(err):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=3)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_stop_place", side_effect=err), \
         patch("tasks._queue_reprotect") as queued, \
         patch("alerts._send") as send:
        execution._restore_stop_after_failed_sell("NVDA")
    queued.assert_called_once()
    assert "UNPROTECTED" in send.call_args[0][0]


def _trail_conversion(trail_effect, place_effect=({"order_id": 3},)):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=126.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": False, "trailing_high": 130.0}
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_place_trailing_stop", side_effect=trail_effect), \
         patch("tasks.manage_stop_place", side_effect=place_effect) as place, \
         patch("tasks._queue_reprotect") as queued, \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.set_trail_params"):
        alerts = rm.check_trailing_stops()
    return place, queued, send, alerts


def test_trail_unknown_places_no_fixed_stop_and_alerts():
    place, queued, send, _ = _trail_conversion(tasks.OrdersCallUnknown("x"))
    place.assert_not_called()
    assert any("STOP STATUS UNKNOWN" in c[0][0] for c in send.call_args_list)


@pytest.mark.parametrize("err", _errors()[1:], ids=lambda e: type(e).__name__)
def test_trail_failure_restores_the_fixed_stop(err):
    place, queued, _, _ = _trail_conversion(err)
    place.assert_called_once_with("NVDA", 7.0, 112.0)
    queued.assert_not_called()


@pytest.mark.parametrize("err", _errors(), ids=lambda e: type(e).__name__)
def test_trail_failure_with_failed_restore_queues_and_flags(err):
    place, queued, _, alerts = _trail_conversion(tasks.OrdersCallExpired("x"), err)
    queued.assert_called_once()
    assert [a for a in alerts if a["action"] == "FIX_STOP"]
