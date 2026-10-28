"""Professor run 7 (2026-10-28): C15 no profit-target sale outside RTH; C16 a
failed TRAIL conversion restores the fixed stop or alerts."""
from unittest.mock import MagicMock, patch

import risk_manager


def test_partial_sale_deferred_outside_rth():
    with patch("market_data.is_market_hours", return_value=False), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("ibkr.place_market_order") as sell:
        assert object.__new__(risk_manager.RiskManager)._partial_sell_and_resize_stop(
            "NVDA", 100, 30, 100.0) is None
    cancel.assert_not_called(); sell.assert_not_called()


def _failed_conversion(place_side_effect):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=126.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    # trailing_high above current, so the old fallback condition is false
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": False, "trailing_high": 130.0}
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_place_trailing_stop", side_effect=RuntimeError("broker")), \
         patch("tasks.manage_stop_place", side_effect=place_side_effect) as place, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.set_trail_params"):
        alerts = rm.check_trailing_stops()
    return place, alerts


def test_failed_trail_conversion_restores_the_fixed_stop():
    place, alerts = _failed_conversion([{"order_id": 3}])
    place.assert_called_once_with("NVDA", 7.0, 112.0)
    assert not [a for a in alerts if a["action"] == "FIX_STOP"]


def test_failed_restore_raises_fix_stop():
    _, alerts = _failed_conversion(RuntimeError("broker down"))
    assert [a for a in alerts if a["action"] == "FIX_STOP"]
