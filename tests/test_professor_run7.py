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
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
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


def test_update_trailing_stop_writes_the_flag():
    import db
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.update_trailing_stop("NVDA", 110.0, 120.0, native_trail=False)
    sql, params = cur.execute.call_args.args
    assert "trailing_active=%s" in sql and params[2] is False


def test_partial_sale_never_places_a_stop_above_the_price(monkeypatch):
    import market_data
    monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=70)
    # recorded stop 130 (stale fixed-STP record) is above the 126 trigger price
    with patch("db.get_position_stop", return_value={"stop_price": 130.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": 30, "order_id": 7}), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place", return_value={"order_id": 8}) as place, \
         patch("db.log_trade"), patch("db.log_order", return_value=1), patch("db.update_order_fill"), \
         patch("alerts._send"):
        result = object.__new__(risk_manager.RiskManager)._partial_sell_and_resize_stop(
            "NVDA", 100, 30, 100.0, decision_price=126.0)
    place.assert_called_once_with("NVDA", 70.0, 100.0)
    assert result["new_stop"] == 100.0


def test_failed_cancel_does_not_stack_a_restored_stop():
    """Professor C19: the restore assumed the cancel worked; if the cancel
    itself failed the old stop is still resting."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=126.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": False, "trailing_high": 130.0}
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=RuntimeError("cancel timeout")), \
         patch("tasks.manage_place_trailing_stop") as trail, \
         patch("tasks.manage_stop_place") as place:
        rm.check_trailing_stops()
    trail.assert_not_called()
    place.assert_not_called()


def test_conversion_skipped_when_stop_moved_while_waiting():
    """Professor C18: state read before the lock must be re-checked inside it."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=126.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": False, "trailing_high": 130.0}
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value={**rec, "stop_price": 118.0}), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        rm.check_trailing_stops()
    cancel.assert_not_called()


def test_trail_does_not_arm_on_an_after_hours_price():
    """Audit A1031-0053-8: a +26% after-hours print must not arm the trail."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=126.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": False}
    with patch("market_data.is_market_hours", return_value=False), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("tasks.manage_place_trailing_stop") as trail, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        assert rm.check_trailing_stops() == []
    trail.assert_not_called(); cancel.assert_not_called()


def test_a_breached_stop_still_alerts_after_hours():
    """The breach check (which feeds the backstop for an unprotected position) still runs."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=100.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator", "entry_price": 120.0}
    with patch("market_data.is_market_hours", return_value=False), \
         patch("db.get_all_position_stops", return_value=[rec]):
        alerts = rm.check_trailing_stops()
    assert [a["action"] for a in alerts] == ["EXIT"]


def _rth_rm(current=126.0, bucket="Accelerator"):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=current)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    return rm, {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": bucket,
                "entry_price": 100.0, "trailing_active": False}


def test_ladder_does_not_ratchet_after_hours():
    rm, rec = _rth_rm(bucket="Breakout")
    with patch("market_data.is_market_hours", return_value=False), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("risk_manager.SETUP_GATES_ENABLED", True), \
         patch.dict(risk_manager.CATEGORY_RISK_PCT_PER_TRADE, {"Breakout": 0.01}), \
         patch.object(risk_manager.RiskManager, "_ratchet_ladder_stop") as ratchet:
        rm.check_trailing_stops()
    ratchet.assert_not_called()


def test_an_unreadable_market_clock_counts_as_closed():
    rm, rec = _rth_rm()
    with patch("market_data.is_market_hours", side_effect=RuntimeError("tz")), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("tasks.manage_place_trailing_stop") as trail, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        assert rm.check_trailing_stops() == []
    trail.assert_not_called(); cancel.assert_not_called()
