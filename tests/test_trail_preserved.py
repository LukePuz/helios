"""A fixed stop never lowers a trail-raised stop, and placing one re-arms the
trail (limitations audit 2026-10-28 12:53, finding 3)."""
import pytest
from unittest.mock import MagicMock, patch

import db
import risk_manager
import tasks



@pytest.fixture(autouse=True)
def _in_market_hours(monkeypatch):
    # Partial sales only run in RTH since professor C15 (2026-10-28).
    import market_data
    monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)

def test_fixed_stop_clears_trailing_active():
    with patch("tasks._is_orders_worker", return_value=True), \
         patch("ibkr.place_stop_order", return_value={"order_id": 5}), \
         patch("db.clear_trailing_active") as clear:
        assert tasks.manage_stop_place("NVDA", 10, 180.0) == {"order_id": 5}
    clear.assert_called_once_with("NVDA")


def test_clear_failure_does_not_hide_the_placed_stop():
    with patch("tasks._is_orders_worker", return_value=True), \
         patch("ibkr.place_stop_order", return_value={"order_id": 5}), \
         patch("db.clear_trailing_active", side_effect=RuntimeError("db down")):
        assert tasks.manage_stop_place("NVDA", 10, 180.0) == {"order_id": 5}


def test_partial_sale_keeps_a_trail_raised_stop():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=70)
    with patch("db.get_position_stop", return_value={"stop_price": 112.0, "trailing_active": True}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": 30, "order_id": 7}), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place", return_value={"order_id": 8}) as place, \
         patch("db.log_trade"), patch("db.log_order", return_value=1), patch("db.update_order_fill"), \
         patch("alerts._send"):
        result = object.__new__(risk_manager.RiskManager)._partial_sell_and_resize_stop(
            "NVDA", 100, 30, 100.0)
    place.assert_called_once_with("NVDA", 70.0, 112.0)  # not breakeven 100
    assert result["new_stop"] == 112.0


def test_mark_profit_target_taken_never_lowers_the_stop():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.mark_profit_target_taken("NVDA", 100.0)
    assert "GREATEST(COALESCE(stop_price, 0), %s)" in cur.execute.call_args.args[0]


def _trail_check(stop_price, atr):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=126.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": stop_price, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": False, "trailing_high": 126.0}
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=atr), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_place_trailing_stop", return_value={"order_id": 9}) as trail, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.set_trail_params", new=_SET_TRAIL_PARAMS):
        rm.check_trailing_stops()
    return cancel, trail


_SET_TRAIL_PARAMS = MagicMock()


def test_trail_rearms_after_a_fixed_stop():
    # +26% on an Accelerator; trail from 126 sits above the 112 fixed stop
    _, trail = _trail_check(stop_price=112.0, atr=4.0)
    trail.assert_called_once()


def test_trail_not_armed_below_the_current_stop():
    # 1.75 x ATR 10 = 17.5 -> 108.5 < 112: keep the fixed stop for now
    cancel, trail = _trail_check(stop_price=112.0, atr=10.0)
    trail.assert_not_called()
    cancel.assert_not_called()


def _already_trailing(rec_extra, current):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=current)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    rec = {"ticker": "NVDA", "stop_price": 112.0, "thesis_bucket": "Accelerator",
           "entry_price": 100.0, "trailing_active": True, "trailing_high": 126.0, **rec_extra}
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch("db.update_trailing_stop") as upd:
        rm.check_trailing_stops()
    return upd


def test_recorded_stop_follows_the_trail_amount():
    upd = _already_trailing({"trail_amount": 7.0}, current=130.0)
    upd.assert_called_once_with("NVDA", 123.0, 130.0)


def test_recorded_stop_follows_the_trail_percent():
    upd = _already_trailing({"trail_amount": None, "trail_percent": 10.0}, current=130.0)
    upd.assert_called_once_with("NVDA", 117.0, 130.0)


def test_recorded_stop_never_drops_and_needs_a_new_high():
    upd = _already_trailing({"trail_amount": 30.0}, current=130.0)
    upd.assert_called_once_with("NVDA", 112.0, 130.0)  # est 100 < 112 keeps 112
    assert not _already_trailing({"trail_amount": 7.0}, current=125.0).called


def test_conversion_stores_trail_params():
    _SET_TRAIL_PARAMS.reset_mock()
    _, trail = _trail_check(stop_price=112.0, atr=4.0)
    trail.assert_called_once()
    _SET_TRAIL_PARAMS.assert_called_once_with("NVDA", 7.0, None)


def test_stop_reader_survives_missing_trail_columns():
    import psycopg2.errors
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    row = ("NVDA", 100, 112, "m", True, 126, None, "Accelerator", 7, None, None, False, None, None,
           None, None)
    cur.execute.side_effect = [psycopg2.errors.UndefinedColumn("trail_amount"), None]
    cur.fetchall.return_value = [row]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        stops = db.get_all_position_stops()
    assert stops[0]["ticker"] == "NVDA" and stops[0]["trail_amount"] is None


def test_migration_58_adds_trail_columns():
    assert any(v == 58 and "trail_amount" in sql for v, sql in db._MIGRATIONS)
