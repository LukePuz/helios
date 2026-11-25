"""After an ADD with lagging position reads (11-10 NVDA): the stale-record
branch leaves the old shares' stop resting (L197) and rewrites the record at
the new shares' lower stop (L198). Every later merge that cancels all stops
and places one must not loosen the old shares below L197: the next ADD,
breakeven and the profit-target re-stop floor at the highest resting stop
(risk_manager._floor_at_resting_stop); the TRAIL conversion and the ladder
ratchet skip a level below it and cancel nothing (_resting_stop_floor).

Prices are scaled: lot 197 entered at 100 with its own stop L197 = 89.3;
the ADD filled 1.8% above L197 with its own stop L198 10.7% below L197."""
from unittest.mock import MagicMock, call, patch

import pytest

import execution
import risk_manager

L197 = 89.3
ADD_FILL = round(L197 * 1.018, 2)          # 90.91
L198 = round(L197 * (1 - 0.107), 2)        # 79.74
RECORD = {"ticker": "NVDA", "stop_price": L198, "entry_price": ADD_FILL,
          "thesis_bucket": "Accelerator", "trailing_active": False, "trailing_high": None}


def _stp(price, qty=1.0, order_type="STP"):
    return {"ticker": "NVDA", "action": "SELL", "order_type": order_type,
            "qty": qty, "stop_price": price, "status": "Submitted"}


RESTING = [_stp(L197), _stp(L198)]      # lot 197's own stop and the ADD's own stop


def _gone_after_cancel(stops):
    calls = []

    def read(ticker, *a, **k):
        calls.append(ticker)
        return list(stops) if len(calls) == 1 else []
    return read


def _rm(current, qty=2):
    rm = object.__new__(risk_manager.RiskManager)
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=qty, current_price=current)]
    rm._client = client
    return rm


# ── next ADD (whole-position path) ───────────────────────────────────────────

def test_next_add_floors_at_the_old_shares_resting_stop():
    new_stop = round(L198 * (1 - 0.062), 2)     # the next ADD's own stop, below the record
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=3, avg_entry_price=95.0)
    with patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(3.0, 95.0)), \
         patch("tasks._resting_sell_stops", return_value=list(RESTING)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send"), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        stop, whole = execution._whole_position_stop("NVDA", new_stop, 1, status={},
                                                     market_price=100.0, action="ADD",
                                                     held_before=2)
    assert stop == L197                          # not max(new, record) = L198
    assert whole == (95.0, 3.0)
    cancel.assert_called_once_with("NVDA", stuck_out=[])


# ── breakeven ────────────────────────────────────────────────────────────────

def test_breakeven_floors_at_the_old_shares_resting_stop():
    level = round((L197 + L198) / 2, 2)          # above the record, below lot 197's stop
    with patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("tasks._resting_sell_stops", return_value=list(RESTING)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 6}) as place, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.mark_profit_target_taken"), patch("tasks._queue_reprotect"), \
         patch("alerts._send"):
        _rm(100.0)._breakeven_without_sale("NVDA", 2, level, current_price=100.0)
    cancel.assert_called_once_with("NVDA")
    assert place.call_args == call("NVDA", 2, L197, market_price=100.0)


# ── profit-target re-stop ────────────────────────────────────────────────────

def test_profit_target_restop_floors_at_the_old_shares_resting_stop():
    level = round((L197 + L198) / 2, 2)
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=1)
    with patch("market_data.is_market_hours", return_value=True), \
         patch("risk_manager._pt_hold_still_blocked", return_value=False), \
         patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("tasks._resting_sell_stops", side_effect=_gone_after_cancel(RESTING)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("stop_lock.assert_held"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": 1, "fill_price": 120.0,
                                                        "order_id": 5}), \
         patch("ibkr.get_client", return_value=client), \
         patch("db.log_trade"), patch("db.log_order", return_value=0), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 8}) as place, \
         patch("alerts._send"):
        rm = _rm(120.0)
        with patch.object(risk_manager.RiskManager, "_changed_while_waiting", return_value=None):
            out = rm._partial_sell_and_resize_stop("NVDA", 2, 1, level, decision_price=120.0)
    cancel.assert_called_once_with("NVDA")
    assert out["new_stop"] == L197
    assert place.call_args == call("NVDA", 1.0, L197, market_price=120.0)


# ── TRAIL conversion: skipped, nothing cancelled ─────────────────────────────

def test_trail_conversion_below_the_old_shares_stop_is_skipped():
    current = round(ADD_FILL * 1.27, 2)          # past the Accelerator activation on the record entry
    broker = MagicMock()
    rec = dict(RECORD)
    # ATR 15: a 1.75 x ATR trail starts 26.25 under the price, between L198 and L197
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=15.0), \
         patch("tasks._resting_sell_stops", side_effect=_gone_after_cancel(RESTING)), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_place_trailing_stop", new=broker.trail), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks._queue_reprotect"), patch("tasks.open_manual_action"), \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.set_trail_params"):
        _rm(current).check_trailing_stops()
    assert L198 < current - 1.75 * 15.0 < L197
    assert broker.mock_calls == []
    assert sum("STOP CHANGE SKIPPED" in c.args[0] for c in send.call_args_list) == 1


# ── ladder ratchet: skipped, nothing cancelled ───────────────────────────────

def test_ladder_ratchet_below_the_old_shares_stop_is_skipped():
    rung = round((L197 + L198) / 2, 2)
    broker = MagicMock()
    with patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("db.get_trade_ladder", return_value={"rungs": [{"price": rung, "confluence_score": 1}]}), \
         patch("tasks._resting_sell_stops", return_value=list(RESTING)), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send"), \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"):
        _rm(100.0)._ratchet_ladder_stop("NVDA", MagicMock(qty=2), 100.0, L198, [])
    assert broker.mock_calls == []


@pytest.mark.parametrize("rung_over_l197", [0.5])
def test_ladder_ratchet_above_the_old_shares_stop_still_moves(rung_over_l197):
    rung = round(L197 + rung_over_l197, 2)
    broker = MagicMock()
    broker.place.return_value = {"order_id": 4}
    with patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("db.get_trade_ladder", return_value={"rungs": [{"price": rung, "confluence_score": 1}]}), \
         patch("tasks._resting_sell_stops", return_value=list(RESTING)), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send"), \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"):
        _rm(100.0)._ratchet_ladder_stop("NVDA", MagicMock(qty=2), 100.0, L198, [])
    assert broker.mock_calls == [call.cancel("NVDA"), call.place("NVDA", 2, rung, market_price=100.0)]
