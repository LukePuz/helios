"""Two order-path guards on the cancel-and-replace stop paths.

Placed guard: once the broker has accepted a replacement stop, a failing DB
write (pool exhausted: db._get_conn raises outside the writers' own try)
must not reach a fallback or restore that places a second stop.

No-loosen floor: the ADD, breakeven and profit-target paths replace the stop
at the DB record, which can sit below a native TRAIL's live trigger. The
replacement is raised to the highest resting SELL stop; a failed read keeps
the old behaviour and alerts."""
from unittest.mock import MagicMock, call, patch

import pytest

import execution
import risk_manager

POOL = RuntimeError("connection pool exhausted")


def _stp(price, order_type="TRAIL"):
    return {"ticker": "NVDA", "action": "SELL", "order_type": order_type,
            "qty": 7.0, "stop_price": price, "status": "Submitted"}


def _gone_after_cancel(stops):
    """tasks._resting_sell_stops side effect: the stops on the first read
    (before the cancel), none after it, as at the broker."""
    calls = []

    def read(ticker, *a, **k):
        calls.append(ticker)
        return list(stops) if len(calls) == 1 else []
    return read


def _rm(current=None):
    rm = object.__new__(risk_manager.RiskManager)
    if current is not None:
        client = MagicMock()
        client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=current)]
        rm._client = client
    return rm


def _sent(send, text):
    return sum(text in c[0][0] for c in send.call_args_list)


@pytest.fixture(autouse=True)
def _in_market_hours(monkeypatch):
    import market_data
    monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)


# ── M227: TRAIL conversion ───────────────────────────────────────────────

def _convert(record_stop, current, trailing_high, set_id_effect=POOL):
    """Accelerator at +26%; 1.75 x ATR 4 = 7.00 trail, starting at current - 7."""
    rm = _rm(current)
    rec = {"ticker": "NVDA", "stop_price": record_stop, "thesis_bucket": "Accelerator",
           "entry_price": round(current / 1.26, 2), "trailing_active": False,
           "trailing_high": trailing_high}
    broker = MagicMock()
    broker.trail.return_value = {"order_id": 9}
    broker.place.return_value = {"order_id": 3}
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks._resting_sell_stops", return_value=[_stp(record_stop, "STP")]), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_place_trailing_stop", new=broker.trail), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action") as manual, \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect) as set_id, \
         patch("db.update_trailing_stop") as update, \
         patch("db.set_trail_params") as params:
        alerts = rm.check_trailing_stops()
    return dict(broker=broker, send=send, alerts=alerts, queue=queue, manual=manual,
                set_id=set_id, update=update, params=params)


def test_record_write_failure_after_the_trail_leaves_exactly_one_stop():
    # current 101 <= trailing_high 105: before the fix this restored a full
    # fixed stop on top of the resting TRAIL
    r = _convert(92.0, 101.0, trailing_high=105.0)
    assert r["broker"].mock_calls == [call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0)]
    r["queue"].assert_not_called()
    r["manual"].assert_not_called()
    assert not [a for a in r["alerts"] if a["action"] == "FIX_STOP"]
    assert _sent(r["send"], "STOP RECORD NOT UPDATED") == 1
    # every write is still tried on its own
    r["set_id"].assert_called_once_with("NVDA", 9)
    r["update"].assert_called_once_with("NVDA", 94.0, 101.0)
    r["params"].assert_called_once_with("NVDA", 7.0, None)


def test_record_write_failure_after_the_trail_skips_the_fallback_move_too():
    # current above the old high: the other fallback would cancel the TRAIL
    # and place a fixed stop
    r = _convert(92.0, 101.0, trailing_high=99.0)
    assert r["broker"].mock_calls == [call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0)]
    assert not [a for a in r["alerts"] if a["action"] == "FIX_STOP"]


def test_trail_conversion_with_working_writes_sends_no_record_alert():
    r = _convert(92.0, 101.0, trailing_high=105.0, set_id_effect=None)
    assert r["broker"].mock_calls == [call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0)]
    assert _sent(r["send"], "STOP RECORD NOT UPDATED") == 0


def test_restore_after_a_failed_trail_is_not_queued_again_when_its_write_fails():
    rm = _rm(101.0)
    rec = {"ticker": "NVDA", "stop_price": 92.0, "thesis_bucket": "Accelerator",
           "entry_price": round(101.0 / 1.26, 2), "trailing_active": False, "trailing_high": 105.0}
    broker = MagicMock()
    broker.trail.side_effect = RuntimeError("rejected")
    broker.place.return_value = {"order_id": 3}
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks._resting_sell_stops", side_effect=_gone_after_cancel([_stp(92.0, "STP")])), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_place_trailing_stop", new=broker.trail), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action") as manual, \
         patch("alerts._send"), \
         patch("db.set_stop_broker_order_id", side_effect=POOL), \
         patch("db.update_trailing_stop"), patch("db.set_trail_params"):
        alerts = rm.check_trailing_stops()
    broker.place.assert_called_once_with("NVDA", 7.0, 92.0, market_price=101.0)
    queue.assert_not_called()
    manual.assert_not_called()
    assert not [a for a in alerts if a["action"] == "FIX_STOP"]


# ── M227: ladder ratchet, breakeven, profit-target re-place ──────────────

def test_ladder_ratchet_record_failure_is_not_reported_as_a_failed_order():
    rm = _rm(100.0)
    broker = MagicMock()
    broker.place.return_value = {"order_id": 4}
    alerts: list = []
    with patch("db.get_position_stop", return_value={"stop_price": 92.0}), \
         patch("db.get_trade_ladder", return_value={"rungs": [{"price": 95.0, "confluence_score": 1}]}), \
         patch("tasks._resting_sell_stops", return_value=[_stp(92.0, "STP")]), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id", side_effect=POOL), patch("db.update_trailing_stop"):
        rm._ratchet_ladder_stop("NVDA", MagicMock(qty=7), 100.0, 92.0, alerts)
    assert broker.mock_calls == [call.cancel("NVDA"), call.place("NVDA", 7.0, 95.0, market_price=100.0)]
    assert alerts == []
    assert _sent(send, "STOP RECORD NOT UPDATED") == 1


def test_breakeven_record_failure_after_placement_does_not_raise():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("db.set_stop_broker_order_id", side_effect=POOL), \
         patch("db.mark_profit_target_taken") as taken, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
         patch("alerts._send") as send:
        assert _rm()._breakeven_without_sale("NVDA", 1, 100.0) is True
    place.assert_called_once()
    taken.assert_called_once_with("NVDA", 100.0)
    assert _sent(send, "STOP RECORD NOT UPDATED") == 1


def test_breakeven_restore_with_a_failed_record_write_is_not_unprotected():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("db.set_stop_broker_order_id", side_effect=POOL), \
         patch("db.mark_profit_target_taken"), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", side_effect=[RuntimeError("down"), {"order_id": 6}]) as place, \
         patch("alerts._send") as send:
        assert _rm()._breakeven_without_sale("NVDA", 2, 100.0) is False
    assert place.call_count == 2
    assert _sent(send, "UNPROTECTED") == 0


def _partial(record, resting, decision_price=120.0, set_id_effect=None, read_effect=None,
             sell=None):
    calls = []
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=70)
    read = (patch("tasks._resting_sell_stops", side_effect=read_effect) if read_effect
            else patch("tasks._resting_sell_stops", side_effect=_gone_after_cancel(resting)))
    with patch("db.get_position_stop", return_value={"stop_price": record}), \
         patch("db.log_trade"), patch("db.log_order", return_value=1), \
         patch("db.update_order_fill"), patch("db.update_trailing_stop") as _upd, \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect), \
         read, patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=lambda t: calls.append("cancel")), \
         patch("ibkr.place_market_order",
               side_effect=sell or (lambda *a: calls.append("sell") or {"filled_qty": 30, "order_id": 7})), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place",
               side_effect=lambda *a, **k: calls.append(("stop",) + a) or {"order_id": 8}), \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0,
                                                     decision_price=decision_price)
    return calls, send, result


def test_profit_target_re_place_record_failure_is_not_unprotected():
    calls, send, _ = _partial(90.0, [], set_id_effect=POOL)
    assert calls == ["cancel", "sell", ("stop", "NVDA", 70.0, 100.0)]
    assert _sent(send, "UNPROTECTED") == 0
    assert _sent(send, "STOP RECORD NOT UPDATED") == 1


# ── M228: floor at the highest resting stop ──────────────────────────────

def test_profit_target_stop_is_floored_at_a_higher_resting_trail():
    # record 90, breakeven 100, broker TRAIL trigger 103
    calls, send, result = _partial(90.0, [_stp(103.0)])
    assert calls == ["cancel", "sell", ("stop", "NVDA", 70.0, 103.0)]
    assert result["new_stop"] == 103.0
    send.assert_not_called()


def test_failed_profit_target_sale_restores_at_the_resting_trail():
    calls, _, result = _partial(90.0, [_stp(103.0)], sell=RuntimeError("rejected"))
    assert result is None
    assert calls == ["cancel", ("stop", "NVDA", 100, 103.0)]


def test_profit_target_floor_read_failure_keeps_the_level_and_alerts():
    # the pre-cancel floor read fails; the post-cancel check reads no stop
    calls, send, _ = _partial(90.0, None, read_effect=[TimeoutError("read timed out"), [], []])
    assert calls == ["cancel", "sell", ("stop", "NVDA", 70.0, 100.0)]
    assert _sent(send, "STOP FLOOR UNREAD") == 1


def test_a_resting_stop_above_the_price_is_not_used_as_the_floor():
    calls, _, _ = _partial(90.0, [_stp(125.0)], decision_price=120.0)
    assert calls == ["cancel", "sell", ("stop", "NVDA", 70.0, 100.0)]


def _breakeven(resting, read_effect=None, place_effect=None):
    read = (patch("tasks._resting_sell_stops", side_effect=read_effect) if read_effect
            else patch("tasks._resting_sell_stops", side_effect=_gone_after_cancel(resting)))
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         read, patch("tasks._reconcile_alert_due", return_value=True), \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.mark_profit_target_taken") as taken, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", side_effect=place_effect or [{"order_id": 5}]) as place, \
         patch("alerts._send") as send:
        ok = _rm()._breakeven_without_sale("NVDA", 1, 100.0, current_price=110.0)
    return ok, place, taken, send


def test_breakeven_stop_is_floored_at_a_higher_resting_trail():
    ok, place, taken, send = _breakeven([_stp(103.0)])
    assert ok is True
    place.assert_called_once_with("NVDA", 1, 103.0, market_price=110.0)
    taken.assert_called_once_with("NVDA", 103.0)
    send.assert_not_called()


def test_failed_breakeven_restores_at_the_resting_trail():
    ok, place, _, _ = _breakeven([_stp(103.0)], place_effect=[RuntimeError("down"), {"order_id": 6}])
    assert ok is False
    assert place.call_count == 2
    assert place.call_args_list[-1] == call("NVDA", 1, 103.0, market_price=110.0)


def test_breakeven_floor_read_failure_keeps_the_level_and_alerts():
    ok, place, _, send = _breakeven(None, read_effect=TimeoutError("read timed out"))
    assert ok is True
    place.assert_called_once_with("NVDA", 1, 100.0, market_price=110.0)
    assert _sent(send, "STOP FLOOR UNREAD") == 1


def _add(resting, read_effect=None, prior=98.0, new_stop=95.0, price=110.0):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=15, avg_entry_price=105.0,
                                                      current_price=price)
    read = (patch("tasks._resting_sell_stops", side_effect=read_effect) if read_effect
            else patch("tasks._resting_sell_stops", return_value=resting))
    with patch("db.get_position_stop", return_value={"stop_price": prior}), \
         patch("ibkr.get_client", return_value=client), \
         read, patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("alerts._send") as send:
        result = execution._whole_position_stop("NVDA", new_stop, 5)
    return result, cancel, send


def test_add_stop_is_floored_at_a_higher_resting_trail():
    (stop, add), cancel, send = _add([_stp(101.0)])
    assert stop == 101.0
    assert add == (105.0, 15.0)
    cancel.assert_called_once_with("NVDA")
    send.assert_not_called()


def test_add_floor_read_failure_keeps_the_level_and_alerts():
    (stop, _), cancel, send = _add(None, read_effect=TimeoutError("read timed out"))
    assert stop == 98.0
    cancel.assert_called_once_with("NVDA")
    assert _sent(send, "STOP FLOOR UNREAD") == 1


def test_add_with_resting_stop_at_the_record_is_unchanged():
    (stop, _), _, send = _add([_stp(98.0, "STP")])
    assert stop == 98.0
    send.assert_not_called()


def test_failed_sell_restore_with_a_failed_record_write_is_not_queued_or_unprotected():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=3)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id", side_effect=POOL), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action") as manual, \
         patch("alerts._send") as send:
        execution._restore_stop_after_failed_sell("NVDA")
    place.assert_called_once_with("NVDA", 3.0, 90.0)
    queue.assert_not_called()
    manual.assert_not_called()
    assert _sent(send, "UNPROTECTED") == 0
    assert _sent(send, "STOP RECORD NOT UPDATED") == 1
