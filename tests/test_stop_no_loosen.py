"""No-loosen guard on the cancel-all-then-replace stop paths: the TRAIL
conversion, its failed-trail restore, and the ladder ratchet. A replacement
never sits below the highest working SELL stop it cancels, even when the
position_stops record is lower than what rests at the broker."""
from unittest.mock import MagicMock, call, patch

import risk_manager


def _gone_after_cancel(stops):
    """tasks._resting_sell_stops side effect: the stops on the first read
    (before the cancel), none after it, as at the broker."""
    calls = []

    def read(ticker, *a, **k):
        calls.append(ticker)
        return list(stops) if len(calls) == 1 else []
    return read


def _stp(price, order_type="STP"):
    return {"ticker": "NVDA", "action": "SELL", "order_type": order_type,
            "qty": 7.0, "stop_price": price, "status": "Submitted"}


def _rm(current):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=current)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    return rm


_last: dict = {}


def _trail(record_stop, current, resting, trail_effect=({"order_id": 9},),
           place_effect=({"order_id": 3},), trailing_high=None, update_effect=None):
    """Accelerator at +26%; 1.75 x ATR 4 = 7.00 trail, so it starts at current - 7."""
    rm = _rm(current)
    rec = {"ticker": "NVDA", "stop_price": record_stop, "thesis_bucket": "Accelerator",
           "entry_price": round(current / 1.26, 2), "trailing_active": False,
           "trailing_high": trailing_high}
    read = (patch("tasks._resting_sell_stops", side_effect=resting) if isinstance(resting, Exception)
            else patch("tasks._resting_sell_stops", side_effect=_gone_after_cancel(resting)))
    broker = MagicMock()
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         read, \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_place_trailing_stop", new=broker.trail) as trail, \
         patch("tasks.manage_stop_place", new=broker.place) as place, \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action"), \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id"), \
         patch("db.update_trailing_stop", side_effect=update_effect) as update, \
         patch("db.set_trail_params"):
        _last.update(queue=queue, update=update)
        trail.side_effect = list(trail_effect) if not isinstance(trail_effect, Exception) else trail_effect
        place.side_effect = list(place_effect) if not isinstance(place_effect, Exception) else place_effect
        alerts = rm.check_trailing_stops()
    return broker, send, alerts


def test_trail_below_a_higher_resting_stop_is_skipped_and_nothing_cancelled():
    # record 85, resting 92, trail would start at 88
    broker, send, _ = _trail(85.0, 95.0, [_stp(92.0)])
    assert broker.mock_calls == []
    assert sum("STOP CHANGE SKIPPED" in c[0][0] for c in send.call_args_list) == 1


def test_a_resting_trail_counts_at_its_trigger_price():
    broker, _, _ = _trail(85.0, 95.0, [_stp(92.0, "TRAIL")])
    assert broker.mock_calls == []


def test_trail_with_record_equal_to_resting_is_unchanged():
    # record 92, resting 92, trail starts at 94: same calls, order and arguments
    broker, send, _ = _trail(92.0, 101.0, [_stp(92.0)])
    assert broker.mock_calls == [call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0)]
    assert not send.called


def test_trail_with_no_resting_stop_still_converts():
    broker, _, _ = _trail(92.0, 101.0, [])
    assert broker.mock_calls == [call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0)]


def test_trail_skipped_when_the_open_order_read_fails():
    broker, send, _ = _trail(92.0, 101.0, TimeoutError("read timed out"))
    assert broker.mock_calls == []
    assert sum("could not be read" in c[0][0] for c in send.call_args_list) == 1


def test_trail_skipped_when_a_resting_price_is_unreadable():
    broker, send, _ = _trail(92.0, 101.0, [_stp(None)])
    assert broker.mock_calls == []
    assert sum("no readable price" in c[0][0] for c in send.call_args_list) == 1


def test_failed_trail_restores_at_the_highest_cancelled_stop():
    # record 85, resting 87, trail start 88 passes; the TRAIL fails and the
    # restore goes back at 87, not the record's 85
    broker, _, alerts = _trail(85.0, 95.0, [_stp(86.0), _stp(87.0)],
                               trail_effect=RuntimeError("broker"), trailing_high=99.0)
    assert broker.mock_calls[0] == call.cancel("NVDA")
    broker.place.assert_called_once_with("NVDA", 7.0, 87.0, market_price=95.0)
    assert not [a for a in alerts if a["action"] == "FIX_STOP"]


def test_failed_trail_restore_failure_names_the_higher_level():
    _, _, alerts = _trail(85.0, 95.0, [_stp(87.0)], trail_effect=RuntimeError("broker"),
                          place_effect=RuntimeError("down"), trailing_high=99.0)
    fix = [a for a in alerts if a["action"] == "FIX_STOP"]
    assert fix and fix[0]["stop_price"] == 87.0 and "$87.00" in fix[0]["reason"]


def test_failed_trail_restores_at_the_record_when_consistent():
    broker, _, _ = _trail(92.0, 101.0, [_stp(92.0)], trail_effect=RuntimeError("broker"),
                          trailing_high=105.0)
    broker.place.assert_called_once_with("NVDA", 7.0, 92.0, market_price=101.0)


def test_failed_trail_restore_raises_the_record_to_the_restored_level():
    broker, _, _ = _trail(85.0, 95.0, [_stp(87.0)], trail_effect=RuntimeError("broker"),
                          trailing_high=99.0)
    broker.place.assert_called_once_with("NVDA", 7.0, 87.0, market_price=95.0)
    _last["update"].assert_called_once_with("NVDA", 87.0, 99.0, native_trail=False)


def test_failed_trail_restore_leaves_the_record_alone_when_consistent():
    _trail(92.0, 101.0, [_stp(92.0)], trail_effect=RuntimeError("broker"), trailing_high=105.0)
    _last["update"].assert_not_called()


def test_a_failed_record_write_still_leaves_the_restored_stop():
    # update_trailing_stop's pool checkout is outside its own try, so it can
    # raise; that must not read as a failed restore (no alert, no queue)
    broker, _, alerts = _trail(85.0, 95.0, [_stp(87.0)], trail_effect=RuntimeError("broker"),
                               trailing_high=99.0, update_effect=RuntimeError("pool exhausted"))
    broker.place.assert_called_once_with("NVDA", 7.0, 87.0, market_price=95.0)
    _last["update"].assert_called_once()
    assert not [a for a in alerts if a["action"] == "FIX_STOP"]
    _last["queue"].assert_not_called()


def test_failed_higher_restore_queues_a_plain_restore():
    _trail(85.0, 95.0, [_stp(87.0)], trail_effect=RuntimeError("broker"),
           place_effect=RuntimeError("down"), trailing_high=99.0)
    _last["queue"].assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    _last["update"].assert_not_called()


def test_failed_restore_queue_entry_unchanged_when_consistent():
    _trail(92.0, 101.0, [_stp(92.0)], trail_effect=RuntimeError("broker"),
           place_effect=RuntimeError("down"), trailing_high=105.0)
    _last["queue"].assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)


def _ratchet(record_stop, resting, rung, current=100.0):
    rm = _rm(current)
    pos = MagicMock(qty=7)
    read = (patch("tasks._resting_sell_stops", side_effect=resting) if isinstance(resting, Exception)
            else patch("tasks._resting_sell_stops", return_value=resting))
    broker = MagicMock()
    broker.place.return_value = {"order_id": 4}
    alerts: list = []
    with patch("db.get_position_stop", return_value={"stop_price": record_stop}), \
         patch("db.get_trade_ladder", return_value={"rungs": [{"price": rung, "confluence_score": 1}]}), \
         read, \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"):
        rm._ratchet_ladder_stop("NVDA", pos, current, record_stop, alerts)
    return broker, send


def test_ratchet_below_a_higher_resting_stop_is_skipped_and_nothing_cancelled():
    broker, send = _ratchet(85.0, [_stp(92.0)], rung=88.0)
    assert broker.mock_calls == []
    assert sum("STOP CHANGE SKIPPED" in c[0][0] for c in send.call_args_list) == 1


def test_ratchet_with_record_equal_to_resting_is_unchanged():
    broker, send = _ratchet(92.0, [_stp(92.0)], rung=95.0)
    assert broker.mock_calls == [call.cancel("NVDA"), call.place("NVDA", 7.0, 95.0, market_price=100.0)]
    assert not send.called


def test_ratchet_skipped_when_the_open_order_read_fails():
    broker, send = _ratchet(92.0, TimeoutError("read timed out"), rung=95.0)
    assert broker.mock_calls == []
    assert sum("could not be read" in c[0][0] for c in send.call_args_list) == 1


def test_a_same_cent_level_is_not_read_as_lower():
    # 91.995 goes on the tick at 92.00: not below a 92.00 resting stop
    rm = _rm(95.0)
    with patch("tasks._resting_sell_stops", return_value=[_stp(92.0)]), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send") as send:
        assert rm._resting_stop_floor("NVDA", 91.995, "x") == (True, 92.0)
        assert rm._resting_stop_floor("NVDA", 91.99, "x") == (False, 92.0)
    assert send.call_count == 1


def test_ratchet_to_a_same_cent_rung_places_the_rung_unchanged():
    broker, send = _ratchet(91.5, [_stp(92.0)], rung=91.995)
    assert broker.mock_calls == [call.cancel("NVDA"), call.place("NVDA", 7.0, 91.995, market_price=100.0)]
    assert not send.called


def test_each_skip_kind_has_its_own_throttle_key():
    rm = _rm(95.0)
    with patch("tasks._reconcile_alert_due", return_value=False) as due, \
         patch("alerts._send") as send:
        with patch("tasks._resting_sell_stops", side_effect=TimeoutError("x")):
            rm._resting_stop_floor("NVDA", 88.0, "x")
        with patch("tasks._resting_sell_stops", return_value=[_stp(None)]):
            rm._resting_stop_floor("NVDA", 88.0, "x")
        with patch("tasks._resting_sell_stops", return_value=[_stp(92.0)]):
            rm._resting_stop_floor("NVDA", 88.0, "x")
    keys = [c.args[0] for c in due.call_args_list]
    assert len(set(keys)) == 3 and all("NVDA" in k for k in keys)
    assert not send.called


def test_skip_alert_is_throttled():
    rm = _rm(95.0)
    with patch("tasks._resting_sell_stops", return_value=[_stp(92.0)]), \
         patch("tasks._reconcile_alert_due", side_effect=[True, False]), \
         patch("alerts._send") as send:
        assert rm._resting_stop_floor("NVDA", 88.0, "x") == (False, 92.0)
        assert rm._resting_stop_floor("NVDA", 88.0, "x") == (False, 92.0)
    assert send.call_count == 1
