"""Follow-ups to the placed guard and the no-loosen floor.

- Restores after a failed replacement are sized to the shares no working stop
  covers: an error after ib.placeOrder that isn't OrdersCallUnknown may leave
  the new stop resting. An unreadable coverage read places nothing.
- The floor also applies to TRIM, manual sell and the failed-sell restore,
  read from the sell path's own pre-cancel read.
- A restore above the record raises the record.
- A failed profit-target mark can't abort check_profit_targets or allow a
  second sale in the same process.
- The ladder ratchet puts the previous stop back when its placement fails."""
import math
from unittest.mock import MagicMock, call, patch

import pytest

import execution
import risk_manager
import tasks

POOL = RuntimeError("connection pool exhausted")


def _stp(price, qty=7.0, order_type="TRAIL"):
    return {"ticker": "NVDA", "action": "SELL", "order_type": order_type,
            "qty": qty, "stop_price": price, "status": "Submitted"}


def _reads(*seq):
    """tasks._resting_sell_stops side effect: one entry per read (a list, or
    an exception to raise); the last entry repeats."""
    calls = []

    def read(ticker, *a, **k):
        calls.append(ticker)
        item = seq[min(len(calls), len(seq)) - 1]
        if isinstance(item, Exception):
            raise item
        return list(item)
    return read


def _sent(send, text):
    return sum(text in c[0][0] for c in send.call_args_list)


@pytest.fixture(autouse=True)
def _in_market_hours(monkeypatch):
    import market_data
    monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _recheck(monkeypatch):
    """No 5 s second look on a cancelling stop, and the coverage re-check
    _restore_cover's callers queue is recorded instead of published."""
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    queued = MagicMock()
    monkeypatch.setattr(tasks.recheck_stop_coverage, "apply_async", queued)
    # the re-check's position cross-check: reads agree
    monkeypatch.setattr(tasks, "_recheck_position_reads", lambda ticker: {})
    return queued


@pytest.fixture(autouse=True)
def _clear_pt_marker():
    risk_manager._PT_TAKEN_UNRECORDED.clear()
    risk_manager._PT_HELD_BY_WORKING_STOP.clear()
    yield
    risk_manager._PT_TAKEN_UNRECORDED.clear()
    risk_manager._PT_HELD_BY_WORKING_STOP.clear()


def _rm(current=None):
    rm = object.__new__(risk_manager.RiskManager)
    if current is not None:
        client = MagicMock()
        client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=7, current_price=current)]
        rm._client = client
    return rm


# ── 1. Residual stacking: TRAIL restore ─────────────────────────────────

def _convert(trail_effect, reads, place_effect=({"order_id": 3},), trailing_high=105.0,
             set_id_effect=None, record=92.0, current=101.0):
    """Accelerator at +26%; 1.75 x ATR 4 = 7.00 trail."""
    rm = _rm(current)
    rec = {"ticker": "NVDA", "stop_price": record, "thesis_bucket": "Accelerator",
           "entry_price": round(current / 1.26, 2), "trailing_active": False,
           "trailing_high": trailing_high}
    broker = MagicMock()
    broker.trail.side_effect = trail_effect
    broker.place.side_effect = list(place_effect)
    with patch("db.get_all_position_stops", return_value=[rec]), \
         patch("db.get_position_stop", return_value=rec), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager.RiskManager, "get_atr", return_value=4.0), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_place_trailing_stop", new=broker.trail), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action") as manual, \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect) as set_id, \
         patch("db.update_trailing_stop") as update, patch("db.set_trail_params"):
        alerts = rm.check_trailing_stops()
    return dict(broker=broker, send=send, alerts=alerts, queue=queue, manual=manual,
                set_id=set_id, update=update)


def test_trail_error_after_placement_with_the_trail_resting_places_no_restore():
    # the TRAIL call raised (e.g. inside the accept wait) but the TRAIL rests
    r = _convert(RuntimeError("worker lost"), [[_stp(92.0, order_type="STP")], [_stp(94.0)]])
    assert r["broker"].mock_calls == [call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0)]
    r["queue"].assert_not_called()


def test_trail_restore_is_sized_to_the_uncovered_shares():
    r = _convert(RuntimeError("x"), [[_stp(92.0, order_type="STP")], [_stp(94.0, qty=3.0)]])
    r["broker"].place.assert_called_once_with("NVDA", 4.0, 92.0, market_price=101.0)


def test_trail_restore_places_nothing_when_coverage_is_unreadable():
    r = _convert(RuntimeError("x"), [[_stp(92.0, order_type="STP")], TimeoutError("read")])
    r["broker"].place.assert_not_called()
    r["queue"].assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    assert _sent(r["send"], "CHECK STOPS") == 1
    assert not [a for a in r["alerts"] if a["action"] == "FIX_STOP"]


def test_trail_fallback_move_with_a_failing_write_places_one_stop():
    # current above the old high: the fixed-stop move branch; its record
    # write fails after the stop is placed
    r = _convert(RuntimeError("x"), [[_stp(92.0, order_type="STP")], []], trailing_high=99.0,
                 set_id_effect=POOL)
    assert r["broker"].mock_calls == [
        call.cancel("NVDA"), call.trail("NVDA", 7.0, trail_amount=7.0),
        call.cancel("NVDA"), call.place("NVDA", 7.0, 94.0, market_price=101.0)]
    assert not [a for a in r["alerts"] if a["action"] == "FIX_STOP"]
    assert _sent(r["send"], "STOP RECORD NOT UPDATED") == 1


# ── 1/4. Breakeven restore: sized, floored, record raised ───────────────

def _breakeven(reads, place_effect, set_id_effect=None, qty=2):
    rec = {"stop_price": 90.0, "trailing_high": 120.0}
    with patch("db.get_position_stop", return_value=rec), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect), \
         patch("db.update_trailing_stop") as update, \
         patch("db.mark_profit_target_taken") as taken, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", side_effect=place_effect) as place, \
         patch("tasks._queue_reprotect") as queue, \
         patch("alerts._send") as send:
        ok = _rm()._breakeven_without_sale("NVDA", qty, 100.0, current_price=110.0)
    return ok, place, update, taken, queue, send


def test_breakeven_error_with_the_new_stop_resting_places_no_restore():
    ok, place, _, taken, _, send = _breakeven(
        [[], [_stp(100.0, qty=2.0, order_type="STP")]], [RuntimeError("accept wait raised")])
    assert ok is False
    assert place.call_count == 1          # the failed move only
    taken.assert_not_called()
    assert _sent(send, "UNPROTECTED") == 0


def test_breakeven_restore_places_nothing_when_coverage_is_unreadable():
    ok, place, _, _, queue, send = _breakeven([[], TimeoutError("read")], [RuntimeError("down")])
    assert ok is False and place.call_count == 1
    queue.assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    assert _sent(send, "CHECK STOPS") == 1 and _sent(send, "UNPROTECTED") == 0


def test_breakeven_restore_at_a_resting_trail_raises_the_record():
    ok, place, update, _, _, _ = _breakeven([[_stp(103.0)], []],
                                            [RuntimeError("down"), {"order_id": 6}])
    assert place.call_args_list[-1] == call("NVDA", 2.0, 103.0, market_price=110.0)
    update.assert_called_once_with("NVDA", 103.0, 120.0, native_trail=False)


def test_breakeven_restore_at_the_record_leaves_the_record_alone():
    _, place, update, _, _, _ = _breakeven([[], []], [RuntimeError("down"), {"order_id": 6}])
    assert place.call_args_list[-1] == call("NVDA", 2.0, 90.0, market_price=110.0)
    update.assert_not_called()


def test_breakeven_restore_with_a_failing_write_is_not_unprotected():
    _, place, _, _, queue, send = _breakeven([[], []], [RuntimeError("down"), {"order_id": 6}],
                                             set_id_effect=POOL)
    assert place.call_count == 2
    queue.assert_not_called()
    assert _sent(send, "UNPROTECTED") == 0 and _sent(send, "STOP RECORD NOT UPDATED") == 1


# ── 1/4. Profit-target failed-sale restore ──────────────────────────────

def _pt_failed_sale(reads, place_effect=({"order_id": 8},), set_id_effect=None):
    with patch("db.get_position_stop", return_value={"stop_price": 90.0, "trailing_high": 125.0}), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect), \
         patch("db.update_trailing_stop") as update, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", side_effect=RuntimeError("rejected")), \
         patch("tasks.manage_stop_place", side_effect=list(place_effect)) as place, \
         patch("tasks._queue_reprotect") as queue, \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0, decision_price=120.0)
    assert result is None
    return place, update, queue, send


def test_pt_restore_is_sized_to_the_uncovered_shares():
    place, _, _, _ = _pt_failed_sale([[], [], [_stp(90.0, qty=40.0, order_type="STP")]])
    place.assert_called_once_with("NVDA", 60.0, 90.0, market_price=120.0)


def test_pt_restore_places_nothing_when_coverage_is_unreadable():
    place, _, queue, send = _pt_failed_sale([[], [], TimeoutError("read")])
    place.assert_not_called()
    queue.assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    assert _sent(send, "UNPROTECTED") == 0 and _sent(send, "CHECK STOPS") == 1


def test_pt_restore_at_a_resting_trail_raises_the_record():
    place, update, _, _ = _pt_failed_sale([[_stp(103.0)], []])
    place.assert_called_once_with("NVDA", 100.0, 103.0, market_price=120.0)
    update.assert_called_once_with("NVDA", 103.0, 125.0, native_trail=False)


def test_pt_restore_with_a_failing_write_is_not_unprotected():
    place, _, queue, send = _pt_failed_sale([[], []], set_id_effect=POOL)
    place.assert_called_once()
    queue.assert_not_called()
    assert _sent(send, "UNPROTECTED") == 0 and _sent(send, "STOP RECORD NOT UPDATED") == 1


# ── 2. Floor on TRIM, manual sell and the failed-sell restore ──────────

def _trim(reads, set_id_effect=None, record=90.0):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=100.0)
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=70.0)]
    approval = {"id": 1, "ticker": "NVDA", "action": "TRIM", "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
                "risk_json": {"shares_initial": 30, "price_at_sizing": 120.0,
                              "stop_loss_price": 90.0}}
    with patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for", return_value=1), \
         patch("ibkr.place_market_order",
               return_value={"order_id": 1, "fill_price": 120.0, "filled_qty": 30.0}), \
         patch("ibkr.get_latest_price", return_value=120.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("config.USE_BRACKET_ORDERS", False), \
         patch("db.log_trade"), patch("db.log_order", return_value=1), \
         patch("db.update_order_fill"), patch("db.log_trade_outcome"), \
         patch("db.get_position_stop", return_value={"stop_price": record, "trailing_high": 125.0}), \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect), \
         patch("db.update_trailing_stop") as update, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action"), \
         patch("alerts.send_trade_alert"), patch("alerts._send") as send:
        execution.execute_approved(approval)
    return place, update, queue, send


def test_trim_re_place_is_floored_at_a_resting_trail():
    place, update, _, send = _trim([[_stp(103.0, qty=100.0)], []])
    place.assert_called_once_with("NVDA", 70.0, 103.0, market_price=120.0)
    update.assert_called_once_with("NVDA", 103.0, 125.0, native_trail=False)
    assert _sent(send, "UNPROTECTED") == 0


def test_trim_re_place_at_the_record_without_a_higher_stop():
    place, update, _, _ = _trim([[_stp(90.0, qty=100.0, order_type="STP")], []])
    place.assert_called_once_with("NVDA", 70.0, 90.0, market_price=120.0)
    update.assert_not_called()


def test_trim_re_place_with_a_failing_write_is_not_unprotected():
    place, _, queue, send = _trim([[], []], set_id_effect=POOL)
    place.assert_called_once()
    queue.assert_not_called()
    assert _sent(send, "UNPROTECTED") == 0 and _sent(send, "STOP RECORD NOT UPDATED") == 1


def _manual_sell(reads, set_id_effect=None, sell_effect=None):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=5.0, current_price=110.0)
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=3.0)]
    sell = (patch("ibkr.place_market_order", side_effect=sell_effect) if sell_effect else
            patch("ibkr.place_market_order",
                  return_value={"order_id": 7, "fill_price": 110.0, "filled_qty": 2.0}))
    with patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for", return_value=1), sell, \
         patch("ibkr.get_latest_price", return_value=110.0), \
         patch("market_data.get_priced_quote", return_value={"price": 110.0}), \
         patch("db.log_trade"), patch("db.log_order", return_value=5), \
         patch("db.update_order_fill"), patch("db.close_trade_outcome"), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0, "trailing_high": 115.0}), \
         patch("db.set_stop_broker_order_id", side_effect=set_id_effect), \
         patch("db.update_trailing_stop") as update, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action"), \
         patch("alerts._send") as send:
        r = execution.execute_manual("NVDA", "SELL", 2, "test")
    return r, place, update, queue, send


def test_manual_sell_re_place_is_floored_at_a_resting_trail():
    _, place, update, _, _ = _manual_sell([[_stp(104.0, qty=5.0)], []])
    place.assert_called_once_with("NVDA", 3.0, 104.0, market_price=110.0)
    update.assert_called_once_with("NVDA", 104.0, 115.0, native_trail=False)


def test_manual_sell_re_place_with_a_failing_write_is_not_unprotected():
    _, place, _, queue, send = _manual_sell([[], []], set_id_effect=POOL)
    place.assert_called_once()
    queue.assert_not_called()
    assert _sent(send, "UNPROTECTED") == 0 and _sent(send, "STOP RECORD NOT UPDATED") == 1


def test_manual_sell_failure_restores_at_the_resting_trail():
    _, place, update, _, _ = _manual_sell([[_stp(104.0, qty=5.0)], []],
                                          sell_effect=RuntimeError("rejected"))
    place.assert_called_once_with("NVDA", 5.0, 104.0)
    update.assert_called_once_with("NVDA", 104.0, 115.0, native_trail=False)


def test_failed_sell_restore_is_floored_and_raises_the_record():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=3, current_price=100.0)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0, "trailing_high": 99.0}), \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop") as update, \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
         patch("alerts._send"):
        execution._restore_stop_after_failed_sell("NVDA", floor=95.0)
    place.assert_called_once_with("NVDA", 3.0, 95.0)
    update.assert_called_once_with("NVDA", 95.0, 99.0, native_trail=False)


def test_floor_before_cancel_comes_from_the_pre_cancel_read_with_no_extra_read():
    prices: list = []
    with patch("tasks._resting_sell_stops", return_value=[_stp(92.0), _stp(97.0)]) as read:
        assert execution._stops_still_resting("NVDA", prices_out=prices) == 14.0
        assert execution._floor_before_cancel("NVDA", prices, "x") == 97.0
        assert execution._floor_before_cancel("NVDA", [], "x") is None
    assert read.call_count == 1


def test_floor_before_cancel_with_an_unreadable_price_alerts_and_uses_the_record():
    with patch("tasks._resting_sell_stops", return_value=[_stp(None)]), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send") as send:
        prices: list = []
        execution._stops_still_resting("NVDA", prices_out=prices)
        assert execution._floor_before_cancel("NVDA", prices, "x") is None
    assert _sent(send, "STOP FLOOR UNREAD") == 1


# ── 3. ADD: the fill's market price, implausible values ignored ─────────

def _add(market_price, resting=101.0):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=15, avg_entry_price=105.0,
                                                      current_price=1.0)   # not used
    with patch("db.get_position_stop", return_value={"stop_price": 98.0}), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", return_value=[_stp(resting)]), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("alerts._send"):
        stop, _ = execution._whole_position_stop("NVDA", 95.0, 5, market_price=market_price)
    return stop


@pytest.mark.parametrize("price", [None, math.nan, 0.0, -5.0, 2e9, "x"])
def test_add_with_an_implausible_price_still_floors(price):
    assert _add(price) == 101.0


def test_add_floor_uses_the_fill_price_not_the_position_price():
    assert _add(110.0) == 101.0          # position current_price 1.0 would have refused it


def test_add_floor_at_or_above_the_market_is_not_used():
    assert _add(101.0) == 98.0
    assert _add(100.0) == 98.0


def test_buy_add_passes_the_fill_price_to_the_whole_position_stop():
    approval = {"id": 1, "ticker": "NVDA", "action": "ADD", "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
                "risk_json": {"shares_initial": 5, "price_at_sizing": 100.0,
                              "stop_loss_price": 90.0}}
    seen = {}

    def whole(ticker, stop, filled, status=None, market_price=None):
        seen["market_price"] = market_price
        raise RuntimeError("stop here")
    with patch("ibkr.get_ib"), patch("ibkr.get_client"), \
         patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
         patch("ibkr.place_market_order",
               return_value={"order_id": 1, "fill_price": 101.5, "filled_qty": 5.0}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("config.USE_BRACKET_ORDERS", False), \
         patch("db.log_trade"), patch("db.log_order", return_value=1), \
         patch("db.update_order_fill"), patch("db.log_trade_outcome"), \
         patch("execution._whole_position_stop", side_effect=whole), \
         patch("alerts._send"), patch("alerts.send_trade_alert"):
        rm_cls.return_value.is_halted.return_value = False
        execution.execute_approved(approval)
    assert seen.get("market_price") == 101.5


# ── 5. Profit-target mark guard ─────────────────────────────────────────

def _pt_run(rm, stops, mark_effect, order):
    with patch("agents.market_regime.get_cached_regime",
               return_value={"trade_style": "Trending/Expanding"}), \
         patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
         patch("db.get_all_position_stops", return_value=stops), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("db.log_trade"), patch("db.log_order", return_value=1), \
         patch("db.update_order_fill"), patch("db.set_order_reconcile_note"), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 2}), \
         patch("ibkr.get_client"), \
         patch("db.mark_profit_target_taken", side_effect=mark_effect) as mark, \
         patch("ibkr.place_market_order", new=order), \
         patch("alerts._send") as send:
        result = rm.check_profit_targets()
    return result, mark, send


def _pt_rm():
    client = MagicMock()
    client.get_all_positions.return_value = [
        MagicMock(symbol="SNOW", qty=100.0, current_price=130.0),
        MagicMock(symbol="NVDA", qty=100.0, current_price=130.0)]
    rm = object.__new__(risk_manager.RiskManager)
    rm._client = client
    return rm


def _pt_stops(opened="2026-11-01"):
    return [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
             "profit_target_taken": False, "opened_at": opened},
            {"ticker": "NVDA", "entry_price": 100.0, "thesis_bucket": "Accelerator",
             "profit_target_taken": False, "opened_at": opened}]


def test_failed_mark_does_not_abort_the_loop_or_allow_a_second_sale():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    result, mark, send = _pt_run(rm, _pt_stops(), POOL, order)
    # both tickers sold once: the failure on the first didn't abort the loop
    assert [r["ticker"] for r in result] == ["SNOW", "NVDA"]
    assert order.call_count == 2
    assert _sent(send, "PROFIT TARGET NOT RECORDED") == 2
    # same process, next run, mark still failing: no second sale
    result, mark, send = _pt_run(rm, _pt_stops(), POOL, order)
    assert result == [] and order.call_count == 2
    assert mark.call_count == 2                   # retried
    assert _sent(send, "PROFIT TARGET NOT RECORDED") == 0   # alerted once


def test_marker_clears_once_the_mark_lands():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    _pt_run(rm, _pt_stops(), POOL, order)
    _pt_run(rm, _pt_stops(), None, order)        # retry succeeds
    assert risk_manager._PT_TAKEN_UNRECORDED == {}
    assert order.call_count == 2


def test_marker_does_not_block_a_new_position():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    _pt_run(rm, _pt_stops("2026-11-01"), POOL, order)
    result, _, _ = _pt_run(rm, _pt_stops("2026-11-03"), None, order)
    assert len(result) == 2 and order.call_count == 4


def test_breakeven_early_mark_failure_returns_false():
    with patch("db.get_position_stop", return_value={"stop_price": 105.0}), \
         patch("db.mark_profit_target_taken", side_effect=POOL), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        assert _rm()._breakeven_without_sale("NVDA", 1, 100.0) is False
    cancel.assert_not_called()


# ── 6. Ladder ratchet: failed placement after the cancel ────────────────

def _ratchet(place_effect, reads, record=92.0):
    rm = _rm(100.0)
    broker = MagicMock()
    broker.place.side_effect = list(place_effect)
    alerts: list = []
    with patch("db.get_position_stop", return_value={"stop_price": record}), \
         patch("db.get_trade_ladder", return_value={"rungs": [{"price": 95.0, "confluence_score": 1}]}), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._queue_reprotect") as queue, patch("tasks.open_manual_action") as manual, \
         patch("alerts._send") as send, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop") as update:
        rm._ratchet_ladder_stop("NVDA", MagicMock(qty=7), 100.0, record, alerts)
    return broker, alerts, queue, manual, send, update


def test_failed_ratchet_restores_the_previous_stop():
    broker, alerts, queue, _, _, _ = _ratchet([RuntimeError("rejected"), {"order_id": 5}],
                                              [[_stp(92.0, order_type="STP")], []])
    assert broker.mock_calls == [call.cancel("NVDA"),
                                 call.place("NVDA", 7.0, 95.0, market_price=100.0),
                                 call.place("NVDA", 7.0, 92.0, market_price=100.0)]
    queue.assert_not_called()


def test_failed_ratchet_restore_at_a_higher_resting_stop_raises_the_record():
    # record 91, resting 93 (below the 95 rung, so the ratchet goes ahead)
    broker, _, _, _, _, update = _ratchet([RuntimeError("rejected"), {"order_id": 5}],
                                          [[_stp(93.0, order_type="STP")], []], record=91.0)
    assert broker.place.call_args_list[-1] == call("NVDA", 7.0, 93.0, market_price=100.0)
    update.assert_called_once_with("NVDA", 93.0, 100.0, native_trail=False)


def test_failed_ratchet_with_the_new_stop_resting_places_no_restore():
    broker, _, queue, _, _, _ = _ratchet([RuntimeError("worker lost")],
                                         [[_stp(92.0, order_type="STP")],
                                          [_stp(95.0, order_type="STP")]])
    assert broker.place.call_count == 1
    queue.assert_not_called()


def test_failed_ratchet_unknown_result_places_no_restore():
    broker, _, queue, _, send, _ = _ratchet([tasks.OrdersCallUnknown("x")],
                                            [[_stp(92.0, order_type="STP")], []])
    assert broker.place.call_count == 1
    assert _sent(send, "STOP STATUS UNKNOWN") == 1


def test_failed_ratchet_with_unreadable_coverage_queues():
    broker, _, queue, _, send, _ = _ratchet([RuntimeError("rejected")],
                                            [[_stp(92.0, order_type="STP")], TimeoutError("read")])
    assert broker.place.call_count == 1
    queue.assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    assert _sent(send, "CHECK STOPS") == 1


def test_failed_ratchet_restore_failure_queues_and_opens_a_manual_action():
    broker, alerts, queue, manual, _, _ = _ratchet([RuntimeError("rejected"), RuntimeError("down")],
                                                   [[_stp(92.0, order_type="STP")], []])
    queue.assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    manual.assert_called_once()
    assert [a for a in alerts if a["action"] == "FIX_STOP"]


def test_ratchet_cancel_failure_places_nothing():
    rm = _rm(100.0)
    broker = MagicMock()
    broker.cancel.side_effect = RuntimeError("cancel failed")
    with patch("db.get_position_stop", return_value={"stop_price": 92.0}), \
         patch("db.get_trade_ladder", return_value={"rungs": [{"price": 95.0, "confluence_score": 1}]}), \
         patch("tasks._resting_sell_stops", return_value=[_stp(92.0, order_type="STP")]), \
         patch("tasks.manage_cancel_stop_orders_for", new=broker.cancel), \
         patch("tasks.manage_stop_place", new=broker.place), \
         patch("tasks._queue_reprotect") as queue:
        rm._ratchet_ladder_stop("NVDA", MagicMock(qty=7), 100.0, 92.0, [])
    broker.place.assert_not_called()
    queue.assert_not_called()


# ── A stop still PendingCancel is not cover for a restore ──────────────

def _pending(price, qty):
    return {**_stp(price, qty=qty, order_type="STP"), "status": "PendingCancel"}


def test_restore_cover_ignores_pending_cancel_stops_only():
    with patch("tasks._resting_sell_stops",
               return_value=[_pending(92.0, 5.0), _stp(94.0, qty=2.0)]):
        assert execution._restore_cover("NVDA") == 2.0
        assert execution._stops_still_resting("NVDA") == 7.0   # unchanged for sell checks
    assert "PendingCancel" not in tasks._NOT_WORKING


def test_trail_restore_ignores_a_pending_cancel_stop():
    r = _convert(RuntimeError("x"), [[_stp(92.0, order_type="STP")], [_pending(92.0, 7.0)]])
    r["broker"].place.assert_called_once_with("NVDA", 7.0, 92.0, market_price=101.0)


def test_breakeven_restore_ignores_a_pending_cancel_stop():
    _, place, _, _, _, _ = _breakeven([[], [_pending(90.0, 2.0)]],
                                      [RuntimeError("down"), {"order_id": 6}])
    assert place.call_count == 2
    assert place.call_args_list[-1] == call("NVDA", 2.0, 90.0, market_price=110.0)


def test_pt_restore_ignores_a_pending_cancel_stop():
    place, _, _, _ = _pt_failed_sale([[], [], [_pending(90.0, 100.0)]])
    place.assert_called_once_with("NVDA", 100.0, 90.0, market_price=120.0)


def test_ratchet_restore_ignores_a_pending_cancel_stop():
    broker, _, _, _, _, _ = _ratchet([RuntimeError("rejected"), {"order_id": 5}],
                                     [[_stp(92.0, order_type="STP")], [_pending(92.0, 7.0)]])
    assert broker.place.call_args_list[-1] == call("NVDA", 7.0, 92.0, market_price=100.0)


def test_failed_sell_restore_ignores_a_pending_cancel_stop():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=3)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", return_value=[_pending(90.0, 3.0)]), \
         patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
         patch("alerts._send"):
        execution._restore_stop_after_failed_sell("NVDA")
    place.assert_called_once_with("NVDA", 3.0, 90.0)


def test_exit_failure_restore_ignores_a_pending_cancel_stop():
    # the sell path's cancel left the stop PendingCancel, then the sell failed
    _, place, _, _, _ = _manual_sell([[_stp(90.0, qty=5.0, order_type="STP")],
                                      [_pending(90.0, 5.0)]],
                                     sell_effect=RuntimeError("rejected"))
    # the post-cancel check holds the sell (a stop still rests) and restores
    place.assert_called_once_with("NVDA", 5.0, 90.0)


# ── PT marker: an ADD before the mark lands re-arms the target ─────────

def test_add_before_the_mark_lands_re_arms_the_target():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    _pt_run(rm, _pt_stops(), POOL, order)                 # sold, mark failed
    assert risk_manager._PT_TAKEN_UNRECORDED["SNOW"]["held_after"] == 75.0
    # same position, more shares than after the sale: an ADD
    rm._client.get_all_positions.return_value = [
        MagicMock(symbol="SNOW", qty=120.0, current_price=130.0),
        MagicMock(symbol="NVDA", qty=75.0, current_price=130.0)]
    result, mark, _ = _pt_run(rm, _pt_stops(), None, order)
    assert [r["ticker"] for r in result] == ["SNOW"]      # SNOW re-armed and sold again
    assert "SNOW" not in risk_manager._PT_TAKEN_UNRECORDED
    # NVDA unchanged: its mark is retried, not re-sold
    assert call("NVDA", 100.0) in mark.call_args_list
    assert "NVDA" not in risk_manager._PT_TAKEN_UNRECORDED


def test_entry_price_change_before_the_mark_lands_re_arms_the_target():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    _pt_run(rm, _pt_stops(), POOL, order)
    rm._client.get_all_positions.return_value = [
        MagicMock(symbol="SNOW", qty=75.0, current_price=130.0),
        MagicMock(symbol="NVDA", qty=75.0, current_price=130.0)]
    stops = _pt_stops()
    stops[0]["entry_price"] = 101.0                       # SNOW's average cost moved
    result, _, _ = _pt_run(rm, stops, None, order)
    assert [r["ticker"] for r in result] == ["SNOW"]


def test_unchanged_position_keeps_the_marker_after_the_sale():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    _pt_run(rm, _pt_stops(), POOL, order)
    rm._client.get_all_positions.return_value = [
        MagicMock(symbol="SNOW", qty=75.0, current_price=130.0),
        MagicMock(symbol="NVDA", qty=75.0, current_price=130.0)]
    result, _, _ = _pt_run(rm, _pt_stops(), POOL, order)
    assert result == [] and order.call_count == 2


def test_a_position_read_still_at_the_pre_sale_quantity_is_not_an_add():
    # the position cache lags the fill: still 100 shares on the next run
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    _pt_run(rm, _pt_stops(), POOL, order)
    result, _, _ = _pt_run(rm, _pt_stops(), POOL, order)
    assert result == [] and order.call_count == 2


# ── Profit-target sale held while a stop still works after the cancel ──

def _pt_sale(reads, record=90.0, alert_due=True):
    sell = MagicMock(return_value={"filled_qty": 30, "order_id": 7, "fill_price": 120.0})
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=70)
    with patch("db.get_position_stop", return_value={"stop_price": record, "trailing_high": 125.0}), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks._reconcile_alert_due", return_value=alert_due) as due, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.log_trade"), patch("db.log_order", return_value=1), \
         patch("db.update_order_fill"), patch("db.set_order_reconcile_note"), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("ibkr.place_market_order", new=sell), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place", return_value={"order_id": 8}) as place, \
         patch("tasks._queue_reprotect") as queue, \
         patch("alerts._send") as send:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0, decision_price=120.0)
    _pt_sale.due = due
    return result, sell, place, queue, send, cancel


def test_pt_sale_held_when_the_stop_is_still_pending_cancel():
    # the cancel returned with the stop PendingCancel: no SELL; protection
    # goes back for all 100 shares (a pending cancel isn't cover)
    stop = {**_stp(90.0, qty=100.0, order_type="STP"), "status": "PendingCancel"}
    result, sell, place, queue, send, cancel = _pt_sale([[stop], [stop], [stop]])
    assert result is None
    cancel.assert_called_once_with("NVDA")
    sell.assert_not_called()
    place.assert_called_once_with("NVDA", 100.0, 90.0, market_price=120.0)
    queue.assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    assert _sent(send, "PROFIT-TARGET SALE HELD") == 1 and _sent(send, "UNPROTECTED") == 0


def test_pt_sale_held_when_a_stop_still_works_after_the_cancel():
    # a working stop the cancel didn't take covers the position: no SELL and
    # no second stop on top of it
    stop = _stp(90.0, qty=100.0, order_type="STP")
    result, sell, place, queue, send, _ = _pt_sale([[stop], [stop], [stop]])
    assert result is None
    sell.assert_not_called()
    place.assert_not_called()
    queue.assert_called_once_with("NVDA", {"intent": "RESTORE"}, None)
    assert _sent(send, "PROFIT-TARGET SALE HELD") == 1


def test_pt_sale_held_when_the_post_cancel_read_fails():
    result, sell, place, queue, send, _ = _pt_sale([[], TimeoutError("read")])
    assert result is None
    sell.assert_not_called()
    place.assert_not_called()          # coverage unknown: nothing stacked
    assert queue.call_count == 2       # the restore's own queue and the hold's
    assert _sent(send, "CHECK STOPS") == 1
    held = [c[0][0] for c in send.call_args_list if "PROFIT-TARGET SALE HELD" in c[0][0]]
    assert len(held) == 1
    assert "Protection not restored yet; queued" in held[0]
    assert "Protection was put back" not in held[0]


def test_pt_sale_goes_ahead_after_a_confirmed_cancel():
    result, sell, place, queue, send, _ = _pt_sale([[_stp(90.0, qty=100.0, order_type="STP")], []])
    sell.assert_called_once_with("NVDA", "SELL", 30)
    place.assert_called_once_with("NVDA", 70.0, 100.0, market_price=120.0)
    assert result["new_stop"] == 100.0
    queue.assert_not_called()
    assert _sent(send, "PROFIT-TARGET SALE HELD") == 0


def test_held_pt_sale_is_not_marked_taken():
    rm = _pt_rm()
    order = MagicMock(return_value={"order_id": 1, "filled_qty": 25})
    stop = {**_stp(90.0, qty=100.0, order_type="STP"), "status": "PendingCancel"}
    with patch("agents.market_regime.get_cached_regime",
               return_value={"trade_style": "Trending/Expanding"}), \
         patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
         patch("db.get_all_position_stops", return_value=_pt_stops()), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks._resting_sell_stops", return_value=[stop]), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 2}), \
         patch("tasks._queue_reprotect"), patch("ibkr.get_client"), \
         patch("db.mark_profit_target_taken") as mark, \
         patch("ibkr.place_market_order", new=order), \
         patch("alerts._send"):
        result = rm.check_profit_targets()
    assert result == []
    order.assert_not_called()
    mark.assert_not_called()


# ── Held-sale alert throttle and the skip while a working stop remains ─

def _working(price, qty, order_id, status="Submitted"):
    return {**_stp(price, qty=qty, order_type="STP"), "order_id": order_id, "status": status}


def test_held_sale_alert_is_throttled_per_ticker():
    stop = _working(90.0, 100.0, 41)
    _, _, _, _, send, _ = _pt_sale([[stop]], alert_due=False)
    assert _sent(send, "PROFIT-TARGET SALE HELD") == 0
    assert call("pt_sale_held_NVDA") in _pt_sale.due.call_args_list


def test_held_sale_alert_says_protection_was_put_back_when_placed():
    stop = {**_working(90.0, 100.0, 41), "status": "PendingCancel"}
    _, _, _, _, send, _ = _pt_sale([[stop]])
    held = [c[0][0] for c in send.call_args_list if "PROFIT-TARGET SALE HELD" in c[0][0]]
    assert len(held) == 1 and "Protection was put back" in held[0]


def test_a_working_stop_hold_skips_the_target_until_that_stop_is_gone():
    foreign = _working(90.0, 100.0, 41)
    # run 1: the cancel leaves stop 41 working: held, skip recorded
    result, sell, _, _, _, _ = _pt_sale([[foreign]])
    assert result is None and sell.call_count == 0
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {"NVDA": {("order", 41)}}
    # run 2: stop 41 still works: skipped before any cancel
    result, sell, place, _, _, cancel = _pt_sale([[foreign, _working(90.0, 100.0, 77)]])
    assert result is None
    cancel.assert_not_called(); sell.assert_not_called(); place.assert_not_called()
    # run 3: stop 41 is gone (only our restored 77 rests): skip cleared, the
    # sale runs and its own cancel confirms
    result, sell, _, _, _, cancel = _pt_sale([[_working(90.0, 100.0, 77)],
                                              [_working(90.0, 100.0, 77)], []])
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {}
    cancel.assert_called_once_with("NVDA")
    sell.assert_called_once_with("NVDA", "SELL", 30)
    assert result["new_stop"] == 100.0


def test_a_pending_cancel_hold_does_not_skip_the_next_run():
    stop = {**_working(90.0, 100.0, 41), "status": "PendingCancel"}
    _pt_sale([[stop]])
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {}


def test_skip_stays_while_the_stops_cannot_be_read():
    risk_manager._PT_HELD_BY_WORKING_STOP["NVDA"] = {("order", 41)}
    result, sell, _, _, _, cancel = _pt_sale([TimeoutError("read")])
    assert result is None
    cancel.assert_not_called(); sell.assert_not_called()
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {"NVDA": {("order", 41)}}


def test_restore_previous_stop_reports_its_outcome():
    rm = _rm()
    with patch("tasks._resting_sell_stops", side_effect=_reads(TimeoutError("read"))), \
         patch("tasks._queue_reprotect"), patch("alerts._send"):
        assert rm._restore_previous_stop("NVDA", 5, 90.0, None, None, "x", None) == "unread"
    with patch("tasks._resting_sell_stops", return_value=[_stp(90.0, qty=5.0, order_type="STP")]):
        assert rm._restore_previous_stop("NVDA", 5, 90.0, None, None, "x", None) == "covered"
    with patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks.manage_stop_place", side_effect=RuntimeError("down")):
        assert rm._restore_previous_stop("NVDA", 5, 90.0, None, None, "x", None) == "failed"
    with patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}), \
         patch("db.set_stop_broker_order_id"):
        assert rm._restore_previous_stop("NVDA", 5, 90.0, None, None, "x", None) == "placed"



# ── Skip identity by permId, and the throttled skip reminder ────────────

def test_stop_identity_prefers_perm_id_and_falls_back_to_order_id():
    assert risk_manager._stop_identity({"perm_id": 9001, "order_id": 41}) == ("perm", 9001)
    assert risk_manager._stop_identity({"perm_id": 0, "order_id": 41}) == ("order", 41)
    assert risk_manager._stop_identity({"perm_id": None, "order_id": 0}) == ("order", 0)
    assert risk_manager._stop_identity({"order_id": 7}) == ("order", 7)


def test_skip_is_keyed_by_perm_id():
    held_by = {**_working(90.0, 100.0, 41), "perm_id": 9001}
    _pt_sale([[held_by]])
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {"NVDA": {("perm", 9001)}}
    # the same stop now shows orderId 0 (a TWS-owned view): still blocking
    same = {**_working(90.0, 100.0, 0), "perm_id": 9001}
    result, sell, _, _, _, cancel = _pt_sale([[same]])
    assert result is None
    cancel.assert_not_called(); sell.assert_not_called()


def test_another_clients_stop_with_the_same_order_id_does_not_block():
    held_by = {**_working(90.0, 100.0, 41), "perm_id": 9001}
    _pt_sale([[held_by]])
    # orderId 41 again, but another permId: a different order, so the
    # blocking stop is gone and the sale runs (and re-checks after its cancel)
    other = {**_working(90.0, 100.0, 41), "perm_id": 5555}
    result, sell, _, _, _, cancel = _pt_sale([[other], [other], []])
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {}
    cancel.assert_called_once_with("NVDA")
    sell.assert_called_once_with("NVDA", "SELL", 30)


def test_order_id_fallback_blocks_until_that_order_is_gone():
    manual = {**_working(90.0, 100.0, 0), "perm_id": 0}
    _pt_sale([[manual]])
    assert risk_manager._PT_HELD_BY_WORKING_STOP == {"NVDA": {("order", 0)}}
    result, sell, _, _, _, _ = _pt_sale([[manual]])
    assert result is None and sell.call_count == 0


def test_skip_sends_a_throttled_reminder():
    risk_manager._PT_HELD_BY_WORKING_STOP["NVDA"] = {("perm", 9001)}
    blocking = {**_working(90.0, 100.0, 41), "perm_id": 9001}
    result, sell, _, _, send, _ = _pt_sale([[blocking]])
    assert result is None and sell.call_count == 0
    assert _sent(send, "PROFIT TARGET STILL SKIPPED") == 1
    assert call("pt_sale_skipped_NVDA") in _pt_sale.due.call_args_list
    # within the throttle window: still skipped, no reminder
    result, sell, _, _, send, _ = _pt_sale([[blocking]], alert_due=False)
    assert result is None and sell.call_count == 0
    assert _sent(send, "PROFIT TARGET STILL SKIPPED") == 0


def test_skip_reminder_also_covers_an_unreadable_read():
    risk_manager._PT_HELD_BY_WORKING_STOP["NVDA"] = {("perm", 9001)}
    _, _, _, _, send, _ = _pt_sale([TimeoutError("read")])
    assert _sent(send, "PROFIT TARGET STILL SKIPPED") == 1


def test_skip_reminder_not_sent_when_the_throttle_is_unreadable():
    risk_manager._PT_HELD_BY_WORKING_STOP["NVDA"] = {("perm", 9001)}
    blocking = {**_working(90.0, 100.0, 41), "perm_id": 9001}
    with patch("tasks._resting_sell_stops", return_value=[blocking]), \
         patch("tasks._reconcile_alert_due", side_effect=RuntimeError("pool")), \
         patch("alerts._send") as send:
        assert risk_manager._pt_hold_still_blocked("NVDA") is True
    send.assert_not_called()


# ── A cancelling stop gets a second look; a restore over one is re-checked ─

def test_restore_cover_second_look_sees_a_slow_cancel_land():
    out = []
    with patch("tasks._resting_sell_stops",
               side_effect=_reads([_pending(92.0, 5.0), _stp(94.0, qty=2.0)],
                                  [_stp(94.0, qty=2.0)])) as read, \
         patch("time.sleep") as sleep:
        assert execution._restore_cover("NVDA", out) == 2.0
    sleep.assert_called_once_with(5)
    assert read.call_count == 2
    assert out == [0.0]


def test_restore_cover_without_a_cancelling_stop_reads_once():
    with patch("tasks._resting_sell_stops", return_value=[_stp(94.0, qty=2.0)]) as read, \
         patch("time.sleep") as sleep:
        assert execution._restore_cover("NVDA") == 2.0
    sleep.assert_not_called()
    assert read.call_count == 1


def _failed_sell_restore(reads, qty=3):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=qty, current_price=100.0)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", side_effect=_reads(*reads)), \
         patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
         patch("alerts._send"):
        execution._restore_stop_after_failed_sell("NVDA")
    return place


def test_failed_sell_restore_after_a_slow_cancel_lands_queues_no_recheck(_recheck):
    # the TRIM's cancel was only slow: gone on the second look, so one
    # restore for the position and nothing to re-check
    place = _failed_sell_restore([[_pending(90.0, 3.0)], []])
    place.assert_called_once_with("NVDA", 3.0, 90.0)
    _recheck.assert_not_called()


def test_failed_sell_restore_over_a_still_cancelling_stop_queues_a_recheck(_recheck):
    # still PendingCancel after the second look: the restore still goes on
    # (unprotected is worse than over-stopped) and a re-check is queued in
    # case the cancel is rejected and the old stop reverts
    place = _failed_sell_restore([[_pending(90.0, 3.0)], [_pending(90.0, 3.0)]])
    place.assert_called_once_with("NVDA", 3.0, 90.0)
    _recheck.assert_called_once_with(args=["NVDA"], kwargs={"attempt": 1}, countdown=60)


def test_failed_sell_restore_survives_a_queue_failure(_recheck):
    _recheck.side_effect = RuntimeError("broker down")
    place = _failed_sell_restore([[_pending(90.0, 3.0)]])
    place.assert_called_once_with("NVDA", 3.0, 90.0)


def test_sized_restore_over_a_still_cancelling_stop_queues_a_recheck(_recheck):
    with patch("tasks._resting_sell_stops", return_value=[_pending(90.0, 5.0)]), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        assert _rm()._restore_previous_stop("NVDA", 5, 90.0, None, None, "x", None) == "placed"
    place.assert_called_once_with("NVDA", 5.0, 90.0, market_price=None)
    _recheck.assert_called_once_with(args=["NVDA"], kwargs={"attempt": 1}, countdown=60)


def test_sized_restore_after_a_slow_cancel_lands_queues_no_recheck(_recheck):
    with patch("tasks._resting_sell_stops", side_effect=_reads([_pending(90.0, 5.0)], [])), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        assert _rm()._restore_previous_stop("NVDA", 5, 90.0, None, None, "x", None) == "placed"
    place.assert_called_once_with("NVDA", 5.0, 90.0, market_price=None)
    _recheck.assert_not_called()


def test_recheck_trims_a_reverted_cancel(_recheck):
    # the cancel was rejected: both stops work, so _fix_overstop trims
    with patch("tasks._resting_sell_stops", return_value=[_stp(90.0, qty=3.0), _stp(90.0, qty=3.0)]), \
         patch("tasks._fix_overstop_safe", return_value="🔧 NVDA: trimmed") as fix, \
         patch("alerts._send") as send:
        tasks.recheck_stop_coverage("NVDA")
    fix.assert_called_once_with("NVDA", recheck_try=2)
    assert _sent(send, "STOP RECONCILE — 🔧 NVDA") == 1
    _recheck.assert_not_called()


def test_recheck_with_coverage_fine_sends_nothing(_recheck):
    with patch("tasks._resting_sell_stops", return_value=[_stp(90.0, qty=3.0)]), \
         patch("tasks._fix_overstop_safe",
               return_value="NVDA: stop coverage fine on re-check (3 vs 3)") as fix, \
         patch("alerts._send") as send:
        tasks.recheck_stop_coverage("NVDA")
    fix.assert_called_once_with("NVDA", recheck_try=2)
    send.assert_not_called()


def test_recheck_waits_while_a_cancel_is_still_pending(_recheck):
    # _fix_overstop would count the cancelling stop as cover: wait instead
    with patch("tasks._resting_sell_stops", return_value=[_pending(90.0, 3.0), _stp(90.0, qty=3.0)]), \
         patch("tasks._fix_overstop_safe") as fix:
        tasks.recheck_stop_coverage("NVDA")
        _recheck.assert_called_once_with(args=["NVDA"], kwargs={"attempt": 2}, countdown=60)
        _recheck.reset_mock()
        tasks.recheck_stop_coverage("NVDA", attempt=tasks._COVERAGE_RECHECK_TRIES)
    fix.assert_not_called()
    _recheck.assert_not_called()   # last try: left to reconcile


def test_recheck_with_unreadable_stops_does_nothing(_recheck):
    with patch("tasks._resting_sell_stops", side_effect=TimeoutError("read")), \
         patch("tasks._fix_overstop_safe") as fix:
        tasks.recheck_stop_coverage("NVDA")
    fix.assert_not_called()
    _recheck.assert_not_called()


# ── A sized restore uses the shares held now ───────────────────────────

def _sized_restore(positions, held=7.0, covered=()):
    client = MagicMock()
    if isinstance(positions, Exception):
        client.get_all_positions.side_effect = positions
    else:
        client.get_all_positions.return_value = positions
    with patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", return_value=list(covered)), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place:
        outcome, _ = execution._restore_sized_stop("NVDA", held, 90.0, "x", market_price=100.0)
    return outcome, place


def test_sized_restore_uses_a_fresh_position_read():
    # 7 when the caller read it; 4 now (part of a sale filled since)
    outcome, place = _sized_restore([MagicMock(symbol="NVDA", qty=4.0)],
                                    covered=[_stp(90.0, qty=1.0, order_type="STP")])
    assert outcome == "placed"
    place.assert_called_once_with("NVDA", 3.0, 90.0, market_price=100.0)


def test_sized_restore_keeps_the_callers_qty_when_the_read_fails():
    _, place = _sized_restore(TimeoutError("read"))
    place.assert_called_once_with("NVDA", 7.0, 90.0, market_price=100.0)


def test_sized_restore_keeps_the_callers_qty_when_the_read_shows_none():
    # the portfolio cache can lag a fill: a flat read is not trusted
    _, place = _sized_restore([MagicMock(symbol="AMD", qty=10.0)])
    place.assert_called_once_with("NVDA", 7.0, 90.0, market_price=100.0)


@pytest.mark.parametrize("fresh, sized", [({"NVDA": 5.0}, 5.0), ({"NVDA": 9.0}, 7.0),
                                          ({}, 7.0), (TimeoutError("read"), 7.0)])
def test_sized_restore_uses_the_smaller_of_cache_and_fresh_read(fresh, sized):
    # cache 7; a fresh read below it wins, one above it or without the
    # ticker (or failing) leaves the cache's 7
    kw = {"side_effect": fresh} if isinstance(fresh, Exception) else {"return_value": fresh}
    with patch("ibkr.get_fresh_stock_positions", **kw):
        _, place = _sized_restore([MagicMock(symbol="NVDA", qty=7.0)])
    place.assert_called_once_with("NVDA", sized, 90.0, market_price=100.0)


# ── Held PT sale with no recorded stop says nothing was put back ───────

def test_held_sale_alert_without_an_old_stop_says_none_was_put_back():
    with patch("tasks._queue_reprotect"), patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_stop_place") as place, \
         patch("alerts._send") as send:
        _rm()._hold_profit_target_sale("NVDA", 100, None, None, 120.0, None, None)
    place.assert_not_called()
    held = [c[0][0] for c in send.call_args_list if "PROFIT-TARGET SALE HELD" in c[0][0]]
    assert len(held) == 1
    assert "No previous stop was recorded, so none was put back" in held[0]
    assert "Protection was put back" not in held[0]


def test_held_sale_alert_when_working_stops_cover_says_nothing_was_placed():
    stop = _stp(90.0, qty=100.0, order_type="STP")
    _, _, place, _, send, _ = _pt_sale([[stop], [stop], [stop]])
    place.assert_not_called()
    held = [c[0][0] for c in send.call_args_list if "PROFIT-TARGET SALE HELD" in c[0][0]]
    assert len(held) == 1
    assert "No stop was placed: working stops already cover the position" in held[0]
    assert "Protection was put back" not in held[0]


# ── The failed-sell restore's floor sees the position's last price ─────

def test_failed_sell_restore_floor_at_the_market_uses_the_record():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=3, current_price=100.0)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop") as update, \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks.manage_stop_place", return_value={"order_id": 5}) as place, \
         patch("alerts._send"):
        execution._restore_stop_after_failed_sell("NVDA", floor=101.0)
    place.assert_called_once_with("NVDA", 3.0, 90.0)
    update.assert_not_called()
