"""Alerts sent after the stop work they report on, not between a cancel and
the place or restore that follows it (sweep "cancel-to-place windows",
2026-11-21): a Telegram send can take tens of seconds while the position has
no stop. Each test checks the order of events and that the alert still goes
out when the place or restore raises."""
import importlib.util
import inspect
import json
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import execution
import ibkr
import tasks

_H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_order_row_before_fill_wait.py")
_spec = importlib.util.spec_from_file_location("_order_row_harness_cpa", _H)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def _recorder(events, name, ret=None, raises=None):
    def fn(*a, **k):
        events.append(name)
        if raises is not None:
            raise raises
        return ret
    return fn


def _alert_recorder(events):
    def send(text, *a, **k):
        events.append(("alert", str(text).splitlines()[0]))
        return 1
    return send


# ---- 1. execute_approved SELL: a stop that would not cancel ----

def _run_sell_stuck(after_cancel, restore_raises=None):
    """A TRIM whose cancel left stops resting (after_cancel = the re-read:
    a share count, or None for unreadable). Returns the event list."""
    fake = harness.FakeOrders()
    events = []
    reads = iter([0.0, after_cancel])
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=42.0)
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client),
                  patch("tasks.manage_cancel_stop_orders_for",
                        side_effect=_recorder(events, "cancel", ret=1)),
                  patch("execution._stops_still_resting",
                        side_effect=lambda t, prices_out=None: next(reads)),
                  patch("ibkr.place_market_order",
                        side_effect=AssertionError("sold on top of a resting stop")),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=None),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("config.USE_BRACKET_ORDERS", False),
                  patch("alerts._send", side_effect=_alert_recorder(events)),
                  patch("alerts.send_trade_alert"),
                  patch("execution._restore_stop_after_failed_sell",
                        side_effect=_recorder(events, "restore", raises=restore_raises)),
                  patch("execution._schedule_held_sell_retry",
                        side_effect=_recorder(events, "schedule"))):
            st.enter_context(p)
        # A raising restore reaches execute_approved's own handler: a failed outcome.
        assert execution.execute_approved(harness._approval("TRIM")) is False
    return events


def test_sell_stuck_stop_restores_before_the_alert():
    events = _run_sell_stuck(4.0)
    assert events[:2] == ["cancel", "restore"]
    assert events[2][0] == "alert" and "NOT SENT" in events[2][1]
    assert "schedule" not in events


def test_sell_unreadable_after_cancel_restores_and_schedules_before_the_alert():
    events = _run_sell_stuck(None)
    assert events[:3] == ["cancel", "restore", "schedule"]
    assert events[3][0] == "alert" and "NOT SENT" in events[3][1]


def test_sell_stuck_alert_still_sent_when_the_restore_raises():
    events = _run_sell_stuck(None, restore_raises=RuntimeError("pool exhausted"))
    assert events[:2] == ["cancel", "restore"]
    assert events[-1][0] == "alert" and "NOT SENT" in events[-1][1]


# ---- 2. backstop sell ----

_BACKSTOP_ALERT = {"current_price": 140.0, "stop_price": 150.0}


def _backstop(place_effect, open_orders, retry=False, due=True):
    events = []
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]

    def place(*a, **k):
        events.append("sell")
        if isinstance(place_effect, BaseException):
            raise place_effect
        return place_effect

    with patch("ibkr.get_client", return_value=client), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=_recorder(events, "cancel")), \
         patch("ibkr.get_open_orders", return_value=open_orders), \
         patch("ibkr.place_market_order", side_effect=place), \
         patch("ibkr.get_broker_stock_qty", return_value=0.0), patch("ibkr.get_ib"), \
         patch("db.log_trade", side_effect=_recorder(events, "log_trade")), \
         patch("db.delete_position_stop", side_effect=_recorder(events, "delete_stop")), \
         patch("db.log_order", side_effect=_recorder(events, "log_order", ret=5)), \
         patch("db.update_order_fill", side_effect=_recorder(events, "update_fill")), \
         patch("db.close_trade_outcome", side_effect=_recorder(events, "close_outcome")), \
         patch("tasks._schedule_backstop_retry", side_effect=_recorder(events, "schedule_retry")), \
         patch("tasks._claim_backstop_retry_send", return_value=True), \
         patch("tasks._reconcile_alert_due", return_value=due) as due_check, \
         patch("alerts._send", side_effect=_alert_recorder(events)):
        out = (tasks.retry_backstop_sell.run("AMD", _BACKSTOP_ALERT) if retry
               else tasks._execute_backstop_sell("AMD", _BACKSTOP_ALERT))
    _backstop.due_keys = [c.args[0] for c in due_check.call_args_list]
    return out, events


def _stop(qty):
    return {"order_id": 41, "ticker": "AMD", "action": "SELL", "order_type": "STP", "qty": qty,
            "stop_price": 150.0, "status": "PreSubmitted", "remaining": qty}


def _mkt(rem):
    return {"order_id": 77, "ticker": "AMD", "action": "SELL", "order_type": "MKT", "qty": 10.0,
            "stop_price": None, "status": "Submitted", "remaining": rem}


def test_backstop_books_the_sale_before_any_alert():
    """Review SF1: a worker killed during the sends must not leave a sold
    position unbooked, so every alert comes after the orders and trade rows."""
    out, events = _backstop({"order_id": 78, "filled_qty": 4, "fill_price": 140.0},
                            [_stop(4.0), _mkt(2.0)])
    assert out is True
    first_alert = next(i for i, e in enumerate(events) if isinstance(e, tuple))
    assert events[:first_alert] == ["cancel", "sell", "log_order", "update_fill", "log_trade"]
    alerts = [e[1] for e in events[first_alert:]]
    assert len(alerts) == 2 and all("BACKSTOP" in a for a in alerts)


def test_backstop_full_sale_books_and_closes_before_the_alerts():
    out, events = _backstop({"order_id": 78, "filled_qty": 10, "fill_price": 140.0}, [])
    assert out is True and not any(isinstance(e, tuple) for e in events)   # nothing held, nothing sent
    assert events == ["cancel", "sell", "log_order", "update_fill", "log_trade", "delete_stop",
                      "close_outcome"]


def test_backstop_held_text_goes_out_first_when_the_sell_raises():
    """Review SF2: the held text, then the except path's own (as before)."""
    out, events = _backstop(RuntimeError("broker down"), [_stop(4.0)], retry=True)
    assert events[:2] == ["cancel", "sell"]
    alerts = [e[1] for e in events if isinstance(e, tuple)]
    assert len(alerts) == 2
    assert alerts[0].startswith("⚠️ BACKSTOP") and "RETRY FAILED" in alerts[1]


def test_backstop_held_text_sent_once_on_a_first_try_failure():
    out, events = _backstop(RuntimeError("broker down"), [_stop(4.0)])
    assert out is False
    alerts = [e for e in events if isinstance(e, tuple)]
    assert len(alerts) == 1 and "BACKSTOP" in alerts[0][1]
    assert events.index(alerts[0]) < events.index("schedule_retry")


def test_backstop_alert_still_sent_when_nothing_is_left_to_sell():
    out, events = _backstop({"filled_qty": 0}, [_stop(10.0)])   # the stuck stop covers all 10
    assert out is False
    assert "sell" not in events
    assert len(events) == 2 and events[1][0] == "alert" and "BACKSTOP" in events[1][1]


def test_send_alert_texts_throttles_keyed_entries_at_send_time():
    """Professor m118/m124: an (issue_key, text) entry is sent only when
    _reconcile_alert_due(issue_key); a plain text is never checked."""
    texts = ["plain", ("key_due", "due text"), ("key_not_due", "throttled text")]
    with patch("tasks._reconcile_alert_due", side_effect=lambda k: k == "key_due") as due, \
         patch("alerts._send") as send:
        tasks._send_alert_texts(texts)
    assert [c.args[0] for c in send.call_args_list] == ["plain", "due text"]
    assert [c.args[0] for c in due.call_args_list] == ["key_due", "key_not_due"]
    assert texts == []


def test_send_alert_texts_sends_when_the_throttle_check_fails():
    texts = [("key", "text")]
    with patch("tasks._reconcile_alert_due", side_effect=RuntimeError("db down")), \
         patch("alerts._send") as send:
        tasks._send_alert_texts(texts)
    send.assert_called_once_with("text")
    assert texts == []


def _throttle_state():
    """In-memory system_state for the real _reconcile_alert_due."""
    st = {}
    return st, patch("db.get_state", side_effect=lambda k, d=None: st.get(k, d)), \
        patch("db.set_state", side_effect=lambda k, v: st.__setitem__(k, v))


def _age_s(st, key):
    from datetime import datetime, timezone
    return (datetime.now(timezone.utc) - datetime.fromisoformat(st[f"reconcile_alert_{key}"])).total_seconds()


def test_a_sent_keyed_alert_spends_the_throttle_window():
    st, gs, ss = _throttle_state()
    with gs, ss, patch("alerts._send", return_value=41) as send:
        tasks._send_alert_texts([("queued_exit_dropped_AMD", "dropped")])
        tasks._send_alert_texts([("queued_exit_dropped_AMD", "dropped")])   # the next pass
    assert send.call_count == 1
    assert _age_s(st, "queued_exit_dropped_AMD") < 60


@pytest.mark.parametrize("failure", [{"return_value": None}, {"side_effect": RuntimeError("telegram")}])
def test_a_failed_keyed_send_is_due_again_after_the_retry_wait_not_every_pass(failure):
    """Review of 33fe560: the issue key was recorded before the send, so a
    failed 'QUEUED … DROPPED' send spent the whole window, and a later drop
    within it was silenced although nothing went out. Now the issue is due
    again after _HELD_ALERT_RETRY_S: not at once (no send every pass while
    Telegram is down), not after the full window."""
    import config
    key = "queued_exit_dropped_AMD"
    st, gs, ss = _throttle_state()
    with gs, ss, patch("alerts._send", **failure) as send:
        tasks._send_alert_texts([(key, "dropped")])
        window = config.RECONCILE_ALERT_THROTTLE_HOURS * 3600
        assert abs(_age_s(st, key) - (window - tasks._HELD_ALERT_RETRY_S)) < 60
        for _ in range(3):                                   # the next passes, minutes later
            tasks._send_alert_texts([(key, "dropped")])
        assert send.call_count == 1
    # the retry wait passes: due again, and this time it goes out
    from datetime import datetime, timedelta
    st[f"reconcile_alert_{key}"] = (datetime.fromisoformat(st[f"reconcile_alert_{key}"])
                                    - timedelta(seconds=tasks._HELD_ALERT_RETRY_S + 60)).isoformat()
    with gs, ss, patch("alerts._send", return_value=42) as send:
        tasks._send_alert_texts([(key, "dropped again")])
        tasks._send_alert_texts([(key, "dropped again")])
    assert [c.args[0] for c in send.call_args_list] == ["dropped again"]
    assert _age_s(st, key) < 60                              # sent: the full window from now


def test_a_failed_plain_send_writes_no_throttle_state():
    st, gs, ss = _throttle_state()
    with gs, ss, patch("alerts._send", return_value=None):
        tasks._send_alert_texts(["plain"])
    assert st == {}


def test_a_failed_retry_stamp_never_raises():
    texts = [("k", "t"), "after"]
    with patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks._reconcile_alert_retry_soon", side_effect=RuntimeError("db down")) as soon, \
         patch("alerts._send", side_effect=[None, 7]) as send:
        tasks._send_alert_texts(texts)
    soon.assert_called_once_with("k")
    assert send.call_count == 2 and texts == []


def test_a_throttled_or_unkeyed_entry_never_restamps():
    with patch("tasks._reconcile_alert_due", return_value=False), \
         patch("tasks._reconcile_alert_retry_soon") as soon, patch("alerts._send") as send:
        tasks._send_alert_texts([("k", "t")])
    send.assert_not_called(); soon.assert_not_called()
    with patch("tasks._reconcile_alert_retry_soon") as soon, patch("alerts._send", return_value=None):
        tasks._send_alert_texts(["plain"])
    soon.assert_not_called()


@pytest.mark.parametrize("due", [True, False])
def test_backstop_stuck_stop_alert_is_throttled_and_the_sale_is_not(due):
    """Professor m124: the 'would not cancel' alert repeated every trigger
    cycle. Throttled at send time; the sale of the uncovered rest stands."""
    out, events = _backstop({"order_id": 78, "filled_qty": 6, "fill_price": 140.0}, [_stop(4.0)], due=due)
    assert "sell" in events
    assert _backstop.due_keys == ["backstop_stop_not_cancelled_AMD"]
    sent = [e[1] for e in events if isinstance(e, tuple)]   # first lines only
    assert sum("BACKSTOP" in s for s in sent) == (1 if due else 0)


@pytest.mark.parametrize("due", [True, False])
def test_backstop_working_sell_alert_is_throttled_and_the_sale_is_not(due):
    working = {"order_id": 77, "ticker": "AMD", "action": "SELL", "order_type": "MKT", "qty": 10.0,
               "stop_price": None, "status": "Submitted", "remaining": 4.0}
    out, events = _backstop({"order_id": 78, "filled_qty": 6, "fill_price": 140.0}, [working], due=due)
    assert "sell" in events
    assert _backstop.due_keys == ["backstop_sell_working_AMD"]
    sent = [e[1] for e in events if isinstance(e, tuple)]   # first lines only
    assert sum("BACKSTOP" in s for s in sent) == (1 if due else 0)


def test_send_alert_texts_empties_the_list_and_never_raises():
    texts = ["a", "b"]
    with patch("alerts._send", side_effect=[RuntimeError("telegram"), 1]) as send:
        tasks._send_alert_texts(texts)
    assert texts == [] and send.call_count == 2
    with patch("alerts._send") as send:
        tasks._send_alert_texts(texts)                 # a second call sends nothing
    send.assert_not_called()


# ---- 3. ADD whole-position stop: queue before the alert, thesis after the place ----

def _run_open_refused(manage_place):
    fake = harness.FakeOrders()
    events = []
    place, _ = harness._broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_ib"), patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=quote),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}),
                  patch("config.USE_BRACKET_ORDERS", False), patch("time.sleep"),
                  patch("db.set_stop_broker_order_id"),
                  patch("alerts._send", side_effect=_alert_recorder(events)),
                  patch("alerts.send_trade_alert"),
                  patch("execution._whole_position_stop", return_value=(90.0, (95.0, 8.0))),
                  patch("execution._write_stop_record", return_value=True),
                  patch("execution._record_add_lot_r"),
                  patch("execution._capture_original_thesis_if_new",
                        side_effect=_recorder(events, "thesis")),
                  patch("tasks._queue_reprotect", side_effect=_recorder(events, "queue")),
                  patch("tasks._apply_buy_outcome"),
                  patch("tasks.manage_stop_place", side_effect=manage_place(events))):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        execution.execute_approved(harness._open_approval())
    return events


def _refused(events):
    def place(*a, **k):
        events.append("place")
        raise ibkr.SellExceedsPosition("ZZZ: SELL 8 refused; broker shows 5 shares")
    return place


def _placed(events):
    def place(*a, **k):
        events.append("place")
        return {"order_id": 9}
    return place


def test_a_refused_whole_position_stop_is_queued_before_the_alert():
    events = _run_open_refused(_refused)
    i = events.index("place")
    assert events[i + 1] == "queue"
    assert events[i + 2][0] == "alert" and "STOP REFUSED" in events[i + 2][1]


def test_the_thesis_is_captured_after_the_stop_place():
    events = _run_open_refused(_placed)
    assert events.index("place") < events.index("thesis")
    events = _run_open_refused(_refused)                # and when the place is refused
    assert events.index("queue") < events.index("thesis")


def _failing(events):
    def place(*a, **k):
        events.append("place")
        raise RuntimeError("rejected")
    return place


def test_the_thesis_comes_after_the_unprotected_alert_on_a_failed_place():
    events = _run_open_refused(_failing)
    assert events.count("place") == 3
    unprotected = next(i for i, e in enumerate(events)
                       if isinstance(e, tuple) and "UNPROTECTED" in e[1])
    assert unprotected < events.index("thesis")


def test_manual_buy_queues_before_the_refused_alert():
    src = inspect.getsource(execution.execute_manual)
    i = src.index('if type(e).__name__ == "SellExceedsPosition":')
    block = src[i:i + 400]
    assert block.index("_queue_stop_for_retry(") < block.index("_alert_stop_refused(")


# ---- 6. the queued re-protect pass ----

def _pending_pass(pending, results):
    state = {tasks._REPROTECT_STATE_KEY: json.dumps(pending)}
    events = []
    it = iter(results)

    def reprotect(ticker, *a, **k):
        events.append(("reprotect", ticker))
        r = next(it)
        if isinstance(r, BaseException):
            raise r
        return r

    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
         patch("tasks._reprotect_after_late_sell", side_effect=reprotect), \
         patch("tasks.open_manual_action"), \
         patch("alerts._send", side_effect=_alert_recorder(events)):
        try:
            tasks._retry_pending_reprotects()
        except RuntimeError:
            events.append("raised")
    return events, state


def test_gave_up_alerts_go_out_after_the_other_tickers_reprotect():
    pending = {"AAA": {"intent": "RESTORE", "tries": tasks._REPROTECT_MAX_TRIES - 1, "nonce": "a"},
               "BBB": {"intent": "RESTORE", "nonce": "b"}}
    events, state = _pending_pass(pending, [False, True])
    assert events[:2] == [("reprotect", "AAA"), ("reprotect", "BBB")]
    assert events[2][0] == "alert" and "STOP NOT RESTORED" in events[2][1]
    assert json.loads(state[tasks._REPROTECT_STATE_KEY]) == {}   # both settled, as before


def test_gave_up_alert_still_sent_when_a_later_ticker_raises():
    pending = {"AAA": {"intent": "RESTORE", "tries": tasks._REPROTECT_MAX_TRIES - 1, "nonce": "a"},
               "BBB": {"intent": "RESTORE", "nonce": "b"}}
    events, _ = _pending_pass(pending, [False, RuntimeError("worker lost")])
    assert events[:2] == [("reprotect", "AAA"), ("reprotect", "BBB")]
    assert events[2][0] == "alert" and "STOP NOT RESTORED" in events[2][1]
    assert events[3:] == ["raised"]                    # sent in the finally, then raised


def test_flat_buy_alert_is_held_to_the_end_too():
    pending = {"AAA": {"intent": "BUY", "tries": tasks._BUY_FLAT_MAX_TRIES - 1, "nonce": "a"},
               "BBB": {"intent": "RESTORE", "nonce": "b"}}
    events, _ = _pending_pass(pending, [None, True])
    assert events[:2] == [("reprotect", "AAA"), ("reprotect", "BBB")]
    assert events[2][0] == "alert" and "STOP NOT PLACED" in events[2][1]


# ---- 7. the reconcile resize guard ----

def test_overstop_guard_alerts_are_collected_not_sent():
    out = []
    with patch("config.STOP_GUARD_MODE", "shadow"), \
         patch("risk_manager._stop_guard_row"), \
         patch("alerts._send", side_effect=AssertionError("sent inside the window")):
        level = tasks._overstop_guarded_level("AMD", 210.0, 200.0, [], out)
    assert level == 210.0 and len(out) == 1 and "stop guard" in out[0]


def _overstop_run(place_raises=False):
    """Under-covered: 10 held, a stop for 6, record $150 below last $200. The
    guard (forced to report) runs before the place; its alert must come after."""
    events = []
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]

    def guard(ticker, level, fallback, mp, what, path, resting=None, record_follows=False,
              price_is_fill=False, alerts_out=None, cancel_window=False):
        events.append("guard")
        alerts_out.append("ℹ️ stop guard (shadow) — test")
        return level

    def place(*a, **k):
        events.append("place")
        if place_raises:
            raise RuntimeError("orders worker lost")
        return {"order_id": 9}

    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[_stop(6.0)]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("risk_manager._stop_below_market", side_effect=guard), \
         patch("tasks.manage_stop_place", side_effect=place), \
         patch("db.set_stop_broker_order_id"), \
         patch("alerts._send", side_effect=_alert_recorder(events)):
        msg = tasks._fix_overstop("AMD")
    return events, msg


def test_overstop_guard_alert_comes_after_the_place():
    events, msg = _overstop_run()
    assert events[:2] == ["guard", "place"] and events[2][0] == "alert"
    assert "placed a stop" in msg


def test_overstop_guard_alert_still_sent_when_the_place_raises():
    events, msg = _overstop_run(place_raises=True)
    assert events[:2] == ["guard", "place"] and events[2][0] == "alert"
    assert "fix failed" in msg


def test_refused_over_position_sends_the_guard_alerts_after_the_place():
    src = inspect.getsource(tasks._fix_refused_over_position)
    assert "_overstop_guarded_level(ticker, stop_price, last, stuck, guard_alerts)" in src
    i = src.index("manage_stop_place(")
    assert src.index("_send_alert_texts(guard_alerts)") > i
