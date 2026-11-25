"""Alert ordering on two record/alert paths (cross-branch review 2 of
batch-1124, 2026-11-22).

- S1: execute_approved's BUY stop block sends the texts
  _whole_position_stop held back (status["alerts_out"]) after the place
  loop's own alerts: the STOP RECORD MISSING / CHECK STOPS alerts and, on
  three failed places, the reprotect queueing and the UNPROTECTED alert,
  which no longer wait behind those informational sends. If the block
  raises, the held texts still go out once before it propagates.
- S2: risk_manager._hwm_guard_send: a failed hwm-guard alert send puts the
  day's dedupe key back, so the next unusable-mark check sends again.
"""
import importlib.util
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import alert_policy
import config
import execution
import risk_manager
from risk_manager import TRADING_ENV as ENV

_H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_order_row_before_fill_wait.py")
_spec = importlib.util.spec_from_file_location("_order_row_harness_hao", _H)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

HELD = (("alert", "held one"), ("alert", "held two"))


# ---- S1: execute_approved ----

def _execute(manage_place, record_ok=True, wps_status=None, wps_raises=None,
             prior_record=None, extra=(), events=None):
    """execute_approved's BUY branch with a _whole_position_stop that holds
    two texts. Events: "wps", "place", "reprotect", "manual", ("alert", text)."""
    fake = harness.FakeOrders()
    events = [] if events is None else events
    place, _ = harness._broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def wps(ticker, new_stop, filled_qty, status=None, **k):
        events.append("wps")
        status["alerts_out"].extend(["held one", "held two"])
        status.update(wps_status or {})
        if wps_raises:
            raise wps_raises
        return 90.0, (95.0, 8.0)

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
                  patch("db.get_position_stop",
                        return_value=prior_record or {"stop_price": 92.0}),
                  patch("alerts._send", side_effect=lambda t, *a, **k: events.append(("alert", t))),
                  patch("alerts.send_trade_alert"),
                  patch("execution._whole_position_stop", side_effect=wps),
                  patch("execution._write_stop_record", return_value=record_ok),
                  patch("execution._record_add_lot_r"),
                  patch("execution._record_entry_basis"),
                  patch("execution._record_entry_ref_age"),
                  patch("execution._capture_original_thesis_if_new"),
                  patch("tasks._queue_reprotect", side_effect=lambda *a, **k: events.append("reprotect")),
                  patch("tasks.open_manual_action", side_effect=lambda *a, **k: events.append("manual")),
                  patch("tasks._apply_buy_outcome"),
                  patch("tasks.manage_stop_place", side_effect=manage_place(events)),
                  *extra):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        try:
            execution.execute_approved(harness._open_approval())
        except Exception as e:      # execute_approved's own handling decides; record it
            events.append(("raised", type(e).__name__))
    return events


def _placed(events):
    def place(*a, **k):
        events.append("place")
        return {"order_id": 9}
    return place


def _failing(events):
    def place(*a, **k):
        events.append("place")
        raise RuntimeError("ib")
    return place


def _first(events, pred):
    return next(i for i, e in enumerate(events) if pred(e))


def _alert_index(events, prefix):
    return _first(events, lambda e: isinstance(e, tuple) and e[0] == "alert"
                  and str(e[1]).startswith(prefix))


def _held_indexes(events):
    idx = [i for i, e in enumerate(events) if e in HELD]
    assert len(idx) == 2, events                           # once each
    return idx


def test_three_failed_places_queue_and_alert_unprotected_before_the_held_texts():
    events = _execute(_failing)
    assert events.count("place") == 3
    held = _held_indexes(events)
    unprotected = _alert_index(events, "🚨 UNPROTECTED POSITION")
    assert events.index("reprotect") < unprotected < min(held)
    assert events.index("manual") < min(held)
    last_place = len(events) - 1 - events[::-1].index("place")
    # Nothing sent between the last failed place and the UNPROTECTED alert
    # except the reprotect queueing.
    assert events[last_place + 1:unprotected] == ["reprotect"]


def test_a_missing_record_alert_goes_before_the_held_texts():
    events = _execute(_placed, record_ok=False)
    held = _held_indexes(events)
    assert _alert_index(events, "⚠️ STOP RECORD MISSING") < min(held)
    assert events.index("place") < min(held)


def test_a_prior_unread_alert_goes_before_the_held_texts():
    events = _execute(_placed, wps_status={"prior_unreadable": True})
    held = _held_indexes(events)
    assert _alert_index(events, "⚠️ CHECK STOPS") < min(held)


def test_a_clean_place_sends_the_held_texts_once_after_it():
    events = _execute(_placed)
    held = _held_indexes(events)
    assert events.index("place") < min(held)
    assert not any(isinstance(e, tuple) and str(e[1]).startswith("🚨") for e in events)


def test_a_raise_in_the_place_block_still_sends_the_held_texts_once():
    events = _execute(_placed, wps_raises=RuntimeError("boom"))
    assert events.count(HELD[0]) == 1 and events.count(HELD[1]) == 1
    assert "place" not in events


class _Probe(Exception):
    pass


def test_a_raise_after_the_place_loop_still_sends_the_held_texts_once(caplog):
    """Review SF-1: the record-missing alert raising after a placed stop. The
    raise still reaches execute_approved's own handler, as before."""
    events = _execute(_placed, record_ok=False,
                      extra=(patch("execution._alert_stop_record_missing", side_effect=_Probe("x")),))
    assert events.index("place") < min(_held_indexes(events))
    assert "Order execution failed: x" in caplog.text


def test_a_malformed_kept_record_still_sends_the_held_texts_once(caplog):
    """Review SF-1: float() on a kept record's stop_price in the prior-unread alert."""
    events = _execute(_placed, wps_status={"prior_unreadable": True},
                      prior_record={"stop_price": "not a number"})
    _held_indexes(events)
    assert "could not convert string to float" in caplog.text


def test_a_raise_in_the_not_placed_block_still_sends_the_held_texts_once():
    """A BaseException past the block's own except Exception (the reprotect
    queueing): sent once, and it propagates."""
    events = []
    with pytest.raises(KeyboardInterrupt):
        _execute(_failing, events=events,
                 extra=(patch("tasks._queue_reprotect", side_effect=KeyboardInterrupt),))
    _held_indexes(events)


def test_a_base_exception_sends_the_held_texts_and_propagates():
    events = []
    with pytest.raises(KeyboardInterrupt):
        _execute(_placed, wps_raises=KeyboardInterrupt(), events=events)
    _held_indexes(events)


def test_the_held_texts_have_one_send_site_in_a_finally():
    import inspect
    src = inspect.getsource(execution.execute_approved)
    assert src.count('tasks._send_alert_texts(prior_info.get("alerts_out") or [])') == 1
    assert "except BaseException" not in src


# ---- S2: risk_manager._hwm_guard_send ----

KEY = f"hwm_guard_alerted_{ENV}_non_positive"
TODAY = "2026-11-23"


def _send_with(send, state, verify_fails=False):
    """_hwm_guard_send with alerts._send = send over the dict state."""
    def verified(writes):
        if verify_fails:
            return [k for k, _ in writes]
        state.update(writes)
        return []
    with patch("alerts._send", send), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state_verified", side_effect=verified) as sv, \
         patch("db.set_state") as ss:
        risk_manager._hwm_guard_send(KEY, state.get("prev"), TODAY, "text")
    ss.assert_not_called()                                    # only the verified write
    return sv


@pytest.mark.parametrize("send", [MagicMock(return_value=None),
                                  MagicMock(side_effect=RuntimeError("telegram"))])
@pytest.mark.parametrize("prev, restored", [("2026-01-01", "2026-01-01"), (None, "")])
def test_a_failed_send_puts_the_key_back(send, prev, restored, caplog):
    send.reset_mock()
    state = {KEY: TODAY, "prev": prev}
    sv = _send_with(send, state)
    assert send.call_args.kwargs == {"severity": alert_policy.CRITICAL}
    sv.assert_called_once_with([(KEY, restored)])
    assert state[KEY] == restored and "put back so the next check sends it" in caplog.text


def test_a_sent_alert_leaves_the_key():
    state = {KEY: TODAY}
    sv = _send_with(MagicMock(return_value=101), state)
    sv.assert_not_called()
    assert state[KEY] == TODAY


@pytest.mark.parametrize("now", ["2026-11-24", "", None])
def test_a_key_that_moved_on_is_left_alone(now, caplog):
    """Review NIT-2: a midnight crossing or another writer's value is not reverted."""
    state = {KEY: now, "prev": "2026-01-01"}
    sv = _send_with(MagicMock(return_value=None), state)
    sv.assert_not_called()
    assert state[KEY] == now and "left as it is" in caplog.text


def test_an_unverified_restore_says_so(caplog):
    """Review NIT-3: set_state swallows failures; the verified write is checked."""
    state = {KEY: TODAY}
    _send_with(MagicMock(return_value=None), state, verify_fails=True)
    assert "could not be put back" in caplog.text
    assert "put back so the next check sends it" not in caplog.text


def test_a_failed_restore_never_raises(caplog):
    with patch("alerts._send", return_value=None), patch("db.get_state", return_value=TODAY), \
         patch("db.set_state_verified", side_effect=RuntimeError("db")):
        risk_manager._hwm_guard_send(KEY, None, TODAY, "text")
    assert "dedupe key restore failed" in caplog.text


class _SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.target, self.args, self.kwargs, self.daemon = target, args, kwargs or {}, daemon

    def start(self):
        self.target(*self.args, **self.kwargs)


def _check(state, send):
    strict = {f"high_water_mark_{ENV}": "0"}

    def verified(writes):
        for k, v in writes:
            state[k] = v
        return []
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1000.0)
    with patch.object(config, "HWM_GUARD_MODE", "shadow"), \
         patch("db.get_state_strict", side_effect=lambda k, d=None: strict.get(k, d)), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.set_state_verified", side_effect=verified), \
         patch("db.log_halt_event"), patch("db.log_activity_queued"), \
         patch("alerts._send", send), \
         patch.object(risk_manager.threading, "Thread", _SyncThread):
        return risk_manager.RiskManager(client).is_halted()


def test_after_a_failed_send_the_next_check_that_day_sends_again():
    state = {KEY: "2026-01-01"}
    send = MagicMock(side_effect=[None, 101])
    assert _check(state, send) is False
    assert state[KEY] == "2026-01-01"                       # put back
    assert _check(state, send) is False
    assert send.call_count == 2 and state[KEY] != "2026-01-01"
    assert _check(state, send) is False                     # sent: deduped for the day
    assert send.call_count == 2
