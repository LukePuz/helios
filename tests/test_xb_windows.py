"""Slow-call windows on the stop re-places (cross-branch review of
batch-1124, 2026-11-21): no broker read or alert send between the point
where the shares are bare (or the cancel) and the place, and every held alert
still goes out after the place, once.

- S1: tasks._reprotect_after_late_sell's floor_and_guard (queued reprotect
  and late-SELL re-place): the coverage read is reused by the no-loosen
  floor and the above-market guard (none when nothing rests); their alert
  texts are sent after manage_stop_place.
- S3: execution._whole_position_stop (ADD merge): the stop-size, floor-read
  and guard alerts are held in status["alerts_out"] and execute_approved
  sends them after its place loop; the guard reuses the floor's read.
- N2: ibkr.get_today_sell_executions keeps the session on a timeout.
S2 (the profit-target remainder count from cached_stock_qty) is tested in
tests/test_stop_qty_more.py."""
import importlib.util
import inspect
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
import execution
import ibkr
import tasks

_H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_order_row_before_fill_wait.py")
_spec = importlib.util.spec_from_file_location("_order_row_harness_xbw", _H)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

READS = ("read:open_orders", "read:resting", "read:minute", "read:positions")


def _stp(price, qty, status="Submitted"):
    return {"ticker": "CRWD", "action": "SELL", "order_type": "STP", "qty": qty,
            "stop_price": price, "status": status}


def _alerts(events):
    return [e for e in events if isinstance(e, tuple) and e[0] == "alert"]


def _reprotect(intent, resting_seq, from_queue, place_raises=None, order_extra=None):
    """One _reprotect_after_late_sell run on 20 CRWD shares with a 150 stop
    record and a 140 fill a minute ago (the guard's price, so no minute bar
    is read; the shadow guard alerts, 150 being above it). resting_seq: what
    each _resting_sell_stops read returns (an Exception raises)."""
    events = []
    seq = list(resting_seq)

    def resting(*a, **k):
        events.append("read:resting")
        r = seq.pop(0) if seq else []
        if isinstance(r, Exception):
            raise r
        return r

    def rec(name, ret=None, raises=None):
        def fn(*a, **k):
            events.append(name)
            if raises is not None:
                raise raises
            return ret
        return fn

    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=20)]
    order = {"intent": intent, "own_stop": 130.0, **(order_extra or {})}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=rec("read:open_orders", [])), \
         patch("ibkr.get_fresh_minute_price", side_effect=rec("read:minute", 140.0)), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("db.mark_profit_target_taken"), \
         patch("tasks._resting_sell_stops", side_effect=resting), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=rec("cancel", 1)), \
         patch("tasks.manage_stop_place", side_effect=rec("place", {"order_id": 3}, place_raises)), \
         patch("db.set_stop_broker_order_id"), patch("db.log_activity_queued"), \
         patch("tasks._record_floored_stop"), patch("tasks._record_guarded_stop"), \
         patch("tasks._record_seeded_stop"), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("time.sleep"), \
         patch("alerts._send", side_effect=lambda t, *a, **k: events.append(("alert", t.splitlines()[0]))):
        out = tasks._reprotect_after_late_sell("CRWD", order, None, from_queue=from_queue,
                                               exit_avg=140.0,
                                               exit_fill_at=datetime.now(timezone.utc))
    return out, events


def _nothing_slow_between(events, start, end="place"):
    """No broker read or alert between events[start] and the place."""
    j = events.index(end)
    between = events[start + 1:j]
    assert not [e for e in between if e in READS or (isinstance(e, tuple) and e[0] == "alert")], events
    return j


# ---- S1: queued reprotect ----

def test_queued_bare_shares_read_once_and_alert_after_the_place():
    """A late OPEN with nothing resting (covered 0): no floor read, the guard
    gets resting=[], and its shadow alert goes out after the place."""
    out, events = _reprotect("BUY", [[]], from_queue=True)
    assert out is True
    assert events.count("read:resting") == 1 and "cancel" not in events
    j = _nothing_slow_between(events, events.index("read:resting"))
    al = _alerts(events)
    assert len(al) == 1 and "stop guard (shadow)" in al[0][1]
    assert events.index(al[0]) > j


def test_queued_partly_covered_reuses_the_cover_read_before_the_cancel():
    """5 of 20 covered by a stop at 145: the floor and the guard use that
    read; the only other read is the existing look after the cancel."""
    out, events = _reprotect("BUY", [[_stp(145.0, 5.0)], []], from_queue=True)
    assert out is True
    first = events.index("read:resting")
    assert events[first + 1] == "cancel"                  # nothing between the read and the cancel
    assert events.count("read:resting") == 2              # the cover read + the after-cancel look
    j = events.index("place")
    assert not _alerts(events[:j])                        # no alert before the place
    assert len(_alerts(events)) == 1                      # the guard's, after it


def test_queued_alert_still_sent_once_when_the_place_raises():
    out, events = _reprotect("BUY", [[]], from_queue=True, place_raises=RuntimeError("ib"))
    assert out is False
    j = events.index("place")
    al = _alerts(events)
    assert len(al) == 1 and events.index(al[0]) > j


def test_queued_already_covered_sends_nothing_and_places_nothing():
    out, events = _reprotect("BUY", [[_stp(145.0, 20.0)]], from_queue=True)
    assert out is True and "place" not in events and _alerts(events) == []


# ---- S1: late-SELL re-place ----

def test_late_trim_reuses_the_cover_read_and_alerts_after_the_place():
    out, events = _reprotect("TRIM", [[_stp(145.0, 5.0)]], from_queue=False)
    assert out is True
    i = events.index("read:resting")
    assert events[i + 1:i + 3] == ["cancel", "place"]     # nothing slow between
    assert events.count("read:resting") == 1
    al = _alerts(events)
    assert len(al) == 1 and events.index(al[0]) > events.index("place")


def test_late_profit_target_floor_read_failure_is_alerted_after_the_place():
    """A profit-target re-place doesn't read cover, so the floor reads once;
    that read fails: the guard doesn't read again, and both the floor's
    STOP FLOOR UNREAD and the guard's alert go out after the place."""
    out, events = _reprotect("PROFIT_TARGET", [TimeoutError("read")], from_queue=False,
                             order_extra={"stop_price": 150.0})
    assert out is True
    assert events.count("read:resting") == 1
    j = _nothing_slow_between(events, events.index("read:resting"))
    al = _alerts(events)
    assert [a for a in al if "STOP FLOOR UNREAD" in a[1]] and len(al) == 2
    assert all(events.index(a) > j for a in al)


def test_late_sell_alerts_sent_once_when_the_place_raises():
    out, events = _reprotect("TRIM", [[]], from_queue=False, place_raises=RuntimeError("ib"))
    assert out is False
    al = _alerts(events)
    assert len(al) == 1 and events.index(al[0]) > events.index("place")


# ---- S3: ADD whole-position stop ----

def _wps(resting_effect, broker_qty=1.0):
    """_whole_position_stop for an ADD of 1 on 1 held (2 expected) with a
    record at 150, a fill at 140 and its own stop at 130; the broker count is
    broker_qty (1: undersized, alerted)."""
    events = []
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=2, avg_entry_price=145.0)

    def resting(*a, **k):
        events.append("read:resting")
        if isinstance(resting_effect, Exception):
            raise resting_effect
        return list(resting_effect)
    status = {"alerts_out": []}
    with patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(2.0, 145.0)), \
         patch("ibkr.cached_stock_qty", return_value=broker_qty), \
         patch("tasks._resting_sell_stops", side_effect=resting), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued"), \
         patch("alerts._send", side_effect=lambda t, *a, **k: events.append(("alert", t))), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=lambda *a, **k: events.append("cancel")):
        out = execution._whole_position_stop("CRWD", 130.0, 1.0, status=status, market_price=140.0,
                                             action="ADD", held_before=1.0, price_is_fill=True)
    return out, status, events


@pytest.fixture
def add_count_on(monkeypatch):
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", True)   # shipped True; conftest off


@pytest.mark.usefixtures("add_count_on")
def test_add_merge_holds_its_alerts_and_reads_resting_once():
    out, status, events = _wps([_stp(120.0, 1.0)])
    assert _alerts(events) == []                          # nothing sent before the cancel
    assert events == ["read:resting", "cancel"]           # the floor's read, reused by the guard
    held = status["alerts_out"]
    assert any("ADD STOP MAY NOT COVER" in t for t in held)
    assert any("stop guard (shadow)" in t for t in held)
    assert len(held) == 2


@pytest.mark.usefixtures("add_count_on")
def test_add_merge_floor_read_failure_is_held_and_not_repeated():
    out, status, events = _wps(TimeoutError("read"), broker_qty=2.0)
    assert events == ["read:resting", "cancel"]
    held = status["alerts_out"]
    assert any("STOP FLOOR UNREAD" in t for t in held) and _alerts(events) == []


@pytest.mark.usefixtures("add_count_on")
def test_add_merge_without_a_list_still_sends_inline():
    """Callers that don't pass status["alerts_out"] keep today's behaviour."""
    events = []
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=2, avg_entry_price=145.0)
    with patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(2.0, 145.0)), \
         patch("ibkr.cached_stock_qty", return_value=1.0), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued"), \
         patch("alerts._send", side_effect=lambda t, *a, **k: events.append(t)), \
         patch("tasks.manage_cancel_stop_orders_for"):
        execution._whole_position_stop("CRWD", 130.0, 1.0, status={}, market_price=140.0,
                                       action="ADD", held_before=1.0, price_is_fill=True)
    assert len(events) == 2


def _execute_with_held(manage_place):
    """execute_approved's BUY branch with a _whole_position_stop that holds
    two texts in status["alerts_out"]."""
    fake = harness.FakeOrders()
    events = []
    place, _ = harness._broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def wps(ticker, new_stop, filled_qty, status=None, **k):
        events.append("wps")
        status["alerts_out"].extend(["held one", "held two"])
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
                  patch("alerts._send", side_effect=lambda t, *a, **k: events.append(("alert", t))),
                  patch("alerts.send_trade_alert"),
                  patch("execution._whole_position_stop", side_effect=wps),
                  patch("execution._write_stop_record", return_value=True),
                  patch("execution._record_add_lot_r"),
                  patch("execution._capture_original_thesis_if_new"),
                  patch("tasks._queue_reprotect"),
                  patch("tasks._apply_buy_outcome"),
                  patch("tasks.manage_stop_place", side_effect=manage_place(events))):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        execution.execute_approved(harness._open_approval())
    return events


def _placed(events):
    def place(*a, **k):
        events.append("place")
        return {"order_id": 9}
    return place


def _raising(events):
    def place(*a, **k):
        events.append("place")
        raise RuntimeError("ib")
    return place


@pytest.mark.parametrize("manage_place", [_placed, _raising])
def test_execute_sends_the_held_texts_after_the_place_loop_once(manage_place):
    events = _execute_with_held(manage_place)
    last_place = len(events) - 1 - events[::-1].index("place")
    held = [i for i, e in enumerate(events) if e in (("alert", "held one"), ("alert", "held two"))]
    assert len(held) == 2                                 # once each
    assert all(i > last_place for i in held)
    assert events.index("wps") < events.index("place")


# ---- N2, N4 ----

def test_today_sell_executions_keep_the_session_on_a_timeout():
    src = inspect.getsource(ibkr.get_today_sell_executions)
    assert 'ib.reqExecutions, drop_session=False' in src
    assert "ib.RequestTimeout" in inspect.getsource(ibkr._bounded_req)


def test_shipped_defaults_these_paths_run_under():
    """As config.py ships them (conftest turns the broker counts off)."""
    import ast
    tree = ast.parse(open(config.__file__).read())
    shipped = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert shipped["STOP_GUARD_MODE"] == "shadow"
    assert shipped["PT_STOP_QTY_FROM_BROKER"] is True and shipped["ADD_STOP_QTY_FROM_BROKER"] is True


def test_held_texts_are_sent_when_the_whole_position_stop_raises():
    """Review follow-up: the _whole_position_stop call sits inside the
    try/finally, so texts it held before raising still go out."""
    fake = harness.FakeOrders()
    events = []
    place, _ = harness._broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}

    def wps(ticker, new_stop, filled_qty, status=None, **k):
        status["alerts_out"].append("held before the raise")
        raise RuntimeError("boom")

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
                  patch("alerts._send", side_effect=lambda t, *a, **k: events.append(("alert", t))),
                  patch("alerts.send_trade_alert"),
                  patch("execution._whole_position_stop", side_effect=wps),
                  patch("execution._write_stop_record", return_value=True),
                  patch("execution._record_add_lot_r"),
                  patch("execution._capture_original_thesis_if_new"),
                  patch("tasks._queue_reprotect"), patch("tasks._apply_buy_outcome"),
                  patch("tasks.manage_stop_place", return_value={"order_id": 9})):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        execution.execute_approved(harness._open_approval())
    assert events.count(("alert", "held before the raise")) == 1


def test_cached_stock_qty_is_signed():
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.positions.return_value = [MagicMock(contract=MagicMock(symbol="NVDA", secType="STK"), position=-2.0),
                                 MagicMock(contract=MagicMock(symbol="NVDA", secType="OPT"), position=5.0)]
    with patch.object(ibkr, "_ib", ib):
        assert ibkr.cached_stock_qty("NVDA") == -2.0
    assert "signed" in ibkr.cached_stock_qty.__doc__


def test_the_enforce_gap_is_noted_at_the_guard_switch():
    src = open(config.__file__).read()
    i = src.index("STOP_GUARD_MODE = ")
    note = src[i - 2000:i]
    assert "floor_read_failed" in note and "reread_skipped_shadow" in note
    assert "Enforce prerequisites" in note
    assert "RESTORE" in note and "drops the" in note and "accept the drop" in note
