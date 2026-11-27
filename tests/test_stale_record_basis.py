"""stale_record_at_entry rows say which case they are (record only).

entry_record_basis (always): "fresh_after_confirmed_exit" (the old position
looks closed: nothing held before the order or an OPEN over a leftover
record, the post-fill count exactly the fill, and the old stop gone at the
broker or a SELL execution today), "inherited_lagged_read" (the old stop is
still working, or a count was held before the order), else
"inherited_unknown". would_reset (always) is True only for the first.
prior_order_status / prior_sell_fill_seen are read only where the new stop
is already placed (execute_approved after its stop work, the bracket path).
The message no longer claims an exit on the lagged path. Nothing about the
stop record or the orders changes."""
import inspect
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

import pytest

import config
import execution

FRESH, LAGGED, UNKNOWN = ("fresh_after_confirmed_exit", "inherited_lagged_read",
                          "inherited_unknown")


# ── _entry_record_basis (pure) ───────────────────────────────────────────────

@pytest.mark.parametrize("kw, want", [
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "Submitted"}, LAGGED),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "PreSubmitted"}, LAGGED),
    ({"held_before": 2.0, "post_fill_qty": 1}, LAGGED),
    ({"held_before": 2.0, "post_fill_qty": 1, "prior_order_status": "not_open"}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "not_open"}, FRESH),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "Filled"}, FRESH),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "Cancelled"}, FRESH),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_sell_fill_seen": True}, FRESH),
    ({"held_before": 5.0, "post_fill_qty": 1, "over_leftover": True,
      "prior_order_status": "not_open"}, FRESH),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "Inactive"}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "Inactive",
      "prior_sell_fill_seen": True}, FRESH),                         # Inactive is no evidence
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "unknown"}, UNKNOWN),
    ({"held_before": 2.0, "post_fill_qty": 1, "prior_sell_fill_seen": True}, UNKNOWN),
    # The stop path's own read of any working stop (old_stop_state) wins over
    # the record's order id, which can be stale or missing.
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "not_open",
      "prior_sell_fill_seen": True, "old_stop_state": "working"}, LAGGED),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_sell_fill_seen": True,
      "old_stop_state": "working"}, LAGGED),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "not_open",
      "prior_sell_fill_seen": True, "old_stop_state": "unread"}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "not_open",
      "old_stop_state": "absent"}, FRESH),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_order_status": "read_error",
      "prior_sell_fill_seen": None}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": 1, "prior_sell_fill_seen": False}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": 1}, UNKNOWN),               # no broker reads
    ({"held_before": None, "post_fill_qty": 1, "prior_order_status": "not_open"}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": 0, "prior_order_status": "not_open"}, UNKNOWN),
    ({"held_before": 0.0, "post_fill_qty": None, "prior_order_status": "not_open"}, UNKNOWN),
])
def test_basis(kw, want):
    assert execution._entry_record_basis(1, **kw) == want


# ── _stale_entry_evidence ────────────────────────────────────────────────────

class _Exec:
    def __init__(self, ticker="NVDA", side="SLD", order_id=9, perm_id=0):
        self.contract = MagicMock(symbol=ticker)
        self.execution = MagicMock(side=side, orderId=order_id, permId=perm_id)


def _evidence(prior_order, orders=None, orders_error=None, execs=(), execs_error=None,
              recorded=(set(), set()), connected=True):
    ib = MagicMock()
    ib.isConnected.return_value = connected
    ib.RequestTimeout = 0
    if execs_error is not None:
        ib.reqExecutions.side_effect = execs_error
    else:
        ib.reqExecutions.return_value = list(execs)
    with patch("ibkr.get_open_orders", return_value=orders or [],
               side_effect=orders_error) as open_orders, \
         patch("ibkr._ib", ib), \
         patch("ibkr.get_ib") as get_ib, \
         patch("db.recorded_sell_order_ids", return_value=recorded):
        out = execution._stale_entry_evidence("NVDA", prior_order)
        import ibkr
        same_session = ibkr._ib is ib
    get_ib.assert_not_called()                      # no connect, no heartbeat
    return out, open_orders, ib, same_session


def _order(oid, status, ticker="NVDA"):
    return {"order_id": oid, "ticker": ticker, "action": "SELL", "order_type": "STP",
            "qty": 1.0, "status": status}


def test_a_working_old_stop_is_conclusive():
    out, open_orders, ib, _ = _evidence(4242, orders=[_order(4242, "Submitted")])
    assert out == {"prior_order_status": "Submitted"}
    open_orders.assert_called_once_with(timeout=execution._STALE_EVIDENCE_TIMEOUT_S, reconnect=False)
    ib.reqExecutions.assert_not_called()


def test_an_old_stop_not_open_also_reads_todays_sells():
    out, _, ib, _ = _evidence(4242, orders=[_order(4242, "Submitted", ticker="AMD"),
                                            _order(7, "Submitted")],
                              execs=[_Exec(ticker="AMD"), _Exec(side="BOT"), _Exec()])
    assert out == {"prior_order_status": "not_open", "prior_sell_fill_seen": True}
    ib.reqExecutions.assert_called_once_with()


def test_a_listed_order_without_a_status_is_unknown():
    out, _, _, _ = _evidence(4242, orders=[_order(4242, "")])
    assert out["prior_order_status"] == "unknown"
    assert "prior_sell_fill_seen" in out              # not a working stop: the sells are read


def test_sells_the_bot_booked_are_no_exit():
    # A same-day TRIM: booked by order id, or by perm id.
    out, _, _, _ = _evidence(None, execs=[_Exec(order_id=31), _Exec(order_id=5, perm_id=88)],
                             recorded=({31}, {88}))
    assert out == {"prior_sell_fill_seen": False}


def test_unreadable_booked_sells_leave_the_fill_unknown():
    out, _, ib, _ = _evidence(None, execs=[_Exec()], recorded=None)
    assert out == {"prior_sell_fill_seen": None}
    ib.reqExecutions.assert_not_called()


def test_an_executions_timeout_keeps_the_session():
    out, _, _, same_session = _evidence(None, execs_error=TimeoutError())
    assert out == {"prior_sell_fill_seen": None}
    assert same_session                                # drop_session=False


def test_failed_reads_are_marked():
    out, _, _, _ = _evidence(4242, orders_error=ConnectionError("down"),
                             execs_error=RuntimeError("x"))
    assert out == {"prior_order_status": "read_error", "prior_sell_fill_seen": None}


def test_no_order_id_reads_only_the_sells():
    out, open_orders, _, _ = _evidence(None)
    assert out == {"prior_sell_fill_seen": False}
    open_orders.assert_not_called()


def test_no_session_reads_no_sells():
    out, _, ib, _ = _evidence(None, connected=False)
    assert out == {"prior_sell_fill_seen": None}
    ib.reqExecutions.assert_not_called()


# ── _note_stale_record: deferred for the bracket path ────────────────────────

def test_note_stale_record_can_defer_its_row():
    defer: dict = {}
    with patch("db.get_position_stop", return_value={"stop_price": 98.0}), \
         patch("db.get_all_position_stops", return_value=[{"ticker": "NVDA", "broker_order_id": 4242}]), \
         patch("execution._record_stale_entry") as stale:
        execution._note_stale_record("NVDA", 5, "OPEN", path="bracket", held_before=None,
                                     read_broker=True, defer_out=defer)
    stale.assert_not_called()
    assert defer["prior_order_ref"] == (4242, "ok") and defer["read_broker"] is True
    assert defer["path"] == "bracket" and defer["decision_basis"] == "approval_action"
    assert defer["filled_qty"] == 5 and defer["prior"] == {"stop_price": 98.0}


# ── _record_stale_entry: keys and wording ────────────────────────────────────

def _row(read_broker=False, evidence=None, **kw):
    with patch("db.log_activity_direct", return_value=True) as act, \
         patch("db.get_all_position_stops", return_value=[{"ticker": "NVDA", "broker_order_id": 4242}]), \
         patch("execution._stale_entry_evidence", return_value=evidence or {}) as ev:
        execution._record_stale_entry("NVDA", {"stop_price": 98.0, "opened_at": None}, 1, 1,
                                      read_broker=read_broker, **kw)
    stage, ticker, message, _, extra = act.call_args.args
    assert stage == "stale_record_at_entry"
    return message, extra, ev


def test_without_broker_reads_the_evidence_keys_are_absent():
    message, extra, ev = _row(held_before=0.0)
    ev.assert_not_called()
    assert extra["old_stop_state"] is None                # no stop-path read ran
    assert extra["entry_record_basis"] == UNKNOWN and extra["would_reset"] is False
    assert extra["held_before"] == 0.0
    assert "prior_order_status" not in extra and "prior_sell_fill_seen" not in extra
    assert "unbooked exit" not in message and "unknown" in message


def test_a_working_old_stop_is_a_lagged_read_and_claims_no_exit():
    message, extra, ev = _row(read_broker=True, held_before=0.0,
                              evidence={"prior_order_status": "Submitted"})
    ev.assert_called_once_with("NVDA", 4242)
    assert extra["entry_record_basis"] == LAGGED and extra["would_reset"] is False
    assert extra["prior_order_status"] == "Submitted"
    assert "unbooked exit" not in message and "no exit confirmed" in message


def test_a_working_stop_seen_by_the_stop_path_is_lagged_and_stamped():
    message, extra, _ = _row(read_broker=True, held_before=0.0, old_stop_state="working",
                             evidence={"prior_order_status": "not_open",
                                       "prior_sell_fill_seen": True})
    assert extra["old_stop_state"] == "working"
    assert extra["entry_record_basis"] == LAGGED and extra["would_reset"] is False


def test_a_confirmed_exit_would_reset():
    message, extra, _ = _row(read_broker=True, held_before=0.0,
                             evidence={"prior_order_status": "not_open",
                                       "prior_sell_fill_seen": True})
    assert extra["entry_record_basis"] == FRESH and extra["would_reset"] is True
    assert extra["prior_sell_fill_seen"] is True
    assert "looks closed" in message


def test_a_given_prior_order_ref_is_used_instead_of_a_read():
    with patch("db.log_activity_direct", return_value=True) as act, \
         patch("db.get_all_position_stops") as read, \
         patch("execution._stale_entry_evidence", return_value={}) as ev:
        execution._record_stale_entry("NVDA", {"stop_price": 98.0}, 1, 1, read_broker=True,
                                      prior_order_ref=(4242, "ok"))
    read.assert_not_called()
    ev.assert_called_once_with("NVDA", 4242)
    assert act.call_args.args[4]["prior_broker_order_id"] == 4242


def test_a_failing_evidence_read_still_writes_the_row():
    with patch("db.log_activity_direct", return_value=True) as act, \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("ibkr.get_open_orders", side_effect=RuntimeError("x")), \
         patch("ibkr._ib", None):
        execution._record_stale_entry("NVDA", {"stop_price": 98.0}, 1, 1, read_broker=True)
    extra = act.call_args.args[4]
    assert extra["entry_record_basis"] == UNKNOWN
    assert "prior_order_status" not in extra and extra["prior_sell_fill_seen"] is None


# ── execute_approved: the row after the stop work, the old order id ──────────

RECORD = {"stop_price": 146.24, "entry_price": 190.0, "opened_at": None, "stop_method": "atr"}
NEW_STOP = 137.15
FILL = 162.63


def _execute(monkeypatch, action, held_before, orders, execs=(), bracket=False, stops=None):
    import db
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", True)
    approval = {"id": 174, "ticker": "NVDA", "action": action, "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": NEW_STOP,
                "risk_json": {"shares_initial": 1, "price_at_sizing": FILL,
                              "stop_loss_price": NEW_STOP}}
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["get_position_stop"] = MagicMock(return_value=dict(RECORD))
    events = []
    db_fns["upsert_position_stop"] = MagicMock(
        side_effect=lambda *a, **k: (events.append("record_write"), True)[1])
    # The record names the old stop until the caller's write, then the new one.
    db_fns["get_all_position_stops"] = MagicMock(side_effect=lambda *a, **k: [
        {"ticker": "NVDA", "broker_order_id": 50 if "record_write" in events else 4242}])
    db_fns["log_activity_direct"] = MagicMock(
        side_effect=lambda stage, *a, **k: (events.append(stage), True)[1])
    db_fns["set_stop_broker_order_id"] = MagicMock(
        side_effect=lambda *a, **k: events.append("order_id_write"))
    db_fns["recorded_sell_order_ids"] = MagicMock(return_value=(set(), set()))
    monkeypatch.setattr(config, "USE_BRACKET_ORDERS", bracket)

    def place(*a, **k):
        events.append("place")
        return {"order_id": 50}
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.RequestTimeout = 0
    ib.reqExecutions.return_value = list(execs)
    with ExitStack() as st:
        st.enter_context(patch.multiple("db", **db_fns))
        st.enter_context(patch("ibkr.get_ib"))
        st.enter_context(patch("ibkr._ib", ib))
        client = st.enter_context(patch("ibkr.get_client"))
        st.enter_context(patch("ibkr.place_market_order", return_value={
            "filled_qty": 1, "fill_price": FILL, "order_id": 141, "status": "Filled"}))
        st.enter_context(patch("ibkr.get_latest_price", return_value=FILL))
        st.enter_context(patch("ibkr.get_fresh_stock_position", return_value=(1.0, 190.0)))
        st.enter_context(patch("ibkr.cached_stock_qty", return_value=1.0))
        st.enter_context(patch("ibkr.get_open_orders", return_value=orders))
        st.enter_context(patch("ibkr.place_bracket_order", return_value={
            "parent_order_id": 141, "stop_order_id": 77}))
        st.enter_context(patch("market_data.get_priced_quote", return_value={
            "price": FILL, "is_stale": False, "age_minutes": 1.0, "market_open": True}))
        st.enter_context(patch("market_data.check_gap_confirmation",
                               return_value={"status": "no_gap"}))
        st.enter_context(patch("market_data.is_market_hours", return_value=True))
        st.enter_context(patch("execution._held_before_order", return_value=held_before))
        st.enter_context(patch("tasks._resting_sell_stops",
                               **(stops if stops is not None else {"return_value": []})))
        st.enter_context(patch("tasks.manage_cancel_stop_orders_for"))
        st.enter_context(patch("tasks._queue_stop_coverage_recheck", return_value=True))
        st.enter_context(patch("tasks._queue_reprotect"))
        place_mock = st.enter_context(patch("tasks.manage_stop_place", side_effect=place))
        st.enter_context(patch("execution._queue_stop_for_retry"))
        st.enter_context(patch("time.sleep"))
        st.enter_context(patch("tasks.open_manual_action"))
        st.enter_context(patch("alerts._send"))
        st.enter_context(patch("alerts.send_trade_alert"))
        client.return_value.get_open_position.return_value = MagicMock(qty=1, avg_entry_price=190.0)
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=1)]
        execution.execute_approved(approval)
    rows = [c.args for c in db_fns["log_activity_direct"].call_args_list
            if c.args[0] == "stale_record_at_entry"]
    return rows, events, place_mock


def test_execute_lagging_add_rows_after_the_place_with_the_old_order(monkeypatch):
    rows, events, place = _execute(monkeypatch, "ADD", 0.0, [_order(4242, "Submitted")])
    assert place.call_args_list == [call("NVDA", 1, NEW_STOP, market_price=FILL)]   # unchanged
    assert len(rows) == 1
    extra = rows[0][4]
    assert extra["prior_broker_order_id"] == 4242          # read before the record write
    assert extra["prior_order_status"] == "Submitted"
    assert extra["entry_record_basis"] == LAGGED and extra["would_reset"] is False
    assert extra["held_before"] == 0.0 and extra["decision_basis"] == "post_fill_read"
    assert "unbooked exit" not in rows[0][2]
    assert events.index("stale_record_at_entry") > events.index("place")


def test_execute_open_over_a_closed_position_would_reset(monkeypatch):
    rows, events, _ = _execute(monkeypatch, "OPEN", 0.0, [], execs=[_Exec()])
    extra = rows[0][4]
    assert extra["prior_order_status"] == "not_open" and extra["prior_sell_fill_seen"] is True
    assert extra["entry_record_basis"] == FRESH and extra["would_reset"] is True
    assert "record_write" in events                         # the record is still written as before


def test_bracket_row_comes_after_the_record_and_its_order_id(monkeypatch):
    rows, events, place = _execute(monkeypatch, "OPEN", None, [], execs=[_Exec()], bracket=True)
    place.assert_not_called()                               # the bracket carried the stop
    assert len(rows) == 1
    extra = rows[0][4]
    assert extra["path"] == "bracket"
    assert extra["prior_broker_order_id"] == 4242           # read before the record write
    assert extra["prior_order_status"] == "not_open" and extra["prior_sell_fill_seen"] is True
    assert extra["entry_record_basis"] == UNKNOWN           # held_before not read on this path
    assert (events.index("record_write") < events.index("order_id_write")
            < events.index("stale_record_at_entry"))


def test_execute_passes_the_stop_paths_old_stop_state_to_the_row(monkeypatch):
    # Whatever _whole_position_stop leaves in status["old_stop_state"] reaches
    # the row (on this branch alone nothing sets it: the row carries None).
    real = execution._whole_position_stop

    def wps(*a, **k):
        out = real(*a, **k)
        k["status"]["old_stop_state"] = "working"
        return out
    monkeypatch.setattr(execution, "_whole_position_stop", wps)
    rows, _, _ = _execute(monkeypatch, "ADD", 0.0, [], execs=[_Exec()])
    extra = rows[0][4]
    assert extra["old_stop_state"] == "working" and extra["entry_record_basis"] == LAGGED


_GUARD = pytest.mark.skipif(not hasattr(execution, "_old_stop_state"),
                            reason="needs the add-stale-guard routing (merged trees)")


@_GUARD
def test_merged_another_working_stop_makes_the_row_agree_with_the_kept_route(monkeypatch):
    """The record's order id (4242) is no longer open and an untracked SELL
    printed today, but another stop of ours (999) still works: the guard keeps
    the record, so the row must not call it a confirmed exit."""
    other = dict(_order(999, "Submitted"), stop_price=140.0)
    rows, events, _ = _execute(monkeypatch, "ADD", 0.0, [], execs=[_Exec()],
                               stops={"return_value": [other]})
    assert "record_write" not in events                     # the kept-record route
    extra = rows[0][4]
    assert extra["prior_order_status"] == "not_open" and extra["prior_sell_fill_seen"] is True
    assert extra["entry_record_basis"] == LAGGED and extra["would_reset"] is False
    assert extra["old_stop_state"] == "working"


@_GUARD
def test_merged_an_unread_stop_check_is_never_a_confirmed_exit(monkeypatch):
    rows, events, _ = _execute(monkeypatch, "ADD", 0.0, [], execs=[_Exec()],
                               stops={"side_effect": ConnectionError("down")})
    assert "record_write" not in events
    extra = rows[0][4]
    assert extra["entry_record_basis"] == UNKNOWN and extra["would_reset"] is False
    assert extra["old_stop_state"] == "unread"
