"""An ADD whose post-fill reads show no more than the fill while a stop record
exists (execution._whole_position_stop's stale-record branch), with no
count above 0 held before the order (held_before 0.0 or None).

Before: it always took the fresh-entry route. The record was rewritten at
the new shares' stop with this fill as the entry, a stop went on for the new
shares only, the old stop was never cancelled and nothing alerted (11-10
NVDA: cache and fresh read both at the fill, two stops against one record).

Now the old stop's state decides (_old_stop_state, one bounded open-orders
read):
- a SELL stop for the ticker still working (a lagging read) or open orders
  unreadable: the kept-record route. The record and the old stop are kept,
  the new shares get their own stop and the caller's prior-stop alert goes
  out. No coverage re-check is queued: as after the held_before > 0 guard,
  reconcile checks the two stops.
- no stop working (a real exit not yet booked): the fresh-entry route as
  before; the record is rewritten and no old stop is left behind.
add_stale_record says which (record, old_stop_state). An OPEN with nothing
held before, and an ADD with a count above 0 held before, are unchanged."""
import inspect
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

import pytest

import config
import execution

RECORD = {"stop_price": 146.24, "entry_price": 190.0, "opened_at": None, "stop_method": "atr"}
NEW_STOP = 137.15
FILL = 162.63
OLD = {"ticker": "NVDA", "action": "SELL", "order_type": "STP", "qty": 1.0,
       "stop_price": 146.24, "status": "Submitted"}
STOPS = {"working": {"return_value": [dict(OLD)]},
         "cancelling": {"return_value": [dict(OLD, status="PendingCancel")]},
         "absent": {"return_value": []},
         "unread": {"side_effect": ConnectionError("IBKR not connected")}}


# ── _old_stop_state ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("state, want", [("working", "working"), ("cancelling", "working"),
                                         ("absent", "absent"), ("unread", "unread")])
def test_old_stop_state(state, want):
    # A stop still cancelling counts as working: it may still fill.
    with patch("tasks._resting_sell_stops", **STOPS[state]) as read:
        assert execution._old_stop_state("NVDA") == want
    import tasks
    read.assert_called_once_with("NVDA", timeout=tasks._REPROTECT_FRESH_TIMEOUT_S, reconnect=False)


# ── _whole_position_stop ─────────────────────────────────────────────────────

def _wps(action="ADD", held_before=0.0, cache=1, fresh=1.0, stops="working", status=None):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=cache, avg_entry_price=190.0)
    status = {} if status is None else status
    with ExitStack() as st:
        st.enter_context(patch("db.get_position_stop", return_value=dict(RECORD)))
        st.enter_context(patch("ibkr.get_client", return_value=client))
        st.enter_context(patch("ibkr.get_fresh_stock_position", return_value=(fresh, 190.0)))
        st.enter_context(patch("db.log_activity_direct"))
        st.enter_context(patch("db.get_all_position_stops", return_value=[]))
        queued = st.enter_context(patch("db.log_activity_queued"))
        read = st.enter_context(patch("tasks._resting_sell_stops", **STOPS[stops]))
        recheck = st.enter_context(patch("tasks._queue_stop_coverage_recheck", return_value=True))
        cancel = st.enter_context(patch("tasks.manage_cancel_stop_orders_for"))
        out = execution._whole_position_stop("NVDA", NEW_STOP, 1.0, status=status, action=action,
                                             held_before=held_before, market_price=FILL)
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "add_stale_record"]
    _wps.messages = [c.args[2] for c in queued.call_args_list if c.args[0] == "add_stale_record"]
    return out, status, rows, cancel, recheck, read


@pytest.mark.parametrize("held_before", [0.0, None], ids=["held_before_0", "held_before_none"])
def test_add_with_the_old_stop_working_keeps_the_record(held_before):
    out, status, rows, cancel, recheck, _ = _wps(held_before=held_before, stops="working")
    assert out == (NEW_STOP, None)                    # the new shares only, at their own stop
    assert status["prior_unreadable"] is True         # the caller keeps the record
    assert "exit_possible" not in status
    assert "still working" in status["unknown_reason"]
    assert "stale_record" not in status and "recheck_after_place" not in status
    assert not execution._wants_coverage_recheck(status)
    cancel.assert_not_called()                        # the old stop stays
    recheck.assert_not_called()
    assert [(r["record"], r["old_stop_state"]) for r in rows] == [("kept_unconfirmed", "working")]
    assert rows[0]["held_before"] == held_before
    assert status["old_stop_state"] == "working"


def test_add_with_open_orders_unreadable_keeps_the_record():
    out, status, rows, cancel, recheck, _ = _wps(stops="unread")
    assert out == (NEW_STOP, None) and status["prior_unreadable"] is True
    assert status["exit_possible"] is True
    assert "couldn't be read" in status["unknown_reason"]
    assert "stale_record" not in status and "recheck_after_place" not in status
    cancel.assert_not_called()
    recheck.assert_not_called()
    assert [(r["record"], r["old_stop_state"]) for r in rows] == [("kept_unconfirmed", "unread")]
    assert status["old_stop_state"] == "unread"


def test_add_with_no_old_stop_working_takes_the_fresh_entry_route():
    out, status, rows, cancel, recheck, _ = _wps(stops="absent")
    assert out == (NEW_STOP, None) and status.get("stale_record") is True
    assert "prior_unreadable" not in status
    cancel.assert_not_called()
    recheck.assert_not_called()
    assert [(r["record"], r["old_stop_state"]) for r in rows] == [("rewritten_at_new_stop", "absent")]
    assert status["old_stop_state"] == "absent"
    assert "no SELL stop is working" in _wps.messages[0]


def test_without_a_status_dict_nothing_is_queued():
    out, status, rows, cancel, recheck, _ = _wps(status=None)
    assert out == (NEW_STOP, None)
    recheck.assert_not_called()
    cancel.assert_not_called()


def test_open_with_nothing_held_before_keeps_the_fresh_entry_route():
    out, status, rows, cancel, recheck, read = _wps(action="OPEN", held_before=0.0)
    assert out == (NEW_STOP, None) and status.get("stale_record") is True
    assert "prior_unreadable" not in status
    assert rows == [] and "old_stop_state" not in status
    read.assert_not_called()                          # no open-orders read for an OPEN
    cancel.assert_not_called()


def test_add_with_a_count_held_before_keeps_the_existing_guard():
    out, status, rows, cancel, recheck, read = _wps(held_before=1.0)
    assert out == (NEW_STOP, None) and status["prior_unreadable"] is True
    assert "don't yet show" in status["unknown_reason"]
    assert "stale_record" not in status and "exit_possible" not in status
    assert "old_stop_state" not in status
    assert rows == []                                 # returns before the stale branch
    read.assert_not_called()
    cancel.assert_not_called()


# ── execute_approved ─────────────────────────────────────────────────────────

def _execute(monkeypatch, action="ADD", held_before=0.0, stops="working", new_stop=NEW_STOP):
    import db
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", True)
    approval = {"id": 174, "ticker": "NVDA", "action": action, "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": new_stop,
                "risk_json": {"shares_initial": 1, "price_at_sizing": FILL,
                              "stop_loss_price": new_stop}}
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["get_position_stop"] = MagicMock(return_value=dict(RECORD))
    db_fns["upsert_position_stop"] = MagicMock(return_value=True)
    events = []

    def place(*a, **k):
        events.append("place")
        return {"order_id": 50}
    with ExitStack() as st:
        st.enter_context(patch.multiple("db", **db_fns))
        st.enter_context(patch("ibkr.get_ib"))
        client = st.enter_context(patch("ibkr.get_client"))
        st.enter_context(patch("ibkr.place_market_order", return_value={
            "filled_qty": 1, "fill_price": FILL, "order_id": 141, "status": "Filled"}))
        st.enter_context(patch("ibkr.get_latest_price", return_value=FILL))
        st.enter_context(patch("ibkr.get_fresh_stock_position", return_value=(1.0, 190.0)))
        st.enter_context(patch("ibkr.cached_stock_qty", return_value=1.0))
        st.enter_context(patch("market_data.get_priced_quote", return_value={
            "price": FILL, "is_stale": False, "age_minutes": 1.0, "market_open": True}))
        st.enter_context(patch("market_data.check_gap_confirmation",
                               return_value={"status": "no_gap"}))
        st.enter_context(patch("market_data.is_market_hours", return_value=True))
        st.enter_context(patch("execution._held_before_order", return_value=held_before))
        st.enter_context(patch("tasks._resting_sell_stops", **STOPS[stops]))
        cancel = st.enter_context(patch("tasks.manage_cancel_stop_orders_for"))
        recheck = st.enter_context(patch("tasks._queue_stop_coverage_recheck", return_value=True))
        reprotect = st.enter_context(patch("tasks._queue_reprotect"))
        place_mock = st.enter_context(patch("tasks.manage_stop_place", side_effect=place))
        st.enter_context(patch("execution._queue_stop_for_retry"))
        st.enter_context(patch("time.sleep"))
        st.enter_context(patch("tasks.open_manual_action",
                               side_effect=lambda t, what, **k: events.append(("manual", what))))
        st.enter_context(patch("alerts._send",
                               side_effect=lambda t, *a, **k: events.append(("alert", t))))
        st.enter_context(patch("alerts.send_trade_alert"))
        lot_r = st.enter_context(patch("execution._record_add_lot_r"))
        client.return_value.get_open_position.return_value = MagicMock(qty=1, avg_entry_price=190.0)
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=1)]
        execution.execute_approved(approval)
    alerts = [e[1] for e in events if isinstance(e, tuple) and e[0] == "alert"]
    kept = [t for t in alerts if "The existing record was KEPT" in t]
    return place_mock, db_fns, cancel, recheck, reprotect, kept, lot_r


@pytest.mark.parametrize("held_before", [0.0, None], ids=["held_before_0", "held_before_none"])
def test_execute_lagging_add_keeps_the_record_and_alerts(monkeypatch, held_before):
    place, db_fns, cancel, recheck, reprotect, kept, lot_r = _execute(
        monkeypatch, held_before=held_before, stops="working")
    assert place.call_args_list == [call("NVDA", 1, NEW_STOP, market_price=FILL)]
    cancel.assert_not_called()                                # the old stop stays
    db_fns["upsert_position_stop"].assert_not_called()        # the record is kept
    db_fns["set_stop_broker_order_id"].assert_not_called()    # still names the old stop
    recheck.assert_not_called()                               # reconcile checks the two stops
    reprotect.assert_not_called()
    assert len(kept) == 1 and "still working (a lagging read), so the old stop" in kept[0]
    assert "The record is right for the old shares" in kept[0]
    assert "LOWER" not in kept[0]                             # the new stop is below the old
    lot_r.assert_not_called()                                 # no stale_record lot-R row


def test_execute_add_with_open_orders_unreadable_keeps_the_record(monkeypatch):
    place, db_fns, cancel, recheck, _, kept, _ = _execute(monkeypatch, stops="unread")
    assert place.call_args_list == [call("NVDA", 1, NEW_STOP, market_price=FILL)]
    db_fns["upsert_position_stop"].assert_not_called()
    recheck.assert_not_called()
    assert len(kept) == 1 and "If the old shares are still held" in kept[0]


def test_execute_add_after_a_real_exit_rewrites_the_record(monkeypatch):
    place, db_fns, cancel, recheck, _, kept, lot_r = _execute(monkeypatch, stops="absent")
    assert place.call_args_list == [call("NVDA", 1, NEW_STOP, market_price=FILL)]
    db_fns["upsert_position_stop"].assert_called_once()       # rewritten and reset
    up = db_fns["upsert_position_stop"].call_args.kwargs
    assert up["stop_price"] == NEW_STOP and up["entry_price"] == FILL
    db_fns["set_stop_broker_order_id"].assert_called_once_with("NVDA", 50)
    cancel.assert_not_called()
    recheck.assert_not_called()
    assert kept == []
    lot_r.assert_called_once()                                # the stale_record lot-R row


def test_execute_lagging_add_above_the_old_stop_warns_it_could_be_lowered(monkeypatch):
    above = round(RECORD["stop_price"] * 1.05, 2)
    place, db_fns, _, _, _, kept, _ = _execute(monkeypatch, stops="working", new_stop=above)
    assert place.call_args_list == [call("NVDA", 1, above, market_price=FILL)]
    db_fns["upsert_position_stop"].assert_not_called()
    assert len(kept) == 1 and "could LOWER the new shares' stop" in kept[0]


def test_execute_open_with_nothing_held_before_rewrites_the_record(monkeypatch):
    place, db_fns, _, recheck, _, kept, _ = _execute(monkeypatch, action="OPEN", held_before=0.0)
    assert place.call_args_list == [call("NVDA", 1, NEW_STOP, market_price=FILL)]
    db_fns["upsert_position_stop"].assert_called_once()
    assert db_fns["upsert_position_stop"].call_args.kwargs["stop_price"] == NEW_STOP
    recheck.assert_not_called()
    assert kept == []


def test_execute_add_with_a_count_held_before_is_unchanged(monkeypatch):
    place, db_fns, _, recheck, _, kept, _ = _execute(monkeypatch, held_before=1.0)
    assert place.call_args_list == [call("NVDA", 1, NEW_STOP, market_price=FILL)]
    db_fns["upsert_position_stop"].assert_not_called()
    recheck.assert_not_called()
    assert len(kept) == 1 and "don't yet show" in kept[0]
    assert "The record is right for the old shares" in kept[0]
