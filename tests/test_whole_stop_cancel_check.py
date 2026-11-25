"""An ADD's whole-position stop when the old stop's cancel isn't confirmed
(execution._whole_position_stop). A cancel that raised used to return
(new_stop, None) without prior_unreadable, so the caller wrote the new
shares' level over the record; a cancel that left a stop resting got the
whole-position stop placed on top of it with no follow-up.

Now any cancel that raised (OrdersCallUnknown included: the worker may still
be cancelling) or reported a stop that didn't cancel (stuck_out, from
ibkr.cancel_stop_orders_for's own list) is treated the same way. The
whole-position stop still goes on for every share at the whole-position
level, exactly as after a clean cancel, and the record matches it. After the
place the caller queues the stop coverage re-check, which trims the
over-stop against the record, and one CHECK STOPS alert goes out. The clean
path is unchanged and reads nothing more.

stuck_out carries each stop's cancel status (cancel-order-confirm): an old
stop that FILLED during the cancel takes the kept-record route (only the
new shares get a stop, and the alert names the fill), and an Inactive stop,
which rests nowhere, sets no re-check. A NotFound stop (gone from openTrades
before its cancel: it may have filled) stays unconfirmed."""
import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from ib_async.objects import TradeLogEntry

import config
import execution
import ibkr
import tasks
import test_overstop as _ov   # its fake broker (a module import: no re-collection)

RECORD = {"stop_price": 146.24, "entry_price": 190.0}
NEW_STOP = 137.15
LEVEL = 146.24          # the whole-position level: max(new, record), floored at the resting stop
OLD = {"ticker": "NVDA", "action": "SELL", "order_type": "STP", "qty": 1.0,
       "stop_price": 146.24, "status": "Submitted"}


def _cancel(effect=None, stuck=()):
    """manage_cancel_stop_orders_for: raises effect, or reports stuck ids."""
    def cancel(ticker, timeout=30, stuck_out=None):
        if effect is not None:
            raise effect
        if stuck_out is not None:
            stuck_out[:] = list(stuck)
        return 1 - len(stuck) if stuck else 1
    return cancel


NOT_CLEAN = [
    pytest.param({"effect": RuntimeError("socket")}, "the old stop's cancel failed (RuntimeError)",
                 id="raised"),
    pytest.param({"effect": tasks.OrdersCallUnknown("no result")},
                 "the old stop's cancel failed (OrdersCallUnknown)", id="orders_call_unknown"),
    pytest.param({"effect": tasks.OrdersCallExpired("late")},
                 "the old stop's cancel failed (OrdersCallExpired)", id="orders_call_expired"),
    pytest.param({"stuck": [322]}, "1 old stop(s) did not confirm their cancel", id="stuck"),
    pytest.param({"stuck": [(322, "PreSubmitted")]}, "1 old stop(s) did not confirm their cancel",
                 id="stuck_with_status"),
    pytest.param({"stuck": [[322, "PendingCancel"]]}, "1 old stop(s) did not confirm their cancel",
                 id="stuck_from_worker"),                 # JSON: the pair arrives as a list
    pytest.param({"stuck": [(322, "NotSent"), (8, "Inactive")]},
                 "1 old stop(s) did not confirm their cancel", id="stuck_beside_inactive"),
    pytest.param({"stuck": [(8, "NotFound")]}, "1 old stop(s) did not confirm their cancel",
                 id="not_found"),                         # may have filled: kept unconfirmed
    pytest.param({"stuck": [(8, "Inactive"), [9, "NotFound"]]},
                 "1 old stop(s) did not confirm their cancel", id="not_found_beside_inactive"),
]


def _wps(cancel, status=None, lookup_error=None, action="ADD", held=1.0):
    client = MagicMock()
    if lookup_error is not None:
        client.get_open_position.side_effect = lookup_error
    else:
        client.get_open_position.return_value = MagicMock(qty=2, avg_entry_price=190.0)
    with patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(2.0, 190.0)), \
         patch("tasks._resting_sell_stops", return_value=[dict(OLD)]), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued"), patch("alerts._send") as send, \
         patch("tasks._queue_stop_coverage_recheck", return_value=True) as recheck, \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel) as cancel_mock:
        out = execution._whole_position_stop("NVDA", NEW_STOP, 1.0, status=status,
                                             market_price=162.63, action=action,
                                             held_before=held, price_is_fill=True)
    sent = [c.args[0] for c in send.call_args_list if "CHECK STOPS" in c.args[0]]
    return out, recheck, cancel_mock, sent


def _held_checks(status):
    return [t for t in status.get("alerts_out", []) if "CHECK STOPS" in t]


# ── _whole_position_stop ─────────────────────────────────────────────────────

def test_clean_cancel_is_unchanged():
    status: dict = {"alerts_out": []}
    out, recheck, cancel, sent = _wps(_cancel(), status=status)
    assert out == (LEVEL, (190.0, 2.0))
    cancel.assert_called_once_with("NVDA", stuck_out=[])
    assert "cancel_unconfirmed" not in status and "prior_unreadable" not in status
    assert _held_checks(status) == [] and sent == []
    recheck.assert_not_called()


@pytest.mark.parametrize("kw, reason", NOT_CLEAN)
def test_unconfirmed_cancel_still_places_the_whole_stop(kw, reason):
    status: dict = {"alerts_out": []}
    out, recheck, _, sent = _wps(_cancel(**kw), status=status)
    assert out == (LEVEL, (190.0, 2.0))                  # same as a clean cancel
    assert status["cancel_unconfirmed"] == reason
    assert "prior_unreadable" not in status              # the record follows the whole stop
    assert status["prior_stop"] == 146.24
    (text,) = _held_checks(status)                       # held: sent after the place
    assert text.startswith("⚠️ CHECK STOPS — <b>NVDA</b>")
    assert reason in text and "over-stopped until the stop coverage re-check (about 1-6 min)" in text
    assert "for all 2 shares @ $146.24 is being placed" in text
    assert sent == []
    recheck.assert_not_called()                          # the caller queues it after its place


@pytest.mark.parametrize("kw, reason", NOT_CLEAN)
def test_without_a_status_dict_it_queues_and_sends_inline(kw, reason):
    out, recheck, _, sent = _wps(_cancel(**kw), status=None)
    assert out == (LEVEL, (190.0, 2.0))
    recheck.assert_called_once_with("NVDA")
    assert len(sent) == 1 and reason in sent[0]


def test_status_without_a_held_list_sends_inline_and_flags_the_caller():
    status: dict = {}
    _, recheck, _, sent = _wps(_cancel(stuck=[322]), status=status)
    assert status["cancel_unconfirmed"] and len(sent) == 1
    recheck.assert_not_called()


def test_failed_position_lookup_keeps_the_record():
    status: dict = {}
    out, recheck, cancel, _ = _wps(_cancel(), status=status, lookup_error=ConnectionError("down"))
    assert out == (NEW_STOP, None)
    assert status["prior_unreadable"] is True
    assert status["unknown_reason"] == "the broker position lookup failed"
    cancel.assert_not_called()
    recheck.assert_not_called()


def test_open_with_no_position_still_rewrites_a_leftover_record():
    status: dict = {}
    out, _, cancel, _ = _wps(_cancel(), status=status, action="OPEN", held=0.0,
                             lookup_error=Exception("No open position in NVDA"))
    assert out == (NEW_STOP, None)
    assert "prior_unreadable" not in status              # the caller writes the record
    cancel.assert_not_called()


# ── the stuck list: ibkr → orders worker → manage_cancel_stop_orders_for ─────

class _Trade:
    def __init__(self, order_id, status):
        self.contract = MagicMock(symbol="NVDA")
        self.order = MagicMock(orderId=order_id, permId=0, action="SELL", orderType="STP",
                               totalQuantity=1)
        self.orderStatus = MagicMock(status=status)
        self.log = []

    def cancelled(self):
        # As ib_async records a broker cancel: status Cancelled plus a
        # trade.log entry with code 202 (Order Canceled), which
        # cancel_order requires before it counts the cancel as confirmed.
        self.orderStatus.status = "Cancelled"
        self.log.append(TradeLogEntry(datetime.now(timezone.utc), "Cancelled", "", 202))


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_inactive_and_working_stuck_stops_are_reported_with_their_statuses(parallel):
    """openTrades holds an Inactive stop (it stays Inactive: nothing to
    cancel) and a working stop that refuses to cancel (another clientId):
    neither is confirmed, and stuck_out names each with its status."""
    inactive, working = _Trade(8, "Inactive"), _Trade(322, "PreSubmitted")
    ib = MagicMock()
    ib.openTrades.return_value = [inactive, working]
    stuck: list = []
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=parallel):
        confirmed = ibkr.cancel_stop_orders_for("NVDA", stuck_out=stuck)
    assert confirmed == 0
    assert stuck == [(8, "Inactive"), (322, "PreSubmitted")]
    assert inactive.orderStatus.status == "Inactive"


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_stop_that_fills_during_the_cancel_is_reported_filled(parallel):
    sold, other = _Trade(8, "Submitted"), _Trade(9, "Submitted")

    def cancel_order(order):
        if order.orderId == 8:
            sold.orderStatus.status = "Filled"
        else:
            other.cancelled()
    ib = MagicMock()
    ib.openTrades.return_value = [sold, other]
    ib.cancelOrder.side_effect = cancel_order
    stuck: list = []
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=parallel):
        assert ibkr.cancel_stop_orders_for("NVDA", stuck_out=stuck) == 1
    assert stuck == [(8, "Filled")]


def test_cancel_count_without_stuck_out_is_unchanged():
    t = _Trade(8, "PreSubmitted")
    ib = MagicMock()
    ib.openTrades.return_value = [t]
    ib.cancelOrder.side_effect = lambda o: t.cancelled()
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=False):
        assert ibkr.cancel_stop_orders_for("NVDA") == 1


def test_worker_task_detail_returns_the_stuck_ids():
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [322]
        return 1
    with patch("tasks._mark_started", return_value=True), \
         patch("ibkr.cancel_stop_orders_for", side_effect=cancel) as c:
        assert tasks._broker_cancel_stop_orders_for.run("NVDA", deadline=None, detail=True) == \
            {"confirmed": 1, "stuck": [322]}
    with patch("tasks._mark_started", return_value=True), \
         patch("ibkr.cancel_stop_orders_for", return_value=2) as c:
        assert tasks._broker_cancel_stop_orders_for.run("NVDA", deadline=None) == 2
    c.assert_called_once_with("NVDA")                    # old messages: the count, as before


def test_manage_cancel_passes_the_stuck_ids_back(monkeypatch):
    monkeypatch.setenv("IB_CLIENT_ID", "1")              # not the orders worker
    fake = MagicMock()
    fake.get.return_value = {"confirmed": 1, "stuck": [322]}
    stuck: list = []
    with patch("stop_lock.assert_held"), \
         patch.object(tasks._broker_cancel_stop_orders_for, "apply_async", return_value=fake) as d:
        assert tasks.manage_cancel_stop_orders_for("NVDA", stuck_out=stuck) == 1
    assert stuck == [322]
    kw = dict(d.call_args.kwargs["kwargs"]); kw.pop("deadline")
    assert kw == {"detail": True}
    fake.get.return_value = 3
    with patch("stop_lock.assert_held"), \
         patch.object(tasks._broker_cancel_stop_orders_for, "apply_async", return_value=fake) as d:
        assert tasks.manage_cancel_stop_orders_for("NVDA") == 3
    kw = dict(d.call_args.kwargs["kwargs"]); kw.pop("deadline")
    assert kw == {}                                      # every other caller: unchanged message


def test_manage_cancel_in_the_orders_worker_fills_stuck_out(monkeypatch):
    stuck: list = []

    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [322]
        return 0
    with patch("stop_lock.assert_held"), patch("tasks._is_orders_worker", return_value=True), \
         patch("ibkr.cancel_stop_orders_for", side_effect=cancel):
        assert tasks.manage_cancel_stop_orders_for("NVDA", stuck_out=stuck) == 0
    assert stuck == [322]


# ── execute_approved: one whole stop, the record at its level, re-check after ─

def _execute(monkeypatch, cancel, place_effect=None, events=None, record=None):
    import db
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", True)
    approval = {"id": 174, "ticker": "NVDA", "action": "ADD", "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": NEW_STOP,
                "risk_json": {"shares_initial": 1, "price_at_sizing": 162.63,
                              "stop_loss_price": NEW_STOP}}
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["get_position_stop"] = (MagicMock(side_effect=record) if record is not None
                                   else MagicMock(return_value=dict(RECORD)))
    db_fns["upsert_position_stop"] = MagicMock(return_value=True)
    events = [] if events is None else events

    def place(*a, **k):
        events.append("place")
        if place_effect is not None:
            raise place_effect
        return {"order_id": 50}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 1, "fill_price": 162.63,
                                                        "order_id": 141, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=162.63), \
         patch("ibkr.get_fresh_stock_position", return_value=(2.0, 190.0)), \
         patch("ibkr.cached_stock_qty", return_value=2.0), \
         patch("market_data.get_priced_quote", return_value={"price": 162.63, "is_stale": False,
                                                             "age_minutes": 1.0, "market_open": True}), \
         patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks._resting_sell_stops", return_value=[dict(OLD)]), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel), \
         patch("tasks._queue_stop_coverage_recheck",
               side_effect=lambda t: (events.append("recheck"), True)[1]) as recheck, \
         patch("tasks._queue_reprotect") as reprotect, \
         patch("tasks.manage_stop_place", side_effect=place) as place, \
         patch("execution._queue_stop_for_retry") as retry_queue, \
         patch("time.sleep"), \
         patch("tasks.open_manual_action",
               side_effect=lambda t, what, **k: events.append(("manual", what))) as manual, \
         patch("alerts._send",
               side_effect=lambda t, *a, **k: events.append(("alert", t))) as send, \
         patch("alerts.send_trade_alert"):
        client.return_value.get_open_position.return_value = MagicMock(qty=1, avg_entry_price=190.0)
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=1)]
        execution.execute_approved(approval)
    _execute.retry_queue = retry_queue
    _execute.manual = manual
    checks = [e[1] for e in events if isinstance(e, tuple) and "CHECK STOPS" in e[1]]
    return place, db_fns, recheck, reprotect, checks, events


def test_execute_clean_cancel_places_one_whole_stop(monkeypatch):
    place, db_fns, recheck, _, checks, _ = _execute(monkeypatch, _cancel())
    assert place.call_args_list == [call("NVDA", 2.0, LEVEL, market_price=162.63)]
    db_fns["upsert_position_stop"].assert_called_once()
    assert db_fns["upsert_position_stop"].call_args.kwargs["stop_price"] == LEVEL
    db_fns["set_stop_broker_order_id"].assert_called_once_with("NVDA", 50)
    recheck.assert_not_called()
    assert checks == []


@pytest.mark.parametrize("kw, reason", NOT_CLEAN)
def test_execute_unconfirmed_cancel_places_the_whole_stop_then_rechecks(monkeypatch, kw, reason):
    place, db_fns, recheck, reprotect, checks, events = _execute(monkeypatch, _cancel(**kw))
    # Exactly one stop, for every share at the whole-position level: never
    # one for only the new share, so a late cancel leaves nothing bare.
    assert place.call_args_list == [call("NVDA", 2.0, LEVEL, market_price=162.63)]
    db_fns["upsert_position_stop"].assert_called_once()
    assert db_fns["upsert_position_stop"].call_args.kwargs["stop_price"] == LEVEL
    db_fns["set_stop_broker_order_id"].assert_called_once_with("NVDA", 50)
    recheck.assert_called_once_with("NVDA")
    reprotect.assert_not_called()
    assert len(checks) == 1 and reason in checks[0]
    assert "old stop was left in place" not in checks[0]
    # After the place: the re-check, then the held alert.
    order = [e if isinstance(e, str) else "alert" for e in events
             if e in ("place", "recheck") or (isinstance(e, tuple) and "CHECK STOPS" in e[1])]
    assert order == ["place", "recheck", "alert"]


@pytest.mark.parametrize("place_effect, outcome", [
    (RuntimeError("socket"), "failed"),
    (tasks.OrdersCallExpired("busy"), "queued"),
], ids=["place_failed", "place_queued"])
def test_execute_unconfirmed_cancel_rechecks_even_when_the_place_fails(monkeypatch, place_effect,
                                                                       outcome):
    place, db_fns, recheck, reprotect, checks, events = _execute(
        monkeypatch, _cancel(stuck=[322]), place_effect=place_effect)
    recheck.assert_called_once_with("NVDA")
    assert len(checks) == 1
    if outcome == "failed":
        assert place.call_count == 3                     # the usual three tries
        reprotect.assert_called_once()                   # the usual unprotected path
    else:
        assert place.call_count == 1
        _execute.retry_queue.assert_called_once()
        reprotect.assert_not_called()
    assert events.index("recheck") > max(i for i, e in enumerate(events) if e == "place")


# ── the re-check after an unconfirmed ADD cancel ─────────────────────────────

def _after_add(cache, fresh):
    """An ADD of 1 on 2 held: our whole-position stop for 3 plus the old
    lot's stop for 2 from another client, which refused the cancel."""
    b = _ov._Broker(held=cache)
    b.fresh = fresh
    b.add(1, 2, cross=True, perm=7001)    # the old stop: reverts every cancel
    b.add(2, 3)                           # our whole-position stop
    return b


def test_recheck_trims_the_over_stop_around_a_stuck_old_stop():
    b = _after_add(cache=3, fresh=3)
    b.run(lambda: tasks.recheck_stop_coverage("AMD"))
    b.run_rechecks()
    assert b.cover() == 3                                # exact cover once the chain settles
    assert b.orders[1]["status"] == "Submitted" and b.orders[1]["qty"] == 2
    assert b.cancels[1] == 1                             # the refused stop: cancelled once


def test_recheck_with_disagreeing_reads_looks_again_then_trims():
    """The cache still shows the pre-fill 2 a minute after the ADD; the fresh
    read shows 3. No cancel then, a ⚠️ and another look; once the reads
    agree the over-stop is trimmed."""
    b = _after_add(cache=2, fresh=3)
    b.run(lambda: tasks.recheck_stop_coverage("AMD"))
    assert b.cancels == {} and b.cover() == 5
    assert [a for _, a in b.queued] == [2]
    (line,) = [m for m in b.sent if "disagree" in m]
    assert line.startswith("STOP RECONCILE — ⚠️ AMD:") and "re-checking in 60 s" in line
    assert not any("🚨" in m for m in b.sent)
    b.held, b.fresh = 3, None                            # the cache catches up
    b.run_rechecks()
    assert b.cover() == 3


def test_disagreeing_reads_on_the_last_try_keep_the_alarm():
    b = _after_add(cache=2, fresh=3)
    b.run(lambda: tasks.recheck_stop_coverage("AMD", attempt=tasks._COVERAGE_RECHECK_TRIES))
    assert b.queued == [] and b.cancels == {}
    assert any(m.startswith("STOP RECONCILE — 🚨 AMD:") and "disagree" in m for m in b.sent)


def test_fix_overstop_text_is_unchanged_for_reconcile():
    """outcome_out only reports; reconcile's calls get the same 🚨 line."""
    b = _after_add(cache=2, fresh=3)
    out: dict = {}
    with_out = b.run(lambda: tasks._fix_overstop("AMD", allow_cancel=False, max_qty=2.0,
                                                 outcome_out=out))
    without = b.run(lambda: tasks._fix_overstop("AMD", allow_cancel=False, max_qty=2.0))
    assert with_out == without and with_out.startswith("🚨 AMD:")
    assert with_out == f"🚨 {out['disagree_over_cover']} — check the stops in TWS"


# ── an old stop that FILLED during the cancel; one that is not resting ───────

def test_util_stuck_pairs_reads_tuples_lists_and_bare_ids():
    assert execution._stuck_pairs([(8, "Filled"), [9, "Inactive"], 322, (7, None)]) == [
        (8, "Filled"), (9, "Inactive"), (322, None), (7, None)]
    assert execution._stuck_pairs(None) == [] and execution._stuck_pairs([]) == []


@pytest.mark.parametrize("kw", [
    pytest.param({"stuck": [(8, "Filled")]}, id="filled"),
    pytest.param({"stuck": [[8, "Filled"], (9, "PreSubmitted")]}, id="filled_and_stuck"),
], )
def test_a_filled_old_stop_stops_only_the_new_shares_and_keeps_the_record(kw):
    status: dict = {"alerts_out": []}
    out, recheck, _, sent = _wps(_cancel(**kw), status=status)
    assert out == (NEW_STOP, None)                        # the new lot only, at its own stop
    assert status["prior_unreadable"] is True and status["old_stop_filled"] is True
    assert "FILLED during the cancel" in status["unknown_reason"]
    assert "cancel_unconfirmed" not in status and "prior_stop" not in status
    assert _held_checks(status) == [] and sent == []       # the caller's alert names the fill
    # Another old stop may have cancelled (bare shares) or still rest sized
    # for sold ones: the caller queues the coverage re-check after its place.
    assert status["recheck_after_place"] is True
    recheck.assert_not_called()


def test_a_filled_old_stop_without_a_status_dict_alerts_inline():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=2, avg_entry_price=190.0)
    with patch("db.get_position_stop", return_value=dict(RECORD)), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(2.0, 190.0)), \
         patch("tasks._resting_sell_stops", return_value=[dict(OLD)]), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued"), patch("alerts._send") as send, \
         patch("tasks._queue_stop_coverage_recheck") as recheck, \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=_cancel(stuck=[(8, "Filled")])):
        out = execution._whole_position_stop("NVDA", NEW_STOP, 1.0, status=None,
                                             market_price=162.63, action="ADD", held_before=1.0)
    assert out == (NEW_STOP, None)
    texts = [c.args[0] for c in send.call_args_list]
    assert any(t.startswith("🚨 CHECK POSITION — <b>NVDA</b>") and "FILLED" in t for t in texts)
    recheck.assert_called_once_with("NVDA")              # no caller to queue it


@pytest.mark.parametrize("stuck", [
    pytest.param([(8, "Inactive")], id="inactive"),
    pytest.param([(8, "Inactive"), [9, "Inactive"]], id="two_inactive"),
])
def test_a_stop_that_is_not_resting_sets_no_recheck(stuck, caplog):
    status: dict = {"alerts_out": []}
    caplog.set_level("INFO")
    out, recheck, _, sent = _wps(_cancel(stuck=stuck), status=status)
    assert out == (LEVEL, (190.0, 2.0))                   # the whole stop, as on a clean cancel
    assert "cancel_unconfirmed" not in status and "prior_unreadable" not in status
    assert _held_checks(status) == [] and sent == []
    recheck.assert_not_called()
    assert "(not resting); no coverage re-check" in caplog.text


def test_a_raised_cancel_beside_an_inactive_stop_is_still_unconfirmed():
    status: dict = {"alerts_out": []}

    def cancel(ticker, timeout=30, stuck_out=None):
        stuck_out[:] = [(8, "Inactive")]
        raise tasks.OrdersCallUnknown("no result")
    out, _, _, _ = _wps(cancel, status=status)
    assert out == (LEVEL, (190.0, 2.0))
    assert status["cancel_unconfirmed"] == "the old stop's cancel failed (OrdersCallUnknown)"


_REAL_MANAGE = tasks.manage_cancel_stop_orders_for


def _real_cancel(trades, on_cancel=None):
    """manage_cancel_stop_orders_for in the orders worker over the real
    ibkr.cancel_stop_orders_for and cancel_order, on fake trades."""
    def cancel(ticker, timeout=30, stuck_out=None):
        ib = MagicMock()
        ib.openTrades.return_value = trades
        if on_cancel is not None:
            ib.cancelOrder.side_effect = on_cancel
        with patch("ibkr.get_ib", return_value=ib), \
             patch("ibkr._cancel_parallel_enabled", return_value=False), \
             patch("tasks._is_orders_worker", return_value=True), patch("stop_lock.assert_held"):
            return _REAL_MANAGE(ticker, timeout=timeout, stuck_out=stuck_out)
    return cancel


def test_combined_a_stop_filled_during_the_cancel_through_the_real_cancel(caplog):
    """ADD 1 onto 1: the old stop fills while it is being cancelled. The real
    cancel path reports it Filled, so only the new share is stopped."""
    old = _Trade(8, "Submitted")
    status: dict = {"alerts_out": []}
    out, recheck, _, sent = _wps(
        _real_cancel([old], lambda o: setattr(old.orderStatus, "status", "Filled")), status=status)
    assert out == (NEW_STOP, None)
    assert status["old_stop_filled"] is True and "orderIds [8]" in status["unknown_reason"]
    assert "cancel_unconfirmed" not in status and status["recheck_after_place"] is True
    assert "FILLED while being cancelled" in caplog.text          # ibkr's own error line
    recheck.assert_not_called()


def test_combined_an_inactive_stop_through_the_real_cancel_is_not_a_check_stops():
    inactive, working = _Trade(8, "Inactive"), _Trade(9, "Submitted")
    status: dict = {"alerts_out": []}
    out, recheck, _, sent = _wps(_real_cancel([inactive, working],
                                              lambda o: working.cancelled() if o.orderId == 9
                                              else None), status=status)
    assert inactive.orderStatus.status == "Inactive"
    assert out == (LEVEL, (190.0, 2.0))
    assert "cancel_unconfirmed" not in status and _held_checks(status) == []
    recheck.assert_not_called()


def test_execute_filled_old_stop_places_the_new_share_only_and_names_the_fill(monkeypatch):
    old = _Trade(8, "Submitted")
    place, db_fns, recheck, reprotect, checks, events = _execute(
        monkeypatch, _real_cancel([old], lambda o: setattr(old.orderStatus, "status", "Filled")))
    assert len(place.call_args_list) == 1
    args = place.call_args.args
    assert args[0] == "NVDA" and args[1] == 1 and args[2] < LEVEL   # the new lot, its own level
    db_fns["upsert_position_stop"].assert_not_called()               # the record is kept
    recheck.assert_called_once_with("NVDA")
    reprotect.assert_not_called()
    assert checks == []                                              # no "over-stopped" text
    alerts_ = [e[1] for e in events if isinstance(e, tuple) and e[0] == "alert"]
    (fill,) = [t for t in alerts_ if t.startswith("🚨 CHECK POSITION — NVDA")]
    fill_at = events.index(("alert", fill))
    assert events.index("place") < events.index("recheck") < fill_at
    assert "FILLED during the cancel" in fill and "may already be sold" in fill
    assert "old stop was left in place" not in fill
    manual = [e[1] for e in events if isinstance(e, tuple) and e[0] == "manual"]
    assert manual and "old stop filled during the cancel" in manual[0]


def test_execute_inactive_old_stop_places_the_whole_stop_without_a_recheck(monkeypatch):
    inactive, working = _Trade(8, "Inactive"), _Trade(9, "Submitted")
    place, db_fns, recheck, _, checks, _ = _execute(
        monkeypatch, _real_cancel([inactive, working],
                                  lambda o: working.cancelled() if o.orderId == 9 else None))
    assert place.call_args_list == [call("NVDA", 2.0, LEVEL, market_price=162.63)]
    assert db_fns["upsert_position_stop"].call_args.kwargs["stop_price"] == LEVEL
    recheck.assert_not_called()
    assert checks == []


# ── the re-check right after the place loop; the UNPROTECTED text ────────────

def test_the_recheck_is_queued_before_the_unprotected_alert(monkeypatch):
    place, _, recheck, _, _, events = _execute(monkeypatch, _cancel(stuck=[322]),
                                               place_effect=RuntimeError("socket"))
    recheck.assert_called_once_with("NVDA")
    unprotected = next(i for i, e in enumerate(events) if isinstance(e, tuple)
                       and e[0] == "alert" and "UNPROTECTED POSITION" in e[1])
    assert events.index("recheck") < unprotected
    text = events[unprotected][1]
    assert ("The old stop's cancel was not confirmed (1 old stop(s) did not confirm their "
            "cancel), so it may still rest on the older shares") in text
    manual = [e[1] for e in events if isinstance(e, tuple) and e[0] == "manual"]
    assert any("the old stop may still rest, check its size first" in m for m in manual)


def test_the_unprotected_text_is_unchanged_on_a_clean_cancel(monkeypatch):
    _, _, recheck, _, _, events = _execute(monkeypatch, _cancel(), place_effect=RuntimeError("x"))
    recheck.assert_not_called()
    text = next(e[1] for e in events if isinstance(e, tuple) and e[0] == "alert"
                and "UNPROTECTED POSITION" in e[1])
    assert "old stop's cancel" not in text
    manual = [e[1] for e in events if isinstance(e, tuple) and e[0] == "manual"]
    assert manual and all(m.startswith("place a stop @ $") and "may still rest" not in m
                          for m in manual)


def test_a_raise_before_the_recheck_still_queues_it_once_from_the_finally(monkeypatch):
    events: list = []
    with pytest.raises(KeyboardInterrupt):
        _execute(monkeypatch, _cancel(stuck=[322]), place_effect=KeyboardInterrupt(),
                 events=events)
    assert events.count("recheck") == 1


def test_the_recheck_has_one_guarded_call_in_the_finally():
    src = inspect.getsource(execution.execute_approved)
    assert src.count("tasks._queue_stop_coverage_recheck(ticker)") == 2
    assert '_wants_coverage_recheck(prior_info) and not prior_info.get("recheck_queued")' in src


def test_not_resting_is_inactive_only():
    assert execution._STUCK_NOT_RESTING == ("Inactive",)


def test_execute_a_stop_gone_before_its_cancel_is_unconfirmed_and_rechecked(monkeypatch, caplog):
    """Through the real cancel: the stop is listed, then gone from openTrades
    when cancel_order looks for it (NotFound; it may have filled)."""
    old = _Trade(8, "Submitted")
    reads = iter([[old], []])

    def cancel(ticker, timeout=30, stuck_out=None):
        ib = MagicMock()
        ib.openTrades.side_effect = lambda: next(reads)
        with patch("ibkr.get_ib", return_value=ib), \
             patch("ibkr._cancel_parallel_enabled", return_value=False), \
             patch("tasks._is_orders_worker", return_value=True), patch("stop_lock.assert_held"):
            return _REAL_MANAGE(ticker, timeout=timeout, stuck_out=stuck_out)
    place, db_fns, recheck, _, checks, _ = _execute(monkeypatch, cancel)
    assert "orderId=8 not found among open orders" in caplog.text
    assert "status 8=NotFound" in caplog.text                     # ibkr's stuck warning
    assert place.call_args_list == [call("NVDA", 2.0, LEVEL, market_price=162.63)]
    recheck.assert_called_once_with("NVDA")
    assert len(checks) == 1 and "1 old stop(s) did not confirm their cancel" in checks[0]



# ── review 2: the Filled route re-checks; a Filled survives a cancel raise ───

def test_wants_coverage_recheck():
    assert execution._wants_coverage_recheck({"cancel_unconfirmed": "x"}) is True
    assert execution._wants_coverage_recheck({"recheck_after_place": True}) is True
    assert execution._wants_coverage_recheck({}) is False


def test_execute_filled_with_a_raise_in_the_place_loop_queues_the_recheck_once(monkeypatch):
    old = _Trade(8, "Submitted")
    events: list = []
    with pytest.raises(KeyboardInterrupt):
        _execute(monkeypatch, _real_cancel([old], lambda o: setattr(old.orderStatus, "status",
                                                                    "Filled")),
                 place_effect=KeyboardInterrupt(), events=events)
    assert events.count("recheck") == 1


def test_execute_filled_and_failed_place_queues_the_recheck_once(monkeypatch):
    old = _Trade(8, "Submitted")
    place, _, recheck, reprotect, _, events = _execute(
        monkeypatch, _real_cancel([old], lambda o: setattr(old.orderStatus, "status", "Filled")),
        place_effect=RuntimeError("socket"))
    assert place.call_count == 3
    recheck.assert_called_once_with("NVDA")
    reprotect.assert_called_once()
    text = next(e[1] for e in events if isinstance(e, tuple) and e[0] == "alert"
                and "UNPROTECTED POSITION" in e[1])
    assert "old stop's cancel was not confirmed" not in text        # it filled, it isn't resting


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_ibkr_a_raise_after_a_fill_still_reports_the_fill(parallel):
    """Stop 8 fills during its cancel; the cancel for stop 9 raises. No
    further cancel is sent, and stuck_out names the fill before the raise."""
    sold, boom, later = _Trade(8, "Submitted"), _Trade(9, "Submitted"), _Trade(10, "Submitted")
    sent = []

    def cancel_order(order):
        sent.append(order.orderId)
        if order.orderId == 8:
            sold.orderStatus.status = "Filled"
        elif order.orderId == 9:
            raise RuntimeError("socket")
    ib = MagicMock()
    ib.openTrades.return_value = [sold, boom, later]
    ib.cancelOrder.side_effect = cancel_order
    stuck: list = []
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=parallel), \
         pytest.raises(RuntimeError, match="socket"):
        ibkr.cancel_stop_orders_for("NVDA", stuck_out=stuck)
    assert sent == [8, 9]                                   # nothing sent after the raise
    assert stuck == [(8, "Filled"), (9, "NotSent"), (10, "NotSent")]


def test_combined_a_fill_then_a_raise_through_the_real_sequential_cancel():
    sold, boom = _Trade(8, "Submitted"), _Trade(9, "Submitted")

    def on_cancel(order):
        if order.orderId == 8:
            sold.orderStatus.status = "Filled"
        else:
            raise RuntimeError("socket")
    status: dict = {"alerts_out": []}
    out, recheck, _, sent = _wps(_real_cancel([sold, boom], on_cancel), status=status)
    assert out == (NEW_STOP, None)
    assert status["old_stop_filled"] is True and status["recheck_after_place"] is True
    assert "cancel_unconfirmed" not in status and sent == []


def test_worker_task_detail_returns_the_error_with_what_it_had():
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [(8, "Filled")]
        raise RuntimeError("socket")
    with patch("tasks._mark_started", return_value=True), \
         patch("ibkr.cancel_stop_orders_for", side_effect=cancel):
        out = tasks._broker_cancel_stop_orders_for.run("NVDA", deadline=None, detail=True)
    assert out == {"confirmed": None, "stuck": [(8, "Filled")], "error": "RuntimeError: socket"}
    with patch("tasks._mark_started", return_value=True), \
         patch("ibkr.cancel_stop_orders_for", side_effect=RuntimeError("x")), \
         pytest.raises(RuntimeError):
        tasks._broker_cancel_stop_orders_for.run("NVDA", deadline=None)   # no detail: raises


def _via_worker(result):
    """manage_cancel_stop_orders_for's non-worker branch with the orders
    worker's JSON-shaped result."""
    def cancel(ticker, timeout=30, stuck_out=None):
        fake = MagicMock()
        fake.get.return_value = result
        with patch("stop_lock.assert_held"), patch("tasks._is_orders_worker", return_value=False), \
             patch.object(tasks._broker_cancel_stop_orders_for, "apply_async", return_value=fake):
            return _REAL_MANAGE(ticker, timeout=timeout, stuck_out=stuck_out)
    return cancel


def test_manage_cancel_fills_stuck_out_then_raises_the_worker_error():
    stuck: list = []
    with pytest.raises(tasks.OrdersCallFailed) as ei:
        _via_worker({"confirmed": None, "stuck": [[8, "Filled"]],
                     "error": "RuntimeError: socket"})("NVDA", stuck_out=stuck)
    assert stuck == [[8, "Filled"]]
    assert ei.value.error_type == "RuntimeError" and str(ei.value) == "RuntimeError: socket"


@pytest.mark.parametrize("result", [
    pytest.param({"confirmed": None, "stuck": [[8, "Filled"]], "error": "RuntimeError: socket"},
                 id="with_error"),
    pytest.param({"confirmed": 0, "stuck": [[8, "Filled"]]}, id="plain"),
])
def test_combined_a_fill_through_the_worker_result(result):
    status: dict = {"alerts_out": []}
    out, recheck, _, _ = _wps(_via_worker(result), status=status)
    assert out == (NEW_STOP, None)
    assert status["old_stop_filled"] is True and status["recheck_after_place"] is True
    assert "cancel_unconfirmed" not in status


def test_a_worker_error_without_a_fill_is_unconfirmed_named_by_its_type():
    status: dict = {"alerts_out": []}
    out, _, _, _ = _wps(_via_worker({"confirmed": None, "stuck": [[9, "NotSent"]],
                                     "error": "ConnectionError: reset"}), status=status)
    assert out == (LEVEL, (190.0, 2.0))
    assert status["cancel_unconfirmed"] == "the old stop's cancel failed (ConnectionError)"


def test_the_filled_route_drops_the_whole_stop_texts_it_held():
    """Texts held by _add_stop_qty / the floor / the guard describe the
    whole-position stop; it is never placed once an old stop filled. Texts
    held before this call stay."""
    def add_stop_qty(ticker, total, *a, alerts_out=None, **k):
        alerts_out.append("⚠️ ADD STOP MAY NOT COVER — NVDA")
        return total
    for stuck, kept in (([(8, "Filled")], ["earlier"]),
                        ([], ["earlier", "⚠️ ADD STOP MAY NOT COVER — NVDA"])):
        status: dict = {"alerts_out": ["earlier"]}
        with patch("execution._add_stop_qty", side_effect=add_stop_qty):
            _wps(_cancel(stuck=stuck), status=status)
        assert status["alerts_out"] == kept


def test_a_missing_record_after_a_fill_is_written_with_a_warning(monkeypatch, caplog):
    """NIT: the post-place read finds no record on the Filled route."""
    old = _Trade(8, "Submitted")
    seen = {"cancelled": False}
    real = _real_cancel([old], lambda o: setattr(old.orderStatus, "status", "Filled"))

    def cancel(*a, **k):
        seen["cancelled"] = True
        return real(*a, **k)
    place, db_fns, recheck, _, _, _ = _execute(
        monkeypatch, cancel, record=lambda *a, **k: None if seen["cancelled"] else dict(RECORD))
    assert "no stop record after the ADD (1 old stop(s) FILLED during the cancel" in caplog.text
    db_fns["upsert_position_stop"].assert_called_once()                # the new shares' record
    assert db_fns["upsert_position_stop"].call_args.kwargs["stop_price"] < LEVEL
    recheck.assert_called_once_with("NVDA")


# ── review 3: a raise after the cancel went out keeps Filled / SentUnknown ───

def _poll_raises(trades, on_cancel=None, raise_on_sleep=1):
    """A fake ib whose ib.sleep raises on call raise_on_sleep (the poll)."""
    ib = MagicMock()
    ib.openTrades.return_value = trades
    if on_cancel is not None:
        ib.cancelOrder.side_effect = on_cancel
    calls = {"n": 0}

    def sleep(_s):
        calls["n"] += 1
        if calls["n"] >= raise_on_sleep:
            raise ConnectionError("disconnect")
    ib.sleep.side_effect = sleep
    return ib


@pytest.mark.parametrize("fills, want", [(True, "Filled"), (False, "SentUnknown")],
                         ids=["reads_filled", "unknown"])
def test_ibkr_sequential_a_poll_raise_after_the_send_is_not_notsent(fills, want):
    first, second = _Trade(8, "Submitted"), _Trade(9, "Submitted")

    def on_cancel(order):
        if fills and order.orderId == 8:
            first.orderStatus.status = "Filled"
    ib = _poll_raises([first, second], on_cancel)
    stuck: list = []
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=False), \
         pytest.raises(ConnectionError, match="disconnect"):
        ibkr.cancel_stop_orders_for("NVDA", stuck_out=stuck)
    assert [c.args[0].orderId for c in ib.cancelOrder.call_args_list] == [8]   # sent, then raised
    assert stuck == [(8, want), (9, "NotSent")]


def test_ibkr_cancel_order_keeps_the_status_when_its_poll_raises():
    t = _Trade(8, "Submitted")
    ib = _poll_raises([t])
    with patch("ibkr.get_ib", return_value=ib), pytest.raises(ConnectionError):
        ibkr.cancel_order(8, symbol="NVDA")
    assert ibkr._LAST_CANCEL_STATUS[8] == "SentUnknown"


@pytest.mark.parametrize("raise_on_sleep, want", [
    (1, [(8, "Filled"), (9, "SentUnknown"), (10, "SentUnknown")]),
    (2, [(8, "Filled"), (10, "SentUnknown")]),           # 9 confirmed on the first poll
], ids=["first_poll", "second_poll"])
def test_ibkr_parallel_a_poll_raise_fills_stuck_out_then_reraises(raise_on_sleep, want):
    a, b, c = _Trade(8, "Submitted"), _Trade(9, "Submitted"), _Trade(10, "Submitted")

    def on_cancel(order):
        if order.orderId == 8:
            a.orderStatus.status = "Filled"
        elif order.orderId == 9:
            b.cancelled()
    ib = _poll_raises([a, b, c], on_cancel, raise_on_sleep=raise_on_sleep)
    stuck: list = []
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=True), \
         pytest.raises(ConnectionError, match="disconnect"):
        ibkr.cancel_stop_orders_for("NVDA", stuck_out=stuck)
    assert ib.cancelOrder.call_count == 3                 # all sent before the poll
    assert stuck == want


def test_status_after_raise():
    assert ibkr._status_after_raise(_Trade(8, "Filled")) == "Filled"
    assert ibkr._status_after_raise(_Trade(8, "PendingCancel")) == "SentUnknown"
    assert ibkr._status_after_raise(object()) == "SentUnknown"          # unreadable


def _real_cancel_ib(ib, parallel):
    def cancel(ticker, timeout=30, stuck_out=None):
        with patch("ibkr.get_ib", return_value=ib), \
             patch("ibkr._cancel_parallel_enabled", return_value=parallel), \
             patch("tasks._is_orders_worker", return_value=True), patch("stop_lock.assert_held"):
            return _REAL_MANAGE(ticker, timeout=timeout, stuck_out=stuck_out)
    return cancel


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_combined_a_fill_seen_before_a_poll_raise_takes_the_filled_route(parallel):
    a, b = _Trade(8, "Submitted"), _Trade(9, "Submitted")

    def on_cancel(order):
        if order.orderId == 8:
            a.orderStatus.status = "Filled"
    status: dict = {"alerts_out": []}
    out, _, _, _ = _wps(_real_cancel_ib(_poll_raises([a, b], on_cancel), parallel), status=status)
    assert out == (NEW_STOP, None)
    assert status["old_stop_filled"] is True and status["recheck_after_place"] is True


def test_combined_a_sent_unknown_is_unconfirmed():
    a = _Trade(8, "Submitted")
    status: dict = {"alerts_out": []}
    out, _, _, _ = _wps(_real_cancel_ib(_poll_raises([a]), False), status=status)
    assert out == (LEVEL, (190.0, 2.0))
    assert status["cancel_unconfirmed"] == "the old stop's cancel failed (ConnectionError)"
