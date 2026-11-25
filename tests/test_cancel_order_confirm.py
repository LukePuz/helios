"""ibkr.cancel_order confirms a cancel from trade.log, not from the status
alone (the same rule as place_market_order's market-closed cancel).

ib_async's error handler sets orderStatus.status = Cancelled locally on any
order error, a rejected cancel (10148, 161, 10147) among them. cancel_order
used to return True on that at once, so a stop still working at the broker
read as gone. Now True only when the newest trade.log entry with the status
has error code 0 (an orderStatus message) or 202 (Order Canceled), polled
through _poll_after_cancel within the same 5 s bound; Filled and Inactive end
the poll at once and return False. The same per-stop rule holds in
_cancel_stops_parallel. cancel_stop_orders_for's stuck warning and the
orphan reconcile's line name the status, and a fill says FILLED.
"""
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ib_async.objects import TradeLogEntry

import config
import ibkr
import tasks


class _Trade:
    """A stop trade. set(status, code) changes the status and writes the
    trade.log entry ib_async writes for it: code 0 from an orderStatus
    message, the error code from its error handler."""

    def __init__(self, oid=7, symbol="NVDA", perm=None):
        self.order = SimpleNamespace(orderId=oid, permId=perm if perm is not None else 1000 + oid,
                                     action="SELL", orderType="STP", totalQuantity=10.0)
        self.contract = SimpleNamespace(symbol=symbol)
        self.orderStatus = SimpleNamespace(status="PreSubmitted")
        self.log = []

    def set(self, status, code=0):
        self.orderStatus.status = status
        self.log.append(TradeLogEntry(datetime.now(timezone.utc), status, "", code))


class _IB:
    """script[orderId] = {n: (status, code)}: on the n-th poll read after
    that order's cancel, the status changes (1-based)."""

    def __init__(self, trades, script):
        self.trades, self.script = trades, script
        self.reads = {t.order.orderId: None for t in trades}
        self.sleeps = 0

    def openTrades(self):
        return [t for t in self.trades
                if t.orderStatus.status not in ("Filled", "Cancelled", "ApiCancelled", "Inactive")]

    def cancelOrder(self, order):
        self.reads[order.orderId] = 0

    def sleep(self, _):
        self.sleeps += 1
        for t in self.trades:
            n = self.reads[t.order.orderId]
            if n is None:
                continue
            n += 1
            self.reads[t.order.orderId] = n
            step = self.script.get(t.order.orderId, {}).get(n)
            if step:
                t.set(*step)


def _cancel(trade, script, **kw):
    ib = _IB([trade], {trade.order.orderId: script})
    with patch("ibkr.get_ib", return_value=ib):
        ok = ibkr.cancel_order(trade.order.orderId, **kw)
    return ok, ib


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger="ibkr")
    getattr(ibkr, "_LAST_CANCEL_STATUS", {}).clear()


# ── cancel_order ───────────────────────────────────────────────────────────

def test_a_broker_cancelled_with_code_0_is_confirmed():
    trade = _Trade()
    ok, ib = _cancel(trade, {2: ("Cancelled", 0)})
    assert ok is True
    assert ib.sleeps == 2
    assert ibkr._LAST_CANCEL_STATUS[7] == "Cancelled"


def test_order_canceled_202_is_confirmed():
    ok, _ = _cancel(_Trade(), {1: ("Cancelled", 202)})
    assert ok is True


def test_api_cancelled_with_code_0_is_confirmed():
    ok, _ = _cancel(_Trade(), {1: ("ApiCancelled", 0)})
    assert ok is True


@pytest.mark.parametrize("code", [10148, 161, 10147])
def test_a_rejected_cancel_returns_false_after_the_bound(code, caplog):
    trade = _Trade()
    ok, ib = _cancel(trade, {1: ("Cancelled", code)})
    assert ok is False
    assert ib.sleeps == 5
    assert ibkr._LAST_CANCEL_STATUS[7] == ibkr.CANCEL_REJECTED_STATUS
    assert any("it was rejected (status CancelRejected" in r.getMessage() for r in caplog.records)


def test_a_rejection_then_the_broker_status_is_still_not_confirmed():
    """After a rejection the broker's next orderStatus brings the order's
    real state back (still working)."""
    ok, _ = _cancel(_Trade(), {1: ("Cancelled", 10148), 2: ("PreSubmitted", 0)})
    assert ok is False
    assert ibkr._LAST_CANCEL_STATUS[7] == "PreSubmitted"


def test_the_newest_cancelled_entry_decides():
    trade = _Trade()
    trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "Cancelled", "", 0))
    ok, _ = _cancel(trade, {1: ("Cancelled", 10148)})
    assert ok is False


def test_filled_returns_false_at_once(caplog):
    ok, ib = _cancel(_Trade(), {1: ("Filled", 0)})
    assert ok is False
    assert ib.sleeps == 1
    assert ibkr._LAST_CANCEL_STATUS[7] == "Filled"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("but the order FILLED during the cancel" in m for m in msgs)
    assert not any("needs manual cancel" in m for m in msgs)


def test_inactive_returns_false_at_once():
    ok, ib = _cancel(_Trade(), {2: ("Inactive", 0)})
    assert ok is False
    assert ib.sleeps == 2
    assert ibkr._LAST_CANCEL_STATUS[7] == "Inactive"


def test_pending_cancel_through_the_bound_returns_false(caplog):
    ok, ib = _cancel(_Trade(), {1: ("PendingCancel", 0)})
    assert ok is False
    assert ib.sleeps == 5
    assert any("Cancel sent for orderId=7 but status is still 'PendingCancel' after 5s" in
               r.getMessage() for r in caplog.records)


def test_the_bound_is_never_longer_than_poll_seconds():
    for script in ({}, {1: ("Cancelled", 10148)}, {1: ("PendingCancel", 0)}):
        _, ib = _cancel(_Trade(), script)
        assert ib.sleeps <= 5
    _, ib = _cancel(_Trade(), {}, _poll_seconds=3)
    assert ib.sleeps == 3


def test_the_symbol_refusal_is_unchanged():
    trade = _Trade(symbol="NVDA")
    ib = _IB([trade], {})
    ib.cancelOrder = MagicMock()
    with patch("ibkr.get_ib", return_value=ib):
        assert ibkr.cancel_order(7, symbol="AMD") is False
    ib.cancelOrder.assert_not_called()


def test_the_status_record_is_capped():
    for i in range(ibkr._LAST_CANCEL_STATUS_MAX + 10):
        ibkr._record_cancel_status(i, "PendingCancel")
    assert len(ibkr._LAST_CANCEL_STATUS) == ibkr._LAST_CANCEL_STATUS_MAX
    assert 0 not in ibkr._LAST_CANCEL_STATUS


# ── cancel_stop_orders_for, serial and parallel ───────────────────────────

def _stops(parallel, scripts):
    trades = [_Trade(oid) for oid in sorted(scripts)]
    ib = _IB(trades, scripts)
    with patch("ibkr.get_ib", return_value=ib), \
         patch.object(config, "CANCEL_PARALLEL_ENABLED", parallel):
        n = ibkr.cancel_stop_orders_for("NVDA")
    return n, ib


@pytest.mark.parametrize("parallel", [False, True])
def test_a_rejected_stop_is_not_counted_and_its_status_is_named(parallel, caplog):
    n, ib = _stops(parallel, {1: {1: ("Cancelled", 0)}, 2: {1: ("Cancelled", 10148)}})
    assert n == 1
    stuck = [r.getMessage() for r in caplog.records if "did NOT actually cancel" in r.getMessage()]
    assert len(stuck) == 1
    assert "orderIds=[2]; status 2=CancelRejected" in stuck[0]


def test_the_parallel_path_runs_with_the_flag_on():
    with patch("ibkr._cancel_stops_parallel", wraps=ibkr._cancel_stops_parallel) as spy:
        n, ib = _stops(True, {1: {1: ("Cancelled", 0)}, 2: {1: ("Cancelled", 161)}})
    spy.assert_called_once()
    assert n == 1
    assert ib.sleeps == 5        # one window for both: the rejected one to the bound


@pytest.mark.parametrize("parallel", [False, True])
def test_a_filled_stop_is_reported_as_filled(parallel, caplog):
    n, _ = _stops(parallel, {1: {1: ("Filled", 0)}, 2: {1: ("Cancelled", 202)}})
    assert n == 1
    msgs = [r.getMessage() for r in caplog.records]
    assert any("stop order(s) FILLED while being cancelled (orderIds=[1]; status Filled)"
               " — shares were sold" in m for m in msgs)
    assert not any("did NOT actually cancel" in m for m in msgs)


@pytest.mark.parametrize("parallel", [False, True])
def test_filled_and_stuck_stops_get_their_own_lines(parallel, caplog):
    n, _ = _stops(parallel, {1: {1: ("Filled", 0)}, 2: {1: ("PendingCancel", 0)}})
    assert n == 0
    msgs = [r.getMessage() for r in caplog.records]
    assert any("FILLED while being cancelled (orderIds=[1]" in m for m in msgs)
    assert any("did NOT actually cancel (orderIds=[2]; status 2=PendingCancel)" in m for m in msgs)


def test_the_parallel_inactive_stop_ends_its_poll():
    n, ib = _stops(True, {1: {1: ("Inactive", 0)}, 2: {2: ("Cancelled", 0)}})
    assert n == 1
    assert ib.sleeps == 2
    assert ibkr._LAST_CANCEL_STATUS[1] == "Inactive"


# ── reconcile's orphaned-stop cancel (tasks._cancel_orphan_stop) ──────────

_ORPHAN = {"order_id": 41, "perm_id": 5041, "ticker": "SNDK", "qty": 12.0, "stop_price": 90.0}


def _orphan_cycle(state, cancel_ok, completed=None, completed_raises=None, o=_ORPHAN):
    all_actions, new_actions = [], []
    get_completed = MagicMock(return_value=completed or [], side_effect=completed_raises)
    with patch("tasks.manage_cancel_order", return_value=cancel_ok) as cancel, \
         patch("ibkr.get_completed_orders", get_completed), \
         patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        tasks._cancel_orphan_stop(o, "orphan_pending_SNDK", all_actions, new_actions)
    cancel.assert_called_once_with(41, symbol="SNDK", perm_id=o["perm_id"])
    return all_actions, new_actions


def test_a_rejected_orphan_cancel_retries_each_cycle_and_alerts_once():
    state = {"orphan_pending_SNDK": "armed"}
    for cycle in range(3):
        all_actions, new_actions = _orphan_cycle(state, False)
        assert len(all_actions) == 1
        assert "Could NOT cancel orphaned stop" in all_actions[0]
        assert "(status Working)" in all_actions[0]
        assert new_actions == (all_actions if cycle == 0 else [])
    assert state["orphan_pending_SNDK"] == "armed"     # kept, so the next cycle retries
    assert "reconcile_alert_cancel_fail_41" in state


def test_a_filled_orphan_stop_says_check_for_a_short():
    state = {}
    all_actions, new_actions = _orphan_cycle(
        state, False, completed=[{"perm_id": 5041, "status": "Filled", "ticker": "SNDK"}])
    assert "FILLED while being cancelled" in all_actions[0]
    assert "check TWS for a short" in all_actions[0]
    assert "manual cancel" not in all_actions[0]
    assert new_actions == all_actions


def test_a_confirmed_orphan_cancel_clears_the_marker():
    state = {"orphan_pending_SNDK": "armed"}
    all_actions, new_actions = _orphan_cycle(state, True)
    assert all_actions[0].startswith("Cancelled ORPHANED stop")
    assert state["orphan_pending_SNDK"] == ""
    assert new_actions == all_actions


def test_an_unreadable_orphan_status_reads_unknown():
    all_actions, _ = _orphan_cycle({}, False, completed_raises=ConnectionError("down"))
    assert "(status unknown)" in all_actions[0]
    all_actions, _ = _orphan_cycle({}, False, o={**_ORPHAN, "perm_id": None})
    assert "(status unknown)" in all_actions[0]


def test_an_orphan_that_went_cancelled_late_is_said_so():
    all_actions, _ = _orphan_cycle(
        {}, False, completed=[{"perm_id": 5041, "status": "Cancelled", "ticker": "SNDK"}])
    assert "now reads Cancelled at the broker" in all_actions[0]


# ── delta: stale statuses, the shared read, part fills, edges ─────────────

class _IBLater(_IB):
    """openTrades returns every trade on its first call (the match read in
    cancel_stop_orders_for), then only `later` (a stop gone, or another
    client's, by the time its cancel looks it up). raise_on: orderIds whose
    cancelOrder raises."""

    def __init__(self, trades, script, later, raise_on=()):
        super().__init__(trades, script)
        self.later, self.n, self.raise_on = later, 0, set(raise_on)

    def openTrades(self):
        self.n += 1
        live = super().openTrades()
        return live if self.n == 1 else [t for t in live if t.order.orderId in self.later]

    def cancelOrder(self, order):
        if order.orderId in self.raise_on:
            raise ConnectionError("not connected")
        super().cancelOrder(order)


@pytest.mark.parametrize("parallel", [False, True])
def test_a_stale_filled_status_never_labels_a_stop_not_found_now(parallel, caplog):
    ibkr._record_cancel_status(2, "Filled")         # an older call, a reused orderId
    trades = [_Trade(1), _Trade(2)]
    ib = _IBLater(trades, {1: {1: ("Cancelled", 0)}}, later={1})
    with patch("ibkr.get_ib", return_value=ib), \
         patch.object(config, "CANCEL_PARALLEL_ENABLED", parallel):
        assert ibkr.cancel_stop_orders_for("NVDA") == 1
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("FILLED" in m for m in msgs)
    assert any("did NOT actually cancel (orderIds=[2]; status 2=NotFound)" in m for m in msgs)


def test_a_stale_filled_status_never_labels_a_stop_not_sent(caplog):
    ibkr._record_cancel_status(3, "Filled")
    trades = [_Trade(1), _Trade(2), _Trade(3)]
    ib = _IBLater(trades, {1: {1: ("Cancelled", 0)}}, later={1, 2, 3}, raise_on={2})
    with patch("ibkr.get_ib", return_value=ib), \
         patch.object(config, "CANCEL_PARALLEL_ENABLED", True), \
         pytest.raises(ConnectionError):
        ibkr.cancel_stop_orders_for("NVDA")
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("FILLED" in m for m in msgs)
    assert any("orderIds=[2, 3]; status 2=NotSent, 3=NotSent" in m for m in msgs)


def test_cancel_stop_orders_for_clears_its_stops_old_statuses():
    ibkr._record_cancel_status(1, "Filled")
    ibkr._record_cancel_status(99, "Filled")      # not this ticker's stop: kept
    with patch("ibkr.get_ib", return_value=_IB([], {})):
        ibkr.cancel_stop_orders_for("NVDA")
    assert ibkr._LAST_CANCEL_STATUS[1] == "Filled"   # no stop 1 matched: untouched
    trade = _Trade(1)
    with patch("ibkr.get_ib", return_value=_IB([trade], {1: {1: ("Cancelled", 0)}})):
        assert ibkr.cancel_stop_orders_for("NVDA") == 1
    assert ibkr._LAST_CANCEL_STATUS[1] == "Cancelled"
    assert ibkr._LAST_CANCEL_STATUS[99] == "Filled"


def test_a_rejection_then_an_identical_broker_cancelled_reads_rejected():
    """ib_async logs an orderStatus only when a field changed: a broker
    Cancelled identical to the local one adds no entry, so the rejection's
    entry stays newest. A false alarm in the safe direction."""
    trade = _Trade()

    def no_log_cancelled():
        trade.orderStatus.status = "Cancelled"      # unchanged fields: no log entry
    ib = _IB([trade], {7: {1: ("Cancelled", 10148)}})
    orig = ib.sleep

    def sleep(x):
        orig(x)
        if ib.reads[7] == 2:
            no_log_cancelled()
    ib.sleep = sleep
    with patch("ibkr.get_ib", return_value=ib):
        assert ibkr.cancel_order(7) is False
    assert ibkr._LAST_CANCEL_STATUS[7] == ibkr.CANCEL_REJECTED_STATUS


@pytest.mark.parametrize("script, ok", [({1: ("Cancelled", 0)}, True),
                                        ({1: ("Cancelled", 10148)}, False)])
def test_poll_seconds_below_one_still_reads_once_with_the_log_check(script, ok):
    for secs in (0, -3):
        trade = _Trade()
        ok_now, ib = _cancel(trade, script, _poll_seconds=secs)
        assert ok_now is ok
        assert ib.sleeps == 1


def test_get_completed_orders_passes_its_timeout():
    ib = MagicMock()
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._bounded_req", return_value=[]) as req:
        ibkr.get_completed_orders(timeout=5.0)
    assert req.call_args.kwargs["timeout"] == 5.0
    assert "drop_session" not in req.call_args.kwargs     # a timeout still drops it


def _orphans_run(orphans, cancel_ok, completed=None, completed_raises=None):
    """One reconcile run's section 1 over several orphans, sharing a cache."""
    state, cache = {}, {}
    all_actions, new_actions = [], []
    get_completed = MagicMock(return_value=completed or [], side_effect=completed_raises)
    with patch("tasks.manage_cancel_order", return_value=cancel_ok), \
         patch("ibkr.get_completed_orders", get_completed), \
         patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        for o in orphans:
            tasks._cancel_orphan_stop(o, f"orphan_pending_{o['ticker']}", all_actions,
                                      new_actions, completed=cache)
    return all_actions, get_completed


def test_one_completed_orders_read_per_run_with_a_short_timeout():
    two = [_ORPHAN, {**_ORPHAN, "order_id": 42, "perm_id": 5042, "ticker": "VRT"}]
    all_actions, get_completed = _orphans_run(two, False)
    get_completed.assert_called_once_with(timeout=tasks._ORPHAN_STATUS_READ_TIMEOUT_S)
    assert tasks._ORPHAN_STATUS_READ_TIMEOUT_S <= 5
    assert len(all_actions) == 2


def test_a_timed_out_read_reads_unknown_for_every_orphan_in_the_run():
    two = [_ORPHAN, {**_ORPHAN, "order_id": 42, "perm_id": 5042, "ticker": "VRT"}]
    all_actions, get_completed = _orphans_run(two, False,
                                              completed_raises=TimeoutError("5s"))
    get_completed.assert_called_once()
    assert all("(status unknown)" in a for a in all_actions)


def test_a_part_filled_then_cancelled_orphan_says_check_for_a_short():
    row = [{"perm_id": 5041, "status": "Cancelled", "filled_qty": 4.0, "ticker": "SNDK"}]
    for cancel_ok in (False, True):
        state = {"orphan_pending_SNDK": "armed"}
        all_actions, _ = _orphan_cycle(state, cancel_ok, completed=row)
        assert "part-FILLED (4 filled, status Cancelled)" in all_actions[0]
        assert "check TWS for a short" in all_actions[0]
        assert "manual cancel" not in all_actions[0]
        assert state["orphan_pending_SNDK"] == ("" if cancel_ok else "armed")


def test_a_confirmed_orphan_cancel_with_nothing_filled_reads_as_before():
    row = [{"perm_id": 5041, "status": "Cancelled", "filled_qty": 0.0, "ticker": "SNDK"}]
    all_actions, _ = _orphan_cycle({}, True, completed=row)
    assert all_actions[0].startswith("Cancelled ORPHANED stop: SELL 12.0 SNDK @ $90.00")
