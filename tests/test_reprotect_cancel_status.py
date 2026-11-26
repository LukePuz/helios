"""The re-protect paths' "cancel ours" reads each stop's cancel status
(tasks._reprotect_cancel) and sizes the stop from the smaller of the
portfolio cache and a fresh reqPositions() read taken BEFORE the cancel
(tasks._reprotect_fresh_held; read again only after a stop of ours filled). The stop coverage re-check is queued once per
ticker at a time (tasks._queue_stop_coverage_recheck), and a request that
dedupe skips keeps its full count of looks (the restart marker).

- An unusable fresh read (failed, empty, flat) while stops rest: nothing is
  cancelled, so those stay, and the call returns False (the retry). Nothing
  resting: the cache, as before.
- A stop of ours that FILLED (or part-filled) during the cancel: a second
  fresh read sizes the stop, one CHECK POSITION alert says what was placed,
  and the coverage re-check is queued. That read unusable: nothing placed,
  False.
- A retry whose cache still shows the pre-fill count checks cover and sizes
  from the fresh read, other clients' stops netted on both paths.
- A stop that did not cancel and may still rest (any status but Inactive,
  NotFound and SentUnknown included): the stop goes on as before, then the
  re-check trims a double cover. A raised cancel keeps each path's old
  handling (no place).
The cancel goes through the real manage_cancel_stop_orders_for (orders
worker branch, and once the detail=True worker reply), ibkr's
cancel_stop_orders_for and cancel_order, on fake trades. No broker, DB or
Redis (conftest's _FakeRedis, on its test clock)."""
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from ib_async.objects import TradeLogEntry

from conftest import _FakeRedis
import ibkr
import tasks

HELD = 20.0
RECORD = 150.0
_REAL_FRESH_HELD = tasks._reprotect_fresh_held     # conftest fakes it per test
_REAL_RECHECK_REDIS = tasks._recheck_redis


@pytest.fixture(autouse=True)
def _real_fresh_read(monkeypatch):
    monkeypatch.setattr(tasks, "_reprotect_fresh_held", _REAL_FRESH_HELD)


class _Trade:
    def __init__(self, order_id, status="Submitted", qty=20.0, symbol="CRWD"):
        self.contract = MagicMock(symbol=symbol)
        self.order = MagicMock(orderId=order_id, permId=0, action="SELL", orderType="STP",
                               totalQuantity=qty)
        self.orderStatus = MagicMock(status=status)
        self.log = []

    def cancelled(self, part_filled=None):
        # As ib_async records a broker cancel: Cancelled plus a 202 log entry.
        self.orderStatus.status = "Cancelled"
        if part_filled is not None:
            self.orderStatus.filled = float(part_filled)
        self.log.append(TradeLogEntry(datetime.now(timezone.utc), "Cancelled", "", 202))


def _ib(trades, on_cancel=None, sleep_raises=False):
    ib = MagicMock()
    ib.openTrades.return_value = trades
    if on_cancel is not None:
        ib.cancelOrder.side_effect = on_cancel
    if sleep_raises:
        ib.sleep.side_effect = ConnectionError("disconnect")
    return ib


def _fills(trade):
    return lambda order: setattr(trade.orderStatus, "status", "Filled")


def _cancels(*trades, part=None):
    by_id = {t.order.orderId: t for t in trades}
    return lambda order: by_id[order.orderId].cancelled(part)


LAST: dict = {}


def _fresh_value(v):
    return v if isinstance(v, (dict, Exception)) else {"CRWD": v}


def _run(ib, from_queue=True, before=5.0, after=(), held=HELD, fresh=HELD, place_effect=None,
         intent="RESTORE", worker=True, record_effect=None, exact_reads=None, cover=None,
         outcome=None):
    """_reprotect_after_late_sell on CRWD with the real cancel path on ib.
    before: the coverage read before the cancel (part cover, so the cancel
    runs); after: the stops a re-read finds after it; held: the cache's
    count; fresh: the fresh read's count (a dict for the whole read, an
    exception to raise; a list: one per read, the first before the cancel,
    the second after a fill). worker False: the cancel goes through the orders
    worker's detail=True reply, built by the real worker task over ib and
    JSON-encoded. cover: the cover reads as row lists, one per read (with
    their statuses; replaces before). outcome: the caller's outcome_out."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=held)]
    if exact_reads is not None:
        fresh_kw = {"side_effect": [_fresh_value(v) for v in exact_reads]}
    elif isinstance(fresh, list):
        # the queued path's early sign-check read sees the same first value
        reads = ([fresh[0]] if from_queue else []) + list(fresh)
        fresh_kw = {"side_effect": [_fresh_value(v) for v in reads]}
    elif isinstance(fresh, Exception):
        fresh_kw = {"side_effect": fresh}
    else:
        fresh_kw = {"return_value": _fresh_value(fresh)}
    with ExitStack() as st:
        for p in (patch("ibkr.get_client", return_value=client),
                  patch("ibkr.get_open_orders", return_value=[]),
                  patch("ibkr.get_ib", return_value=ib),

                  patch("ibkr._cancel_parallel_enabled", return_value=False),
                  patch("db.get_position_stop", return_value={"stop_price": RECORD}),
                  patch("db.set_stop_broker_order_id", side_effect=record_effect),
                  patch("db.mark_profit_target_taken"),

                  patch("risk_manager._floor_at_resting_stop",
                        side_effect=lambda t, level, *a, **k: (level, None)),
                  patch("risk_manager._stop_below_market",
                        side_effect=lambda t, level, *a, **k: level)):
            st.enter_context(p)
        if worker:
            st.enter_context(patch("tasks._is_orders_worker", return_value=True))
        else:
            def reply(*a, **k):
                out = tasks._broker_cancel_stop_orders_for.run("CRWD", deadline=None, detail=True)
                fake = MagicMock()
                fake.get.return_value = json.loads(json.dumps(out))   # as the result backend
                return fake
            st.enter_context(patch("tasks._is_orders_worker", return_value=False))
            st.enter_context(patch.object(tasks._broker_cancel_stop_orders_for, "apply_async",
                                          side_effect=reply))
        if cover is not None:
            reads = iter(cover)

            def cover_read(ticker, rows_out=None, **k):
                rows = next(reads)
                if rows_out is not None:
                    rows_out[:] = rows
                return sum(float(o["qty"]) for o in rows)
            LAST["cover"] = st.enter_context(patch("execution._stops_still_resting",
                                                   side_effect=cover_read))
        else:
            LAST["cover"] = st.enter_context(patch("execution._stops_still_resting",
                                                   return_value=before))
        LAST["sleep"] = st.enter_context(patch("time.sleep"))
        LAST["fresh"] = st.enter_context(patch("ibkr.get_fresh_stock_positions", **fresh_kw))
        LAST["ib"] = ib
        LAST["left"] = st.enter_context(patch("tasks._resting_sell_stops",
                                              return_value=list(after)))
        place = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 3},
                                       side_effect=place_effect))
        send = st.enter_context(patch("alerts._send"))
        queue = st.enter_context(patch.object(tasks.recheck_stop_coverage, "apply_async"))
        order = {"intent": intent, **({"stop_price": RECORD} if intent == "PROFIT_TARGET" else {})}
        out = tasks._reprotect_after_late_sell("CRWD", order, 140.0, from_queue=from_queue,
                                               **({"outcome_out": outcome}
                                                  if outcome is not None else {}))
    checks = [c.args[0] for c in send.call_args_list if "CHECK POSITION" in c.args[0]]
    return out, place, queue, checks


# ── _reprotect_cancel ────────────────────────────────────────────────────────

def test_reprotect_cancel_reads_the_statuses_through_the_real_path():
    sold, stuck, dead = _Trade(8), _Trade(9), _Trade(10, "Inactive")

    def on_cancel(order):
        if order.orderId == 8:
            sold.orderStatus.status = "Filled"
    with patch("ibkr.get_ib", return_value=_ib([sold, stuck, dead], on_cancel)), \
         patch("ibkr._cancel_parallel_enabled", return_value=False), \
         patch("tasks._is_orders_worker", return_value=True), patch("stop_lock.assert_held"):
        filled, resting, exc = tasks._reprotect_cancel("CRWD")
    assert filled == [8] and resting == [9] and exc is None       # 10 Inactive: not resting


@pytest.mark.parametrize("stuck, want_resting", [
    ([(9, "NotFound")], [9]), ([(9, "SentUnknown")], [9]), ([(9, "Inactive")], []),
    ([9], [9]), ([[9, "PendingCancel"]], [9]),
])
def test_reprotect_cancel_keeps_not_found_and_sent_unknown_unconfirmed(stuck, want_resting):
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = stuck
        return 0
    with patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel):
        assert tasks._reprotect_cancel("CRWD") == ([], want_resting, None)


def test_reprotect_cancel_never_raises():
    boom = RuntimeError("socket")
    with patch("tasks.manage_cancel_stop_orders_for", side_effect=boom):
        assert tasks._reprotect_cancel("CRWD") == ([], [], boom)


# ── _reprotect_fresh_held ────────────────────────────────────────────────────

@pytest.mark.parametrize("read, want", [
    pytest.param({"CRWD": 12.0, "NVDA": 3.0}, 12.0, id="held"),
    pytest.param({"NVDA": 3.0}, 0.0, id="others_but_not_this_one"),
    pytest.param({"CRWD": -4.0}, -4.0, id="short"),
    pytest.param({}, None, id="empty_read"),
    pytest.param(ConnectionError("down"), None, id="read_fails"),
])
def test_fresh_held(read, want):
    kw = {"side_effect": read} if isinstance(read, Exception) else {"return_value": read}
    with patch("ibkr.get_fresh_stock_positions", **kw):
        assert tasks._reprotect_fresh_held("CRWD") == want


# ── a stop of ours filled during the cancel ──────────────────────────────────

@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_fill_during_the_cancel_stops_the_fresh_count_and_says_so(from_queue):
    ours = _Trade(8)
    out, place, queue, checks = _run(_ib([ours], _fills(ours)), from_queue=from_queue,
                                     fresh=[20.0, 12.0])
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)          # not the 20 before the fill
    # before the cancel (after the cover read), after the fill; queued: also the early sign check
    assert LAST["fresh"].call_count == (3 if from_queue else 2)
    queue.assert_called_once_with(args=["CRWD"], kwargs={"attempt": 1}, countdown=60)
    (text,) = checks
    assert text.startswith("🚨 CHECK POSITION — <b>CRWD</b>")
    assert "FILLED during the cancel (orderIds [8])" in text and "20 shares" in text
    assert "A stop went on 12 shares, sized from a fresh position read" in text


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
@pytest.mark.parametrize("fresh", [
    pytest.param(ConnectionError("down"), id="unread"),
    pytest.param({}, id="empty"),
    pytest.param({"NVDA": 3.0}, id="flat"),
])
def test_a_fill_with_no_usable_fresh_read_places_nothing_and_rechecks(from_queue, fresh):
    ours = _Trade(8)
    out, place, queue, checks = _run(_ib([ours], _fills(ours)), from_queue=from_queue,
                                     fresh=[20.0, fresh])
    assert out is False                                   # the retry runs
    place.assert_not_called()
    queue.assert_called_once()
    (text,) = checks
    assert "No stop was placed now" in text


def test_queued_a_fill_with_other_stops_covering_the_fresh_count_places_nothing():
    ours = _Trade(8)
    other = {"order_id": 9, "ticker": "CRWD", "qty": 12.0, "stop_price": 140.0}
    out, place, queue, checks = _run(_ib([ours], _fills(ours)), fresh=[20.0, 12.0], after=[other])
    assert out is True
    place.assert_not_called()
    queue.assert_called_once()
    assert "No new stop was needed" in checks[0]


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_fill_then_a_raise_places_nothing_and_rechecks(from_queue):
    ours, other = _Trade(8), _Trade(9)

    def on_cancel(order):
        if order.orderId == 8:
            ours.orderStatus.status = "Filled"
        else:
            raise RuntimeError("socket")
    out, place, queue, checks = _run(_ib([ours, other], on_cancel), from_queue=from_queue)
    assert out is False and len(checks) == 1 and "No stop was placed now" in checks[0]
    place.assert_not_called()
    queue.assert_called_once()


def test_queued_a_poll_raise_after_a_fill_places_nothing():
    ours = _Trade(8)
    out, place, queue, checks = _run(_ib([ours], _fills(ours), sleep_raises=True))
    assert out is False and len(checks) == 1
    place.assert_not_called()


def test_a_fill_reaches_the_re_protect_through_the_worker_s_detail_reply():
    ours = _Trade(8)
    out, place, queue, checks = _run(_ib([ours], _fills(ours)), fresh=[20.0, 12.0], worker=False)
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)
    assert len(checks) == 1 and "orderIds [8]" in checks[0]


# ── SF-1: a retry whose cache still shows the pre-fill count ─────────────────

def test_queued_retry_after_a_fill_finds_the_good_stop_covers_the_fresh_count():
    """The re-check's good stop (12, ours) rests; the cache still shows 20:
    the cover check uses the fresh 12, so nothing is cancelled or placed. The
    two reads disagree (one lags), so the re-check looks again."""
    good = _Trade(8, qty=12.0)
    out, place, queue, checks = _run(_ib([good], _cancels(good)), before=12.0, fresh=12.0)
    assert out is True and checks == []
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_not_called()
    queue.assert_called_once()


def test_queued_retry_after_a_fill_with_another_client_s_stop_never_over_covers():
    """Ours (3) and another client's 5-share stop rest; the cache shows 20,
    the fresh read 12. Ours is cancelled, theirs refuses: ours goes on 7, so
    the stops total the 12 held (not 20 - 5 = 15 over a 12-share position)."""
    good, theirs = _Trade(8, qty=3.0), _Trade(9, qty=5.0)
    other = {"order_id": 9, "ticker": "CRWD", "qty": 5.0, "stop_price": 140.0,
             "status": "Submitted"}
    ib = _ib([good, theirs], lambda o: good.cancelled() if o.orderId == 8 else None)
    out, place, queue, _ = _run(ib, before=8.0, fresh=12.0, after=[other])
    assert out is True
    place.assert_called_once_with("CRWD", 7.0, RECORD)
    queue.assert_called_once()                            # 9 did not cancel: re-check


def test_late_sell_nets_another_client_s_stop():
    """Pre-existing on the late path: it placed all of remaining over a stop
    the cancel could not remove. Now netted, as on the queued path."""
    theirs = _Trade(9, qty=5.0)
    other = {"order_id": 9, "ticker": "CRWD", "qty": 5.0, "stop_price": 140.0,
             "status": "Submitted"}
    out, place, queue, _ = _run(_ib([theirs]), from_queue=False, before=5.0, after=[other])
    assert out is True
    place.assert_called_once_with("CRWD", 15.0, RECORD)
    queue.assert_called_once()


def test_late_sell_others_covering_all_places_nothing():
    other = {"order_id": 9, "ticker": "CRWD", "qty": 20.0, "stop_price": 140.0,
             "status": "Submitted"}
    out, place, queue, _ = _run(_ib([_Trade(9)]), from_queue=False, before=5.0, after=[other])
    assert out is True
    place.assert_not_called()
    queue.assert_called_once()


def test_profit_target_retry_with_a_stale_cache_sizes_from_the_fresh_read():
    good = _Trade(8, qty=12.0)
    out, place, _, _ = _run(_ib([good], _cancels(good)), before=12.0, fresh=12.0,
                            intent="PROFIT_TARGET")
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)


def test_late_sell_sizes_from_the_fresh_read():
    good = _Trade(8, qty=12.0)
    out, place, _, _ = _run(_ib([good], _cancels(good)), from_queue=False, fresh=12.0)
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_fresh_read_above_the_cache_keeps_the_cache(from_queue):
    good = _Trade(8)
    out, place, _, _ = _run(_ib([good], _cancels(good)), from_queue=from_queue, fresh=25.0)
    assert out is True
    place.assert_called_once_with("CRWD", 20.0, RECORD)


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
@pytest.mark.parametrize("fresh", [
    pytest.param(ConnectionError("down"), id="unread"),
    pytest.param({}, id="empty"),
    pytest.param({"NVDA": 3.0}, id="flat"),
])
def test_an_unusable_fresh_read_with_stops_resting_cancels_nothing(from_queue, fresh):
    """S1: the fresh read comes before the cancel; unusable, the stops
    resting now stay (not cancelled into a bare position)."""
    good = _Trade(8, qty=5.0)
    out, place, queue, checks = _run(_ib([good], _cancels(good)), from_queue=from_queue,
                                     fresh=fresh)
    assert out is False and checks == []
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_not_called()
    queue.assert_called_once()                            # 5 of 20 covered: the rest may be bare


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_an_unread_fresh_read_with_nothing_resting_places_from_the_cache(from_queue):
    out, place, queue, _ = _run(_ib([]), from_queue=from_queue, before=0.0,
                                fresh=ConnectionError("down"))
    assert out is True
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_called_once()                            # sized from the cache: it may lag


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_the_cover_read_and_the_fresh_read_failing_together_cancel_nothing(from_queue):
    good = _Trade(8, qty=5.0)
    out, place, queue, _ = _run(_ib([good], _cancels(good)), from_queue=from_queue, before=None,
                                fresh=ConnectionError("down"))
    assert out is False
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_not_called()


def test_late_profit_target_with_an_unusable_fresh_read_cancels_nothing():
    good = _Trade(8, qty=5.0)
    out, place, _, _ = _run(_ib([good], _cancels(good)), from_queue=False, intent="PROFIT_TARGET",
                            fresh=ConnectionError("down"))
    assert out is False
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_not_called()


def test_queued_sizes_from_a_fresh_read_taken_after_the_cover_read():
    """S3: the sign check's early read (20) comes before the price, order
    and cover reads; a fill in that window shows in the read taken after the
    cover read (12), which sizes the stop."""
    good = _Trade(8)
    out, place, _, _ = _run(_ib([good], _cancels(good)), exact_reads=[20.0, 12.0])
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)
    assert LAST["fresh"].call_count == 2
    for c in LAST["fresh"].call_args_list:
        assert c.kwargs == {"timeout": tasks._REPROTECT_FRESH_TIMEOUT_S, "reconnect": False}


def test_queued_a_short_by_the_fresh_read_is_a_sign_flip_and_cancels_nothing():
    good = _Trade(8)
    with patch("tasks._sign_flip_alert") as flip:
        out, place, _, _ = _run(_ib([good], _cancels(good)), fresh=-4.0)
    assert out is False
    flip.assert_called_once()
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_not_called()


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_late_a_working_sell_covering_the_fresh_count_places_nothing(from_queue):
    good = _Trade(8)
    with patch("tasks._working_mkt_sell_qty", return_value=12.0):
        out, place, queue, _ = _run(_ib([good], _cancels(good)), from_queue=from_queue,
                                    fresh=12.0)
    assert out is True
    place.assert_not_called()
    LAST["ib"].cancelOrder.assert_not_called()


# ── N1: a record write failing after a good place ────────────────────────────

@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_failed_order_id_write_after_a_good_place_still_reports_it_placed(from_queue, caplog):
    ours = _Trade(8)
    out, place, queue, checks = _run(_ib([ours], _fills(ours)), from_queue=from_queue,
                                     fresh=[20.0, 12.0], record_effect=RuntimeError("db down"))
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)
    (text,) = checks
    assert "A stop went on 12 shares" in text
    assert "broker order id was not recorded" in caplog.text


# ── stops that may still rest after the cancel ───────────────────────────────

def test_queued_a_raised_cancel_without_a_fill_waits_and_rechecks():
    """S5: ours may be gone (shares bare) or still resting: the re-check."""
    out, place, queue, checks = _run(_ib([_Trade(8)], sleep_raises=True))   # SentUnknown
    assert out is False and checks == []
    place.assert_not_called()
    queue.assert_called_once()


def test_queued_a_stop_still_resting_is_netted_then_rechecked():
    """Another client's stop (12 shares) refuses the cancel: it stays as
    cover, ours goes on 8, and the re-check is queued for a double cover."""
    other = {"order_id": 9, "ticker": "CRWD", "qty": 12.0, "stop_price": 140.0,
             "status": "Submitted"}
    out, place, queue, checks = _run(_ib([_Trade(9)]), after=[other])
    assert out is True and checks == []
    place.assert_called_once_with("CRWD", 8.0, RECORD)
    queue.assert_called_once()


def test_queued_a_stop_still_resting_that_covers_it_all_still_rechecks():
    other = {"order_id": 9, "ticker": "CRWD", "qty": 20.0, "stop_price": 140.0,
             "status": "Submitted"}
    out, place, queue, _ = _run(_ib([_Trade(9)]), after=[other])
    assert out is True
    place.assert_not_called()
    queue.assert_called_once()


def test_queued_a_failed_place_after_an_unconfirmed_cancel_still_rechecks():
    other = {"order_id": 9, "ticker": "CRWD", "qty": 12.0, "stop_price": 140.0}
    out, place, queue, _ = _run(_ib([_Trade(9)]), after=[other],
                                place_effect=RuntimeError("x"))
    assert out is False and place.call_count == 1
    queue.assert_called_once()


def test_queued_an_inactive_stop_is_not_a_recheck():
    out, place, queue, checks = _run(_ib([_Trade(9, "Inactive")]))
    assert out is True and checks == []
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_not_called()


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_clean_cancel_is_unchanged(from_queue):
    t = _Trade(8)
    out, place, queue, checks = _run(_ib([t], _cancels(t)), from_queue=from_queue)
    assert out is True and checks == []
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_not_called()


def test_late_a_stop_still_resting_places_then_rechecks():
    out, place, queue, checks = _run(_ib([_Trade(9)]), from_queue=False)
    assert out is True and checks == []
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_called_once()


def test_late_a_raised_cancel_without_a_fill_waits_and_rechecks(caplog):
    out, place, queue, checks = _run(_ib([_Trade(9)], sleep_raises=True), from_queue=False)
    assert out is False and checks == []
    place.assert_not_called()                             # the cancel raised: no place
    queue.assert_called_once()                            # S5
    assert "the cancel failed" in caplog.text and "stop placement" not in caplog.text   # N2


# ── NIT 1: a confirmed cancel after a part fill reads as Filled ──────────────

@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_ibkr_a_cancel_after_a_part_fill_is_counted_and_listed_filled(parallel, caplog):
    part, clean = _Trade(8), _Trade(9)
    ib = _ib([part, clean], lambda o: (part.cancelled(5) if o.orderId == 8 else clean.cancelled()))
    stuck: list = []
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._cancel_parallel_enabled", return_value=parallel):
        assert ibkr.cancel_stop_orders_for("CRWD", stuck_out=stuck) == 2
    assert stuck == [(8, "Filled")]
    assert "but 5 shares FILLED before it" in caplog.text


def test_ibkr_cancel_order_still_confirms_a_part_filled_cancel():
    part = _Trade(8)
    with patch("ibkr.get_ib", return_value=_ib([part], lambda o: part.cancelled(5))):
        assert ibkr.cancel_order(8, symbol="CRWD") is True
    assert ibkr._LAST_CANCEL_STATUS[8] == "Filled"


@pytest.mark.parametrize("filled", [0.0, None, MagicMock()], ids=["zero", "none", "mock"])
def test_ibkr_a_cancel_with_nothing_filled_is_unchanged(filled):
    t = _Trade(8)
    t.orderStatus.filled = filled
    stuck: list = []
    with patch("ibkr.get_ib", return_value=_ib([t], _cancels(t))), \
         patch("ibkr._cancel_parallel_enabled", return_value=False):
        assert ibkr.cancel_stop_orders_for("CRWD", stuck_out=stuck) == 1
    assert stuck == []


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_re_protect_a_part_filled_cancel_takes_the_fill_route(from_queue):
    part = _Trade(8)
    out, place, queue, checks = _run(_ib([part], _cancels(part, part=5)), from_queue=from_queue,
                                     fresh=[20.0, 15.0])
    assert out is True
    place.assert_called_once_with("CRWD", 15.0, RECORD)
    queue.assert_called_once()
    assert len(checks) == 1 and "orderIds [8]" in checks[0]


def test_add_a_part_filled_cancel_takes_the_fill_route():
    """The ADD's whole-position stop (execution._whole_position_stop): an old
    stop cancelled after a part fill is reported Filled, so only the new
    shares are stopped and the record is kept."""
    from test_whole_stop_cancel_check import NEW_STOP, _Trade as _AddTrade, _real_cancel, _wps
    old = _AddTrade(8, "Submitted")

    def part(o):
        old.orderStatus.filled = 1.0
        old.cancelled()
    status: dict = {"alerts_out": []}
    out, _, _, _ = _wps(_real_cancel([old], part), status=status)
    assert out == (NEW_STOP, None)
    assert status["old_stop_filled"] is True and "orderIds [8]" in status["unknown_reason"]


# ── the re-check dedupe ──────────────────────────────────────────────────────

def _queue(*attempts, ticker="CRWD", apply_effect=None):
    with patch.object(tasks.recheck_stop_coverage, "apply_async",
                      side_effect=apply_effect) as aa:
        outs = [tasks._queue_stop_coverage_recheck(ticker, a) for a in attempts]
    return outs, aa


def _key(ticker="CRWD"):
    return tasks._RECHECK_KEY.format(ticker)


def _restart(ticker="CRWD"):
    return tasks._RECHECK_RESTART_KEY.format(ticker)


def test_dedupe_one_queued_recheck_per_ticker():
    outs, aa = _queue(1, 1)
    assert outs == [True, True]
    aa.assert_called_once_with(args=["CRWD"], kwargs={"attempt": 1}, countdown=60)
    assert _FakeRedis.expires[_key()] == 50.0


def test_dedupe_key_expires_before_the_queued_run():
    assert tasks._RECHECK_DEDUPE_S < tasks._RECHECK_COUNTDOWN_S == 60
    assert tasks._RECHECK_RESTART_S > tasks._RECHECK_COUNTDOWN_S


def test_a_request_after_the_key_expires_queues():
    _queue(1)
    _FakeRedis.clock += tasks._RECHECK_DEDUPE_S + 1
    outs, aa = _queue(1)
    assert outs == [True]
    aa.assert_called_once()
    assert _restart() not in _FakeRedis.store            # not a skip: no marker


def test_dedupe_is_per_ticker():
    _, aa1 = _queue(1)
    _, aa2 = _queue(1, ticker="NVDA")
    assert aa1.call_count == 1 and aa2.call_count == 1


def test_the_chain_s_next_looks_always_queue_and_refresh_the_key():
    outs, aa = _queue(1, 2, 3)
    assert outs == [True, True, True] and aa.call_count == 3


def test_the_cap_is_unchanged():
    outs, aa = _queue(tasks._COVERAGE_RECHECK_TRIES + 1)
    assert outs == [False]
    aa.assert_not_called()


@pytest.mark.parametrize("fail", ["client", "set"])
def test_an_unreadable_redis_queues_as_before(fail, caplog):
    broken = MagicMock()
    broken.set.side_effect = ConnectionError("down")
    with patch("tasks._recheck_redis", side_effect=ConnectionError("no client")) if fail == "client" \
            else patch("tasks._recheck_redis", return_value=broken):
        outs, aa = _queue(1, 1)
    assert outs == [True, True] and aa.call_count == 2    # fail open: duplicates, not misses
    assert "dedupe unreadable" in caplog.text


def test_a_failed_queue_clears_the_key_so_the_next_request_queues():
    outs, aa = _queue(1, 1, apply_effect=[RuntimeError("broker down"), None])
    assert outs == [False, True] and aa.call_count == 2


def test_an_add_and_a_re_protect_within_the_minute_start_one_chain():
    """The ADD's whole-position stop (execution) and a re-protect on the same
    ticker both ask for the re-check: one is queued."""
    ours = _Trade(8)
    _queue(1)                                             # as execute_approved after its place
    out, place, queue, checks = _run(_ib([ours], _fills(ours)), fresh=[20.0, 12.0])
    assert out is True and len(checks) == 1
    queue.assert_not_called()                             # deduped: the first one covers it
    assert _restart() in _FakeRedis.store


# ── SF-2: a skipped request keeps its full count of looks ────────────────────

def test_a_skip_leaves_the_restart_marker():
    _queue(1)
    _queue(1)
    assert _FakeRedis.store.get(_restart()) == "1"
    assert _FakeRedis.expires[_restart()] == tasks._RECHECK_RESTART_S


def test_a_failed_marker_write_queues_instead_of_skipping():
    class _NoMarker(_FakeRedis):
        def set(self, key, value, ex=None, nx=False):
            if key == _restart():
                raise ConnectionError("down")
            return super().set(key, value, ex=ex, nx=nx)
    with patch("tasks._recheck_redis", return_value=_NoMarker()):
        _queue(1)
        outs, aa = _queue(1)
    assert outs == [True]
    aa.assert_called_once()                               # queued anyway


def _recheck(attempt, pending=True, busy=False):
    """recheck_stop_coverage with a cancel still pending (so it continues)
    or, busy, the ticker lock taken. Returns the follow-up attempts queued."""
    import stop_lock
    rows = [{"status": "PendingCancel"}] if pending else []
    lock = (patch("tasks.ticker_lock", side_effect=stop_lock.TickerBusy("CRWD")) if busy
            else patch("tasks.ticker_lock"))
    with lock, patch("tasks._resting_sell_stops", return_value=rows), \
         patch("tasks._recheck_position_reads", return_value={}), \
         patch("tasks._fix_overstop_safe", return_value=""), \
         patch.object(tasks.recheck_stop_coverage, "apply_async") as aa:
        tasks.recheck_stop_coverage.run("CRWD", attempt=attempt)
    return [c.kwargs["kwargs"]["attempt"] for c in aa.call_args_list]


def test_the_last_look_with_a_skip_continues_with_a_fresh_count():
    """The chain's attempt 6 was queued (key refreshed), then a new request
    was skipped: attempt 6 is that request's first look, so its follow-up is
    attempt 2 and five more looks remain, not none."""
    _queue(tasks._COVERAGE_RECHECK_TRIES)                 # the chain's last look, queued
    _queue(1)                                             # skipped
    _FakeRedis.clock += tasks._RECHECK_COUNTDOWN_S
    assert _recheck(tasks._COVERAGE_RECHECK_TRIES) == [2]
    assert _restart() not in _FakeRedis.store             # taken


def test_without_a_skip_the_last_look_stops_at_the_cap():
    assert _recheck(tasks._COVERAGE_RECHECK_TRIES) == []


def test_a_restarted_chain_gets_the_full_count():
    _queue(tasks._COVERAGE_RECHECK_TRIES)
    _queue(1)
    looks, attempt = 1, tasks._COVERAGE_RECHECK_TRIES     # the skipped request's first look
    while True:
        _FakeRedis.clock += tasks._RECHECK_COUNTDOWN_S
        nxt = _recheck(attempt)
        if not nxt:
            break
        (attempt,) = nxt
        looks += 1
    assert looks == tasks._COVERAGE_RECHECK_TRIES


def test_a_busy_ticker_with_a_skip_restarts_at_attempt_1():
    _queue(tasks._COVERAGE_RECHECK_TRIES)
    _queue(1)
    _FakeRedis.clock += tasks._RECHECK_COUNTDOWN_S
    assert _recheck(tasks._COVERAGE_RECHECK_TRIES, busy=True) == [1]


def test_a_clean_run_takes_the_marker_and_stops():
    _queue(3)
    _queue(1)
    _FakeRedis.clock += tasks._RECHECK_COUNTDOWN_S
    assert _recheck(3, pending=False) == []
    assert _restart() not in _FakeRedis.store


def test_an_unreadable_marker_counts_on_as_before():
    with patch("tasks._recheck_redis", side_effect=ConnectionError("down")):
        assert tasks._recheck_take_restart("CRWD") is False


# ── NIT 3: one cached client ─────────────────────────────────────────────────

def test_the_dedupe_client_is_made_once_with_short_timeouts(monkeypatch):
    import redis
    monkeypatch.setattr(tasks, "_RECHECK_REDIS_CLIENT", None)
    with patch.object(redis.Redis, "from_url", return_value=MagicMock()) as make:
        first, second = _REAL_RECHECK_REDIS(), _REAL_RECHECK_REDIS()
    assert first is second
    make.assert_called_once()
    assert make.call_args.kwargs["socket_timeout"] == tasks._RECHECK_REDIS_TIMEOUT_S == 1.0
    assert make.call_args.kwargs["socket_connect_timeout"] == 1.0


def test_add_a_part_filled_cancel_queues_the_coverage_recheck(monkeypatch):
    """N3: the ADD's part-fill route stops the new lot; the re-check covers
    the old shares left."""
    from test_whole_stop_cancel_check import _Trade as _AddTrade, _execute, _real_cancel
    old = _AddTrade(8, "Submitted")

    def part(o):
        old.orderStatus.filled = 1.0
        old.cancelled()
    place, db_fns, recheck, reprotect, checks, events = _execute(monkeypatch,
                                                                 _real_cancel([old], part))
    assert place.call_count == 1 and place.call_args.args[1] == 1      # the new lot only
    recheck.assert_called_once_with("NVDA")
    db_fns["upsert_position_stop"].assert_not_called()                 # the record is kept


# ── N4: a run that took the marker and stopped without a look ────────────────

def test_a_run_without_a_look_writes_the_marker_back():
    _queue(tasks._COVERAGE_RECHECK_TRIES)
    _queue(1)
    _FakeRedis.clock += tasks._RECHECK_COUNTDOWN_S
    with patch("tasks.ticker_lock"), \
         patch("tasks._resting_sell_stops", side_effect=ConnectionError("down")), \
         patch.object(tasks.recheck_stop_coverage, "apply_async") as aa:
        tasks.recheck_stop_coverage.run("CRWD", attempt=tasks._COVERAGE_RECHECK_TRIES)
    aa.assert_not_called()
    assert _FakeRedis.store.get(_restart()) == "1"


def test_without_a_marker_a_run_without_a_look_writes_none():
    with patch("tasks.ticker_lock"), \
         patch("tasks._resting_sell_stops", side_effect=ConnectionError("down")):
        tasks.recheck_stop_coverage.run("CRWD", attempt=2)
    assert _restart() not in _FakeRedis.store


# ── S2: the bounded fresh read ───────────────────────────────────────────────

_REAL_GET_FRESH = ibkr.get_fresh_stock_positions   # before conftest's per-test fake


def _real_fresh(*a, **k):
    return _REAL_GET_FRESH(*a, **k)


def test_the_fresh_read_without_reconnect_uses_this_session_only(monkeypatch):
    monkeypatch.setattr(ibkr, "_ib", None)
    with patch("ibkr.get_ib") as get_ib:
        with pytest.raises(ConnectionError):
            _real_fresh(timeout=5, reconnect=False)
    get_ib.assert_not_called()


def test_the_fresh_read_passes_its_bound(monkeypatch):
    ib = MagicMock()
    ib.isConnected.return_value = True
    monkeypatch.setattr(ibkr, "_ib", ib)
    row = MagicMock(position=12.0, account="A", contract=MagicMock(symbol="CRWD", secType="STK",
                                                                   conId=1))
    with patch("ibkr.get_ib") as get_ib, \
         patch("ibkr._bounded_req", return_value=[row]) as req:
        assert _real_fresh(timeout=5, reconnect=False) == {"CRWD": 12.0}
    get_ib.assert_not_called()
    assert req.call_args.kwargs["timeout"] == 5


def test_the_fresh_read_defaults_are_unchanged():
    ib = MagicMock()
    with patch("ibkr.get_ib", return_value=ib) as get_ib, \
         patch("ibkr._bounded_req", return_value=[]) as req:
        assert _real_fresh() == {}
    get_ib.assert_called_once()
    assert req.call_args.kwargs["timeout"] is None


@pytest.mark.parametrize("action, word", [("SELL", "sold"), ("BUY", "bought")])
def test_the_part_fill_line_follows_the_order_s_side(action, word):
    """N2: cancel_order also cancels stale market BUYs."""
    t = _Trade(8)
    t.order.action = action
    t.orderStatus.filled = 5.0
    line = ibkr._cancel_part_filled_line(8, t, "Cancelled")
    assert f"5 shares FILLED before it — shares were {word}" in line and f"({action} " in line



# ── the fourth review: S1, S2, S4 and a failed place ─────────────────────────

@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_s1_a_fully_covered_position_is_done_even_when_the_fresh_read_fails(from_queue):
    ours = _Trade(8, qty=25.0)
    out, place, queue, _ = _run(_ib([ours], _cancels(ours)), from_queue=from_queue, before=25.0,
                                fresh=ConnectionError("down"))
    assert out is True
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_not_called()


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
@pytest.mark.parametrize("fresh", [pytest.param(-4.0, id="short"),
                                   pytest.param({"NVDA": 3.0}, id="flat")])
def test_s2_short_or_flat_by_the_fresh_read_with_nothing_resting_places_nothing(from_queue,
                                                                                 fresh):
    with patch("tasks._sign_flip_alert") as flip:
        out, place, queue, _ = _run(_ib([]), from_queue=from_queue, before=0.0, fresh=fresh)
    assert out is False
    place.assert_not_called()
    assert flip.call_count == (1 if fresh == -4.0 else 0)


def test_s4_late_with_nothing_resting_cancels_nothing_and_reads_nothing_after():
    out, place, _, _ = _run(_ib([]), from_queue=False, before=0.0)
    assert out is True
    LAST["ib"].openTrades.assert_not_called()             # no cancel pass
    LAST["left"].assert_not_called()                      # no look after a cancel
    place.assert_called_once_with("CRWD", 20.0, RECORD)


def test_s4_late_look_after_the_cancel_is_one_bounded_read():
    t = _Trade(8)
    out, place, _, _ = _run(_ib([t], _cancels(t)), from_queue=False)
    assert out is True
    LAST["left"].assert_called_once_with("CRWD", timeout=tasks._REPROTECT_FRESH_TIMEOUT_S,
                                         reconnect=False)


def test_s4_late_unreadable_look_after_the_cancel_stops_all_and_rechecks():
    t = _Trade(8)
    ib = _ib([t], _cancels(t))
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=HELD)]
    with patch("ibkr.get_client", return_value=client), patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_ib", return_value=ib), patch("ibkr._cancel_parallel_enabled", return_value=False), \
         patch("ibkr.get_fresh_stock_positions", return_value={"CRWD": HELD}), \
         patch("tasks._is_orders_worker", return_value=True), \
         patch("db.get_position_stop", return_value={"stop_price": RECORD}), \
         patch("db.set_stop_broker_order_id"), \
         patch("execution._stops_still_resting", return_value=5.0), \
         patch("tasks._resting_sell_stops", side_effect=TimeoutError("5 s")) as left, \
         patch("risk_manager._floor_at_resting_stop", side_effect=lambda t_, lv, *a, **k: (lv, None)), \
         patch("risk_manager._stop_below_market", side_effect=lambda t_, lv, *a, **k: lv), \
         patch("time.sleep"), patch("alerts._send"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch.object(tasks.recheck_stop_coverage, "apply_async") as queue:
        out = tasks._reprotect_after_late_sell("CRWD", {"intent": "RESTORE"}, 140.0)
    assert out is True
    left.assert_called_once()                             # one try, no second look
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_called_once()


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
@pytest.mark.parametrize("before", [0.0, 5.0], ids=["no_cancel", "after_cancel"])
def test_a_failed_place_always_queues_the_recheck(from_queue, before):
    t = _Trade(8)
    out, place, queue, _ = _run(_ib([t], _cancels(t)), from_queue=from_queue, before=before,
                                place_effect=RuntimeError("x"))
    assert out is False and place.call_count == 1
    queue.assert_called_once()


# ── under-cover: stops still being cancelled, maybe-filled statuses ──────────

def _row(qty, status="Submitted", oid=8):
    return {"order_id": oid, "ticker": "CRWD", "qty": qty, "stop_price": 140.0, "status": status}


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_cancelling_stop_gets_a_second_look_and_is_not_cover(from_queue):
    """A stop still being cancelled covers all 20 on the first look; 5 s
    later it is gone: nothing rests, so ours goes on all 20, no cancel."""
    out, place, queue, _ = _run(_ib([]), from_queue=from_queue,
                                cover=[[_row(20.0, "PendingCancel")], []])
    assert out is True
    assert LAST["cover"].call_count == 2
    LAST["sleep"].assert_any_call(5)
    LAST["ib"].cancelOrder.assert_not_called()
    place.assert_called_once_with("CRWD", 20.0, RECORD)


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_stop_still_cancelling_after_the_second_look_is_not_done(from_queue):
    """Still cancelling 5 s later: covered (20) >= the cache, but none of it
    is cover, so it is not "done": ours is cancelled and placed on 20, and
    the re-check looks again (the cancel may be rejected)."""
    ours = _Trade(8)
    out, place, queue, _ = _run(_ib([ours], _cancels(ours)), from_queue=from_queue,
                                cover=[[_row(20.0, "ApiPendingCancel")]] * 2)
    assert out is True
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_called_once()


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_stop_left_cancelling_after_ours_is_not_netted(from_queue, caplog):
    """After the cancel an 8-share stop is still being cancelled: it is not
    netted (ours goes on all 20, not 12), the re-check looks again, and the
    log doesn't call it cover."""
    ours = _Trade(8, qty=8.0)
    out, place, queue, _ = _run(_ib([ours], _cancels(ours)), from_queue=from_queue,
                                cover=[[_row(8.0)]], after=[_row(8.0, "PendingCancel", oid=9)])
    assert out is True
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_called_once()
    assert "stay covered" not in caplog.text
    assert "8 shares are still being cancelled, so not counted as cover ($140.00)" in caplog.text


def test_the_log_splits_cover_left_from_stops_still_cancelling(caplog):
    ours = _Trade(8, qty=8.0)
    out, place, _, _ = _run(_ib([ours], _cancels(ours)), cover=[[_row(8.0)]],
                            after=[_row(5.0, oid=9), _row(8.0, "PendingCancel", oid=10)])
    assert out is True
    place.assert_called_once_with("CRWD", 15.0, RECORD)
    assert "5 shares stay covered by stop(s) the cancel did not remove ($140.00)" in caplog.text
    assert "8 shares are still being cancelled, so not counted as cover" in caplog.text


@pytest.mark.parametrize("status", ["NotFound", "SentUnknown", None])
@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_status_that_may_mean_filled_sizes_from_a_fresh_read_after(from_queue, status):
    """The old stop sold 8 of 20 and its cancel came back NotFound (or
    SentUnknown, or no status): the read after the cancel shows 12, so ours
    goes on 12 (not 20 over a 12-share position), the re-check looks again,
    and no FILLED alert goes out (the fill is not confirmed)."""
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [(8, status)] if status else [8]
    with patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel):
        out, place, queue, checks = _run(_ib([]), from_queue=from_queue, before=8.0,
                                         fresh=[20.0, 12.0])
    assert out is True
    place.assert_called_once_with("CRWD", 12.0, RECORD)
    queue.assert_called_once()
    assert checks == []


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_maybe_filled_status_with_no_fresh_read_after_places_nothing(from_queue):
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [(8, "NotFound")]
    with patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel):
        out, place, queue, checks = _run(_ib([]), from_queue=from_queue, before=8.0,
                                         fresh=[20.0, ConnectionError("down")])
    assert out is False
    place.assert_not_called()
    queue.assert_called_once()
    assert checks == []


def test_a_pending_cancel_status_is_not_a_maybe_fill():
    """A cancel still pending is not a maybe-fill: no read after the cancel;
    ours goes on as before and the re-check trims a double cover."""
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [(8, "PendingCancel")]
    with patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel):
        out, place, queue, _ = _run(_ib([]), before=8.0, fresh=20.0)
    assert out is True
    assert LAST["fresh"].call_count == 2          # the early sign check and the one before
    place.assert_called_once_with("CRWD", 20.0, RECORD)
    queue.assert_called_once()


def test_reprotect_cancel_lists_the_maybe_filled_ones():
    def cancel(ticker, stuck_out=None):
        stuck_out[:] = [(8, "NotFound"), (9, "SentUnknown"), 10, (11, "PendingCancel"),
                        (12, "Filled"), (13, "Inactive")]
    amb: list = []
    with patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel):
        filled, resting, exc = tasks._reprotect_cancel("CRWD", ambiguous_out=amb)
    assert filled == [12] and resting == [8, 9, 10, 11] and amb == [8, 9, 10] and exc is None


# ── the caller's outcome_out ────────────────────────────────────────────────

@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_outcome_says_covered_when_other_stops_cover_it(from_queue):
    outcome: dict = {}
    out, place, _, _ = _run(_ib([]), from_queue=from_queue, before=20.0, outcome=outcome)
    assert out is True and outcome == {"placed": 0.0, "by": "stops", "broker_qty": 0.0}
    place.assert_not_called()


def test_outcome_says_covered_after_the_cancel_too():
    outcome: dict = {}
    ours = _Trade(8, qty=8.0)
    out, place, _, _ = _run(_ib([ours], _cancels(ours)), before=8.0, outcome=outcome,
                            after=[_row(20.0, oid=9)])
    assert out is True and outcome == {"placed": 0.0, "by": "stops", "broker_qty": 0.0}
    place.assert_not_called()


def test_outcome_gives_the_placed_shares():
    outcome: dict = {}
    out, _, _, _ = _run(_ib([]), before=0.0, outcome=outcome)
    assert out is True and outcome == {"placed": 20.0}


def test_outcome_gives_the_cover_on_a_wait_that_cancelled_nothing():
    """Fresh read unread while an old stop rests: nothing is cancelled, so
    the old stop still covers what it was sized for (5 shares)."""
    outcome: dict = {}
    good = _Trade(8, qty=5.0)
    out, place, _, _ = _run(_ib([good], _cancels(good)), before=5.0, outcome=outcome,
                            fresh=ConnectionError("down"))
    assert out is False and outcome == {"cover": 5.0}
    LAST["ib"].cancelOrder.assert_not_called()


def test_outcome_cover_leaves_out_stops_being_cancelled():
    outcome: dict = {}
    out, _, _, _ = _run(_ib([]), outcome=outcome, fresh=ConnectionError("down"),
                        cover=[[_row(5.0, "PendingCancel"), _row(3.0, oid=9)]] * 2)
    assert out is False and outcome == {"cover": 3.0}


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_outcome_says_lag_when_stops_cover_only_the_fresh_count(from_queue):
    outcome: dict = {}
    out, place, queue, _ = _run(_ib([]), from_queue=from_queue, before=12.0, fresh=12.0,
                                outcome=outcome)
    assert out is True and outcome == {"placed": 0.0, "by": "lag", "broker_qty": 12.0}
    place.assert_not_called()
    queue.assert_called_once()


def test_outcome_says_sell_when_a_working_sell_covers_the_fresh_count():
    """Probe from review 2: cache 20 less a working SELL of 5, fresh 3: the
    SELL covers what is held; no stops rest."""
    outcome: dict = {}
    with patch("tasks._working_mkt_sell_qty", return_value=5.0):
        out, place, queue, _ = _run(_ib([]), from_queue=False, before=0.0, held=25.0,
                                    fresh=3.0, outcome=outcome)
    assert out is True and outcome == {"placed": 0.0, "by": "sell", "broker_qty": 0.0}
    place.assert_not_called()
    queue.assert_called_once()


def test_outcome_says_sell_when_a_working_sell_covers_everything():
    outcome: dict = {}
    with patch("tasks._working_mkt_sell_qty", return_value=20.0):
        out, place, _, _ = _run(_ib([]), from_queue=False, outcome=outcome)
    assert out is True and outcome == {"placed": 0.0, "by": "sell"}
    place.assert_not_called()


@pytest.mark.parametrize("from_queue", [True, False], ids=["queued", "late"])
def test_a_wait_after_a_fill_with_left_unread_logs_no_stopping_all(from_queue, caplog):
    ours = _Trade(8)
    with patch("tasks._reprotect_left_after_cancel", return_value=None):
        out, place, _, _ = _run(_ib([ours], _fills(ours)), from_queue=from_queue, before=8.0,
                                fresh=[20.0, ConnectionError("down")])
    assert out is False
    place.assert_not_called()
    assert "stopping all" not in caplog.text
