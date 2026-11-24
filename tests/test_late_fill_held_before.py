"""Roadmap 04ah: a late fill of an OPEN on a ticker that was already held.
execute_approved stores the held count before the order on its row
(orders.held_before_qty); tasks._late_fill_held_decide reads it when the late
shares are booked. "shadow" changes nothing and writes a late_fill_held_before
row; "enforce" books the OPEN as an ADD (record and its level kept) only when
the count was above 0 and the old shares are still held."""
import importlib.util
import inspect
import os
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
import db
import execution
import ibkr
import tasks

_H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_order_row_before_fill_wait.py")
_spec = importlib.util.spec_from_file_location("_order_row_harness", _H)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

NOW = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
OLD_STOP, NEW_STOP = 95.0, 90.0


def _approval(action="OPEN"):
    return {"id": 9, "ticker": "NVDA", "action": action, "conviction": 7, "status": "APPROVED",
            "thesis_bucket": "Accelerator", "stop_loss_price": NEW_STOP,
            "risk_json": {"stop_loss_price": NEW_STOP, "price_at_sizing": 100.0,
                          "stop_method": "2.0x ATR"}}


def _order(prev_qty=2.0):
    return {"id": 77, "ticker": "NVDA", "side": "BUY", "intent": "OPEN", "approval_id": 9,
            "broker_order_id": 501,
            "qty": 4.0, "filled_qty": prev_qty, "fill_price": 100.0,
            "created_at": NOW - timedelta(minutes=20)}


class _Redis:
    """stop_lock._redis() stand-in: a pipeline of INCR + EXPIRE."""
    def __init__(self):
        self.n = {}
        self.round_trips = 0

    def pipeline(self):
        r, ops = self, []

        class P:
            def incr(self, key):
                ops.append(("incr", key))

            def expire(self, key, ttl):
                ops.append(("expire", key, ttl))

            def execute(self):
                r.round_trips += 1
                out = []
                for op in ops:
                    if op[0] == "incr":
                        r.n[op[1]] = r.n.get(op[1], 0) + 1
                        out.append(r.n[op[1]])
                    else:
                        out.append(True)
                return out
        return P()


def _book(mode, stored, held=None, held_error=None, prev_qty=2.0, fill_qty=2.0,
          action="OPEN", record=True, events=None, progress=None, redis_fake=None,
          reread=None, reread_error=None, other_buys=0, fresh=None, sld=0, other_bot=0):
    """5 held with a $95 stop record; an OPEN of 4 filled 2 in the wait and
    2 late. held: this process's position cache (ibkr.cached_stock_qty);
    reread: the order's cumulative fill re-read (default: what was booked).
    Returns (upserts, reprotect calls, activity rows). events gets
    ("read", stop_work_done) for the held-count read, ("reprotect",) for the
    stop step and ("row",) for the activity row, in order."""
    upserts, rows = [], []
    events = events if events is not None else []
    progress = progress if progress is not None else {}

    def read_stored(order_id):
        events.append(("read", progress.get("stop_work_done")))
        if isinstance(stored, BaseException):
            raise stored
        return stored

    def reprotect_call(*a, **k):
        events.append(("reprotect",))
        return True

    def row(stage, t, msg, s="neutral", extra=None):
        if stage == "late_fill_expect" and config.LATE_FILL_EXPECT_MODE != "off":
            return          # the late-BUY expect shadow's own row (test_stop_qty_more.py)
        events.append(("row",))
        rows.append((stage, t, msg, extra))
    rec = {"ticker": "NVDA", "stop_price": OLD_STOP, "entry_price": 80.0} if record else None
    booked = round(prev_qty + fill_qty, 4)
    with ExitStack() as st:
        st.enter_context(patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", mode))
        st.enter_context(patch("db.get_order_held_before", side_effect=read_stored))
        st.enter_context(patch("stop_lock._redis", return_value=redis_fake or _Redis()))
        st.enter_context(patch("db.get_position_stop", return_value=rec))
        st.enter_context(patch("db.upsert_position_stop",
                               side_effect=lambda **kw: upserts.append(kw)))
        st.enter_context(patch("db.log_activity_queued", side_effect=row))
        st.enter_context(patch("db.count_other_buy_orders_since", return_value=other_buys))
        st.enter_context(patch("ibkr.cached_stock_qty",
                               side_effect=held_error or (lambda t: 0.0 if held is None else held)))
        st.enter_context(patch("ibkr.order_exec_check",
                               side_effect=reread_error or
                               (lambda *a, **k: {"own_bot": booked if reread is None else reread,
                                                 "sld": sld, "other_bot": other_bot})))
        st.enter_context(patch("ibkr.get_fresh_stock_positions",
                               return_value={"NVDA": fresh} if fresh is not None else {}))
        st.enter_context(patch("execution._stop_anchored_to_fill", return_value=NEW_STOP))
        st.enter_context(patch("execution._record_stale_entry"))
        st.enter_context(patch("tasks._apply_buy_outcome"))
        st.enter_context(patch("tasks._queue_reprotect"))
        st.enter_context(patch("alerts._send"))
        reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell",
                                           side_effect=reprotect_call))
        tasks._book_late_fill(_order(prev_qty), fill_qty, 101.0, 100.5, prev_qty,
                              trade_logged=True, approval=_approval(action), progress=progress)
    return upserts, reprotect.call_args_list, rows


# ---- OPEN on a held ticker: 5 held, OPEN of 4, 2 in the wait, 2 late ----

def test_shadow_books_as_today_and_writes_the_row():
    upserts, reprotect, rows = _book("shadow", stored=5.0, held=9.0, fresh=9.0)
    # today's behaviour: the record rewritten at this order's level, the 5 old shares' stop lowered
    assert len(upserts) == 1 and upserts[0]["stop_price"] == NEW_STOP and len(reprotect) == 1
    assert len(rows) == 1
    stage, ticker, msg, extra = rows[0]
    assert stage == "late_fill_held_before" and ticker == "NVDA"
    assert extra == {"order_id": 77, "approval_id": 9, "ticker": "NVDA", "stored_held_before": 5.0,
                     "held_now": 9.0, "held_read": "ok", "order_filled_qty": 4.0,
                     "order_qty": 4.0, "fill_reread": 4.0, "fill_reread_read": "ok",
                     "cum_fill_used": 4.0, "other_buys_since": 0,
                     "sld_execs_since": 0, "other_bot_execs_since": 0,
                     "prev_qty": 2.0, "fill_qty": 2.0, "pass_no": 1,
                     "held_fresh": 9.0, "held_fresh_read": "ok",
                     "action_today": "OPEN", "action_would": "ADD", "mode": "shadow",
                     "applied": False}
    assert "shadow" in msg


def test_enforce_takes_the_add_path_and_keeps_the_floor():
    upserts, reprotect, rows = _book("enforce", stored=5.0, held=9.0)
    assert upserts == []                                     # the $95 record kept, not rewritten at $90
    assert len(reprotect) == 1                               # whole position re-stopped from the record
    assert reprotect[0].kwargs.get("from_queue") is True
    assert rows[0][3]["action_would"] == "ADD" and rows[0][3]["applied"] is True
    assert "held_fresh" not in rows[0][3]                    # no fresh reqPositions in enforce


def test_a_position_read_taken_while_the_order_still_fills_is_not_old_shares():
    """Review repro: held 5, OPEN of 4, the fill lookup saw 2, the position
    reads 4 (the old 5 sold, the order filled in full meanwhile). The re-read
    after the position read shows the full 4: 4 - 4 = 0 of the 5."""
    upserts, _, rows = _book("enforce", stored=5.0, held=4.0, prev_qty=0.0, fill_qty=2.0,
                             reread=4.0)
    assert len(upserts) == 1 and upserts[0]["stop_price"] == NEW_STOP     # today's behaviour
    assert rows[0][3]["action_would"] == "OPEN" and rows[0][3]["applied"] is False
    # the re-read failing falls back to the whole order: still OPEN
    _, _, rows = _book("enforce", stored=5.0, held=4.0, prev_qty=0.0, fill_qty=2.0,
                       reread_error=TimeoutError())
    assert rows[0][3]["cum_fill_used"] == 4.0 and rows[0][3]["action_would"] == "OPEN"


def test_pass_one_of_a_partial_can_qualify_from_the_reread():
    """5 old + 2 filled so far of an OPEN of 4, and the re-read confirms only
    2: 7 - 2 = 5, all the old shares. Against the whole order it was OPEN."""
    _, _, rows = _book("enforce", stored=5.0, held=7.0, prev_qty=0.0, fill_qty=2.0, reread=2.0)
    assert rows[0][3]["cum_fill_used"] == 2.0 and rows[0][3]["action_would"] == "ADD"
    _, _, rows = _book("enforce", stored=5.0, held=7.0, prev_qty=0.0, fill_qty=2.0,
                       reread_error=TimeoutError())
    assert rows[0][3]["action_would"] == "OPEN"


@pytest.mark.parametrize("other, would", [(0, "ADD"), (1, "OPEN"), (None, "OPEN")])
def test_another_buy_since_the_order_blocks_the_add(other, would):
    """Shares another BUY bought since this order would read as old ones."""
    _, _, rows = _book("enforce", stored=5.0, held=9.0, other_buys=other)
    assert rows[0][3]["other_buys_since"] == other and rows[0][3]["action_would"] == would


def test_shadow_reads_only_after_the_stop_step():
    events, progress = [], {}
    _book("shadow", stored=5.0, held=9.0, events=events, progress=progress)
    assert events == [("reprotect",), ("read", True), ("row",)]   # after stop_work_done is set


def test_enforce_decides_before_the_stop_and_writes_its_row_after():
    events = []
    _book("enforce", stored=5.0, held=9.0, events=events)
    assert events == [("read", None), ("reprotect",), ("row",)]


def test_enforce_reads_cannot_drop_the_session():
    """The decision before the stop reads the position cache (no request)
    and re-reads executions bounded without a session drop."""
    src = inspect.getsource(tasks._late_fill_held_decide)
    assert "cached_stock_qty" in src and "get_fresh_stock_positions" not in src
    assert "order_exec_check" in src

    def code(fn):
        return inspect.getsource(fn).split('"""')[2]            # without the docstring
    assert "drop_session=False" in code(ibkr.order_exec_check)
    assert "get_ib(" not in code(ibkr.order_exec_check)
    assert "get_ib(" not in code(ibkr.cached_stock_qty)
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.reqExecutions.side_effect = TimeoutError()
    with patch.object(ibkr, "_ib", ib):
        with pytest.raises(TimeoutError):
            ibkr.order_exec_check(501, "NVDA", timeout=5)
        assert ibkr._ib is ib                                     # session kept
    ib.disconnect.assert_not_called()


def _ex(oid, sym, side, shares, t=None):
    e = MagicMock()
    e.execution.orderId, e.execution.side, e.execution.shares = oid, side, shares
    e.contract.symbol = sym
    e.t = t
    return e


def _exec_ib(execs):
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.reqExecutions.return_value = execs
    return ib


def test_order_exec_check_counts_this_order_other_buys_and_sales():
    ib = _exec_ib([_ex(501, "NVDA", "BOT", 2), _ex(501, "NVDA", "BOT", 1),
                   _ex(501, "AMD", "BOT", 5), _ex(502, "NVDA", "BOT", 7),
                   _ex(503, "NVDA", "SLD", 1), _ex(504, "AMD", "SLD", 1)])
    with patch.object(ibkr, "_ib", ib), patch("ibkr._exec_time", side_effect=lambda e: e.t):
        assert ibkr.order_exec_check(501, "NVDA") == {"own_bot": 3.0, "other_bot": 1, "sld": 1}
    with patch.object(ibkr, "_ib", None), pytest.raises(ConnectionError):
        ibkr.order_exec_check(501, "NVDA")


def _decide_with_execs(execs, held=7.0, prev_qty=0.0, fill_qty=2.0):
    """The real executions pass over a fake IB: created_at = NOW - 20 min."""
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "enforce"), \
         patch("db.get_order_held_before", return_value=5.0), \
         patch("db.count_other_buy_orders_since", return_value=0), \
         patch("ibkr.cached_stock_qty", return_value=held), \
         patch.object(ibkr, "_ib", _exec_ib(execs)), \
         patch("ibkr._exec_time", side_effect=lambda e: e.t):
        return tasks._late_fill_held_decide(_order(prev_qty), _approval(), fill_qty, prev_qty,
                                            "enforce")


def test_the_first_execution_just_before_created_at_still_counts():
    """Review SF2: created_at is NOW() after submission and execution times
    are whole seconds, so the first fill can stamp before it. Held 5, an OPEN
    filling 2 + 2, 2 old shares sold since: 7 held. Dropping the first fill
    gives 7 - 2 = 5 (a false ADD); both give 7 - 4 = 3 (OPEN)."""
    created = _order()["created_at"]
    rec = _decide_with_execs([_ex(501, "NVDA", "BOT", 2, created - timedelta(seconds=1)),
                              _ex(501, "NVDA", "BOT", 2, created + timedelta(minutes=3))])
    assert rec["fill_reread"] == 4.0 and rec["action_would"] == "OPEN"
    # an execution older than the margin is still left out
    rec = _decide_with_execs([_ex(501, "NVDA", "BOT", 2, created - timedelta(minutes=6))])
    assert rec["fill_reread"] == 0.0


@pytest.mark.parametrize("extra_exec", [
    _ex(900, "NVDA", "SLD", 2, NOW - timedelta(minutes=5)),     # a sale since (SF3)
    _ex(901, "NVDA", "BOT", 1, NOW - timedelta(minutes=5)),     # another BUY's fill (SF3)
])
def test_a_sale_or_another_buy_in_the_executions_gives_open(extra_exec):
    own = [_ex(501, "NVDA", "BOT", 2, NOW - timedelta(minutes=10)),
           _ex(501, "NVDA", "BOT", 2, NOW - timedelta(minutes=8))]
    assert _decide_with_execs(own, held=9.0)["action_would"] == "ADD"       # control
    rec = _decide_with_execs(own + [extra_exec], held=9.0)
    assert rec["action_would"] == "OPEN"
    assert rec["sld_execs_since"] + rec["other_bot_execs_since"] == 1


@pytest.mark.parametrize("sld, other_bot", [(1, 0), (0, 1)])
def test_execution_flags_give_open_through_the_booking(sld, other_bot):
    upserts, _, rows = _book("enforce", stored=5.0, held=9.0, sld=sld, other_bot=other_bot)
    assert len(upserts) == 1 and rows[0][3]["action_would"] == "OPEN"


def test_a_working_earlier_buy_counts_as_another_buy():
    """Review SF3: an earlier BUY still working (it fills afterwards) is
    counted, as is one filled since; and the booking then gives OPEN."""
    seen = {}

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, args=None):
            seen.setdefault("sql", []).append((sql, args))

        def fetchone(self):
            return (1,)

    conn = MagicMock()
    conn.cursor.return_value = Cur()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.count_other_buy_orders_since("NVDA", NOW, 77) == 1
    sql, args = seen["sql"][-1]
    assert "status IN ('submitted', 'unconfirmed', 'partial')" in sql
    assert "filled_at >= %s" in sql and "created_at >= %s" in sql
    assert args == ("NVDA", db.TRADING_ENV, 77, NOW, NOW)
    _, _, rows = _book("enforce", stored=5.0, held=9.0, other_buys=1)
    assert rows[0][3]["action_would"] == "OPEN"


def test_is_missing_column_matches_the_whole_name_on_the_primary_message():
    E = db.psycopg2.errors.UndefinedColumn
    assert db._is_missing_column(E('column "held_before_qty" of relation "orders" does not exist'),
                                 "held_before_qty")
    assert db._is_missing_column(E("column orders.filled_at does not exist"), "filled_at")
    assert not db._is_missing_column(E('column "filled_at_source" does not exist'), "filled_at")
    assert not db._is_missing_column(ValueError("held_before_qty"), "held_before_qty")
    e = E("wrapped text without the name")
    diag = MagicMock()
    diag.message_primary = 'column "held_before_qty" does not exist'
    with patch.object(E, "diag", diag, create=True):
        assert db._is_missing_column(e, "held_before_qty")   # read from diag, not str(e)


def test_a_failing_shadow_check_does_not_break_the_booking():
    upserts, reprotect, rows = _book("shadow", stored=RuntimeError("db down"), held=9.0)
    assert len(upserts) == 1 and len(reprotect) == 1 and rows == []


def test_the_pass_number_counts_per_order_in_one_round_trip():
    r = _Redis()
    first = _book("shadow", stored=5.0, held=9.0, redis_fake=r)[2]
    second = _book("shadow", stored=5.0, held=9.0, redis_fake=r)[2]
    assert first[0][3]["pass_no"] == 1 and second[0][3]["pass_no"] == 2
    assert r.round_trips == 2                                  # INCR + EXPIRE pipelined
    with patch("stop_lock._redis", side_effect=OSError("down")), \
         patch("db.log_activity_queued") as row:
        tasks._late_fill_held_record({"order_id": 77, "ticker": "NVDA", "mode": "enforce",
                                      "stored_held_before": 5.0, "held_now": 9.0,
                                      "action_would": "ADD", "action_today": "OPEN"})
    assert row.call_args[0][4]["pass_no"] is None


def test_held_count_queries_have_a_short_statement_timeout():
    for fn in (db.get_order_held_before, db.set_order_held_before, db.count_other_buy_orders_since):
        src = inspect.getsource(fn)
        assert src.index('SET LOCAL statement_timeout = 3000') < src.index("FROM orders" if
                                                                          "FROM orders" in src
                                                                          else "held_before_qty")


def test_enforce_on_old_shares_sold_since_books_as_today():
    """Held 5 at the order, the old 5 sold before the late fill: the record is
    a leftover, so the rewrite (today's behaviour) is right."""
    upserts, _, rows = _book("enforce", stored=5.0, held=4.0)
    assert len(upserts) == 1 and upserts[0]["stop_price"] == NEW_STOP
    assert rows[0][3]["action_would"] == "OPEN" and rows[0][3]["applied"] is False


def test_enforce_with_a_failed_position_read_books_as_today():
    upserts, _, rows = _book("enforce", stored=5.0, held_error=ConnectionError())
    assert len(upserts) == 1
    assert rows[0][3]["held_read"] == "error: ConnectionError" and rows[0][3]["action_would"] == "OPEN"


# ---- the reconcile run's end-of-run flush ----

def test_shadow_rows_wait_for_the_end_of_the_run_and_share_one_fresh_read():
    events, rows = [], []
    fresh_calls = []

    def run():
        for _ in range(2):
            with patch("tasks._reprotect_after_late_sell",
                       side_effect=lambda *a, **k: events.append(("reprotect",)) or True):
                tasks._book_late_fill(_order(), 2.0, 101.0, 100.5, 2.0, trade_logged=True,
                                      approval=_approval())
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "shadow"), \
         patch("db.get_order_held_before", side_effect=lambda i: events.append(("read",)) or 5.0), \
         patch("stop_lock._redis", return_value=_Redis()), \
         patch("db.get_position_stop", return_value={"ticker": "NVDA", "stop_price": OLD_STOP}), \
         patch("db.upsert_position_stop"), \
         patch("db.log_activity_queued", side_effect=lambda *a, **k: rows.append(a)), \
         patch("db.count_other_buy_orders_since", return_value=0), \
         patch("ibkr.cached_stock_qty", return_value=9.0), \
         patch("ibkr.order_exec_check", return_value={"own_bot": 4.0, "sld": 0, "other_bot": 0}), \
         patch("ibkr.get_fresh_stock_positions",
               side_effect=lambda: fresh_calls.append(1) or {"NVDA": 9.0}), \
         patch("execution._stop_anchored_to_fill", return_value=NEW_STOP), \
         patch("execution._record_stale_entry"), patch("tasks._apply_buy_outcome"), \
         patch("tasks._queue_reprotect"), patch("alerts._send"):
        tasks._late_fill_shadow_buffered(run)
    assert events == [("reprotect",), ("reprotect",), ("read",), ("read",)]
    assert len([r for r in rows if r[0] == "late_fill_held_before"]) == 2
    assert len(fresh_calls) == 1                              # one shared read for the run
    assert getattr(tasks._LATE_FILL_SHADOW, "fresh", None) is None


def test_a_callable_item_that_raises_does_not_stop_the_flush():
    done = []
    tasks._LATE_FILL_SHADOW.buf = None

    def run():
        tasks._late_fill_shadow_add(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        tasks._late_fill_shadow_add(lambda: done.append(1))
    tasks._late_fill_shadow_buffered(run)
    assert done == [1]


# ---- OPEN on a flat ticker, missing count, other actions ----

@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_open_on_a_flat_ticker_is_unchanged(mode):
    upserts, reprotect, rows = _book(mode, stored=0.0, held=4.0)
    assert len(upserts) == 1 and upserts[0]["stop_price"] == NEW_STOP and len(reprotect) == 1
    assert rows[0][3]["action_would"] == "OPEN" and rows[0][3]["held_read"] == "not_needed"
    assert rows[0][3]["applied"] is False


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_no_stored_count_is_todays_behaviour_with_no_row(mode):
    upserts, reprotect, rows = _book(mode, stored=None, held=9.0)
    assert len(upserts) == 1 and upserts[0]["stop_price"] == NEW_STOP and rows == []


def test_off_reads_nothing():
    events = []
    upserts, _, rows = _book("off", stored=5.0, held=9.0, events=events)
    assert len(upserts) == 1 and rows == [] and events == [("reprotect",)]


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_an_add_approval_is_not_touched(mode):
    events = []
    upserts, _, rows = _book(mode, stored=5.0, held=9.0, action="ADD", events=events)
    assert upserts == [] and rows == [] and not any(e[0] == "read" for e in events)


def test_the_decision_never_raises():
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "enforce"), \
         patch("db.get_order_held_before", side_effect=RuntimeError("db down")):
        assert tasks._late_fill_held_decision(_order(), _approval(), 2.0, 2.0) == ("OPEN", None)
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "enforce"), \
         patch("db.get_order_held_before", return_value=5.0), \
         patch("ibkr.cached_stock_qty", return_value=9.0), \
         patch("ibkr.order_exec_check", side_effect=RuntimeError("x")), \
         patch("db.count_other_buy_orders_since", side_effect=RuntimeError("queue")):
        assert tasks._late_fill_held_decision(_order(), _approval(), 2.0, 2.0) == ("OPEN", None)
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "enforce"), \
         patch("tasks._late_fill_shadow_add", side_effect=RuntimeError("buf")):
        tasks._late_fill_held_after_stop(_order(), _approval(), 2.0, 2.0, {"order_id": 77})


# ---- the write at execution ----

def _run_open_killed_in_wait(fake, held, mode="shadow"):
    """execute_approved on an OPEN, the worker killed during the fill wait."""
    place, _ = harness._broker(raise_after_submit=harness._Killed())
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=held)
    stored = []

    def set_held(order_row_id, qty):
        stored.append((order_row_id, qty, fake.rows[order_row_id]["status"]))
        return True

    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", mode),
                  patch("db.set_order_held_before", side_effect=set_held),
                  patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client),
                  patch("risk_manager.RiskManager"),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=quote),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}),
                  patch("config.USE_BRACKET_ORDERS", False), patch("time.sleep"),
                  patch("db.set_stop_broker_order_id"), patch("alerts._send"),
                  patch("alerts.send_trade_alert"),
                  patch("execution._capture_original_thesis_if_new")):
            m = st.enter_context(p)
            if getattr(p, "attribute", None) == "RiskManager":
                m.return_value.is_halted.return_value = False
        with pytest.raises(harness._Killed):
            execution.execute_approved(harness._open_approval())
    return stored


@pytest.mark.parametrize("held", [5.0, 0.0])
def test_the_count_goes_in_with_the_rows_insert(held):
    fake = harness.FakeOrders()
    stored = _run_open_killed_in_wait(fake, held)
    assert fake.rows[1]["held_before_qty"] == held and stored == []   # no separate UPDATE


@pytest.mark.parametrize("held", [5.0, 0.0])
def test_without_migration_92_the_count_is_updated_before_the_fill_wait(held):
    fake = harness.FakeOrders(held_column_missing=True)
    stored = _run_open_killed_in_wait(fake, held)
    assert stored == [(1, held, "submitted")]                # written with the row, before the wait


def test_off_stores_nothing():
    fake = harness.FakeOrders()
    assert _run_open_killed_in_wait(fake, 5.0, mode="off") == []
    assert fake.rows[1]["held_before_qty"] is None


def test_store_helpers_never_raise_and_skip_unknowns():
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "shadow"):
        with patch("db.set_order_held_before", side_effect=AssertionError("written")):
            assert execution._store_held_before(None, "NVDA", 5.0) is False
            assert execution._store_held_before(-1, "NVDA", 5.0) is False
            assert execution._store_held_before(3, "NVDA", None) is False
        with patch("db.set_order_held_before", side_effect=RuntimeError("db down")):
            assert execution._store_held_before(3, "NVDA", 5.0) is False
        with patch("db.set_order_held_before", return_value=True):
            assert execution._store_held_before(3, "NVDA", 0.0) is True     # flat is stored too
    early = {"id": None}
    calls = []
    wrapped = execution._with_held_before(lambda o, p: early.update(id=12), early, "NVDA", 5.0)
    with patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "shadow"), \
         patch("db.set_order_held_before", side_effect=lambda i, q: calls.append((i, q)) or True):
        wrapped(501, None)
    assert calls == [(12, 5.0)] and early["held_stored"] is True


def test_the_held_read_comes_before_the_callback_is_built():
    src = inspect.getsource(execution.execute_approved)
    read = src.index("held_before_order = _held_before_order(ticker)")
    assert read < src.index("on_submitted, early = _early_order_row(")
    assert src.index("_with_held_before(on_submitted") > read


def test_migration_92_is_the_column():
    assert dict(db._MIGRATIONS)[92] == "ALTER TABLE orders ADD COLUMN IF NOT EXISTS held_before_qty DECIMAL;"


def _run_open(fake, held, store_ok=(True,), result=None, early_fails=False, extra=()):
    """execute_approved on an OPEN through the fill wait. store_ok: what each
    set_order_held_before call returns in turn. Returns the event list:
    ("store", row id, row status) and ("place",)."""
    place, _ = harness._broker(result=result)
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=held)
    events, oks = [], list(store_ok)

    def set_held(order_row_id, qty):
        events.append(("store", order_row_id, fake.rows.get(order_row_id, {}).get("status")))
        return oks.pop(0) if oks else True

    def stop_place(*a, **k):
        events.append(("place",))
        return {"order_id": 777, "stop_price": 90.0}

    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch.object(config, "LATE_FILL_HELD_BEFORE_MODE", "shadow"),
                  patch("db.set_order_held_before", side_effect=set_held),
                  patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client),
                  patch("risk_manager.RiskManager"),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=quote),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}),
                  patch("config.USE_BRACKET_ORDERS", False), patch("time.sleep"),
                  patch("db.set_stop_broker_order_id"), patch("alerts._send"),
                  patch("alerts.send_trade_alert"),
                  patch("execution._capture_original_thesis_if_new"),
                  patch("execution._whole_position_stop", return_value=(90.0, None)),
                  patch("execution._write_stop_record", return_value=True),
                  patch("execution._record_add_lot_r"),
                  patch("tasks._apply_buy_outcome"),
                  patch("tasks.manage_stop_place", side_effect=stop_place)) + tuple(extra):
            m = st.enter_context(p)
            if getattr(p, "attribute", None) == "RiskManager":
                m.return_value.is_halted.return_value = False
        execution.execute_approved(harness._open_approval())
    return events


_FILLED = {"filled_qty": 4.0, "fill_price": 100.0, "status": "Filled"}


def test_a_failed_store_at_submission_is_retried_after_the_stop():
    fake = harness.FakeOrders(held_column_missing=True)
    events = _run_open(fake, 5.0, store_ok=(False, True), result=_FILLED)
    assert events[0] == ("store", 1, "submitted")           # at submission, failed
    assert events[1:] == [("place",), ("store", 1, "filled")]


def test_a_stored_count_is_not_written_again():
    fake = harness.FakeOrders(held_column_missing=True)
    events = _run_open(fake, 5.0, result=_FILLED)
    assert events == [("store", 1, "submitted"), ("place",)]
    fake = harness.FakeOrders()                              # in with the INSERT: no UPDATE at all
    assert _run_open(fake, 5.0, result=_FILLED) == [("place",)]
    assert fake.rows[1]["held_before_qty"] == 5.0


@pytest.mark.parametrize("which", ["order_row", "trade_row"])
def test_the_retry_is_stored_before_handing_the_fill_to_reconcile(which):
    """NIT3: a fill reconcile takes over still gets the count on its row."""
    fake = harness.FakeOrders(held_column_missing=True)
    target = "execution._mark_early_row_filled" if which == "order_row" else "execution._write_trade_row"
    events = _run_open(fake, 5.0, store_ok=(False, True), result=_FILLED,
                       extra=(patch(target, return_value=False),))
    assert events[0] == ("store", 1, "submitted")
    assert [e[:2] for e in events[1:]] == [("store", 1)]     # stored again, no place



def test_no_early_row_stores_on_the_late_row_after_the_stop():
    fake = harness.FakeOrders(fail_early_write=True)
    events = _run_open(fake, 5.0, result=_FILLED)
    assert [e[0] for e in events] == ["place", "store"]


def test_an_unconfirmed_order_gets_the_count_on_its_row():
    fake = harness.FakeOrders(held_column_missing=True)
    events = _run_open(fake, 5.0, store_ok=(False, True), result={"filled_qty": 0})
    assert events == [("store", 1, "submitted"), ("store", 1, "unconfirmed")]
    fake = harness.FakeOrders(fail_early_write=True)
    events = _run_open(fake, 5.0, result={"filled_qty": 0})
    assert len(events) == 1 and events[0][0] == "store"
    assert fake.rows[events[0][1]]["status"] == "unconfirmed"


# ---- NIT1: the early INSERT carries the count ----

class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.conn.sql.append((sql, args))
        if "held_before_qty" in sql and self.conn.held_missing:
            raise db.psycopg2.errors.UndefinedColumn(
                'column "held_before_qty" of relation "orders" does not exist')

    def fetchone(self):
        return (42,)


class _Conn:
    def __init__(self, held_missing):
        self.held_missing, self.sql, self.rollbacks, self.commits = held_missing, [], 0, 0

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rollbacks += 1

    def commit(self):
        self.commits += 1


@pytest.mark.parametrize("missing", [False, True])
def test_log_order_carries_the_count_and_drops_only_it_when_missing(missing):
    conn = _Conn(missing)
    written = {}
    with patch("db._writer_conn", return_value=conn), patch("db._put_conn"):
        oid = db.log_order("NVDA", "BUY", "MARKET", 4, broker_order_id=501, status="submitted",
                           submitted_at=NOW, submit_price=100.0, reconcile_note=db.AWAITING_FILL_NOTE,
                           held_before_qty=5.0, written=written)
    assert oid == 42 and conn.commits == 1
    last_sql, last_args = conn.sql[-1]
    assert "submitted_at" in last_sql and "reconcile_note" in last_sql   # nothing else dropped
    assert ("held_before_qty" in last_sql) is (not missing)
    assert written == {"held_before_qty": not missing}
    assert len(conn.sql) == (2 if missing else 1) and conn.rollbacks == (1 if missing else 0)


def test_log_order_without_the_count_is_the_old_insert():
    conn = _Conn(False)
    with patch("db._writer_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "BUY", "MARKET", 4) == 42
    assert "held_before_qty" not in conn.sql[0][0] and len(conn.sql) == 1
