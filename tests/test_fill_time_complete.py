"""m141: ib_async marks a market order done on orderStatus Filled, but adds
an execution to trade.fills only when its execDetails arrives. place_market_order
now waits (up to config.FILL_TIME_WAIT_S) while the fills' shares are below the
filled count, and stamps whether they caught up: orders.fill_time_complete and
trade_outcomes.exit_fill_complete (migrations 93, 94)."""
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psycopg2.errors
import pytest

import db
import ibkr
import test_order_row_before_fill_wait as _fw   # its fake orders table (a module import: no re-collection)


def _t(ago_s):
    return (datetime.now(timezone.utc) - timedelta(seconds=ago_s)).replace(microsecond=0)


def _fill(t, shares=1.0, exec_id="0001.aaaa.01.01"):
    """A live Fill: execution.time plus its own receipt time, 50 ms later."""
    return SimpleNamespace(contract=SimpleNamespace(symbol="NVDA"),
                           execution=SimpleNamespace(time=t, execId=exec_id, orderId=79,
                                                     shares=shares, price=100.0, side="BOT",
                                                     permId=0),
                           time=t + timedelta(milliseconds=50))


def _place(fills, filled=2.0, on_poll=None, cap=2.0, done=True, status="Filled"):
    """place_market_order against a fake IB whose trade is already done
    (or, done=False, still working at a short fill timeout).
    on_poll(n) runs on the n-th short ib.sleep (the fill-record wait; the 1 s
    quote sleep isn't counted). Returns (result, the poll sleeps' arguments)."""
    trade = MagicMock()
    trade.isDone.return_value = done
    trade.order.orderId, trade.order.permId = 79, 1903836920
    trade.orderStatus.status, trade.orderStatus.filled = status, filled
    trade.orderStatus.avgFillPrice = 100.0
    trade.fills = list(fills)
    polls = []

    def sleep(secs):
        if secs >= 1:
            return
        polls.append(secs)
        if on_poll is not None:
            on_poll(len(polls), trade)
    fake_ib = MagicMock()
    fake_ib.placeOrder.return_value = trade
    fake_ib.sleep.side_effect = sleep
    with patch("ibkr.get_ib", return_value=fake_ib), patch("ibkr._stock", return_value=MagicMock()), \
         patch("ibkr.ibi.MarketOrder", return_value=MagicMock(orderId=79)), \
         patch("ibkr._record_order_quote"), patch("config.FILL_TIME_WAIT_S", cap):
        return ibkr.place_market_order("NVDA", "BUY", filled, fill_timeout=0.05), polls


class TestTheWait:
    def test_a_late_execution_inside_the_cap_is_waited_for(self):
        t1, t2 = _t(10), _t(5)

        def arrive(n, trade):
            if n == 3:
                trade.fills.append(_fill(t2, exec_id="0001.bbbb.01.01"))
        out, polls = _place([_fill(t1)], on_poll=arrive)
        assert out["fill_time_complete"] is True
        assert out["fill_time"] == t2 + timedelta(milliseconds=50)
        assert out["fill_time_source"] == "receipt"
        assert len(polls) == 3

    def test_still_short_at_the_cap_is_false_with_the_fills_seen(self):
        t1 = _t(10)
        out, polls = _place([_fill(t1)], cap=0.3)
        assert out["fill_time_complete"] is False
        assert out["fill_time"] == t1 + timedelta(milliseconds=50)
        assert out["fill_time_source"] == "receipt"
        assert len(polls) >= 3
        assert all(0 <= p <= ibkr._FILL_RECORDS_POLL_S for p in polls)

    def test_complete_at_once_sleeps_no_more(self):
        out, polls = _place([_fill(_t(10)), _fill(_t(9), exec_id="0001.bbbb.01.01")])
        assert out["fill_time_complete"] is True
        assert polls == []

    def test_a_raise_inside_the_wait_leaves_the_result_as_today(self, caplog):
        t1 = _t(10)

        def boom(n, trade):
            raise RuntimeError("event loop gone")
        with caplog.at_level(logging.WARNING, logger="ibkr"):
            out, _ = _place([_fill(t1)], on_poll=boom)
        assert out["fill_time_complete"] is None
        assert out["fill_price"] == 100.0
        assert out["filled_qty"] == 2.0
        assert out["status"] == "Filled"
        assert out["fill_time"] == t1 + timedelta(milliseconds=50)
        assert any("fill-record wait failed" in r.getMessage() for r in caplog.records)

    def test_zero_turns_the_wait_off(self):
        out, polls = _place([_fill(_t(10))], cap=0)
        assert out["fill_time_complete"] is None
        assert polls == []

    def test_nothing_filled_is_not_stamped(self):
        out, polls = _place([], filled=0.0)
        assert out["fill_time_complete"] is None
        assert polls == []

    def test_a_working_order_is_not_waited_for(self):
        """Part filled at the fill timeout: it can fill more during a wait,
        past the filled_qty already read, so there's no wait and no stamp."""
        out, polls = _place([_fill(_t(10))], done=False, status="Submitted")
        assert out["fill_time_complete"] is None
        assert polls == []
        assert out["filled_qty"] == 2.0

    def test_complete_with_no_readable_time_is_not_stamped(self):
        rec = SimpleNamespace(contract=SimpleNamespace(symbol="NVDA"),
                              execution=SimpleNamespace(time=None, execId="0001.aaaa.01.01",
                                                        shares=2.0),
                              time=None)
        out, polls = _place([rec])
        assert out.get("fill_time") is None
        assert out["fill_time_complete"] is None
        assert polls == []

    def test_a_corrected_execution_counts_once(self):
        t = _t(10)
        trade = SimpleNamespace(fills=[_fill(t, exec_id="0001.aaaa.01.01"),
                                       _fill(t, exec_id="0001.aaaa.01.02")])
        assert ibkr._fill_records_shares(trade) == 1.0

    def test_the_shipped_cap_is_two_seconds(self):
        """Read from config.py itself: conftest sets the cap to 0 for the suite."""
        import ast
        import config
        tree = ast.parse(open(config.__file__).read())
        values = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == "FILL_TIME_WAIT_S" for t in n.targets)]
        assert values == [2.0]


def _conn(fail_column=None):
    calls = []
    cur = MagicMock()
    cur.fetchone.return_value = (None, "BUY")
    cur.rowcount = 1

    def execute(sql, params=None):
        calls.append((sql, params))
        if fail_column and fail_column in sql and sql.startswith("UPDATE"):
            raise psycopg2.errors.UndefinedColumn(f'column "{fail_column}" does not exist')
    cur.execute.side_effect = execute
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, calls


@pytest.fixture
def _fresh_warnings():
    db._missing_column_warned.discard("fill_time_complete")
    db._missing_column_warned.discard("exit_fill_complete")
    yield
    db._missing_column_warned.discard("fill_time_complete")
    db._missing_column_warned.discard("exit_fill_complete")


class TestOrderStamp:
    def _write(self, conn, **kw):
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            return db.update_order_fill(44, 100.0, 2.0, "filled", **kw)

    def test_the_stamp_is_written_after_the_fill(self):
        conn, calls = _conn()
        assert self._write(conn, filled_at="T", fill_time_complete=False) is True
        stamp = [c for c in calls if "fill_time_complete" in c[0]]
        assert len(stamp) == 1
        assert stamp[0][1] == (False, 44)
        assert calls.index(stamp[0]) > [i for i, c in enumerate(calls) if "filled_at=" in c[0]][0]

    @pytest.mark.parametrize("kw", [{"filled_at": "T"}, {}])
    def test_a_reconcile_write_without_a_stamp_resets_it(self, kw):
        """A reconcile write (expect_filled_qty, on an unconfirmed or partial
        row) may add shares or a time the stamp didn't see, with or without
        a time of its own."""
        conn, calls = _conn()
        assert self._write(conn, expect_filled_qty=1.0, **kw) is True
        stamp = [c for c in calls if "fill_time_complete" in c[0]]
        assert len(stamp) == 1
        assert "fill_time_complete = NULL" in stamp[0][0]
        assert "fill_time_complete IS NOT NULL" in stamp[0][0]

    @pytest.mark.parametrize("kw", [{"filled_at": "T"}, {}])
    def test_a_first_write_without_a_stamp_adds_no_statement(self, kw):
        """The immediate paths' first write lands on a 'submitted' row, which
        can't carry a stamp: no extra commit before the stop."""
        conn, calls = _conn()
        assert self._write(conn, **kw) is True
        assert not [c for c in calls if "fill_time_complete" in c[0]]
        conn2, calls2 = _conn()
        with patch("db._get_conn", return_value=conn2), patch("db._put_conn"):
            assert db.mark_submitted_order_fill(44, 100.0, 2.0, "filled", **kw) == 1
        assert not [c for c in calls2 if "fill_time_complete" in c[0]]

    def test_a_write_that_updated_nothing_is_not_stamped(self):
        conn, calls = _conn()
        conn.cursor.return_value.__enter__.return_value.rowcount = 0
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            assert db.update_order_fill(44, 100.0, 2.0, "filled", expect_filled_qty=1.0,
                                        filled_at="T", fill_time_complete=True) is False
        assert not [c for c in calls if "fill_time_complete" in c[0]]

    def test_migration_pending_keeps_the_fill_and_warns_once(self, caplog, _fresh_warnings):
        with caplog.at_level(logging.WARNING):
            for _ in range(2):
                conn, calls = _conn(fail_column="fill_time_complete")
                assert self._write(conn, filled_at="T", fill_time_complete=True) is True
        assert sum("fill_time_complete is missing" in r.getMessage() for r in caplog.records) == 1
        assert not [r for r in caplog.records if "stamp for order row" in r.getMessage()]

    def test_mark_submitted_passes_the_stamp(self):
        conn, calls = _conn()
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            assert db.mark_submitted_order_fill(44, 100.0, 2.0, "filled", filled_at="T",
                                                fill_time_complete=True) == 1
        assert [c[1] for c in calls if "fill_time_complete" in c[0]] == [(True, 44)]


class TestOrderStampRead:
    def _read(self, conn):
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            return db.get_order_fill_complete(44)

    @pytest.mark.parametrize("stored, expect", [(True, True), (False, False), (None, None)])
    def test_reads_the_stamp(self, stored, expect):
        conn, calls = _conn()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (stored,)
        assert self._read(conn) is expect
        assert any("SELECT fill_time_complete FROM orders" in c[0] for c in calls)
        conn.rollback.assert_called()

    def test_no_row_is_none(self):
        conn, _ = _conn()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
        assert self._read(conn) is None

    def test_migration_pending_is_none_with_one_warning(self, caplog, _fresh_warnings):
        def missing(sql, params=None):
            if sql.startswith("SELECT fill_time_complete"):
                raise psycopg2.errors.UndefinedColumn('column "fill_time_complete" does not exist')
        with caplog.at_level(logging.WARNING):
            for _ in range(2):
                conn, _ = _conn()
                conn.cursor.return_value.__enter__.return_value.execute.side_effect = missing
                assert self._read(conn) is None
        assert sum("fill_time_complete is missing" in r.getMessage() for r in caplog.records) == 1

    def test_a_query_error_is_none_rolled_back_and_returned_to_the_pool(self, caplog):
        conn, _ = _conn()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("timeout")
        with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put, \
             caplog.at_level(logging.WARNING):
            assert db.get_order_fill_complete(44) is None
        conn.rollback.assert_called()
        put.assert_called_once_with(conn)
        assert any("get_order_fill_complete(44) failed" in r.getMessage() for r in caplog.records)

    def test_a_read_returns_the_connection(self):
        conn, _ = _conn()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (True,)
        with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
            assert db.get_order_fill_complete(44) is True
        put.assert_called_once_with(conn)

    def test_a_failed_read_is_none_and_never_raises(self):
        with patch("db._get_conn", side_effect=RuntimeError("pool exhausted")), patch("db._put_conn"):
            assert db.get_order_fill_complete(44) is None


class TestOutcomeStamp:
    def test_the_closed_rows_are_stamped(self):
        conn, calls = _conn()
        db._stamp_exit_fill_complete(conn, [11, 10], True)
        stamp = [c for c in calls if "exit_fill_complete" in c[0]]
        assert stamp == [(stamp[0][0], (True, [11, 10]))]

    def test_none_leaves_the_rows_null(self):
        conn, calls = _conn()
        db._stamp_exit_fill_complete(conn, [11], None)
        assert calls == []

    def test_migration_pending_is_skipped_with_one_warning(self, caplog, _fresh_warnings):
        with caplog.at_level(logging.WARNING):
            for _ in range(2):
                conn, _ = _conn(fail_column="exit_fill_complete")
                db._stamp_exit_fill_complete(conn, [11], False)
        assert sum("exit_fill_complete is missing" in r.getMessage() for r in caplog.records) == 1

    def test_close_trade_outcome_stamps_what_it_closed(self, monkeypatch):
        import test_close_outcome_locking as _lk
        monkeypatch.setattr(db, "_stamp_exit_benchmark_at", lambda *a, **k: None)
        seen = []
        monkeypatch.setattr(db, "_stamp_exit_fill_complete",
                            lambda conn, ids, complete: seen.append((list(ids), complete)))
        table = _lk._Table([(11, 100.0, None), (10, 90.0, None)])
        assert _lk._close(table, 110.0, "model_exit", exit_fill_at="T",
                          exit_fill_at_source="receipt", exit_fill_complete=False) is True
        assert seen == [([11, 10], False)]


class TestMigrations:
    def test_one_column_per_migration_with_no_default(self):
        m = dict(db._MIGRATIONS)
        assert m[93] == "ALTER TABLE orders ADD COLUMN IF NOT EXISTS fill_time_complete BOOLEAN;"
        assert m[94] == ("ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS "
                         "exit_fill_complete BOOLEAN;")
        assert db._ADD_COLUMN_RE.match(m[93]).group(2) == "fill_time_complete"
        assert db._ADD_COLUMN_RE.match(m[94]).group(2) == "exit_fill_complete"


def _broker(complete):
    def place(ticker, side, qty, fill_timeout=30.0, on_submitted=None):
        if on_submitted is not None:
            on_submitted(501, None)
        return {"order_id": 501, "perm_id": 9001, "ticker": ticker, "side": side, "qty": qty,
                "fill_price": 101.5, "filled_qty": qty, "status": "Filled",
                "fill_time": _t(5), "fill_time_source": "receipt", "fill_time_complete": complete}
    return place


class TestCallers:
    @pytest.mark.parametrize("complete", [True, False])
    def test_model_exit_stamps_the_order_and_the_outcome(self, complete):
        fake = _fw.FakeOrders()
        with patch("execution._remaining_after_sell", return_value=0.0):
            _fw._run_exit(fake, _broker(complete))
        assert fake.rows[1]["fill_time_complete"] is complete
        kw = fake.mocks["close_trade_outcome"].call_args.kwargs
        assert kw["exit_source"] == "model_exit"
        assert kw["exit_fill_complete"] is complete

    def test_profit_target_partial_sale_stamps_the_order(self, monkeypatch):
        import market_data
        import risk_manager
        monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)
        client = MagicMock()
        client.get_open_position.return_value = MagicMock(qty=70)
        with patch("execution._stops_still_resting", return_value=0.0), \
             patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
             patch("db.set_stop_broker_order_id"), patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.place_market_order", side_effect=_broker(False)), \
             patch("ibkr.get_client", return_value=client), \
             patch("tasks.manage_stop_place", return_value={"order_id": 8}), patch("alerts._send"), \
             patch("db.log_trade"), patch("db.log_order", return_value=11), \
             patch("db.update_order_fill") as fill:
            object.__new__(risk_manager.RiskManager)._partial_sell_and_resize_stop(
                "NVDA", 100, 30, 100.0, decision_price=126.0)
        assert fill.call_args.kwargs["fill_time_complete"] is False

    def test_manual_buy_stamps_the_order(self):
        from contextlib import ExitStack
        import execution
        fake = _fw.FakeOrders()
        with ExitStack() as st:
            for p in fake.patches():
                st.enter_context(p)
            for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                      patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                      patch("ibkr.place_market_order", side_effect=_broker(True)),
                      patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"),
                      patch("tasks.manage_stop_place", return_value={"order_id": 9}),
                      patch("alerts._send")):
                m = st.enter_context(p)
                if p.attribute == "RiskManager":
                    m.return_value.is_halted.return_value = False
                    m.return_value.calculate_stop.return_value = (92.0, "ATR")
            execution.execute_manual("ZZZ", "BUY", 3, "test")
        assert fake.rows[1]["fill_time_complete"] is True

    @pytest.mark.parametrize("stored", [True, False, None])
    def test_a_queued_flat_close_carries_the_orders_stamp(self, stored):
        """A model_exit / backstop_sell kept for a position check closes later
        through the reconcile queue (tasks._close_queued_flat_exit), from its
        order row: the row's fill_time_complete goes with its filled_at."""
        import test_overstop as _ov
        read = MagicMock(return_value=stored)
        r, close, _ = _ov._queued_flat(
            "EXIT", {"status": "filled", "fill_price": 171.5, "broker_order_id": 77,
                     "created_at": _ov._T0, "filled_at": "FILLED-AT", "filled_at_source": "receipt"},
            exit_source="model_exit", fill_complete=read, order_id=5)
        assert r is None
        read.assert_called_once_with(5)   # the order row's id, not the broker order id
        kw = close.call_args.kwargs
        assert kw["exit_source"] == "model_exit"
        assert kw["exit_fill_at"] == "FILLED-AT"
        assert kw["exit_fill_complete"] is stored

    def test_reconcile_does_not_stamp(self):
        fake = _fw.FakeOrders()
        fake.log_order("NVDA", "SELL", "MARKET", 42.0, broker_order_id=501, status="unconfirmed")
        fake.rows[1]["created_at"] = _t(60)
        real_update = fake.update_order_fill
        fake.update_order_fill = MagicMock(side_effect=real_update)
        _fw._reconcile(fake, {"filled_qty": 42.0, "fill_price": 99.8, "fill_time": _t(30),
                              "fill_time_source": "execution"})
        assert fake.update_order_fill.called
        assert fake.update_order_fill.call_args.kwargs.get("fill_time_complete") is None
