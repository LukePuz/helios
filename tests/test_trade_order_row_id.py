"""trades.order_row_id (migration 78): a trade row names the orders row it was
booked from, so the two join by id instead of a time-window match. None
keeps the old statement and values; a missing column (migration 78 not yet
applied) keeps the trade row, written without the id."""
import inspect
import re
from datetime import timedelta
from unittest.mock import MagicMock, patch

import psycopg2.errors
import pytest

import db
import execution
import tasks
import test_order_row_before_fill_wait as _fw   # its fake orders table (a module import: no re-collection)

_OLD_SQL = ("INSERT INTO trades (symbol, trading_env, side, quantity, price, reason, price_source) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)")


def _sqls(cur):
    return [c.args[0] for c in cur.execute.call_args_list]


# --- the migration ---------------------------------------------------------

def test_migration_78_adds_the_column_once():
    versions = [v for v, _ in db._MIGRATIONS]
    assert len(versions) == len(set(versions))
    assert versions == sorted(versions) and 78 in versions
    sql = dict(db._MIGRATIONS)[78]
    assert sql == "ALTER TABLE trades ADD COLUMN IF NOT EXISTS order_row_id INTEGER;"
    assert sum("order_row_id" in s for _, s in db._MIGRATIONS) == 1


# --- log_trade ---------------------------------------------------------------

def test_log_trade_without_an_id_is_unchanged():
    conn, cur = _fw._conn()
    with patch("db._writer_conn", return_value=conn) as wc, patch("db._put_conn"):
        assert db.log_trade("NVDA", "SELL", 3, 10.0, "x", price_source="fill") is True
    assert [c.args for c in cur.execute.call_args_list] == [
        (_OLD_SQL, ("NVDA", db.TRADING_ENV, "SELL", 3, 10.0, "x", "fill"))]
    assert "order_row_id" not in wc.call_args.args[1]              # spill record unchanged
    conn.commit.assert_called_once()


def test_log_trade_writes_the_id():
    conn, cur = _fw._conn()
    with patch("db._writer_conn", return_value=conn) as wc, patch("db._put_conn"):
        assert db.log_trade("NVDA", "SELL", 3, 10.0, "x", price_source="fill",
                            order_row_id=17) is True
    ins = [c.args for c in cur.execute.call_args_list if c.args[0].startswith("INSERT")]
    assert len(ins) == 1
    assert "price_source, order_row_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)" in ins[0][0]
    assert ins[0][1] == ("NVDA", db.TRADING_ENV, "SELL", 3, 10.0, "x", "fill", 17)
    assert wc.call_args.args[1]["order_row_id"] == 17               # kept in a spill
    conn.commit.assert_called_once()


def test_log_trade_keeps_the_row_when_the_column_is_missing():
    conn, cur = _fw._conn()

    def execute(sql, *a):
        if sql.startswith("INSERT INTO trades") and "order_row_id" in sql:
            raise psycopg2.errors.UndefinedColumn("column \"order_row_id\" does not exist")
    cur.execute.side_effect = execute
    with patch("db._writer_conn", return_value=conn), patch("db._put_conn"), \
         patch("db._record_write_failure") as fail:
        assert db.log_trade("NVDA", "SELL", 3, 10.0, "x", price_source="fill",
                            order_row_id=17) is True
    calls = [c.args for c in cur.execute.call_args_list]
    assert "ROLLBACK TO SAVEPOINT trade_order_row_id" in _sqls(cur)
    assert (_OLD_SQL, ("NVDA", db.TRADING_ENV, "SELL", 3, 10.0, "x", "fill")) in calls
    conn.commit.assert_called_once()
    fail.assert_not_called()


# --- claim_and_log_trade -----------------------------------------------------

def test_claim_and_log_trade_without_an_id_is_unchanged():
    conn, cur = _fw._conn(rowcount=1)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r", "fill") is True
    assert [c.args for c in cur.execute.call_args_list][1] == (
        _OLD_SQL, ("NVDA", db.TRADING_ENV, "SELL", 3, 10.0, "r", "fill"))
    assert cur.execute.call_count == 2


def test_claim_and_log_trade_writes_the_id():
    conn, cur = _fw._conn(rowcount=1)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r", "fill",
                                      order_row_id=4) is True
    ins = [c.args for c in cur.execute.call_args_list if c.args[0].startswith("INSERT")]
    assert ins[0][1] == ("NVDA", db.TRADING_ENV, "SELL", 3, 10.0, "r", "fill", 4)
    conn.commit.assert_called_once()


def test_claim_and_log_trade_keeps_the_claim_and_row_when_the_column_is_missing():
    conn, cur = _fw._conn(rowcount=1)

    def execute(sql, *a):
        if sql.startswith("INSERT INTO trades") and "order_row_id" in sql:
            raise psycopg2.errors.UndefinedColumn("column \"order_row_id\" does not exist")
    cur.execute.side_effect = execute
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r", "fill",
                                      order_row_id=4) is True
    sqls = _sqls(cur)
    assert sqls[0].startswith("UPDATE orders SET reconcile_note")
    assert "ROLLBACK TO SAVEPOINT trade_order_row_id" in sqls and _OLD_SQL in sqls
    conn.rollback.assert_not_called()                      # the claim was not undone
    conn.commit.assert_called_once()


# --- callers -----------------------------------------------------------------

def test_execution_path_passes_the_early_rows_id_through_the_claim():
    fake = _fw.FakeOrders()
    place, _ = _fw._broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _fw._run_exit(fake, place)
    assert [t["order_row_id"] for t in fake.trades] == [1]


def test_execution_path_passes_the_id_when_the_fill_mark_failed():
    """Not trade_pending (the fill write failed), but the early row exists."""
    fake = _fw.FakeOrders()
    fake.mark_submitted_order_fill = lambda *a, **k: None
    fake.order_reconcile_note = lambda oid: _fw.AWAIT
    place, _ = _fw._broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _fw._run_exit(fake, place)
    assert [t["order_row_id"] for t in fake.trades] == [1]


def test_execution_path_without_an_early_row_leaves_it_null():
    fake = _fw.FakeOrders(fail_early_write=True)
    place, _ = _fw._broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    with patch("db.find_awaiting_order_row", return_value=None):
        _fw._run_exit(fake, place)
    assert len(fake.trades) == 1 and fake.trades[0]["order_row_id"] is None


def test_manual_path_passes_the_early_rows_id():
    fake = _fw.FakeOrders()
    _fw._run_manual_open(fake)
    assert [t["order_row_id"] for t in fake.trades] == [1]


def test_recovery_and_late_fill_pass_the_orders_row_id():
    """_recover_trade_pending_order books the part (claim), _book_late_fill
    the rest: both name the orders row."""
    fake = _fw.FakeOrders()
    with patch("tasks._queue_reprotect"):
        _fw._killed_before_trade_row(fake, {"fill_price": 101.5, "filled_qty": 20.0,
                                            "status": "Submitted"})
    _fw._age(fake)
    _fw._reconcile(fake, {"filled_qty": 42.0, "fill_price": 101.0, "fill_time": None})
    assert [(t["quantity"], t["order_row_id"]) for t in fake.trades] == [(20.0, 1), (22.0, 1)]


def test_adopted_orphan_late_fill_passes_the_orders_row_id():
    fake = _fw.FakeOrders()
    place, _ = _fw._broker(raise_after_submit=_fw._Killed())
    with pytest.raises(_fw._Killed):
        _fw._run_exit(fake, place)
    fake.rows[1]["created_at"] -= timedelta(minutes=11)
    _fw._reconcile(fake, {"filled_qty": 42.0, "fill_price": 99.8, "fill_time": None})
    assert [t["order_row_id"] for t in fake.trades] == [1]


def test_every_caller_with_an_orders_row_passes_it():
    """Source check: the callers that know the orders row id pass it (the
    backstop sell too since professor m301: its orders row now comes
    first); the ones that write the trade row before their orders row
    (profit-target partial sell) or have none (stale-stop reconciled exit)
    don't."""
    import risk_manager
    src = {f: inspect.getsource(f) for f in (
        execution._write_trade_row, tasks._book_late_fill, tasks._recover_trade_pending_order,
        tasks._execute_backstop_sell)}
    for f, s in src.items():
        for call in re.findall(r"db\.(?:claim_and_)?log_trade\((?:[^()]|\([^()]*\))*\)", s):
            assert "order_row_id=" in call, (f.__name__, call)
    for f in (tasks._reconcile_stop_orders_body,
              risk_manager.RiskManager._partial_sell_and_resize_stop):
        assert "order_row_id" not in inspect.getsource(f), f.__name__
