"""orders.submitted_at / submit_price and the reconcile_note for rows left
'submitted' (status is never rewritten by these)."""
from datetime import datetime
from unittest.mock import MagicMock, patch

import psycopg2.errors

import db
import execution
import tasks


def _conn(side_effect=None, rowcount=1, fetch=(11,)):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.execute.side_effect = side_effect
    cur.rowcount = rowcount
    cur.fetchone.return_value = fetch
    return conn, cur


def test_migrations_75_and_76_are_idempotent_and_the_newest():
    versions = dict(db._MIGRATIONS)
    assert len(versions) == len(db._MIGRATIONS)                 # no duplicate numbers
    assert "submitted_at" in versions[75] and "submit_price" in versions[75]
    assert "reconcile_note" in versions[76]
    for v in (75, 76):
        assert versions[v].count("ADD COLUMN IF NOT EXISTS") == versions[v].count("ADD COLUMN")
    assert max(versions) >= 76


def test_log_order_writes_submitted_at_and_submit_price():
    conn, cur = _conn()
    at = datetime(2026, 11, 2, 14, 31)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "BUY", "MARKET", 3, broker_order_id=7,
                            submitted_at=at, submit_price=101.25) == 11
    sql, params = cur.execute.call_args.args
    assert "submitted_at, submit_price" in sql and "COALESCE(%s, NOW())" in sql
    assert params[-2:] == (at, 101.25)


def test_log_order_falls_back_when_migration_75_is_pending():
    conn, cur = _conn(side_effect=[psycopg2.errors.UndefinedColumn("submitted_at"), None])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "BUY", "MARKET", 3, submitted_at=datetime.now()) == 11
    assert "submitted_at" not in cur.execute.call_args.args[0]


def test_reconcile_note_touches_only_its_column():
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.set_order_reconcile_note(5, "fill_write_failed") is True
    sql = cur.execute.call_args.args[0]
    assert sql.startswith("UPDATE orders SET reconcile_note=%s WHERE id=%s AND "
                          "(reconcile_note IS NULL OR reconcile_note=%s)")
    assert cur.execute.call_args.args[1] == ("fill_write_failed", 5, db.AWAITING_FILL_NOTE)
    assert db.set_order_reconcile_note(-1, "x") is False


def test_stale_submitted_rows_get_a_note_not_a_status():
    conn, cur = _conn(rowcount=4)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.mark_stale_submitted_orders(6) == 4
    sql, params = cur.execute.call_args.args
    assert "SET reconcile_note=%s WHERE status='submitted'" in sql and "SET status" not in sql
    assert "trading_env" not in sql and params == (db.STALE_SUBMITTED_NOTE, 6)


def test_stale_note_failure_returns_none():
    conn, _ = _conn(side_effect=RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.mark_stale_submitted_orders() is None


def test_review_task_runs_the_stale_note_and_survives_its_failure():
    with patch("db.get_open_outcomes", return_value=[]), \
         patch("db.mark_stale_submitted_orders", return_value=2) as mark:
        tasks.review_trade_outcomes.run()
    mark.assert_called_once()
    with patch("db.get_open_outcomes", return_value=[]), \
         patch("db.mark_stale_submitted_orders", side_effect=RuntimeError("down")):
        tasks.review_trade_outcomes.run()                        # no retry raised


def _manual(result, fill_ok=True):
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value=result) as place, \
         patch("db.log_trade"), patch("db.log_order", return_value=5) as log_order, \
         patch("db.update_order_fill", return_value=fill_ok), \
         patch("db.set_order_reconcile_note") as note, patch("db.upsert_position_stop"), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}), patch("alerts._send"):
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.return_value = (92.0, "ATR")
        execution.execute_manual("ZZZ", "BUY", 3, "test")
    return place, log_order, note


def test_manual_orders_carry_submission_time_and_quote():
    place, log_order, note = _manual({"order_id": 78, "fill_price": 100.0, "filled_qty": 3})
    assert place.call_args.args == ("ZZZ", "BUY", 3)               # order unchanged
    kw = log_order.call_args.kwargs
    assert isinstance(kw["submitted_at"], datetime) and kw["submitted_at"].tzinfo is not None
    assert kw["submit_price"] == 100.0 and kw["status"] == "submitted"
    note.assert_not_called()


def test_a_failed_fill_write_is_noted_on_the_row():
    _, _, note = _manual({"order_id": 78, "fill_price": 100.0, "filled_qty": 3}, fill_ok=False)
    note.assert_called_once_with(5, "fill_write_failed")


def test_profit_target_priceless_fill_is_noted():
    """The sale path leaves a priceless fill 'submitted'; it gets a note."""
    import inspect
    import risk_manager
    src = inspect.getsource(risk_manager)
    assert 'set_order_reconcile_note(local_id, "filled_no_price")' in src
