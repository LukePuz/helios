"""orders.filled_at_source (migration 80) and trade_outcomes.exit_fill_at_source
(migration 81) missing: the fill and outcome writes, and the reads that select
the source, retry without that column instead of failing, warning once."""
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import psycopg2.errors
import pytest

import db


@pytest.fixture(autouse=True)
def _no_exit_benchmark_stamp(monkeypatch):
    """These tests count the close's own SQL; the post-commit
    exit_benchmark_at stamp is tested in test_exit_benchmark_at.py."""
    import db
    monkeypatch.setattr(db, "_stamp_exit_benchmark_at", lambda *a, **k: None)



def _undefined(column):
    return psycopg2.errors.UndefinedColumn(f'column "{column}" does not exist')


def _conn(fail_column, fetchone=(None, "SELL"), fetchall=None):
    """A connection whose cursor raises UndefinedColumn for any statement
    naming fail_column."""
    cur = MagicMock()
    calls = []

    def execute(sql, params=None):
        calls.append((sql, params))
        if fail_column and fail_column in sql:
            raise _undefined(fail_column)
    cur.execute.side_effect = execute
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = fetchall or []
    cur.rowcount = 1
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, calls


@pytest.fixture(autouse=True)
def _fresh_warnings():
    getattr(db, "_missing_column_warned", set()).clear()
    yield
    getattr(db, "_missing_column_warned", set()).clear()


def test_fill_write_without_the_source_column_keeps_every_other_field(caplog):
    conn, calls = _conn("filled_at_source")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         caplog.at_level(logging.WARNING):
        assert db.update_order_fill(44, 171.5, 50.0, "filled", expect_filled_qty=0.0,
                                    filled_at="T", filled_at_source="receipt") is True
        assert db.mark_submitted_order_fill(44, 171.5, 50.0, "filled", filled_at="T",
                                            filled_at_source="receipt") == 1
    conn.rollback.assert_called()
    retried = [c for c in calls if c[0].startswith("UPDATE") and "filled_at_source" not in c[0]]
    assert len(retried) == 2
    sql, params = retried[0]
    assert "filled_at=COALESCE(%s, filled_at)" in sql and "reconcile_note" in sql
    assert params == [171.5, 50.0, "filled", None, "T", db.AWAITING_FILL_NOTE, 44, 0.0]
    assert retried[1][0].endswith("AND status='submitted'")
    assert sum("filled_at_source is missing" in r.getMessage() for r in caplog.records) == 1


def test_other_errors_still_fail_the_fill_write():
    cur = MagicMock()
    cur.fetchone.return_value = (None, "SELL")
    cur.execute.side_effect = [None, psycopg2.errors.UndefinedColumn('column "x" does not exist')]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.update_order_fill(44, 171.5, 50.0, "filled", filled_at_source="receipt") is False


def test_outcome_close_without_the_source_column_closes_every_row(caplog):
    conn, calls = _conn("exit_fill_at_source", fetchall=[(11, 100.0, None), (10, 90.0, None)])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("yfinance.Ticker"), caplog.at_level(logging.WARNING):
        assert db.close_trade_outcome("NVDA", exit_price=110.0, exit_source="model_exit",
                                      exit_price_source="fill", exit_fill_at="T",
                                      exit_fill_at_source="receipt") is True
    conn.rollback.assert_called_once()
    conn.commit.assert_called_once()
    retried = [c for c in calls if c[0].startswith("UPDATE") and "exit_fill_at_source" not in c[0]]
    assert [p[-1] for _, p in retried] == [11, 10]
    assert all(p[-4:] == ("model_exit", "fill", "T", p[-1]) and len(p) == 11 for _, p in retried)
    assert sum("exit_fill_at_source is missing" in r.getMessage() for r in caplog.records) == 1


def test_source_reads_fall_back_without_the_column():
    now = datetime.now(timezone.utc)
    conn, _ = _conn("filled_at_source", fetchone=("filled", 10.0, 3.0, 501, now, now))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        row = db.get_order_fill_state(4)
    assert row["fill_price"] == 10.0 and row["filled_at"] == now and row["filled_at_source"] is None
    conn, calls = _conn("filled_at_source", fetchall=[
        (4, "NVDA", "SELL", 5, 602, None, None, "TRIM", None, now, 5, 10.0, None, now, "filled",
         db.TRADE_PENDING_NOTE)])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        rows = db.get_stale_trade_pending_orders(10)
    assert rows[0]["reconcile_note"] == db.TRADE_PENDING_NOTE and "filled_at_source" not in rows[0]
