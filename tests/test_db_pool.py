"""Pool put-back and health check never swap the pool under a caller (audit A1031-1653 N2)."""
from unittest.mock import MagicMock

import psycopg2.pool
import pytest

import db

_real_get_pool = db._get_pool   # conftest blocks db._get_pool per test


def _fake_pool():
    pool = MagicMock()
    pool.closed = False
    return pool


def test_a_connection_goes_back_to_the_pool_that_issued_it(monkeypatch):
    first, second = _fake_pool(), _fake_pool()
    monkeypatch.setattr(db, "_get_pool", lambda: first)
    conn = db._get_conn()
    monkeypatch.setattr(db, "_pool", second)
    monkeypatch.setattr(db, "_get_pool", lambda: second)
    db._put_conn(conn)
    first.putconn.assert_called_once_with(conn)
    second.putconn.assert_not_called()


def test_a_refused_put_closes_the_connection_instead_of_raising(monkeypatch):
    pool = _fake_pool()
    pool.putconn.side_effect = psycopg2.pool.PoolError("trying to put unkeyed connection")
    monkeypatch.setattr(db, "_get_pool", lambda: pool)
    conn = db._get_conn()
    db._put_conn(conn)                      # no raise
    conn.close.assert_called_once()
    closed = _fake_pool()
    closed.closed = True
    monkeypatch.setattr(db, "_get_pool", lambda: closed)
    conn = db._get_conn()
    db._put_conn(conn)
    closed.putconn.assert_not_called()
    conn.close.assert_called_once()


def test_a_dead_connection_frees_its_slot(monkeypatch):
    """putconn's rollback raises on a dead connection before freeing the slot."""
    pool = _fake_pool()
    pool.putconn.side_effect = [psycopg2.OperationalError("server closed the connection"), None]
    monkeypatch.setattr(db, "_get_pool", lambda: pool)
    conn = db._get_conn()
    db._put_conn(conn)                      # no raise
    assert pool.putconn.call_args_list[-1].kwargs == {"close": True}


@pytest.mark.parametrize("failure", ["probe", "exhausted"])
def test_a_failed_health_check_keeps_the_pool(monkeypatch, failure):
    pool = _fake_pool()
    conn = MagicMock()
    if failure == "probe":
        pool.getconn.return_value = conn
        conn.cursor.return_value.execute.side_effect = RuntimeError("server closed the connection")
    else:
        pool.getconn.side_effect = psycopg2.pool.PoolError("connection pool exhausted")
    monkeypatch.setattr(db, "_pool", pool)
    monkeypatch.setattr(db, "_last_pool_health_check", 0.0)
    assert _real_get_pool() is pool
    pool.closeall.assert_not_called()
    if failure == "probe":
        pool.putconn.assert_called_once_with(conn, close=True)


def test_a_failed_write_on_a_dead_connection_is_still_spilled(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = psycopg2.OperationalError("gone")
    conn.rollback.side_effect = psycopg2.InterfaceError("connection already closed")
    monkeypatch.setattr(db, "_writer_conn", lambda writer, rec: conn)
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    spilled = []
    monkeypatch.setattr(db, "_record_write_failure", lambda w, e, r: spilled.append(w))
    db.log_trade("AAA", "BUY", 1, 10.0, "test")
    db.log_signal("AAA", "HOLD", 5, "test")
    assert spilled == ["log_trade", "log_signal"]
