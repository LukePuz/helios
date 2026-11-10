"""close_trade_outcome locks the open rows it closes (SELECT ... FOR UPDATE)
and closes only rows still open, so a second close of the same ticker never
overwrites the first; the missing-column retry locks the rows again."""
from unittest.mock import MagicMock, patch

import psycopg2.errors

import db
import pytest


@pytest.fixture(autouse=True)
def _no_exit_benchmark_stamp(monkeypatch):
    """These tests count the close's own SQL; the post-commit
    exit_benchmark_at stamp is tested in test_exit_benchmark_at.py."""
    import db
    monkeypatch.setattr(db, "_stamp_exit_benchmark_at", lambda *a, **k: None)



class _Table:
    """trade_outcomes rows keyed by id, behind a cursor that runs the two
    statements close_trade_outcome sends. stale_select returns the rows as
    they were before any close, the race an unlocked read allows."""

    def __init__(self, rows, stale_select=False, fail_column=None):
        self.rows = {r[0]: {"entry_price": r[1], "bench": r[2], "exit_price": None,
                            "exit_date": None, "exit_source": None} for r in rows}
        self.initial = list(rows)
        self.stale_select = stale_select
        self.fail_column = fail_column
        self.calls = []

    def conn(self):
        cur = MagicMock()
        state = {"fetch": []}

        def execute(sql, params=None):
            self.calls.append((sql, params))
            if self.fail_column and self.fail_column in sql:
                raise psycopg2.errors.UndefinedColumn(f'column "{self.fail_column}" does not exist')
            if sql.startswith("SELECT"):
                state["fetch"] = list(self.initial) if self.stale_select else [
                    (i, r["entry_price"], r["bench"]) for i, r in self.rows.items()
                    if r["exit_date"] is None]
                return
            row = self.rows[params[-1]]
            if "exit_date IS NULL" in sql.split("WHERE", 1)[1] and row["exit_date"] is not None:
                cur.rowcount = 0
                return
            row.update(exit_price=params[0], exit_date=params[1], exit_source=params[7])
            cur.rowcount = 1

        cur.execute.side_effect = execute
        cur.fetchall.side_effect = lambda: state["fetch"]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        conn.commit.side_effect = lambda: self.calls.append(("COMMIT", None))
        conn.rollback.side_effect = lambda: self.calls.append(("ROLLBACK", None))
        return conn


def _close(table, price, source, **kw):
    with patch("db._get_conn", return_value=table.conn()), patch("db._put_conn"), \
         patch("yfinance.Ticker"):
        return db.close_trade_outcome("NVDA", exit_price=price, exit_date=f"D{price}",
                                      exit_source=source, **kw)


def test_the_open_rows_select_locks_them():
    table = _Table([(11, 100.0, None)])
    assert _close(table, 110.0, "model_exit") is True
    selects = [s for s, _ in table.calls if s.startswith("SELECT")]
    assert len(selects) == 2 and "FOR UPDATE" not in selects[0]   # unlocked look, then the lock
    assert selects[1].rstrip().endswith("FOR UPDATE")


def test_a_second_close_does_not_overwrite_the_first():
    table = _Table([(11, 100.0, None), (10, 90.0, None)], stale_select=True)
    assert _close(table, 110.0, "model_exit") is True
    # the second call read the rows before the first committed
    assert _close(table, 95.0, "late_exit") is False
    assert {i: (r["exit_price"], r["exit_source"]) for i, r in table.rows.items()} == {
        11: (110.0, "model_exit"), 10: (110.0, "model_exit")}
    updates = [s for s, _ in table.calls if s.startswith("UPDATE")]
    assert all(s.endswith("WHERE id=%s AND exit_date IS NULL") for s in updates)


def test_the_missing_column_retry_locks_the_rows_again():
    getattr(db, "_missing_column_warned", set()).clear()
    table = _Table([(11, 100.0, None)], fail_column="exit_fill_at_source")
    assert _close(table, 110.0, "model_exit", exit_fill_at="T", exit_fill_at_source="receipt") is True
    kinds = [s.split()[0] + (" FOR UPDATE" if "FOR UPDATE" in s else "") for s, _ in table.calls
             if s.split()[0] in ("SELECT", "UPDATE")]
    assert kinds == ["SELECT", "SELECT FOR UPDATE", "UPDATE", "SELECT FOR UPDATE", "UPDATE"]
    assert table.rows[11]["exit_price"] == 110.0
    getattr(db, "_missing_column_warned", set()).clear()


def test_no_spy_fetch_while_the_rows_are_locked():
    """A hung Yahoo call must not hold the row locks: the fetch comes before
    the FOR UPDATE, and nothing calls yfinance between it and the commit,
    on the retry path too."""
    for kw in ({}, {"fail_column": "exit_fill_at_source"}):
        getattr(db, "_missing_column_warned", set()).clear()
        table = _Table([(11, 100.0, None), (10, 90.0, None)], **kw)

        def ticker(symbol):
            table.calls.append(("YF " + symbol, None))
            return MagicMock()
        with patch("db._get_conn", return_value=table.conn()), patch("db._put_conn"), \
             patch("yfinance.Ticker", side_effect=ticker):
            assert db.close_trade_outcome("NVDA", exit_price=110.0, exit_source="model_exit",
                                          exit_fill_at="T", exit_fill_at_source="receipt") is True
        order = [s.split()[0] + (" FOR UPDATE" if "FOR UPDATE" in s else "") for s, _ in table.calls]
        assert order.count("YF") == 1
        first_lock = order.index("SELECT FOR UPDATE")
        assert order.index("YF") < first_lock
        assert "YF" not in order[first_lock:order.index("COMMIT")]
    getattr(db, "_missing_column_warned", set()).clear()


def test_a_dead_connection_still_returns_false():
    table = _Table([(11, 100.0, None)])
    conn = table.conn()
    cur = conn.cursor.return_value.__enter__.return_value
    real = cur.execute.side_effect

    def execute(sql, params=None):
        if "FOR UPDATE" in sql:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        return real(sql, params)
    cur.execute.side_effect = execute
    conn.rollback.side_effect = psycopg2.InterfaceError("connection already closed")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), patch("yfinance.Ticker"):
        assert db.close_trade_outcome("NVDA", exit_price=110.0) is False


def _racing(table, losses, error=psycopg2.errors.DeadlockDetected):
    """table's connection, with the first `losses` UPDATEs losing a lock race."""
    conn = table.conn()
    cur = conn.cursor.return_value.__enter__.return_value
    real = cur.execute.side_effect
    left = {"n": losses}

    def execute(sql, params=None):
        if sql.startswith("UPDATE") and left["n"] > 0:
            left["n"] -= 1
            table.calls.append(("LOST " + sql, params))
            raise error("deadlock detected")
        return real(sql, params)
    cur.execute.side_effect = execute
    return conn


def _close_on(conn, yf_calls):
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("yfinance.Ticker", side_effect=lambda s: yf_calls.append(s) or MagicMock()):
        return db.close_trade_outcome("NVDA", exit_price=110.0, exit_date="D", exit_source="model_exit")


def test_a_lost_lock_race_is_retried_once_and_closes_the_rows():
    for error in (psycopg2.errors.DeadlockDetected, psycopg2.errors.SerializationFailure):
        table = _Table([(11, 100.0, None), (10, 90.0, None)])
        yf_calls = []
        assert _close_on(_racing(table, 1, error), yf_calls) is True
        assert yf_calls == ["SPY"]                          # the fetched price is reused
        kinds = [s.split()[0] + (" FOR UPDATE" if "FOR UPDATE" in s else "") for s, _ in table.calls]
        assert kinds == ["SELECT", "SELECT FOR UPDATE", "LOST", "ROLLBACK",
                         "SELECT FOR UPDATE", "UPDATE", "UPDATE", "COMMIT"]
        assert all(r["exit_price"] == 110.0 for r in table.rows.values())


def test_a_second_lost_race_logs_an_error_and_returns_false(caplog):
    import logging
    table = _Table([(11, 100.0, None)])
    with caplog.at_level(logging.ERROR):
        assert _close_on(_racing(table, 2), []) is False
    assert table.rows[11]["exit_price"] is None
    assert [c[0] for c in table.calls].count("COMMIT") == 0
    assert any(r.levelno == logging.ERROR and "NVDA" in r.getMessage() for r in caplog.records)
