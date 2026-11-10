"""When the exit SPY price was fetched, stamped on the closed outcome rows
after the close commits (audit 2026-11-07 13:07 #3). Record-only."""
import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock

import db


def test_stamp_updates_the_closed_rows_in_its_own_commit():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    at = datetime(2026, 11, 5, 14, 45, 49, tzinfo=timezone.utc)
    db._stamp_exit_benchmark_at(conn, [7, 9], at, "ok")
    sql, params = cur.execute.call_args.args
    assert "SET exit_benchmark_at=%s, exit_benchmark_status=%s WHERE id = ANY(%s)" in sql
    assert params == (at, "ok", [7, 9])
    conn.commit.assert_called_once()


def test_a_failed_fetch_still_stamps_its_status():
    """Professor m624: NULL exit_benchmark_at had three meanings."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    db._stamp_exit_benchmark_at(conn, [3], None, "fetch_failed")
    assert cur.execute.call_args.args[1] == (None, "fetch_failed", [3])


def test_nothing_to_stamp_or_a_failure_never_raises():
    conn = MagicMock()
    db._stamp_exit_benchmark_at(conn, [], datetime.now(timezone.utc))
    db._stamp_exit_benchmark_at(conn, [1], None)
    db._stamp_exit_benchmark_at(conn, [], None, "no_price")
    conn.cursor.assert_not_called()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("no column")
    db._stamp_exit_benchmark_at(conn, [1], datetime.now(timezone.utc))
    conn.rollback.assert_called()


def test_close_stamps_after_its_commit_and_the_migration_exists():
    src = inspect.getsource(db.close_trade_outcome)
    i = src.index("conn.commit()\n        _stamp_exit_benchmark_at(conn, closed_ids, exit_benchmark_at, "
                  "exit_benchmark_status)")
    assert i > src.index('"UPDATE trade_outcomes SET exit_price=%s')
    assert "exit_benchmark_at = datetime.now(_tz.utc)" in src
    assert 'exit_benchmark_status = "fetch_failed"' in src
    assert "exit_benchmark_at TIMESTAMPTZ" in dict(db._MIGRATIONS)[87]
    assert "exit_benchmark_status VARCHAR(16)" in dict(db._MIGRATIONS)[88]


def test_each_write_attempt_starts_with_no_closed_ids():
    src = inspect.getsource(db.close_trade_outcome)
    w = src.index("    def write(conn) -> bool:")
    assert src.index("closed_ids.clear()", w) < src.index("with conn.cursor() as cur:", w)


def test_a_nan_quote_is_no_price():
    src = inspect.getsource(db.close_trade_outcome)
    assert "math.isfinite(float(spy_price))" in src
