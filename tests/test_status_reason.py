"""pending_approvals.status_reason (migration 57, SEF ask 2026-10-28)."""
from unittest.mock import MagicMock, patch

import db


def _sql_for(**kw):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.update_approval_status(5, "EXPIRED", **kw)
    return cur.execute.call_args.args


def test_reason_is_written_when_given():
    sql, params = _sql_for(reason="superseded: x")
    assert "status_reason=%s" in sql and params == ("EXPIRED", None, "superseded: x", 5)


def test_no_reason_leaves_the_column_alone():
    sql, params = _sql_for()
    assert "status_reason" not in sql and params == ("EXPIRED", None, 5)


def test_migration_57_adds_the_column():
    assert any(v == 57 and "status_reason" in sql for v, sql in db._MIGRATIONS)
