"""Paper verdicts must not reach live decisions (professor M35, 2026-10-28)."""
from unittest.mock import MagicMock, patch

import db


def _sql(fn, *args):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        try:
            fn(*args)
        except Exception:
            pass
    return [c.args for c in cur.execute.call_args_list if "FROM signals" in c.args[0]]


def test_latest_signal_filters_by_env():
    (sql, params), = _sql(db.get_latest_signal_for, "NVDA")
    assert "trading_env=%s" in sql and db.TRADING_ENV in params


def test_position_context_verdicts_filter_by_env():
    calls = _sql(db.get_position_context, "NVDA")
    assert calls and all("trading_env=%s" in sql for sql, _ in calls)
