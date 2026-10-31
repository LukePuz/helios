"""Professor M142: a failed state query must raise for the halt checks,
where get_state returns the default ("false" for the halt flag)."""
from unittest.mock import MagicMock, patch

import pytest

import db


def _conn(execute_error=None, row=None):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    if execute_error:
        cur.execute.side_effect = execute_error
    cur.fetchone.return_value = row
    return conn


def test_a_query_error_raises_where_get_state_returns_the_default():
    with patch("db._get_conn", return_value=_conn(RuntimeError("statement timeout"))), \
         patch("db._put_conn"):
        assert db.get_state("trading_halted_live", "false") == "false"
    with patch("db._get_conn", return_value=_conn(RuntimeError("statement timeout"))), \
         patch("db._put_conn"):
        with pytest.raises(RuntimeError):
            db._real_get_state_strict("trading_halted_live", "false")


def test_reads_a_value_or_the_default():
    with patch("db._get_conn", return_value=_conn(row=("true",))), patch("db._put_conn"):
        assert db._real_get_state_strict("trading_halted_live", "false") == "true"
    with patch("db._get_conn", return_value=_conn(row=None)), patch("db._put_conn"):
        assert db._real_get_state_strict("trading_halted_live", "false") == "false"
