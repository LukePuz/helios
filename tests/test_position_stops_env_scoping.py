"""
Tests that every position_stops read/write is scoped to db.TRADING_ENV.

Incident 2026-08-14: position_stops had no notion of which broker account
a stop belonged to. Switching a live-connected process onto the same DB
that held 41 paper-account stop rows made reconcile_stop_orders conclude
all 41 positions had "disappeared with no matching SELL execution" (none
of them existed on the empty live account), alert on every one, and delete
all 41 records -- while the real paper positions sat completely untouched
at the broker the whole time.

Mock-only (no real Postgres) so this runs in CI, which has no DB
credentials configured (see .github/workflows/tests.yml).
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import db


@contextmanager
def _mocked_conn():
    """Patches db._get_conn/_put_conn with a MagicMock connection and
    yields the mock cursor so a test can inspect exactly what SQL/params
    were sent, the same way the real position_stops functions would call
    them (conn.cursor() as a context manager)."""
    cur = MagicMock()
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), \
         patch("db._put_conn"):
        yield cur


class TestPositionStopsEnvScoping:
    def test_upsert_writes_trading_env_and_conflict_target(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "live"):
            db.upsert_position_stop("NVDA", 100.0, 90.0, "ATR", "Accelerator", 7)

        sql, params = cur.execute.call_args[0]
        assert "trading_env" in sql
        assert "ON CONFLICT (ticker, trading_env)" in sql
        assert "live" in params

    def test_get_all_position_stops_filters_by_trading_env(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "live"):
            db.get_all_position_stops()

        sql, params = cur.execute.call_args[0]
        assert "WHERE trading_env=%s" in sql
        assert params == ("live",)

    def test_get_position_stop_filters_by_trading_env(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "paper"):
            db.get_position_stop("NVDA")

        sql, params = cur.execute.call_args[0]
        assert "trading_env=%s" in sql
        assert params == ("NVDA", "paper")

    def test_delete_position_stop_filters_by_trading_env(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "live"):
            db.delete_position_stop("NVDA")

        sql, params = cur.execute.call_args[0]
        assert "trading_env=%s" in sql
        assert params == ("NVDA", "live")

    def test_update_trailing_stop_filters_by_trading_env(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "live"):
            db.update_trailing_stop("NVDA", 95.0, 100.0)

        sql, params = cur.execute.call_args[0]
        assert "trading_env=%s" in sql
        assert params[-1] == "live"

    def test_get_open_position_symbols_filters_by_trading_env(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "paper"):
            db.get_open_position_symbols()

        sql, params = cur.execute.call_args[0]
        assert "trading_env=%s" in sql
        assert params == ("paper",)

    def test_set_stop_broker_order_id_filters_by_trading_env(self):
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "live"):
            db.set_stop_broker_order_id("NVDA", 12345)

        sql, params = cur.execute.call_args[0]
        assert "trading_env=%s" in sql
        assert params[-1] == "live"

    def test_live_and_paper_envs_use_different_scope_values(self):
        """The same call with a different db.TRADING_ENV must produce a
        different bound parameter -- this is what actually prevents one
        env's reconciliation pass from ever seeing the other env's rows."""
        with _mocked_conn() as cur, patch("db.TRADING_ENV", "paper"):
            db.get_all_position_stops()
        _, paper_params = cur.execute.call_args[0]

        with _mocked_conn() as cur, patch("db.TRADING_ENV", "live"):
            db.get_all_position_stops()
        _, live_params = cur.execute.call_args[0]

        assert paper_params != live_params
