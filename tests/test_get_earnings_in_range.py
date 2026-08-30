"""
Tests for db.get_earnings_in_range (2026-08-30, user request) -- the bulk
query ingestion/sector_calendar.py uses to detect sector earnings clusters.

Mock-only (no real Postgres), same pattern as
test_position_stops_env_scoping.py.
"""
from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, patch

import db


@contextmanager
def _mocked_conn(fetchall_return):
    cur = MagicMock()
    cur.fetchall.return_value = fetchall_return
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), \
         patch("db._put_conn"):
        yield cur


class TestGetEarningsInRange:
    def test_queries_by_date_range_and_orders_ascending(self):
        with _mocked_conn([]) as cur:
            db.get_earnings_in_range(date(2026, 9, 1), date(2026, 9, 30))

        sql, params = cur.execute.call_args[0]
        assert "earnings_date >= %s" in sql
        assert "earnings_date <= %s" in sql
        assert "ORDER BY earnings_date ASC" in sql
        assert params == (date(2026, 9, 1), date(2026, 9, 30))

    def test_returns_list_of_dicts_with_expected_keys(self):
        rows = [("NVDA", date(2026, 9, 10), True), ("AMD", date(2026, 9, 12), False)]
        with _mocked_conn(rows):
            result = db.get_earnings_in_range(date(2026, 9, 1), date(2026, 9, 30))

        assert result == [
            {"ticker": "NVDA", "earnings_date": date(2026, 9, 10), "confirmed": True},
            {"ticker": "AMD", "earnings_date": date(2026, 9, 12), "confirmed": False},
        ]

    def test_db_exception_fails_safe_to_empty_list(self):
        conn = MagicMock()
        conn.cursor.side_effect = Exception("connection lost")
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            result = db.get_earnings_in_range(date(2026, 9, 1), date(2026, 9, 30))
        assert result == []
