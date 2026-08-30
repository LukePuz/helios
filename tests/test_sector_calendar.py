"""
Tests for ingestion/sector_calendar.py (2026-08-30, user request).
"""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from catalyst_service import client, storage
from ingestion.sector_calendar import refresh_sector_calendar, _week_start


class WeekStartTests(unittest.TestCase):
    def test_monday_maps_to_itself(self):
        monday = date(2026, 9, 14)  # a real Monday
        self.assertEqual(_week_start(monday), monday)

    def test_sunday_maps_to_prior_monday(self):
        sunday = date(2026, 9, 20)
        self.assertEqual(_week_start(sunday), date(2026, 9, 14))


class SectorCalendarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = storage.DB_PATH
        self.old_service_url = client.SERVICE_URL
        client.SERVICE_URL = ""
        storage.DB_PATH = Path(self.tmp.name) / "catalyst_calendar.db"
        storage.init_db()

        self._constituents = {
            "AAPL": {"gics_sector": "Information Technology"},
            "MSFT": {"gics_sector": "Information Technology"},
            "NVDA": {"gics_sector": "Information Technology"},
            "AMD":  {"gics_sector": "Information Technology"},
            "JPM":  {"gics_sector": "Financials"},
        }

    def tearDown(self):
        storage.DB_PATH = self.old_db_path
        client.SERVICE_URL = self.old_service_url
        self.tmp.cleanup()

    def _same_week_rows(self):
        monday = _week_start(date.today() + timedelta(days=10))
        return [
            {"ticker": "AAPL", "earnings_date": monday, "confirmed": True},
            {"ticker": "MSFT", "earnings_date": monday + timedelta(days=1), "confirmed": True},
            {"ticker": "NVDA", "earnings_date": monday + timedelta(days=2), "confirmed": True},
            {"ticker": "JPM",  "earnings_date": monday + timedelta(days=1), "confirmed": True},
        ]

    def test_three_or_more_same_week_same_sector_creates_cluster(self):
        with patch("db.get_earnings_in_range", return_value=self._same_week_rows()), \
             patch("ingestion.sp500_universe.get_sp500_constituents", return_value=self._constituents):
            count = refresh_sector_calendar(min_count=3)
        self.assertEqual(count, 1)  # only Info Tech (3 names) clusters, Financials (1 name) doesn't
        rows = storage.query_events(scope="sector")
        self.assertEqual(rows[0]["ticker"], "XLK")
        self.assertIn("AAPL", rows[0]["thesis_note"])
        self.assertIn("MSFT", rows[0]["thesis_note"])
        self.assertIn("NVDA", rows[0]["thesis_note"])

    def test_below_min_count_creates_no_cluster(self):
        rows = self._same_week_rows()[:2]  # only 2 Info Tech names
        with patch("db.get_earnings_in_range", return_value=rows), \
             patch("ingestion.sp500_universe.get_sp500_constituents", return_value=self._constituents):
            count = refresh_sector_calendar(min_count=3)
        self.assertEqual(count, 0)

    def test_non_constituent_tickers_are_skipped_not_crashed_on(self):
        rows = [{"ticker": "QBTS", "earnings_date": date.today() + timedelta(days=5), "confirmed": False}]
        with patch("db.get_earnings_in_range", return_value=rows), \
             patch("ingestion.sp500_universe.get_sp500_constituents", return_value=self._constituents):
            count = refresh_sector_calendar(min_count=1)
        self.assertEqual(count, 0)

    def test_same_ticker_multiple_rows_same_week_counts_once(self):
        # Real incident (2026-08-30 live-DB seed): AVGO had two
        # earnings_calendar rows in the same week (a stale estimated date
        # alongside a firmer confirmed one -- different dates, so both
        # satisfy the table's own (ticker, earnings_date) unique key) and
        # was double-counted toward the cluster / listed twice in the note.
        monday = _week_start(date.today() + timedelta(days=10))
        rows = [
            {"ticker": "AAPL", "earnings_date": monday, "confirmed": True},
            {"ticker": "MSFT", "earnings_date": monday, "confirmed": True},
            {"ticker": "NVDA", "earnings_date": monday + timedelta(days=1), "confirmed": True},
            {"ticker": "NVDA", "earnings_date": monday + timedelta(days=2), "confirmed": False},
        ]
        with patch("db.get_earnings_in_range", return_value=rows), \
             patch("ingestion.sp500_universe.get_sp500_constituents", return_value=self._constituents):
            count = refresh_sector_calendar(min_count=3)
        self.assertEqual(count, 1)
        sector_rows = storage.query_events(scope="sector")
        self.assertEqual(sector_rows[0]["thesis_note"].count("NVDA"), 1)
        self.assertIn("3 Information Technology", sector_rows[0]["thesis_note"])

    def test_empty_earnings_range_returns_zero(self):
        with patch("db.get_earnings_in_range", return_value=[]):
            count = refresh_sector_calendar()
        self.assertEqual(count, 0)

    def test_mostly_unconfirmed_cluster_is_estimated_confidence(self):
        monday = _week_start(date.today() + timedelta(days=10))
        rows = [
            {"ticker": "AAPL", "earnings_date": monday, "confirmed": False},
            {"ticker": "MSFT", "earnings_date": monday, "confirmed": False},
            {"ticker": "NVDA", "earnings_date": monday, "confirmed": True},
        ]
        with patch("db.get_earnings_in_range", return_value=rows), \
             patch("ingestion.sp500_universe.get_sp500_constituents", return_value=self._constituents):
            refresh_sector_calendar(min_count=3)
        sector_rows = storage.query_events(scope="sector")
        self.assertEqual(sector_rows[0]["confidence"], "estimated")


if __name__ == "__main__":
    unittest.main()
