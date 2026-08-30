"""
Tests for ingestion/macro_calendar.py (2026-08-30, user request).
"""
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from catalyst_service import client, storage
from ingestion.macro_calendar import (
    CPI_2026, ELECTION_2026, FOMC_2026, JOBS_REPORT_2026, refresh_macro_calendar,
)


class MacroCalendarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = storage.DB_PATH
        self.old_service_url = client.SERVICE_URL
        client.SERVICE_URL = ""
        storage.DB_PATH = Path(self.tmp.name) / "catalyst_calendar.db"
        storage.init_db()

    def tearDown(self):
        storage.DB_PATH = self.old_db_path
        client.SERVICE_URL = self.old_service_url
        self.tmp.cleanup()

    def test_refresh_upserts_all_future_events(self):
        with patch("ingestion.macro_calendar.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 30)
            count = refresh_macro_calendar()
        expected_future = sum(
            1 for d, _ in FOMC_2026 if d >= "2026-08-30"
        ) + sum(
            1 for d, _, _ in CPI_2026 if d >= "2026-08-30"
        ) + sum(
            1 for d, _ in JOBS_REPORT_2026 if d >= "2026-08-30"
        ) + sum(
            1 for d, _ in ELECTION_2026 if d >= "2026-08-30"
        )
        self.assertEqual(count, expected_future)

    def test_refresh_skips_past_events(self):
        # A date well past every hardcoded entry -- nothing should upsert.
        with patch("ingestion.macro_calendar.date") as mock_date:
            mock_date.today.return_value = date(2027, 6, 1)
            count = refresh_macro_calendar()
        self.assertEqual(count, 0)

    def test_refresh_is_idempotent(self):
        with patch("ingestion.macro_calendar.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 30)
            first = refresh_macro_calendar()
            second = refresh_macro_calendar()
        self.assertEqual(first, second)
        rows = storage.query_events(scope="macro")
        self.assertEqual(len(rows), first)  # no duplicate rows created

    def test_fomc_events_land_as_confirmed_tier_1(self):
        with patch("ingestion.macro_calendar.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 30)
            refresh_macro_calendar()
        fomc_rows = storage.query_events(scope="macro", event_type="fomc_meeting")
        self.assertTrue(fomc_rows)
        for row in fomc_rows:
            self.assertEqual(row["confidence"], "confirmed")
            self.assertEqual(row["source_tier"], 1)

    def test_upcoming_events_surfaces_next_fomc_after_refresh(self):
        with patch("ingestion.macro_calendar.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 30)
            refresh_macro_calendar()
        events = client.upcoming_events(ticker=None, days_ahead=30)
        fomc = [e for e in events if e["event_type"] == "fomc_meeting"]
        self.assertEqual(fomc[0]["event_date"], "2026-09-16")


if __name__ == "__main__":
    unittest.main()
