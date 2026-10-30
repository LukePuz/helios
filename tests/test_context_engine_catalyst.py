"""
Tests for context_engine._catalyst_calendar_section (2026-08-30, user
request) -- the shared helper that surfaces ticker/sector/macro catalyst
events into both assemble_brief and build_scan_brief.
"""
import unittest
from unittest.mock import patch

from context_engine import _catalyst_calendar_section


class CatalystCalendarSectionTests(unittest.TestCase):
    def test_no_events_returns_empty_string(self):
        with patch("catalyst_service.client.upcoming_events", return_value=[]):
            result = _catalyst_calendar_section("NVDA")
        self.assertEqual(result, "")

    def test_events_render_scope_tagged_lines(self):
        events = [
            {"scope": "ticker", "event_date": "2026-09-10", "event_type": "earnings",
             "confidence": "confirmed", "thesis_note": "NVDA Q2 earnings", "ticker": "NVDA"},
            {"scope": "sector", "event_date": "2026-09-14", "event_type": "sector_earnings_cluster",
             "confidence": "estimated", "thesis_note": "3 Info Tech names report", "ticker": "XLK"},
            {"scope": "macro", "event_date": "2026-09-16", "event_type": "fomc_meeting",
             "confidence": "confirmed", "thesis_note": "FOMC rate decision", "ticker": "MARKET"},
        ]
        # The ticker's own earnings line comes from Postgres (Luke D5, 10-29).
        from datetime import date, timedelta
        with patch("catalyst_service.client.upcoming_events", return_value=events), \
             patch("db.get_next_earnings_row", return_value=(date.today() + timedelta(days=5), True)):
            result = _catalyst_calendar_section("NVDA")
        self.assertIn("UPCOMING CATALYST CALENDAR", result)
        self.assertIn("[TICKER", result)
        self.assertIn("NVDA earnings", result)
        self.assertIn("[SECTOR", result)
        self.assertIn("XLK", result)
        self.assertIn("[MACRO ", result)
        self.assertIn("MARKET", result)
        self.assertIn("FOMC rate decision", result)

    def test_import_failure_fails_safe_to_empty_string(self):
        with patch("catalyst_service.client.upcoming_events", side_effect=Exception("db locked")):
            result = _catalyst_calendar_section("NVDA")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
