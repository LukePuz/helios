"""Luke D5 (10-29): the brief's own-earnings line comes from the Postgres
calendar the bot acts on; the SQLite store's ticker earnings rows are dropped."""
from datetime import date, timedelta
from unittest.mock import patch

import context_engine

SQLITE = [
    {"scope": "ticker", "event_type": "earnings", "event_date": "2026-12-01", "confidence": "estimated", "ticker": "NET"},
    {"scope": "ticker", "event_type": "earnings", "event_date": "2026-12-08", "confidence": "estimated", "ticker": "NET"},
    {"scope": "macro", "event_type": "cpi_release", "event_date": "2026-11-10", "confidence": "confirmed",
     "ticker": "__MACRO__", "thesis_note": "CPI release covering Oct 2026"},
]


def test_own_earnings_from_postgres_and_duplicates_dropped():
    nxt = date.today() + timedelta(days=7)
    with patch("catalyst_service.client.upcoming_events", return_value=SQLITE), \
         patch("db.get_next_earnings_row", return_value=(nxt, True)):
        out = context_engine._catalyst_calendar_section("NET")
    assert f"{nxt} — NET earnings (confirmed)" in out
    assert "2026-12-01" not in out and "2026-12-08" not in out
    assert "CPI release" in out


def test_no_postgres_date_shows_no_own_earnings():
    with patch("catalyst_service.client.upcoming_events", return_value=SQLITE[:2]), \
         patch("db.get_next_earnings_row", return_value=None):
        assert context_engine._catalyst_calendar_section("NET") == ""
