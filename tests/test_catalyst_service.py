import sqlite3
import tempfile
import unittest
from pathlib import Path

from catalyst_service import client
from catalyst_service import storage
from catalyst_service.gates import catalyst_gate_check


class CatalystServiceTests(unittest.TestCase):
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

    def test_reconcile_single_aggregator_date_is_estimated(self):
        event = storage.reconcile(aggregator_dates=["2026-07-29"])
        self.assertEqual(event["event_date"], "2026-07-29")
        self.assertEqual(event["event_date_precision"], "estimated")
        self.assertEqual(event["confidence"], "estimated")

    def test_reconcile_disagreement_creates_conflicting_window(self):
        event = storage.reconcile(aggregator_dates=["2026-07-29", "2026-08-05"])
        self.assertEqual(event["event_date"], "2026-07-29")
        self.assertEqual(event["window_start"], "2026-07-29")
        self.assertEqual(event["window_end"], "2026-08-05")
        self.assertEqual(event["confidence"], "conflicting")

    def test_upsert_and_query_event_round_trip_json_fields(self):
        event_id = storage.upsert_event(
            {
                "ticker": "MU",
                "event_date": "2026-07-29",
                "event_date_precision": "estimated",
                "event_type": "earnings",
                "confidence": "estimated",
                "source_tier": 2,
                "sources": [{"name": "yfinance"}],
                "action_flags": {"tighten_sizing": True},
            }
        )
        rows = storage.query_events(ticker="MU")
        self.assertEqual(rows[0]["event_id"], event_id)
        self.assertEqual(rows[0]["sources"], [{"name": "yfinance"}])
        self.assertEqual(rows[0]["action_flags"], {"tighten_sizing": True})

    def test_catalyst_gate_blocks_conflicting_event(self):
        storage.upsert_event(
            {
                "ticker": "MU",
                "event_date": "2099-01-01",
                "event_date_precision": "window",
                "window_start": "2099-01-01",
                "window_end": "2099-01-08",
                "event_type": "earnings",
                "confidence": "conflicting",
                "source_tier": 2,
                "sources": [{"name": "source-a"}, {"name": "source-b"}],
                "action_flags": {"halt_auto_execution": True},
            }
        )
        result = catalyst_gate_check("MU", days_ahead=36500)
        self.assertFalse(result["approved"])
        self.assertEqual(result["blocking_events"][0]["ticker"], "MU")


class CatalystScopeTests(unittest.TestCase):
    """Sector/macro catalyst scope (2026-08-30, user request)."""

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

    def test_ticker_event_defaults_to_ticker_scope(self):
        # Pre-existing callers (upsert_earnings_from_sources et al.) never
        # pass scope -- must still work exactly as before.
        storage.upsert_event({
            "ticker": "MU", "event_date": "2026-09-20", "event_date_precision": "exact",
            "event_type": "earnings", "confidence": "confirmed", "source_tier": 1,
            "sources": [{"name": "8-K"}],
        })
        rows = storage.query_events(ticker="MU")
        self.assertEqual(rows[0]["scope"], "ticker")

    def test_upsert_macro_event_uses_market_sentinel(self):
        storage.upsert_macro_event(
            event_type="fomc_meeting", event_date="2026-09-16",
            confidence="confirmed", source_tier=1,
            sources=[{"name": "federalreserve.gov"}],
            thesis_note="FOMC meeting -- rate decision",
        )
        rows = storage.query_events(scope="macro")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], storage.MACRO_SENTINEL)
        self.assertEqual(rows[0]["event_type"], "fomc_meeting")

    def test_upsert_macro_event_rejects_ticker_level_type(self):
        with self.assertRaises(ValueError):
            storage.upsert_macro_event(
                event_type="earnings", event_date="2026-09-16",
                confidence="confirmed", source_tier=1, sources=[],
            )

    def test_upsert_sector_event_uses_etf_symbol_and_carries_gics_name(self):
        storage.upsert_sector_event(
            sector_etf="XLK", event_type="sector_earnings_cluster",
            event_date="2026-09-14", confidence="estimated", source_tier=2,
            sources=[{"name": "derived cluster"}],
            gics_sector="Information Technology",
            thesis_note="3 Info Tech names report this week",
        )
        rows = storage.query_events(scope="sector")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "XLK")
        self.assertEqual(rows[0]["bucket_relevance"], "Information Technology")

    def test_upsert_sector_event_rejects_macro_level_type(self):
        with self.assertRaises(ValueError):
            storage.upsert_sector_event(
                sector_etf="XLK", event_type="election",
                event_date="2026-09-14", confidence="confirmed",
                source_tier=1, sources=[],
            )

    def test_query_events_scope_filter_isolates_each_scope(self):
        storage.upsert_event({
            "ticker": "MU", "event_date": "2026-09-20", "event_date_precision": "exact",
            "event_type": "earnings", "confidence": "confirmed", "source_tier": 1, "sources": [],
        })
        storage.upsert_macro_event(
            event_type="election", event_date="2026-11-03", confidence="confirmed",
            source_tier=1, sources=[],
        )
        storage.upsert_sector_event(
            sector_etf="XLK", event_type="sector_earnings_cluster", event_date="2026-09-14",
            confidence="estimated", source_tier=2, sources=[],
        )
        self.assertEqual(len(storage.query_events(scope="ticker")), 1)
        self.assertEqual(len(storage.query_events(scope="sector")), 1)
        self.assertEqual(len(storage.query_events(scope="macro")), 1)
        self.assertEqual(len(storage.query_events()), 3)

    def test_upcoming_events_combines_ticker_sector_and_macro(self):
        # NVDA -> Information Technology -> XLK, per the real S&P constituent
        # map (ingestion.sp500_universe), so this exercises the real
        # resolution path, not a mock.
        storage.upsert_event({
            "ticker": "NVDA", "event_date": "2026-09-10", "event_date_precision": "exact",
            "event_type": "earnings", "confidence": "confirmed", "source_tier": 1, "sources": [],
        })
        storage.upsert_sector_event(
            sector_etf="XLK", event_type="sector_earnings_cluster", event_date="2026-09-14",
            confidence="estimated", source_tier=2, sources=[],
        )
        storage.upsert_macro_event(
            event_type="fomc_meeting", event_date="2026-09-16", confidence="confirmed",
            source_tier=1, sources=[],
        )
        events = client.upcoming_events(ticker="NVDA", days_ahead=45)
        scopes = [e["scope"] for e in events]
        self.assertEqual(set(scopes), {"ticker", "sector", "macro"})
        # sorted by date ascending
        self.assertEqual([e["event_date"] for e in events],
                          ["2026-09-10", "2026-09-14", "2026-09-16"])

    def test_upcoming_events_with_no_ticker_returns_macro_only(self):
        storage.upsert_macro_event(
            event_type="election", event_date="2026-11-03", confidence="confirmed",
            source_tier=1, sources=[],
        )
        events = client.upcoming_events(ticker=None, days_ahead=90)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["scope"], "macro")

    def test_migration_preserves_existing_rows_as_ticker_scope(self):
        # Simulates a DB written by the pre-scope schema (no `scope`
        # column) -- init_db()/_migrate_add_scope() must rebuild it in
        # place without losing the row, defaulting it to scope="ticker".
        storage.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(storage.DB_PATH)
        conn.executescript("""
            DROP TABLE IF EXISTS catalyst_events;

            CREATE TABLE catalyst_events (
                event_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, related_ticker TEXT,
                event_date TEXT NOT NULL, event_date_precision TEXT NOT NULL,
                window_start TEXT, window_end TEXT, event_type TEXT NOT NULL,
                bucket_relevance TEXT, confidence TEXT NOT NULL, source_tier INTEGER NOT NULL,
                sources TEXT NOT NULL, thesis_note TEXT, action_flags TEXT NOT NULL,
                superseded_by TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                last_verified_at TEXT
            );
            INSERT INTO catalyst_events VALUES (
                'evt-1', 'MU', NULL, '2026-09-20', 'exact', NULL, NULL, 'earnings',
                NULL, 'confirmed', 1, '[]', NULL, '{}', NULL,
                '2026-08-01T00:00:00', '2026-08-01T00:00:00', '2026-08-01T00:00:00'
            );
        """)
        conn.commit()
        conn.close()

        storage.init_db()  # triggers _migrate_add_scope

        rows = storage.query_events(ticker="MU")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_id"], "evt-1")
        self.assertEqual(rows[0]["scope"], "ticker")


if __name__ == "__main__":
    unittest.main()
