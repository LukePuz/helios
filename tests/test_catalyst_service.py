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


if __name__ == "__main__":
    unittest.main()
