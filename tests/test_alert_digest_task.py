"""Hourly digest task for held low-severity alerts, and the weekly
candidate scan's staleness slack."""
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks


def test_digest_task_is_scheduled_hourly_while_awake():
    entry = tasks.app.conf.beat_schedule["flush-alert-digest"]
    assert entry["task"] == "tasks.flush_alert_digest"
    assert entry["options"]["expires"] <= 3600


def test_digest_task_flushes_and_swallows_errors():
    with patch("alerts.flush_digest") as flush:
        tasks.flush_alert_digest.run()
    flush.assert_called_once()
    with patch("alerts.flush_digest", side_effect=RuntimeError("x")):
        tasks.flush_alert_digest.run()   # no raise


def test_weekly_candidate_scan_is_not_stale_at_its_own_run_time():
    """Mon 09:00:47 check vs last Monday's scan: 10139 min > 10080 alerted
    on 2026-11-02 while that week's scan was running."""
    class _DT(datetime):
        fixed = None

        @classmethod
        def now(cls, tz=None):
            return cls.fixed.astimezone(tz) if tz else cls.fixed

    fixed = datetime(2026, 11, 2, 9, 0, 47, tzinfo=ZoneInfo("America/New_York"))
    _DT.fixed = fixed
    rows = [{"source": "candidate_scan", "last_success_at": fixed - timedelta(minutes=10139)}]
    with patch("db.get_ingestion_health", return_value=rows), patch("datetime.datetime", _DT):
        assert not any("candidate_scan" in s for s in tasks._get_stale_ingestion_sources())
    rows = [{"source": "candidate_scan", "last_success_at": fixed - timedelta(days=9)}]
    with patch("db.get_ingestion_health", return_value=rows), patch("datetime.datetime", _DT):
        assert any("candidate_scan" in s for s in tasks._get_stale_ingestion_sources())
