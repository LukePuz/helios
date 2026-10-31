"""Deferred entries older than 16 h are re-analysed, not executed (2026-10-27,
limitations audit finding 13)."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import tasks


@pytest.fixture(autouse=True)
def _claim_wins():
    """retry_deferred_entry claims the row before executing (FAILED -> PROCESSING)."""
    with patch("db.claim_deferred_entry_for_retry", return_value=True):
        yield



def _approval(hours_old):
    return {"id": 7, "ticker": "SNOW", "action": "OPEN", "status": "FAILED",
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=hours_old)}


def test_stale_approval_is_expired_and_reanalysed():
    with patch("db.get_approval_by_id", return_value=_approval(60)), \
         patch("db.update_approval_status") as status, \
         patch("execution.execute_approved") as execute, \
         patch("tasks.deep_analyze_event.apply_async") as analyse:
        tasks.retry_deferred_entry.run(7)
    execute.assert_not_called()
    assert status.call_args.args == (7, "EXPIRED") and "60h old" in status.call_args.kwargs["reason"]
    event = analyse.call_args.kwargs["args"][0]
    assert event["ticker"] == "SNOW" and event["source"] == "stale_entry_recheck"


def test_fresh_approval_still_executes():
    with patch("db.get_approval_by_id", return_value=_approval(2)), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("db.update_approval_status") as status, \
         patch("execution.execute_approved", return_value=True) as execute, \
         patch("tasks.deep_analyze_event.apply_async") as analyse:
        tasks.retry_deferred_entry.run(7)
    execute.assert_called_once()
    analyse.assert_not_called()
    status.assert_called_once_with(7, "APPROVED", reason=None)
