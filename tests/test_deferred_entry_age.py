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
         patch("db.expire_failed_entry", return_value=True) as expire, \
         patch("execution.execute_approved") as execute, \
         patch("tasks.deep_analyze_event.apply_async") as analyse:
        tasks.retry_deferred_entry.run(7)
    execute.assert_not_called()
    assert expire.call_args.args[0] == 7 and "60h old" in expire.call_args.args[1]
    event = analyse.call_args.kwargs["args"][0]
    assert event["ticker"] == "SNOW" and event["source"] == "stale_entry_recheck"


def test_duplicate_delivery_of_a_stale_retry_queues_one_reanalysis():
    """Several queued copies read FAILED; only the copy that expires the row re-analyses."""
    outcomes = iter([True, False, False])
    with patch("db.get_approval_by_id", return_value=_approval(60)), \
         patch("db.expire_failed_entry", side_effect=lambda *a: next(outcomes)), \
         patch("execution.execute_approved") as execute, \
         patch("tasks.deep_analyze_event.apply_async") as analyse:
        for _ in range(3):
            tasks.retry_deferred_entry.run(7)
    execute.assert_not_called()
    assert analyse.call_count == 1


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
