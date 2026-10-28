"""A deferred OPEN/ADD is dropped at the open when a newer buy approval or a
newer non-buy verdict exists for the ticker (audit 2026-10-28 10:53 #2)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import tasks


def _approval(action="OPEN"):
    return {"id": 7, "ticker": "SNOW", "action": action, "status": "FAILED",
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=3)}


def _run(approval, newer_approval=False, latest=None):
    with patch("db.get_approval_by_id", return_value=approval), \
         patch("db.has_newer_buy_approval", return_value=newer_approval), \
         patch("db.get_latest_signal_for", return_value=latest), \
         patch("db.update_approval_status") as status, \
         patch("execution.execute_approved", return_value=True) as execute:
        tasks.retry_deferred_entry.run(7)
    return status, execute


def test_newer_buy_approval_drops_the_older_one():
    status, execute = _run(_approval(), newer_approval=True)
    execute.assert_not_called()
    assert status.call_args.args == (7, "EXPIRED") and status.call_args.kwargs["reason"].startswith("superseded: ")


def test_newer_contrary_verdict_drops_the_buy():
    latest = {"signal": "WATCH", "confidence": 5.0,
              "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)}
    status, execute = _run(_approval("ADD"), latest=latest)
    execute.assert_not_called()
    assert status.call_args.args == (7, "EXPIRED") and status.call_args.kwargs["reason"].startswith("superseded: ")


def test_older_contrary_verdict_does_not_drop_it():
    latest = {"signal": "WATCH", "confidence": 5.0,
              "timestamp": datetime.now(timezone.utc) - timedelta(hours=5)}
    status, execute = _run(_approval(), latest=latest)
    execute.assert_called_once()
    status.assert_called_once_with(7, "APPROVED")


def test_newer_buy_verdict_keeps_it():
    latest = {"signal": "OPEN", "confidence": 7.0,
              "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)}
    _, execute = _run(_approval(), latest=latest)
    execute.assert_called_once()


def test_sells_are_not_checked():
    with patch("db.has_newer_buy_approval") as newer:
        assert tasks._deferred_buy_superseded({"id": 1, "ticker": "X", "action": "EXIT"}) is None
    newer.assert_not_called()
