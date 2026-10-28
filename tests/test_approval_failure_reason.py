"""Why an execution returned False reaches status_reason (professor M-6)."""
from unittest.mock import patch

import execution
import stop_lock
from conftest import _FakeLock


def test_ibkr_unreachable_reason():
    with patch("ibkr.get_ib", side_effect=ConnectionError("down")), patch("alerts._send"):
        assert execution.execute_approved({"ticker": "AMD", "action": "OPEN", "id": 1}) is False
    assert execution.last_outcome_reason() == "failed: IBKR unreachable"


def test_busy_ticker_reason(monkeypatch):
    monkeypatch.setattr(stop_lock, "DEFAULT_WAIT_S", 0)
    _FakeLock.held.add("helios:stop_lock:AMD")
    assert execution.execute_approved({"ticker": "AMD", "action": "EXIT"}) is False
    assert execution.last_outcome_reason().startswith("skipped: another stop change")


def _fresh_approval():
    from datetime import datetime, timedelta, timezone
    return {"id": 7, "ticker": "SNOW", "action": "OPEN", "status": "FAILED",
            "risk_json": {"shares_initial": 4},
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=2)}


def _retry(execute_effect, shares_now=None):
    import tasks
    with patch("db.get_approval_by_id", return_value=_fresh_approval()), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=shares_now), \
         patch("db.update_approval_status") as status, \
         patch("execution.execute_approved", side_effect=execute_effect):
        tasks.retry_deferred_entry.run(7)
    return status


def test_retry_failure_records_why():
    status = _retry(lambda *a, **k: execution._outcome("deferred: gap re-check scheduled"))
    status.assert_called_once_with(7, "FAILED", reason="deferred: gap re-check scheduled")


def test_retry_resize_is_recorded_on_success():
    status = _retry(lambda *a, **k: True, shares_now=2)
    status.assert_called_once_with(7, "APPROVED", reason="reduced 4 -> 2 shares at execution")
