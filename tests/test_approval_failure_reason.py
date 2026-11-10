"""Why an execution returned False reaches status_reason (professor M-6)."""
import pytest
from unittest.mock import patch

import execution
import stop_lock
from conftest import _FakeLock


@pytest.fixture(autouse=True)
def _claim_wins():
    """retry_deferred_entry claims the row before executing (FAILED -> PROCESSING)."""
    with patch("db.claim_deferred_entry_for_retry", return_value=True):
        yield



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


def test_retry_left_to_reconcile_is_recorded_approved():
    """A fill left to reconcile filled: APPROVED with that reason, not FAILED."""
    status = _retry(lambda *a, **k: execution._outcome(
        f"{execution._LEFT_TO_RECONCILE} (trade row recovered)"))
    status.assert_called_once_with(7, "APPROVED",
                                   reason="skipped: fill left to reconcile (trade row recovered)")


def test_auto_execution_left_to_reconcile_is_recorded_approved():
    import inspect
    import agents.crew as crew
    src = inspect.getsource(crew)
    assert "status, reason = approval_outcome(success)" in src
    assert "db.update_approval_status(approval_id, status, reason=reason)" in src
    assert "_report_auto_execution_not_done(" in src


# A pre-market OPEN deferred to the open is not an execution failure
# (ANET approval #167, 2026-11-04 08:33 ET).

def _closed_market_open(schedule_raises=False):
    """execute_approved on an OPEN while the market is closed; returns
    (result, reason, sent alert texts, retry task mock)."""
    import datetime
    from unittest.mock import MagicMock
    approval = {"id": 167, "ticker": "ANET", "action": "OPEN",
                "risk_json": {"shares_initial": 5}}
    eta = datetime.datetime(2026, 11, 4, 14, 30, tzinfo=datetime.timezone.utc)
    retry = MagicMock()
    if schedule_raises:
        retry.apply_async.side_effect = RuntimeError("broker down")
    with patch("ibkr.get_ib"), patch("ibkr.get_client"), \
         patch("risk_manager.RiskManager") as rm, \
         patch("ibkr.place_market_order") as order, \
         patch("market_data.is_market_hours", return_value=False), \
         patch("market_data.next_market_open", return_value=eta), \
         patch("tasks.retry_deferred_entry", retry), \
         patch("alerts._send") as send:
        rm.return_value.is_halted.return_value = False
        result = execution.execute_approved(approval, trigger="auto")
    order.assert_not_called()
    return result, execution.last_outcome_reason(), [c.args[0] for c in send.call_args_list], retry


def _crew_notice(status, reason):
    from unittest.mock import MagicMock
    import agents.crew as crew
    feed, send = MagicMock(), MagicMock()
    crew._report_auto_execution_not_done(feed, send, "ANET", "OPEN", 167, status, reason)
    return feed, [c.args[0] for c in send.call_args_list]


def test_market_closed_deferral_is_not_reported_as_failure():
    result, reason, sent, retry = _closed_market_open()
    assert result is False
    retry.apply_async.assert_called_once()
    assert any("ORDER DEFERRED" in t for t in sent)
    assert execution.outcome_deferred(reason)
    status, why = execution.approval_outcome(False)
    assert status == "FAILED"            # retry_deferred_entry claims FAILED -> PROCESSING
    assert why == reason
    feed, crew_sent = _crew_notice(status, why)
    assert not any("AUTO-EXECUTION FAILED" in t for t in crew_sent)
    assert crew_sent == []
    (_, _, text, kind), = [c.args for c in feed.call_args_list]
    assert "deferred" in text and "FAILED" not in text and kind == "info"


def test_unscheduled_market_open_retry_still_alerts_as_failure():
    result, reason, sent, _ = _closed_market_open(schedule_raises=True)
    assert result is False
    assert reason == "failed: market closed, market-open retry not scheduled"
    assert not execution.outcome_deferred(reason)
    feed, crew_sent = _crew_notice(*execution.approval_outcome(False))
    assert len(crew_sent) == 1 and crew_sent[0].startswith("⚠️ AUTO-EXECUTION FAILED")


def test_real_failure_still_sends_auto_execution_failed():
    with patch("ibkr.get_ib", side_effect=ConnectionError("down")), patch("alerts._send"):
        assert execution.execute_approved({"ticker": "ANET", "action": "OPEN", "id": 167}) is False
    status, reason = execution.approval_outcome(False)
    assert status == "FAILED" and reason == "failed: IBKR unreachable"
    feed, crew_sent = _crew_notice(status, reason)
    assert crew_sent == [
        "⚠️ AUTO-EXECUTION FAILED\n"
        "<b>OPEN ANET</b> (approval #167) could not be executed.\n"
        "Check helios.log for details."
    ]
    feed.assert_called_once_with("execution", "ANET",
                                 "Auto-execution FAILED for OPEN ANET — check logs", "alert")


def test_left_to_reconcile_still_sends_no_failure_alert():
    feed, crew_sent = _crew_notice("APPROVED", f"{execution._LEFT_TO_RECONCILE} (trade row recovered)")
    assert crew_sent == []


def test_deferred_retry_still_claims_the_failed_row():
    import tasks
    with patch("db.get_approval_by_id", return_value=_fresh_approval()), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=None), \
         patch("db.update_approval_status") as status, \
         patch("db.claim_deferred_entry_for_retry", return_value=True) as claim, \
         patch("execution.execute_approved", return_value=True) as ex:
        tasks.retry_deferred_entry.run(7)
    claim.assert_called_once_with(7)
    ex.assert_called_once()
    status.assert_called_once_with(7, "APPROVED", reason=None)



def _run_failed_retry(reason):
    import tasks
    with patch("db.get_approval_by_id", return_value=_fresh_approval()), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=None), \
         patch("db.update_approval_status") as status, \
         patch("db.claim_deferred_entry_for_retry", return_value=True), \
         patch("execution.execute_approved", return_value=False), \
         patch("execution.last_outcome_reason", return_value=reason), \
         patch("alerts._send") as send:
        tasks.retry_deferred_entry.run(7)
    return status, send


def test_deferred_retry_that_finally_fails_alerts():
    """The deferral no longer sends AUTO-EXECUTION FAILED, so the retry's own
    real failure must, or a deferred entry that never fills goes unreported."""
    status, send = _run_failed_retry("failed: order rejected")
    status.assert_called_once_with(7, "FAILED", reason="failed: order rejected")
    send.assert_called_once()
    assert "AUTO-EXECUTION FAILED" in send.call_args[0][0]


def test_deferred_retry_deferred_again_stays_quiet():
    status, send = _run_failed_retry("deferred: gap re-check scheduled")
    status.assert_called_once_with(7, "FAILED", reason="deferred: gap re-check scheduled")
    send.assert_not_called()

def test_manual_approval_path_handles_deferral():
    import inspect
    import telegram_bot
    src = inspect.getsource(telegram_bot)
    assert "outcome_deferred(reason)" in src
    assert "Order deferred: {action} {ticker}" in src


def test_every_deferral_sends_its_own_alert():
    """outcome_deferred callers send no failure alert, so each "deferred:"
    return in execute_approved must come after its own Telegram alert, with
    no other return in between (m484: the gap-check-exception deferral had
    none). _send_wait_alert counts: it sends on a wait's first check and
    logs the later ones (audit 2026-11-07 19:07 #7)."""
    import inspect
    import re
    import execution
    lines = inspect.getsource(execution.execute_approved).splitlines()
    returns = [i for i, ln in enumerate(lines) if re.search(r'_outcome\(\s*f?"deferred:', ln)]
    assert len(returns) >= 4
    for i in returns:
        j = i - 1
        while j >= 0 and "alert_module._send(" not in lines[j] and "_send_wait_alert(" not in lines[j]:
            assert not re.search(r"\breturn\b(?!.*failed:)", lines[j]), \
                f"no deferral alert before: {lines[i].strip()}"
            j -= 1
        assert j >= 0 and i - j <= 15, f"no deferral alert before: {lines[i].strip()}"
