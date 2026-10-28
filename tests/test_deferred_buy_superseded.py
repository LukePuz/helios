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
    status.assert_called_once_with(7, "APPROVED", reason=None)


def test_newer_buy_verdict_keeps_it():
    latest = {"signal": "OPEN", "confidence": 7.0,
              "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)}
    _, execute = _run(_approval(), latest=latest)
    execute.assert_called_once()


def test_sells_are_not_checked():
    with patch("db.has_newer_buy_approval") as newer:
        assert tasks._deferred_buy_superseded({"id": 1, "ticker": "X", "action": "EXIT"}) is None
    newer.assert_not_called()


def _sized_approval(shares=5):
    a = _approval()
    a["risk_json"] = {"shares_initial": shares, "price_at_sizing": 100.0}
    a["thesis_bucket"], a["conviction"] = "Accelerator", 7
    return a


def _run_resized(approval, shares_now):
    with patch("db.get_approval_by_id", return_value=approval), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=shares_now), \
         patch("db.update_approval_status") as status, \
         patch("execution.execute_approved", return_value=True) as execute:
        tasks.retry_deferred_entry.run(7)
    return status, execute


def test_buy_that_sizes_to_zero_now_is_dropped():
    status, execute = _run_resized(_sized_approval(), 0)
    execute.assert_not_called()
    assert status.call_args.kwargs["reason"].startswith("re-check at execution")


def test_buy_is_reduced_to_what_fits_now():
    _, execute = _run_resized(_sized_approval(5), 2)
    assert execute.call_args.args[0]["risk_json"]["shares_initial"] == 2


def test_buy_is_never_increased():
    _, execute = _run_resized(_sized_approval(5), 9)
    assert execute.call_args.args[0]["risk_json"]["shares_initial"] == 5


def test_recheck_failure_keeps_the_approved_size():
    with patch("ibkr.get_client", side_effect=RuntimeError("no broker")):
        assert tasks._deferred_buy_shares_now(_sized_approval()) is None


def test_recheck_uses_sizing_and_caps():
    from unittest.mock import MagicMock
    rm = MagicMock()
    rm.size_position.return_value = {"shares_initial": 4}
    rm.shares_after_concentration.return_value = 3
    with patch("ibkr.get_client"), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("risk_manager.RiskManager", return_value=rm):
        assert tasks._deferred_buy_shares_now(_sized_approval()) == 3
    assert rm.size_position.call_args.args == ("SNOW", 7, "Accelerator", "Neutral", 100.0)
