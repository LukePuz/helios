"""Alert-only check for deferred entries whose scheduled retry was lost: the
row stays FAILED with a "deferred:" reason and nothing touches it again."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import tasks
from config import TRADING_ENV

UTC = timezone.utc


def _row(aid=81, reason="deferred: waiting for gap confirmation, re-check scheduled",
         updated=None, action="OPEN", ticker="AMD"):
    return {"id": aid, "ticker": ticker, "action": action, "status_reason": reason,
            "updated_at": updated or datetime(2026, 11, 4, 15, 0, tzinfo=UTC)}


@contextmanager
def _env(rows, state=None):
    """In-memory system_state; every pending_approvals writer is a mock that
    the test can assert was never called."""
    state = {} if state is None else state
    with patch("db.get_deferred_entries", return_value=rows), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.update_approval_status") as upd, \
         patch("db.expire_failed_entry") as exp, \
         patch("db.claim_deferred_entry_for_retry") as claim, \
         patch.object(tasks.retry_deferred_entry, "apply_async") as retry, \
         patch("execution.execute_approved") as execute, \
         patch("alerts._send") as send:
        yield send, state
    for writer in (upd, exp, claim, retry, execute):
        writer.assert_not_called()


def test_a_stale_deferred_row_alerts_once():
    now = datetime(2026, 11, 4, 16, 0, tzinfo=UTC)   # 60 min after a 5-min re-check deferral
    with _env([_row()]) as (send, state):
        assert tasks._check_stuck_deferred_entries(now) == [81]
        assert tasks._check_stuck_deferred_entries(now + timedelta(minutes=30)) == []
    send.assert_called_once()
    text = send.call_args.args[0]
    assert text.startswith("⚠️ DEFERRED ENTRY STUCK — AMD approval #81 (OPEN) was deferred "
                           "(waiting for gap confirmation, re-check scheduled) but its retry hadn't run by ")
    assert "Nothing was executed; check and re-approve or expire." in text
    assert "81" in state[tasks._DEFERRED_STUCK_ALERTED_KEY.format(env=TRADING_ENV)]


def test_a_fresh_deferred_row_does_not_alert():
    now = datetime(2026, 11, 4, 15, 30, tzinfo=UTC)   # 30 min: inside 5 + 40 min
    with _env([_row()]) as (send, _):
        assert tasks._check_stuck_deferred_entries(now) == []
    send.assert_not_called()


def test_a_market_closed_deferral_waits_for_the_next_open_plus_slack():
    # Tue 11-03 16:05 ET; next open Wed 11-04 09:30 ET = 14:30 UTC.
    row = _row(reason="deferred: market closed, retry scheduled for the open",
               updated=datetime(2026, 11, 3, 21, 5, tzinfo=UTC))
    with _env([row]) as (send, _):
        for now in (datetime(2026, 11, 4, 3, 0, tzinfo=UTC),     # overnight, hours after the write
                    datetime(2026, 11, 4, 14, 0, tzinfo=UTC),    # before the open
                    datetime(2026, 11, 4, 15, 5, tzinfo=UTC)):   # open + 35 min
            assert tasks._check_stuck_deferred_entries(now) == []
        send.assert_not_called()
        assert tasks._check_stuck_deferred_entries(datetime(2026, 11, 4, 15, 15, tzinfo=UTC)) == [81]
    assert "(market closed, retry scheduled for the open)" in send.call_args.args[0]


def test_a_friday_evening_deferral_is_not_flagged_over_the_weekend():
    row = _row(reason="deferred: market closed, retry scheduled for the open",
               updated=datetime(2026, 11, 6, 22, 0, tzinfo=UTC))   # Fri 17:00 ET
    with _env([row]) as (send, _):
        assert tasks._check_stuck_deferred_entries(datetime(2026, 11, 8, 20, 0, tzinfo=UTC)) == []
        assert tasks._check_stuck_deferred_entries(datetime(2026, 11, 9, 15, 15, tzinfo=UTC)) == [81]


def test_a_recorded_retry_run_moves_the_due_time():
    """The re-size re-check leaves the row FAILED without rewriting it; its
    recorded run counts as the retry having happened."""
    now = datetime(2026, 11, 4, 18, 0, tzinfo=UTC)
    ran_key = tasks._DEFERRED_RETRY_RAN_KEY.format(env=TRADING_ENV)
    state = {ran_key: '{"81": "%s"}' % (now - timedelta(minutes=10)).isoformat()}
    with _env([_row()], state) as (send, _):
        assert tasks._check_stuck_deferred_entries(now) == []
    send.assert_not_called()


def test_non_deferred_failed_rows_are_not_selected():
    import db
    cur = MagicMock()
    cur.fetchall.return_value = []
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.get_deferred_entries() == []
    sql, params = cur.execute.call_args[0]
    assert sql.lstrip().upper().startswith("SELECT")
    assert not any(w in sql.upper() for w in ("UPDATE", "INSERT", "DELETE"))
    assert "status='FAILED'" in sql and "trading_env=%s" in sql
    assert params[0] == db.TRADING_ENV and params[1] == "deferred:%" and "swept:" in params[2]


def test_the_check_never_raises():
    with patch("db.get_deferred_entries", side_effect=RuntimeError("db down")):
        assert tasks._check_stuck_deferred_entries() == []


def test_auto_expire_runs_the_check():
    with patch("db.expire_stale_approvals", return_value=0), \
         patch("db.sweep_stuck_processing", return_value=[]), \
         patch("tasks._check_stuck_deferred_entries") as check, \
         patch("alerts._send"):
        tasks.auto_expire_approvals.run()
    check.assert_called_once()


def test_a_failed_recheck_schedule_alerts():
    approval = {"id": 7, "ticker": "SNOW", "action": "OPEN", "status": "FAILED",
                "timestamp": datetime.now(UTC) - timedelta(hours=3)}
    state = {}
    with patch("db.claim_deferred_entry_for_retry", return_value=True), \
         patch("db.get_approval_by_id", return_value=approval), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=tasks._RESIZE_UNREADABLE), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.expire_failed_entry") as expire, \
         patch("db.update_approval_status") as status, \
         patch.object(tasks.retry_deferred_entry, "apply_async", side_effect=RuntimeError("broker down")), \
         patch("execution.execute_approved") as execute, \
         patch("alerts._send") as send:
        tasks.retry_deferred_entry.run(7)
    execute.assert_not_called()
    status.assert_not_called()
    expire.assert_not_called()
    send.assert_called_once()
    text = send.call_args.args[0]
    assert text.startswith("⚠️ DEFERRED ENTRY RETRY NOT SCHEDULED — SNOW approval #7 (OPEN)")
    assert "broker down" in text and "Nothing was executed" in text
    # The stuck check won't report the same approval again.
    assert "7" in state[tasks._DEFERRED_STUCK_ALERTED_KEY.format(env=TRADING_ENV)]
