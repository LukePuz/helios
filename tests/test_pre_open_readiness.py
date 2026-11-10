"""08:45 pre-open check: code and stop coverage, Telegram only on a problem
(audit 2026-11-08 11:07 #3)."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks

OK = {"ok": True, "failed": [], "warnings": []}


def test_problems_keep_code_failures_and_coverage_only():
    v = {"failed": ["no_screens_since_0500", "dirty_tree"], "warnings": []}
    assert tasks._readiness_problems(v, {"readable": True, "ratio": {}, "uncovered": []}) == ["dirty_tree"]
    assert tasks._readiness_problems(OK, {"readable": False}) == [
        "positions or open orders unreadable from IBKR, stop coverage unknown"]
    assert tasks._readiness_problems(OK, {"readable": True, "uncovered": ["AMD", "NVDA"]}) == [
        "without a full working stop: AMD, NVDA"]


def _run(coverage, verdict=OK, day=datetime(2026, 11, 9, 8, 45)):
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return day.replace(tzinfo=ZoneInfo("America/New_York"))
    with patch("tasks._canary_verdict", return_value=verdict), \
         patch("db.get_canary_counts", return_value={}), \
         patch("agents.analyst._git_code_sha", return_value={}), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks._snapshot_positions_or_none", return_value={}), \
         patch("tasks._stop_coverage", return_value=coverage), \
         patch("db.log_activity_direct") as row, \
         patch("alerts._send") as send, \
         patch("datetime.datetime", _DT):
        tasks.pre_open_readiness()
    return row, send


def test_ready_sends_nothing_but_writes_the_row():
    row, send = _run({"readable": True, "ratio": {"NVDA": 1.0}, "uncovered": []})
    assert not send.called
    assert row.call_args[0][0] == "pre_open_readiness" and row.call_args[0][4]["problems"] == []


def test_a_naked_position_is_sent():
    row, send = _run({"readable": True, "ratio": {"AMD": 0.0}, "uncovered": ["AMD"]})
    assert "without a full working stop: AMD" in send.call_args[0][0]
    assert row.call_args[0][3] == "alert"


def test_weekend_does_nothing():
    row, send = _run({"readable": False}, day=datetime(2026, 11, 8, 8, 45))
    assert not row.called and not send.called


def test_scheduled_on_ops_before_the_open():
    e = tasks.app.conf.beat_schedule["pre-open-readiness"]
    assert e["task"] == "tasks.pre_open_readiness" and e["options"]["queue"] == "ops"
    assert "tasks.pre_open_readiness" in tasks.app.tasks


def test_db_outage_is_reported_and_still_sent():
    v = {"ok": False, "failed": ["counts_unreadable"], "warnings": []}
    assert tasks._readiness_problems(v, {"readable": True, "uncovered": []}) == ["counts_unreadable"]
    row, send = _run({"readable": True, "ratio": {}, "uncovered": []}, verdict=v)
    assert "counts_unreadable" in send.call_args[0][0]


def test_row_write_failure_does_not_stop_the_alert():
    from unittest.mock import patch as p2
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 9, 8, 45, tzinfo=ZoneInfo("America/New_York"))
    with p2("tasks._canary_verdict", return_value={"failed": ["counts_unreadable"]}), \
         p2("db.get_canary_counts", return_value=None), \
         p2("agents.analyst._git_code_sha", return_value={}), \
         p2("ibkr.get_open_orders", return_value=[]), \
         p2("tasks._snapshot_positions_or_none", return_value={}), \
         p2("tasks._stop_coverage", return_value={"readable": True, "uncovered": []}), \
         p2("db.log_activity_direct", side_effect=RuntimeError("db down")), \
         p2("alerts._send") as send, p2("datetime.datetime", _DT):
        tasks.pre_open_readiness()
    assert send.called
