"""Any task other than the two event tasks that expires in the queue or is
revoked gets a task_lost activity row (audit 2026-11-07 13:07 #8). Log-only."""
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import tasks


def _join():
    for t in threading.enumerate():
        if t.name == "task-lost":
            t.join(timeout=2)


def test_an_expired_ops_task_writes_a_task_lost_row():
    eta = datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc)
    request = SimpleNamespace(id="tid-1", task_name="tasks.check_trailing_stops", eta=eta,
                              expires="2026-10-19T14:25:00+00:00", args=[])
    with patch("db.log_activity_queued") as act, patch("db.log_signal") as sig:
        tasks._on_event_task_revoked(sender=tasks.check_trailing_stops, request=request, expired=True)
        _join()
    sig.assert_not_called()
    (call,) = act.call_args_list
    assert call.args[0] == "task_lost" and call.args[1] == ""
    extra = call.args[4]
    assert extra["task"] == "tasks.check_trailing_stops" and extra["reason"] == "expired"
    assert extra["task_id"] == "tid-1" and extra["eta"] == eta.isoformat()
    assert extra["expires"] == "2026-10-19T14:25:00+00:00"


def test_revoked_reason_and_event_tasks_keep_event_lost():
    request = SimpleNamespace(id="t2", task_name="tasks.review_positions", args=[])
    with patch("db.log_activity_queued") as act:
        tasks._on_event_task_revoked(sender=None, request=request, expired=False)
        _join()
    assert act.call_args.args[4]["reason"] == "revoked"
    with patch("db.log_activity_queued") as act, patch("db.log_signal"):
        tasks._on_event_task_revoked(sender=tasks.screen_event,
                                     request=SimpleNamespace(args=[{}], task_name="tasks.screen_event"),
                                     expired=True)
        _join()
    act.assert_not_called()


def test_a_failed_row_never_raises():
    request = SimpleNamespace(id="t3", task_name="tasks.reconcile_stop_orders", args=[])
    with patch("db.log_activity_queued", side_effect=RuntimeError("db")):
        tasks._on_event_task_revoked(sender=None, request=request, expired=True)
        _join()
    tasks._on_event_task_revoked(sender=None, request=None, expired=True)   # no name: nothing
