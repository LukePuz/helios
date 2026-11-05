"""A screened-in event leaves a SCREENED_IN row, and a screen or analysis task
that expires, is revoked or fails its last retry leaves an EVENT_LOST row.
Both are log-only and are not verdicts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import db
import tasks
from agents import crew
from helios3d.replay import harness

_EVENT = {"ticker": "nvda", "source": "news", "title": "NVIDIA signs supply deal",
          "url": "https://example.com/a", "text": "body text that must not be logged"}


def _rows(log_signal, kind):
    return [c for c in log_signal.call_args_list if c.args[1] == kind]


def test_screened_in_row():
    prov = {"model": "qwen2.5:7b", "new_fact": True, "code_sha": "abc"}
    screening = {"relevant": True, "urgency": "immediate", "reason": "deal", "new_fact": True,
                 "provenance": prov}
    with patch("vector_store.store"), patch("agents.screener.screen", return_value=screening), \
         patch("db.log_signal") as log_signal:
        crew.screen_event_only(dict(_EVENT, ticker="NVDA"))
    (row,) = _rows(log_signal, "SCREENED_IN")
    assert row.args[0] == "NVDA" and row.args[2] == 0.0 and row.args[3] == "deal"
    assert row.kwargs["model"] == "qwen2.5:7b" and row.kwargs["provenance"] == prov
    d = row.kwargs["decision"]
    assert d["urgency"] == "immediate" and d["new_fact"] is True
    assert d["title"] == _EVENT["title"] and d["url"] == _EVENT["url"] and d["source"] == "news"
    assert "text" not in d


def test_screened_out_writes_no_screened_in():
    with patch("vector_store.store"), \
         patch("agents.screener.screen", return_value={"relevant": False, "reason": "noise"}), \
         patch("db.log_signal") as log_signal:
        crew.screen_event_only(dict(_EVENT))
    assert not _rows(log_signal, "SCREENED_IN")


@pytest.mark.parametrize("task, expired, reason", [
    (tasks.screen_event, True, "expired"),
    (tasks.deep_analyze_event, True, "expired"),
    (tasks.deep_analyze_event, False, "revoked"),
])
def test_revoked_handler(task, expired, reason):
    request = SimpleNamespace(args=[dict(_EVENT), {"relevant": True}], task_name=task.name)
    with patch("db.log_signal") as log_signal:
        tasks._on_event_task_revoked(sender=task, request=request, terminated=False,
                                     signum=None, expired=expired)
    (row,) = _rows(log_signal, "EVENT_LOST")
    assert row.args[0] == "NVDA"
    d = row.kwargs["decision"]
    assert d["task"] == task.name and d["reason"] == reason
    assert d["title"] == _EVENT["title"] and d["url"] == _EVENT["url"] and d["item_id"] == _EVENT["url"]
    assert "error_class" not in d and "text" not in d
    assert row.kwargs.get("provenance") is None


def test_revoked_handler_ignores_other_tasks():
    request = SimpleNamespace(args=[], task_name="tasks.poll_news")
    with patch("db.log_signal") as log_signal:
        tasks._on_event_task_revoked(sender=tasks.poll_news, request=request, expired=True)
    log_signal.assert_not_called()


@pytest.mark.parametrize("task", [tasks.screen_event, tasks.deep_analyze_event])
@pytest.mark.parametrize("retries, reason", [(1, "max_retries"), (0, "RuntimeError")])
def test_final_failure_handler(task, retries, reason):
    task.push_request(retries=retries)
    try:
        with patch("db.log_signal") as log_signal:
            task.on_failure(RuntimeError("ollama down"), "tid", [dict(_EVENT)], {}, None)
    finally:
        task.pop_request()
    (row,) = _rows(log_signal, "EVENT_LOST")
    d = row.kwargs["decision"]
    assert d["task"] == task.name and d["reason"] == reason and d["error_class"] == "RuntimeError"
    assert row.args[0] == "NVDA"


def test_handlers_never_raise():
    with patch("db.log_signal", side_effect=RuntimeError("db down")):
        tasks._log_event_lost("tasks.screen_event", "expired", [None])
        tasks._on_event_task_revoked(sender=tasks.screen_event, request=None, expired=True)


def test_kinds_are_not_verdicts():
    for kind in ("SCREENED_IN", "EVENT_LOST"):
        assert kind in db._NON_ANALYST_SIGNALS and kind in db._RECORD_ONLY_SIGNALS
        assert kind not in harness.ROUTE_KINDS


def test_on_failure_runs_through_celery_after_the_last_retry():
    with patch("agents.crew.screen_event_only", side_effect=RuntimeError("ollama down")), \
         patch("db.log_signal") as log_signal:
        res = tasks.screen_event.apply(args=[dict(_EVENT)], retries=1)
    assert res.failed()
    (row,) = _rows(log_signal, "EVENT_LOST")             # the trace's task_failure adds none
    assert row.kwargs["decision"]["reason"] == "max_retries"


# ── hard time limit / lost worker ─────────────────────────────────────────────
# The worker's parent handles these (celery.worker.request.Request): the
# child that would call the task's on_failure is dead.

def _worker_request(task):
    from celery.contrib.testing.mocks import TaskMessage
    from celery.worker.request import Request
    msg = TaskMessage(task.name, args=[dict(_EVENT), {"relevant": True}])
    return Request(msg, app=tasks.app)


def _exc_info(exc):
    from billiard.einfo import ExceptionInfo
    try:
        raise exc
    except type(exc):
        return ExceptionInfo()


@pytest.mark.parametrize("task", [tasks.screen_event, tasks.deep_analyze_event])
@pytest.mark.parametrize("make_exc, reason", [
    (lambda: __import__("billiard.exceptions").exceptions.TimeLimitExceeded(300), "time_limit"),
    (lambda: __import__("billiard.exceptions").exceptions.WorkerLostError("SIGKILL"), "worker_lost"),
])
def test_killed_task_writes_event_lost_via_task_failure(task, make_exc, reason):
    req = _worker_request(task)
    with patch.object(task.backend, "mark_as_failure"), \
         patch.object(task, "on_failure") as task_on_failure, \
         patch("db.log_signal") as log_signal:
        if reason == "time_limit":
            req.on_timeout(soft=False, timeout=300)          # billiard calls this, then
        req.on_failure(_exc_info(make_exc()))                # sets the job's error result
    task_on_failure.assert_not_called()                      # on_failure never sees it
    (row,) = _rows(log_signal, "EVENT_LOST")
    d = row.kwargs["decision"]
    assert d["task"] == task.name and d["reason"] == reason and d["title"] == _EVENT["title"]


def test_task_failure_signal_leaves_ordinary_exceptions_to_on_failure():
    with patch("db.log_signal") as log_signal:
        tasks._on_event_task_killed(sender=tasks.screen_event, task_id="t",
                                    exception=RuntimeError("x"), args=[dict(_EVENT)])
        from billiard.exceptions import WorkerLostError
        tasks._on_event_task_killed(sender=tasks.poll_news, task_id="t",
                                    exception=WorkerLostError("x"), args=[])
    log_signal.assert_not_called()
