"""Warm shutdown of Celery workers (limitations audit 18:53, #7)."""
from unittest.mock import MagicMock, patch

import psutil
import pytest

import process_manager as pm


class _Proc:
    def __init__(self, pid, exits=True):
        self.pid = pid
        self.exits = exits
        self.kids = [MagicMock(name=f"child-{pid}")]
        self.terminated = self.killed = False
        self.waited = None

    def children(self, recursive=False):
        return self.kids

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = timeout
        if not self.exits:
            raise psutil.TimeoutExpired(timeout)


@pytest.fixture
def procs(monkeypatch):
    table = {}
    state = {}
    monkeypatch.setattr(pm, "_load", lambda: {n: {"pid": p.pid} for n, p in table.items()
                                              if n not in state.get("gone", set())})
    monkeypatch.setattr(pm, "_save", lambda pids: state.__setitem__(
        "gone", set(table) - set(pids)))
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(psutil, "Process", lambda pid: next(p for p in table.values() if p.pid == pid))
    return table


def test_celery_worker_gets_sigterm_on_main_only(procs):
    procs["celery_llm_worker"] = p = _Proc(1)
    assert pm.stop("celery_llm_worker") == "ok"
    assert p.terminated
    p.kids[0].terminate.assert_not_called()  # pool children finish their task
    assert 400 < p.waited <= 420


def test_non_celery_service_terminates_its_tree(procs):
    procs["catalyst_service"] = p = _Proc(2)
    pm.stop("catalyst_service")
    p.kids[0].terminate.assert_called_once()
    assert p.waited <= pm._OTHER_GRACE_S


def test_worker_still_running_after_grace_is_killed(procs):
    procs["celery_worker"] = p = _Proc(3, exits=False)
    assert pm.stop("celery_worker").startswith("killed after 60s")
    assert p.killed
    p.kids[0].kill.assert_called_once()


def test_stop_all_stops_beat_first_and_waits_in_parallel(procs):
    order = []
    procs["celery_beat"] = beat = _Proc(10)
    procs["celery_llm_worker"] = llm = _Proc(11)
    procs["celery_worker"] = ops = _Proc(12)
    beat.terminate = lambda: order.append("beat")
    llm.terminate = lambda: order.append("llm")
    ops.terminate = lambda: order.append("ops")
    with patch("time.monotonic", return_value=1000.0):
        results = pm.stop_all()
    assert order[0] == "beat" and set(order[1:]) == {"llm", "ops"}
    assert set(results.values()) == {"ok"}
    # each waited against its own deadline from the same start, not one after another
    assert llm.waited == 420 and ops.waited == 60


def test_orders_worker_stops_after_the_other_workers(procs):
    order = []
    procs["celery_orders_worker"] = o = _Proc(20)
    procs["celery_llm_worker"] = l = _Proc(21)
    o.terminate = lambda: order.append("orders")
    l.terminate = lambda: order.append("llm")
    pm.stop_all()
    assert order == ["llm", "orders"]


def test_rotation_archives_the_previous_backup(tmp_path, monkeypatch):
    """Professor M76: the old .1 used to be deleted."""
    monkeypatch.setattr(pm, "_LOG_ROTATE_MAX_BYTES", 10)
    log = tmp_path / "celery_llm_worker.log"
    (tmp_path / "celery_llm_worker.log.1").write_text("old verdicts")
    log.write_text("x" * 50)
    pm._rotate_log_if_large(log)
    assert (tmp_path / "celery_llm_worker.log.1").read_text() == "x" * 50
    [archived] = list((tmp_path / "archive").iterdir())
    assert archived.read_text() == "old verdicts" and archived.name.startswith("celery_llm_worker.")
