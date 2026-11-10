"""
Tests for deploy/watchdog.py's snapshot-staleness diagnosis — regression
coverage for the 2026-08-03 incident: TWS hung (application-level -- no
crash, no systemd restart, port 7497 stopped accepting connections) for
9+ hours. The staleness message used to blame "the ops worker" unconditionally,
which sent whoever was triaging looking in the wrong place for 39 separate
alerts that day. A reachability probe against TWS's own port now tells a
hung/down broker apart from a genuinely stuck ops worker.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from deploy import watchdog


@pytest.fixture(autouse=True)
def _unsent_ping_count_in_tmp(tmp_path, monkeypatch):
    """The unsent-ping count file defaults to the repo's logs/; keep tests off it."""
    monkeypatch.setattr(watchdog, "_ORDERS_PING_UNSENT_STATE_FILE", tmp_path / "unsent")


def _fake_conn(snapshot_age_minutes):
    ts = datetime.now(timezone.utc) - timedelta(minutes=snapshot_age_minutes)
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (ts,)
    return conn


class TestSnapshotFreshnessDiagnosis:
    def test_fresh_snapshot_reports_no_problem(self):
        with patch("db._get_conn", return_value=_fake_conn(2)), \
             patch("db._put_conn"):
            assert watchdog._check_snapshot_freshness() == []

    def test_stale_snapshot_with_ibkr_unreachable_blames_ibkr(self):
        with patch("db._get_conn", return_value=_fake_conn(90)), \
             patch("db._put_conn"), \
             patch.object(watchdog, "_ibkr_port_reachable", return_value=False):
            problems = watchdog._check_snapshot_freshness()

        assert len(problems) == 1
        assert "IBKR/TWS is NOT reachable" in problems[0]
        assert "ops worker" not in problems[0].split("—")[1].split("(")[0]

    def test_stale_snapshot_with_ibkr_reachable_blames_ops_worker(self):
        """If TWS's port IS accepting connections but snapshots still
        aren't landing, the ops worker itself is the more likely culprit --
        must not blame IBKR in this case."""
        with patch("db._get_conn", return_value=_fake_conn(90)), \
             patch("db._put_conn"), \
             patch.object(watchdog, "_ibkr_port_reachable", return_value=True):
            problems = watchdog._check_snapshot_freshness()

        assert len(problems) == 1
        assert "ops worker itself may be stuck" in problems[0]
        assert "IBKR/TWS is NOT reachable" not in problems[0]


class TestIbkrPortReachable:
    def test_returns_true_when_something_is_listening(self):
        with patch("socket.create_connection") as mock_connect:
            mock_connect.return_value.__enter__.return_value = MagicMock()
            assert watchdog._ibkr_port_reachable() is True

    def test_returns_false_on_connection_refused(self):
        import socket as socket_module
        with patch("socket.create_connection", side_effect=socket_module.error("refused")):
            assert watchdog._ibkr_port_reachable() is False

    def test_returns_false_on_timeout(self):
        with patch("socket.create_connection", side_effect=TimeoutError()):
            assert watchdog._ibkr_port_reachable() is False


class TestPlannedRestart:
    """2026-10-28: the 16:05 deploy restart was caught mid-stop; the watchdog
    alerted Luke and self-healed (a second restart). Transitional states and a
    fresh planned-restart marker now skip the run."""

    def _main(self, state, marker_fresh=False, tmp_path=None, monkeypatch=None):
        from unittest.mock import patch
        import time as _t
        marker = tmp_path / ".planned_restart"
        if marker_fresh:
            marker.write_text("")
        monkeypatch.setattr(watchdog, "_PLANNED_RESTART_MARKER", marker)
        monkeypatch.setattr(watchdog.sys, "argv", ["watchdog.py", "--heal"])
        with patch.object(watchdog, "_service_state", return_value=state), \
             patch.object(watchdog, "_check_processes", return_value=[]) as procs, \
             patch.object(watchdog, "_check_snapshot_freshness", return_value=[]), \
             patch.object(watchdog, "_ping_orders_worker", return_value=None), \
             patch.object(watchdog, "_orders_active_tasks", return_value=None), \
             patch.object(watchdog, "_attempt_heal", return_value=[]) as heal, \
             patch.object(watchdog, "_emit") as emit:
            watchdog.main()
        return procs, heal, emit

    def test_deactivating_is_skipped(self, tmp_path, monkeypatch):
        procs, heal, emit = self._main("deactivating", tmp_path=tmp_path, monkeypatch=monkeypatch)
        heal.assert_not_called(); emit.assert_not_called()

    def test_fresh_marker_skips_even_when_inactive(self, tmp_path, monkeypatch):
        _, heal, emit = self._main("inactive", True, tmp_path, monkeypatch)
        heal.assert_not_called(); emit.assert_not_called()

    def test_inactive_without_marker_still_heals(self, tmp_path, monkeypatch):
        _, heal, emit = self._main("inactive", False, tmp_path, monkeypatch)
        heal.assert_called_once(); emit.assert_called_once()

    def test_stale_marker_is_ignored(self, tmp_path, monkeypatch):
        import os, time as _t
        marker = tmp_path / ".planned_restart"
        marker.write_text("")
        old = _t.time() - 3600
        os.utime(marker, (old, old))
        monkeypatch.setattr(watchdog, "_PLANNED_RESTART_MARKER", marker)
        assert watchdog._planned_restart_in_progress() is False


class TestCriticalWorkers:
    """The orders worker is the only process that can move a stop; a dead
    one went unnoticed because only ops, LLM and beat were checked."""

    def _pids(self, dead):
        import process_manager as pm
        return {n: {"pid": 1, "dead": n in dead} for n in pm.SERVICES}

    def _run(self, dead):
        import process_manager as pm
        with patch.object(pm, "_load", return_value=self._pids(dead)), \
             patch.object(pm, "_alive", side_effect=lambda e: not e["dead"]):
            return watchdog._check_processes()

    def test_dead_orders_worker_is_reported(self):
        problems = self._run({"celery_orders_worker"})
        assert problems == ["Celery Worker (orders) is not running"]

    def test_dead_screen_and_ingestion_workers_are_reported(self):
        problems = self._run({"celery_screen_worker", "celery_ingestion_worker"})
        assert sorted(problems) == ["Celery Worker (ingestion) is not running",
                                    "Celery Worker (screen) is not running"]

    def test_missing_entry_is_reported(self):
        import process_manager as pm
        with patch.object(pm, "_load", return_value={}):
            problems = watchdog._check_processes()
        assert "Celery Worker (orders) is not running" in problems

    def _main(self, dead, monkeypatch, tmp_path):
        import process_manager as pm
        monkeypatch.setattr(watchdog, "_PLANNED_RESTART_MARKER", tmp_path / "none")
        monkeypatch.setattr(watchdog.sys, "argv", ["watchdog.py", "--heal"])
        with patch.object(pm, "_load", return_value=self._pids(dead)), \
             patch.object(pm, "_alive", side_effect=lambda e: not e["dead"]), \
             patch.object(watchdog, "_service_state", return_value="active"), \
             patch.object(watchdog, "_check_snapshot_freshness", return_value=[]), \
             patch.object(watchdog, "_ping_orders_worker", return_value=None), \
             patch.object(watchdog, "_orders_active_tasks", return_value=None), \
             patch.object(watchdog, "_attempt_heal", return_value=[]) as heal, \
             patch.object(watchdog, "_emit") as emit:
            watchdog.main()
        return heal, emit

    def test_dead_orders_worker_alerts_without_restarting(self, monkeypatch, tmp_path):
        heal, emit = self._main({"celery_orders_worker"}, monkeypatch, tmp_path)
        heal.assert_not_called()
        assert "Celery Worker (orders) is not running" in emit.call_args[0][0]

    def test_ops_llm_beat_dead_still_heals_with_others_up(self, monkeypatch, tmp_path):
        heal, _ = self._main({"celery_worker", "celery_llm_worker", "celery_beat"},
                             monkeypatch, tmp_path)
        heal.assert_called_once()



class TestOrdersWorkerPing:
    """m275: a live PID can be a hung orders worker; it must answer a ping."""

    def _check(self, tmp_path, monkeypatch, answers, proc_problems=()):
        monkeypatch.setattr(watchdog, "_ORDERS_PING_STATE_FILE", tmp_path / "misses")
        out = []
        for a in answers:
            with patch.object(watchdog, "_ping_orders_worker", return_value=a):
                out.append(watchdog._check_orders_worker_responds(list(proc_problems)))
        return out

    def test_two_missed_pings_in_a_row_alert(self, tmp_path, monkeypatch):
        out = self._check(tmp_path, monkeypatch, [False, False, False])
        assert out[0] == []
        assert "has not answered a Celery ping on 2 checks" in out[1][0]
        assert out[1][0].startswith("Celery Worker (orders)") and out[2]

    def test_an_answer_resets_the_count(self, tmp_path, monkeypatch):
        assert self._check(tmp_path, monkeypatch, [False, True, False]) == [[], [], []]

    def test_an_unsendable_ping_counts_nothing_toward_misses(self, tmp_path, monkeypatch):
        assert self._check(tmp_path, monkeypatch, [False, None]) == [[], []]
        assert (tmp_path / "misses").read_text() == "1"

    def test_two_unsendable_pings_in_a_row_say_the_broker_is_down(self, tmp_path, monkeypatch):
        """Audit 2026-11-07 05:07 #5: a Redis outage left no problem line."""
        out = self._check(tmp_path, monkeypatch, [None, None, True, None])
        assert out[0] == [] and out[2] == [] and out[3] == []
        assert out[1][0].startswith("Celery broker (Redis) unreachable, or the ping could not be built")
        assert "on 2 checks in a row" in out[1][0]
        assert (tmp_path / "unsent").read_text() == "1"

    def test_a_stale_unsent_count_restarts(self, tmp_path, monkeypatch):
        import os
        f = tmp_path / "unsent"
        f.write_text("5")
        old = f.stat().st_mtime - watchdog._ORDERS_PING_STALE_S - 60
        os.utime(f, (old, old))
        assert self._check(tmp_path, monkeypatch, [None]) == [[]]
        assert f.read_text() == "1"

    def test_a_dead_worker_is_not_pinged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(watchdog, "_ORDERS_PING_STATE_FILE", tmp_path / "misses")
        with patch.object(watchdog, "_ping_orders_worker") as ping:
            assert watchdog._check_orders_worker_responds(
                ["Celery Worker (orders) is not running"]) == []
        ping.assert_not_called()

    def test_the_ping_targets_the_orders_worker(self):
        import socket
        import tasks
        with patch.object(tasks.app.control, "ping", return_value=[{"orders@x": {"ok": "pong"}}]) as p:
            assert watchdog._ping_orders_worker() is True
        assert p.call_args.kwargs["destination"] == [f"orders@{socket.gethostname()}"]
        with patch.object(tasks.app.control, "ping", return_value=[]):
            assert watchdog._ping_orders_worker() is False
        with patch.object(tasks.app.control, "ping", side_effect=OSError("broker down")):
            assert watchdog._ping_orders_worker() is None

    def test_a_ping_that_never_returns_is_abandoned(self, monkeypatch):
        import threading
        import time as _t
        import tasks
        release = threading.Event()
        monkeypatch.setattr(watchdog, "_ORDERS_PING_GRACE_S", 0.2)
        with patch.object(tasks.app.control, "ping", side_effect=lambda **k: release.wait(30)):
            t0 = _t.monotonic()
            assert watchdog._ping_orders_worker(timeout=0.1) is None
            assert _t.monotonic() - t0 < 5
        release.set()

    def test_a_missing_count_file_starts_at_one(self, tmp_path, monkeypatch):
        f = tmp_path / "misses"
        monkeypatch.setattr(watchdog, "_ORDERS_PING_STATE_FILE", f)
        with patch.object(watchdog, "_ping_orders_worker", return_value=False):
            assert watchdog._check_orders_worker_responds([]) == []
        assert f.read_text() == "1"

    def test_an_error_inside_the_ping_thread_is_none_and_says_so(self, monkeypatch):
        import tasks
        said = []
        monkeypatch.setattr(watchdog, "_say", said.append)
        with patch.object(tasks.app.control, "ping", side_effect=ImportError("no celery")):
            assert watchdog._ping_orders_worker(timeout=0.1) is None
        assert said and "could not be sent: no celery" in said[-1]

    def test_a_timed_out_ping_is_logged_as_a_timeout(self, monkeypatch):
        import threading
        import tasks
        release, said = threading.Event(), []
        monkeypatch.setattr(watchdog, "_say", said.append)
        monkeypatch.setattr(watchdog, "_ORDERS_PING_GRACE_S", 0.2)
        with patch.object(tasks.app.control, "ping", side_effect=lambda **k: release.wait(30)):
            assert watchdog._ping_orders_worker(timeout=0.1) is None
        release.set()
        assert "did not finish within" in said[-1]

    def test_a_stale_count_does_not_carry_over(self, tmp_path, monkeypatch):
        import os
        f = tmp_path / "misses"
        f.write_text("1")
        old = watchdog.time.time() - watchdog._ORDERS_PING_STALE_S - 60
        os.utime(f, (old, old))
        monkeypatch.setattr(watchdog, "_ORDERS_PING_STATE_FILE", f)
        with patch.object(watchdog, "_ping_orders_worker", return_value=False):
            assert watchdog._check_orders_worker_responds([]) == []
        assert f.read_text() == "1"

    def test_main_alerts_on_a_hung_worker_but_never_heals_on_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(watchdog, "_PLANNED_RESTART_MARKER", tmp_path / "none")
        monkeypatch.setattr(watchdog.sys, "argv", ["watchdog.py", "--heal"])
        hung = ["Celery Worker (orders) is running but has not answered a Celery ping"]
        with patch.object(watchdog, "_service_state", return_value="active"), \
             patch.object(watchdog, "_check_processes", return_value=[]), \
             patch.object(watchdog, "_check_snapshot_freshness", return_value=[]), \
             patch.object(watchdog, "_ping_orders_worker", return_value=None), \
             patch.object(watchdog, "_orders_active_tasks", return_value=None), \
             patch.object(watchdog, "_check_orders_worker_responds", return_value=hung), \
             patch.object(watchdog, "_attempt_heal", return_value=[]) as heal, \
             patch.object(watchdog, "_emit") as emit:
            watchdog.main()
        heal.assert_not_called()
        assert emit.call_args[0][0] == hung


class TestOrdersTaskNotStuck:
    """Professor M219: a hung child passes the ping; a long-active task shows it."""

    def test_a_task_running_past_the_limit_is_a_problem(self):
        now = watchdog.time.time()
        active = [{"name": "tasks._broker_place_stop_order", "time_start": now - watchdog._ORDERS_TASK_STUCK_S - 60},
                  {"name": "tasks.manage_stop_place", "time_start": now - 30}]
        with patch.object(watchdog, "_orders_active_tasks", return_value=active):
            out = watchdog._check_orders_task_not_stuck([])
        assert len(out) == 1 and "_broker_place_stop_order" in out[0] and "may be hung" in out[0]
        assert "16 min" in out[0]
        assert watchdog._problems_fingerprint(out) == watchdog._problems_fingerprint(
            [out[0].replace("16 min", "47 min")])   # the minute count doesn't re-fire it

    def test_nothing_active_or_unknown_is_quiet(self):
        for a in (None, [], [{"name": "x"}], [{"name": "x", "time_start": "bad"}]):
            with patch.object(watchdog, "_orders_active_tasks", return_value=a):
                assert watchdog._check_orders_task_not_stuck([]) == []

    def test_a_dead_worker_is_not_inspected(self):
        with patch.object(watchdog, "_orders_active_tasks") as insp:
            assert watchdog._check_orders_task_not_stuck(["Celery Worker (orders) is not running"]) == []
        insp.assert_not_called()

    def test_the_inspect_call_is_bounded_and_targets_the_orders_node(self, monkeypatch):
        import socket
        import threading
        import tasks
        node = f"orders@{socket.gethostname()}"
        insp = tasks.app.control.inspect
        fake = type("I", (), {"active": lambda self: {node: [{"name": "t", "time_start": 1.0}]}})
        with patch.object(tasks.app.control, "inspect", return_value=fake()) as i:
            assert watchdog._orders_active_tasks() == [{"name": "t", "time_start": 1.0}]
        assert i.call_args.kwargs["destination"] == [node]
        release = threading.Event()
        monkeypatch.setattr(watchdog, "_ORDERS_PING_GRACE_S", 0.2)
        slow = type("I", (), {"active": lambda self: release.wait(30)})
        try:
            with patch.object(tasks.app.control, "inspect", return_value=slow()):
                assert watchdog._orders_active_tasks(timeout=0.1) is None
        finally:
            release.set()
        with patch.object(tasks.app.control, "inspect", side_effect=OSError("down")):
            assert watchdog._orders_active_tasks() is None

    def test_main_alerts_on_a_stuck_task_but_never_heals(self, monkeypatch, tmp_path):
        monkeypatch.setattr(watchdog, "_PLANNED_RESTART_MARKER", tmp_path / "none")
        monkeypatch.setattr(watchdog.sys, "argv", ["watchdog.py", "--heal"])
        stuck = ["Celery Worker (orders) has been running _broker_place_stop_order for 20 min; x"]
        with patch.object(watchdog, "_service_state", return_value="active"), \
             patch.object(watchdog, "_check_processes", return_value=[]), \
             patch.object(watchdog, "_check_snapshot_freshness", return_value=[]), \
             patch.object(watchdog, "_check_orders_worker_responds", return_value=[]), \
             patch.object(watchdog, "_check_orders_task_not_stuck", return_value=stuck), \
             patch.object(watchdog, "_attempt_heal", return_value=[]) as heal, \
             patch.object(watchdog, "_emit") as emit:
            watchdog.main()
        heal.assert_not_called()
        assert emit.call_args[0][0] == stuck
