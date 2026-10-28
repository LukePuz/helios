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

from deploy import watchdog


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
