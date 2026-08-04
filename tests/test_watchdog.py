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
