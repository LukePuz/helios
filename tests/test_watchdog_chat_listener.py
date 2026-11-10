"""Neither helios-local nor helios-standby up on two checks in a row is a
problem line (roadmap 04f)."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog  # noqa: E402


def _states(local, standby):
    return lambda u: local if u.startswith("helios-local") else standby


def test_two_misses_in_a_row_alert(tmp_path):
    f = tmp_path / "misses"
    with patch.object(watchdog, "_planned_restart_in_progress", return_value=False):
        assert watchdog._check_chat_listener(_states("inactive", "inactive"), f) == []
        out = watchdog._check_chat_listener(_states("inactive", "inactive"), f)
    assert len(out) == 1 and "No Telegram chat listener" in out[0]


def test_either_up_or_in_transition_resets(tmp_path):
    f = tmp_path / "misses"
    with patch.object(watchdog, "_planned_restart_in_progress", return_value=False):
        watchdog._check_chat_listener(_states("inactive", "inactive"), f)
        assert watchdog._check_chat_listener(_states("activating", "inactive"), f) == []
        assert watchdog._check_chat_listener(_states("inactive", "inactive"), f) == []   # count restarted
        assert watchdog._check_chat_listener(_states("inactive", "active"), f) == []


def test_planned_restart_counts_as_up(tmp_path):
    f = tmp_path / "misses"
    with patch.object(watchdog, "_planned_restart_in_progress", return_value=True):
        for _ in range(3):
            assert watchdog._check_chat_listener(_states("inactive", "inactive"), f) == []


def test_never_raises(tmp_path):
    def boom(u):
        raise OSError("x")
    assert watchdog._check_chat_listener(boom, tmp_path / "m") == []
