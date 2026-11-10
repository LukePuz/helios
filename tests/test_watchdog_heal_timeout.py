"""The watchdog's heal waits out helios-local's own start timeout
(TimeoutStartSec=300) instead of giving up at 90 s (audit 2026-11-08 15:08
#4)."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog  # noqa: E402


def test_start_gets_the_long_timeout():
    calls = []

    def fake(verb, unit, timeout=90.0):
        calls.append((verb, timeout))
        return True, ""
    with patch.object(watchdog, "_heal_allowed", return_value=(True, "")), \
         patch.object(watchdog, "_heal_history_record"), \
         patch.object(watchdog, "_heal_history", return_value=[]), \
         patch.object(watchdog, "_sudo_systemctl", side_effect=fake), \
         patch.object(watchdog, "_service_is_active", return_value=True), \
         patch.object(watchdog, "_check_processes", return_value=[]), \
         patch.object(watchdog.time, "sleep"):
        watchdog._attempt_heal("test")
    assert ("start", 320.0) in calls
    assert ("reset-failed", 90.0) in calls
    assert watchdog._HEAL_START_TIMEOUT_S > 300
