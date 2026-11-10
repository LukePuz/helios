"""The gateway check alerts on an inactive ibc-live inside the hours it must
be up (limitations audit 2026-11-08 11:07 #3), and stays silent outside."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog  # noqa: E402

ET = ZoneInfo("America/New_York")


def _at(*a):
    return datetime(*a, tzinfo=ET)


def test_expected_up_windows():
    assert watchdog._gateway_expected_up(_at(2026, 11, 9, 5, 0))        # Mon 05:00
    assert watchdog._gateway_expected_up(_at(2026, 11, 9, 18, 55))
    assert not watchdog._gateway_expected_up(_at(2026, 11, 9, 19, 0))
    assert not watchdog._gateway_expected_up(_at(2026, 11, 7, 12, 0))   # Sat
    assert not watchdog._gateway_expected_up(_at(2026, 11, 8, 10, 25))  # Sun morning
    assert watchdog._gateway_expected_up(_at(2026, 11, 8, 17, 30))      # Sun evening


def test_inactive_alerts_only_inside_the_window():
    with patch.object(watchdog, "_service_is_active", return_value=False):
        msgs = watchdog._check_gateway_login(_at(2026, 11, 9, 5, 5))
        assert len(msgs) == 1 and "NOT running" in msgs[0]
        assert watchdog._check_gateway_login(_at(2026, 11, 8, 10, 25)) == []


def test_active_paths_unchanged():
    with patch.object(watchdog, "_service_is_active", return_value=True), \
         patch.object(watchdog, "_ibkr_port_reachable", return_value=True):
        assert watchdog._check_gateway_login(_at(2026, 11, 9, 5, 5)) == []
    with patch.object(watchdog, "_service_is_active", return_value=True), \
         patch.object(watchdog, "_ibkr_port_reachable", return_value=False):
        assert "2FA" in watchdog._check_gateway_login(_at(2026, 11, 8, 10, 25))[0]
