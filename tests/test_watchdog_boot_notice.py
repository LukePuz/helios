"""One Telegram notice per machine boot: when the box came back, whether the
previous run ended cleanly, downtime, which units are up (limitations audit
2026-11-08 11:07 #5)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog  # noqa: E402

# 2026-11-07 22:58:46 EST and 2026-11-08 10:21:20 EST, in microseconds.
PREV_END, CUR_START = 1794110326_000000, 1794151280_000000
BOOTS = [{"index": -1, "boot_id": "aaa", "first_entry": 1, "last_entry": PREV_END},
         {"index": 0, "boot_id": "bbb", "first_entry": CUR_START, "last_entry": CUR_START + 1}]


def _run(tail):
    def run(cmd, **kw):
        if "--list-boots" in cmd:
            return SimpleNamespace(stdout=json.dumps(BOOTS))
        return SimpleNamespace(stdout=tail)
    return run


def test_facts_tell_clean_from_unclean():
    f = watchdog._boot_facts(run=_run("x\nsystemd-shutdown[1]: Syncing filesystems\n"))
    assert f["boot_id"] == "bbb" and f["prev_clean"] is True
    assert watchdog._boot_facts(run=_run("kernel: something\n"))["prev_clean"] is False


def test_notice_once_per_boot(tmp_path):
    marker = tmp_path / "boot_id"
    facts = watchdog._boot_facts(run=_run("no shutdown lines"))
    with patch.object(watchdog, "_service_is_active", side_effect=lambda u: u == "ibc-live.service"):
        text = watchdog._boot_notice(facts, marker)
    assert "Helios box restarted" in text and "Sun 11-08 10:21 ET" in text
    assert "NOT a clean shutdown" in text and "down 11.4 h" in text
    assert "ibc-live up, helios-local down, helios-standby down" in text
    marker.write_text("bbb")
    assert watchdog._boot_notice(facts, marker) is None
    assert watchdog._boot_notice(None, marker) is None


def test_marker_written_only_after_a_send(tmp_path):
    marker = tmp_path / "boot_id"
    facts = {"boot_id": "bbb", "started": CUR_START / 1e6}
    with patch.object(watchdog, "_BOOT_NOTICE_MARKER", marker), \
         patch.object(watchdog, "_boot_facts", return_value=facts), \
         patch.object(watchdog, "_service_is_active", return_value=True), \
         patch.object(watchdog, "_send_telegram", return_value=False):
        watchdog._send_boot_notice()
    assert not marker.exists()
    with patch.object(watchdog, "_BOOT_NOTICE_MARKER", marker), \
         patch.object(watchdog, "_boot_facts", return_value=facts), \
         patch.object(watchdog, "_service_is_active", return_value=True), \
         patch.object(watchdog, "_send_telegram", return_value=True):
        watchdog._send_boot_notice()
    assert marker.read_text() == "bbb"
