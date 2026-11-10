"""Installed units that differ from the repo copies are named in the boot
notice, key names only (audit 2026-11-08 11:07 #8)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog  # noqa: E402


def _tree(tmp_path):
    repo, inst = tmp_path / "repo", tmp_path / "etc"
    repo.mkdir(); inst.mkdir()
    return repo, inst


def test_names_changed_keys_and_ignores_comments_and_placeholder(tmp_path):
    repo, inst = _tree(tmp_path)
    (repo / "a.service").write_text("# note\n[Service]\nUser=YOUR_USER\nTimeoutStartSec=300\n")
    (inst / "a.service").write_text("[Service]\nUser=luke\n\nTimeoutStartSec=90\n")
    (repo / "b.timer").write_text("# new comment\n[Timer]\nOnCalendar=*:0/5\n")
    (inst / "b.timer").write_text("[Timer]\nOnCalendar=*:0/5\n")
    (repo / "c.service").write_text("[Service]\nUser=x\n")          # not installed: skipped
    out = watchdog._unit_drift(repo, inst, user="luke")
    assert out == ["a.service (TimeoutStartSec)"]


def test_missing_drop_in_is_reported(tmp_path):
    repo, inst = _tree(tmp_path)
    (repo / "a.service.d").mkdir()
    (repo / "a.service.d" / "db-port.conf").write_text("[Service]\nEnvironment=SUPABASE_PORT=6543\n")
    assert watchdog._unit_drift(repo, inst, user="luke") == ["a.service.d/db-port.conf (not installed)"]


def test_values_are_never_shown(tmp_path):
    repo, inst = _tree(tmp_path)
    (repo / "a.service").write_text("[Service]\nEnvironment=SECRET=one\n")
    (inst / "a.service").write_text("[Service]\nEnvironment=SECRET=two\n")
    out = watchdog._unit_drift(repo, inst, user="luke")
    assert out == ["a.service (Environment)"] and "one" not in out[0] and "two" not in out[0]


def test_never_raises(tmp_path):
    assert watchdog._unit_drift(tmp_path / "nope", tmp_path / "nope2") == []
