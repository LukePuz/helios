"""code_sha names the last code commit (research-log/ excluded) and is read
once at import (professor 0k, 2026-10-28)."""
from unittest.mock import patch

from agents import analyst


def _fake_git(outputs):
    def run(cmd, **kw):
        class R:
            stdout = outputs[cmd[3]]
        return R()
    return run


def test_code_sha_skips_research_log_and_its_dirty_files():
    out = {"rev-parse": "headsha\n", "log": "codesha\n",
           "status": " M research-log/transcripts-manifest.csv\n?? notes.md\n"}
    with patch("subprocess.run", side_effect=_fake_git(out)):
        assert analyst._git_code_sha() == {"code_sha": "codesha", "head_sha": "headsha"}


def test_code_changes_and_untracked_py_mark_dirty():
    for status in (" M agents/crew.py\n", "?? scratch.py\n"):
        out = {"rev-parse": "h", "log": "c", "status": status}
        with patch("subprocess.run", side_effect=_fake_git(out)):
            assert analyst._git_code_sha()["code_sha"] == "c-dirty"


def test_git_failure_gives_none():
    with patch("subprocess.run", side_effect=OSError):
        assert analyst._git_code_sha() == {"code_sha": None, "head_sha": None}


def test_untracked_package_marks_dirty_and_status_lists_every_file():
    """An untracked package is listed file by file, so its modules count."""
    calls = []
    out = {"rev-parse": "h", "log": "c", "status": "?? agents/newmod/__init__.py\n"}

    def run(cmd, **kw):
        calls.append(cmd)
        return _fake_git(out)(cmd, **kw)
    with patch("subprocess.run", side_effect=run):
        assert analyst._git_code_sha()["code_sha"] == "c-dirty"
    status_cmd = next(c for c in calls if c[3] == "status")
    assert "--untracked-files=all" in status_cmd


def test_git_timeout_is_retried_once():
    import subprocess
    out = {"rev-parse": "h", "log": "c", "status": ""}
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[3])
        if cmd[3] == "rev-parse" and seen.count("rev-parse") == 1:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return _fake_git(out)(cmd, **kw)
    with patch("subprocess.run", side_effect=run):
        assert analyst._git_code_sha() == {"code_sha": "c", "head_sha": "h"}
    assert seen.count("rev-parse") == 2


def test_git_calls_share_one_deadline():
    """Every command hanging: the whole check stops within _GIT_DEADLINE_S,
    with no retry once the budget is spent."""
    import subprocess
    clock = [1000.0]
    timeouts = []

    def run(cmd, **kw):
        timeouts.append(kw["timeout"])
        clock[0] += kw["timeout"]                 # the command used its whole timeout
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    with patch("subprocess.run", side_effect=run), patch("time.monotonic", lambda: clock[0]):
        assert analyst._git_code_sha() == {"code_sha": None, "head_sha": None}
    assert sum(timeouts) <= analyst._GIT_DEADLINE_S
    assert timeouts == [10.0, 5.0]                # retry gets only what is left


def test_code_mixed_is_none_when_the_disk_read_fails():
    analyst._DISK_CHECK.update(at=0.0, value=None)
    try:
        with patch.dict(analyst._CODE_SHA, {"code_sha": "aaa111", "head_sha": "aaa111"}), \
             patch.object(analyst, "_git_code_sha", return_value={"code_sha": None, "head_sha": None}):
            v = analyst._code_mixed_check()
        assert v["code_mixed"] is None and v["disk_code_sha"] is None
    finally:
        analyst._DISK_CHECK.update(at=0.0, value=None)
