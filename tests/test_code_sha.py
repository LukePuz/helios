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
