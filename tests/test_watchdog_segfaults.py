"""Celery pool children that died on an unplanned signal since helios-local
started become one problem line (audits 2026-11-08 15:08 #3, 17:17 #9)."""
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog  # noqa: E402

START = int(datetime(2026, 11, 9, 5, 0, 0).timestamp())


def _run(since=f"@{START}"):
    return lambda cmd, **kw: SimpleNamespace(stdout=since)


def _line(ts, sig):
    return (f"[{ts},123: ERROR/MainProcess] Process 'ForkPoolWorker-2' pid:4242 "
            f"exited with 'signal {sig}'")


def test_counts_unplanned_signals_since_start(tmp_path):
    (tmp_path / "celery_llm_worker.log").write_text("\n".join([
        _line("2026-11-09 06:01:02", "11 (SIGSEGV)"),
        _line("2026-11-09 06:05:00", "11 (SIGSEGV)"),
        _line("2026-11-09 07:00:00", "15 (SIGTERM)"),        # planned
        _line("2026-11-08 12:00:00", "11 (SIGSEGV)"),        # before start
        "unrelated"]))
    (tmp_path / "celery_worker.log").write_text(_line("2026-11-09 08:00:00", "6 (SIGABRT)"))
    out = watchdog._check_worker_crashes(run=_run(), log_dir=tmp_path)
    assert out == ["3 pool worker crash(es) since helios-local.service started "
                   "(celery_llm_worker SIGSEGV x2, celery_worker SIGABRT x1)"]


def test_nothing_when_clean(tmp_path):
    (tmp_path / "celery_worker.log").write_text(_line("2026-11-09 07:00:00", "9 (SIGKILL)"))
    assert watchdog._check_worker_crashes(run=_run(), log_dir=tmp_path) == []


def test_inactive_service_is_skipped(tmp_path):
    (tmp_path / "celery_worker.log").write_text(_line("2026-11-09 08:00:00", "11 (SIGSEGV)"))
    assert watchdog._check_worker_crashes(run=_run(since=""), log_dir=tmp_path) == []


def test_never_raises(tmp_path):
    def boom(cmd, **kw):
        raise OSError("no systemctl")
    assert watchdog._check_worker_crashes(run=boom, log_dir=tmp_path) == []


def test_rotated_log_counts_into_the_same_worker(tmp_path):
    (tmp_path / "celery_worker.log").write_text(_line("2026-11-09 09:00:00", "11 (SIGSEGV)"))
    (tmp_path / "celery_worker.log.1").write_text(_line("2026-11-09 06:00:00", "11 (SIGSEGV)"))
    out = watchdog._check_worker_crashes(run=_run(), log_dir=tmp_path)
    assert out == ["2 pool worker crash(es) since helios-local.service started (celery_worker SIGSEGV x2)"]


def test_exit_codes_count_and_planned_ones_do_not(tmp_path):
    lines = [
        "[2026-11-09 06:00:00,1: ERROR/MainProcess] Process 'ForkPoolWorker-1' pid:1 exited with 'exitcode 70'",
        "[2026-11-09 06:00:01,1: ERROR/MainProcess] Process 'ForkPoolWorker-2' pid:2 exited with 'exitcode 70'",
        "[2026-11-09 07:00:00,1: ERROR/MainProcess] Process 'ForkPoolWorker-3' pid:3 exited with 'exitcode 15'",
    ]
    (tmp_path / "celery_screen_worker.log").write_text("\n".join(lines))
    out = watchdog._check_worker_crashes(run=_run(), log_dir=tmp_path)
    assert out == ["2 pool worker crash(es) since helios-local.service started "
                   "(celery_screen_worker exitcode 70 x2)"]
