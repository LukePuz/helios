"""Boot record of beat entries that came due while the stack was down
(read-only on beat's schedule file)."""
import shelve
from datetime import datetime, timedelta, timezone

from celery import Celery
from celery.beat import ScheduleEntry
from celery.schedules import crontab

import beat_missed

_APP = Celery("t", set_as_current=False)
_APP.conf.timezone = "America/New_York"
_NOW = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)           # Mon 05:00 EST
_SCHED = {
    "daily-0715": {"task": "x", "schedule": crontab(hour=7, minute=15)},
    "every-5m": {"task": "y", "schedule": timedelta(minutes=5)},
    "sat-weekly": {"task": "z", "schedule": crontab(hour=6, minute=0, day_of_week=6)},
    "new-entry": {"task": "w", "schedule": timedelta(hours=1)},
}


def _write(path, last_runs):
    with shelve.open(str(path)) as s:
        s["entries"] = {n: ScheduleEntry(name=n, task="t", last_run_at=t, schedule=_SCHED[n]["schedule"],
                                         app=_APP) for n, t in last_runs.items()}


def test_lists_entries_due_since_their_last_run(tmp_path):
    path = tmp_path / "celerybeat-schedule"
    fri_close = datetime(2026, 10, 30, 23, 8, tzinfo=timezone.utc)   # Fri 19:08 EDT
    _write(path, {"daily-0715": datetime(2026, 10, 30, 11, 15, tzinfo=timezone.utc),
                  "every-5m": fri_close,
                  "sat-weekly": datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)})
    res = beat_missed.missed_beat_entries(_SCHED, _APP, path, now=_NOW)
    assert res["error"] is None
    due = [n for n, _ in res["due"]]
    assert "every-5m" in due and "sat-weekly" in due
    assert "daily-0715" in due                       # Sat/Sun/Mon-before-07:15 came due over the weekend
    assert res["no_record"] == ["new-entry"]


def test_not_due_when_last_run_is_recent(tmp_path):
    path = tmp_path / "celerybeat-schedule"
    _write(path, {"every-5m": _NOW - timedelta(minutes=1)})
    res = beat_missed.missed_beat_entries({"every-5m": _SCHED["every-5m"]}, _APP, path, now=_NOW)
    assert res["due"] == [] and res["no_record"] == []


def test_log_line_and_read_only(tmp_path, caplog):
    path = tmp_path / "celerybeat-schedule"
    _write(path, {"every-5m": datetime(2026, 10, 30, 23, 8, tzinfo=timezone.utc)})
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    with caplog.at_level("INFO"):
        line = beat_missed.log_missed_beat_entries({"every-5m": _SCHED["every-5m"]}, _APP, path, now=_NOW)
    assert line.startswith("Beat entries due at boot: 1 of 1: every-5m (last 2026-10-30 19:08 EDT)")
    assert line in caplog.text
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()} == before


def test_missing_file_is_reported_not_raised(tmp_path):
    line = beat_missed.log_missed_beat_entries(_SCHED, _APP, tmp_path / "nope", now=_NOW)
    assert "unknown (no beat schedule file" in line
    assert not any(tmp_path.iterdir())               # nothing created


def test_every_real_beat_entry_can_be_checked(tmp_path):
    import tasks
    sched = dict(tasks.app.conf.beat_schedule)
    path = tmp_path / "celerybeat-schedule"
    fri = datetime(2026, 10, 30, 23, 8, tzinfo=timezone.utc)
    with shelve.open(str(path)) as s:
        s["entries"] = {n: ScheduleEntry(name=n, task=c["task"], last_run_at=fri, schedule=c["schedule"],
                                         app=tasks.app) for n, c in sched.items()}
    res = beat_missed.missed_beat_entries(sched, tasks.app, path, now=_NOW)
    assert res["error"] is None and res["no_record"] == []
    assert 0 < len(res["due"]) <= len(sched)
