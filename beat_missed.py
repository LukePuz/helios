"""Read-only boot record of the beat entries that came due while the stack
was down, so it is known which catch-up runs beat fires at start.

Beat's PersistentScheduler keeps each entry's last_run_at in the shelve
file it runs with (celerybeat-schedule in the repo root). Before beat
starts, this opens that file read-only and checks every entry of
tasks.app.conf.beat_schedule against it. Nothing is written or scheduled.
"""
import logging
import shelve
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
DEFAULT_SCHEDULE_PATH = Path(__file__).resolve().parent / "celerybeat-schedule"


def missed_beat_entries(beat_schedule: dict, app, schedule_path=DEFAULT_SCHEDULE_PATH,
                        now: datetime | None = None) -> dict:
    """{"due": [(name, last_run_at)], "no_record": [name], "error": str|None}.
    due: entries whose next run time has passed since their last recorded
    run (beat fires each once at start). no_record: entries the file has
    no last run for. Never raises."""
    from celery.schedules import maybe_schedule
    out = {"due": [], "no_record": [], "error": None}
    now = now or datetime.now(timezone.utc)
    try:
        if not Path(schedule_path).exists() and not any(
                Path(str(schedule_path) + ext).exists() for ext in (".db", ".dat")):
            out["error"] = f"no beat schedule file at {schedule_path}"
            return out
        with shelve.open(str(schedule_path), flag="r") as store:
            entries = dict(store.get("entries") or {})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    for name, conf in sorted(beat_schedule.items()):
        entry = entries.get(name)
        last = getattr(entry, "last_run_at", None)
        if last is None:
            out["no_record"].append(name)
            continue
        try:
            sched = maybe_schedule(conf["schedule"], app=app)
            sched.nowfun = lambda: now
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if sched.remaining_estimate(last).total_seconds() <= 0:
                out["due"].append((name, last))
        except Exception as e:
            log.debug(f"beat entry {name}: due check failed: {e}")
    return out


def log_missed_beat_entries(beat_schedule: dict | None = None, app=None,
                            schedule_path=DEFAULT_SCHEDULE_PATH, now=None) -> str:
    """One INFO line naming the entries due at boot, with each one's last
    run in ET. Returns the line. Never raises."""
    try:
        if beat_schedule is None or app is None:
            import tasks
            app = app or tasks.app
            beat_schedule = beat_schedule if beat_schedule is not None else dict(app.conf.beat_schedule)
        res = missed_beat_entries(beat_schedule, app, schedule_path, now)
        if res["error"]:
            line = f"Beat entries due at boot: unknown ({res['error']})"
        else:
            due = ", ".join(f"{n} (last {t.astimezone(_ET):%Y-%m-%d %H:%M %Z})" for n, t in res["due"])
            line = (f"Beat entries due at boot: {len(res['due'])} of {len(beat_schedule)}"
                    + (f": {due}" if due else "")
                    + (f"; no last-run record: {', '.join(res['no_record'])}" if res["no_record"] else ""))
    except Exception as e:
        line = f"Beat entries due at boot: unknown ({type(e).__name__}: {e})"
    log.info(line)
    return line
