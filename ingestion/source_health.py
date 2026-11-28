"""Per-source data health (record and report only; roadmap 04e4).

Each poller calls note(source, outcome) once per poll with "ok" (the poll
returned data), "empty" (it ran and returned nothing) or "error" (it
failed). note() only writes a Redis hash, never the DB, with a 0.2 s socket
timeout; after a connection error it skips Redis for _DOWN_BACKOFF_S, so a
Redis outage costs a poller at most one short timeout a minute.

Once a day, tasks.send_daily_digest calls daily_record(): it reads the
hashes, keeps a per-source streak book in system_state
(source_health_<env>), writes one activity row (stage source_health) and
returns the names of the sources dead for SOURCE_HEALTH_DEAD_SESSIONS or
more sessions, for one line in the digest.

Definitions (engineering health, not a study metric):
- a poll is "ok" when it returned data; "empty" when it ran and returned
  nothing; "error" when it failed. EDGAR and insider passes that ran count
  as ok even with no new filings (an empty pass is normal there).
- a source is "dead" for a session (ET date) when it was polled that day and
  no poll was ok. A source not polled that day is not judged and keeps its
  streak.
- dead_sessions is the count of consecutive judged sessions that were dead.
- a poll is "blocked" when the vendor refused it for the plan (Polygon: the
  endpoint is in polygon_stream._ENTITLEMENT_BLOCKED after the call, i.e. a
  403 NOT_AUTHORIZED). It is counted but never judges a session: a date with
  only blocked polls is not judged (the streak is kept), and a source whose
  polls that date were all blocked is never in the digest line.

Polygon and FRED are noted by instrument(), not by an edit in their modules:
ingestion.polygon_stream and ingestion.macro are hashed whole into
brief_template_full_sha, so their source stays as it is. instrument() wraps
polygon_stream._get and macro._fetch_series in the Celery worker's parent at
boot (tasks._instrument_source_health, worker_init), and judges each call by
what it returned: Polygon {} is blocked when the endpoint is entitlement-
blocked after the call, else error (it returns {} only on a failure or a
missing key), anything else ok; FRED [] is empty (a failed fetch also
returns []) unless FRED_API_KEY is missing (error), a non-empty list ok.

Nothing here retries, falls back, or changes what a poller returns. No data
values are stored: outcomes, counts and times only.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)

OK, EMPTY, ERROR, BLOCKED = "ok", "empty", "error", "blocked"
OUTCOMES = (OK, EMPTY, ERROR, BLOCKED)
_JUDGED = (OK, EMPTY, ERROR)
STAGE = "source_health"
_SOCKET_TIMEOUT_S = 0.2
_DOWN_BACKOFF_S = 60.0
_KEY_TTL_S = 21 * 86400
_DAY_FIELD = "day:{day}:{outcome}"

_client = None
_down_until = 0.0


def _enabled() -> bool:
    try:
        from config import SOURCE_HEALTH_RECORD_ENABLED
        return bool(SOURCE_HEALTH_RECORD_ENABLED)
    except Exception:
        return False


def _env() -> str:
    try:
        from config import TRADING_ENV
        return TRADING_ENV
    except Exception:
        return os.getenv("TRADING_ENV", "paper")


def _prefix(env: str | None = None) -> str:
    return f"helios:source_health:{env or _env()}:"


def _et_day(now: datetime) -> str:
    from zoneinfo import ZoneInfo
    return now.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _redis():
    """This module's own client: 0.2 s socket timeouts. None when redis
    can't be imported."""
    global _client
    if _client is None:
        try:
            import redis
            _client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                           socket_timeout=_SOCKET_TIMEOUT_S,
                                           socket_connect_timeout=_SOCKET_TIMEOUT_S,
                                           decode_responses=True)
        except Exception:
            return None
    return _client


def _is_connection_error(e: Exception) -> bool:
    try:
        import redis
        if isinstance(e, (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError)):
            return True
    except Exception:
        pass
    return isinstance(e, (OSError, TimeoutError))


def note(source: str, outcome: str, now: datetime | None = None) -> bool:
    """Record one poll's outcome for source. True when written. Never raises
    and never logs per call (callers may be on hot paths)."""
    global _down_until
    if outcome not in OUTCOMES or not source:
        return False
    try:
        if not _enabled() or time.monotonic() < _down_until:
            return False
        cli = _redis()
        if cli is None:
            return False
        now = now or datetime.now(timezone.utc)
        ts = now.isoformat()
        key = _prefix() + source
        pipe = cli.pipeline(transaction=False)
        pipe.hset(key, mapping={"last_poll_at": ts, "last_outcome": outcome})
        if outcome == OK:
            pipe.hset(key, mapping={"last_ok_at": ts, "consec_empty": 0, "consec_error": 0,
                                    "consec_not_ok": 0})
        elif outcome == BLOCKED:
            pipe.hset(key, "last_blocked_at", ts)   # streak counters untouched
        elif outcome == EMPTY:
            pipe.hincrby(key, "consec_empty", 1)
            pipe.hset(key, "consec_error", 0)
            pipe.hincrby(key, "consec_not_ok", 1)
        else:
            pipe.hincrby(key, "consec_error", 1)
            pipe.hset(key, mapping={"consec_empty": 0, "last_error_at": ts})
            pipe.hincrby(key, "consec_not_ok", 1)
        pipe.hincrby(key, _DAY_FIELD.format(day=_et_day(now), outcome=outcome), 1)
        pipe.expire(key, _KEY_TTL_S)
        pipe.execute()
        return True
    except Exception as e:
        if _is_connection_error(e):
            _down_until = time.monotonic() + _DOWN_BACKOFF_S
        return False


def _polygon_source(args, kwargs) -> str:
    endpoint = str(args[0] if args else kwargs.get("endpoint", ""))
    return "polygon_snapshot" if "/snapshot/" in endpoint else "polygon"


def _polygon_outcome(args, kwargs, out) -> str:
    if out:
        return OK
    try:
        from ingestion import polygon_stream
        endpoint = str(args[0] if args else kwargs.get("endpoint", ""))
        if endpoint.rsplit("/", 1)[0] in polygon_stream._ENTITLEMENT_BLOCKED:
            return BLOCKED
    except Exception:
        pass
    return ERROR


def _fred_source(args, kwargs) -> str:
    return f"fred:{args[0] if args else kwargs.get('series_id', '')}"


def _fred_outcome(args, kwargs, out) -> str:
    if out:
        return OK
    try:
        from ingestion import macro
        missing_key = not macro.FRED_API_KEY
    except Exception:
        missing_key = False
    return ERROR if missing_key else EMPTY


def _wrap(fn, source, judge):
    """fn, plus one note() after each call: judged from what it returned, or
    error when fn raised (noted first, then the same exception is re-raised
    unchanged). The return value passes through unchanged."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            out = fn(*args, **kwargs)
        except Exception:
            try:
                note(source(args, kwargs), ERROR)
            except Exception:
                pass
            raise
        try:
            note(source(args, kwargs), judge(args, kwargs, out))
        except Exception:
            pass
        return out
    wrapped._source_health_wrapped = True
    return wrapped


def instrument() -> list[str]:
    """Wrap polygon_stream._get and macro._fetch_series once (see the module
    note). Returns what it wrapped. Off (SOURCE_HEALTH_RECORD_ENABLED False):
    wraps nothing. Never raises."""
    done = []
    if not _enabled():
        return done
    for mod_name, attr, source, judge in (
            ("ingestion.polygon_stream", "_get", _polygon_source, _polygon_outcome),
            ("ingestion.macro", "_fetch_series", _fred_source, _fred_outcome)):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr)
            if not getattr(fn, "_source_health_wrapped", False):
                setattr(mod, attr, _wrap(fn, source, judge))
            done.append(f"{mod_name}.{attr}")
        except Exception:
            continue
    return done


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _read_hashes(cli, env: str) -> dict:
    out = {}
    prefix = _prefix(env)
    for key in cli.scan_iter(match=prefix + "*", count=200):
        out[key[len(prefix):]] = cli.hgetall(key) or {}
    return out


def _judge(book_entry: dict, h: dict, day: str) -> dict:
    """One source's entry for day: today's counts, streaks and dead_sessions
    (idempotent within a day: a rerun recomputes from the stored base)."""
    counts = {o: _int(h.get(_DAY_FIELD.format(day=day, outcome=o))) for o in OUTCOMES}
    polls = sum(counts[o] for o in _JUDGED)
    prev = book_entry or {}
    base = prev.get("dead_base", prev.get("dead_sessions", 0)) if prev.get("last_session") == day \
        else prev.get("dead_sessions", 0)
    base = _int(base)
    judged = polls > 0
    dead = judged and counts[OK] == 0
    if judged:
        dead_sessions = base + 1 if dead else 0
    else:
        dead_sessions = base
    last_ok = h.get("last_ok_at") or prev.get("last_success_at")
    days_since = None
    if last_ok:
        try:
            from datetime import date
            days_since = (date.fromisoformat(day)
                          - date.fromisoformat(_et_day(datetime.fromisoformat(last_ok)))).days
        except Exception:
            days_since = None
    blocked_only = counts[BLOCKED] > 0 and not judged
    return {"polls": polls, **counts, "judged": judged, "dead_today": dead,
            "blocked_only": blocked_only, "last_blocked_at": h.get("last_blocked_at"),
            "consec_empty": _int(h.get("consec_empty")), "consec_error": _int(h.get("consec_error")),
            "consec_not_ok": _int(h.get("consec_not_ok")),
            "last_poll_at": h.get("last_poll_at") or prev.get("last_poll_at"),
            "last_success_at": last_ok, "last_error_at": h.get("last_error_at"),
            "days_since_success": days_since, "dead_sessions": dead_sessions,
            "_book": {"last_session": day if judged else prev.get("last_session"),
                      "dead_base": base if judged else prev.get("dead_base", base),
                      "dead_sessions": dead_sessions, "last_success_at": last_ok,
                      "last_poll_at": h.get("last_poll_at") or prev.get("last_poll_at")}}


def daily_record(now: datetime | None = None) -> list[str]:
    """Once a day (tasks.send_daily_digest, its own try): judge each source
    for today, update the streak book, write one source_health activity row,
    and return the names dead for SOURCE_HEALTH_DEAD_SESSIONS+ sessions
    (sorted). [] when off, or when nothing is dead. A failed Redis read
    writes a row saying so, leaves the book alone, and returns []. Never
    raises."""
    try:
        if not _enabled():
            return []
        import db
        from config import SOURCE_HEALTH_DEAD_SESSIONS as threshold
        now = now or datetime.now(timezone.utc)
        env, day = _env(), _et_day(now)
        cli = _redis()
        try:
            if cli is None:
                raise RuntimeError("redis unavailable")
            hashes = _read_hashes(cli, env)
        except Exception as e:
            db.log_activity_direct(STAGE, "", f"Source health {day}: Redis read failed",
                                   "neutral", {"day": day, "env": env, "read": type(e).__name__,
                                               "sources": {}, "dead": []})
            return []
        book_key = f"source_health_{env}"
        try:
            book = json.loads(db.get_state_strict(book_key) or "{}")
            if not isinstance(book, dict):
                book = {}
        except Exception:
            book = None   # unreadable: judge from a blank book, don't overwrite it
        sources, new_book = {}, dict(book or {})
        for name in sorted(set(hashes) | set(book or {})):
            entry = _judge((book or {}).get(name), hashes.get(name, {}), day)
            new_book[name] = entry.pop("_book")
            sources[name] = entry
            # today's day fields stay; older ones go (the hash keeps one day)
            try:
                old = [f for f in hashes.get(name, {}) if f.startswith("day:")
                       and not f.startswith(f"day:{day}:")]
                if old:
                    cli.hdel(_prefix(env) + name, *old)
            except Exception:
                pass
        dead = sorted(n for n, s in sources.items()
                      if s["dead_sessions"] >= threshold and not s["blocked_only"])
        if book is not None:
            db.set_state(book_key, json.dumps(new_book, default=str))
        db.log_activity_direct(
            STAGE, "", (f"Source health {day}: {len(dead)} dead {threshold}+ sessions"
                        + (f" ({', '.join(dead)})" if dead else "")),
            "negative" if dead else "neutral",
            {"day": day, "env": env, "read": "ok" if book is not None else "book_unreadable",
             "threshold": threshold, "dead": dead, "sources": sources})
        return dead
    except Exception as e:
        log.warning(f"source health daily record failed: {type(e).__name__}: {e}")
        return []
