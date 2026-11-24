"""Record only: the ticker's recent-news timeline at each analyst verdict.

The brief has no recent-news timeline for the ticker: its memory section has
no ticker filter, no recency limit and no dates (context_engine.py, the
semantic memory block). A screener-rejected headline therefore never reaches
the prompt, whatever it said (limitations audit 2026-11-22 02:56 #2: IONQ
OPEN 94689 never saw the 11-13 CFO exit or the 11-16/11-17 downgrades). For
each analyst verdict on config.NEWS_TIMELINE_PATHS this records what a
"last 7 days" block for the ticker would have held, and whether each item
reached the prompt that was sent.

Off the verdict path: analyst._news_timeline_enqueue puts a job (the inputs
already in memory, no I/O) on this module's own bounded queue; one daemon
thread per process reads news_log and the screener rows on its own short
connection (never the pool; on the transaction pooler, port 6543, so not a
session-pooler session), builds the row and queues it on db.log_activity_queued
(activity stage "news_timeline"), so the write shares the process's activity
writer and this thread holds no write session of its own. The window is
closed ([verdict - 7 d, verdict)), so computing it seconds later reads the
same rows. A full queue drops the job (counted); a job still queued when the
process exits is lost (drain_now at a Celery child's exit lets the thread
finish what it can). Nothing here is read by the prompt, the brief, the
gates or routing.

Row extra: shadow_key "{env}:{signal_id}", signal_id, env, path, verdict_at,
prompt_at (when the prompt was built; the window is [verdict - 7 d, verdict),
so headlines logged during inference are in it, flagged after_prompt, with
n_items_before_prompt, n_negative_before_prompt and
n_negative_rejected_before_prompt counting the rest), window_days, cap, read ("ok" or "error: <class>"), the counts below over
every story in the window, and items: the newest `cap` stories, newest
first, each {date_et, title (160 chars), source, screen ("passed": a
SCREENED_IN row; "rejected": SCREENED_OUT rows and no SCREENED_IN; or
"no_screen_row": none found; screener rows carry the item's url and title
only from 2026-11-09), screen_reason (the newest matching row's, 160 chars),
n_screens, tone (agents.screener.title_tone: pos/neg/mixed/neutral),
in_prompt (any copy's title, as logged or without a trailing " - Publisher":
its first 60 characters, lower-cased, in the lower-cased prompt sent),
after_prompt (first copy logged at or after prompt_at; None without one),
is_trigger, n_copies}. "negative" means tone "neg" (not
"mixed"). Stories are deduped on agents.story_dedup.normalise_title (the
" - Publisher" suffix removed on news.google.com links), keeping the newest
copy; news_log.published_at is the time the headline was logged.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

STAGE = "news_timeline"
WINDOW_DAYS = 7
CAP = 10
TITLE_CHARS = 160
PROMPT_MATCH_CHARS = 60
READ_LIMIT = 3000                # rows per read (news_log, screener rows)
READ_TIMEOUT_MS = 5000           # each statement; on the side thread
QUEUE_MAX = 200
DRAIN_S = 3.0
_PUBLISHER_SUFFIX = re.compile(r"\s+[-\u2013\u2014|]\s+[^-\u2013\u2014|]{2,40}$")   # as story_dedup
DEDUPE = "story_dedup.normalise_title (publisher suffix removed on news.google.com links)"
_ET = ZoneInfo("America/New_York")

_NEWS_SQL = ("SELECT published_at, title, url FROM news_log WHERE ticker = %s "
             "AND published_at >= %s AND published_at < %s ORDER BY published_at DESC LIMIT %s")
_SCREENS_SQL = ("SELECT timestamp, signal, decision->>'url', decision->>'title', "
                "left(reasoning, 200) FROM signals WHERE symbol = %s "
                "AND signal IN ('SCREENED_IN', 'SCREENED_OUT') AND trading_env = %s "
                "AND timestamp >= %s AND timestamp < %s ORDER BY timestamp DESC LIMIT %s")

_QUEUE = None
_WORKER = None            # (pid, thread)
_START_LOCK = None        # (pid, lock): a lock made at import isn't fork-safe
DROPPED = {"queue_full": 0, "at_exit": 0, "dead_thread": 0}


# ---------------------------------------------------------------- pure parts

def _story(title, url):
    from agents.story_dedup import is_google_news, normalise_title
    return normalise_title(title, strip_publisher=is_google_news(url))


def _url_key(url):
    from agents.story_dedup import url_key
    return url_key(str(url)[:300]) if url else None


def _aware(ts):
    if not isinstance(ts, datetime):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _in_prompt(titles, low: str) -> bool:
    """Any copy's title, as logged or with a trailing " - Publisher"
    removed, has its first PROMPT_MATCH_CHARS characters (lower-cased) in
    the lower-cased prompt."""
    for t in titles:
        t = (t or "").strip()
        for form in (t, _PUBLISHER_SUFFIX.sub("", t).strip()):
            if form and form[:PROMPT_MATCH_CHARS].lower() in low:
                return True
    return False


def build_record(news_rows, screen_rows, prompt, trigger=None, cap: int = CAP,
                 prompt_at=None) -> dict:
    """The counts and items from the two reads (news_rows: (published_at,
    title, url), screen_rows: (timestamp, signal, url, title, reasoning),
    both newest first) and the prompt that was sent. prompt_at (when the
    prompt was built; None if unknown) sets each item's after_prompt (its
    first copy was logged at or after prompt_at, so it could not have been
    in the prompt) and the *_before_prompt counts; the window itself is
    not changed. Pure; never raises ({"build": "error: <class>"} instead)."""
    try:
        from agents.screener import title_tone
        from agents.story_dedup import story_domain
        stories = {}
        for ts, title, url in news_rows or ():
            key = _story(title, url)
            if not key:
                continue
            s = stories.get(key)
            if s is None:
                stories[key] = s = {"ts": _aware(ts), "first": _aware(ts), "title": str(title),
                                    "url": url, "urls": set(), "titles": [], "n": 0}
            s["n"] += 1
            if _aware(ts) and (s["first"] is None or _aware(ts) < s["first"]):
                s["first"] = _aware(ts)
            if len(s["titles"]) < 5 and str(title) not in s["titles"]:
                s["titles"].append(str(title))
            if _url_key(url):
                s["urls"].add(_url_key(url))
        by_url, by_story = {}, {}
        for ts, sig, url, title, reason in screen_rows or ():
            row = (_aware(ts), sig, reason)
            if _url_key(url):
                by_url.setdefault(_url_key(url), []).append(row)
            key = _story(title, url)
            if key:
                by_story.setdefault(key, []).append(row)
        low = (prompt or "").lower()
        prompt_at = _aware(prompt_at)
        trig = trigger if isinstance(trigger, dict) else {}
        trig_url, trig_story = _url_key(trig.get("url")), _story(trig.get("title"), trig.get("url"))
        items = []
        for key, s in stories.items():
            seen, screens = set(), []
            for row in [r for u in s["urls"] for r in by_url.get(u, ())] + by_story.get(key, []):
                if id(row) not in seen:
                    seen.add(id(row))
                    screens.append(row)
            screens.sort(key=lambda r: r[0] or datetime.min.replace(tzinfo=timezone.utc),
                         reverse=True)
            passed = [r for r in screens if r[1] == "SCREENED_IN"]
            decision = "passed" if passed else "rejected" if screens else "no_screen_row"
            newest = (passed or screens or [None])[0]
            title = s["title"].strip()
            items.append({
                "ts": s["ts"],
                "date_et": s["ts"].astimezone(_ET).date().isoformat() if s["ts"] else None,
                "title": title[:TITLE_CHARS], "source": story_domain(s["url"], title, None),
                "screen": decision,
                "screen_reason": (str(newest[2])[:TITLE_CHARS] if newest and newest[2] else None),
                "n_screens": len(screens), "tone": title_tone(title),
                "in_prompt": _in_prompt(s["titles"], low),
                "after_prompt": ((s["first"] >= prompt_at) if prompt_at and s["first"] else None),
                "is_trigger": bool((trig_url and trig_url in s["urls"])
                                   or (trig_story and trig_story == key)),
                "n_copies": s["n"]})
        items.sort(key=lambda i: i["ts"] or datetime.min.replace(tzinfo=timezone.utc),
                   reverse=True)
        neg = [i for i in items if i["tone"] == "neg"]
        rec = {"n_news_rows_7d": len(news_rows or ()), "n_screen_rows_7d": len(screen_rows or ()),
               "n_items_7d": len(items),
               "n_passed_7d": sum(i["screen"] == "passed" for i in items),
               "n_rejected_7d": sum(i["screen"] == "rejected" for i in items),
               "n_no_screen_row_7d": sum(i["screen"] == "no_screen_row" for i in items),
               "n_negative_7d": len(neg),
               "n_negative_rejected_7d": sum(i["screen"] == "rejected" for i in neg),
               "n_positive_7d": sum(i["tone"] == "pos" for i in items),
               "n_in_prompt_7d": sum(i["in_prompt"] for i in items),
               "n_negative_in_prompt_7d": sum(i["in_prompt"] for i in neg),
               **_before_prompt(items, neg, prompt_at),
               "items": [{k: v for k, v in i.items() if k != "ts"} for i in items[:cap]]}
        return rec
    except Exception as e:
        return {"build": f"error: {type(e).__name__}"}


def _before_prompt(items, neg, prompt_at) -> dict:
    """The window counts restricted to items first logged before prompt_at
    (None without a prompt_at)."""
    if prompt_at is None:
        return {"n_items_before_prompt": None, "n_negative_before_prompt": None,
                "n_negative_rejected_before_prompt": None}
    return {"n_items_before_prompt": sum(i["after_prompt"] is False for i in items),
            "n_negative_before_prompt": sum(i["after_prompt"] is False for i in neg),
            "n_negative_rejected_before_prompt": sum(i["after_prompt"] is False
                                                     and i["screen"] == "rejected" for i in neg)}


def _tone_lexicon() -> dict:
    try:
        from agents.screener import TONE_SHA
        return {"tone_lexicon": "agents.screener.title_tone", "tone_sha": TONE_SHA}
    except Exception:
        return {"tone_lexicon": "agents.screener.title_tone", "tone_sha": None}


# ---------------------------------------------------------------- the reads

def read_inputs(cur, ticker: str, env: str, since, until, limit: int = READ_LIMIT):
    """(news_rows, screen_rows) on an open cursor. Raises on any error."""
    cur.execute("SET LOCAL statement_timeout = %s", (int(READ_TIMEOUT_MS),))
    cur.execute(_NEWS_SQL, (str(ticker).upper(), since, until, int(limit)))
    news = cur.fetchall()
    cur.execute(_SCREENS_SQL, (str(ticker).upper(), env, since, until, int(limit)))
    return news, cur.fetchall()


def read_port() -> int:
    """NEWS_TIMELINE_READ_PORT from the environment (e.g. a local Postgres
    clone), else db._TRANSACTION_POOLER_PORT (6543). An unusable value
    falls back to the default."""
    import db
    try:
        return int(os.getenv("NEWS_TIMELINE_READ_PORT") or db._TRANSACTION_POOLER_PORT)
    except ValueError:
        return int(db._TRANSACTION_POOLER_PORT)


def _read(ticker: str, env: str, since, until):
    """Both reads on a short connection of their own (not the pool), on
    read_port() (the transaction pooler, 6543, unless NEWS_TIMELINE_READ_PORT
    is set) whatever port the bot uses, so it never holds one of the session
    pooler's sessions. One
    transaction: SET TRANSACTION READ ONLY and SET LOCAL statement_timeout
    sit inside it, which transaction mode allows. Rolled back and closed.
    Raises on any error."""
    import db
    conn = db.psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"), port=read_port(),
        dbname=os.getenv("SUPABASE_DB", "postgres"), user=os.getenv("SUPABASE_USER", "postgres"),
        password=os.getenv("SUPABASE_PASSWORD"), sslmode="require", connect_timeout=5,
        keepalives=1, keepalives_idle=10, keepalives_interval=5, keepalives_count=3,
        tcp_user_timeout=10000)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            return read_inputs(cur, ticker, env, since, until)
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()


def run_job(job: dict, read=None, write=None) -> dict:
    """Reads, builds and writes one verdict's row; returns the row's extra.
    read(ticker, env, since, until) and write(stage, ticker, message,
    sentiment, extra) default to _read and db.log_activity_queued (the row
    joins the process's activity writer, so this thread never holds a
    write session of its own; extra gains queued_at there). A failed
    read writes a row with read "error: <class>" and no counts. Never
    raises."""
    try:
        verdict_at = job["verdict_at"]
        since = verdict_at - timedelta(days=WINDOW_DAYS)
        env, sid = job.get("env"), job.get("signal_id")
        extra = {"shadow_key": f"{env}:{sid}", "signal_id": sid, "env": env,
                 "path": job.get("path"), "verdict_at": verdict_at.isoformat(),
                 "prompt_at": (job["prompt_at"].isoformat()
                               if isinstance(job.get("prompt_at"), datetime) else None),
                 "window_days": WINDOW_DAYS, "cap": CAP, "dedupe": DEDUPE, **_tone_lexicon()}
        t0 = time.monotonic()
        try:
            news, screens = (read or _read)(job["ticker"], env, since, verdict_at)
            extra["read"] = "ok"
            extra["limit_hit"] = len(news) >= READ_LIMIT or len(screens) >= READ_LIMIT
            extra.update(build_record(news, screens, job.get("prompt"), job.get("trigger"),
                                      prompt_at=job.get("prompt_at")))
        except Exception as e:
            extra["read"] = f"error: {type(e).__name__}"
        extra["read_ms"] = round((time.monotonic() - t0) * 1000)
        extra["lag_s"] = round((datetime.now(timezone.utc) - verdict_at).total_seconds(), 1)
        if extra["read"] == "ok" and "n_items_7d" in extra:
            msg = (f"{job['ticker']} last {WINDOW_DAYS} d: {extra['n_items_7d']} stories, "
                   f"{extra['n_negative_7d']} negative ({extra['n_negative_rejected_7d']} "
                   f"screener-rejected), {extra['n_in_prompt_7d']} in the prompt")
        else:
            msg = f"{job['ticker']} news timeline not read ({extra.get('read') or extra.get('build')})"
        if write is None:
            import db
            write = db.log_activity_queued
        write(STAGE, job["ticker"], msg, "neutral", extra)
        return extra
    except Exception as e:
        log.debug(f"news timeline job failed: {e}")
        return {"job": f"error: {type(e).__name__}"}


# ---------------------------------------------------------------- the lane

def _worker(q) -> None:
    while True:
        job = q.get()
        try:
            run_job(job)
        except Exception as e:
            log.debug(f"news timeline job failed: {e}")
        finally:
            q.task_done()


def enqueue(job: dict) -> bool:
    """Queue one job for this process's news-timeline thread, starting it
    when there is none, when it died, or in a forked child. No I/O; returns
    at once (False when dropped). Never raises."""
    global _QUEUE, _WORKER, _START_LOCK
    try:
        import queue
        if _START_LOCK is None or _START_LOCK[0] != os.getpid():
            _START_LOCK = (os.getpid(), threading.Lock())
        with _START_LOCK[1]:
            w = _WORKER
            if _QUEUE is None or w is None or w[0] != os.getpid() or not w[1].is_alive():
                import atexit
                if _QUEUE is not None and w is not None and w[0] == os.getpid():
                    DROPPED["dead_thread"] += _QUEUE.qsize()    # this process's, never ran
                _QUEUE = queue.Queue(maxsize=QUEUE_MAX)
                t = threading.Thread(target=_worker, args=(_QUEUE,), name="news-timeline",
                                     daemon=True)
                t.start()
                _WORKER = (os.getpid(), t)
                atexit.register(drain_now)
            q = _QUEUE
        try:
            q.put_nowait(job)
            return True
        except queue.Full:
            DROPPED["queue_full"] += 1
            if DROPPED["queue_full"] in (1, 10, 100) or DROPPED["queue_full"] % 1000 == 0:
                log.warning(f"news timeline queue full: {DROPPED['queue_full']} jobs dropped")
            return False
    except Exception as e:
        log.debug(f"news timeline enqueue failed: {e}")
        return False


def drain_now(timeout_s: float = DRAIN_S) -> int:
    """At a process's exit (atexit, or tasks.py's Celery child-exit hook):
    let the thread finish for up to timeout_s, then drop what is still
    queued (counted in DROPPED["at_exit"]; returned). Only this process's
    own queue. Never raises."""
    import queue
    try:
        w, q = _WORKER, _QUEUE
        if q is None or w is None or w[0] != os.getpid():
            return 0
        end = time.monotonic() + timeout_s
        while q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.05)
        n = 0
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            q.task_done()
            n += 1
        DROPPED["at_exit"] += n
        return n
    except Exception as e:
        log.debug(f"news timeline drain failed: {e}")
        return 0
