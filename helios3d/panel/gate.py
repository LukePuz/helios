"""Live-lane gate: the panel calls a model only while the live LLM lane is
idle and no kill file exists. Every Redis command used is a read.

Checked before every call:
1. Kill file (HELIOS3D_PANEL_KILL_FILE, default logs/.h3d_off): present
   means stop now. Checked first and again while waiting.
2. Queued live work: LLEN of each live LLM queue (llm_p0, llm_p1, llm_p2,
   the names in tasks.py) and of kombu's priority sub-queues
   ("<queue>\\x06\\x16<n>" for n in 3, 6, 9). Any non-zero length blocks.
3. Reserved live work: kombu's ``unacked`` hash (HSCAN) holds messages a
   worker has taken but not acked; one whose routing key is a live LLM
   queue blocks (the llm worker prefetches one message while it runs).
   A SIGKILLed llm worker leaves its unacked entry behind. kombu restores
   it to the queue only once it is older than the broker's
   visibility_timeout, which tasks.py sets to 6 days (for eta tasks), so
   such an entry can hold the gate closed for up to 6 days. The gate
   result says "live_task_reserved"; clearing it is a broker admin step
   (restart the llm worker, or remove the entry by hand), never done here.
4. A running live task: tasks.py sets task_track_started, so a running task
   has a result key ``celery-task-meta-<id>`` with status STARTED and the
   worker's hostname in ``result.hostname``. A STARTED key whose hostname is
   the llm worker's (``llm@...``, process_manager.py) blocks. The key's age
   is the result TTL minus its remaining TTL; one older than
   HELIOS3D_PANEL_LIVE_TASK_MAX_AGE_S is reported as stale and does not block
   (a killed worker leaves STARTED behind for a day).
   Cost: SCAN pages of up to 1000 keys, one MGET per page, and json.loads
   only for values that contain "STARTED". More than
   HELIOS3D_PANEL_GATE_MAX_KEYS result keys (or unacked entries) fails
   closed ("too_many_keys") instead of scanning on.
Redis unreachable, or any read error, counts as "not clear".

wait_clear() polls until the lane is clear, the kill file appears, or
HELIOS3D_PANEL_GATE_WAIT_S passes; the runner then yields (stops the run
and leaves the cursor where it is).

What this cannot see: a model call made outside the Celery llm worker on the
same Ollama server. The gate also does not stop a live task from starting
right after the check; a shadow call in flight then delays it by up to one
call (the timeout caps that at HELIOS3D_PANEL_TIMEOUT_S).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

LIVE_LLM_QUEUES = ("llm_p0", "llm_p1", "llm_p2")   # tasks.LLM_QUEUE_*
KOMBU_PRIORITY_SEP = "\x06\x16"
KOMBU_PRIORITY_STEPS = (3, 6, 9)                     # step 0 is the bare name
UNACKED_KEY = "unacked"
RESULT_PREFIX = "celery-task-meta-"
SCAN_COUNT = 1000
LIVE_LLM_WORKER_PREFIX = "llm@"                      # process_manager: -n llm@%h


def queue_keys(queues=LIVE_LLM_QUEUES) -> list[str]:
    return [k for q in queues for k in [q] + [f"{q}{KOMBU_PRIORITY_SEP}{p}"
                                                for p in KOMBU_PRIORITY_STEPS]]


@dataclass
class GateResult:
    ok: bool
    reason: str
    detail: dict = field(default_factory=dict)
    waited_s: float = 0.0


def _s(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)


class TooManyKeys(RuntimeError):
    pass


class LiveLaneGate:
    def __init__(self, settings, redis_client=None, clock=time.monotonic, sleep=time.sleep):
        self.settings = settings
        self._redis = redis_client
        self.clock = clock
        self.sleep = sleep

    # -- sources --------------------------------------------------------
    def redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis.from_url(self.settings.live_redis_url,
                                               socket_timeout=3, socket_connect_timeout=3)
        return self._redis

    def kill_file_present(self) -> bool:
        return os.path.exists(self.settings.kill_file)

    def queued(self, r) -> dict:
        return {k: int(n) for k in queue_keys() for n in [r.llen(k)] if int(n or 0) > 0}

    def reserved(self, r) -> int:
        n = seen = 0
        for _tag, raw in r.hscan_iter(UNACKED_KEY, count=500):
            seen += 1
            if seen > self.settings.gate_max_keys:
                raise TooManyKeys(f"unacked has more than {self.settings.gate_max_keys} entries")
            try:
                rk = json.loads(_s(raw))[2]
            except (ValueError, IndexError, TypeError):
                continue
            if _s(rk) in LIVE_LLM_QUEUES:
                n += 1
        return n

    def running(self, r) -> tuple[int, int]:
        """(live llm tasks STARTED and fresh, stale STARTED entries)."""
        # Why every celery-task-meta key is scanned: the key
        # is celery-task-meta-<task uuid>, with no queue, task name or worker
        # in it, so no SCAN MATCH pattern narrows it to the llm lane. The
        # cheaper reads cannot see a running task: tasks.py acks early
        # (no task_acks_late), so a running message has left both the queue
        # and the unacked hash, and Celery's active/reserved lists are only
        # reachable through a control broadcast, which writes to the broker.
        # result_extended is off, so the value names no queue either; the
        # worker hostname inside a STARTED value is the only lane marker.
        # The scan runs only after the queue and unacked checks are clear,
        # costs one MGET per page plus one TTL call per STARTED value whose
        # hostname is the llm worker's, and stops at gate_max_keys.
        fresh = stale = seen = 0
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=f"{RESULT_PREFIX}*", count=SCAN_COUNT)
            seen += len(keys)
            if seen > self.settings.gate_max_keys:
                raise TooManyKeys(f"more than {self.settings.gate_max_keys} result keys")
            values = r.mget(keys) if keys else []
            for key, raw in zip(keys, values):
                if raw is None:
                    continue
                text = _s(raw)
                if "STARTED" not in text:     # cheap skip before json.loads
                    continue
                try:
                    meta = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(meta, dict) or meta.get("status") != "STARTED":
                    continue
                res = meta.get("result") if isinstance(meta.get("result"), dict) else {}
                if not str(res.get("hostname") or "").startswith(LIVE_LLM_WORKER_PREFIX):
                    continue
                ttl = r.ttl(key)
                age = (self.settings.result_expires_s - ttl) if ttl is not None and ttl >= 0 else None
                if age is not None and age > self.settings.live_task_max_age_s:
                    stale += 1
                else:
                    fresh += 1
            if int(cursor) == 0:
                return fresh, stale

    # -- decisions ------------------------------------------------------
    def check(self) -> GateResult:
        if self.kill_file_present():
            return GateResult(False, "kill_file", {"path": self.settings.kill_file})
        try:
            r = self.redis()
            q = self.queued(r)
            if q:
                return GateResult(False, "live_queue_not_empty", {"queued": q})
            n = self.reserved(r)
            if n:
                return GateResult(False, "live_task_reserved", {"reserved": n})
            fresh, stale = self.running(r)
            if fresh:
                return GateResult(False, "live_task_running", {"running": fresh, "stale": stale})
        except TooManyKeys as e:
            return GateResult(False, "too_many_keys", {"error": str(e)})
        except Exception as e:  # unreachable or unreadable: not clear
            return GateResult(False, "redis_unavailable", {"error": type(e).__name__})
        return GateResult(True, "clear", {"stale_started": stale} if stale else {})

    def wait_clear(self) -> GateResult:
        t0 = self.clock()
        while True:
            res = self.check()
            res.waited_s = round(self.clock() - t0, 3)
            if res.ok or res.reason == "kill_file":
                return res
            if res.waited_s >= self.settings.gate_wait_s:
                return res
            self.sleep(max(0.0, min(self.settings.gate_poll_s,
                                    self.settings.gate_wait_s - res.waited_s)))
