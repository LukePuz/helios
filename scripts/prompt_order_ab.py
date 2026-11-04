"""Replay stored analyst prompts under the answer order before and after
31d52f8 (rewritten hash 01aecd8, 2026-10-27), which moved the verdict fields
(action, conviction, the four sub-scores) from the top of the JSON schema to
the bottom, after the evidence and reasoning, and added a three-line "fill
the fields in the order shown" instruction.

Arms (each built from the stored prompt, signals.prompt, by swapping only
the schema block; the text before and after it stays byte-identical, which
is asserted per signal):
  A  the pre-31d52f8 block (scripts/prompt_ab_old_schema.txt)
  B  the post-31d52f8 block as shipped, instruction included
     (scripts/prompt_order_ab_new_schema.txt)
  C  (only with --arms ABC) B's field order under A's header line, without
     the three-line instruction: A and C differ only in field order.
  A2 (only with --repeat-arm A2) a second independent call with arm A's
     prompt and options, on every signal or on a seeded --repeat-subsample
     N of them: the A-vs-A2 discordance is the run-to-run noise floor. It
     joins the shuffled call order and counts in the stop-time sizing.

The sample is chosen by the caller; there is no default:
  --signal-ids-file FILE              ids, one per line ('#' comments ok)
  --from/--to --trading-env --sample N --seed S
                                      N ids drawn (seeded) from the rows with
                                      a stored prompt in [from, to)
--seed also seeds the per-signal arm order (random, fixed per signal id).

Modes:
  --dry-run (default)  read the stored prompts, build the arms, print their
                       hashes, the identity checks and the schema diff, and
                       the live call settings. Calls no model.
  --apply              call the model. Needs --out and --stop-at HH:MM (ET).

The call is the live analyst call: the payload (model tag, options with
num_ctx, temperature, num_predict, repeat_penalty, logprobs) is captured
from agents.analyst._call_ollama itself with its HTTP post stubbed out, so
it follows the live code, and so does the timeout: --timeout defaults to,
and may not exceed, the timeout of that call (400 s). Every arm uses the live num_predict (1600) unless
--arm-a-num-predict 1200 gives arm A the pre-31d52f8 value.

Only while the stack is down. --apply refuses to start, and stops before
each pair, when helios-local is not inactive (systemctl is-active), a
Celery beat process is running (it schedules check_ollama_gpu_health, a
72B probe every 30 min, and the live analysis tasks), or it is 05:00-19:00
ET on a weekday (helios-start.timer / helios-evening-stop.timer). --stop-at
must be at most 10 h away and no later than the next 05:00 ET.

Before every call the live lane must also be idle: the kill file
(--kill-file) is absent; in the live Redis broker (read-only: LLEN, HSCAN,
SCAN, MGET, TTL) no task is queued on llm_p0/llm_p1/llm_p2 or their kombu
priority sub-queues, none is reserved in kombu's unacked hash, and no fresh
STARTED result belongs to the llm@ worker (the same reads as the shadow
panel's live-lane gate, which live-side scripts may not import; a test runs
both on the same fixtures); and the Ollama server's /api/ps shows no model
other than the analyst's, and the analyst model, if loaded, at the live
num_ctx and not past its keep-alive expiry. /api/ps cannot show an
in-flight request by itself. The tool waits (polling, logging each new
block reason) until all are clear. It starts a pair only if all its calls
can end by --stop-at (arms x timeout), and each later call only if it can
end by then; otherwise it stops, and the unfinished pair is redone on the
next run.

Output (--out, JSONL, appended): a "run" line per start (sample, seed,
timeout, arms, model digest, Ollama version, schema hashes, tool_sha, git
head), a
"skip" line per signal that can't be replayed (no stored prompt, unknown
schema block), and a "pair" line per attempt with ids, the stored verdict,
each arm's action/conviction, raw reply (account figures redacted as in
live storage), token counts and timings, prompt hashes. Prompts themselves
are not written. Signals with a pair line are skipped on resume
(--retry-errors makes a new attempt for those whose last attempt had a
failed comparison arm: A, B or C; status covers only those, and A2's
outcome is repeat_status, never retried). The trading DB is only read, in a READ ONLY transaction.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/prompt_order_ab.py \\
        --signal-ids-file IDS --seed S [--dry-run]
    env -C ~/Desktop/trading_bot venv/bin/python scripts/prompt_order_ab.py \\
        --signal-ids-file IDS --seed S --apply --out DIR/ab.jsonl --stop-at 04:30
Then: scripts/prompt_order_ab_report.py DIR/ab.jsonl
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from unittest import mock
from zoneinfo import ZoneInfo

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prompt_ab  # noqa: E402  (schema_block and the committed pre-31d52f8 block)

ET = ZoneInfo("America/New_York")
SCHEMA_A_FILE = prompt_ab.OLD_SCHEMA_FILE
SCHEMA_B_FILE = os.path.join(REPO, "scripts", "prompt_order_ab_new_schema.txt")
# 31d52f8 after the 2026-10-31 history rewrite
# (research-log/history-rewrite-2026-10-31-commit-map.txt).
COMMIT_B = "01aecd8ccb6d94a10690ad94dc7b506d4d2cd0c1"
COMMIT_A = COMMIT_B + "^"
DEFAULT_KILL_FILE = os.path.join(REPO, "logs", ".prompt_order_ab_off")
ARM_SETS = {"AB": ("A", "B"), "ABC": ("A", "B", "C")}
REPEAT_ARM = "A2"   # optional second independent draw of arm A (noise floor)
# The live stack: helios-start.timer starts it Mon-Fri 05:00 ET,
# helios-evening-stop.timer stops it at 19:00 ET.
LIVE_DAY_START, LIVE_DAY_END = 5, 19
MAX_RUN_HOURS = 10
_BODY_SEP = "\n\n{"


class PromptShapeError(ValueError):
    """The stored prompt does not hold exactly one known schema block."""


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


TOOL_FILES = (os.path.abspath(__file__), SCHEMA_B_FILE, SCHEMA_A_FILE)


# The live analyst functions the replay reads replies with.
TOOL_FUNCS = ("_extract_json", "_conviction_distribution", "redact_account_figures")


def _analyst_sources(names=TOOL_FUNCS) -> list[str]:
    import inspect
    from agents import analyst
    return [inspect.getsource(getattr(analyst, n)) for n in names]


def tool_sha(paths=TOOL_FILES, sources=None) -> str:
    """sha256 of the concatenated bytes of this script and the two committed
    schema blocks, then the source of the analyst parsing functions in
    TOOL_FUNCS (in that order): what the report compares across resumes
    (the repo head is recorded for information only, so a deploy between
    nights doesn't split a run)."""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    for src in (_analyst_sources() if sources is None else sources):
        h.update(src.encode())
    return h.hexdigest()


def load_blocks() -> tuple[str, str]:
    with open(SCHEMA_A_FILE) as f:
        a = f.read()
    with open(SCHEMA_B_FILE) as f:
        b = f.read()
    return a, b


def block_c(block_a: str, block_b: str) -> str:
    """B's JSON body (new field order) under A's header line (no fill-order
    instruction)."""
    return block_a[:block_a.index(_BODY_SEP)] + block_b[block_b.index(_BODY_SEP):]


# ---------------------------------------------------------------------------
# Prompt transform
# ---------------------------------------------------------------------------

def build_pair(stored: str, block_a: str, block_b: str, arms=("A", "B")) -> dict:
    """The arms from one stored prompt. Finds which block the prompt holds
    (exactly once, and not the other), and swaps only that span. Asserts the
    text before and after the block is byte-identical across the arms and
    that the arm in the stored order equals the stored prompt."""
    n_a, n_b = stored.count(block_a), stored.count(block_b)
    if n_b == 1 and n_a == 0:
        native, block = "B", block_b
    elif n_a == 1 and n_b == 0:
        native, block = "A", block_a
    else:
        raise PromptShapeError(f"schema blocks found: A x{n_a}, B x{n_b} (need exactly one)")
    i = stored.index(block)
    prefix, suffix = stored[:i], stored[i + len(block):]
    blocks = {"A": block_a, "B": block_b}
    if "C" in arms:
        blocks["C"] = block_c(block_a, block_b)
    prompts = {k: prefix + v + suffix for k, v in blocks.items()}
    assert prompts[native] == stored
    for k, p in prompts.items():
        assert p[:i] == prefix and p[i + len(blocks[k]):] == suffix
    # round trip: swapping back gives the stored prompt exactly
    for k, p in prompts.items():
        assert p.replace(blocks[k], block, 1) == stored
    out = {"native": native, "block_start": i, "sha_stored": sha12(stored),
           "sha_prefix": sha12(prefix), "sha_suffix": sha12(suffix)}
    for k, p in prompts.items():
        out[k] = p
        out[f"sha_{k}"] = sha12(p)
        out[f"schema_sha_{k}"] = sha12(blocks[k])
    return out


def schema_diff(prompt_x: str, prompt_y: str, x="A", y="B") -> list[str]:
    return list(difflib.unified_diff(prompt_x.splitlines(), prompt_y.splitlines(),
                                     f"arm {x}", f"arm {y}", lineterm="", n=1))


def arm_order(seed: int, signal_id: int, arms=("A", "B")) -> tuple:
    """Random arm order per signal, fixed by (seed, signal id) so a resumed
    run gives every signal the order it would have had."""
    rng = random.Random(f"{seed}:{signal_id}")
    if tuple(arms) == ("A", "B"):
        return ("A", "B") if rng.random() < 0.5 else ("B", "A")
    order = list(arms)
    rng.shuffle(order)
    return tuple(order)


# ---------------------------------------------------------------------------
# Live call settings, read from the live code
# ---------------------------------------------------------------------------

class _Captured(Exception):
    pass


def live_payload() -> dict:
    """{url, payload, timeout} exactly as agents.analyst._call_ollama would
    send them, captured by stubbing its requests.post (nothing is sent).
    payload["prompt"] is a placeholder; callers set it per arm."""
    from agents import analyst
    seen: dict = {}

    def fake_post(url, json=None, timeout=None, **kw):
        seen.update(url=url, payload=copy.deepcopy(json), timeout=timeout)
        raise _Captured()

    quiet = analyst.log.level
    analyst.log.setLevel(logging.CRITICAL)
    try:
        with mock.patch.object(analyst.requests, "post", fake_post):
            analyst._call_ollama("<prompt>")
    finally:
        analyst.log.setLevel(quiet)
    if not seen:
        raise RuntimeError("could not capture the live analyst payload")
    seen["host"] = seen["url"].rsplit("/api/", 1)[0]
    return seen


def resolve_timeout(requested: float | None, live_timeout) -> float:
    """The per-call timeout: the live call's unless a smaller one is asked for."""
    if not isinstance(live_timeout, (int, float)) or live_timeout <= 0:
        raise ValueError(f"the live analyst call has no usable timeout ({live_timeout!r})")
    if requested is None:
        return float(live_timeout)
    if not 0 < requested <= live_timeout:
        raise ValueError(f"--timeout must be in (0, {live_timeout}] (the live analyst call's timeout)")
    return float(requested)


def arm_options(live: dict, arm: str, arm_a_num_predict: int | None = None) -> dict:
    opts = dict(live["payload"].get("options") or {})
    if arm == "A" and arm_a_num_predict:
        opts["num_predict"] = arm_a_num_predict
    return opts


# ---------------------------------------------------------------------------
# Stack-down guard
# ---------------------------------------------------------------------------

def in_live_window(now: datetime) -> bool:
    t = now.astimezone(ET)
    return t.weekday() < 5 and LIVE_DAY_START <= t.hour < LIVE_DAY_END


def next_live_start(now: datetime) -> datetime:
    """The next 05:00 ET after now (any day: the check is conservative)."""
    t = now.astimezone(ET)
    s = t.replace(hour=LIVE_DAY_START, minute=0, second=0, microsecond=0)
    return s if s > t else s + timedelta(days=1)


def helios_local_state(run=subprocess.run) -> str:
    try:
        return run(["systemctl", "is-active", "helios-local"], capture_output=True,
                   text=True, timeout=10).stdout.strip() or "unknown"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def beat_running(run=subprocess.run) -> bool | None:
    """Whether a Celery beat process for this app runs (pgrep). None when
    pgrep itself fails."""
    try:
        r = run(["pgrep", "-f", "celery -A tasks beat"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if r.returncode == 0:
        return True
    return False if r.returncode == 1 else None


def stack_down_reason(now: datetime, run=subprocess.run) -> str | None:
    """None when calls may run; else why not. Fails closed."""
    if in_live_window(now):
        return "live_hours (05:00-19:00 ET on a weekday)"
    state = helios_local_state(run)
    if state not in ("inactive", "failed"):
        return f"helios-local is {state}"
    beat = beat_running(run)
    if beat is not False:
        return "celery beat is running" if beat else "could not check for celery beat"
    return None


def stop_deadline(hhmm: str, now: datetime) -> datetime:
    """The next HH:MM ET after now. Refused when more than MAX_RUN_HOURS away
    or later than the next 05:00 ET."""
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", hhmm or "")
    if not m:
        raise ValueError(f"--stop-at must be HH:MM, not {hhmm!r}")
    now_et = now.astimezone(ET)
    t = now_et.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    if t <= now_et:
        t += timedelta(days=1)
    if t - now_et > timedelta(hours=MAX_RUN_HOURS):
        raise ValueError(f"--stop-at {hhmm} is {t - now_et} away (more than {MAX_RUN_HOURS} h)")
    if t > next_live_start(now):
        raise ValueError(f"--stop-at {hhmm} is after the next {LIVE_DAY_START:02d}:00 ET")
    return t


# ---------------------------------------------------------------------------
# Queue gate
# ---------------------------------------------------------------------------

def _parse_ts(v):
    if not isinstance(v, str) or not v:
        return None
    s = re.sub(r"(\.\d{6})\d+", r"\1", v.strip()).replace("Z", "+00:00")
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def ollama_ps_state(host: str, model: str, num_ctx: int, get=None, now=None) -> tuple[bool, str, dict]:
    """Whether the Ollama server looks idle for this model, from /api/ps."""
    import requests
    get = get or requests.get
    now = now or datetime.now(timezone.utc)
    try:
        resp = get(f"{host}/api/ps", timeout=(2, 5))
        resp.raise_for_status()
        models = resp.json().get("models") or []
    except Exception as e:
        return False, "ollama_ps_unavailable", {"error": type(e).__name__}
    names = [m.get("name") or m.get("model") for m in models]
    other = sorted(n for n in names if n != model)
    if other:
        return False, "other_model_loaded", {"loaded": names}
    for m in models:
        ctx = m.get("context_length")
        if ctx not in (None, 0) and int(ctx) != int(num_ctx):
            return False, "loaded_at_other_num_ctx", {"context_length": ctx}
        exp = _parse_ts(m.get("expires_at"))
        if exp is not None and exp < now:
            return False, "loaded_past_expiry", {"expires_at": m.get("expires_at")}
    return True, "clear", {"loaded": names}


LIVE_LLM_QUEUES = ("llm_p0", "llm_p1", "llm_p2")   # tasks.LLM_QUEUE_*
LIVE_LLM_WORKER_PREFIX = "llm@"                      # process_manager: -n llm@%h
LIVE_TASK_MAX_AGE_S = 1800      # an older STARTED entry is a killed worker's leftover
MAX_KEYS = 50000                # read no more than this many keys; fail closed


def _celery_result_expires_s() -> float:
    """Celery's default result TTL; tasks.py doesn't set result_expires (a
    test checks tasks.app.conf against this)."""
    try:
        from celery.app.defaults import DEFAULTS
        v = DEFAULTS["result_expires"]
        return float(v.total_seconds() if hasattr(v, "total_seconds") else v)
    except Exception:
        return 86400.0


RESULT_EXPIRES_S = _celery_result_expires_s()


def _s(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)


class LiveLane:
    """Whether the live Celery LLM lane is idle, from read-only Redis reads.
    check() -> (ok, reason, detail). Redis unreachable counts as not clear."""

    def __init__(self, redis_url: str, kill_file: str, redis_client=None):
        self.redis_url, self.kill_file, self._r = redis_url, kill_file, redis_client

    def redis(self):
        if self._r is None:
            import redis
            self._r = redis.Redis.from_url(self.redis_url, socket_timeout=3, socket_connect_timeout=3)
        return self._r

    def check(self) -> tuple[bool, str, dict]:
        if os.path.exists(self.kill_file):
            return False, "kill_file", {"path": self.kill_file}
        try:
            r = self.redis()
            keys = [k for q in LIVE_LLM_QUEUES for k in [q] + [f"{q}\x06\x16{p}" for p in (3, 6, 9)]]
            queued = {k: int(n) for k in keys for n in [r.llen(k)] if int(n or 0) > 0}
            if queued:
                return False, "live_queue_not_empty", {"queued": queued}
            reserved = seen = 0
            for _tag, raw in r.hscan_iter("unacked", count=500):
                seen += 1
                if seen > MAX_KEYS:
                    return False, "too_many_keys", {}
                try:
                    if _s(json.loads(_s(raw))[2]) in LIVE_LLM_QUEUES:
                        reserved += 1
                except (ValueError, IndexError, TypeError):
                    continue
            if reserved:
                return False, "live_task_reserved", {"reserved": reserved}
            running = stale = seen = 0
            cursor = 0
            while True:
                cursor, ks = r.scan(cursor=cursor, match="celery-task-meta-*", count=1000)
                seen += len(ks)
                if seen > MAX_KEYS:
                    return False, "too_many_keys", {}
                for k, raw in zip(ks, r.mget(ks) if ks else []):
                    text = _s(raw) if raw is not None else ""
                    if "STARTED" not in text:
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
                    ttl = r.ttl(k)
                    age = RESULT_EXPIRES_S - ttl if ttl is not None and ttl >= 0 else None
                    if age is not None and age > LIVE_TASK_MAX_AGE_S:
                        stale += 1
                    else:
                        running += 1
                if int(cursor) == 0:
                    break
            if running:
                return False, "live_task_running", {"running": running, "stale": stale}
        except Exception as e:
            return False, "redis_unavailable", {"error": type(e).__name__}
        return True, "clear", {"stale_started": stale} if stale else {}


class QueueGate:
    """LiveLane (Redis, read-only) plus /api/ps, polled until clear."""

    def __init__(self, lane, host: str, model: str, num_ctx: int, poll_s: float = 15.0,
                 ps_check=None, clock=time.time, sleep=time.sleep, log=print):
        self.lane = lane
        self.host, self.model, self.num_ctx = host, model, num_ctx
        self.poll_s = poll_s
        self.ps_check = ps_check or (lambda: ollama_ps_state(self.host, self.model, self.num_ctx))
        self.clock, self.sleep, self.log = clock, sleep, log

    def check(self) -> tuple[bool, str, dict]:
        ok, reason, detail = self.lane.check()
        if not ok:
            return False, reason, detail
        return self.ps_check()

    def wait(self, give_up_at: float) -> dict:
        """Poll until clear. Returns {ok, reason, detail, waited_s}; not ok
        when give_up_at (epoch seconds) passes first or the kill file is set.
        Each new block reason is logged once."""
        t0 = self.clock()
        last = None
        while True:
            ok, reason, detail = self.check()
            out = {"ok": ok, "reason": reason, "detail": detail, "waited_s": round(self.clock() - t0, 1)}
            if not ok and reason != last:
                self.log(f"gate blocked: {reason} {detail}")
                last = reason
            if ok or reason == "kill_file":
                return out
            if self.clock() + self.poll_s > give_up_at:
                out["reason"] = f"gave_up_waiting ({reason})"
                return out
            self.sleep(self.poll_s)


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------

def parse_reply(raw: str | None) -> dict:
    """The model's own action and conviction, read as live reads them
    (agents.analyst._extract_json; conviction clamped to 1-10)."""
    from agents import analyst
    parsed = analyst._extract_json(raw) if raw else None
    if not parsed:
        return {"parse_ok": False, "salvaged": False, "action": None, "conviction": None,
                "conviction_raw": None}
    salvaged = bool(parsed.pop("_salvaged", False))
    raw_c = parsed.get("conviction")
    try:
        conv = max(1, min(10, int(raw_c or 1))) if raw_c is not None else None
    except (TypeError, ValueError):
        conv = None
    act = parsed.get("action")
    return {"parse_ok": True, "salvaged": salvaged,
            "action": str(act).upper() if act is not None else None,
            "conviction": conv, "conviction_raw": raw_c,
            "sub_scores": {k: parsed.get(k) for k in ("thesis_strength", "catalyst_quality",
                                                      "valuation_risk_reward", "data_completeness")}}


def call_arm(live: dict, prompt: str, timeout_s: float, options: dict | None = None,
             post=None, clock=time.monotonic) -> dict:
    import requests
    from agents import analyst
    post = post or requests.post
    payload = dict(live["payload"], prompt=prompt)
    if options is not None:
        payload["options"] = options
    t0 = clock()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: dict = {"started_at": started, "error": None, "num_predict": payload.get("options", {}).get("num_predict")}
    try:
        resp = post(f"{live['host']}/api/generate", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        body = resp.json()
    except requests.exceptions.Timeout:
        out.update(error=f"timeout after {timeout_s:.0f} s", wall_s=round(clock() - t0, 1))
        return out
    except Exception as e:
        out.update(error=type(e).__name__, wall_s=round(clock() - t0, 1))
        return out
    out["wall_s"] = round(clock() - t0, 1)
    raw = body.get("response") or ""
    out.update({k: body.get(k) for k in ("done_reason", "eval_count", "prompt_eval_count",
                                         "total_duration", "load_duration", "prompt_eval_duration",
                                         "eval_duration")})
    try:
        out["conviction_logprobs"] = analyst._conviction_distribution(body.get("logprobs"))
    except Exception:
        out["conviction_logprobs"] = None
    out["raw_sha"] = sha12(raw)
    out["raw"] = analyst.redact_account_figures(raw)
    out.update(parse_reply(raw))
    return out


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

def _records(path: str):
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue   # a line cut by a crash; that pair is redone


def file_state(path: str) -> dict:
    """{last pair status per signal, attempts per signal, skipped ids}."""
    last, attempts, skipped = {}, {}, set()
    for rec in _records(path):
        sid = rec.get("signal_id")
        if sid is None:
            continue
        if rec.get("kind") == "pair":
            last[sid] = rec.get("status")
            attempts[sid] = attempts.get(sid, 0) + 1
        elif rec.get("kind") == "skip":
            skipped.add(sid)
    return {"last": last, "attempts": attempts, "skipped": skipped}


def done_ids(path: str, retry_errors: bool = False) -> set:
    """Signal ids with a pair line in the file. With retry_errors, a signal
    whose last line has status error is not counted as done."""
    last = file_state(path)["last"]
    return {s for s, st in last.items() if not (retry_errors and st != "ok")}


def _append(path: str, rec: dict) -> None:
    lead = ""
    try:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            lead = "" if f.read(1) == b"\n" else "\n"   # a line cut by a crash stays its own line
    except OSError:
        pass   # missing or empty file
    with open(path, "a") as f:
        f.write(lead + json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Sample and stored rows (DB, read only)
# ---------------------------------------------------------------------------

def read_ids_file(path: str) -> list[int]:
    with open(path) as f:
        text = f.read()
    try:
        vals = json.loads(text)
        if not isinstance(vals, list):
            raise ValueError
    except ValueError:
        vals = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    out, seen = [], set()
    for v in vals:
        if v in ("", None):
            continue
        i = int(v)
        if i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise SystemExit(f"{path}: no signal ids")
    return out


def draw_sample(population: list[int], n: int, seed: int) -> list[int]:
    pop = sorted(population)
    if n > len(pop):
        raise SystemExit(f"--sample {n} is larger than the {len(pop)} rows with a stored prompt in range")
    return random.Random(seed).sample(pop, n)


def _et(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=ET)


def _readonly(sql: str, params) -> list[tuple]:
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.rollback()
        return rows
    finally:
        db._put_conn(conn)


def population_ids(start: datetime, end: datetime, trading_env: str) -> list[int]:
    rows = _readonly("SELECT id FROM signals WHERE prompt IS NOT NULL AND trading_env=%s "
                     "AND timestamp >= %s AND timestamp < %s ORDER BY id", (trading_env, start, end))
    return [r[0] for r in rows]


def fetch_signals(ids: list[int]) -> dict:
    rows = _readonly(
        "SELECT id, symbol, timestamp, trading_env, signal, prompt, "
        "provenance->>'prompt_sha', provenance->>'model_digest', provenance->>'code_sha', "
        "decision->>'model_action', decision->>'model_conviction' "
        "FROM signals WHERE id = ANY(%s)", (list(ids),))
    return {r[0]: {"signal_id": r[0], "ticker": r[1], "signal_ts": r[2], "trading_env": r[3],
                   "live_signal": r[4], "prompt": r[5], "provenance_prompt_sha": r[6],
                   "provenance_model_digest": r[7], "provenance_code_sha": r[8],
                   "live_model_action": r[9], "live_model_conviction": r[10]} for r in rows}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_pairs(signals: list[dict], *, live: dict, blocks: tuple[str, str], seed: int, out: str,
              timeout_s: float, deadline: datetime, gate: QueueGate, call=call_arm,
              now=lambda: datetime.now(timezone.utc), retry_errors: bool = False,
              run_model_digest: str | None = None, arms=("A", "B"),
              arm_a_num_predict: int | None = None, run_id: str | None = None,
              repeat_ids=frozenset(), stack_guard=None, log=print) -> dict:
    """Calls every arm per signal in its seeded order, appending one pair
    line per attempt and one skip line per unusable signal. For a signal in
    repeat_ids, arm A2 (a second independent call with arm A's prompt and
    options) joins the shuffled order. Returns counts and why it stopped.
    stack_guard(now) -> None or the reason to stop."""
    state = file_state(out)
    done = {s for s, st in state["last"].items() if not (retry_errors and st != "ok")}
    stats = {"written": 0, "skipped_done": 0, "skipped_shape": 0, "missing": 0, "stopped": None}
    for sig in signals:
        sid = sig["signal_id"]
        if sid in done:
            stats["skipped_done"] += 1
            continue
        reason = None
        if not sig.get("prompt"):
            stats["missing"] += 1
            reason = "no_stored_prompt"
        else:
            try:
                pair = build_pair(sig["prompt"], *blocks, arms=arms)
            except PromptShapeError as e:
                stats["skipped_shape"] += 1
                reason = f"schema_block: {e}"
        if reason:
            log(f"{sid}: skipped ({reason})")
            if sid not in state["skipped"]:
                _append(out, {"kind": "skip", "signal_id": sid, "reason": reason, "run_id": run_id})
                state["skipped"].add(sid)
            continue
        if stack_guard is not None:
            why = stack_guard(now())
            if why:
                stats["stopped"] = why
                log(f"stopping: {why}")
                break
        sig_arms = tuple(arms) + ((REPEAT_ARM,) if sid in repeat_ids else ())
        order = arm_order(seed, sid, sig_arms)
        if now() + timedelta(seconds=len(order) * timeout_s) > deadline:
            stats["stopped"] = "stop_at"
            break
        results: dict = {}
        for pos, arm in enumerate(order, start=1):
            remaining_calls = len(order) - pos + 1
            give_up = (deadline - timedelta(seconds=remaining_calls * timeout_s)).timestamp()
            g = gate.wait(give_up)
            if not g["ok"]:
                stats["stopped"] = g["reason"]
                log(f"stopping: {g['reason']}")
                break
            if now() + timedelta(seconds=remaining_calls * timeout_s) > deadline:
                stats["stopped"] = "stop_at"
                break
            base = "A" if arm == REPEAT_ARM else arm
            res = call(live, pair[base], timeout_s, options=arm_options(live, base, arm_a_num_predict))
            res.update(position=pos, gate_waited_s=g["waited_s"])
            results[arm] = res
            log(f"{sid} {sig.get('ticker')} arm {arm}: action={res.get('action')} "
                f"conviction={res.get('conviction')} {res.get('wall_s')} s"
                + (f" error={res['error']}" if res.get("error") else ""))
        if len(results) < len(order):
            break   # stopped mid-pair: nothing written, redone on resume
        # status covers the comparison arms only; A2's outcome is repeat_status
        # and never makes --retry-errors redraw the pair.
        status = "ok" if all(not a.get("error") for k, a in results.items() if k != REPEAT_ARM) else "error"
        repeat_status = (("ok" if not results[REPEAT_ARM].get("error") else "error")
                         if REPEAT_ARM in results else None)
        prov_sha = sig.get("provenance_prompt_sha")
        attempt = state["attempts"].get(sid, 0) + 1
        state["attempts"][sid] = attempt
        rec = {"kind": "pair", "status": status, "signal_id": sid, "attempt": attempt, "run_id": run_id,
               "ticker": sig.get("ticker"),
               "signal_ts": sig.get("signal_ts"), "trading_env": sig.get("trading_env"),
               "live_signal": sig.get("live_signal"),
               "live_model_action": sig.get("live_model_action"),
               "live_model_conviction": sig.get("live_model_conviction"),
               "provenance_prompt_sha": prov_sha,
               # False when the stored copy was redacted (account figures) or
               # otherwise differs from what the model saw live.
               "stored_matches_provenance_sha": (pair["sha_stored"] == prov_sha) if prov_sha else None,
               "provenance_model_digest": sig.get("provenance_model_digest"),
               "provenance_code_sha": sig.get("provenance_code_sha"),
               "run_model_digest": run_model_digest,
               "native_order": pair["native"], "order": list(order), "seed": seed,
               **{k: pair[k] for k in ("sha_stored", "sha_prefix", "sha_suffix")},
               **{f"{p}_{a}": pair[f"{p}_{a}"] for a in arms for p in ("sha", "schema_sha")},
               "model": live["payload"].get("model"), "options": live["payload"].get("options"),
               "arm_a_num_predict": arm_a_num_predict,
               "repeat_arm": REPEAT_ARM if REPEAT_ARM in sig_arms else None,
               "repeat_status": repeat_status,
               "timeout_s": timeout_s, "arms": results,
               "written_at": now().isoformat(timespec="seconds")}
        _append(out, rec)
        stats["written"] += 1
    return stats


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--signal-ids-file")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--trading-env")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, required=True, help="seeds the sample draw and the arm order")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--out", help="JSONL output path (appended; resumable)")
    ap.add_argument("--stop-at", help="HH:MM ET: no call may run past it (required with --apply; "
                                      f"at most {MAX_RUN_HOURS} h away and before the next "
                                      f"{LIVE_DAY_START:02d}:00 ET)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per call; default and maximum: the live analyst call's timeout")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="AB",
                    help="AB (default) or ABC: C = the new order without the fill-order instruction")
    ap.add_argument("--arm-a-num-predict", type=int, choices=(1200, 1600), default=None,
                    help="num_predict for arm A (default: the live value, as for the other arms)")
    ap.add_argument("--repeat-arm", choices=(REPEAT_ARM,), default=None,
                    help="A2: a second independent draw of arm A, as a noise floor (default off)")
    ap.add_argument("--repeat-subsample", type=int, default=None,
                    help="run A2 on only N signals of the sample (seeded by --seed)")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between queue checks while waiting")
    ap.add_argument("--kill-file", default=DEFAULT_KILL_FILE)
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args(argv)
    by_file = bool(args.signal_ids_file)
    by_range = any(v is not None for v in (args.date_from, args.date_to, args.sample, args.trading_env))
    if by_file == by_range:
        ap.error("give the sample: either --signal-ids-file, or --from --to --trading-env --sample")
    if by_range and None in (args.date_from, args.date_to, args.sample, args.trading_env):
        ap.error("--from, --to, --trading-env and --sample all go together")
    if args.apply:
        if not args.out:
            ap.error("--apply needs --out")
        if not args.stop_at:
            ap.error("--apply needs --stop-at HH:MM (ET)")
        try:
            stop_deadline(args.stop_at, datetime.now(timezone.utc))
        except ValueError as e:
            ap.error(str(e))
    if args.timeout is not None and not args.timeout > 0:
        ap.error("--timeout must be positive")
    if args.repeat_subsample is not None and (not args.repeat_arm or args.repeat_subsample < 1):
        ap.error("--repeat-subsample needs --repeat-arm A2 and a positive N")
    return args


def repeat_ids_for(ids: list[int], repeat_arm: str | None, subsample: int | None, seed: int) -> frozenset:
    """The signals that get arm A2: none when off, all, or a seeded N of
    them (fixed by seed and the sample, so a resume picks the same ones)."""
    if not repeat_arm:
        return frozenset()
    if subsample is None:
        return frozenset(ids)
    pool = sorted(set(ids))
    if subsample > len(pool):
        raise SystemExit(f"--repeat-subsample {subsample} is larger than the sample ({len(pool)})")
    return frozenset(random.Random(f"{seed}:repeat").sample(pool, subsample))


def main(argv=None):
    args = parse_args(argv)
    arms = ARM_SETS[args.arms]
    if args.apply:
        why = stack_down_reason(datetime.now(timezone.utc))
        if why:
            sys.exit(f"refusing to start: {why}")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    blocks = load_blocks()
    live = live_payload()
    try:
        args.timeout = resolve_timeout(args.timeout, live["timeout"])
    except ValueError as e:
        sys.exit(str(e))

    from agents import analyst
    current_b = prompt_ab.schema_block(open(os.path.join(REPO, "agents", "analyst.py")).read())
    if current_b != blocks[1]:
        sys.exit("the live analyst schema block differs from arm B's committed block "
                 f"({SCHEMA_B_FILE}); not run")

    if args.signal_ids_file:
        ids = read_ids_file(args.signal_ids_file)
        sample = {"mode": "ids_file", "file": os.path.abspath(args.signal_ids_file)}
    else:
        start, end = _et(args.date_from), _et(args.date_to)
        pop = population_ids(start, end, args.trading_env)
        ids = draw_sample(pop, args.sample, args.seed)
        sample = {"mode": "range", "from": start.isoformat(), "to": end.isoformat(),
                  "trading_env": args.trading_env, "population": len(pop), "n": args.sample}
    repeat_ids = repeat_ids_for(ids, args.repeat_arm, args.repeat_subsample, args.seed)
    rows = fetch_signals(ids)
    signals = [rows.get(i, {"signal_id": i, "prompt": None}) for i in ids]
    print(f"{len(ids)} signal ids; {sum(1 for s in signals if s.get('prompt'))} with a stored prompt")
    print(f"live call: {live['host']} model={live['payload']['model']} options={live['payload']['options']} "
          f"logprobs={live['payload'].get('logprobs')} top_logprobs={live['payload'].get('top_logprobs')} "
          f"live timeout={live['timeout']} s, this run {args.timeout} s, arms {''.join(arms)}, "
          f"arm A num_predict {args.arm_a_num_predict or 'live'}, "
          f"repeat arm {args.repeat_arm or 'off'} on {len(repeat_ids)} signals")

    if not args.apply:
        shown = False
        for s in signals:
            if not s.get("prompt"):
                print(f"{s['signal_id']}: no stored prompt")
                continue
            try:
                p = build_pair(s["prompt"], *blocks, arms=arms)
            except PromptShapeError as e:
                print(f"{s['signal_id']} {s.get('ticker')}: skipped ({e})")
                continue
            prov = s.get("provenance_prompt_sha")
            print(f"{s['signal_id']} {s.get('ticker')} {s.get('signal_ts')}: native={p['native']} "
                  f"stored={p['sha_stored']} " + " ".join(f"{a}={p['sha_' + a]}" for a in arms)
                  + f" prefix={p['sha_prefix']} suffix={p['sha_suffix']} "
                  f"order={'/'.join(arm_order(args.seed, s['signal_id'], arms + ((REPEAT_ARM,) if s['signal_id'] in repeat_ids else ())))} "
                  f"stored==provenance_sha={p['sha_stored'] == prov if prov else None}")
            if not shown:
                print("\n".join(schema_diff(p["A"], p["B"])))
                if "C" in arms:
                    print("\n".join(schema_diff(p["A"], p["C"], "A", "C")))
                shown = True
        print("dry run: no model called")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = datetime.now(timezone.utc)
    deadline = stop_deadline(args.stop_at, started)
    ident = analyst._ollama_identity()
    run_id = f"{started.isoformat(timespec='seconds')}/{os.getpid()}"
    sha_c = sha12(block_c(*blocks)) if "C" in arms else None
    _append(args.out, {"kind": "run", "run_id": run_id, "started_at": started.isoformat(timespec="seconds"),
                       "sample": sample, "signal_ids": ids, "seed": args.seed,
                       "stop_at": deadline.isoformat(), "timeout_s": args.timeout,
                       "arms": "".join(arms), "arm_a_num_predict": args.arm_a_num_predict,
                       "repeat_arm": args.repeat_arm, "repeat_subsample": args.repeat_subsample,
                       "repeat_ids": sorted(repeat_ids),
                       "model": live["payload"]["model"], "options": live["payload"]["options"],
                       "logprobs": live["payload"].get("logprobs"),
                       "top_logprobs": live["payload"].get("top_logprobs"),
                       "live_timeout_s": live["timeout"], "ollama_identity": ident,
                       "model_digest": ident.get("model_digest"),
                       "ollama_version": ident.get("ollama_version"),
                       "schema_sha_A": sha12(blocks[0]), "schema_sha_B": sha12(blocks[1]),
                       "schema_sha_C": sha_c,
                       "commit_A": COMMIT_A, "commit_B": COMMIT_B, "tool_sha": tool_sha(),
                       "head": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip()})
    redis_url = os.getenv("REDIS_URL") or "redis://localhost:6379/0"   # the live broker (tasks.py)
    gate = QueueGate(LiveLane(redis_url, args.kill_file), live["host"], live["payload"]["model"],
                     live["payload"]["options"]["num_ctx"], poll_s=args.poll)
    stats = run_pairs(signals, live=live, blocks=blocks, seed=args.seed, out=args.out,
                      timeout_s=args.timeout, deadline=deadline, gate=gate,
                      retry_errors=args.retry_errors, run_model_digest=ident.get("model_digest"),
                      arms=arms, arm_a_num_predict=args.arm_a_num_predict, run_id=run_id,
                      repeat_ids=repeat_ids, stack_guard=stack_down_reason)
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
