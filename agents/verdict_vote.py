"""Shadow self-consistency vote (config.VERDICT_VOTE_SHADOW_ENABLED), record
only. A routed OPEN/ADD (pass_to_risk True) gets 2 more analyst draws on the
same stored prompt, with the same model and options, after its routing. One
activity_feed row, stage verdict_vote_shadow, records the 3 draws and their
vote. Nothing reads it back: routing, approval and the order go on from the
first draw as before.

Input. signals.prompt holds the rendered prompt with the account's dollar
figures redacted, and provenance.prompt_sha is the hash of the unredacted
prompt. A draw runs only when sha256(stored prompt)[:12] equals that hash,
so the text is byte-exact. Otherwise the row says prompt_not_exact and no
call is made. The brief is never rebuilt.

GPU. Each draw is its own task on llm_p3, the last queue the llm worker
drains (queue_order_strategy=sorted). Before its call, a draw checks the live
LLM queues (llm_p0..p2, with kombu's priority sub-queues) and kombu's unacked
hash for a message the worker reserved from one, read-only (the same reads
as the research panel's live-lane gate). When live work is waiting, it re-queues itself after
_BUSY_RETRY_S, up to _BUSY_MAX_TRIES times, then writes a gave_up_busy row.
A call is never preempted, so a live analysis that arrives while a draw runs
still waits for it, up to one call (the 400 s timeout). That is why the
switch ships off.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from collections import Counter
from datetime import datetime, timezone

log = logging.getLogger(__name__)

STAGE = "verdict_vote_shadow"
QUEUE = "llm_p3"                 # tasks.SHADOW_LLM_QUEUE; sorts after llm_p2
LIVE_LLM_QUEUES = ("llm_p0", "llm_p1", "llm_p2")   # tasks.LLM_QUEUE_*
_KOMBU_PRIORITY_KEYS = ("", "\x06\x163", "\x06\x166", "\x06\x169")   # kombu sub-queues
_UNACKED_MAX = 50000
EXTRA_DRAWS = 2
_BUY_ACTIONS = ("OPEN", "ADD")
_TASK_EXPIRES_S = 6 * 3600
_BUSY_RETRY_S = 120
_BUSY_MAX_TRIES = 90             # about 3 h of a busy lane, then gave_up_busy


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "VERDICT_VOTE_SHADOW_ENABLED", False))
    except Exception:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(v):
    """A finite float, else None (no bools, no strings that aren't numbers)."""
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _action(v):
    return (str(v).strip().upper() or None) if v is not None else None


# ---------------------------------------------------------------- enqueue ---

def _enqueue(payload: dict, countdown: int | None = None) -> None:
    from tasks import verdict_vote_draw
    kw = {"countdown": countdown} if countdown else {}
    verdict_vote_draw.apply_async(args=[payload], queue=QUEUE, expires=_TASK_EXPIRES_S,
                                  retry=False, **kw)


def observe_routed(result: dict) -> dict | None:
    """Called by crew after a verdict's routing returned. Enqueues the first
    re-draw for a pass_to_risk OPEN/ADD with a stored signal row. Only an
    enqueue; returns the payload or None. Never raises."""
    try:
        if not _enabled() or not result.get("pass_to_risk") \
                or result.get("action") not in _BUY_ACTIONS:
            return None
        prov = result.get("provenance")
        prov = prov if isinstance(prov, dict) else {}
        signal_id = prov.get("signal_id")
        if not isinstance(signal_id, int) or isinstance(signal_id, bool):
            log.info(f"[{result.get('ticker')}] verdict vote shadow skipped: no signal row")
            return None
        payload = {"signal_id": signal_id, "ticker": result.get("ticker"),
                   "action": result.get("action"), "conviction": result.get("conviction"),
                   "udr": result.get("upside_downside_ratio"),
                   "prompt_sha": prov.get("prompt_sha"), "queued_at": _now(),
                   "draws": [], "busy_tries": 0}
        _enqueue(payload)
        return payload
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] verdict vote shadow enqueue failed: {e}")
        return None


# ------------------------------------------------------------------ reads ---

def _fetch_signal(signal_id: int) -> dict | None:
    """The verdict's stored prompt, prompt_sha, model output and call record,
    read in a READ ONLY transaction. None when there is no such row."""
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SELECT prompt, provenance->>'prompt_sha', decision->'model_output', "
                        "provenance->'llm_call', provenance->>'model_digest' "
                        "FROM signals WHERE id = %s", (signal_id,))
            row = cur.fetchone()
        conn.rollback()
    finally:
        db._put_conn(conn)
    if not row:
        return None
    prompt, sha, model_output, llm_call, digest = row
    return {"prompt": prompt, "prompt_sha": sha,
            "model_output": model_output if isinstance(model_output, dict) else {},
            "llm_call": llm_call if isinstance(llm_call, dict) else {}, "model_digest": digest}


def prompt_is_exact(prompt, prompt_sha) -> bool:
    return (isinstance(prompt, str) and isinstance(prompt_sha, str) and bool(prompt_sha)
            and hashlib.sha256(prompt.encode()).hexdigest()[:len(prompt_sha)] == prompt_sha
            and len(prompt_sha) >= 12)


def _lane_busy() -> dict | None:
    """What is waiting in the live LLM lane (llm_p0..p2 queued, or reserved
    by the worker), or None when it is clear. Read-only. A read error counts
    as busy. The running-task check is left out: it would see this draw."""
    try:
        r = _redis()
        q = {k: int(n) for k in (q + p for q in LIVE_LLM_QUEUES for p in _KOMBU_PRIORITY_KEYS)
             for n in [r.llen(k)] if int(n or 0) > 0}
        if q:
            return {"queued": q}
        n = seen = 0
        for _tag, raw in r.hscan_iter("unacked", count=500):
            seen += 1
            if seen > _UNACKED_MAX:
                return {"error": "too_many_unacked"}
            try:
                rk = json.loads(raw.decode() if isinstance(raw, bytes) else raw)[2]
            except (ValueError, IndexError, TypeError, AttributeError):
                continue
            if (rk.decode() if isinstance(rk, bytes) else rk) in LIVE_LLM_QUEUES:
                n += 1
        return {"reserved": n} if n else None
    except Exception as e:
        return {"error": type(e).__name__}


def _redis():
    import os
    import redis
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                socket_timeout=3, socket_connect_timeout=3)


# ------------------------------------------------------------------ draws ---

def _raw(v):
    """A JSON-safe copy of a field as the model wrote it."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    return str(v)[:50]


def _model_draw(parsed: dict | None) -> dict:
    """The model's own fields from one reply. pass_to_risk is read as live
    reads it (bool(parsed.get("pass_to_risk")), analyst._analyze_inner), and
    kept as written in model_pass_raw. All None for a missing reply."""
    if not isinstance(parsed, dict) or not parsed:
        return {"action": None, "conviction": None, "udr": None,
                "model_pass": None, "model_pass_raw": None}
    return {"action": _action(parsed.get("action")),
            "conviction": _num(parsed.get("conviction")),
            "udr": _num(parsed.get("upside_downside_ratio")),
            "model_pass": bool(parsed.get("pass_to_risk", False)),
            "model_pass_raw": _raw(parsed.get("pass_to_risk"))}


def _draw(prompt: str) -> dict:
    """One analyst call on the stored prompt: the model's own action,
    conviction, UDR and pass_to_risk, before any gate."""
    from agents import analyst
    raw = analyst._call_ollama(prompt)
    stats = dict(analyst._LAST_CALL_STATS)
    out = {"at": _now(), "eval_count": stats.get("eval_count"),
           "total_duration": stats.get("total_duration"),
           "done_reason": stats.get("done_reason"),
           "model_digest": analyst._ollama_identity().get("model_digest"), "parse_mode": None}
    if not raw:
        return {**_model_draw(None), **out,
                "error": analyst._LAST_CALL_FAILURE.get("kind") or "empty_reply"}
    parsed = analyst._extract_json(raw)
    if not parsed:
        return {**_model_draw(None), **out, "error": "parse_failed"}
    out["parse_mode"] = analyst._reply_parse_mode(raw, parsed)
    return {**_model_draw(parsed), **out, "error": None}


def vote(draws: list[dict]) -> dict:
    """The vote over the draws (the first draw first). Lists keep draw order,
    with None for a draw without the field. The majority action (2+ draws,
    else None), majority_count, differ_from_first and buy_votes count only
    draws with an action; differ_from_first is None when the first has none.
    Medians and spreads are over the draws with a number. The majority of the
    model's own pass_to_risk: 2+ draws True or False, else None.
    would_pass_bar stays None: what counts as the majority still passing the
    system's entry bar is a readout definition not made here, and it can be
    computed later from the stored draws."""
    actions = [d.get("action") for d in draws]
    convs = [d.get("conviction") for d in draws]
    udrs = [d.get("udr") for d in draws]
    valid = [a for a in actions if a]
    counts = Counter(valid)
    top = counts.most_common(1)
    majority = top[0][0] if top and top[0][1] >= 2 else None
    passes = [d.get("model_pass") for d in draws]
    n_true, n_false = passes.count(True), passes.count(False)
    cv = [c for c in convs if c is not None]
    uv = [u for u in udrs if u is not None]
    first = actions[0] if actions else None
    return {"actions": actions, "convictions": convs, "udrs": udrs,
            "majority_action": majority,
            "majority_count": top[0][1] if top else 0,
            "median_conviction": statistics.median(cv) if cv else None,
            "median_udr": statistics.median(uv) if uv else None,
            "conviction_spread": (max(cv) - min(cv)) if cv else None,
            "udr_spread": round(max(uv) - min(uv), 4) if uv else None,
            "buy_votes": sum(1 for a in valid if a in _BUY_ACTIONS),
            "differ_from_first": (sum(1 for a in actions[1:] if a and a != first)
                                  if first else None),
            "model_passes": passes,
            "model_pass_majority": True if n_true >= 2 else False if n_false >= 2 else None,
            "would_pass_bar": None,
            "n_valid": sum(1 for d in draws if not d.get("error")),
            "n_valid_actions": len(valid)}


def _write(payload: dict, status: str, first: dict | None = None, extra: dict | None = None,
           first_options: dict | None = None) -> bool:
    """One verdict_vote_shadow row. Every status writes the same keys, null
    where they don't apply."""
    import db
    import config
    from agents import analyst
    draws = ([first] if first else []) + list(payload.get("draws") or [])
    options = analyst._ollama_options()
    rec = {"signal_id": payload.get("signal_id"), "status": status,
           "env": getattr(config, "TRADING_ENV", None),
           "routed": {"action": payload.get("action"), "conviction": payload.get("conviction"),
                      "udr": payload.get("udr")},
           "draws": draws, **vote(draws),
           "prompt_sha": payload.get("prompt_sha"), "model": analyst.ANALYST_MODEL,
           "options": options, "first_options": first_options,
           "options_match": (first_options == options) if isinstance(first_options, dict) else None,
           "busy_tries": payload.get("busy_tries", 0), "lane": None, "error": None,
           "queued_at": payload.get("queued_at"), "written_at": _now(),
           **analyst.shadow_versions()}
    rec.update({k: v for k, v in (extra or {}).items() if k in ("lane", "error")})
    v = rec.get("actions")
    msg = (f"Vote shadow {status}: draws {v}, majority {rec.get('majority_action')}"
           if v else f"Vote shadow {status}")
    return db.log_activity(STAGE, payload.get("ticker") or "", msg, "neutral", extra=rec)


def run(payload: dict, lane_busy=None, fetch=None, draw=None, enqueue=None, write=None) -> str:
    """One re-draw task. Returns what it did: disabled, no_signal,
    prompt_not_exact, requeued_busy, gave_up_busy, next_draw, written or
    error. The lane is checked last, right before the call. Never raises
    (the task's time limit aside)."""
    lane_busy, fetch = lane_busy or _lane_busy, fetch or _fetch_signal
    draw, enqueue, write = draw or _draw, enqueue or _enqueue, write or _write
    ticker = payload.get("ticker")
    first = first_options = None
    try:
        if not _enabled():
            return "disabled"
        sig = fetch(payload["signal_id"])
        if sig is None:
            write(payload, "no_signal")
            return "no_signal"
        first = {**_model_draw(sig["model_output"]), "first": True,
                 "at": None, "eval_count": sig["llm_call"].get("eval_count"),
                 "total_duration": sig["llm_call"].get("total_duration"),
                 "done_reason": sig["llm_call"].get("done_reason"),
                 "model_digest": sig.get("model_digest"), "parse_mode": None, "error": None}
        first_options = sig["llm_call"].get("options")
        first_options = first_options if isinstance(first_options, dict) else None
        if not prompt_is_exact(sig["prompt"], sig["prompt_sha"]) \
                or sig["prompt_sha"] != payload.get("prompt_sha"):
            write(payload, "prompt_not_exact", first, first_options=first_options)
            return "prompt_not_exact"
        busy = lane_busy()
        if busy:
            payload = {**payload, "busy_tries": int(payload.get("busy_tries") or 0) + 1}
            if payload["busy_tries"] > _BUSY_MAX_TRIES:
                write(payload, "gave_up_busy", first, extra={"lane": busy},
                      first_options=first_options)
                return "gave_up_busy"
            enqueue(payload, countdown=_BUSY_RETRY_S)
            return "requeued_busy"
        payload = {**payload, "draws": list(payload.get("draws") or []) + [draw(sig["prompt"])]}
        if len(payload["draws"]) < EXTRA_DRAWS:
            enqueue(payload)
            return "next_draw"
        write(payload, "ok", first, first_options=first_options)
        return "written"
    except Exception as e:
        log.warning(f"[{ticker}] verdict vote shadow failed: {type(e).__name__}: {e}")
        try:
            write(payload, "error", first, extra={"error": f"{type(e).__name__}: {e}"[:300]},
                  first_options=first_options)
        except Exception:
            pass
        return "error"
