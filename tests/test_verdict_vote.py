"""agents/verdict_vote.py: the shadow self-consistency vote. Enqueues only for
a routed OPEN/ADD, after routing, and never touches it; each draw runs on the
stored prompt only when it is byte-exact, waits while the live LLM lanes have
work, and the row's vote math. Nothing here calls a model or Redis."""
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

import config
from agents import crew, verdict_vote as vv

PROMPT = "You are the analyst.\nRegime: Neutral\nTicker: NVDA\n..."      # no account figures
SHA = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]


def _result(action="OPEN", ptr=True, signal_id=41, **kw):
    return {"ticker": "NVDA", "action": action, "conviction": 7, "upside_downside_ratio": 2.6,
            "pass_to_risk": ptr, "provenance": {"signal_id": signal_id, "prompt_sha": SHA}, **kw}


@pytest.fixture
def on():
    with patch.object(config, "VERDICT_VOTE_SHADOW_ENABLED", True):
        yield


def test_switch_ships_off():
    assert config.VERDICT_VOTE_SHADOW_ENABLED is False


# ------------------------------------------------------------- enqueue ---

def test_enqueue_only_for_a_routed_open_or_add(on):
    with patch.object(vv, "_enqueue") as enq:
        for r in (_result("OPEN"), _result("ADD")):
            p = vv.observe_routed(r)
            assert p["signal_id"] == 41 and p["draws"] == [] and p["busy_tries"] == 0
        for r in (_result(ptr=False), _result("TRIM"), _result("EXIT"), _result("HOLD"),
                  _result(signal_id=None), _result(signal_id=True), _result(signal_id="41")):
            assert vv.observe_routed(r) is None
    assert enq.call_count == 2


def test_nothing_is_enqueued_with_the_switch_off():
    with patch.object(vv, "_enqueue") as enq:
        assert vv.observe_routed(_result()) is None
    enq.assert_not_called()


def test_enqueue_goes_to_the_last_llm_queue(on):
    task = MagicMock()
    with patch("tasks.verdict_vote_draw", task):
        vv.observe_routed(_result())
    kw = task.apply_async.call_args.kwargs
    assert kw["queue"] == "llm_p3" and kw["retry"] is False and "countdown" not in kw


def test_an_enqueue_failure_returns_none(on):
    with patch.object(vv, "_enqueue", side_effect=ConnectionError("redis down")):
        assert vv.observe_routed(_result()) is None


def test_routing_runs_first_and_a_shadow_failure_never_touches_it(on):
    order = []
    with patch.object(crew, "_route_to_approval", side_effect=lambda *a: order.append("route")), \
         patch("agents.verdict_vote.observe_routed",
               side_effect=lambda r: (order.append("shadow"), 1 / 0)):
        crew._route_guarded(_result(), {"regime": "Neutral"}, None)        # does not raise
    assert order == ["route", "shadow"]
    with patch.object(crew, "_route_to_approval") as route, \
         patch.object(vv, "_enqueue", side_effect=RuntimeError("x")):
        crew._route_guarded(_result(), {"regime": "Neutral"}, None)
    route.assert_called_once()


def test_a_routing_error_still_reaches_the_shadow_after_it(on):
    with patch.object(crew, "_route_to_approval", side_effect=ValueError("boom")), \
         patch.object(crew, "_log_route_exit") as exit_row, \
         patch.object(crew, "_alert_route_error"), \
         patch("agents.verdict_vote.observe_routed") as obs:
        crew._route_guarded(_result(), {"regime": "Neutral"}, None)
    exit_row.assert_called_once()
    obs.assert_called_once()


# --------------------------------------------------------------- draws ---

def _sig(prompt=PROMPT, sha=SHA):
    return {"prompt": prompt, "prompt_sha": sha,
            "model_output": {"action": "OPEN", "conviction": 7, "upside_downside_ratio": 2.6,
                             "pass_to_risk": True},
            "llm_call": {"eval_count": 900, "total_duration": 190e9,
                         "options": {"temperature": 0.2, "num_predict": 1600, "num_ctx": 32768,
                                     "repeat_penalty": 1.1}},
            "model_digest": "abc123def456"}


class _Harness:
    def __init__(self, busy=None, sig=None, draws=None):
        self.busy, self.sig = busy, sig if sig is not None else _sig()
        self.draws = list(draws or [{"action": "WATCH", "conviction": 6, "udr": 1.8,
                                     "model_pass": False, "error": None},
                                    {"action": "OPEN", "conviction": 8, "udr": 3.0,
                                     "model_pass": True, "error": None}])
        self.enqueued, self.written, self.prompts = [], [], []

    def run(self, payload):
        return vv.run(payload, lane_busy=self._lane, fetch=lambda sid: self.sig,
                      draw=self._draw, enqueue=lambda p, countdown=None:
                      self.enqueued.append((p, countdown)),
                      write=lambda p, status, first=None, extra=None, first_options=None:
                      self.written.append((p, status, first, extra)))

    lane_calls = 0

    def _lane(self):
        self.lane_calls += 1
        return self.busy

    def _draw(self, prompt):
        self.prompts.append(prompt)
        return self.draws.pop(0)


def _payload(**kw):
    return {"signal_id": 41, "ticker": "NVDA", "action": "OPEN", "conviction": 7, "udr": 2.6,
            "prompt_sha": SHA, "queued_at": "t", "draws": [], "busy_tries": 0, **kw}


def test_two_draws_one_task_each_then_one_row(on):
    h = _Harness()
    assert h.run(_payload()) == "next_draw"
    assert h.written == [] and len(h.enqueued) == 1 and h.enqueued[0][1] is None
    nxt = h.enqueued[0][0]
    assert len(nxt["draws"]) == 1
    assert h.run(nxt) == "written"
    assert h.prompts == [PROMPT, PROMPT]                       # the stored prompt, as stored
    p, status, first, _ = h.written[0]
    assert status == "ok" and len(p["draws"]) == 2
    assert first["action"] == "OPEN" and first["first"] is True and first["model_pass"] is True


def test_disabled_mid_chain_does_nothing():
    h = _Harness()
    assert h.run(_payload()) == "disabled"
    assert h.enqueued == h.written == h.prompts == []


def test_a_busy_lane_requeues_without_a_call_then_gives_up(on):
    h = _Harness(busy={"queued": {"llm_p0": 1}})
    assert h.run(_payload()) == "requeued_busy"
    (p, countdown), = h.enqueued
    assert countdown == vv._BUSY_RETRY_S and p["busy_tries"] == 1 and h.prompts == []
    assert h.run(_payload(busy_tries=vv._BUSY_MAX_TRIES)) == "gave_up_busy"
    assert h.written[0][1] == "gave_up_busy" and h.written[0][3] == {"lane": h.busy}
    assert h.prompts == []


@pytest.mark.parametrize("sig", [
    _sig(sha="000000000000"),                                       # stored text differs
    _sig(prompt=PROMPT + " "),
    _sig(sha=None), _sig(prompt=None), _sig(sha=SHA[:6])])
def test_no_call_unless_the_stored_prompt_is_exact(on, sig):
    h = _Harness(sig=sig)
    assert h.run(_payload()) == "prompt_not_exact"
    assert h.prompts == [] and h.enqueued == [] and h.written[0][1] == "prompt_not_exact"
    assert h.lane_calls == 0                                   # the lane is read last


def test_a_redacted_prompt_is_not_exact(on):
    """A prompt that had account figures: prompt_sha is of the text the model
    saw, the stored copy is redacted, so no draw runs."""
    from agents.analyst import redact_account_figures
    seen = "You are the analyst.\nPortfolio Value: $1,000\nCash Available: $10\n..."
    stored = redact_account_figures(seen)
    assert stored != seen
    sha = hashlib.sha256(seen.encode()).hexdigest()[:12]
    h = _Harness(sig=_sig(prompt=stored, sha=sha))
    assert h.run(_payload(prompt_sha=sha)) == "prompt_not_exact" and h.prompts == []


def test_a_prompt_sha_other_than_the_routed_one_is_not_exact(on):
    h = _Harness()
    assert h.run(_payload(prompt_sha="ffffffffffff")) == "prompt_not_exact" and h.prompts == []


def test_a_missing_signal_row_is_recorded(on):
    h = _Harness()
    assert vv.run(_payload(), lane_busy=lambda: None, fetch=lambda sid: None,
                  draw=h._draw, enqueue=lambda *a, **k: None,
                  write=lambda p, s, first=None, extra=None: h.written.append(s)) == "no_signal"
    assert h.written == ["no_signal"] and h.prompts == []


def test_an_exception_writes_an_error_row_and_never_raises(on):
    h = _Harness()
    h._draw = MagicMock(side_effect=RuntimeError("gpu"))
    assert h.run(_payload()) == "error"
    assert h.written[0][1] == "error" and "RuntimeError: gpu" in h.written[0][3]["error"]
    h.run = lambda p: vv.run(p, lane_busy=lambda: None, fetch=lambda s: 1 / 0,
                             draw=None, enqueue=None,
                             write=MagicMock(side_effect=OSError("db")))
    assert h.run(_payload()) == "error"


def test_draw_records_the_models_own_fields():
    reply = json.dumps({"action": "open", "conviction": 7.5, "upside_downside_ratio": "2.4",
                        "pass_to_risk": True})
    with patch("agents.analyst._call_ollama", return_value=reply), \
         patch("agents.analyst._ollama_identity", return_value={"model_digest": "d"}), \
         patch.dict("agents.analyst._LAST_CALL_STATS", {"eval_count": 5}, clear=True):
        d = vv._draw(PROMPT)
    assert (d["action"], d["conviction"], d["udr"], d["model_pass"]) == ("OPEN", 7.5, 2.4, True)
    assert d["error"] is None and d["parse_mode"] == "json" and d["model_digest"] == "d"
    with patch("agents.analyst._call_ollama", return_value=None), \
         patch("agents.analyst._ollama_identity", return_value={}), \
         patch.dict("agents.analyst._LAST_CALL_FAILURE", {"kind": "timeout"}):
        d = vv._draw(PROMPT)
    assert d["error"] == "timeout" and d["action"] is None and d["model_pass"] is None
    with patch("agents.analyst._call_ollama", return_value="not json at all"), \
         patch("agents.analyst._ollama_identity", return_value={}):
        assert vv._draw(PROMPT)["error"] == "parse_failed"


# ---------------------------------------------------------------- vote ---

def _d(action, conv, udr, ptr, error=None):
    return {"action": action, "conviction": conv, "udr": udr, "model_pass": ptr, "error": error}


def test_vote_majority_medians_and_spread():
    v = vv.vote([_d("OPEN", 7, 2.6, True), _d("WATCH", 6, 1.8, False), _d("OPEN", 8, 3.0, True)])
    assert v["actions"] == ["OPEN", "WATCH", "OPEN"] and v["majority_action"] == "OPEN"
    assert v["majority_count"] == 2 and v["median_conviction"] == 7 and v["median_udr"] == 2.6
    assert v["conviction_spread"] == 2 and v["udr_spread"] == pytest.approx(1.2)
    assert v["buy_votes"] == 2 and v["differ_from_first"] == 1 and v["n_valid"] == 3
    assert v["model_passes"] == [True, False, True] and v["model_pass_majority"] is True
    assert v["would_pass_bar"] is None


def test_vote_without_a_majority_and_with_failed_draws():
    v = vv.vote([_d("OPEN", 7, 2.6, True), _d("WATCH", 5, 1.5, False), _d("HOLD", 6, 2.0, None)])
    assert v["majority_action"] is None and v["majority_count"] == 1
    assert v["model_pass_majority"] is None and v["differ_from_first"] == 2
    v = vv.vote([_d("ADD", 7, 2.2, True), _d(None, None, None, None, "timeout"),
                 _d(None, None, None, None, "parse_failed")])
    assert v["majority_action"] is None and v["n_valid"] == 1 and v["n_valid_actions"] == 1
    assert v["median_conviction"] == 7 and v["conviction_spread"] == 0 and v["buy_votes"] == 1
    assert v["differ_from_first"] == 0 and v["majority_count"] == 1     # failed draws don't count
    v = vv.vote([_d(None, None, None, None), _d("OPEN", 7, 2.6, True), _d("OPEN", 8, 3, True)])
    assert v["differ_from_first"] is None and v["majority_action"] == "OPEN"
    assert v["n_valid_actions"] == 2
    v = vv.vote([_d("OPEN", 7, 2.6, True), _d("WATCH", 4, 1.0, False), _d("WATCH", 5, 1.2, False)])
    assert v["majority_action"] == "WATCH" and v["model_pass_majority"] is False
    assert v["median_conviction"] == 5 and v["buy_votes"] == 1


def test_numbers_are_finite_floats_only():
    assert [vv._num(x) for x in (7, "7.5", True, "x", None, float("nan"), float("inf"))] == \
        [7.0, 7.5, None, None, None, None, None]
    assert vv._model_draw({"action": " add ", "conviction": "7", "pass_to_risk": "true"}) == \
        {"action": "ADD", "conviction": 7.0, "udr": None, "model_pass": True,
         "model_pass_raw": "true"}                       # bool() as live reads it
    assert vv._model_draw({"action": "OPEN"})["model_pass"] is False
    assert vv._model_draw({"action": "OPEN"})["model_pass_raw"] is None
    assert vv._model_draw({"pass_to_risk": 0})["model_pass"] is False
    assert vv._model_draw(None)["model_pass"] is None and vv._model_draw({})["action"] is None


def test_the_row(on):
    rows = []
    with patch("db.log_activity", side_effect=lambda *a, **k: rows.append((a, k)) or True):
        first = {**_d("OPEN", 7, 2.6, True), "first": True}
        vv._write(_payload(draws=[_d("WATCH", 6, 1.8, False), _d("OPEN", 8, 3.0, True)]),
                  "ok", first, first_options=_sig()["llm_call"]["options"])
        vv._write(_payload(), "no_signal")
        vv._write(_payload(), "gave_up_busy", first, extra={"lane": {"queued": {"llm_p0": 1}}},
                  first_options={"temperature": 0.7})
        vv._write(_payload(), "error", extra={"error": "x", "other": 1})
    (stage, ticker, msg, sentiment), kw = rows[0]
    rec = kw["extra"]
    assert rec["options_match"] is True and rec["env"] == config.TRADING_ENV
    assert {"code_sha", "env_sha", "lib_versions"} <= set(rec)
    keys = set(rec)
    assert all(set(r[1]["extra"]) == keys for r in rows)       # one shape for every status
    assert rows[1][1]["extra"]["options_match"] is None and rows[1][1]["extra"]["draws"] == []
    assert rows[2][1]["extra"]["options_match"] is False
    assert rows[2][1]["extra"]["lane"] == {"queued": {"llm_p0": 1}}
    assert rows[3][1]["extra"]["error"] == "x" and "other" not in rows[3][1]["extra"]
    assert stage == "verdict_vote_shadow" and ticker == "NVDA" and "$" not in msg
    assert rec["signal_id"] == 41 and rec["status"] == "ok" and len(rec["draws"]) == 3
    assert rec["majority_action"] == "OPEN" and rec["model_pass_majority"] is True
    assert rec["would_pass_bar"] is None and rec["options"]["temperature"] == 0.2
    assert rec["prompt_sha"] == SHA and "prompt" not in rec


def test_prompt_is_exact():
    assert vv.prompt_is_exact(PROMPT, SHA)
    assert not vv.prompt_is_exact(PROMPT + " ", SHA)
    assert not vv.prompt_is_exact(PROMPT, SHA[:8]) and not vv.prompt_is_exact(PROMPT, "")


# ---------------------------------------------------------------- lane ---

class _Redis:
    def __init__(self, lens=None, unacked=None, fail=False):
        self.lens, self.unacked, self.fail = lens or {}, unacked or {}, fail

    def llen(self, k):
        if self.fail:
            raise ConnectionError("down")
        return self.lens.get(k, 0)

    def hscan_iter(self, key, count=500):
        return iter(self.unacked.items())


def _lane(r):
    with patch.object(vv, "_redis", return_value=r):
        return vv._lane_busy()


def test_lane_check():
    assert _lane(_Redis()) is None
    assert _lane(_Redis(lens={"llm_p0": 2})) == {"queued": {"llm_p0": 2}}
    assert _lane(_Redis(lens={"llm_p1\x06\x166": 1})) == {"queued": {"llm_p1\x06\x166": 1}}
    assert _lane(_Redis(lens={"llm_p3": 5})) is None                       # its own queue
    msg = lambda q: json.dumps([{"body": ""}, "", q])
    assert _lane(_Redis(unacked={"t1": msg("llm_p1")})) == {"reserved": 1}
    assert _lane(_Redis(unacked={"t1": msg("llm_p3")})) is None
    assert _lane(_Redis(unacked={b"t1": msg("llm_p2").encode(), b"t2": b"garbage"})) == {"reserved": 1}
    assert _lane(_Redis(fail=True)) == {"error": "ConnectionError"}


def test_the_lane_is_checked_right_before_the_call(on):
    order = []
    vv.run(_payload(), lane_busy=lambda: order.append("lane"),
           fetch=lambda sid: order.append("fetch") or _sig(),
           draw=lambda p: order.append("draw") or {"action": "OPEN", "error": None},
           enqueue=lambda *a, **k: None, write=lambda *a, **k: None)
    assert order == ["fetch", "lane", "draw"]


def test_the_shadow_queue_is_drained_last():
    import process_manager
    import tasks
    cmd = process_manager.SERVICES["celery_llm_worker"]["cmd"]
    assert cmd[cmd.index("-Q") + 1] == "llm_p0,llm_p1,llm_p2,llm_p3"
    live = [tasks.LLM_QUEUE_HELD, tasks.LLM_QUEUE_BULK, tasks.LLM_QUEUE_RESEARCH]
    assert tuple(live) == vv.LIVE_LLM_QUEUES
    qs = live + [tasks.SHADOW_LLM_QUEUE]
    assert sorted(qs) == qs and tasks.SHADOW_LLM_QUEUE == vv.QUEUE
    assert tasks.app.conf.broker_transport_options["queue_order_strategy"] == "sorted"


def test_no_beat_entry_for_the_shadow():
    import tasks
    assert all("verdict_vote" not in str(v.get("task")) for v in tasks.app.conf.beat_schedule.values())


def test_the_task_delegates_to_run():
    import tasks
    assert not tasks.verdict_vote_draw.ignore_result            # STARTED is tracked
    assert tasks.app.conf.task_track_started is True
    with patch("agents.verdict_vote.run", return_value="disabled") as run:
        assert tasks.verdict_vote_draw.run({"signal_id": 1}) == "disabled"
    run.assert_called_once_with({"signal_id": 1})
