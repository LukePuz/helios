"""prompt_order_ab: the arm prompts differ only in the schema block, the
call settings come from the live code, resume, the queue wait, the
stack-down guard and stop-at work without touching Ollama, Redis or the DB,
and the report's math."""
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "prompt_order_ab_stored_prompt.txt")
ET = ZoneInfo("America/New_York")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load("prompt_order_ab")
rep = _load("prompt_order_ab_report")


@pytest.fixture
def stored():
    # Rendered by agents.analyst._build_prompt (post-31d52f8 template) with a
    # synthetic brief and track record (ticker EXMP, made-up numbers), then
    # redact_account_figures, as signals.prompt holds it.
    with open(FIXTURE) as f:
        return f.read()


@pytest.fixture
def blocks():
    return ab.load_blocks()


# -- transform -------------------------------------------------------------

def test_fixture_is_synthetic(stored):
    assert "Ticker: EXMP" in stored and "Held: EXMP 9 sh | P&L: +9.99%" in stored
    assert stored.count("SYNTHETIC TEST DATA: Conviction") == 2
    assert "Portfolio Value: $[REDACTED]" in stored


def test_round_trip_on_a_stored_prompt(stored, blocks):
    a_block, b_block = blocks
    p = ab.build_pair(stored, a_block, b_block)
    assert p["native"] == "B" and p["B"] == stored
    assert a_block in p["A"] and b_block not in p["A"]
    assert p["A"].replace(a_block, b_block, 1) == stored
    # starting from the A arm gives the same two prompts back
    q = ab.build_pair(p["A"], a_block, b_block)
    assert q["native"] == "A" and q["A"] == p["A"] and q["B"] == stored
    assert (q["sha_A"], q["sha_B"], q["sha_prefix"], q["sha_suffix"]) == \
           (p["sha_A"], p["sha_B"], p["sha_prefix"], p["sha_suffix"])


def test_text_outside_the_schema_block_is_byte_identical(stored, blocks):
    a_block, b_block = blocks
    p = ab.build_pair(stored, a_block, b_block, arms=("A", "B", "C"))
    i = p["block_start"]
    c_block = ab.block_c(a_block, b_block)
    parts = {k: (p[k][:i], p[k][i + len(blk):]) for k, blk in (("A", a_block), ("B", b_block), ("C", c_block))}
    assert parts["A"] == parts["B"] == parts["C"]
    pre, post = parts["A"]
    for text in ("=== SITUATION BRIEF ===", "Portfolio Value: $[REDACTED]",
                 "Example Corp announces $2B buyback", "Answer these questions in order:",
                 "\nRules:\n- Set pass_to_risk", "Output ONLY the JSON object"):
        assert text in pre + post and text not in a_block and text not in b_block
    changed = [ln[1:] for ln in ab.schema_diff(p["A"], p["B"])
               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert changed and all(ln in a_block or ln in b_block for ln in changed)


def test_arm_c_is_the_new_order_without_the_instruction(blocks):
    a_block, b_block = blocks
    c = ab.block_c(a_block, b_block)
    assert "Fill the fields" not in c
    assert c.splitlines()[0] == a_block.splitlines()[0]
    # only the order differs from A (a field's trailing comma follows its position)
    assert sorted(ln.rstrip(",") for ln in c.splitlines()) == sorted(ln.rstrip(",") for ln in a_block.splitlines())
    assert c[c.index("\n\n{"):] == b_block[b_block.index("\n\n{"):]      # same body as B
    assert c.index('"action"') > c.index('"reasoning"')


def test_blocks_are_the_two_orders(blocks):
    a_block, b_block = blocks
    assert a_block.index('"action"') < a_block.index('"reasoning"')
    assert b_block.index('"action"') > b_block.index('"reasoning"')
    assert "Fill the fields in the order shown" in b_block and "Fill the fields" not in a_block


def test_committed_b_block_is_the_live_template_block(blocks):
    src = open(os.path.join(REPO, "agents", "analyst.py")).read()
    assert ab.prompt_ab.schema_block(src) == blocks[1]


@pytest.mark.parametrize("rev,idx", [(ab.COMMIT_A, 0), (ab.COMMIT_B, 1)])
def test_committed_blocks_match_git(rev, idx, blocks):
    r = subprocess.run(["git", "-C", REPO, "show", f"{rev}:agents/analyst.py"],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"{rev} not in this history; the committed file is the record")
    assert ab.prompt_ab.schema_block(r.stdout) == blocks[idx]


def test_unknown_or_doubled_block_is_refused(stored, blocks):
    a_block, b_block = blocks
    with pytest.raises(ab.PromptShapeError):
        ab.build_pair("no schema here", a_block, b_block)
    with pytest.raises(ab.PromptShapeError):
        ab.build_pair(stored + b_block, a_block, b_block)
    with pytest.raises(ab.PromptShapeError):
        ab.build_pair(stored.replace("{\n  \"sources\"", "{\n  \"sourcez\"", 1), a_block, b_block)


def test_arm_order_is_seeded_per_signal():
    orders = [ab.arm_order(7, sid) for sid in range(200)]
    assert orders == [ab.arm_order(7, sid) for sid in range(200)]
    assert set(orders) == {("A", "B"), ("B", "A")}
    assert 60 < sum(o == ("A", "B") for o in orders) < 140
    assert orders != [ab.arm_order(8, sid) for sid in range(200)]
    three = [ab.arm_order(7, sid, ("A", "B", "C")) for sid in range(300)]
    assert len(set(three)) == 6 and all(sorted(o) == ["A", "B", "C"] for o in three)


# -- live settings -----------------------------------------------------------

def test_live_payload_comes_from_the_live_call(monkeypatch):
    from agents import analyst
    import requests

    def no_network(*a, **k):
        raise AssertionError("network call")
    monkeypatch.setattr(requests, "post", no_network)
    live = ab.live_payload()
    p = live["payload"]
    assert p["model"] == analyst.ANALYST_MODEL
    assert p["options"] == analyst._ollama_options()
    assert p["options"]["num_ctx"] == analyst._NUM_CTX == 32768
    assert p["options"]["num_predict"] == analyst._NUM_PREDICT
    assert p["logprobs"] is True and p["stream"] is False
    assert live["host"] == analyst.OLLAMA_HOST and live["timeout"] == 400
    assert requests.post is no_network   # the stub was undone


def test_timeout_defaults_to_and_is_capped_by_the_live_call():
    assert ab.resolve_timeout(None, 400) == 400.0
    assert ab.resolve_timeout(300, 400) == 300.0 and ab.resolve_timeout(400, 400) == 400.0
    with pytest.raises(ValueError):
        ab.resolve_timeout(401, 400)
    with pytest.raises(ValueError):
        ab.resolve_timeout(None, None)


def test_main_uses_the_captured_live_timeout(tmp_path, monkeypatch, blocks):
    import dotenv
    ids = tmp_path / "ids.txt"
    ids.write_text("1\n")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(ab, "live_payload", lambda: dict(LIVE, timeout=377))
    seen = {}

    def fetch(ids):
        raise SystemExit("stop here")
    monkeypatch.setattr(ab, "fetch_signals", fetch)
    real = ab.resolve_timeout
    monkeypatch.setattr(ab, "resolve_timeout", lambda req, live_t: seen.setdefault("t", real(req, live_t)))
    with pytest.raises(SystemExit, match="stop here"):
        ab.main(["--seed", "1", "--signal-ids-file", str(ids)])
    assert seen["t"] == 377.0
    seen.clear()
    with pytest.raises(SystemExit, match=r"\(0, 377\]"):
        ab.main(["--seed", "1", "--signal-ids-file", str(ids), "--timeout", "378"])


def test_arm_a_num_predict_option():
    live = {"payload": {"options": {"num_predict": 1600, "num_ctx": 32768}}}
    assert ab.arm_options(live, "A")["num_predict"] == 1600
    assert ab.arm_options(live, "A", 1200) == {"num_predict": 1200, "num_ctx": 32768}
    assert ab.arm_options(live, "B", 1200)["num_predict"] == 1600
    assert live["payload"]["options"]["num_predict"] == 1600   # not mutated


class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_call_arm_parses_and_redacts():
    sent = {}

    def post(url, json=None, timeout=None):
        sent.update(url=url, json=json, timeout=timeout)
        return _Resp({"response": '{"reasoning": "Portfolio Value: $1,000 now", "conviction": 12, "action": "open"}',
                      "done_reason": "stop", "eval_count": 900, "prompt_eval_count": 9000})
    live = {"host": "http://x:1", "payload": {"model": "m", "options": {"num_ctx": 32768, "num_predict": 1600}}}
    out = ab.call_arm(live, "PROMPT", 240, options={"num_ctx": 32768, "num_predict": 1200}, post=post)
    assert sent["json"]["prompt"] == "PROMPT" and sent["timeout"] == 240
    assert sent["json"]["options"]["num_predict"] == 1200 and out["num_predict"] == 1200
    assert sent["url"] == "http://x:1/api/generate"
    assert out["action"] == "OPEN" and out["conviction"] == 10 and out["conviction_raw"] == 12
    assert "$[REDACTED]" in out["raw"] and "1,000" not in out["raw"]
    assert out["error"] is None and out["eval_count"] == 900 and out["prompt_eval_count"] == 9000


def test_call_arm_timeout_is_an_error():
    import requests

    def post(*a, **k):
        raise requests.exceptions.Timeout()
    out = ab.call_arm({"host": "h", "payload": {}}, "P", 240, post=post)
    assert out["error"].startswith("timeout") and "conviction" not in out


# -- stack-down guard and stop-at ------------------------------------------------

class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode = stdout, returncode


def _runner(state="inactive", beat_rc=1):
    def run(cmd, **kw):
        if cmd[0] == "systemctl":
            return _Proc(state + "\n")
        if cmd[0] == "pgrep":
            return _Proc("", beat_rc)
        raise AssertionError(cmd)
    return run


SAT_NOON = datetime(2026, 11, 7, 12, 0, tzinfo=ET)
TUE_22 = datetime(2026, 11, 3, 22, 0, tzinfo=ET)


@pytest.mark.parametrize("now,live", [
    (datetime(2026, 11, 3, 4, 59, tzinfo=ET), False), (datetime(2026, 11, 3, 5, 0, tzinfo=ET), True),
    (datetime(2026, 11, 3, 18, 59, tzinfo=ET), True), (datetime(2026, 11, 3, 19, 0, tzinfo=ET), False),
    (SAT_NOON, False),
])
def test_live_window(now, live):
    assert ab.in_live_window(now) is live


@pytest.mark.parametrize("now,state,beat_rc,reason", [
    (TUE_22, "inactive", 1, None),
    (TUE_22, "failed", 1, None),
    (datetime(2026, 11, 4, 6, 0, tzinfo=ET), "inactive", 1, "live_hours"),
    (TUE_22, "active", 1, "helios-local is active"),
    (TUE_22, "activating", 1, "helios-local is activating"),
    (TUE_22, "inactive", 0, "celery beat is running"),
    (TUE_22, "inactive", 2, "could not check"),
])
def test_stack_down_reason(now, state, beat_rc, reason):
    got = ab.stack_down_reason(now, run=_runner(state, beat_rc))
    assert (got is None) if reason is None else got.startswith(reason)


def test_stack_down_reason_fails_closed_without_systemctl():
    def run(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    assert ab.stack_down_reason(TUE_22, run=run).startswith("helios-local is unknown")


def test_stop_deadline_stays_before_the_next_live_start():
    assert ab.stop_deadline("04:30", TUE_22) == datetime(2026, 11, 4, 4, 30, tzinfo=ET)
    assert ab.stop_deadline("23:15", TUE_22) == datetime(2026, 11, 3, 23, 15, tzinfo=ET)
    assert ab.stop_deadline("05:00", TUE_22) == datetime(2026, 11, 4, 5, 0, tzinfo=ET)
    # a resume at 06:00 with --stop-at 04:30 would run through the live day
    with pytest.raises(ValueError, match="more than 10 h"):
        ab.stop_deadline("04:30", datetime(2026, 11, 4, 6, 0, tzinfo=ET))
    with pytest.raises(ValueError, match="after the next 05:00"):
        ab.stop_deadline("06:00", TUE_22)
    with pytest.raises(ValueError, match="more than 10 h"):
        ab.stop_deadline("04:30", datetime(2026, 11, 3, 18, 0, tzinfo=ET))
    with pytest.raises(ValueError):
        ab.stop_deadline("4:3", TUE_22)


# -- queue gate ----------------------------------------------------------------

NOW = datetime(2026, 11, 4, 3, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("models,reason", [
    ([], "clear"),
    ([{"name": "qwen2.5:72b", "context_length": 32768, "expires_at": "2026-11-04T03:04:00.123456789Z"}], "clear"),
    ([{"name": "qwen2.5:7b"}], "other_model_loaded"),
    ([{"name": "qwen2.5:72b", "context_length": 8192}], "loaded_at_other_num_ctx"),
    ([{"name": "qwen2.5:72b", "expires_at": "2026-11-03T21:00:00-05:00"}], "loaded_past_expiry"),
])
def test_ollama_ps_state(models, reason):
    ok, why, _ = ab.ollama_ps_state("http://h:1", "qwen2.5:72b", 32768,
                                    get=lambda url, timeout: _Resp({"models": models}), now=NOW)
    assert why == reason and ok == (reason == "clear")


def test_ollama_ps_unreachable_is_not_clear():
    def get(*a, **k):
        raise ConnectionError()
    ok, why, _ = ab.ollama_ps_state("http://h:1", "m", 1, get=get)
    assert not ok and why == "ollama_ps_unavailable"


class FakeRedis:
    """Only the read commands the lane gates use; anything else fails."""

    def __init__(self, lists=None, unacked=None, results=None, ttl=86000):
        self.lists, self.unacked, self.results, self._ttl = lists or {}, unacked or {}, results or {}, ttl

    def llen(self, k):
        return self.lists.get(k, 0)

    def hscan_iter(self, name, count=None):
        assert name == "unacked"
        return iter(self.unacked.items())

    def scan(self, cursor=0, match=None, count=None):
        return 0, [k for k in self.results if k.startswith(match.rstrip("*"))]

    def mget(self, keys):
        return [self.results[k] for k in keys]

    def ttl(self, k):
        return self._ttl


def _started(host):
    return json.dumps({"status": "STARTED", "result": {"hostname": host}}).encode()


LANE_CASES = [
    (FakeRedis(), "clear"),
    (FakeRedis(lists={"llm_p1": 2}), "live_queue_not_empty"),
    (FakeRedis(lists={"llm_p0\x06\x169": 1}), "live_queue_not_empty"),
    (FakeRedis(lists={"ops": 9}), "clear"),
    (FakeRedis(unacked={b"t": json.dumps([{}, "", "llm_p2"]).encode()}), "live_task_reserved"),
    (FakeRedis(unacked={b"t": json.dumps([{}, "", "ops"]).encode()}), "clear"),
    (FakeRedis(unacked={b"t": b"not json"}), "clear"),
    (FakeRedis(results={"celery-task-meta-1": _started("llm@box")}), "live_task_running"),
    (FakeRedis(results={"celery-task-meta-1": _started("ops@box")}), "clear"),
    (FakeRedis(results={"celery-task-meta-1": _started("llm@box")}, ttl=1000), "clear"),   # stale
    (FakeRedis(results={"celery-task-meta-1": _started("llm@box")}, ttl=-1), "live_task_running"),
    (FakeRedis(results={"celery-task-meta-1": json.dumps({"status": "SUCCESS"}).encode(),
                        "celery-task-meta-2": None}), "clear"),
]


@pytest.mark.parametrize("r,reason", LANE_CASES)
def test_live_lane_reads(tmp_path, r, reason):
    lane = ab.LiveLane("redis://unused", str(tmp_path / "off"), redis_client=r)
    ok, why, _ = lane.check()
    assert why == reason and ok == (reason == "clear")


@pytest.mark.parametrize("r,reason", LANE_CASES)
def test_live_lane_matches_the_panel_gate(tmp_path, r, reason):
    # The panel gate can't be imported by a live-side script; compare
    # behaviour on the same fixtures instead.
    from helios3d.panel.config import PanelSettings
    from helios3d.panel.gate import LiveLaneGate
    kill = str(tmp_path / "off")
    panel = LiveLaneGate(PanelSettings(kill_file=kill), redis_client=r).check()
    ours = ab.LiveLane("redis://unused", kill, redis_client=r).check()
    assert (ours[0], ours[1]) == (panel.ok, panel.reason) == (reason == "clear", reason)
    assert ab.LIVE_TASK_MAX_AGE_S == PanelSettings().live_task_max_age_s
    assert ab.RESULT_EXPIRES_S == PanelSettings().result_expires_s


def test_result_expires_is_the_live_celery_value():
    import tasks
    v = tasks.app.conf.result_expires
    assert ab.RESULT_EXPIRES_S == (v.total_seconds() if hasattr(v, "total_seconds") else float(v))


def test_live_lane_kill_file_and_unreachable(tmp_path):
    class Down:
        def llen(self, k):
            raise ConnectionError()
    kill = tmp_path / "off"
    assert ab.LiveLane("x", str(kill), redis_client=Down()).check()[1] == "redis_unavailable"
    kill.write_text("")
    assert ab.LiveLane("x", str(kill), redis_client=FakeRedis()).check()[1] == "kill_file"


class FakeClock:
    def __init__(self, t0):
        self.t = t0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def dt(self):
        return datetime.fromtimestamp(self.t, timezone.utc)


class FakeLane:
    def __init__(self, busy_checks, reason="live_task_running"):
        self.busy, self.reason, self.checks = busy_checks, reason, 0

    def check(self):
        self.checks += 1
        if self.checks <= self.busy:
            return False, self.reason, {}
        return True, "clear", {}


def _gate(lane, clock, ps=lambda: (True, "clear", {}), log=lambda *_: None):
    return ab.QueueGate(lane, "http://h:1", "m", 32768, poll_s=15, ps_check=ps,
                        clock=clock, sleep=clock.sleep, log=log)


def test_gate_waits_until_lane_and_ollama_are_clear_and_logs_reasons():
    clock = FakeClock(NOW.timestamp())
    ps_calls = {"n": 0}
    logged = []

    def ps():
        ps_calls["n"] += 1
        return (ps_calls["n"] > 2, "other_model_loaded" if ps_calls["n"] <= 2 else "clear", {})
    g = _gate(FakeLane(3), clock, ps, log=logged.append)
    res = g.wait(give_up_at=clock.t + 3600)
    assert res["ok"] and res["waited_s"] == 15 * 5   # 3 lane-busy + 2 ps-busy polls
    assert len(logged) == 2 and "live_task_running" in logged[0] and "other_model_loaded" in logged[1]


def test_gate_gives_up_and_kill_file_stops():
    clock = FakeClock(NOW.timestamp())
    res = _gate(FakeLane(10 ** 6), clock).wait(give_up_at=clock.t + 100)
    assert not res["ok"] and res["reason"].startswith("gave_up_waiting") and clock.t <= NOW.timestamp() + 100
    res = _gate(FakeLane(10 ** 6, reason="kill_file"), clock).wait(give_up_at=clock.t + 10 ** 6)
    assert not res["ok"] and res["reason"] == "kill_file"


# -- run loop: resume, skips, wait, stop-at ------------------------------------------

def _signals(stored, ids):
    return [{"signal_id": i, "ticker": "EXMP", "prompt": stored, "provenance_prompt_sha": None} for i in ids]


def _fake_call(clock, seconds=200, conviction=7, log=None, error=None):
    def call(live, prompt, timeout_s, options=None):
        clock.t += seconds
        if log is not None:
            log.append((prompt, options))
        return {"action": "WATCH", "conviction": None if error else conviction, "error": error,
                "wall_s": seconds}
    return call


LIVE = {"host": "http://h:1", "payload": {"model": "m", "options": {"num_ctx": 32768, "num_predict": 1600}}}


def _run(stored, blocks, out, ids, clock, calls=None, **kw):
    kw.setdefault("deadline", NOW + timedelta(hours=5))
    kw.setdefault("gate", _gate(FakeLane(0), clock))
    kw.setdefault("call", _fake_call(clock, log=calls))
    sigs = kw.pop("signals", None) or _signals(stored, ids)
    return ab.run_pairs(sigs, live=LIVE, blocks=blocks, seed=kw.pop("seed", 1), out=str(out),
                        timeout_s=kw.pop("timeout_s", 240), now=clock.dt, log=lambda *_: None, **kw)


def _lines(path):
    out = []
    with open(path) as f:
        for x in f:
            try:
                out.append(json.loads(x))
            except ValueError:
                out.append(None)
    return out


def test_resume_skips_signals_already_written(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    out.write_text(json.dumps({"kind": "pair", "signal_id": 1, "status": "ok"}) + "\n"
                   + json.dumps({"kind": "pair", "signal_id": 2, "status": "error"}) + "\n"
                   + '{"kind": "pair", "signal_id": 9, "sta')   # a line cut by a crash
    clock = FakeClock(NOW.timestamp())
    calls = []
    stats = _run(stored, blocks, out, [1, 2, 3, 9], clock, calls)
    assert stats["skipped_done"] == 2 and stats["written"] == 2 and len(calls) == 4
    lines = _lines(out)
    assert lines[2] is None   # the cut line stays on its own
    assert [r["signal_id"] for r in lines[3:]] == [3, 9]
    assert all(r["attempt"] == 1 for r in lines[3:])
    # with --retry-errors only signal 2 is redone, as its second attempt
    calls.clear()
    stats = _run(stored, blocks, out, [1, 2, 3, 9], clock, calls, retry_errors=True)
    assert stats["written"] == 1 and len(calls) == 2
    assert _lines(out)[-1]["signal_id"] == 2 and _lines(out)[-1]["attempt"] == 2
    assert ab.done_ids(str(out), retry_errors=True) == {1, 2, 3, 9}


def test_skip_lines_are_written_once(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    sigs = [{"signal_id": 1, "prompt": None}, {"signal_id": 2, "prompt": "no schema"}] + _signals(stored, [3])
    for _ in range(2):
        stats = _run(stored, blocks, out, None, clock, signals=sigs)
    skips = [r for r in _lines(out) if r["kind"] == "skip"]
    assert [(r["signal_id"], r["reason"].split(":")[0]) for r in skips] == \
           [(1, "no_stored_prompt"), (2, "schema_block")]
    assert stats["missing"] == 1 and stats["skipped_shape"] == 1


def test_pair_record_has_both_arms_in_seeded_order(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    sent = []
    _run(stored, blocks, out, [5], clock, sent, seed=3, gate=_gate(FakeLane(2), clock), run_id="r1",
         arm_a_num_predict=1200)
    rec = _lines(out)[0]
    p = ab.build_pair(stored, *blocks)
    order = ab.arm_order(3, 5)
    assert rec["order"] == list(order) and [s[0] for s in sent] == [p[order[0]], p[order[1]]]
    assert {a: s[1]["num_predict"] for a, s in zip(order, sent)} == {"A": 1200, "B": 1600}
    assert rec["arms"][order[0]]["position"] == 1 and rec["arms"][order[0]]["gate_waited_s"] == 30
    assert (rec["sha_A"], rec["sha_B"]) == (p["sha_A"], p["sha_B"]) and rec["status"] == "ok"
    assert rec["run_id"] == "r1" and rec["arm_a_num_predict"] == 1200 and rec["attempt"] == 1
    text = json.dumps(rec)
    assert "Example Corp announces" not in text and "SITUATION BRIEF" not in text   # no prompt text


def test_three_arms(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    sent = []
    stats = _run(stored, blocks, out, [5], clock, sent, arms=("A", "B", "C"))
    rec = _lines(out)[0]
    assert stats["written"] == 1 and sorted(rec["arms"]) == ["A", "B", "C"] and len(sent) == 3
    assert rec["sha_C"] == ab.build_pair(stored, *blocks, arms=("A", "B", "C"))["sha_C"]
    # three timeouts must fit before the deadline
    clock2 = FakeClock(NOW.timestamp())
    stats = _run(stored, blocks, tmp_path / "x.jsonl", [6], clock2, arms=("A", "B", "C"),
                 deadline=NOW + timedelta(seconds=600))
    assert stats["stopped"] == "stop_at" and stats["written"] == 0


def test_stack_guard_stops_before_a_pair(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    calls = []
    seen = []

    def guard(now):
        seen.append(now)
        return "helios-local is active" if len(seen) > 1 else None
    stats = _run(stored, blocks, out, [1, 2, 3], clock, calls, stack_guard=guard)
    assert stats["written"] == 1 and stats["stopped"] == "helios-local is active" and len(calls) == 2


def test_stop_at_blocks_a_pair_that_cannot_finish(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    calls = []
    stats = _run(stored, blocks, out, [1], clock, calls, deadline=NOW + timedelta(seconds=479))
    assert stats["stopped"] == "stop_at" and not calls and not out.exists()


class _BusyAfterFirst(FakeLane):
    def check(self):
        self.checks += 1
        return self.checks == 1, "clear" if self.checks == 1 else "live_queue_not_empty", {}


@pytest.mark.parametrize("lane,call_s,stopped", [
    (FakeLane(0), 300, "stop_at"),                         # first call ran long
    (_BusyAfterFirst(0), 200, "gave_up_waiting"),          # lane busy until too late
])
def test_stop_at_mid_pair_writes_nothing(tmp_path, stored, blocks, lane, call_s, stopped):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    calls = []
    stats = _run(stored, blocks, out, [1, 2], clock, deadline=NOW + timedelta(seconds=500),
                 gate=_gate(lane, clock), call=_fake_call(clock, seconds=call_s, log=calls))
    assert len(calls) == 1 and stats["written"] == 0 and stats["stopped"].startswith(stopped)
    assert clock.dt() <= NOW + timedelta(seconds=500)
    assert not out.exists() and ab.done_ids(str(out)) == set()


# -- arguments and main -----------------------------------------------------------

def test_args_need_a_sample_and_apply_needs_stop_at(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("101\n102 # note\n\n101\n")
    base = ["--seed", "1", "--signal-ids-file", str(ids)]
    with pytest.raises(SystemExit):
        ab.parse_args(["--seed", "1"])
    with pytest.raises(SystemExit):
        ab.parse_args(["--seed", "1", "--from", "2026-10-28", "--to", "2026-11-01"])
    with pytest.raises(SystemExit):
        ab.parse_args(base + ["--apply", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        ab.parse_args(base + ["--timeout", "0"])
    with pytest.raises(SystemExit):
        ab.parse_args(base + ["--arm-a-num-predict", "1000"])
    a = ab.parse_args(base)
    assert not a.apply and a.timeout is None and a.arms == "AB" and a.arm_a_num_predict is None
    assert ab.parse_args(base + ["--timeout", "400"]).timeout == 400   # checked against live in main
    a = ab.parse_args(base + ["--arms", "ABC", "--arm-a-num-predict", "1200"])
    assert a.arms == "ABC" and a.arm_a_num_predict == 1200
    assert ab.read_ids_file(str(ids)) == [101, 102]


def test_apply_refuses_to_start_while_the_stack_is_up(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("1\n")
    monkeypatch.setattr(ab, "stack_down_reason", lambda now: "helios-local is active")
    monkeypatch.setattr(ab, "stop_deadline", lambda hhmm, now: now + timedelta(hours=1))
    with pytest.raises(SystemExit, match="refusing to start: helios-local is active"):
        ab.main(["--seed", "1", "--signal-ids-file", str(ids), "--apply", "--out", str(tmp_path / "o"),
                 "--stop-at", "04:30"])


def test_schema_mismatch_aborts(tmp_path, monkeypatch, blocks):
    import dotenv
    ids = tmp_path / "ids.txt"
    ids.write_text("1\n")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(ab, "load_blocks", lambda: (blocks[0], blocks[1] + " "))
    monkeypatch.setattr(ab, "fetch_signals", lambda ids: pytest.fail("read the DB"))
    with pytest.raises(SystemExit, match="differs from arm B"):
        ab.main(["--seed", "1", "--signal-ids-file", str(ids)])


def test_draw_sample_is_seeded_and_bounded():
    pop = list(range(100, 200))
    assert ab.draw_sample(pop, 10, 4) == ab.draw_sample(list(reversed(pop)), 10, 4)
    assert ab.draw_sample(pop, 10, 4) != ab.draw_sample(pop, 10, 5)
    with pytest.raises(SystemExit):
        ab.draw_sample(pop, 101, 4)


# -- report ------------------------------------------------------------------

def _arm(conv, action="WATCH", pos=1, **kw):
    return {"conviction": conv, "action": action, "position": pos, "parse_ok": conv is not None,
            "conviction_logprobs": {"expected": conv - 0.25} if conv is not None else None, **kw}


def _pair(sid, ca, cb, aa="WATCH", ba="WATCH", status="ok", first="A", run_id="r1", a_kw=None, b_kw=None):
    pa, pb = (1, 2) if first == "A" else (2, 1)
    return {"kind": "pair", "signal_id": sid, "status": status, "run_id": run_id,
            "arms": {"A": _arm(ca, aa, pa, **(a_kw or {})), "B": _arm(cb, ba, pb, **(b_kw or {}))}}


RUN = {"kind": "run", "run_id": "r1", "seed": 1, "timeout_s": 240, "arms": "AB", "model_digest": "d1",
       "ollama_version": "0.9", "schema_sha_A": "a", "schema_sha_B": "b", "head": "h1",
       "signal_ids": list(range(5)) + [10, 11, 12, 13, 14, 20, 21, 22]}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _toy_rows():
    return ([RUN] + [_pair(i, 7, 5, "OPEN", "WATCH") for i in range(5)]       # A yes, B no
            + [_pair(10, 5, 8, "WATCH", "OPEN", first="B")]                  # A no, B yes
            + [_pair(11, 8, 7, "OPEN", "OPEN", a_kw={"prompt_eval_count": 9000},
                     b_kw={"prompt_eval_count": 9100})]
            + [_pair(12, 4, 4, "PASS", "PASS")]
            + [_pair(13, 7, None, status="error", b_kw={"error": "timeout after 240 s"}),
               _pair(13, 7, 3, status="ok")]                                  # retried
            + [_pair(14, 3, 9, b_kw={"salvaged": True})]
            + [_pair(20, 6, None, status="ok", b_kw={"done_reason": "length", "parse_ok": False})]
            + [{"kind": "skip", "signal_id": 21, "reason": "no_stored_prompt"}])


RUNNER_SHA_AT_9B913D7 = "6986d414e4458de52f1254c1d44d738ef01831344d6a1f1323ad326d40e79f5f"


def test_runner_is_byte_identical_to_9b913d7():
    # its hash keys the resume of a run in progress
    import hashlib
    with open(os.path.join(REPO, "scripts", "prompt_order_ab.py"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == RUNNER_SHA_AT_9B913D7


def test_bounds_treat_a_cut_off_reply_with_a_conviction_as_cut_off():
    last = [_pair(1, 7, 8, b_kw={"done_reason": "length"}),   # B cut off, parsed c8
            _pair(2, 5, 9),
            _pair(3, 8, 4)]
    b = rep.bounds(last, "A", "B")
    assert b["failed_or_cut"] == {"A": 0, "B": 1}
    assert b["dropped"]["n"] == 2
    assert b["dropped"]["x"] == pytest.approx(1 / 2) and b["dropped"]["y"] == pytest.approx(1 / 2)
    assert b["not_c7"]["x"] == pytest.approx(2 / 3) and b["not_c7"]["y"] == pytest.approx(1 / 3)
    assert b["not_c7"]["diff"] == pytest.approx(-1 / 3)
    text = rep.render_bounds(b, "A", "B")
    assert "bound 'dropped'" in text and "bound 'not-c7'" in text


def test_report_prints_its_own_sha_and_the_runner_tool_sha(tmp_path, capsys):
    import hashlib
    path = tmp_path / "r.jsonl"
    rows = _toy_rows()
    rows[0] = {**rows[0], "tool_sha": "t" * 64}
    _write(path, rows)
    rep.main([str(path), "--primary", "c7_share", "--boot", "50"])
    first = capsys.readouterr().out.splitlines()[0]
    with open(os.path.join(REPO, "scripts", "prompt_order_ab_report.py"), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert first == f"report_sha256: {sha}; runner tool_sha on run lines: {'t' * 64}"


def test_report_math_on_a_toy_file(tmp_path):
    path = tmp_path / "r.jsonl"
    _write(path, _toy_rows())
    data = rep.load(str(path))
    (g,) = rep.groups(data)
    last = [recs[-1] for recs in g["attempts"].values()]
    assert len(last) == 11 and len(g["attempts"][13]) == 2

    clean = rep.usable(last, "A", "B", clean_only=True)
    s = rep.summarize(clean, 2000, 0)
    assert s["n"] == 9 and s["c7_x"] == 7 and s["c7_y"] == 2
    assert s["diff"] == pytest.approx(-5 / 9)
    m = s["mcnemar"]
    assert (m["x_yes_y_no"], m["x_no_y_yes"]) == (6, 1)
    assert m["p_two_sided"] == pytest.approx(2 * (1 + 7) / 128)        # exact binomial, n=7, k=1
    lo, hi = s["boot"][0.95]
    assert lo <= -5 / 9 <= hi and s["boot"][0.99][0] <= lo and s["boot"][0.99][1] >= hi
    assert s == rep.summarize(clean, 2000, 0)                          # seeded
    assert s["xtab"][("OPEN", "WATCH")] == 5 and s["xtab"][("PASS", "PASS")] == 1
    assert s["dist_x"][7] == 6 and s["dist_y"][5] == 5
    assert s["mean_x"] == pytest.approx((7 * 6 + 5 + 8 + 4) / 9)
    assert s["expected_mean_x"] == pytest.approx(s["mean_x"] - 0.25)
    assert s["by_position"]["A@1"] == (7, 8) and s["by_position"]["B@1"] == (1, 1)
    assert rep.summarize(rep.usable(last, "A", "B", clean_only=False), 100, 0)["n"] == 10

    # missing B conviction (signal 20): pair dropped, or counted as not c7+
    b = rep.bounds(last, "A", "B")
    assert b["n"] == 11 and b["failed_or_cut"] == {"A": 0, "B": 1}
    assert b["dropped"]["n"] == 10
    assert b["dropped"]["x"] == pytest.approx(7 / 10) and b["dropped"]["y"] == pytest.approx(3 / 10)
    assert b["dropped"]["diff"] == pytest.approx(-4 / 10)
    assert b["not_c7"]["x"] == pytest.approx(7 / 11) and b["not_c7"]["y"] == pytest.approx(3 / 11)
    assert b["not_c7"]["diff"] == pytest.approx(-4 / 11)

    t = rep.arm_tables(last, ("A", "B"))
    assert t["B"]["counts"]["parse_failed"] == 1 and t["B"]["counts"]["num_predict_cut"] == 1
    assert t["B"]["counts"]["salvaged"] == 1 and t["A"]["counts"]["ok"] == 11
    assert t["A"]["prompt_eval"] == (1, 9000.0, 9000, 9000)

    text = rep.report_group(g, data["skips"], 200, 0, "c7_share")
    assert "sample 13, skipped 1 {'no_stored_prompt': 1}, attempted 11" in text
    assert "not attempted 1" in text and "1 x10, 2 x1" in text
    assert "McNemar" in text and "95% percentile interval" in text and "prompt_eval_count B" in text


def test_report_arm_failures_by_type():
    last = [_pair(1, 7, None, b_kw={"error": "timeout after 240 s"}),
            _pair(2, 7, None, b_kw={"error": "ConnectionError"}),
            _pair(3, None, 5)]
    t = rep.arm_tables(last, ("A", "B"))
    assert t["B"]["counts"]["timeout"] == 1 and t["B"]["counts"]["call_error"] == 1
    assert t["A"]["counts"]["parse_failed"] == 1 and t["A"]["counts"]["ok"] == 2
    assert rep.arm_failure(None) == "not_called"


def test_report_stops_on_mixed_configurations(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    run2 = dict(RUN, run_id="r2", model_digest="d2", signal_ids=[30])
    _write(path, [RUN, _pair(1, 7, 5), run2, _pair(30, 5, 5, run_id="r2")])
    with pytest.raises(SystemExit):
        rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    assert "differ in configuration" in capsys.readouterr().err
    rep.main([str(path), "--primary", "c7_share", "--boot", "20", "--split"])
    out = capsys.readouterr().out
    assert "configuration 1 of 2" in out and "configuration 2 of 2" in out and '"model_digest": "d2"' in out


def test_report_same_configuration_resumed_is_one_group(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    _write(path, [RUN, _pair(1, 7, 5), dict(RUN, run_id="r2"), _pair(2, 5, 5, run_id="r2")])
    rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    out = capsys.readouterr().out
    assert "######## configuration" not in out and "attempted 2" in out


def test_report_three_arms(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    p = _pair(1, 7, 5)
    p["arms"]["C"] = _arm(6, pos=3)
    _write(path, [dict(RUN, arms="ABC"), p])
    rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    out = capsys.readouterr().out
    assert "--- B vs A ---" in out and "--- C vs A ---" in out and "--- B vs C ---" in out


def test_mcnemar_and_bootstrap_edges():
    assert rep.mcnemar_exact(0, 0) == 1.0
    assert rep.mcnemar_exact(3, 3) == 1.0
    assert rep.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert rep.bootstrap_diff([], [], 10, 0) == []
    assert rep.percentile_interval([], 0.95) is None
    assert rep.percentile_interval([0.0, 1.0], 0.5) == (0.25, 0.75)
    empty = rep.summarize([], 10, 0)
    assert empty["n"] == 0 and rep.render([], "A", "B", "c7_share", 10, 0) == "pairs: 0"
    assert rep.bounds([], "A", "B") == {"n": 0}


@pytest.mark.parametrize("arms,room_s,runs", [(("A", "B"), 799, False), (("A", "B"), 800, True),
                                              (("A", "B", "C"), 1199, False), (("A", "B", "C"), 1200, True)])
def test_stop_at_is_sized_by_arms_times_the_live_timeout(tmp_path, stored, blocks, arms, room_s, runs):
    clock = FakeClock(NOW.timestamp())
    calls = []
    stats = _run(stored, blocks, tmp_path / "ab.jsonl", [1], clock, timeout_s=400, arms=arms,
                 deadline=NOW + timedelta(seconds=room_s), call=_fake_call(clock, seconds=1, log=calls))
    assert (stats["written"] == 1 and len(calls) == len(arms)) if runs else \
           (stats["stopped"] == "stop_at" and not calls)


LIVE_RUN = dict(RUN, model="qwen2.5:72b", live_timeout_s=400,
                options={"num_ctx": 32768, "temperature": 0.2, "repeat_penalty": 1.1, "num_predict": 1600})


@pytest.mark.parametrize("change", [
    {"options": dict(LIVE_RUN["options"], num_ctx=8192)},
    {"options": dict(LIVE_RUN["options"], temperature=0.0)},
    {"options": dict(LIVE_RUN["options"], repeat_penalty=1.0)},
    {"options": dict(LIVE_RUN["options"], num_predict=1200)},
    {"arm_a_num_predict": 1200},
    {"model": "qwen2.5:32b"},
    {"live_timeout_s": 300},
])
def test_report_stops_when_live_settings_change_between_resumes(tmp_path, capsys, change):
    path = tmp_path / "r.jsonl"
    _write(path, [LIVE_RUN, _pair(1, 7, 5), dict(LIVE_RUN, run_id="r2", **change),
                  _pair(2, 5, 5, run_id="r2")])
    with pytest.raises(SystemExit):
        rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    assert "differ in configuration" in capsys.readouterr().err
    rep.main([str(path), "--primary", "c7_share", "--boot", "20", "--split"])
    assert "configuration 2 of 2" in capsys.readouterr().out


def test_num_predict_by_arm_in_the_config():
    assert rep.run_config(dict(LIVE_RUN, arm_a_num_predict=1200))["num_predict_by_arm"] == {"A": 1200, "B": 1600}
    assert rep.run_config(dict(LIVE_RUN, arms="ABC"))["num_predict_by_arm"] == {"A": 1600, "B": 1600, "C": 1600}


def test_primary_is_required_and_printed_first(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    _write(path, [LIVE_RUN, _pair(1, 7, 5, "OPEN", "WATCH"), _pair(2, 6, 5)])
    with pytest.raises(SystemExit):
        rep.main([str(path)])
    capsys.readouterr()
    rep.main([str(path), "--primary", "mean_conviction", "--boot", "50"])
    out = capsys.readouterr().out
    sec = out.index("secondary:")
    assert out.index("primary (mean_conviction):") < out.index("mean_conviction: A") < sec
    for m in ("c7_share:", "action_change:", "logprob_ev:"):
        assert out.index(m) > sec


def test_logprob_ev_not_computable_is_said_and_excluded():
    pairs = [_pair(1, 7, 5), _pair(2, 6, 5)]
    for r in pairs:
        for a in r["arms"].values():
            a["conviction_logprobs"] = None
    text = rep.render(pairs, "A", "B", "logprob_ev", 50, 0)
    assert "logprob_ev: cannot be computed from the stored output" in text and "excluded" in text
    assert text.index("primary (logprob_ev):") < text.index("cannot be computed")
    ok = rep.metric_lines("logprob_ev", [_pair(1, 7, 5), _pair(2, 6, 5)], "A", "B", 50, 0)
    assert ok[0].startswith("logprob_ev: expected conviction A 6.25, B 4.75 (n 2 of 2 pairs")


def test_secondary_metric_math():
    # mean difference: d = (-2, -1, -3); exact sign-flip p: only the two
    # all-same-sign flips reach |sum| = 6, so 2 of 8
    t = rep.mean_diff_test([7, 6, 8], [5, 5, 5], 100, 0)
    assert t["diff"] == pytest.approx(-2.0) and t["p"] == pytest.approx(2 / 8)
    # action change: 3 OPEN->WATCH, 0 WATCH->OPEN, 1 unchanged
    pairs = [_pair(i, 7, 5, "OPEN", "WATCH") for i in range(3)] + [_pair(9, 5, 5, "PASS", "PASS")]
    a = rep.action_change_test(pairs, "A", "B", 100, 0)
    assert (a["changed"], a["share"], a["bowker"]) == (3, 0.75, 3.0)
    assert a["p"] == pytest.approx(2 / 8)
    assert rep.bowker_stat([("OPEN", "WATCH"), ("WATCH", "OPEN")]) == 0.0
    assert rep.sign_flip_p(lambda f: 0.0, 0, 10, 0) == 1.0
    # the Monte Carlo branch (n > 16) is seeded
    d = [1.0] * 20

    def stat(f):
        return abs(sum(fi * di for fi, di in zip(f, d)))
    p1 = rep.sign_flip_p(stat, 20, 500, 3)
    assert p1 == rep.sign_flip_p(stat, 20, 500, 3) and 0 < p1 <= 2 / 501 + 1e-9


def test_report_labels_action_test_and_secondary_status():
    pairs = [_pair(1, 7, 5, "OPEN", "WATCH"), _pair(2, 6, 5)]
    text = rep.render(pairs, "A", "B", "c7_share", 50, 0)
    assert "direction of changes (symmetry), p = " in text and "not a test of the change share" in text
    sec = text.index("secondary:")
    note = text.index("only the primary metric is confirmatory")
    assert sec < note < text.index("mean_conviction:") and "no multiplicity correction" in text


# -- repeat arm A2 (noise floor) ---------------------------------------------------

def test_repeat_ids_are_off_all_or_a_seeded_subsample():
    ids = list(range(100, 140))
    assert ab.repeat_ids_for(ids, None, None, 1) == frozenset()
    assert ab.repeat_ids_for(ids, "A2", None, 1) == frozenset(ids)
    sub = ab.repeat_ids_for(ids, "A2", 10, 1)
    assert len(sub) == 10 and sub <= set(ids)
    assert sub == ab.repeat_ids_for(list(reversed(ids)), "A2", 10, 1)   # stable on resume
    assert sub != ab.repeat_ids_for(ids, "A2", 10, 2)
    with pytest.raises(SystemExit):
        ab.repeat_ids_for(ids, "A2", 41, 1)


def test_repeat_args(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("1\n")
    base = ["--seed", "1", "--signal-ids-file", str(ids)]
    assert ab.parse_args(base).repeat_arm is None
    with pytest.raises(SystemExit):
        ab.parse_args(base + ["--repeat-subsample", "5"])
    with pytest.raises(SystemExit):
        ab.parse_args(base + ["--repeat-arm", "A2", "--repeat-subsample", "0"])
    with pytest.raises(SystemExit):
        ab.parse_args(base + ["--repeat-arm", "B2"])
    a = ab.parse_args(base + ["--repeat-arm", "A2", "--repeat-subsample", "5"])
    assert (a.repeat_arm, a.repeat_subsample) == ("A2", 5)


def test_repeat_arm_is_a_second_call_of_arm_a(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    sent = []
    stats = _run(stored, blocks, out, [5, 6], clock, sent, repeat_ids=frozenset({5}), arm_a_num_predict=1200)
    recs = _lines(out)
    assert stats["written"] == 2 and len(sent) == 5
    r5, r6 = recs
    assert sorted(r5["arms"]) == ["A", "A2", "B"] and r5["repeat_arm"] == "A2"
    assert sorted(r6["arms"]) == ["A", "B"] and r6["repeat_arm"] is None
    assert r5["order"] == list(ab.arm_order(1, 5, ("A", "B", "A2")))
    p = ab.build_pair(stored, *blocks)
    a2_call = sent[r5["order"].index("A2")]
    assert a2_call[0] == p["A"] and a2_call[1]["num_predict"] == 1200   # arm A's prompt and options
    assert "sha_A2" not in r5


@pytest.mark.parametrize("room_s,runs", [(1199, False), (1200, True)])
def test_repeat_arm_counts_in_the_stop_time(tmp_path, stored, blocks, room_s, runs):
    clock = FakeClock(NOW.timestamp())
    calls = []
    stats = _run(stored, blocks, tmp_path / "ab.jsonl", [1], clock, timeout_s=400, repeat_ids=frozenset({1}),
                 deadline=NOW + timedelta(seconds=room_s), call=_fake_call(clock, seconds=1, log=calls))
    assert (len(calls) == 3) if runs else (stats["stopped"] == "stop_at" and not calls)


def _with_a2(p, conv, action="WATCH"):
    p["arms"]["A2"] = _arm(conv, action, 3)
    p["repeat_arm"] = "A2"
    return p


def test_report_prints_the_noise_floor(tmp_path, capsys):
    rows = [dict(LIVE_RUN, repeat_arm="A2", repeat_subsample=3),
            _with_a2(_pair(1, 7, 5, "OPEN", "WATCH"), 7, "OPEN"),      # A=A2; A vs B differ
            _with_a2(_pair(2, 7, 5, "OPEN", "WATCH"), 6, "WATCH"),     # A vs A2 differ
            _with_a2(_pair(3, 5, 5), 5),
            _pair(4, 8, 4, "OPEN", "PASS")]                            # no A2
    path = tmp_path / "r.jsonl"
    _write(path, rows)
    rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    out = capsys.readouterr().out
    assert "noise floor on 3 signals with arm A2 (of 3 with it called):" in out
    assert "A vs A2 (same prompt, two draws): c7+ discordant 1/3 = 0.333, action differs 1/3 = 0.333" in out
    assert "A vs B on the same signals:          c7+ discordant 2/3 = 0.667, action differs 2/3 = 0.667" in out
    assert "A2 (3 signals)" in out
    last = [r for r in rows[1:]]
    assert rep.noise_floor_lines([_pair(9, 7, 5)], "A", "B", False) == []
    assert rep.noise_floor_lines(last, "A", "B", False)[0].startswith("noise floor on 3")


def test_a2_only_failure_keeps_status_ok_and_is_not_redrawn(tmp_path, stored, blocks):
    out = tmp_path / "ab.jsonl"
    clock = FakeClock(NOW.timestamp())
    order = ab.arm_order(1, 5, ("A", "B", "A2"))
    calls = []

    def call(live, prompt, timeout_s, options=None):
        calls.append(prompt)
        clock.t += 10
        if len(calls) == order.index("A2") + 1:
            return {"action": None, "conviction": None, "error": "timeout after 400 s", "wall_s": 10}
        return {"action": "WATCH", "conviction": 6, "error": None, "wall_s": 10}
    _run(stored, blocks, out, [5], clock, call=call, repeat_ids=frozenset({5}))
    rec = _lines(out)[0]
    assert rec["status"] == "ok" and rec["repeat_status"] == "error"
    assert rec["arms"]["A2"]["error"].startswith("timeout")
    calls.clear()
    stats = _run(stored, blocks, out, [5], clock, call=call, repeat_ids=frozenset({5}), retry_errors=True)
    assert stats["skipped_done"] == 1 and stats["written"] == 0 and not calls
    # a failed comparison arm is still redrawn; repeat_status is None without A2
    out2 = tmp_path / "b.jsonl"
    _run(stored, blocks, out2, [7], clock, call=_fake_call(clock, error="ConnectionError"))
    assert _lines(out2)[0]["status"] == "error" and _lines(out2)[0]["repeat_status"] is None
    stats = _run(stored, blocks, out2, [7], clock, retry_errors=True)
    assert stats["written"] == 1 and _lines(out2)[-1]["attempt"] == 2


def test_noise_floor_by_a2_position():
    def rec(sid, a, a2, pos_a, pos_b, pos_a2):
        p = _pair(sid, a, 5)
        p["arms"]["A"]["position"], p["arms"]["B"]["position"] = pos_a, pos_b
        p["arms"]["A2"] = _arm(a2, pos=pos_a2)
        return p
    last = [rec(1, 7, 7, 1, 3, 2),     # right after A, same c7
            rec(2, 7, 5, 2, 1, 3),     # right after A, differs
            rec(3, 7, 5, 2, 3, 1),     # before A, differs
            rec(4, 5, 5, 1, 2, 3)]     # not right after A, same
    lines = rep.noise_floor_lines(last, "A", "B", False)
    assert lines[-1].startswith("  A vs A2 by position (right after A shares A's prompt cache): ")
    assert "right after A: c7+ discordant 1/2 = 0.500" in lines[-1]
    assert "not: c7+ discordant 1/2 = 0.500" in lines[-1]


def test_runs_pool_across_heads_but_not_across_tool_sha(tmp_path, capsys):
    r1 = dict(LIVE_RUN, tool_sha="t1", head="h1")
    path = tmp_path / "r.jsonl"
    _write(path, [r1, _pair(1, 7, 5), dict(r1, run_id="r2", head="h2"), _pair(2, 5, 5, run_id="r2")])
    rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    out = capsys.readouterr().out
    assert "######## configuration" not in out and "attempted 2" in out
    assert "head" not in rep.run_config(r1)
    _write(path, [r1, _pair(1, 7, 5), dict(r1, run_id="r2", tool_sha="t2"), _pair(2, 5, 5, run_id="r2")])
    with pytest.raises(SystemExit):
        rep.main([str(path), "--primary", "c7_share", "--boot", "20"])
    assert "differ in configuration" in capsys.readouterr().err


def test_tool_sha_covers_the_script_and_both_schema_files(tmp_path):
    files = []
    for i, name in enumerate(("s.py", "b.txt", "a.txt")):
        p = tmp_path / name
        p.write_bytes(f"x{i}".encode())
        files.append(str(p))
    import hashlib
    assert ab.tool_sha(files, sources=[]) == hashlib.sha256(b"x0x1x2").hexdigest()
    assert ab.tool_sha(files, sources=["f1", "f2", "f3"]) == hashlib.sha256(b"x0x1x2f1f2f3").hexdigest()
    before = ab.tool_sha(files)
    (tmp_path / "a.txt").write_bytes(b"changed")
    assert ab.tool_sha(files) != before
    assert [os.path.basename(p) for p in ab.TOOL_FILES] == [
        "prompt_order_ab.py", "prompt_order_ab_new_schema.txt", "prompt_ab_old_schema.txt"]
    assert len(ab.tool_sha()) == 64


@pytest.mark.parametrize("name", ab.TOOL_FUNCS)
def test_tool_sha_changes_when_an_analyst_parsing_function_changes(monkeypatch, name):
    import inspect
    from agents import analyst
    before = ab.tool_sha()
    assert ab._analyst_sources() == [inspect.getsource(getattr(analyst, n)) for n in ab.TOOL_FUNCS]
    real = inspect.getsource

    def changed(obj):
        src = real(obj)
        return src + "    # edited\n" if obj is getattr(analyst, name) else src
    monkeypatch.setattr(inspect, "getsource", changed)
    assert ab.tool_sha() != before
