import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import g33_resume_check as rc  # noqa: E402

RUN = {"kind": "run", "run_id": "r1", "started_at": "2026-11-04T19:12:00", "seed": 1, "timeout_s": 400,
       "arms": "ABC", "arm_a_num_predict": None, "model_digest": "d1", "ollama_version": "0.9",
       "schema_sha_A": "a", "schema_sha_B": "b", "schema_sha_C": "c", "tool_sha": "t", "model": "m",
       "live_timeout_s": 400, "repeat_arm": "A2", "repeat_subsample": 10,
       "options": {"num_ctx": 8192, "temperature": 0.2, "repeat_penalty": 1.1, "num_predict": 900}}


def _write(tmp_path, *recs):
    p = tmp_path / "ab.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(p)


def test_matching_resume_passes(tmp_path, capsys):
    pair = {"kind": "pair", "signal_id": 5, "run_id": "r1", "A": {"conviction": 7}}
    p = _write(tmp_path, RUN, pair, dict(RUN, run_id="r2", started_at="2026-11-05T19:12:00"))
    assert rc.main([p]) == 0
    out = capsys.readouterr().out
    assert "2 run line(s)" in out and "conviction" not in out


def test_digest_change_between_nights_fails(tmp_path, capsys):
    p = _write(tmp_path, RUN, dict(RUN, run_id="r2", model_digest="d2"))
    assert rc.main([p]) == 2
    assert "model_digest: 'd1' -> 'd2'" in capsys.readouterr().out


def test_option_change_fails(tmp_path):
    p = _write(tmp_path, RUN, dict(RUN, run_id="r2", options=dict(RUN["options"], temperature=0.3)))
    assert rc.main([p]) == 2


def test_no_run_line(tmp_path):
    assert rc.main([_write(tmp_path, {"kind": "pair", "signal_id": 1})]) == 1


def test_run_lines_ignores_pair_lines(tmp_path):
    p = _write(tmp_path, RUN, {"kind": "pair", "note": "kind run inside text"}, {"kind": "skip", "signal_id": 2})
    assert [r["run_id"] for r in rc.run_lines(p)] == ["r1"]


def test_preflight_compares_current_config(tmp_path, monkeypatch):
    p = _write(tmp_path, RUN)
    monkeypatch.setattr(rc, "current_run", lambda first: dict(first, tool_sha="t2"))
    assert rc.main([p, "--preflight"]) == 2
    monkeypatch.setattr(rc, "current_run", lambda first: dict(first))
    assert rc.main([p, "--preflight"]) == 0


def test_a_redrawn_sample_or_logprob_change_fails(tmp_path):
    base = dict(RUN, signal_ids=[1, 2, 3], repeat_ids=[2], logprobs=True, top_logprobs=5,
                sample={"mode": "range", "from": "a", "to": "b", "trading_env": "live", "population": 9, "n": 3})
    assert rc.main([_write(tmp_path, base, dict(base, run_id="r2", repeat_ids=[2]))]) == 0
    assert rc.main([_write(tmp_path, base, dict(base, run_id="r2", signal_ids=[1, 2, 4]))]) == 2
    assert rc.main([_write(tmp_path, base, dict(base, run_id="r2", top_logprobs=3))]) == 2
    pop = dict(base["sample"], population=10)
    assert rc.main([_write(tmp_path, base, dict(base, run_id="r2", sample=pop))]) == 2
    opts = dict(base["options"], top_k=40)
    assert rc.main([_write(tmp_path, base, dict(base, run_id="r2", options=opts))]) == 2


def test_stored_shas_keeps_only_id_and_sha(tmp_path):
    pair = {"kind": "pair", "signal_id": 7, "sha_stored": "abc", "arms": {"A": {"reply": "x"}}}
    assert rc.stored_shas(_write(tmp_path, RUN, pair)) == {7: "abc"}


def test_preflight_flags_a_rewritten_stored_prompt(tmp_path, monkeypatch, capsys):
    import prompt_order_ab as ab
    pair = {"kind": "pair", "signal_id": 7, "sha_stored": ab.sha12("old prompt")}
    p = _write(tmp_path, dict(RUN, signal_ids=[7, 8]), pair)
    monkeypatch.setattr(rc, "current_run", lambda first: dict(first))
    monkeypatch.setattr(ab, "fetch_signals", lambda ids: {7: {"prompt": "old prompt"}})
    assert rc.main([p, "--preflight"]) == 0
    monkeypatch.setattr(ab, "fetch_signals", lambda ids: {7: {"prompt": "redacted prompt"}})
    assert rc.main([p, "--preflight"]) == 2
    assert "signal ids [7]" in capsys.readouterr().out
