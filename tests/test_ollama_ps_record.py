"""How the server held the analyst model after the call (/api/ps), recorded
in provenance.llm_call.ollama_ps, also after a failed call (audit 2026-11-07 09:07 #3). Log-only."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import requests

from agents import analyst


def _resp(body):
    r = MagicMock()
    r.json.return_value = body
    return r


_GEN = {"response": "{}", "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop",
        "total_duration": 1}


def _ps(models):
    return _resp({"models": models})


def _analyze(get_effect=None, get_value=None, post_value=None, post_effect=None):
    log_signal = MagicMock()
    with ExitStack() as stack:
        for p in (patch("agents.analyst.requests.post", return_value=post_value,
                        side_effect=post_effect),
                  patch("agents.analyst.requests.get", return_value=get_value,
                        side_effect=get_effect),
                  patch("agents.analyst._ollama_identity", return_value={}),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1}),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.log_signal", log_signal), patch("feed.emit")):
            stack.enter_context(p)
        analyst.analyze({"ticker": "MSFT", "role": "Compounder"}, situation_brief="b",
                        regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})
    return log_signal.call_args.kwargs["provenance"]


_BODY = {**_GEN, "response": json.dumps({"action": "WATCH", "conviction": 5, "reasoning": "r",
                                         "sources": []})}


def test_split_is_in_the_verdict_provenance():
    m = {"name": analyst.ANALYST_MODEL, "size": 40_000, "size_vram": 39_000,
         "context_length": 32768, "digest": "e80424694df1abcdef"}
    prov = _analyze(get_value=_ps([m]), post_value=_resp(_BODY))
    assert prov["llm_call"]["ollama_ps"] == {
        "size": 40_000, "size_vram": 39_000, "vram_share": 0.975,
        "context_length": 32768, "digest": "e80424694df1"}


def test_split_is_recorded_on_a_timed_out_call_too():
    m = {"name": analyst.ANALYST_MODEL, "size": 40_000, "size_vram": 20_000}
    prov = _analyze(get_value=_ps([m]), post_effect=requests.exceptions.Timeout())
    assert prov["llm_call"]["ollama_ps"]["vram_share"] == 0.5


def test_a_failed_ps_read_never_fails_the_verdict():
    prov = _analyze(get_effect=RuntimeError("x"), post_value=_resp(_BODY))
    assert prov["llm_call"]["ollama_ps"] is None and prov["llm_call"]["eval_count"] == 5


def test_call_ollama_source_is_untouched():
    """The read sits in analyze, not _call_ollama, whose source feeds the
    prompt template sha (test_shadow_prompts_unchanged)."""
    import inspect
    assert "_ollama_ps_state" not in inspect.getsource(analyst._call_ollama)


def test_other_models_only_or_a_failed_read_give_none():
    other = {"name": "qwen2.5:7b", "size": 1, "size_vram": 1}
    with patch("agents.analyst.requests.get", return_value=_ps([other])):
        assert analyst._ollama_ps_state() is None
    with patch("agents.analyst.requests.get", side_effect=requests.exceptions.Timeout()):
        assert analyst._ollama_ps_state() is None
    with patch("agents.analyst.requests.get", return_value=_resp(None)):
        assert analyst._ollama_ps_state() is None


def test_zero_size_gives_no_share():
    m = {"name": analyst.ANALYST_MODEL, "size": 0, "size_vram": 0}
    with patch("agents.analyst.requests.get", return_value=_ps([m])):
        assert analyst._ollama_ps_state()["vram_share"] is None


def test_a_tagless_model_name_matches_its_latest_form(monkeypatch):
    monkeypatch.setattr(analyst, "ANALYST_MODEL", "helios-analyst")
    m = {"name": "helios-analyst:latest", "size": 100, "size_vram": 50}
    with patch("agents.analyst.requests.get", return_value=_ps([m])):
        assert analyst._ollama_ps_state()["vram_share"] == 0.5
