"""Sampling options sent, Ollama version and a refreshed model digest are
recorded in provenance (M7). The values sent are unchanged."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import requests

from agents import analyst


def _ok_response(text="{}"):
    resp = MagicMock()
    resp.json.return_value = {"response": text, "prompt_eval_count": 10, "eval_count": 5,
                              "done_reason": "stop", "total_duration": 1}
    return resp


def test_options_sent_are_unchanged_and_recorded():
    with patch("agents.analyst.requests.post", return_value=_ok_response()) as post:
        analyst._call_ollama("p", num_ctx=8192)
    sent = post.call_args.kwargs["json"]
    assert sent["options"] == {"temperature": 0.2, "num_predict": analyst._NUM_PREDICT,
                               "num_ctx": 8192, "repeat_penalty": 1.1}
    assert "seed" not in sent["options"] and sent["top_logprobs"] == 5
    assert analyst._LAST_CALL_SENT == {"options": sent["options"], "seed": None, "top_logprobs": 5}


def test_options_are_recorded_for_a_failed_call_too():
    analyst._LAST_CALL_SENT.clear()
    with patch("agents.analyst.requests.post", side_effect=requests.exceptions.Timeout()):
        assert analyst._call_ollama("p") is None
    assert analyst._LAST_CALL_SENT["options"]["num_ctx"] == analyst._NUM_CTX
    assert analyst._LAST_CALL_STATS == {}


def _identity_mocks(digest, version="0.12.3"):
    def get(url, timeout):
        r = MagicMock()
        if url.endswith("/api/tags"):
            r.json.return_value = {"models": [{"name": analyst.ANALYST_MODEL, "digest": digest}]}
        else:
            r.json.return_value = {"version": version}
        return r
    show = MagicMock()
    show.json.return_value = {"parameters": 'stop "<|im_end|>"\ntop_p 0.8', "template": "{{ .Prompt }}",
                              "details": {"quantization_level": "Q4_K_M"}}
    return get, show


def test_identity_records_version_parameters_and_refreshes_the_digest(monkeypatch):
    monkeypatch.setattr(analyst, "_OLLAMA_ID", {"at": 0.0, "value": None})
    get, show = _identity_mocks("aaaaaaaaaaaaaaaa")
    with patch.object(analyst.requests, "get", side_effect=get), \
         patch.object(analyst.requests, "post", return_value=show):
        first = analyst._ollama_identity()
    assert first["model_digest"] == "aaaaaaaaaaaa" and first["ollama_version"] == "0.12.3"
    assert "top_p 0.8" in first["model_parameters"] and first["model_quantization"] == "Q4_K_M"
    assert len(first["model_template_sha"]) == 12
    # Within the TTL: cached, no new requests.
    with patch.object(analyst.requests, "get", side_effect=AssertionError("refetched")), \
         patch.object(analyst.requests, "post", side_effect=AssertionError("refetched")):
        assert analyst._ollama_identity()["model_digest"] == "aaaaaaaaaaaa"
    # After the TTL: re-read, so a pulled model shows its new digest.
    analyst._OLLAMA_ID["at"] -= analyst._OLLAMA_ID_TTL_S + 1
    get, show = _identity_mocks("bbbbbbbbbbbbbbbb")
    with patch.object(analyst.requests, "get", side_effect=get), \
         patch.object(analyst.requests, "post", return_value=show):
        assert analyst._ollama_identity()["model_digest"] == "bbbbbbbbbbbb"


def test_tags_failure_skips_the_other_calls_and_is_cached_briefly(monkeypatch):
    monkeypatch.setattr(analyst, "_OLLAMA_ID", {"at": 0.0, "value": None})
    get = MagicMock(side_effect=requests.exceptions.ConnectionError())
    post = MagicMock()
    with patch.object(analyst.requests, "get", get), patch.object(analyst.requests, "post", post):
        assert analyst._ollama_identity()["model_digest"] is None
        assert get.call_count == 1 and not post.called          # no /api/version, no /api/show
        assert get.call_args.kwargs["timeout"] == (2, 3)
        analyst._ollama_identity()                               # within 60 s: cached
        assert get.call_count == 1
        analyst._OLLAMA_ID["at"] -= analyst._OLLAMA_ID_FAIL_TTL_S + 1
        analyst._ollama_identity()                               # retried after it
        assert get.call_count == 2


def test_verdict_provenance_carries_the_options_and_identity():
    body = json.dumps({"action": "WATCH", "conviction": 5, "reasoning": "r", "sources": []})
    log_signal = MagicMock()
    with ExitStack() as stack:
        for p in (patch("agents.analyst.requests.post", return_value=_ok_response(body)),
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
    prov = log_signal.call_args.kwargs["provenance"]
    assert prov["llm_call"]["options"]["temperature"] == 0.2
    assert prov["llm_call"]["seed"] is None
    assert prov["model_digest"] == "testdigest" and prov["ollama_version"] == "test"
