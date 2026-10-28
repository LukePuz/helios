"""SCREENED_OUT rows carry the screener model and code version (SEF run 7)."""
from unittest.mock import patch

from agents import crew, screener


def test_screen_result_has_provenance():
    with patch("agents.screener._call_ollama", return_value='{"relevant": false, "urgency": "low", "reason": "noise"}'), \
         patch("feed.emit"):
        result = screener.screen({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"})
    prov = result["provenance"]
    assert prov["model"] == screener.SCREENER_MODEL and prov["code_sha"] == "test"
    assert prov["head_sha"] == "testhead"  # SEF run 12: was missing on screener rows
    assert prov["model_digest"] == "testdigest"
    assert len(prov["prompt_template_sha"]) == 12 and len(prov["prompt_sha"]) == 12


def test_screened_out_signal_logs_model_and_provenance():
    prov = {"code_sha": "abc", "model": "qwen2.5:7b"}
    with patch("vector_store.store"), \
         patch("agents.screener.screen", return_value={"relevant": False, "reason": "noise", "provenance": prov}), \
         patch("db.log_signal") as log_signal:
        crew.screen_event_only({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"})
    assert log_signal.call_args.kwargs == {"model": "qwen2.5:7b", "provenance": prov}
