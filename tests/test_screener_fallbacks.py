"""The 7B screener's fallbacks: a reply with no JSON is judged by the keyword
fallback, so model_used and provenance must not name the 7B; a 7B reply
without urgency gets the fallback's "eod" instead of losing the event; a
"relevant" that isn't a bool or "true"/"false" counts as missing."""
import json
from unittest.mock import patch

import pytest

from agents import crew, screener

_EVENT = {"ticker": "NVDA", "source": "news", "title": "NVIDIA wins DoD contract", "text": "x"}


def _screen(reply):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value=None):
        return screener.screen(dict(_EVENT))


def test_no_json_reply_is_recorded_as_keyword_fallback():
    r = _screen("Sure! This looks relevant to me.")
    assert r["model_used"] == "keyword_fallback"
    assert r["provenance"]["model"] == "keyword_fallback"
    assert r["provenance"]["model_digest"] is None
    assert r["provenance"]["fallback_reason"] == "no_json"
    assert r["new_fact"] is None


def test_fallback_reasons():
    assert _screen(None)["provenance"]["fallback_reason"] == "ollama_unavailable"
    assert _screen(None)["provenance"]["model_digest"] is None
    assert _screen('{"relevant": tru}')["provenance"]["fallback_reason"] == "invalid_json"
    ok = _screen('{"relevant": true, "urgency": "eod", "reason": "deal"}')
    assert ok["model_used"] == screener.SCREENER_MODEL and "fallback_reason" not in ok["provenance"]


def test_missing_urgency_defaults_to_eod():
    r = _screen('{"relevant": true, "reason": "deal", "new_fact": true}')
    assert r["relevant"] is True and r["urgency"] == "eod"
    assert r["model_used"] == screener.SCREENER_MODEL and r["new_fact"] is True
    assert r["provenance"]["urgency_missing"] is True


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ('"true"', True), ('"FALSE"', False), ('" False "', False),
])
def test_relevant_parsed_like_new_fact(value, expected):
    v = json.dumps(value) if isinstance(value, bool) else value
    r = _screen('{"relevant": %s, "urgency": "eod", "reason": "x"}' % v)
    assert r["relevant"] is expected and "relevant_invalid" not in r["provenance"]


@pytest.mark.parametrize("value", ['"no"', "1", "0", "null", '""', "[]"])
def test_invalid_relevant_is_not_relevant(value):
    r = _screen('{"relevant": %s, "urgency": "eod", "reason": "x"}' % value)
    assert not r.get("relevant") and "relevant" not in r      # as for a missing value
    assert r["provenance"]["relevant_invalid"] is True


def test_string_false_is_screened_out():
    with patch("agents.screener._call_ollama",
               return_value='{"relevant": "false", "urgency": "eod", "reason": "recap"}'), \
         patch("feed.emit"), patch("db.get_position_stop", return_value=None), \
         patch("vector_store.store"), patch("db.log_signal") as log_signal:
        out = crew.screen_event_only(dict(_EVENT))
    assert out["relevant"] is False
    assert log_signal.call_args.args[1] == "SCREENED_OUT"


def test_present_urgency_kept():
    r = _screen('{"relevant": true, "urgency": "immediate", "reason": "deal"}')
    assert r["urgency"] == "immediate" and "urgency_missing" not in r["provenance"]
