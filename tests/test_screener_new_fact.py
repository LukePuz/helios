"""The 7B screener's new_fact field: true for an item reporting new facts,
false for opinion/recap/listicle/price-move commentary, None when missing,
invalid or not judged (keyword fallback). Stored in the screener provenance
(SCREENED_OUT rows) and on the event verdict's candidate and trigger."""
import json
from unittest.mock import patch

import pytest

from agents import crew, screener

_EVENT = {"ticker": "NVDA", "source": "news", "title": "NVIDIA signs supply deal", "text": "x"}


def _screen(reply):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value=None):
        return screener.screen(dict(_EVENT))


def _reply(**extra):
    return json.dumps({"relevant": True, "urgency": "eod", "affected_tickers": ["NVDA"],
                       "reason": "deal", "indirect_effects": None, **extra})


def test_prompt_asks_for_new_fact():
    p = screener._SCREENER_PROMPT
    assert '"new_fact": true or false' in p
    assert "rating change with a stated reason" in p and "commentary on a price move" in p
    for field in ("relevant", "urgency", "affected_tickers", "reason", "indirect_effects"):
        assert f'"{field}":' in p                                 # every existing field kept


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("true", True), ("FALSE", False), (" True ", True),
    (None, None), ("yes", None), (1, None), (0, None), ("", None), ([], None),
])
def test_parse_new_fact(value, expected):
    assert screener.parse_new_fact(value) is expected


@pytest.mark.parametrize("value, expected", [(True, True), (False, False), ("maybe", None)])
def test_screen_returns_new_fact(value, expected):
    r = _screen(_reply(new_fact=value))
    assert r["new_fact"] is expected and r["provenance"]["new_fact"] is expected
    assert (r["relevant"], r["urgency"], r["reason"]) == (True, "eod", "deal")


def test_missing_new_fact_is_none():
    r = _screen(_reply())
    assert r["new_fact"] is None and r["provenance"]["new_fact"] is None
    assert r["relevant"] is True


def test_keyword_fallback_and_bad_json_are_none():
    assert _screen(None)["new_fact"] is None                       # Ollama unavailable
    assert _screen("not json at all")["new_fact"] is None
    assert _screen('{"relevant": tru')["new_fact"] is None


def test_screened_out_row_carries_new_fact():
    with patch("agents.screener._call_ollama",
               return_value='{"relevant": false, "urgency": "eod", "reason": "recap", "new_fact": false}'), \
         patch("feed.emit"), patch("db.get_position_stop", return_value=None), \
         patch("vector_store.store"), patch("db.log_signal") as log_signal:
        crew.screen_event_only(dict(_EVENT))
    assert log_signal.call_args.args[1] == "SCREENED_OUT"
    assert log_signal.call_args.kwargs["provenance"]["new_fact"] is False


@pytest.mark.parametrize("screening, expected", [
    ({"relevant": True, "new_fact": True}, True),
    ({"relevant": True, "new_fact": False}, False),
    ({"relevant": True}, None),
    (None, None),
])
def test_deep_analyze_records_new_fact_on_candidate_and_trigger(screening, expected):
    seen = {}

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        seen["candidate"], seen["trigger"] = candidate, trigger
        return {"action": "HOLD", "ticker": "NVDA", "conviction": 5, "pass_to_risk": False}
    with patch("ibkr.IBKRClient", side_effect=RuntimeError("no broker")), \
         patch("agents.analyst.analyze", fake_analyze), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("agents.crew._c6_bar_unrouted"):
        crew.deep_analyze(dict(_EVENT), screening)
    assert seen["candidate"]["new_fact"] is expected
    assert seen["trigger"]["new_fact"] is expected and seen["trigger"]["path"] == "event"
