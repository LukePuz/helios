"""Screener provenance keeps the reply's affected_tickers and
indirect_effects (audit 2026-11-07 15:07 #5). Record-only."""
from unittest.mock import patch

from agents import screener


def _prov(reply, ticker="NVDA"):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value=None):
        r = screener.screen({"ticker": ticker, "source": "news", "title": "t", "text": "x"})
    return r, r["provenance"]


def test_affected_tickers_and_indirect_effects_recorded():
    other = next(t for t in screener.SYMBOLS if t != "NVDA")
    reply = ('{"relevant": false, "urgency": "eod", "reason": "peer", '
             f'"affected_tickers": ["nvda", "{other}", "{other}", "ZZZQ"], '
             '"indirect_effects": "supplier read-through"}')
    r, prov = _prov(reply)
    assert prov["affected_tickers"] == ["NVDA", other, "ZZZQ"]
    assert prov["affected_others"] == [other, "ZZZQ"]
    assert prov["affected_others_on_watchlist"] == [other]
    assert prov["indirect_effects"] == "supplier read-through"
    assert r["relevant"] is False


def test_missing_or_bad_values_are_none():
    _, prov = _prov('{"relevant": false, "urgency": "eod", "reason": "x", '
                    '"affected_tickers": "NVDA", "indirect_effects": "null"}')
    assert prov["affected_tickers"] is None and prov["affected_others"] is None
    assert prov["affected_others_on_watchlist"] is None and prov["indirect_effects"] is None
    assert prov["affected_tickers_type"] == "str"
    _, prov = _prov('{"relevant": false, "urgency": "eod", "reason": "x"}')
    assert prov["affected_tickers"] is None and prov["indirect_effects"] is None
    assert "affected_tickers_type" not in prov
    _, prov = _prov('{"relevant": false, "urgency": "eod", "reason": "x", "affected_tickers": []}')
    assert prov["affected_tickers"] == [] and "affected_tickers_type" not in prov


def test_long_lists_and_text_are_capped():
    names = ", ".join(f'"T{i}"' for i in range(30))
    _, prov = _prov('{"relevant": false, "urgency": "eod", "reason": "x", '
                    f'"affected_tickers": [{names}], "indirect_effects": "{"a" * 500}"}}')
    assert len(prov["affected_tickers"]) == 20 and len(prov["indirect_effects"]) == 300
