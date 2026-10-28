"""Anthropic IPO watch (Luke, 2026-10-28): tracked, never tradeable until listed."""
from unittest.mock import MagicMock, patch

import knowledge_loader
import tasks


def test_anthropic_is_pending_and_not_in_the_tradeable_watchlist():
    pending = knowledge_loader.load_ipo_watch()
    assert [e["company"] for e in pending] == ["Anthropic"]
    assert pending[0]["status"] == "pending listing"
    names = {d["company"].lower() for d in knowledge_loader.load_watchlist().values()}
    assert "anthropic" not in names


def _search(quotes):
    return MagicMock(quotes=quotes)


def test_only_a_us_listed_equity_counts():
    noise = [{"symbol": "ANTW", "quoteType": "ETF", "exchange": "PCX", "shortname": "Anthropic AI Lab Ecosystem ETF"},
             {"symbol": "ANTHROPIC-USD", "quoteType": "CRYPTOCURRENCY", "exchange": "CCC", "shortname": "Anthropic tokenized stock"}]
    with patch("yfinance.Search", return_value=_search(noise)):
        assert tasks._find_us_listing("Anthropic") is None
    listed = noise + [{"symbol": "ANTH", "quoteType": "EQUITY", "exchange": "NMS", "longname": "Anthropic PBC"}]
    with patch("yfinance.Search", return_value=_search(listed)):
        assert tasks._find_us_listing("Anthropic")["symbol"] == "ANTH"


def test_alert_once_when_listed():
    with patch("knowledge_loader.load_ipo_watch", return_value=[{"company": "Anthropic"}]), \
         patch("tasks._find_us_listing", return_value={"symbol": "ANTH", "exchange": "NMS", "name": "Anthropic PBC"}), \
         patch("db.get_state", return_value=None), patch("db.set_state") as set_state, \
         patch("alerts._send") as send:
        tasks.check_ipo_watch()
    set_state.assert_called_once_with("ipo_watch_listed_anthropic", "ANTH")
    assert "ANTH" in send.call_args[0][0]
