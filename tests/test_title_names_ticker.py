"""Screener rows record whether the title names the ticker or its company
(audit 11-06 09:31 #3): the held-negative override fires on tone alone."""
from unittest.mock import patch

from agents import screener

_WL = {"NVDA": {"company": "Nvidia"}, "ANET": {"company": "Arista Networks"}}


def test_names_the_company_or_ticker():
    with patch("knowledge_loader.load_watchlist", return_value=_WL):
        assert screener._title_names_ticker("NVDA", "Nvidia Stock Rises on Dip-Buying") is True
        assert screener._title_names_ticker("ANET", "Should You Add ANET Stock to Your Portfolio") is True


def test_market_wide_recap_does_not_name_it():
    with patch("knowledge_loader.load_watchlist", return_value=_WL):
        assert screener._title_names_ticker(
            "NVDA", "Stock Market Today: Dow Slides 500 Points Ahead Of Jobs Report") is False


def test_unreadable_watchlist_is_none():
    with patch("knowledge_loader.load_watchlist", side_effect=OSError("missing")):
        assert screener._title_names_ticker("NVDA", "Nvidia falls") is None
