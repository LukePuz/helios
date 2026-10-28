"""ETFs are skipped by the analyst-ratings poller (2026-10-28)."""
from unittest.mock import patch

import knowledge_loader
from ingestion import analyst_ratings


def test_watchlist_etfs_detected():
    etfs = knowledge_loader.get_etf_tickers()
    assert {"DRAM", "QTUM"} <= etfs
    assert "NVDA" not in etfs


def test_poller_skips_etfs():
    with patch("knowledge_loader.get_all_tickers", return_value=["DRAM", "NVDA"]), \
         patch("knowledge_loader.get_etf_tickers", return_value={"DRAM"}), \
         patch("ingestion.analyst_ratings._fetch_for_ticker", return_value=[]) as fetch, \
         patch("ingestion.analyst_ratings.time.sleep"):
        analyst_ratings.poll_once()
    assert [c.args[0] for c in fetch.call_args_list] == ["NVDA"]
