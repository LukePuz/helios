"""Per-poll counts and failures for the analyst ratings feed (audit
2026-11-07 03:08 #5). Log-only."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from ingestion import analyst_ratings as ar


def _df(days_ago=(1, 30)):
    now = datetime.now(timezone.utc)
    idx = pd.DatetimeIndex([now - timedelta(days=d) for d in days_ago])
    return pd.DataFrame([{"Firm": f"F{i}", "ToGrade": "Buy", "FromGrade": "Hold", "Action": "up",
                          "priceTargetAction": "", "currentPriceTarget": None, "priorPriceTarget": None}
                         for i in range(len(days_ago))], index=idx)


def _poll(by_ticker, caplog):
    def ticker(sym):
        v = by_ticker[sym]
        t = type("T", (), {})()
        if isinstance(v, Exception):
            type(t).upgrades_downgrades = property(lambda self: (_ for _ in ()).throw(v))
        else:
            t.upgrades_downgrades = v
        return t
    with patch("knowledge_loader.get_all_tickers", return_value=list(by_ticker)), \
         patch("knowledge_loader.get_etf_tickers", return_value=set()), \
         patch("yfinance.Ticker", side_effect=ticker), \
         patch("ingestion.analyst_ratings._is_seen", return_value=False), \
         patch("ingestion.analyst_ratings.time.sleep"), caplog.at_level("INFO"):
        return ar.poll_once()


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_counts_failed_empty_recent_and_returned(caplog):
    items = _poll({"NVDA": _df(), "AMD": pd.DataFrame(), "MSFT": RuntimeError("blocked")}, caplog)
    assert len(items) == 1
    assert ("Analyst ratings this poll (log-only): tickers 3 failed 1 empty 1 recent 1 returned 1"
            in caplog.text)
    assert ar._STATS["first_error"] == "MSFT: RuntimeError: blocked"
    assert _warnings(caplog) == []


def test_warns_when_every_ticker_failed(caplog):
    _poll({"NVDA": RuntimeError("x"), "AMD": RuntimeError("y")}, caplog)
    w = _warnings(caplog)
    assert len(w) == 1 and "failed for all 2" in w[0] and "NVDA: RuntimeError: x" in w[0]


def test_warns_when_no_ticker_returned_rows(caplog):
    _poll({"NVDA": pd.DataFrame(), "AMD": None, "MSFT": RuntimeError("z")}, caplog)
    assert _warnings(caplog) == ["Analyst ratings feed read no rows for any of 3 ticker(s) this poll (1 failed)"]


def test_stats_reset_each_poll(caplog):
    _poll({"NVDA": RuntimeError("x")}, caplog)
    _poll({"AMD": _df((1,))}, caplog)
    assert ar._STATS["tickers"] == 1 and ar._STATS["failed"] == 0 and ar._STATS["returned"] == 1


def test_stats_still_log_when_a_ticker_raises_mid_poll(caplog):
    import pytest
    def fetch(t):
        ar._STATS["tickers"] += 1
        if t == "AMD":
            raise ValueError("bad row")
        return []
    with patch("ingestion.analyst_ratings._fetch_for_ticker", side_effect=fetch), \
         patch("knowledge_loader.get_all_tickers", return_value=["NVDA", "AMD"]), \
         patch("knowledge_loader.get_etf_tickers", return_value=set()), \
         patch("ingestion.analyst_ratings.time.sleep"), caplog.at_level("INFO"), \
         pytest.raises(ValueError):
        ar.poll_once()
    assert "Analyst ratings this poll (log-only): tickers 2" in caplog.text
