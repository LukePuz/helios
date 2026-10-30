"""Luke D4 (10-29): the scanner's catalyst points come from 8-Ks the EDGAR
poller saw and from the earnings calendar (they were 0 on every scan since June)."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import db
from agents import scanner


def test_recent_filings_read_from_seen_items():
    with patch("db.recent_filing_tickers", return_value={"NVDA"}) as q:
        assert scanner._get_recent_filings() == {"NVDA"}
    q.assert_called_once_with("8-K", days=3)


def _cat(ticker, filings=(), earnings=None, tier="Watch"):
    with patch("db.get_next_earnings", return_value=earnings), \
         patch("ingestion.congressional.get_gov_contracts", return_value=[]), \
         patch("ingestion.options_flow.get_darkpool_activity", return_value={}):
        return scanner._score_catalyst({"ticker": ticker, "tier": tier, "role": "Accelerator"}, list(filings))


def test_8k_and_near_earnings_score():
    score, notes, kind = _cat("NVDA", ["NVDA"], date.today() + timedelta(days=10))
    assert score == 25 and kind == "CATALYST" and "earnings in 10d" in notes


def test_far_or_missing_earnings_scores_nothing():
    assert _cat("NVDA", earnings=date.today() + timedelta(days=40))[0] == 0
    assert _cat("NVDA", earnings=None)[0] == 0


def test_mark_seen_stores_ticker_and_form():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.mark_seen("edgar", "urn:x", ticker="NVDA", form="8-K")
    sql, params = cur.execute.call_args.args
    assert "ticker, form" in sql and params == ("edgar", "urn:x", "NVDA", "8-K")


def test_dispatch_passes_edgar_ticker_and_form():
    import tasks
    with patch.object(tasks.screen_event, "apply_async"), patch("db.mark_seen") as ms:
        tasks._dispatch_event({"ticker": "NVDA", "form": "8-K", "_seen_key": "urn:x"}, seen_source="edgar")
    ms.assert_called_once_with("edgar", "urn:x", ticker="NVDA", form="8-K")
