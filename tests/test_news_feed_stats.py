"""Per-feed counts and failures for each news poll, so a feed that fails or
dries up shows in the logs (audit 2026-11-07 03:08 #5). Log-only."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, patch

from ingestion import news


def _rss(n):
    pub = format_datetime(datetime.now(timezone.utc) - timedelta(hours=1))
    items = "".join(f"<item><title>Nvidia headline {i}</title><link>https://x/{i}</link>"
                    f"<pubDate>{pub}</pubDate></item>" for i in range(n))
    return MagicMock(content=f"<rss><channel>{items}</channel></rss>".encode(), raise_for_status=lambda: None)


def _fresh(**patches):
    news._FEED_STATS.clear()
    return (patch("ingestion.news._is_seen", return_value=False),
            patch("ingestion.news._is_seen_title", return_value=False))


def test_google_counts_raw_and_returned_and_a_failure():
    a, b = _fresh()
    with a, b, patch("ingestion.news.requests.get", return_value=_rss(3)):
        news._google_news_rss("NVDA", "Nvidia")
    with a, b, patch("ingestion.news.requests.get", side_effect=TimeoutError("slow")):
        news._google_news_rss("AMD", "AMD")
    st = news._FEED_STATS["google_news"]
    assert (st["tickers"], st["raw"], st["returned"], st["failed"]) == (2, 3, 3, 1)
    assert st["first_error"] == "AMD: TimeoutError: slow"


def test_yfinance_failure_is_counted():
    a, b = _fresh()
    with a, b, patch("ingestion.news.yf.Ticker", side_effect=RuntimeError("blocked")):
        assert news._yfinance_news("NVDA") == []
    st = news._FEED_STATS["yfinance"]
    assert st["failed"] == 1 and st["tickers"] == 1 and "blocked" in st["first_error"]


def _poll_with(y, g, caplog):
    with patch("knowledge_loader.get_all_tickers", return_value=["NVDA", "AMD"]), \
         patch("knowledge_loader.load_watchlist", return_value={}), \
         patch("ingestion.news._yfinance_news", side_effect=y), \
         patch("ingestion.news._google_news_rss", side_effect=g), \
         patch("ingestion.news.time.sleep"), caplog.at_level("INFO"):
        news.poll_once()


def test_poll_logs_one_line_and_warns_when_a_feed_failed_for_every_ticker(caplog):
    def y(t):
        news._note_feed("yfinance", t, error=RuntimeError("blocked"))
        return []

    def g(t, company=""):
        news._note_feed("google_news", t, raw=4, returned=1)
        return []
    _poll_with(y, g, caplog)
    assert ("News feeds this poll (log-only): google_news tickers 2 raw 8 returned 2 failed 0; "
            "yfinance tickers 2 raw 0 returned 0 failed 2") in caplog.text
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1 and "yfinance failed for all 2" in warns[0].getMessage()
    assert "NVDA: RuntimeError: blocked" in warns[0].getMessage()


def test_a_partial_failure_does_not_warn(caplog):
    def y(t):
        news._note_feed("yfinance", t, raw=0 if t == "AMD" else 3,
                        error=RuntimeError("x") if t == "AMD" else None)
        return []

    def g(t, company=""):
        news._note_feed("google_news", t, raw=1)
        return []
    _poll_with(y, g, caplog)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_feed_that_reads_nothing_for_every_ticker_warns(caplog):
    def y(t):
        news._note_feed("yfinance", t, raw=0, returned=0)
        return []

    def g(t, company=""):
        news._note_feed("google_news", t, raw=2 if t == "NVDA" else 0)
        return []
    _poll_with(y, g, caplog)
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warns == ["News feed yfinance read no items for any of 2 ticker(s) this poll (0 failed)"]
