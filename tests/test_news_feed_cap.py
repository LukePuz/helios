"""Each news source reads only its first 8 items; the poll logs when that cap
was full of unseen in-window items, and how many unseen items the age filter
dropped (audit 11-06 19:37 #3). Log-only."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, patch

from ingestion import news


def _rss(n, age_h=1):
    pub = format_datetime(datetime.now(timezone.utc) - timedelta(hours=age_h))
    items = "".join(f"<item><title>Nvidia headline {i}</title><link>https://x/{age_h}/{i}</link>"
                    f"<pubDate>{pub}</pubDate></item>" for i in range(n))
    return MagicMock(content=f"<rss><channel>{items}</channel></rss>".encode(), raise_for_status=lambda: None)


def _poll(resp):
    news._CAP_HITS.clear()
    news._AGED_OUT.clear()
    with patch("ingestion.news.requests.get", return_value=resp), \
         patch("ingestion.news._is_seen", return_value=False), \
         patch("ingestion.news._is_seen_title", return_value=False):
        return news._google_news_rss("NVDA", "Nvidia")


def test_a_full_cap_of_fresh_items_on_a_longer_feed_is_a_cap_hit():
    assert len(_poll(_rss(12))) == 8
    assert news._CAP_HITS == {"google_news": ["NVDA"]}


def test_a_short_feed_is_not_a_cap_hit():
    _poll(_rss(5))
    assert news._CAP_HITS == {}


def test_unseen_items_past_the_age_window_are_counted():
    assert _poll(_rss(12, age_h=60)) == []
    assert news._AGED_OUT == {"google_news:NVDA": 8} and news._CAP_HITS == {}


def test_poll_logs_both_summaries(caplog):
    def g(ticker, company=""):
        news._CAP_HITS.setdefault("google_news", []).append(ticker)
        news._AGED_OUT[f"google_news:{ticker}"] = 3
        return []
    with patch("knowledge_loader.get_all_tickers", return_value=["NVDA"]), \
         patch("knowledge_loader.load_watchlist", return_value={}), \
         patch("ingestion.news._yfinance_news", return_value=[]), \
         patch("ingestion.news._google_news_rss", side_effect=g), \
         patch("ingestion.news.time.sleep"), caplog.at_level("INFO"):
        news.poll_once()
    assert "News feed cap (8) hit" in caplog.text and "'google_news': ['NVDA']" in caplog.text
    assert "News age filter dropped 3 unseen item(s)" in caplog.text and "google_news:NVDA" in caplog.text
    assert "a lower bound" in caplog.text
