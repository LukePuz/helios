"""News ingestion dedupe and staleness (limitations audit 2026-10-28, finding 6)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ingestion import news


def test_publisher_suffix_does_not_change_the_title_key():
    assert news._title_key("Tesla beats on earnings - Reuters") == news._title_key("Tesla beats on earnings")


def test_old_items_are_too_old_and_unknown_dates_are_kept():
    old = datetime.now(timezone.utc) - timedelta(days=4)
    assert news._too_old(old)
    assert not news._too_old(datetime.now(timezone.utc) - timedelta(hours=2))
    assert not news._too_old(None)


def test_rfc822_and_iso_dates_parse():
    assert news._parse_published("Mon, 26 Oct 2026 13:05:00 GMT").year == 2026
    assert news._parse_published("2026-10-26T13:05:00Z").hour == 13


def test_same_headline_from_both_feeds_is_kept_once():
    item = lambda t: {"ticker": "TSLA", "title": t, "_seen_keys": []}
    with patch("knowledge_loader.get_all_tickers", return_value=["TSLA"]), \
         patch("ingestion.news._yfinance_news", return_value=[item("Tesla beats on earnings")]), \
         patch("ingestion.news._google_news_rss", return_value=[item("Tesla beats on earnings - Reuters")]), \
         patch("ingestion.news.time.sleep"):
        assert len(news.poll_once()) == 1
