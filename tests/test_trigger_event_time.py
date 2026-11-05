"""provenance.trigger records the item's own time (event_time) and which
event key it came from (event_time_source). Sources with a structured time
carry it on the event. Log-only: the brief never shows it."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.analyst import trigger_record


@pytest.mark.parametrize("event, value, key", [
    ({"timestamp": "2026-11-03T09:01:00-05:00"}, "2026-11-03T09:01:00-05:00", "timestamp"),
    ({"published_at": "2026-11-03T14:00:00+00:00"}, "2026-11-03T14:00:00+00:00", "published_at"),
    ({"event_time": "2026-11-02T00:00:00+00:00"}, "2026-11-02T00:00:00+00:00", "event_time"),
    ({"filing_date": "2026-11-01"}, "2026-11-01", "filing_date"),
    ({"date": "2026-10-30"}, "2026-10-30", "date"),
    ({"published_at": None, "filing_date": "2026-11-01"}, "2026-11-01", "filing_date"),
    ({}, None, None),
])
def test_trigger_event_time(event, value, key):
    rec = trigger_record("event", dict(event, source="x", title="t"))
    assert rec["event_time"] == value and rec["event_time_source"] == key


def test_trigger_published_at_unchanged():
    rec = trigger_record("event", {"filing_date": "2026-11-01"})
    assert rec["published_at"] is None
    assert trigger_record("event", {"timestamp": "T"})["published_at"] == "T"


def test_analyst_rating_carries_grade_date():
    from ingestion import analyst_ratings
    when = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=1)).floor("s")
    df = pd.DataFrame([{"Firm": "Acme", "ToGrade": "Buy", "FromGrade": "Hold", "Action": "up",
                        "priceTargetAction": "", "currentPriceTarget": None,
                        "priorPriceTarget": None}], index=pd.DatetimeIndex([when]))
    with patch("yfinance.Ticker", return_value=SimpleNamespace(upgrades_downgrades=df)), \
         patch("ingestion.analyst_ratings._is_seen", return_value=False):
        (item,) = analyst_ratings._fetch_for_ticker("NVDA")
    assert item["event_time"] == when.isoformat()
    assert trigger_record("event", item)["event_time_source"] == "event_time"


def test_news_items_carry_published_at():
    from ingestion import news
    pub = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=1)
    yf_item = {"content": {"uuid": "u1", "title": "NVIDIA signs deal", "summary": "s",
                           "pubDate": pub.isoformat().replace("+00:00", "Z")}}
    with patch("ingestion.news.yf.Ticker", return_value=SimpleNamespace(news=[yf_item])), \
         patch("ingestion.news._is_seen", return_value=False), \
         patch("ingestion.news._is_seen_title", return_value=False):
        (item,) = news._yfinance_news("NVDA")
    assert item["published_at"] == pub.isoformat()

    rss = (f"<rss><channel><item><title>NVIDIA Corporation signs deal</title><link>https://x/a</link>"
           f"<pubDate>{pub.strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate></item></channel></rss>")
    resp = MagicMock(content=rss.encode())
    with patch("ingestion.news.requests.get", return_value=resp), \
         patch("ingestion.news._is_seen", return_value=False), \
         patch("ingestion.news._is_seen_title", return_value=False):
        (item,) = news._google_news_rss("NVDA", "NVIDIA Corporation")
    assert item["published_at"] == pub.isoformat()
