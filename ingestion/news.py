"""
News ingestion — free sources only (yfinance + Google News RSS).

Before this module, the codebase had NO general news ingestion at all.
The old news_agent.py (since removed) only computed a keyword sentiment score against Alpaca's news API and was
never wired into the scanner, analyst, or context engine. This module
fetches actual per-ticker headlines and feeds them through the same
screen → analyze → route pipeline that SEC filings and options flow use,
so the bot can react to news for the first time.
"""

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
import yfinance as yf

log = logging.getLogger(__name__)

_SOURCE = "news"
_TITLE_SOURCE = "news_title"   # cross-source dedup: same headline via both feeds
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (trading-bot; +245081114+LukePuz@users.noreply.github.com)"}


def _is_seen(item_id: str) -> bool:
    try:
        import db
        return db.is_seen(_SOURCE, item_id)
    except Exception:
        return False


def _title_key(title: str) -> str:
    """Normalized-title hash — yfinance dedups by uuid and Google RSS by
    link, so the identical headline from both sources gets two different
    IDs under the "news" source and was analyzed twice. This is a second,
    content-based dedup key shared across both fetchers."""
    normalized = re.sub(r"\W+", "", title.lower())
    return hashlib.sha1(normalized.encode()).hexdigest()


def _is_seen_title(title: str) -> bool:
    try:
        import db
        return db.is_seen(_TITLE_SOURCE, _title_key(title))
    except Exception:
        return False


def _pick_thumbnail(content: dict) -> str | None:
    """yfinance news items carry a real thumbnail ~90-100% of the time
    (verified live 2026-07-26) -- previously fetched but never captured.
    Prefers a compact resolution (~140-220px wide, good for a list
    thumbnail) over the "original" (often 1200px+, wasteful to embed in a
    headline list); falls back to the original/base URL if no resolution
    list is present."""
    thumbnail = content.get("thumbnail") or {}
    resolutions = thumbnail.get("resolutions") or []
    for res in resolutions:
        width = res.get("width") or 0
        if 140 <= width <= 220 and res.get("url"):
            return res["url"]
    return thumbnail.get("originalUrl") or (resolutions[0].get("url") if resolutions else None)


def _yfinance_news(ticker: str) -> list[dict]:
    """Fetches recent headlines for a ticker via yfinance's news feed."""
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as e:
        log.debug(f"[{ticker}] yfinance news fetch failed: {e}")
        return []

    results = []
    for item in raw_items[:8]:
        # yfinance news items nest content under different keys across versions
        content = item.get("content", item)
        item_id = content.get("uuid") or item.get("uuid") or content.get("title", "")
        if not item_id or _is_seen(item_id):
            continue

        title   = content.get("title", "")
        summary = content.get("summary", "") or content.get("description", "")
        url     = (content.get("clickThroughUrl") or {}).get("url") \
                  or (content.get("canonicalUrl") or {}).get("url") \
                  or item.get("link", "")

        if not title or _is_seen_title(title):
            continue

        results.append({
            "source": "news",
            "ticker": ticker,
            "title":  title,
            "url":    url,
            "image_url": _pick_thumbnail(content),
            "text":   f"{title}\n\n{summary}".strip(),
            # Not marked seen yet — tasks.poll_news marks both keys after a
            # successful dispatch, so a Celery retry can't lose the item.
            "_seen_keys": [(_SOURCE, item_id), (_TITLE_SOURCE, _title_key(title))],
        })
    return results


def _google_news_rss(ticker: str) -> list[dict]:
    """Fetches recent headlines for a ticker via Google News RSS search."""
    query = quote(f"{ticker} stock")
    url = f"{GOOGLE_NEWS_RSS}?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        log.debug(f"[{ticker}] Google News RSS fetch failed: {e}")
        return []

    results = []
    for item in root.findall(".//item")[:8]:
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link") or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()
        item_id = link or title
        if not title or not item_id or _is_seen(item_id) or _is_seen_title(title):
            continue

        results.append({
            "source": "news",
            "ticker": ticker,
            "title":  title,
            "url":    link,
            # Google News RSS carries no image data at all (verified live
            # 2026-07-26 -- no <media:thumbnail>/<enclosure> despite the
            # feed declaring the media: namespace) -- filled by
            # ingestion.company_photos's fallback at log time instead.
            "image_url": None,
            "text":   f"{title} (published {pub})",
            "_seen_keys": [(_SOURCE, item_id), (_TITLE_SOURCE, _title_key(title))],
        })
    return results


def poll_once() -> list[dict]:
    """
    Fetches new news items for every watchlist ticker from both free sources.
    Deduped persistently across restarts via db.seen_items.
    """
    from knowledge_loader import get_all_tickers

    all_items = []
    for ticker in get_all_tickers():
        all_items.extend(_yfinance_news(ticker))
        time.sleep(0.1)
        all_items.extend(_google_news_rss(ticker))
        time.sleep(0.3)  # be polite to Google's unofficial RSS endpoint
    return all_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent news for the watchlist...\n")
    items = poll_once()
    for i in items[:10]:
        print(f"[{i['ticker']}] {i['title']}")
