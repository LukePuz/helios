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
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests
import yfinance as yf

log = logging.getLogger(__name__)

_SOURCE = "news"
_TITLE_SOURCE = "news_title"   # cross-source dedup: same headline via both feeds
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (trading-bot; +245081114+LukePuz@users.noreply.github.com)"}
# Older items are dropped: a days-old "beats on earnings" story was being
# analysed as a fresh trigger (limitations audit 2026-10-28, finding 6).
MAX_ITEM_AGE = timedelta(hours=48)


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
    # Google News titles end in " - Publisher"; strip it so the same story
    # from yfinance and Google gets the same key (audit 2026-10-28).
    title = re.sub(r"\s+[-\u2013\u2014|]\s+[^-\u2013\u2014|]{2,40}$", "", title.strip())
    normalized = re.sub(r"\W+", "", title.lower())
    return hashlib.sha1(normalized.encode()).hexdigest()


def _too_old(published: datetime | None) -> bool:
    """Unknown publish time is kept; a known one older than MAX_ITEM_AGE is not."""
    if published is None:
        return False
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - published > MAX_ITEM_AGE


def _parse_published(value) -> datetime | None:
    if not value:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        value = str(value)
        if re.match(r"\d{4}-\d{2}-\d{2}T", value):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsedate_to_datetime(value)
    except Exception:
        return None


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
        published = _parse_published(content.get("pubDate") or content.get("displayTime")
                                     or item.get("providerPublishTime"))
        if _too_old(published):
            continue
        pub_note = f" (published {published:%Y-%m-%d %H:%M} UTC)" if published else ""

        results.append({
            "source": "news",
            "ticker": ticker,
            "title":  title,
            "url":    url,
            "image_url": _pick_thumbnail(content),
            "text":   f"{title}{pub_note}\n\n{summary}".strip(),
            # Not marked seen yet — tasks.poll_news marks both keys after a
            # successful dispatch, so a Celery retry can't lose the item.
            "_seen_keys": [(_SOURCE, item_id), (_TITLE_SOURCE, _title_key(title))],
        })
    return results


_ENTITY_DROPS: dict[str, int] = {}

# Tickers that are also ordinary words or match unrelated text: a bare
# uppercase match isn't enough for these, the company name or $TICKER is.
_WORDLIKE_TICKERS = {"NOW", "NET", "V", "MA", "ON", "ALL", "IT", "A", "DAVE", "OPEN", "KEY",
                     "PLAY", "FUN", "ARE", "CAN", "AI", "BE", "GO", "SO", "ANY", "TEAM", "DRAM"}
# Short names headlines use instead of the watchlist name (review 4, 10-29:
# "Sandisk", "D-Wave", "Google" were all dropped by the full-name match).
_NAME_ALIASES = {
    "GOOG": ["Google", "GOOGL"], "QBTS": ["D-Wave"], "SMR": ["NuScale"], "RGTI": ["Rigetti"],
    "PWR": ["Quanta"], "ANET": ["Arista"], "AUR": ["Aurora"],
    "KLIC": ["Kulicke"], "GEV": ["Vernova"], "BWXT": ["BWX Technologies"],
    "ASTS": ["AST"], "QTUM": ["Defiance"], "DRAM": ["Roundhill"], "AMD": ["AMD"],
}
# Names to use instead of the watchlist name, which matches generic headlines.
_NAME_OVERRIDES = {"QUBT": ["Quantum Computing Inc", "QCi"]}
# Names that are also ordinary words: matched in exact case only.
_WORDLIKE_NAMES = {"Dave", "Visa", "Coherent", "Aurora", "Constellation", "Meta", "Arm", "Credo",
                   "Oracle", "Quanta", "Defiance", "Amazon", "Core Scientific"}
_NAME_SUFFIXES = re.compile(
    r",?\s+(Inc\.?|Corp\.?|Corporation|Holdings|Technologies|Technology|Group|Ltd\.?|plc|Co\.?|"
    r"Company|Systems|Platforms|Incorporated)$", re.I)


def _company_core(company: str) -> str:
    core = (company or "").strip()
    while True:
        new = _NAME_SUFFIXES.sub("", core).strip()
        if new == core:
            return core
        core = new


def _company_names(ticker: str, company: str) -> list[str]:
    if ticker in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[ticker]
    core = _company_core(company)
    return ([core] if core else []) + _NAME_ALIASES.get(ticker, [])


def mentions_company(ticker: str, company: str, text: str) -> bool:
    """Is this headline about the ticker's company? Google News search for
    "NOW stock" returns "Is Now the Right Time to Buy..." about other names
    (audit A1853-2; Luke D8). Accepts the company name or a short alias as a
    whole word (any case, exact case for names that are ordinary words),
    $TICKER, (TICKER), "TICKER stock/shares", or the bare uppercase ticker
    unless it's word-like."""
    text = text or ""
    for name in _company_names(ticker, company):
        flags = 0 if name in _WORDLIKE_NAMES else re.I
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text, flags):
            return True
    t = re.escape(ticker)
    if re.search(rf"(\${t}\b|\({t}\)|\b(?:NYSE|NASDAQ|Nasdaq):\s*{t}\b)", text):
        return True
    # "DAVE Stock" names the stock; a word-like ticker needs "Stock" right after
    # it, and not in an all-caps headline ("BUY NOW STOCK...", review 5 #6).
    if re.search(rf"\b{t}\s+(?:[Ss]tock|[Ss]hares)\b", text) and (
            ticker not in _WORDLIKE_TICKERS
            or (re.search(rf"\b{t}\s+Stock\b", text) and text != text.upper())):
        return True
    return ticker not in _WORDLIKE_TICKERS and bool(re.search(rf"\b{t}\b", text))


def _google_news_rss(ticker: str, company: str = "") -> list[dict]:
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
        if _too_old(_parse_published(pub)):
            continue
        if company and not mentions_company(ticker, company, title):
            _ENTITY_DROPS[ticker] = _ENTITY_DROPS.get(ticker, 0) + 1
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
    try:
        from knowledge_loader import load_watchlist
        companies = {t: e.get("company", "") for t, e in load_watchlist().items()}
    except Exception as e:
        log.warning(f"News entity filter off this poll (watchlist unavailable): {e}")
        companies = {}

    all_items = []
    batch_keys = set()
    _ENTITY_DROPS.clear()
    for ticker in get_all_tickers():
        fetched = _yfinance_news(ticker)
        time.sleep(0.1)
        fetched += _google_news_rss(ticker, companies.get(ticker, ""))
        time.sleep(0.3)  # be polite to Google's unofficial RSS endpoint
        # Items are only marked seen after dispatch, so the same headline from
        # both feeds used to pass in one poll (TSLA analysed 3x on 10-26).
        for it in fetched:
            key = (ticker, _title_key(it["title"]))
            if key in batch_keys:
                continue
            batch_keys.add(key)
            all_items.append(it)
    if _ENTITY_DROPS:
        log.info(f"News entity filter dropped {sum(_ENTITY_DROPS.values())} Google News item(s) "
                 f"not naming the company: {dict(sorted(_ENTITY_DROPS.items()))}")
    return all_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent news for the watchlist...\n")
    items = poll_once()
    for i in items[:10]:
        print(f"[{i['ticker']}] {i['title']}")
