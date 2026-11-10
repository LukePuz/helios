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
from ingestion.contact import BROWSER_USER_AGENT

HEADERS = {"User-Agent": BROWSER_USER_AGENT}
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
        _note_feed("yfinance", ticker, error=e)
        return []

    results = []
    fresh = 0
    for item in raw_items[:_FEED_CAP]:
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
            _AGED_OUT[f"yfinance:{ticker}"] = _AGED_OUT.get(f"yfinance:{ticker}", 0) + 1
            continue
        fresh += 1
        pub_note = f" (published {published:%Y-%m-%d %H:%M} UTC)" if published else ""

        results.append({
            "source": "news",
            "ticker": ticker,
            "title":  title,
            "url":    url,
            "image_url": _pick_thumbnail(content),
            "text":   f"{title}{pub_note}\n\n{summary}".strip(),
            "published_at": published.isoformat() if published else None,
            # Not marked seen yet — tasks.poll_news marks both keys after a
            # successful dispatch, so a Celery retry can't lose the item.
            "_seen_keys": [(_SOURCE, item_id), (_TITLE_SOURCE, _title_key(title))],
        })
    if len(raw_items) > _FEED_CAP and fresh == _FEED_CAP:
        _CAP_HITS.setdefault("yfinance", []).append(ticker)
    _note_feed("yfinance", ticker, raw=len(raw_items), returned=len(results))
    return results


_ENTITY_DROPS: dict[str, int] = {}
# Log-only, per poll (audit 2026-11-07 03:08 #5): health checks only that the
# poll ran, and a fetch failure is a debug line returning [], so a feed that
# fails for every ticker looked like a quiet one. Per feed: tickers polled,
# raw items read, items returned (unseen, in window, named), and failures
# with the first error seen.
_FEED_STATS: dict[str, dict] = {}


def _note_feed(feed: str, ticker: str, raw: int = 0, returned: int = 0, error=None) -> None:
    st = _FEED_STATS.setdefault(feed, {"tickers": 0, "raw": 0, "returned": 0,
                                       "failed": 0, "first_error": None})
    st["tickers"] += 1
    st["raw"] += raw
    st["returned"] += returned
    if error is not None:
        st["failed"] += 1
        if st["first_error"] is None:
            st["first_error"] = f"{ticker}: {type(error).__name__}: {str(error)[:120]}"


def _log_feed_stats() -> None:
    """One INFO line per poll with each feed's counts, and a WARNING when a
    feed failed for every ticker it polled, or read no items for any of them
    (yfinance answers a block with an empty list, not an error: professor
    63v). Never raises."""
    try:
        if not _FEED_STATS:
            return
        parts = [f"{f} tickers {st['tickers']} raw {st['raw']} returned {st['returned']} "
                 f"failed {st['failed']}" for f, st in sorted(_FEED_STATS.items())]
        log.info(f"News feeds this poll (log-only): {'; '.join(parts)}")
        for f, st in sorted(_FEED_STATS.items()):
            down = False
            if st["tickers"] and st["failed"] == st["tickers"]:
                down = True
                log.warning(f"News feed {f} failed for all {st['tickers']} ticker(s) this poll "
                            f"(first: {st['first_error']})")
            elif st["tickers"] and st["raw"] == 0:
                down = True
                log.warning(f"News feed {f} read no items for any of {st['tickers']} ticker(s) "
                            f"this poll ({st['failed']} failed)")
            if st["tickers"]:
                _track_feed_outage(f, down, st["first_error"])
    except Exception as e:
        log.debug(f"news feed stats failed: {e}")


# A feed down for every ticker on this many polls in a row, in market hours,
# sends one Telegram per feed per ET day (audit 2026-11-09 16:53 #10: Yahoo
# was down 16:34-17:05 and only a WARNING line said so). A healthy poll resets
# the count. Alert only; nothing else changes.
_FEED_DOWN_ALERT_POLLS = 3


def _track_feed_outage(feed: str, down: bool, first_error, now_utc: datetime | None = None) -> bool:
    """Count consecutive all-ticker failures for the feed in Redis and send
    the outage alert when due. True when an alert was sent. Never raises."""
    try:
        import stop_lock
        r = stop_lock._redis()
        key = f"helios:news_feed_down:{feed}"
        if not down:
            r.delete(key)
            return False
        n = int(r.incr(key))
        r.expire(key, 6 * 3600)
        from market_data import is_market_hours
        if n < _FEED_DOWN_ALERT_POLLS or not is_market_hours(now_utc):
            return False
        from zoneinfo import ZoneInfo
        day = (now_utc or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if not r.set(f"helios:news_feed_down_alert:{feed}:{day}", "1", nx=True, ex=86400):
            return False
        import alerts
        alerts._send(f"⚠️ NEWS FEED DOWN\n<b>{feed}</b> returned nothing for every ticker on {n} "
                     f"polls in a row (first error: {str(first_error)[:120]}). Until it recovers the "
                     f"analyst only sees the other feeds.")
        return True
    except Exception as e:
        log.debug(f"feed outage tracking failed for {feed}: {e}")
        return False
# Log-only, per poll (audit 11-06 19:37 #3, A2053-6): each source reads only
# its first _FEED_CAP items, before the seen and age checks. A cap hit is a
# feed longer than that whose first _FEED_CAP were all unseen and in the age
# window, so more may have been cut (the Monday boot poll covers ~58 h). It is
# a lower bound: feeds aren't strictly date-ordered, so items past the cap can
# be cut even when one of the first _FEED_CAP was seen. Aged out: unseen
# items dropped by _too_old, keyed "source:ticker" (re-counted every poll).
_FEED_CAP = 8
_CAP_HITS: dict[str, list[str]] = {}
_AGED_OUT: dict[str, int] = {}   # "source:ticker" -> count

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
        _note_feed("google_news", ticker, error=e)
        return []

    results = []
    fresh = 0
    feed = root.findall(".//item")
    for item in feed[:_FEED_CAP]:
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link") or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()
        item_id = link or title
        if not title or not item_id or _is_seen(item_id) or _is_seen_title(title):
            continue
        published = _parse_published(pub)
        if _too_old(published):
            _AGED_OUT[f"google_news:{ticker}"] = _AGED_OUT.get(f"google_news:{ticker}", 0) + 1
            continue
        fresh += 1
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
            "published_at": published.isoformat() if published else None,
            "_seen_keys": [(_SOURCE, item_id), (_TITLE_SOURCE, _title_key(title))],
        })
    if len(feed) > _FEED_CAP and fresh == _FEED_CAP:
        _CAP_HITS.setdefault("google_news", []).append(ticker)
    _note_feed("google_news", ticker, raw=len(feed), returned=len(results))
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
    _CAP_HITS.clear()
    _AGED_OUT.clear()
    _FEED_STATS.clear()
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
    if _CAP_HITS:
        log.info(f"News feed cap ({_FEED_CAP}) hit with every item unseen and in window, so more may "
                 f"have been cut (log-only, a lower bound): "
                 f"{dict(sorted((k, sorted(v)) for k, v in _CAP_HITS.items()))}")
    if _AGED_OUT:
        log.info(f"News age filter dropped {sum(_AGED_OUT.values())} unseen item(s) older than the "
                 f"window (log-only): {dict(sorted(_AGED_OUT.items()))}")
    _log_feed_stats()
    return all_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent news for the watchlist...\n")
    items = poll_once()
    for i in items[:10]:
        print(f"[{i['ticker']}] {i['title']}")
