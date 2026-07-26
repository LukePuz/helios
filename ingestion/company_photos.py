"""
Fallback "related" image lookup for news headlines with no photo directly
attached (added 2026-07-26). Two tiers, both free/no-API-key:
  1. Wikipedia's own page image for the company -- usually a headquarters
     photo or logo, a real, properly-licensed image genuinely related to
     the company, not a hallucinated stock photo.
  2. Google's favicon service for the company's domain, as a last resort
     if Wikipedia has no page or no image for it.

Deliberately does NOT attempt a live "photo of the CEO speaking" image
search: no image-search API is configured here (would need a paid key),
and pulling an arbitrary photo of a specific named person from an
uncontrolled web search carries real rights/attribution risk that every
other "free, no-API-key" source in this codebase doesn't have to deal
with. Wikipedia's own page image is the closest reliable, safe substitute
for "something related."

Cached to disk per ticker (company photos essentially never change) --
same TTL-cache-file pattern as ingestion/edgar.py's CIK map and
ingestion/sp500_universe.py's constituent list.
"""

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (trading-bot; +245081114+LukePuz@users.noreply.github.com)"}
_CACHE_PATH = Path(__file__).parent / ".company_photo_cache.json"
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # a month -- these barely ever change


def _wikipedia_page_image(company: str) -> str | None:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=10,
            params={
                "action": "query", "titles": company, "prop": "pageimages",
                "piprop": "thumbnail", "pithumbsize": 400, "format": "json", "redirects": 1,
            },
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                return thumb
    except Exception as e:
        log.debug(f"Wikipedia page image lookup failed for {company}: {e}")
    return None


def _favicon_fallback(ticker: str) -> str | None:
    try:
        import yfinance as yf
        website = yf.Ticker(ticker).info.get("website", "")
        if not website:
            return None
        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=256"
    except Exception as e:
        log.debug(f"Favicon fallback failed for {ticker}: {e}")
        return None


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except Exception as e:
        log.debug(f"Company photo cache read failed: {e}")
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache))
    except Exception as e:
        log.debug(f"Company photo cache write failed: {e}")


def get_fallback_image(ticker: str, company: str) -> str | None:
    """Returns a "related to this company" image URL, or None if every
    source fails. Cached per ticker -- these don't need to be fresh."""
    cache = _load_cache()
    entry = cache.get(ticker)
    if entry and time.time() - entry.get("_fetched_at", 0) < _CACHE_TTL_SECONDS:
        return entry.get("url")

    url = _wikipedia_page_image(company) or _favicon_fallback(ticker)
    cache[ticker] = {"url": url, "_fetched_at": time.time()}
    _save_cache(cache)
    return url


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for t, c in [("NVDA", "NVIDIA"), ("KO", "The Coca-Cola Company"), ("IONQ", "IonQ")]:
        print(t, "->", get_fallback_image(t, c))
