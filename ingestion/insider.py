"""
Insider trading ingestion via SEC Form 4 filings.
When a CEO or CFO buys their own stock, that's one of the strongest
bullish signals in finance. This watches for those events in real time.

Uses SEC EDGAR directly — no API key required.
"""

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_BASE   = "https://www.sec.gov"
HEADERS      = {"User-Agent": "trading-bot lpuzenski@gmail.com"}

from config import SYMBOLS

_seen_form4s: set[str] = set()


def _search_form4(ticker: str) -> list[dict]:
    """Searches SEC full-text search for recent Form 4 filings."""
    params = {
        "q":        f'"{ticker}"',
        "forms":    "4",
        "dateRange": "custom",
        "startdt":  "2024-01-01",
        "_source":  "filing-index",
    }
    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"Form 4 search failed for {ticker}: {e}")
        return []

    results = []
    for hit in data.get("hits", {}).get("hits", [])[:5]:
        src = hit.get("_source", {})
        filing_id = hit.get("_id", "")
        if filing_id in _seen_form4s:
            continue
        _seen_form4s.add(filing_id)

        results.append({
            "source":      "sec_form4",
            "ticker":      ticker,
            "form":        "4",
            "title":       src.get("file_date", "") + f" Form 4 — {ticker}",
            "filing_date": src.get("file_date", ""),
            "url":         EDGAR_BASE + src.get("file_index", ""),
            "text":        f"Insider trading disclosure for {ticker}. "
                           f"Filed: {src.get('file_date', '')}. "
                           f"Entity: {src.get('entity_name', '')}.",
        })
    return results


def poll_once() -> list[dict]:
    all_results = []
    for ticker in SYMBOLS:
        results = _search_form4(ticker)
        all_results.extend(results)
        time.sleep(0.15)
    return all_results


def run_forever(on_new_filing, poll_interval_seconds: int = 1800) -> None:
    """Polls every 30 minutes — insider filings don't need sub-minute latency."""
    log.info("Insider (Form 4) watcher started.")
    while True:
        try:
            filings = poll_once()
            for filing in filings:
                log.info(f"New Form 4: {filing['ticker']} — {filing['title']}")
                try:
                    on_new_filing(filing)
                except Exception as e:
                    log.error(f"on_new_filing callback failed: {e}")
        except Exception as e:
            log.error(f"Form 4 poll error: {e}")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent Form 4 insider filings...\n")
    filings = poll_once()
    for f in filings[:10]:
        print(f"[{f['ticker']}] {f['title']} — {f['url']}")
