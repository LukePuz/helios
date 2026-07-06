"""
SEC EDGAR ingestion — monitors 8-K, 10-Q, 10-K, and Form 4 filings
for every symbol in our watchlist. Runs continuously, fires a Celery
task whenever a new filing drops.

SEC rate limit: 10 requests/second. We stay well under that.
No API key required.
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# CIK numbers map tickers to SEC identifiers
# Pre-populated for our watchlist — add more as needed
TICKER_TO_CIK: dict[str, str] = {
    "META":  "0001326801",
    "AAPL":  "0000320193",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "ORCL":  "0001341439",
    "MSFT":  "0000789019",
    "AMZN":  "0001018724",
    "TSLA":  "0001318605",
    "AMD":   "0000002488",
    "AVGO":  "0001730168",
    "QCOM":  "0000804328",
    "MU":    "0000723125",
    "ARM":   "0001973229",
    "AMAT":  "0000796343",
    "LRCX":  "0000707549",
    "ASML":  "0937556",
    "KLAC":  "0000319201",
    "PLTR":  "0001321655",
    "CRM":   "0001108524",
    "NOW":   "0001373715",
    "SNOW":  "0001640147",
    "NET":   "0001477333",
    "DDOG":  "0001561550",
    "IONQ":  "0001821806",
    "RGTI":  "0001801368",
    "QBTS":  "0001907982",
    "QUBT":  "0001709005",
    "ARQQ":  "0001832176",
    "LITE":  "0001040570",
    "COHR":  "0000021510",
}

FILING_FORMS = ["8-K", "10-Q", "10-K", "4"]  # Form 4 = insider trades
EDGAR_RSS = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_DOC = "https://www.sec.gov"
HEADERS = {"User-Agent": "trading-bot 245081114+LukePuz@users.noreply.github.com"}  # SEC requires this

_SOURCE = "edgar"

# Only process filings published within this window — prevents old annual
# reports and quarterly filings from flooding the pipeline on every restart.
_MAX_AGE_HOURS = 48


def _is_seen(filing_id: str) -> bool:
    try:
        import db
        return db.is_seen(_SOURCE, filing_id)
    except Exception:
        return False  # DB unavailable — fail open rather than silently drop filings


def _mark_seen(filing_id: str) -> None:
    try:
        import db
        db.mark_seen(_SOURCE, filing_id)
    except Exception as e:
        log.debug(f"mark_seen failed for {filing_id}: {e}")


def _fetch_filing_text(filing_url: str) -> str:
    """Downloads the actual text of a filing from SEC EDGAR."""
    try:
        resp = requests.get(filing_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text[:50_000]  # cap at 50k chars to stay within model context
    except Exception as e:
        log.warning(f"Failed to fetch filing text: {e}")
        return ""


def _get_recent_filings(ticker: str, cik: str, form: str) -> list[dict]:
    """Returns new filings for a ticker/form combo not seen before."""
    params = {
        "action": "getcompany",
        "CIK": cik,
        "type": form,
        "dateb": "",
        "owner": "include",
        "count": "5",
        "search_text": "",
        "output": "atom",
    }
    try:
        resp = requests.get(EDGAR_RSS, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"EDGAR fetch failed for {ticker} {form}: {e}")
        return []

    filings = []
    try:
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=_MAX_AGE_HOURS)

        for entry in root.findall("atom:entry", ns):
            filing_id = entry.findtext("atom:id", namespaces=ns) or ""
            if _is_seen(filing_id):
                continue

            title   = entry.findtext("atom:title",   namespaces=ns) or ""
            updated = entry.findtext("atom:updated",  namespaces=ns) or ""
            link_el = entry.find("atom:link",         namespaces=ns)
            url     = link_el.get("href", "") if link_el is not None else ""

            # Skip filings older than _MAX_AGE_HOURS regardless of seen state
            if updated:
                try:
                    filed_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    if filed_at < cutoff:
                        _mark_seen(filing_id)  # mark so we don't re-check
                        continue
                except ValueError:
                    pass

            _mark_seen(filing_id)
            text = _fetch_filing_text(url) if url else ""

            filings.append({
                "ticker":    ticker,
                "cik":       cik,
                "form":      form,
                "title":     title,
                "url":       url,
                "timestamp": updated,
                "text":      text,
            })
    except ET.ParseError as e:
        log.warning(f"XML parse error for {ticker} {form}: {e}")

    return filings


def poll_once() -> list[dict]:
    """
    Checks all tickers for new filings. Returns list of new filing dicts.
    Call this on a schedule (every 10 minutes is plenty).
    """
    new_filings = []
    for ticker, cik in TICKER_TO_CIK.items():
        for form in FILING_FORMS:
            filings = _get_recent_filings(ticker, cik, form)
            if filings:
                log.info(f"New {form} filing(s) for {ticker}: {len(filings)}")
                new_filings.extend(filings)
            time.sleep(0.12)  # stay under 10 req/sec SEC limit
    return new_filings


def run_forever(on_new_filing, poll_interval_seconds: int = 600) -> None:
    """
    Runs continuously. Calls on_new_filing(filing_dict) for each new filing.
    Designed to run as a Celery beat task or in a background thread.
    """
    log.info("EDGAR watcher started.")
    while True:
        try:
            filings = poll_once()
            for filing in filings:
                try:
                    on_new_filing(filing)
                except Exception as e:
                    log.error(f"on_new_filing callback failed: {e}")
        except Exception as e:
            log.error(f"EDGAR poll error: {e}")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Polling SEC EDGAR for recent filings...\n")
    filings = poll_once()
    if filings:
        for f in filings:
            print(f"[{f['form']}] {f['ticker']} — {f['title']}")
            print(f"  URL: {f['url']}")
            print(f"  Text preview: {f['text'][:200].strip()}\n")
    else:
        print("No new filings found (all already seen, or none recently filed).")
