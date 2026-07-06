"""
Insider trading ingestion via SEC Form 4 filings.
When a CEO or CFO buys their own stock, that's one of the strongest
bullish signals in finance. This watches for those events in real time.

Uses SEC EDGAR directly — no API key required.

NOTE: this previously hit `efts.sec.gov/LATEST/search-index`, a full-text
search endpoint that returns a different response shape than expected and
silently returned zero results. It now reuses the same company-filing atom
feed that ingestion/edgar.py already uses successfully (browse-edgar with a
CIK + form type), then opens each filing's primary Form 4 XML to read the
actual transaction code — filtering to open-market purchases (code "P") by
officers/directors, which is the actual bullish signal. Sales, grants, and
option exercises are not surfaced as buy signals.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

EDGAR_RSS = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_BASE = "https://www.sec.gov"
HEADERS = {"User-Agent": "trading-bot 245081114+LukePuz@users.noreply.github.com"}

from ingestion.edgar import TICKER_TO_CIK

_SOURCE = "insider"

# Form 4 transaction codes that represent an open-market BUY by the insider.
# ("A" = grant/award, "M" = option exercise, "S" = sale, "G" = gift — none
# of those are a genuine "insider is putting their own money in" signal.)
_BUY_CODES = {"P"}


def _is_seen(filing_id: str) -> bool:
    try:
        import db
        return db.is_seen(_SOURCE, filing_id)
    except Exception:
        return False


def _mark_seen(filing_id: str) -> None:
    try:
        import db
        db.mark_seen(_SOURCE, filing_id)
    except Exception as e:
        log.debug(f"mark_seen failed for {filing_id}: {e}")


def _find_primary_doc_url(index_url: str) -> str | None:
    """
    Given a filing's -index.htm page, finds the primary Form 4 XML document.

    The index page links to paths like '.../xslF345X06/wk-form4_123.xml' —
    that "xslF345Xnn/" subfolder is EDGAR's XSLT VIEWER route and returns a
    human-readable HTML page (identical file extension, different content),
    not the machine-readable XML. The actual raw XML lives at the same
    filename one directory up, without the xslF345Xnn/ prefix.
    """
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.debug(f"Could not fetch filing index {index_url}: {e}")
        return None

    # Look for any .xml link that isn't the index/table-of-contents itself
    candidates = re.findall(r'href="([^"]+\.xml)"', resp.text, flags=re.IGNORECASE)

    def _strip_xsl_viewer_path(path: str) -> str:
        # e.g. ".../000119.../xslF345X06/wk-form4_123.xml" -> ".../000119.../wk-form4_123.xml"
        return re.sub(r"/xslF345X\d+/", "/", path, flags=re.IGNORECASE)

    def _to_url(path: str) -> str:
        path = _strip_xsl_viewer_path(path)
        return path if path.startswith("http") else EDGAR_BASE + path

    for c in candidates:
        if "primary_doc" in c.lower() or "form4" in c.lower():
            return _to_url(c)
    if candidates:
        return _to_url(candidates[0])
    return None


def _parse_form4_transactions(xml_url: str) -> dict | None:
    """
    Fetches and parses a Form 4 primary document. Returns:
    {owner_name, is_officer, is_director, officer_title, buy_codes: [...],
     total_shares_bought, total_value_bought} or None on failure.
    """
    try:
        resp = requests.get(xml_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        log.debug(f"Could not parse Form 4 XML {xml_url}: {e}")
        return None

    def _text(el, path):
        node = el.find(path)
        if node is None:
            return ""
        val_node = node.find("value")
        return (val_node.text if val_node is not None else node.text) or ""

    owner_name = _text(root, ".//reportingOwnerId/rptOwnerName") or "Unknown"
    rel = root.find(".//reportingOwnerRelationship")
    is_officer  = (_text(rel, "isOfficer") if rel is not None else "0") in ("1", "true")
    is_director = (_text(rel, "isDirector") if rel is not None else "0") in ("1", "true")
    officer_title = _text(rel, "officerTitle") if rel is not None else ""

    buy_codes: list[str] = []
    total_shares = 0.0
    total_value = 0.0

    for txn in root.findall(".//nonDerivativeTransaction"):
        code = _text(txn, "transactionCoding/transactionCode")
        if code not in _BUY_CODES:
            continue
        try:
            shares = float(_text(txn, "transactionAmounts/transactionShares") or 0)
            price  = float(_text(txn, "transactionAmounts/transactionPricePerShare") or 0)
        except ValueError:
            shares, price = 0.0, 0.0
        buy_codes.append(code)
        total_shares += shares
        total_value  += shares * price

    if not buy_codes:
        return None

    return {
        "owner_name":          owner_name,
        "is_officer":          is_officer,
        "is_director":         is_director,
        "officer_title":       officer_title,
        "total_shares_bought": total_shares,
        "total_value_bought":  total_value,
    }


def _search_form4(ticker: str, cik: str) -> list[dict]:
    """Fetches recent Form 4 filings for a ticker via the company-filing atom feed."""
    params = {
        "action": "getcompany",
        "CIK": cik,
        "type": "4",
        "dateb": "",
        "owner": "include",
        "count": "10",
        "output": "atom",
    }
    try:
        resp = requests.get(EDGAR_RSS, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Form 4 fetch failed for {ticker}: {e}")
        return []

    results = []
    try:
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            filing_id = entry.findtext("atom:id", namespaces=ns) or ""
            if _is_seen(filing_id):
                continue
            _mark_seen(filing_id)

            link_el = entry.find("atom:link", namespaces=ns)
            index_url = link_el.get("href", "") if link_el is not None else ""
            filed_at = entry.findtext("atom:updated", namespaces=ns) or ""
            if not index_url:
                continue

            primary_doc = _find_primary_doc_url(index_url)
            if not primary_doc:
                continue
            parsed = _parse_form4_transactions(primary_doc)
            if not parsed:
                continue  # no open-market purchase in this filing — not a buy signal

            role = parsed["officer_title"] or ("Director" if parsed["is_director"] else "Insider")
            results.append({
                "source":      "sec_form4",
                "ticker":      ticker,
                "form":        "4",
                "title":       f"Insider BUY — {ticker}: {parsed['owner_name']} ({role}) "
                               f"bought {parsed['total_shares_bought']:,.0f} shares "
                               f"(${parsed['total_value_bought']:,.0f})",
                "filing_date": filed_at,
                "url":         index_url,
                "text": (
                    f"Insider open-market purchase disclosure for {ticker}. "
                    f"{parsed['owner_name']} ({role}) bought "
                    f"{parsed['total_shares_bought']:,.0f} shares worth "
                    f"~${parsed['total_value_bought']:,.0f}. Filed: {filed_at}."
                ),
            })
    except ET.ParseError as e:
        log.warning(f"Form 4 atom parse error for {ticker}: {e}")

    return results


def poll_once() -> list[dict]:
    all_results = []
    for ticker, cik in TICKER_TO_CIK.items():
        results = _search_form4(ticker, cik)
        all_results.extend(results)
        time.sleep(0.15)  # stay under SEC's 10 req/sec limit
    return all_results


def run_forever(on_new_filing, poll_interval_seconds: int = 1800) -> None:
    """Polls every 30 minutes — insider filings don't need sub-minute latency."""
    log.info("Insider (Form 4) watcher started.")
    while True:
        try:
            filings = poll_once()
            for filing in filings:
                log.info(f"New insider buy: {filing['ticker']} — {filing['title']}")
                try:
                    on_new_filing(filing)
                except Exception as e:
                    log.error(f"on_new_filing callback failed: {e}")
        except Exception as e:
            log.error(f"Form 4 poll error: {e}")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent Form 4 insider BUY filings...\n")
    filings = poll_once()
    if filings:
        for f in filings[:10]:
            print(f"[{f['ticker']}] {f['title']}")
    else:
        print("No new insider buys found (all already seen, or none recently filed).")
