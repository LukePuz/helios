"""
SEC EDGAR ingestion — monitors 8-K, 10-Q, 10-K, and Form 4 filings
for every symbol in our watchlist. Runs continuously, fires a Celery
task whenever a new filing drops.

SEC rate limit: 10 requests/second. We stay well under that.
No API key required.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# Static fallback — used only if the dynamic SEC lookup (below) fails on
# first import (network down, SEC endpoint unreachable). It's what
# TICKER_TO_CIK used to be entirely: hand-maintained, covering only ~29 of
# the ~75 current watchlist tickers, and using "GOOGL" while the watchlist
# itself uses "GOOG" — Alphabet filings silently never matched. The dynamic
# map below fixes both gaps.
_STATIC_TICKER_TO_CIK: dict[str, str] = {
    "META":  "0001326801",
    "AAPL":  "0000320193",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "GOOG":  "0001652044",
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
from ingestion.contact import SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}  # SEC requires this

_SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CIK_CACHE_PATH = Path(__file__).parent / ".cik_cache.json"
_CIK_CACHE_TTL_SECONDS = 7 * 24 * 3600


def _load_cik_map() -> dict[str, str]:
    """Builds {ticker: zero-padded CIK} for every current watchlist ticker
    from SEC's own company_tickers.json — replaces the static, hand-curated
    map that covered under half the watchlist and had GOOG/GOOGL mismatched
    against the watchlist's own naming. Cached to disk for a week; falls
    back to the static map on any failure (offline, SEC endpoint down)."""
    try:
        if _CIK_CACHE_PATH.exists():
            cached = json.loads(_CIK_CACHE_PATH.read_text())
            if time.time() - cached.get("_fetched_at", 0) < _CIK_CACHE_TTL_SECONDS:
                return cached["map"]
    except Exception as e:
        log.debug(f"CIK cache read failed: {e}")

    try:
        from knowledge_loader import get_all_tickers
        wanted = set(get_all_tickers())
    except Exception as e:
        log.warning(f"Could not load watchlist for CIK map ({e}) — using static fallback")
        return dict(_STATIC_TICKER_TO_CIK)

    try:
        resp = requests.get(_SEC_COMPANY_TICKERS_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        raw = resp.json()  # {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
        by_ticker = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in raw.values()}
        result = {t: by_ticker[t] for t in wanted if t in by_ticker}
        missing = wanted - result.keys()
        if missing:
            log.info(f"No SEC CIK found for: {sorted(missing)} (delisted/foreign/ADR — expected for some)")
        try:
            _CIK_CACHE_PATH.write_text(json.dumps({"_fetched_at": time.time(), "map": result}))
        except Exception as e:
            log.debug(f"CIK cache write failed: {e}")
        return result
    except Exception as e:
        log.warning(f"SEC company_tickers.json fetch failed ({e}) — using static fallback")
        return dict(_STATIC_TICKER_TO_CIK)


TICKER_TO_CIK: dict[str, str] = _load_cik_map()

_SOURCE = "edgar"

# Only process filings published within this window — prevents old annual
# reports and quarterly filings from flooding the pipeline on every restart.
_MAX_AGE_HOURS = 48


_SEEN_THIS_PASS: set[str] | None = None   # loaded once per poll_once pass


def _is_seen(filing_id: str) -> bool:
    if _SEEN_THIS_PASS is not None and filing_id in _SEEN_THIS_PASS:
        return True
    if _SEEN_THIS_PASS is not None:
        return False      # the pass-start set is complete for this source
    try:
        import db
        return db.is_seen(_SOURCE, filing_id)
    except Exception:
        return False  # DB unavailable — fail open rather than silently drop filings


def _mark_seen(filing_id: str) -> None:
    if _SEEN_THIS_PASS is not None:
        _SEEN_THIS_PASS.add(filing_id)
    try:
        import db
        db.mark_seen(_SOURCE, filing_id)
    except Exception as e:
        log.debug(f"mark_seen failed for {filing_id}: {e}")


_SKIP_TAGS = frozenset({
    "script", "style",
    # Modern filings are inline-XBRL (iXBRL): <ix:header> wraps a block of
    # machine-readable, non-displayed XBRL taxonomy facts (member/dimension
    # declarations etc.) that can run to 100k+ chars on a large 10-Q —
    # confirmed against a real NVDA 10-Q where this block alone was 116KB,
    # entirely consuming the extraction cap before any narrative text. It's
    # explicitly marked non-visible in the standard (<ix:hidden> lives
    # inside it), so skipping it is correct, not just convenient.
    "ix:header",
})


class _HTMLTextExtractor(HTMLParser):
    """Minimal stdlib HTML-to-text — no new dependency for this. Skips
    script/style/ix:header content and collapses whitespace, which is all
    that's needed to turn a filing's HTML body into clean prose for the LLM."""
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        return re.sub(r"[ \t]+", " ", "\n".join(self._chunks))


def _filing_documents(index_url: str) -> list[tuple[str, str]]:
    """
    Given a filing's -index.htm page, returns [(doc_type, url)] for the .htm
    documents in SEC's "Document Format Files" table, primary document first
    (Seq=1). Fetching the index page itself used to be what got truncated to
    3000 chars downstream, so the LLM mostly saw nav boilerplate.

    The primary href is often prefixed with "/ix?doc=" (the inline-XBRL
    viewer route, an HTML wrapper around the same document) -- the actual
    document lives at that same path with the prefix stripped, same pattern
    as insider.py's xslF345Xnn/ viewer-path handling for Form 4 XML.
    """
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.debug(f"Could not fetch filing index {index_url}: {e}")
        return []

    table_match = re.search(r'tableFile.*?</table>', resp.text, flags=re.IGNORECASE | re.DOTALL)
    if not table_match:
        return []
    docs = []
    for row in re.findall(r"<tr.*?</tr>", table_match.group(), flags=re.IGNORECASE | re.DOTALL):
        href = re.search(r'href="([^"]+\.htm[^"]*)"', row, flags=re.IGNORECASE)
        if not href:
            continue
        cells = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.IGNORECASE | re.DOTALL)]
        doc_type = cells[3] if len(cells) >= 4 else ""
        url = re.sub(r"^/ix\?doc=", "", href.group(1))
        docs.append((doc_type, url if url.startswith("http") else f"{EDGAR_DOC}{url}"))
    return docs


def _html_to_text(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to fetch filing text from {url}: {e}")
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(resp.text)
    except Exception as e:
        log.debug(f"HTML parse failed for {url}, using raw text: {e}")
        return resp.text[:50_000]
    return parser.text()


# 8-K item numbers worth naming in the excerpt header. The screener sees only
# the first 500 chars, so the header has to say what kind of event this is.
_8K_ITEMS = {
    "1.01": "material agreement", "1.02": "agreement terminated",
    "2.01": "acquisition or disposition", "2.02": "results of operations",
    "2.03": "new financial obligation", "2.05": "exit or restructuring costs",
    "2.06": "impairment", "3.01": "delisting notice", "4.02": "non-reliance on financials",
    "5.02": "officer or director change", "7.01": "Reg FD disclosure",
    "8.01": "other events", "9.01": "exhibits",
}


def _skip_cover(text: str, form: str) -> str:
    """Drops the SEC cover page (registrant name, address, check boxes),
    which used to fill the screener's 500 chars and the analyst's excerpt:
    on 10-27 every 8-K and 10-Q was screened out as "no specific new factual
    information" (limitations audit 2026-10-28 05:00, finding 2)."""
    if form.upper().startswith("8-K"):
        m = re.search(r"Item\s+\d\.\d{2}", text)
        return text[m.start():] if m else text
    if form.upper() in ("10-Q", "10-K"):
        # The first hit is the table of contents; the second is the section.
        hits = [m.start() for m in re.finditer(r"Management.s Discussion and Analysis", text, re.IGNORECASE)]
        return text[hits[1]:] if len(hits) >= 2 else text
    return text


def _fetch_filing_text(filing_url: str, form: str = "") -> str:
    """Downloads and cleans a filing from SEC EDGAR. filing_url is the
    -index.htm page from the atom feed. For an 8-K the text starts with the
    item numbers and the EX-99 exhibits (the press release is where the news
    is; the 8-K body often just says "see Exhibit 99.1"), then the body with
    the cover page removed."""
    docs = _filing_documents(filing_url)
    if not docs:
        log.debug(f"No primary document found for {filing_url}")
        return ""
    body = _skip_cover(_html_to_text(docs[0][1]), form)
    parts = []
    if form.upper().startswith("8-K"):
        items = sorted(set(re.findall(r"Item\s+(\d\.\d{2})", body)))
        if items:
            parts.append("8-K items: " + ", ".join(
                f"{i} ({_8K_ITEMS[i]})" if i in _8K_ITEMS else i for i in items))
        for doc_type, url in [d for d in docs[1:] if d[0].upper().startswith("EX-99")][:2]:
            exhibit = _html_to_text(url)
            if exhibit:
                parts.append(f"{doc_type}:\n{exhibit[:20_000]}")
    parts.append(("Filing body:\n" if parts else "") + body)
    return "\n\n".join(parts)[:50_000]  # cap at 50k chars to stay within model context


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

            # NOT marked seen here — a Celery retry after a downstream
            # failure would otherwise see this filing as already-seen and
            # silently drop it forever. tasks.poll_edgar marks it seen only
            # after successfully dispatching it (see _seen_key/_SOURCE).
            text = _fetch_filing_text(url, form) if url else ""

            filings.append({
                "ticker":    ticker,
                "cik":       cik,
                "form":      form,
                # Without a source the screener logged "?", the brief said
                # "Triggering Event: UNKNOWN" and memory filed it as news.
                "source":    "sec_form4" if form == "4" else "sec_filing",
                "type":      f"Form {form}" if form == "4" else form,
                "title":     title,
                "url":       url,
                "timestamp": updated,
                "text":      text,
                "_seen_key": filing_id,
            })
    except ET.ParseError as e:
        log.warning(f"XML parse error for {ticker} {form}: {e}")

    return filings


def poll_once(on_filing=None) -> list[dict]:
    """
    Checks all tickers for new filings. Returns list of new filing dicts.
    Call this on a schedule (every 10 minutes is plenty).
    With on_filing, each filing is handed over as soon as it is found: a
    full pass takes ~11 min, and an 8-K on the first ticker used to wait for
    the whole loop (audit 10:53, A1053-4). A callback failure is logged and
    the filing stays unseen, so the next poll picks it up again.
    """
    global _SEEN_THIS_PASS
    try:
        import db
        _SEEN_THIS_PASS = db.seen_ids(_SOURCE)   # None on failure: per-item checks
    except Exception:
        _SEEN_THIS_PASS = None
    try:
        return _poll_all(on_filing)
    finally:
        _SEEN_THIS_PASS = None


def _poll_all(on_filing=None) -> list[dict]:
    new_filings = []
    for ticker, cik in TICKER_TO_CIK.items():
        for form in FILING_FORMS:
            filings = _get_recent_filings(ticker, cik, form)
            if filings:
                log.info(f"New {form} filing(s) for {ticker}: {len(filings)}")
                new_filings.extend(filings)
                for filing in filings if on_filing else ():
                    key = filing.get("_seen_key")   # the dispatch pops it
                    try:
                        on_filing(filing)
                        if key and _SEEN_THIS_PASS is not None:
                            # Marked seen by the dispatch, not _mark_seen: add it
                            # here so a filing in two feeds (a Form 4 whose owner
                            # is also watched) isn't dispatched twice this pass.
                            _SEEN_THIS_PASS.add(key)
                    except Exception as e:
                        log.error(f"EDGAR: dispatch failed for {ticker} {form}: {e}")
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
