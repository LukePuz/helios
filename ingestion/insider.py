"""
Insider trading ingestion via SEC Form 4 filings.
When a CEO or CFO buys their own stock, that's one of the strongest
bullish signals in finance. This watches for those events in real time.

Uses SEC EDGAR directly — no API key required.

NOTE: this previously hit `efts.sec.gov/LATEST/search-index`, a full-text
search endpoint that returns a different response shape than expected and
silently returned zero results. It now reads the same data.sec.gov
submissions JSON as ingestion/edgar.py (one request per ticker, through
edgar.py's fetch, timezone and 429-backoff helpers; it used to be one
browse-edgar atom request per ticker), then opens each filing's primary
Form 4 XML to read the actual transaction code — filtering to open-market purchases (code "P") by
officers/directors, which is the actual bullish signal. Sales, grants, and
option exercises are not surfaced as buy signals.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

EDGAR_BASE = "https://www.sec.gov"
from ingestion.contact import SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}

from ingestion import edgar
from ingestion.edgar import TICKER_TO_CIK

_SOURCE = "insider"

# The old atom query (type=4, count=10, no date bound): the 10 newest
# filings whose form starts with "4" (4, 4/A, 424B*, 425; browse-edgar's
# type= was a prefix filter), any age. Unchanged here: a filing is marked
# seen once looked at, so in practice only new ones are fetched.
_FORMS = ("4",)
_PER_TICKER_CAP = 10

# A filing whose index or XML fetch failed transiently (timeout, connection
# error, broken chunked body, HTTP 5xx) stays unseen so the next pass retries it, until it
# is this old (by acceptance time); then the next failure marks it seen, so a
# permanently broken filing isn't refetched every 30 min forever.
_TRANSIENT_RETRY_MAX_AGE_H = 72
# A pass stops at this many seconds (edgar's budget) and the next one starts
# at the ticker after where it stopped (insider_next_start_<env>).
_PASS_BUDGET_S = edgar._PASS_BUDGET_S
LAST_PASS: dict = {}

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


class _TransientFetchError(Exception):
    """A per-filing SEC fetch failed in a way worth retrying next pass:
    timeout, connection error, broken chunked body or HTTP 5xx. (A 429
    raises edgar._RateLimited instead.)"""


def _fetch_doc(url: str, what: str):
    """GETs a filing document. Raises _TransientFetchError on a timeout,
    connection error, broken chunked body or 5xx, and edgar._RateLimited on
    a 429 (the pass then starts the shared SEC backoff and stops); returns
    None on any other failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except (requests.Timeout, requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError) as e:
        raise _TransientFetchError(f"{what} {url}: {e}") from e
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 429:
            raise edgar._RateLimited(edgar._retry_after_header(e.response)) from e
        if isinstance(status, int) and status >= 500:
            raise _TransientFetchError(f"{what} {url}: HTTP {status}") from e
        log.debug(f"Could not fetch {what} {url}: {e}")
        return None
    except Exception as e:
        log.debug(f"Could not fetch {what} {url}: {e}")
        return None
    return resp


def _find_primary_doc_url(index_url: str) -> str | None:
    """
    Given a filing's -index.htm page, finds the primary Form 4 XML document.

    The index page links to paths like '.../xslF345X06/wk-form4_123.xml' —
    that "xslF345Xnn/" subfolder is EDGAR's XSLT VIEWER route and returns a
    human-readable HTML page (identical file extension, different content),
    not the machine-readable XML. The actual raw XML lives at the same
    filename one directory up, without the xslF345Xnn/ prefix.
    A transient fetch failure raises _TransientFetchError.
    """
    resp = _fetch_doc(index_url, "filing index")
    if resp is None:
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
     total_shares_bought, total_value_bought} or None when it isn't a buy
    or can't be read. A transient fetch failure raises _TransientFetchError
    instead, so the caller leaves the filing unseen and retries it.
    """
    resp = _fetch_doc(xml_url, "Form 4 XML")
    if resp is None:
        return None
    try:
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


def _search_form4(ticker: str, cik: str) -> list[dict] | None:
    """Fetches recent Form 4 filings for a ticker from its submissions JSON.
    None when the request failed, so poll_once can count failures in a row.
    A 429 propagates as edgar._RateLimited."""
    fetched = edgar._fetch_submissions(ticker, cik, label="Insider Form 4")
    if fetched is None:
        return None
    data, tz = fetched

    results = []
    for row in edgar._parse_submissions(data, tz=tz, wanted=_FORMS,
                                        per_form_cap=_PER_TICKER_CAP, max_age_hours=None):
        # The atom <id>: what seen_items holds for filings seen before the
        # switch, so they stay seen.
        filing_id = edgar._seen_key(row["accession"])
        if _is_seen(filing_id):
            continue

        index_url = edgar._index_url(cik, row["accession"])
        accepted = row["accepted_at"]
        # The atom <updated> format: ISO 8601 in Eastern time with offset.
        filed_at = accepted.astimezone(edgar._ET).isoformat() if accepted else ""

        try:
            primary_doc = _find_primary_doc_url(index_url)
            parsed = _parse_form4_transactions(primary_doc) if primary_doc else None
        except _TransientFetchError as e:
            age = datetime.now(timezone.utc) - accepted if accepted else None
            if age is not None and age < timedelta(hours=_TRANSIENT_RETRY_MAX_AGE_H):
                log.info(f"Insider: {ticker} {row['accession']} fetch failed ({e}), retrying next pass")
                continue
            log.warning(f"Insider: {ticker} {row['accession']} fetch failed ({e}) and the filing "
                        f"is over {_TRANSIENT_RETRY_MAX_AGE_H} h old; marked seen, not retried")
            _mark_seen(filing_id)
            continue
        if not parsed:
            # Not a buy signal, no Form 4 XML, or unreadable XML — mark seen
            # now so we don't keep re-fetching/re-parsing it on every poll.
            _mark_seen(filing_id)
            continue

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
            # Not marked seen yet — tasks.poll_insider marks it after a
            # successful dispatch, so a Celery retry can't lose it.
            "_seen_key": filing_id,
        })

    return results


def poll_once() -> list[dict]:
    """One pass over every ticker, starting after where the last pass
    stopped (insider_next_start_<env>). Skipped while edgar's SEC 429
    backoff (edgar_backoff_until_<env>) runs; a 429 here starts that same
    backoff and ends the pass, as do 5 failed tickers in a row and the
    _PASS_BUDGET_S budget. Afterwards LAST_PASS holds the pass's counts:
    "backoff_skip" (skipped, not a failure) and "poll_failed" (no ticker
    succeeded)."""
    LAST_PASS.clear()
    until = edgar._backoff_until()
    if until and datetime.now(timezone.utc) < until:
        LAST_PASS.update({"backoff_skip": True, "poll_failed": False, "until": until.isoformat()})
        log.info(f"Insider: skipped, SEC rate-limit backoff until {until.isoformat()}")
        return []
    all_results = []
    tickers = list(TICKER_TO_CIK.items())
    total = len(tickers)
    try:
        start = int(edgar._get_state("insider_next_start") or 0) % total if total else 0
    except ValueError:
        start = 0
    stats = {"tickers": total, "ok": 0, "failed": 0, "skipped": 0, "aborted": "", "start": start}
    started = time.monotonic()
    consecutive = 0
    polled = 0
    for n in range(total):
        ticker, cik = tickers[(start + n) % total]
        if consecutive >= edgar._ABORT_AFTER_CONSECUTIVE_FAILS:
            stats["aborted"] = f"{consecutive} tickers in a row failed"
        elif time.monotonic() - started > _PASS_BUDGET_S:
            stats["aborted"] = f"pass ran past {_PASS_BUDGET_S}s"
        if stats["aborted"]:
            break
        if n:
            time.sleep(edgar._REQUEST_GAP_S)  # stay under SEC's 10 req/sec limit
        polled += 1
        try:
            results = _search_form4(ticker, cik)
        except edgar._RateLimited as e:
            until = edgar._start_rate_limit_backoff(e.retry_after)
            stats["failed"] += 1
            stats["aborted"] = f"SEC 429, backing off until {until.isoformat()}"
            break
        except Exception as e:      # one ticker's bad data must not end the pass
            log.warning(f"Insider Form 4: {ticker} failed: {e}")
            results = None
        if results is None:
            stats["failed"] += 1
            consecutive += 1
            continue
        consecutive = 0
        stats["ok"] += 1
        all_results.extend(results)
    stats["skipped"] = total - polled
    if stats["aborted"]:
        log.warning(f"Insider: pass stopped early ({stats['aborted']}), "
                    f"{stats['skipped']} ticker(s) not polled")
    if total:
        edgar._set_state("insider_next_start", str((start + polled) % total))
    stats["seconds"] = round(time.monotonic() - started, 1)
    stats["poll_failed"] = bool(total) and stats["ok"] == 0
    LAST_PASS.update(stats)
    log.info(f"Insider pass: {stats['ok']} ok, {stats['failed']} failed, "
             f"{stats['skipped']} skipped of {total} in {stats['seconds']}s (start {start})")
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
