"""
SEC EDGAR ingestion — monitors 8-K, 10-Q, 10-K, and Form 4 filings
for every symbol in our watchlist. Runs continuously, fires a Celery
task whenever a new filing drops.

One data.sec.gov submissions request per ticker per pass. SEC rate limit:
10 requests/second; we sleep between requests and stay well under it.
No API key required.
"""

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

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
# One JSON per company (all its recent filings, issuer-side Form 4s
# included) replaces one browse-edgar atom request per ticker per form:
# 300 requests a pass took up to 17 min and timed out by the hundred.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
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

    # Set by the test suite: importing this module must not call SEC.
    if os.getenv("HELIOS_EDGAR_OFFLINE"):
        return dict(_STATIC_TICKER_TO_CIK)

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


# ── Submissions fetch ────────────────────────────────────────────────────────

_REQUEST_TIMEOUT = (5, 15)        # (connect, read) seconds, per request
_MAX_ATTEMPTS = 3                 # per ticker: 1 try + 2 retries
_BACKOFF_BASE_S = 1.0             # 1 s, then 2 s
_BACKOFF_CAP_S = 8.0
_RETRY_STATUS = frozenset({500, 502, 503, 504})   # 429 stops the pass instead
_REQUEST_GAP_S = 0.15             # between tickers: under ~7 req/s, SEC allows 10
# A pass stops early when SEC is plainly down (this many tickers in a row
# failed) or when it runs past the budget, so it ends inside its 10 min slot
# instead of making the next runs skip with "previous run still going".
_ABORT_AFTER_CONSECUTIVE_FAILS = 5
_PASS_BUDGET_S = 420
# After a 429 the pass stops and later passes skip for the Retry-After, kept
# between these bounds (a missing or unusable Retry-After gives the minimum).
_RATE_LIMIT_BACKOFF_MIN_S = 60
_RATE_LIMIT_BACKOFF_MAX_S = 3600

_ET = ZoneInfo("America/New_York")
# The atom request asked for count=5 per form; the same cap per form here
# keeps the set of filings that reach the screener unchanged.
_PER_FORM_CAP = 5
_TZ_LOGGED = False       # the acceptanceDateTime sample below, once per process

# acceptanceDateTime carries a "Z", but may be Eastern wall-clock time. EDGAR
# accepts filings only 06:00-22:00 ET. Read as ET, real data has pre-market
# rows (raw hour 06-09) and nothing at 22-05; read as UTC it has evening rows
# (raw 22-05 = 17:00-01:00 ET) and nothing at 06-09 (01:00-05:00 ET). Raw
# hours of the wanted forms' rows are pooled across responses until one side
# is clear: ET when morning >= _TZ_MIN_SIDE and late <= _TZ_MAX_OTHER of the
# pooled rows, UTC the other way round. Until then values are read as UTC
# and nothing is cached; once decided, the decision holds for the process.
_TZ_MIN_SIDE = 10
_TZ_MAX_OTHER = 0.01
_TZ_DECISION = None      # _ET or timezone.utc once decided
_TZ_POOL = {"rows": 0, "morning": 0, "late": 0}

# The atom feed's <title> text, "<form>  - <description>" (two spaces), kept
# so log lines, the screener and the brief see the same title as before.
_FORM_DESCRIPTIONS = {
    "8-K":   "Current report",
    "10-Q":  "Quarterly report [Sections 13 or 15(d)]",
    "10-K":  "Annual report [Section 13 and 15(d), not S-K Item 405]",
    "4":     "Statement of changes in beneficial ownership of securities",
    "425":   "Prospectuses and communications, business combinations",
    "424B2": "Prospectus [Rule 424(b)(2)]",
    "424B3": "Prospectus [Rule 424(b)(3)]",
    "424B5": "Prospectus [Rule 424(b)(5)]",
}


class _RateLimited(Exception):
    """SEC answered 429. Carries the Retry-After seconds, or None."""
    def __init__(self, retry_after: float | None):
        super().__init__(f"HTTP 429 (Retry-After {retry_after})")
        self.retry_after = retry_after


def _retry_after_header(resp) -> float | None:
    """A 429 response's Retry-After in seconds, or None when it is missing,
    not a number (an HTTP date) or not finite. Shared with insider.py."""
    try:
        retry_after = float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError, AttributeError):
        return None
    return retry_after if math.isfinite(retry_after) else None


def _get_submissions(ticker: str, cik: str, label: str = "EDGAR") -> dict | None:
    """GETs one company's submissions JSON. Timeouts, connection errors and
    5xx are retried up to _MAX_ATTEMPTS with capped exponential backoff; any
    other HTTP error or a body that isn't JSON fails at once. Returns None
    on failure (logged once, after the last attempt). A 429 raises
    _RateLimited at once, without retrying: the pass stops on it."""
    url = SUBMISSIONS_URL.format(cik=str(cik).lstrip("0").zfill(10))
    err = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=_REQUEST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as e:
            err = str(e)
        else:
            status = int(getattr(resp, "status_code", 200) or 200)
            if status == 429:
                raise _RateLimited(_retry_after_header(resp))
            if status in _RETRY_STATUS:
                err = f"HTTP {status}"
            elif status >= 400:
                log.warning(f"{label} fetch failed for {ticker}: HTTP {status}")
                return None
            else:
                try:
                    return resp.json()
                except ValueError as e:
                    log.warning(f"{label} fetch failed for {ticker}: bad JSON ({e})")
                    return None
        if attempt < _MAX_ATTEMPTS:
            time.sleep(min(_BACKOFF_BASE_S * 2 ** (attempt - 1), _BACKOFF_CAP_S))
    log.warning(f"{label} fetch failed for {ticker} after {_MAX_ATTEMPTS} attempts: {err}")
    return None


def _wanted_form(actual: str, forms=FILING_FORMS) -> str | None:
    """The forms entry (FILING_FORMS by default) a filing's form falls
    under, or None. A prefix match, as browse-edgar's type= filter was:
    type=8-K returned 8-K/A too, and type=4 returned 4/A, 424B5 and 425 (the
    logs show all of these)."""
    for form in forms:
        if actual.startswith(form):
            return form
    return None


def _seen_key(accession: str) -> str:
    """The atom feed's <id> for this filing, which is what seen_items holds
    for every filing stored before the switch, so they stay seen."""
    return f"urn:tag:sec.gov,2008:accession-number={accession}"


def _index_url(cik: str, accession: str) -> str:
    """The filing's -index.htm page, the atom feed's <link href> format."""
    return (f"{EDGAR_DOC}/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{accession}-index.htm")


def _title(actual: str, doc_description: str = "") -> str:
    """The atom <title> for a form: "8-K  - Current report",
    "8-K/A [Amend]  - Current report". A form not in _FORM_DESCRIPTIONS
    takes the JSON's primaryDocDescription."""
    if actual.endswith("/A"):
        base = actual[:-2]
        return f"{base}/A [Amend]  - {_FORM_DESCRIPTIONS.get(base) or doc_description}"
    return f"{actual}  - {_FORM_DESCRIPTIONS.get(actual) or doc_description}"


def _acceptance_tz(recent: dict) -> tuple:
    """(tzinfo, pooled counts) after adding this response's wanted-form rows
    to the pool; see _TZ_MIN_SIDE. Call once per response. Returns UTC while
    undecided. counts = {rows, morning, late}."""
    global _TZ_DECISION
    if _TZ_DECISION is not None:
        return _TZ_DECISION, dict(_TZ_POOL)
    forms = recent.get("form") or []
    for i, raw in enumerate(recent.get("acceptanceDateTime") or []):
        if i >= len(forms) or not _wanted_form(forms[i] or ""):
            continue
        try:
            hour = int(str(raw)[11:13])
        except ValueError:
            continue
        _TZ_POOL["rows"] += 1
        if 6 <= hour <= 9:
            _TZ_POOL["morning"] += 1
        elif hour >= 22 or hour <= 5:
            _TZ_POOL["late"] += 1
    rows, morning, late = _TZ_POOL["rows"], _TZ_POOL["morning"], _TZ_POOL["late"]
    if morning >= _TZ_MIN_SIDE and late <= _TZ_MAX_OTHER * rows:
        _TZ_DECISION = _ET
    elif late >= _TZ_MIN_SIDE and morning <= _TZ_MAX_OTHER * rows:
        _TZ_DECISION = timezone.utc
    return _TZ_DECISION or timezone.utc, dict(_TZ_POOL)


def _accepted_at(acceptance: str, filing_date: str, tz=timezone.utc) -> datetime | None:
    """acceptanceDateTime ("2026-11-03T20:05:11.000Z") as an aware UTC
    datetime, its clock read in tz (UTC, or ET when _acceptance_tz says the
    "Z" is wrong); the filingDate at midnight ET when it's missing."""
    try:
        if acceptance:
            naive = datetime.fromisoformat(acceptance.replace("Z", "+00:00")).replace(tzinfo=None)
            return naive.replace(tzinfo=tz).astimezone(timezone.utc)
        if filing_date:
            return datetime.fromisoformat(filing_date).replace(tzinfo=_ET)
    except ValueError:
        pass
    return None


def _parse_submissions(data: dict, now: datetime | None = None, tz=None, *,
                       wanted=FILING_FORMS, per_form_cap: int = _PER_FORM_CAP,
                       max_age_hours: float | None = _MAX_AGE_HOURS) -> list[dict]:
    """The filings the old per-form atom queries would have returned and
    kept: for each wanted entry (FILING_FORMS by default), its per_form_cap
    newest filings (the atom request's count=5), then only those accepted
    within max_age_hours (None: no age filter). Returned as {form,
    actual_form, accession, accepted_at, primary_document, doc_description},
    ordered like the old per-form loop: by wanted, newest first within a
    form. filings.recent is column-wise (parallel lists). Acceptance times
    are read in tz, by default the process's decision (UTC while undecided).
    insider.py calls this with wanted=("4",), per_form_cap=10 and no age
    filter, its old atom query (type=4, count=10)."""
    recent = ((data or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    cutoff = None
    if max_age_hours is not None:
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=max_age_hours)
    tz = tz or _TZ_DECISION or timezone.utc

    def col(name, i):
        values = recent.get(name) or []
        return (values[i] if i < len(values) else "") or ""

    by_form: dict[str, list[dict]] = {f: [] for f in wanted}
    for i, actual in enumerate(forms):
        form = _wanted_form(actual or "", wanted)
        accession = col("accessionNumber", i)
        if not form or not accession:
            continue
        accepted = _accepted_at(col("acceptanceDateTime", i), col("filingDate", i), tz)
        by_form[form].append({"form": form, "actual_form": actual, "accession": accession,
                              "accepted_at": accepted, "primary_document": col("primaryDocument", i),
                              "doc_description": col("primaryDocDescription", i)})
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    out = []
    for form in wanted:
        rows = sorted(by_form[form], key=lambda r: r["accepted_at"] or oldest, reverse=True)
        # The cap comes before the age filter, as it did in the atom query.
        # Older filings are skipped without a seen row: this list holds up to
        # 1000, and marking them all would write rows for nothing.
        out.extend(r for r in rows[:per_form_cap]
                   if cutoff is None or r["accepted_at"] is None or r["accepted_at"] >= cutoff)
    return out


def _log_tz_sample(ticker: str, data: dict, tz, counts: dict) -> None:
    """Once per process, on the response where the timezone got decided:
    the decision, the pooled counts, and this response's newest filing's raw
    acceptanceDateTime with its parsed ET time, to check against EDGAR."""
    global _TZ_LOGGED
    if _TZ_LOGGED or _TZ_DECISION is None:
        return
    try:
        recent = ((data or {}).get("filings") or {}).get("recent") or {}
        raws = recent.get("acceptanceDateTime") or []
        _TZ_LOGGED = True
        sample = ""
        if raws:
            i = max(range(len(raws)), key=lambda k: raws[k] or "")
            accession = (recent.get("accessionNumber") or [""] * len(raws))[i]
            parsed = _accepted_at(raws[i] or "", "", tz)
            sample = (f"; {ticker} {accession} raw={raws[i]!r} "
                      f"parsed_et={parsed.astimezone(_ET).isoformat() if parsed else None}")
        log.info(f"EDGAR acceptanceDateTime check: read as {'ET' if tz is _ET else 'UTC'} "
                 f"({counts['rows']} rows pooled, {counts['morning']} at raw 06-09, "
                 f"{counts['late']} at raw 22-05){sample}")
    except Exception as e:
        log.debug(f"EDGAR acceptanceDateTime sample failed: {e}")


def _fetch_submissions(ticker: str, cik: str, label: str = "EDGAR") -> tuple | None:
    """(submissions JSON, acceptance tz) for one ticker: the request, then
    this response added to the timezone pool (and the once-per-process
    sample log). None when the request failed; a 429 propagates as
    _RateLimited. Shared with insider.py; label starts its failure log lines."""
    data = _get_submissions(ticker, cik, label)
    if data is None:
        return None
    recent = ((data or {}).get("filings") or {}).get("recent") or {}
    tz, counts = _acceptance_tz(recent)
    _log_tz_sample(ticker, data, tz, counts)
    return data, tz


def _get_recent_filings(ticker: str, cik: str) -> list[dict] | None:
    """New filings for a ticker, every form in FILING_FORMS, from one
    submissions request. None when the request failed, so the pass can tell
    a quiet ticker from a failed one. A 429 propagates as _RateLimited."""
    fetched = _fetch_submissions(ticker, cik)
    if fetched is None:
        return None
    data, tz = fetched
    filings = []
    for row in _parse_submissions(data, tz=tz):
        filing_id = _seen_key(row["accession"])
        if _is_seen(filing_id):
            continue
        form = row["form"]
        url = _index_url(cik, row["accession"])
        accepted = row["accepted_at"]
        # NOT marked seen here — a Celery retry after a downstream
        # failure would otherwise see this filing as already-seen and
        # silently drop it forever. tasks.poll_edgar marks it seen only
        # after successfully dispatching it (see _seen_key/_SOURCE).
        text = _fetch_filing_text(url, form)
        filings.append({
            "ticker":    ticker,
            "cik":       cik,
            "form":      form,
            # Without a source the screener logged "?", the brief said
            # "Triggering Event: UNKNOWN" and memory filed it as news.
            "source":    "sec_form4" if form == "4" else "sec_filing",
            "type":      f"Form {form}" if form == "4" else form,
            "title":     _title(row["actual_form"], row["doc_description"]),
            "url":       url,
            # The atom <updated> format: ISO 8601 in Eastern time with offset.
            "timestamp": accepted.astimezone(_ET).isoformat() if accepted else "",
            "text":      text,
            "_seen_key": filing_id,
        })
    return filings


# ── Pass health ──────────────────────────────────────────────────────────────

# A pass counts as failed when no ticker succeeded, or when it stopped early
# (5 failures in a row, or a 429) with fewer than half the tickers polled
# successfully. A pass that ran out of time with some successes, or one
# skipped during a 429 backoff, is not failed. This many failed passes in a
# row sends one "EDGAR stale" alert (also throttled across spells); two good
# passes in a row end the spell, with one low-priority recovery message.
_STALE_AFTER_FAILED_PASSES = 3
_GOOD_PASSES_TO_RECOVER = 2
LAST_PASS: dict = {}


def _state_key(name: str) -> str:
    from config import TRADING_ENV
    return f"{name}_{TRADING_ENV}"


def _get_state(name: str) -> str | None:
    try:
        import db
        return db.get_state(_state_key(name))
    except Exception as e:
        log.debug(f"EDGAR state read {name} failed: {e}")
        return None


def _set_state(name: str, value: str) -> None:
    try:
        import db
        db.set_state(_state_key(name), value)
    except Exception as e:
        log.debug(f"EDGAR state write {name} failed: {e}")


def _note_pass_health(failed: bool, detail: str = "") -> None:
    """Failed-pass spell tracking in system_state (edgar_poll_fail_count_<env>,
    edgar_poll_ok_streak_<env>); see the comment above. Never raises."""
    try:
        import db
        fail_key, ok_key = _state_key("edgar_poll_fail_count"), _state_key("edgar_poll_ok_streak")
        count = int(db.get_state_strict(fail_key, "0") or 0)
        if failed:
            count += 1
            db.set_state(fail_key, str(count))
            db.set_state(ok_key, "0")
            if count >= _STALE_AFTER_FAILED_PASSES:
                log.warning(f"EDGAR stale: {count} polls failed ({detail})")
            if count == _STALE_AFTER_FAILED_PASSES:
                from tasks import _reconcile_alert_due
                if _reconcile_alert_due("edgar_stale"):
                    import alerts
                    alerts._send(f"⚠️ EDGAR stale: {count} polls failed ({detail}). "
                                 f"No SEC filings are arriving. A recovery message follows when it works.")
        elif count:
            streak = int(db.get_state_strict(ok_key, "0") or 0) + 1
            if streak < _GOOD_PASSES_TO_RECOVER:
                db.set_state(ok_key, str(streak))
                return
            db.set_state(fail_key, "0")
            db.set_state(ok_key, "0")
            if count >= _STALE_AFTER_FAILED_PASSES:
                import alerts
                alerts._send(f"✅ EDGAR recovered after {count} failed polls.")
    except Exception as e:
        log.warning(f"EDGAR pass-health update failed: {e}")


def _backoff_until() -> datetime | None:
    raw = _get_state("edgar_backoff_until")
    try:
        return datetime.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def _start_rate_limit_backoff(retry_after: float | None) -> datetime:
    """After a 429: sets edgar_backoff_until_<env> to now + Retry-After, kept
    within _RATE_LIMIT_BACKOFF_MIN_S.._RATE_LIMIT_BACKOFF_MAX_S, and returns
    it. One key for both SEC pollers (edgar and insider): the limit is per
    client, so a 429 on either pauses both."""
    if retry_after is not None and not math.isfinite(retry_after):
        retry_after = None
    wait = min(max(retry_after or 0, _RATE_LIMIT_BACKOFF_MIN_S), _RATE_LIMIT_BACKOFF_MAX_S)
    until = datetime.now(timezone.utc) + timedelta(seconds=wait)
    _set_state("edgar_backoff_until", until.isoformat())
    return until


def poll_once(on_filing=None) -> list[dict]:
    """
    Checks all tickers for new filings. Returns list of new filing dicts.
    Call this on a schedule (every 10 minutes is plenty).
    With on_filing, each filing is handed over as soon as it is found, so an
    8-K on the first ticker doesn't wait for the whole loop (audit 10:53,
    A1053-4). A callback failure is logged and the filing stays unseen, so
    the next poll picks it up again. Afterwards LAST_PASS holds the pass's
    counts: "poll_failed" (see _note_pass_health's comment) and
    "backoff_skip" (skipped during a 429 backoff; not a failure).
    """
    global _SEEN_THIS_PASS
    until = _backoff_until()
    if until and datetime.now(timezone.utc) < until:
        LAST_PASS.clear()
        LAST_PASS.update({"backoff_skip": True, "poll_failed": False, "until": until.isoformat()})
        log.info(f"EDGAR: skipped, SEC rate-limit backoff until {until.isoformat()}")
        return []
    try:
        import db
        _SEEN_THIS_PASS = db.seen_ids(_SOURCE)   # None on failure: per-item checks
    except Exception:
        _SEEN_THIS_PASS = None
    try:
        filings = _poll_all(on_filing)
    finally:
        _SEEN_THIS_PASS = None
    _note_pass_health(LAST_PASS.get("poll_failed", False),
                      f"{LAST_PASS.get('ok', 0)} of {LAST_PASS.get('tickers', 0)} tickers ok"
                      + (f", {LAST_PASS['aborted']}" if LAST_PASS.get("aborted") else ""))
    return filings


def _poll_all(on_filing=None) -> list[dict]:
    """One pass, starting at the ticker after where the last pass stopped
    (edgar_next_start_<env>), so a pass cut short by the time budget doesn't
    leave the same tail tickers unpolled every time."""
    new_filings = []
    tickers = list(TICKER_TO_CIK.items())
    total = len(tickers)
    try:
        start = int(_get_state("edgar_next_start") or 0) % total if total else 0
    except ValueError:
        start = 0
    stats = {"tickers": total, "ok": 0, "failed": 0, "skipped": 0, "aborted": "", "start": start}
    started = time.monotonic()
    consecutive = 0
    polled = 0
    LAST_PASS.clear()
    for n in range(total):
        ticker, cik = tickers[(start + n) % total]
        if consecutive >= _ABORT_AFTER_CONSECUTIVE_FAILS:
            stats["aborted"] = f"{consecutive} tickers in a row failed"
        elif time.monotonic() - started > _PASS_BUDGET_S:
            stats["aborted"] = f"pass ran past {_PASS_BUDGET_S}s"
        if stats["aborted"]:
            break
        if n:
            time.sleep(_REQUEST_GAP_S)  # stay under 10 req/sec SEC limit
        polled += 1
        try:
            filings = _get_recent_filings(ticker, cik)
        except _RateLimited as e:
            until = _start_rate_limit_backoff(e.retry_after)
            stats["failed"] += 1
            stats["aborted"] = f"SEC 429, backing off until {until.isoformat()}"
            break
        except Exception as e:      # one ticker's bad data must not end the pass
            log.warning(f"EDGAR: {ticker} failed: {e}")
            filings = None
        if filings is None:
            stats["failed"] += 1
            consecutive += 1
            continue
        consecutive = 0
        stats["ok"] += 1
        if filings:
            log.info(f"New filing(s) for {ticker}: {len(filings)} "
                     f"({', '.join(str(f.get('form')) for f in filings)})")
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
                    log.error(f"EDGAR: dispatch failed for {ticker} {filing.get('form')}: {e}")
    stats["skipped"] = total - polled
    if stats["aborted"]:
        log.warning(f"EDGAR: pass stopped early ({stats['aborted']}), "
                    f"{stats['skipped']} ticker(s) not polled")
    if total:
        _set_state("edgar_next_start", str((start + polled) % total))
    stats["seconds"] = round(time.monotonic() - started, 1)
    stopped_early = stats["aborted"] and not stats["aborted"].startswith("pass ran past")
    stats["poll_failed"] = bool(total) and (stats["ok"] == 0
                                            or bool(stopped_early and stats["ok"] < total / 2))
    LAST_PASS.update(stats)
    log.info(f"EDGAR pass: {stats['ok']} ok, {stats['failed']} failed, "
             f"{stats['skipped']} skipped of {total} in {stats['seconds']}s (start {start})")
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
