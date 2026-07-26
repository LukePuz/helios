"""
S&P 500 universe + GICS sector ingestion — added 2026-07-26 to support
broad-market screening (full-index sector alignment, new-ticker/watchlist
candidate discovery) alongside the existing watchlist-scoped pipeline.

Every other ingestion source in this codebase is scoped to
knowledge_loader.get_all_tickers() (the ~101-ticker AI/tech thesis
watchlist) -- this is the first source that describes the other ~400
S&P 500 names and what GICS sector each belongs to, which the watchlist's
own "sector" field can't (that field is a thesis bucket label like "AI
Infrastructure -- Compute/Accelerators", not a real GICS sector).

No API key required -- scrapes Wikipedia's maintained constituent table,
same "no key needed" posture as edgar.py/insider.py/arxiv_watcher.py.
Cached to disk for a week (mirrors edgar.py's _CIK_CACHE pattern); index
changes/rebalances are quarterly at most, so a week-old constituent list
is never meaningfully wrong.
"""

import json
import logging
import time
from io import StringIO
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (trading-bot; +lpuzenski@gmail.com)"}

_CACHE_PATH = Path(__file__).parent / ".sp500_cache.json"
_CACHE_TTL_SECONDS = 7 * 24 * 3600

# GICS sector name -> SPDR Select Sector ETF. Standard, stable mapping (the
# 11 GICS sectors have had a 1:1 SPDR ETF each since Real Estate split out
# of Financials in 2016) -- not expected to need maintenance.
SECTOR_ETF_MAP: dict[str, str] = {
    "Information Technology":  "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Utilities":               "XLU",
    "Materials":               "XLB",
    "Communication Services":  "XLC",
    "Real Estate":             "XLRE",
}

# Small last-resort fallback (a handful of large-cap names per sector) for
# when both the cache and the live Wikipedia fetch are unavailable -- keeps
# callers from getting a hard failure, same posture as edgar.py's
# _STATIC_TICKER_TO_CIK. Not meant to be exhaustive.
_STATIC_FALLBACK: dict[str, dict] = {
    "AAPL":  {"company": "Apple",           "gics_sector": "Information Technology", "gics_sub_industry": "Technology Hardware"},
    "MSFT":  {"company": "Microsoft",        "gics_sector": "Information Technology", "gics_sub_industry": "Systems Software"},
    "JPM":   {"company": "JPMorgan Chase",   "gics_sector": "Financials",             "gics_sub_industry": "Diversified Banks"},
    "XOM":   {"company": "Exxon Mobil",      "gics_sector": "Energy",                 "gics_sub_industry": "Integrated Oil & Gas"},
    "UNH":   {"company": "UnitedHealth",     "gics_sector": "Health Care",            "gics_sub_industry": "Managed Health Care"},
    "HON":   {"company": "Honeywell",        "gics_sector": "Industrials",            "gics_sub_industry": "Industrial Conglomerates"},
    "AMZN":  {"company": "Amazon",           "gics_sector": "Consumer Discretionary", "gics_sub_industry": "Broadline Retail"},
    "PG":    {"company": "Procter & Gamble", "gics_sector": "Consumer Staples",       "gics_sub_industry": "Household Products"},
    "NEE":   {"company": "NextEra Energy",   "gics_sector": "Utilities",              "gics_sub_industry": "Electric Utilities"},
    "LIN":   {"company": "Linde",            "gics_sector": "Materials",              "gics_sub_industry": "Industrial Gases"},
    "GOOGL": {"company": "Alphabet",         "gics_sector": "Communication Services", "gics_sub_industry": "Interactive Media & Services"},
    "PLD":   {"company": "Prologis",         "gics_sector": "Real Estate",            "gics_sub_industry": "Industrial REITs"},
}


def _fetch_live() -> dict[str, dict]:
    import pandas as pd

    resp = requests.get(_WIKI_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        ticker = str(row["Symbol"]).strip().upper().replace(".", "-")
        if not ticker:
            continue
        result[ticker] = {
            "company":          str(row["Security"]).strip(),
            "gics_sector":      str(row["GICS Sector"]).strip(),
            "gics_sub_industry": str(row["GICS Sub-Industry"]).strip(),
        }
    return result


def _load_constituents() -> dict[str, dict]:
    try:
        if _CACHE_PATH.exists():
            cached = json.loads(_CACHE_PATH.read_text())
            if time.time() - cached.get("_fetched_at", 0) < _CACHE_TTL_SECONDS:
                return cached["map"]
    except Exception as e:
        log.debug(f"S&P 500 cache read failed: {e}")

    try:
        result = _fetch_live()
        try:
            _CACHE_PATH.write_text(json.dumps({"_fetched_at": time.time(), "map": result}))
        except Exception as e:
            log.debug(f"S&P 500 cache write failed: {e}")
        log.info(f"S&P 500 universe refreshed: {len(result)} constituents")
        return result
    except Exception as e:
        log.warning(f"S&P 500 constituent fetch failed ({e}) — using static fallback")
        # Fall back to a stale cache over the static list if one exists at all.
        try:
            if _CACHE_PATH.exists():
                return json.loads(_CACHE_PATH.read_text())["map"]
        except Exception:
            pass
        return dict(_STATIC_FALLBACK)


_CONSTITUENTS: dict[str, dict] = _load_constituents()


def get_sp500_constituents() -> dict[str, dict]:
    """{ticker: {"company", "gics_sector", "gics_sub_industry"}} for all
    current S&P 500 members."""
    return _CONSTITUENTS


def get_sp500_tickers() -> list[str]:
    return sorted(_CONSTITUENTS.keys())


def get_tickers_by_gics_sector(sector: str, limit: int | None = None) -> list[str]:
    """Tickers in a given GICS sector (exact name, e.g. 'Information
    Technology'), sorted alphabetically. `limit` caps the sample size --
    breadth/candidate scans sample rather than fetching bars for an entire
    ~50-90 ticker sector, same rate-limit-conscious posture as
    market_regime._fetch_watchlist_bars_for_breadth's sample_size."""
    tickers = sorted(t for t, d in _CONSTITUENTS.items() if d["gics_sector"] == sector)
    return tickers[:limit] if limit else tickers


_CORPORATE_SUFFIXES = (
    " inc.", " inc", " corp.", " corp", " corporation", " co.", " company",
    " holdings", " holding", " plc", " ltd", " ltd.", " group", " & co.",
)
_MIN_NAME_LENGTH = 4  # skip short/generic names ("3M", "AON") that would false-positive constantly


def _clean_company_name(name: str) -> str:
    lowered = name.lower().strip()
    for suffix in _CORPORATE_SUFFIXES:
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)].strip()
    return lowered


def match_companies_in_text(text: str) -> list[tuple[str, str]]:
    """
    Finds S&P 500 company names mentioned in free text (word-boundary,
    case-insensitive match on the cleaned company name, not the ticker
    symbol itself -- ticker symbols are far too short/ambiguous to search
    for in prose). Returns [(ticker, company), ...] for every match.

    Added 2026-07-26 for arxiv_watcher's company-linkage: a paper naming a
    public company's breakthrough tech is a candidate-discovery signal even
    when that company isn't on the watchlist yet.
    """
    import re

    if not text:
        return []
    lowered_text = text.lower()

    matches = []
    for ticker, info in _CONSTITUENTS.items():
        cleaned = _clean_company_name(info["company"])
        if len(cleaned) < _MIN_NAME_LENGTH:
            continue
        if re.search(rf"\b{re.escape(cleaned)}\b", lowered_text):
            matches.append((ticker, info["company"]))
    return matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    constituents = get_sp500_constituents()
    print(f"{len(constituents)} S&P 500 constituents loaded")
    for sector, etf in SECTOR_ETF_MAP.items():
        tickers = get_tickers_by_gics_sector(sector)
        print(f"  {sector} ({etf}): {len(tickers)} tickers — sample: {tickers[:5]}")
