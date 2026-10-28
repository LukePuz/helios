"""
Knowledge base loader — parses watchlist.md and thesis files for use by agents.
All files live in trading_bot/knowledge/.
"""
import logging
import os
import re
from functools import lru_cache

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")

# Maps section heading keywords → thesis file
_SECTION_TO_FILE: dict[str, str] = {
    "AI Infrastructure":   "ai_infrastructure.md",
    "AI Software":         "ai_software.md",
    "Cyber":               "ai_software.md",
    "Power Infrastructure":"power_infrastructure.md",
    "Nuclear":             "power_infrastructure.md",
    "Robotics":            "robotics.md",
    "Quantum":             "quantum.md",
    "Space":               "space_data_centers.md",
    "Financials":          "financials.md",
    "AI Data":             "ai_data.md",
}


def _read(filename: str) -> str:
    path = os.path.join(_KNOWLEDGE_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[knowledge file not found: {filename}]"


@lru_cache(maxsize=1)
def load_watchlist() -> dict[str, dict]:
    """
    Parse watchlist.md and return a dict keyed by ticker.

    Return shape:
        {
            "NVDA": {
                "company":  "Nvidia",
                "role":     "Compounder",      # Compounder | Accelerator | Moonshot
                "tier":     "Core",            # Core | Watch | Speculative
                "sector":   "AI Infrastructure — Compute / Accelerators",
                "thesis_file": "ai_infrastructure.md",
            },
            ...
        }
    """
    text = _read("watchlist.md")
    watchlist: dict[str, dict] = {}

    current_section = ""
    current_thesis_file = "ai_infrastructure.md"

    for line in text.splitlines():
        # Section headings like "### AI Infrastructure — Compute / Accelerators"
        section_match = re.match(r"^###\s+(.+)", line)
        if section_match:
            current_section = section_match.group(1).strip()
            # Find the best matching thesis file
            for keyword, fname in _SECTION_TO_FILE.items():
                if keyword.lower() in current_section.lower():
                    current_thesis_file = fname
                    break
            continue

        # Table rows: | NVDA | Nvidia | Compounder | Core |
        row_match = re.match(
            r"^\|\s*([A-Z]{1,6})\s*\|\s*(.+?)\s*\|\s*(Compounder|Accelerator|Moonshot)\s*\|\s*(Core|Watch|Speculative)\s*\|",
            line,
        )
        if not row_match and re.match(r"^\|\s*[A-Z]{1,6}\s*\|", line):
            # A ticker row that doesn't parse is silently out of the universe
            # (INTC and NLST since 06-20, role column "Watch"/"Speculative";
            # professor M28). Say so instead.
            logging.getLogger(__name__).warning(
                f"watchlist.md row not parsed (role must be Compounder/Accelerator/Moonshot, "
                f"tier Core/Watch/Speculative): {line.strip()}")
        if row_match:
            ticker, company, role, tier = row_match.groups()
            watchlist[ticker] = {
                "company":     company.strip(),
                "role":        role,
                "tier":        tier,
                "sector":      current_section,
                "thesis_file": current_thesis_file,
            }

    return watchlist


def load_ipo_watch() -> list[dict]:
    """Pending IPO-watch entries from watchlist.md's "IPO Watch" section:
    [{"company", "status", "added"}]. Not part of load_watchlist(), so they
    can't be scanned, analysed or traded until listed (Luke, 2026-10-28)."""
    text = _read("watchlist.md")
    m = re.search(r"^## IPO Watch.*?(?=^## |\Z)", text, flags=re.M | re.S)
    if not m:
        return []
    entries = []
    for line in m.group().splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in ("Company", "") and not set(cells[0]) <= set("-"):
            entries.append({"company": cells[0], "status": cells[1], "added": cells[2]})
    return entries


def get_all_tickers() -> list[str]:
    """Return every ticker in the watchlist (sorted)."""
    return sorted(load_watchlist().keys())


def get_etf_tickers() -> set[str]:
    """Watchlist tickers whose company name says ETF (e.g. DRAM, QTUM). They
    have no analyst ratings, so pollers can skip them."""
    return {t for t, d in load_watchlist().items() if re.search(r"\bETF\b", d["company"])}


def get_core_tickers() -> list[str]:
    """Return only Core-tier tickers."""
    return sorted(t for t, d in load_watchlist().items() if d["tier"] == "Core")


def load_thesis(ticker: str) -> str:
    """
    Return the full text of the thesis file for a given ticker.
    Falls back to ai_infrastructure.md if ticker is not in the watchlist.
    """
    entry = load_watchlist().get(ticker.upper())
    if entry:
        return _read(entry["thesis_file"])
    return _read("ai_infrastructure.md")


@lru_cache(maxsize=1)
def load_philosophy() -> str:
    return _read("portfolio_philosophy.md")


@lru_cache(maxsize=1)
def load_risk_rules() -> str:
    return _read("risk_rules.md")


@lru_cache(maxsize=1)
def load_regime_rules() -> str:
    return _read("market_regime_rules.md")


def get_watchlist_entry(ticker: str) -> dict | None:
    """Return the watchlist dict for a single ticker, or None if not found."""
    return load_watchlist().get(ticker.upper())


def get_tickers_by_role(role: str) -> list[str]:
    """role: 'Compounder' | 'Accelerator' | 'Moonshot'"""
    return sorted(t for t, d in load_watchlist().items() if d["role"] == role)


def get_tickers_by_sector(sector_keyword: str) -> list[str]:
    """Return tickers whose sector contains the given keyword (case-insensitive)."""
    kw = sector_keyword.lower()
    return sorted(t for t, d in load_watchlist().items() if kw in d["sector"].lower())
