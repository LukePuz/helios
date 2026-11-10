"""Boot-time watchlist validator (audit 2026-11-07 15:07 #4, professor).

Finds ticker rows in knowledge/watchlist.md that knowledge_loader drops
(a row that doesn't parse) or overrides (a ticker listed twice: the last
parsed row wins). Read-only and log-only: it changes neither the file nor
which row the loader keeps; that is a knowledge-file edit, Luke's. Kept apart
from knowledge_loader so the analyst's template hash doesn't move.
"""
import logging
import os
import re

log = logging.getLogger(__name__)

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "watchlist.md")
_TICKER_ROW = re.compile(r"^\|\s*([A-Z]{1,6})\s*\|")


def issues(path: str | None = None, loaded: dict | None = None) -> dict:
    """{"duplicates": [{"ticker", "lines", "kept_line"}], "unparsed":
    [{"ticker", "line"}], "rows", "error"}. Line numbers are 1-based.
    kept_line is the row the loader's entry matches (None if none does).
    Never raises."""
    out = {"duplicates": [], "unparsed": [], "rows": 0, "error": None}
    try:
        if loaded is None:
            from knowledge_loader import load_watchlist
            loaded = load_watchlist()
        rows: dict[str, list[tuple[int, list[str]]]] = {}
        with open(path or _PATH, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                m = _TICKER_ROW.match(line)
                if m:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    rows.setdefault(m.group(1), []).append((n, cells))
        out["rows"] = sum(len(v) for v in rows.values())
        for ticker, found in rows.items():
            entry = loaded.get(ticker)
            if entry is None:
                out["unparsed"].extend({"ticker": ticker, "line": n} for n, _ in found)
                continue
            if len(found) > 1:
                kept = [n for n, c in found
                        if len(c) >= 4 and c[1] == entry.get("company") and c[2] == entry.get("role")
                        and c[3] == entry.get("tier")]
                out["duplicates"].append({"ticker": ticker, "lines": [n for n, _ in found],
                                          "kept_line": kept[-1] if kept else None})
    except Exception as e:
        out["error"] = type(e).__name__
    return out


def check_and_warn(path: str | None = None) -> dict:
    """issues(), with one WARNING per duplicate ticker (the loader already
    warns on each unparsed row). For the boot row."""
    res = issues(path)
    for d in res["duplicates"]:
        log.warning(f"watchlist.md lists {d['ticker']} on lines {d['lines']}; "
                    f"the loader keeps line {d['kept_line']} with no other warning")
    return res
