"""Record only: a screener rejection that judged an item by its effect on
other names when the item was about the ticker itself.

The screener's rule 1 asks whether an event is "materially relevant to a
specific watchlist stock — including INDIRECT effects" (agents/screener.py
_SCREENER_PROMPT). Some SCREENED_OUT reasons for a ticker's own earnings or
guidance read it as "another" watchlist stock ("earnings beat and guidance
raise, but no new fact for watchlist stocks"). This flags those rows on the
SCREENED_OUT provenance. Nothing reads the flag to decide: the screener's
prompt, its template sha and every pass or reject are unchanged. Whether the
prompt should change is Luke's. Pure: no DB, no I/O, never raises.
"""
from __future__ import annotations

import hashlib
import re

# The reason cites other or watchlist stocks...
OTHER_PATTERN = (r"\b(?:other|related)\b(?:\s+[\w-]+){0,2}?\s+"
                 r"(?:stocks?|names|tickers|holdings|companies|positions)\b"
                 r"|\bwatch\s?list\b")
# ...and says the item doesn't reach them.
NEGATION_PATTERN = r"\b(?:no|not|nor|without)\b|n't\b"
PATTERNS = (OTHER_PATTERN, NEGATION_PATTERN)
PATTERNS_SHA = hashlib.sha256("\n".join(PATTERNS).encode()).hexdigest()[:12]
DEFINITION = ("own_ticker_rejected: title_names_ticker is True, new_fact is True, and the "
              "screener's reason matches OTHER_PATTERN and NEGATION_PATTERN (case-insensitive); "
              "False when any of the three is false; None when the title test or new_fact is "
              "unknown")
_OTHER = re.compile(OTHER_PATTERN, re.I)
_NEG = re.compile(NEGATION_PATTERN, re.I)
_MATCH_CHARS = 60


def reason_match(reason) -> str | None:
    """The OTHER_PATTERN text the reason matched (at most 60 characters,
    lowercased), when the reason also has a negation; else None."""
    if not isinstance(reason, str) or not reason.strip():
        return None
    m = _OTHER.search(reason)
    if m is None or _NEG.search(reason) is None:
        return None
    return " ".join(m.group(0).lower().split())[:_MATCH_CHARS]


def own_ticker_record(title_names_ticker, new_fact, reason) -> dict:
    """The keys for one SCREENED_OUT row's provenance:
    own_ticker_rejected (see DEFINITION), own_ticker_reason_match (the
    matched phrase or None) and own_ticker_patterns_sha. On a surprise,
    own_ticker_rejected is None with own_ticker_error. Never raises."""
    try:
        match = reason_match(reason)
        if title_names_ticker is False or new_fact is False or match is None:
            flag = False
        elif title_names_ticker is True and new_fact is True:
            flag = True
        else:
            flag = None
        return {"own_ticker_rejected": flag, "own_ticker_reason_match": match,
                "own_ticker_patterns_sha": PATTERNS_SHA}
    except Exception as e:
        return {"own_ticker_rejected": None, "own_ticker_reason_match": None,
                "own_ticker_patterns_sha": PATTERNS_SHA, "own_ticker_error": type(e).__name__}
