"""Record only: what the analyst's cited sources are made of.

The corroboration gate (agents/analyst.py count_distinct_sources and the
OPEN/ADD gate after it) counts every distinct (type, detail[:80]) item the
model lists, so the brief's own FUNDAMENTALS block or PRICE STRUCTURE section
counts as a source, as do types the prompt's schema doesn't name. This module
splits the same deduped items into external, brief-section and other, and
records whether the gate's bar would be met by the external items alone.
Nothing reads the record to decide; whether the gate should count only
external sources is Luke's call. Pure: no DB, no I/O, never raises.
"""
from __future__ import annotations

import re

# The prompt schema's names for the external types (news, EDGAR filings,
# analyst ratings, insider trades, options flow).
EXTERNAL_TYPES = ("news", "edgar_filing", "analyst_rating", "insider_trade", "options_flow")
# The schema's types that come from the brief's own sections (FUNDAMENTALS,
# PRICE STRUCTURE / TECHNICALS).
BRIEF_TYPES = ("fundamentals_data", "price_action")
DEFINITION = ("deduped as the gate: (type, detail[:80]); external: " + ", ".join(EXTERNAL_TYPES)
              + "; brief_section: " + ", ".join(BRIEF_TYPES)
              + "; other: every other type (off-schema ones included)")
_MAX_PUBLISHERS = 10

_URL_HOST = re.compile(r"https?://([^/\s)\"'<>]+)", re.I)
_NOTE = re.compile(r"\s*\((?:published|filed|updated)\b[^)]*\)\s*\.?$", re.I)
_DATE_PAREN = re.compile(r"\s*\(\d[^)]*\)\s*\.?$")
_NAME = r"([A-Z][\w.&'’! ]{1,38}?)"
_PER = re.compile(r"\(?\bper (?:the )?" + _NAME + r"(?: news)?(?: article| report| story)?\)?\.?$")
_PAREN = re.compile(r"\(" + _NAME + r"\)\.?$")
_TITLE_SUFFIX = re.compile(r"\s[-–—]\s([^-–—]{2,40})$")
_PER_TAIL = re.compile(r"\s+per\b.*$", re.I)
_GENERIC = {"news", "event", "event content", "brief", "situation brief", "relevant news",
            "the news", "article", "report", "filing", "filings", "news articles"}


def _name(raw: str | None, require_capital: bool = True) -> str | None:
    """raw as a publisher name, lowercased (a trailing "article"/"report"
    dropped), or None for what reads as a headline fragment or a generic
    reference: no capital start, a digit, a "?", more than 5 words, a
    "section" or brief reference, or an all-caps heading over 5 letters
    (the brief's section names; WSJ, CNBC pass)."""
    if not raw:
        return None
    raw = re.sub(r"\s+(?:articles?|reports?|story|stories)$", "", raw.strip().rstrip("."), flags=re.I)
    if not raw or (require_capital and not raw[0].isupper()):
        return None
    if raw.isupper() and len(raw) > 5:
        return None                     # a brief section heading (POSITION CONTEXT)
    low = " ".join(raw.lower().split())
    if (re.search(r"[\d%$?]|!$", raw) and not low.startswith("yahoo!")) or len(low.split()) > 5:
        return None
    if low in _GENERIC or "section" in low or "brief" in low or low.startswith("per "):
        return None
    return low


def _type_key(t) -> str:
    return str(t).strip().lower()


def _host(text: str) -> str | None:
    m = _URL_HOST.search(text or "")
    if not m:
        return None
    host = m.group(1).lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else (host or None)


def publisher(source: dict) -> tuple[str | None, str | None]:
    """(publisher, basis) of one cited source, or (None, None). Best effort
    on the model's free text, so the basis is kept with it: the host of a
    URL in the item's url/link field or its detail ("url"); then, for news
    only, after dropping a trailing "(published ...)" or date note: a
    closing "per X [news] article" ("per"), a closing "(X)" ("paren"), or a
    headline's trailing " - Publisher" (Google News titles; a "per ..." tail
    dropped) ("title_suffix"). Names are lowercased; a headline fragment or
    a generic reference ("per news article") gives none."""
    for f in ("url", "link", "source_url"):
        v = source.get(f)
        if isinstance(v, str) and _host(v):
            return _host(v), "url"
    detail = source.get("detail") if isinstance(source.get("detail"), str) else ""
    if _host(detail):
        return _host(detail), "url"
    if _type_key(source.get("type")) != "news":
        return None, None
    head = detail.strip()
    for _ in range(3):
        stripped = _DATE_PAREN.sub("", _NOTE.sub("", head))
        if stripped == head:
            break
        head = stripped
    for rx, how in ((_PER, "per"), (_PAREN, "paren")):
        m = rx.search(head)
        if m:
            name = _name(m.group(1))
            if name:
                return name, how
    m = _TITLE_SUFFIX.search(head)
    if m:
        name = _name(_PER_TAIL.sub("", m.group(1)))
        if name:
            return name, "title_suffix"
    return None, None


def source_mix(sources, min_sources: int) -> dict:
    """The record for one verdict's "sources" list. min_sources is the
    gate's bar (analyst._MIN_DISTINCT_SOURCES). Fields:
    source_count (the gate's own count), external_source_count,
    brief_section_sources, other_sources (the three sum to source_count),
    type_counts {type: distinct items}, distinct_publishers (among the
    external items with one), publishers (sorted, at most 10),
    n_external_no_publisher, publisher_basis {basis: n}, min_sources,
    gate_would_pass (source_count >= min_sources), gate_would_pass_external_only
    (external_source_count >= min_sources), definition. Both "would pass"
    flags ignore the action: the gate itself acts only on an OPEN/ADD.
    Never raises: {"error": <class>} instead."""
    try:
        first = {}
        for s in sources if isinstance(sources, list) else []:
            if isinstance(s, dict) and s.get("type"):
                key = (s.get("type"), (s.get("detail") or "")[:80])
                first.setdefault(key, s)         # the gate's dedupe; keep the first item
        types, ext, brief, other = {}, 0, 0, 0
        pubs, no_pub, basis = set(), 0, {}
        for (t, _), s in first.items():
            name = str(t)
            types[name] = types.get(name, 0) + 1
            k = _type_key(t)
            if k in EXTERNAL_TYPES:
                ext += 1
                pub, how = publisher(s)
                if pub:
                    pubs.add(pub)
                    basis[how] = basis.get(how, 0) + 1
                else:
                    no_pub += 1
            elif k in BRIEF_TYPES:
                brief += 1
            else:
                other += 1
        n = len(first)
        return {"source_count": n, "external_source_count": ext,
                "brief_section_sources": brief, "other_sources": other,
                "type_counts": dict(sorted(types.items())),
                "distinct_publishers": len(pubs),
                "publishers": sorted(pubs)[:_MAX_PUBLISHERS],
                "n_external_no_publisher": no_pub, "publisher_basis": dict(sorted(basis.items())),
                "min_sources": min_sources,
                "gate_would_pass": n >= min_sources,
                "gate_would_pass_external_only": ext >= min_sources,
                "definition": DEFINITION}
    except Exception as e:
        return {"error": type(e).__name__}
