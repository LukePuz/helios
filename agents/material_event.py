"""Record only: deal, offering and guidance-withdrawal terms in a screened
item's title, and a per-ticker deal record built from them and from the
EDGAR filing's own form.

The own_ticker record (agents/own_ticker.py) flags a reject only when the
screener's reason cites the watchlist, so a takeover headline rejected as
a "recap" (or with no reason), or a Form 425 whose title doesn't name the
ticker, leaves no trace. material_event_record reads the title alone,
independent of the reason, and goes on the SCREENED_OUT provenance. The
deal record (system_state key deal_state_<env>, one entry per ticker) is
written from every screened item (in or out) that carries a deal term and
names the ticker, or is a business-combination filing (DEAL_FORMS).

Nothing reads either to decide: the screener's prompt, the analyst's
prompt and brief, every gate and every template sha are unchanged. Which
items count as a deal, and whether anything should act on it, is Luke's.
The pure functions never raise; deal_state_note never raises either.
"""
from __future__ import annotations

import hashlib
import logging
import re

log = logging.getLogger(__name__)

# (term, group, pattern), case-insensitive, matched against the title only.
# group: "deal", "offering" or "guidance". "acquire" includes "acquisition".
TERMS = (
    ("acquire", "deal", r"\bacquir(?:e|es|ed|ing)\b|\bacquisitions?\b"),
    ("to buy", "deal", r"\bto\s+buy\b"),
    ("merger", "deal", r"\bmerg(?:er|ers|e|es|ed|ing)\b"),
    ("buyout", "deal", r"\bbuy-?outs?\b"),
    ("all-stock", "deal", r"\ball[- ]stock\b"),
    ("definitive agreement", "deal", r"\bdefinitive\s+(?:merger\s+)?agreement\b"),
    ("tender offer", "deal", r"\btender\s+offers?\b"),
    ("offering", "offering", r"\bofferings?\b"),
    ("withdraws guidance", "guidance",
     r"\bwithdr(?:aw|aws|awn|awing|ew)\b(?:\s+\S+){0,3}?\s+guidance\b"),
)
TERMS_SHA = hashlib.sha256("\n".join("\t".join(t) for t in TERMS).encode()).hexdigest()[:12]
_TERM_RES = tuple((name, group, re.compile(p, re.I)) for name, group, p in TERMS)

# EDGAR forms (the filing's own form, the poller's actual_form). Only the
# forms ingestion/edgar.py polls ever arrive (FILING_FORMS prefixes: 8-K,
# 10-Q, 10-K and "4", which takes in 424B* and 425); the rest are listed so
# the record reads them if the poller ever carries them.
DEAL_FORMS = ("425", "S-4", "DEFM14A", "PREM14A", "DEFA14A", "SC TO-T", "SC TO-I", "SC 14D9")
OFFERING_FORM_PREFIXES = ("424B", "S-3", "S-1", "F-3", "F-1")

# Role and consideration, title only. {NAME} is the ticker's own names
# (ingestion.news._company_names: the watchlist company's core name and
# aliases, or $TICKER / (TICKER) / EXCHANGE:TICKER); {BUYER} is another
# watchlist company's names. A plain "to buy" reads as a deal only with a
# named object or a named watchlist buyer: "Time to Buy Oracle Stock",
# "Upgrades Oklo to Buy From Hold" are not deals.
# A gap inside one clause: no sentence or clause mark, but a decimal point
# ("$43.9 Billion") is fine.
_GAP = r"(?:[^.;:|]|(?<=\d)\.(?=\d))"
# A capitalized object after "to buy", not a rating or listicle word.
_CAP_OBJECT = (r"(?!(?-i:From|Now|Rating|Right|Today|Before|After|On|In|At|Here|Stock|Shares|"
               r"The|This|These|It|Or|And|Hold|Sell)\b)(?-i:[A-Z$])")
_SUFFIX = r"(?:\s+(?:Holdings|Inc\.?|Corp\.?|Co\.?|Group))?(?:\s+\(\w+(?::\w+)?\))?"
ROLE_PATTERNS = (
    # acquirer: the name, then an acquiring verb
    ("acquirer", r"{NAME}" + _SUFFIX + r"\s+(?:to\s+acquire|acquires|will\s+acquire|"
                 r"(?:agree(?:s|d)?|moves|in\s+talks|plans|offers|bids)\s+to\s+acquire|"
                 r"explores\s+(?:an?\s+)?acquisition|announces\s+" + _GAP + r"{0,40}?\bacquisition|"
                 r"nears?\s+(?:a\s+)?deal\s+to\s+(?:acquire|buy)|"
                 r"is\s+spending\s+" + _GAP + r"{0,30}?\s+to\s+buy|"
                 r"(?:(?:agree(?:s|d)?|moves|in\s+talks|plans|offers|bids)\s+)?(?:to|will)\s+"
                 r"(?:buy|purchase)\s+" + _CAP_OBJECT + r")"),
    # acquirer: the name's acquisition / buyout / takeover / bid / deal
    ("acquirer", r"{NAME}(?:'s|’s)\s+(?:\S+\s+){0,3}?(?:acquisition|buyout|takeover|bid|deal)\b"),
    # target: an acquiring verb or noun, then the name
    ("target", r"\b(?:acquire|acquires|acquiring|acquisition\s+of|buyout\s+of|takeover\s+of|"
               r"bid\s+for|offer\s+for|deal\s+for)\s+(?:\S+\s+){0,2}?{NAME}(?![\w])"),
    # target: another watchlist company buys the name
    ("target", r"{BUYER}" + _SUFFIX + r"\s+(?:\S+\s+){0,4}?(?:to|will)\s+(?:buy|purchase)\s+"
               r"(?:\S+\s+){0,2}?{NAME}(?![\w])"),
    # target: "<name> acquisition" (not the name's own; "<name> buyout" is
    # left out: it reads either way, "CrowdStrike Buyout" of Zscaler)
    ("target", r"{NAME}(?!['’]s)\s+acquisition\b"),
    # target: the name, to be / agreeing to be acquired
    ("target", r"{NAME}(?:\s+\S+){0,8}?\s+(?:to\s+be|agree(?:s|d|ing)?\s+to\s+be|being|is\s+being)"
               r"\s+acquired\b"),
)
# A holdings line ("$SNOW Shares Acquired by First Trust", "acquired 86
# shares of"): not a deal, whatever its terms.
HOLDINGS_PATTERN = (r"\b(?:shares|stock|stake|position)\s+(?:\w+\s+){0,2}?acquired\s+by\b"
                    r"|\bacquire[sd]?\s+[\d,.]+\s+shares\b")
_HOLDINGS = re.compile(HOLDINGS_PATTERN, re.I)
CONSIDERATION_PATTERNS = (
    ("cash_and_stock", r"\bcash[- ]and[- ]stock\b|\bstock[- ]and[- ]cash\b"),
    ("stock", r"\ball[- ]stock\b|\bstock[- ]swap\b|\bstock\s+deal\b|\bstock[- ]for[- ]stock\b"),
    ("cash", r"\ball[- ]cash\b|\bcash\s+deal\b|\bin\s+cash\b|\bcash\s+offer\b"),
)
DEAL_SHA = hashlib.sha256("\n".join(
    ["\t".join(r) for r in ROLE_PATTERNS] + ["\t".join(c) for c in CONSIDERATION_PATTERNS]
    + [_CAP_OBJECT, HOLDINGS_PATTERN] + list(DEAL_FORMS) + list(OFFERING_FORM_PREFIXES)).encode()).hexdigest()[:12]
_CONSIDERATION_RES = tuple((k, re.compile(p, re.I)) for k, p in CONSIDERATION_PATTERNS)
_DEFINITION = ("material_event_terms: the TERMS names whose pattern matches the title "
               "(case-insensitive, title only, the screener's reason not read); [] when none; "
               "None when there is no title")


def material_event_terms(title) -> list | None:
    """The TERMS names whose pattern matches title, in TERMS order; [] when
    none; None when title is not a non-blank string."""
    if not isinstance(title, str) or not title.strip():
        return None
    return [name for name, _, rx in _TERM_RES if rx.search(title)]


def form_kind(actual_form) -> str | None:
    """"deal" for a DEAL_FORMS filing, "offering" for one starting with an
    OFFERING_FORM_PREFIXES entry, else None."""
    if not isinstance(actual_form, str) or not actual_form.strip():
        return None
    f = actual_form.strip().upper()
    if f in DEAL_FORMS:
        return "deal"
    if f.startswith(OFFERING_FORM_PREFIXES):
        return "offering"
    return None


def material_event_record(title, actual_form=None) -> dict:
    """The keys for one SCREENED_OUT row's provenance:
    material_event_terms (see _DEFINITION), material_event_groups (the
    matched terms' groups, sorted), material_event_form_kind (form_kind of
    the filing's own form; None off EDGAR) and material_event_terms_sha. On
    a surprise, material_event_terms is None with material_event_error.
    Never raises."""
    try:
        terms = material_event_terms(title)
        groups = sorted({g for name, g, _ in _TERM_RES if terms and name in terms})
        return {"material_event_terms": terms, "material_event_groups": groups,
                "material_event_form_kind": form_kind(actual_form),
                "material_event_terms_sha": TERMS_SHA}
    except Exception as e:
        return {"material_event_terms": None, "material_event_groups": [],
                "material_event_form_kind": None, "material_event_terms_sha": TERMS_SHA,
                "material_event_error": type(e).__name__}


def _names(ticker: str, company: str) -> list:
    """(pattern, case_sensitive) for the ticker's own names: its company
    names as mentions_company reads them (exact case for word-like names)
    and $T / (T) / EXCHANGE:T."""
    from ingestion.news import _WORDLIKE_NAMES, _company_names
    out = [(r"(?<![\w])" + re.escape(n), n in _WORDLIKE_NAMES)
           for n in _company_names(ticker, company or "") if n]
    t = re.escape(ticker)
    out.append((rf"(?:\${t}\b|\({t}\)|\b(?:NYSE|NASDAQ|Nasdaq):\s*{t}\b)", True))
    return out


def _alt(names: list) -> str:
    """One alternation of _names entries, each keeping its own case rule."""
    return "(?:" + "|".join(f"(?-i:{p})" if cs else f"(?:{p})" for p, cs in names) + ")"


def _buyer_names(ticker: str, watchlist: dict | None) -> list:
    """_names of every other watchlist ticker (the {BUYER} side)."""
    out = []
    for t, row in (watchlist or {}).items():
        if t != ticker and isinstance(row, dict):
            out += [(p, cs) for p, cs in _names(t, row.get("company", "")) if p]
    return out


def deal_role(ticker: str, company: str, title, watchlist: dict | None = None) -> str | None:
    """The ticker's role in a deal by its title (ROLE_PATTERNS): "acquirer",
    "target", "mixed" when patterns of both match, None when neither does
    or there is no title. watchlist ({ticker: {"company": ...}}) gives the
    {BUYER} names; None loads knowledge_loader.load_watchlist (no buyer
    names if that fails). Never raises."""
    try:
        if not isinstance(title, str) or not title.strip():
            return None
        if watchlist is None:
            try:
                from knowledge_loader import load_watchlist
                watchlist = load_watchlist()
            except Exception:
                watchlist = {}
        name = _alt(_names(ticker, company))
        buyers = _buyer_names(ticker, watchlist)
        buyer = _alt(buyers) if buyers else None
        found = set()
        for role, pat in ROLE_PATTERNS:
            if role in found or ("{BUYER}" in pat and buyer is None):
                continue
            p = pat.replace("{NAME}", name).replace("{BUYER}", buyer or "")
            if re.search(p, title, re.I):
                found.add(role)
        if len(found) == 2:
            return "mixed"
        return next(iter(found), None)
    except Exception:
        return None


def deal_consideration(title) -> str | None:
    """"cash_and_stock", "stock" or "cash" when the title states it
    (CONSIDERATION_PATTERNS, first match in that order), else None."""
    if not isinstance(title, str):
        return None
    return next((k for k, rx in _CONSIDERATION_RES if rx.search(title)), None)


def deal_item(ticker: str, company: str, title, actual_form, terms, title_names_ticker,
              watchlist: dict | None = None) -> dict | None:
    """One screened item's deal fields, or None when it is not a deal item:
    a business-combination filing (form_kind "deal"), or a title that names
    the ticker (title_names_ticker True) with a "deal" group term. "to buy"
    alone counts only when deal_role finds a role ("Time to Buy?", "Stocks
    to Buy" are not deals), and a holdings line (HOLDINGS_PATTERN) never
    does. watchlist: passed to deal_role. The fields: via ("form", "title" or
    "form+title"), role (deal_role; None for a filing title),
    consideration (deal_consideration), terms (the deal-group terms
    matched). Never raises."""
    try:
        deal_terms = [n for n, g, _ in _TERM_RES if g == "deal" and terms and n in terms]
        by_form = form_kind(actual_form) == "deal"
        holdings = isinstance(title, str) and bool(_HOLDINGS.search(title))
        role = (deal_role(ticker, company, title, watchlist)
                if deal_terms and title_names_ticker is True and not holdings else None)
        by_title = (title_names_ticker is True and bool(deal_terms) and not holdings
                    and (deal_terms != ["to buy"] or role is not None))
        if not (by_form or by_title):
            return None
        return {"via": "form+title" if by_form and by_title else ("form" if by_form else "title"),
                "role": role if by_title else None,
                "consideration": deal_consideration(title), "terms": deal_terms}
    except Exception:
        return None


# ── the per-ticker deal record (system_state deal_state_<env>) ──────────────

DEAL_STATE_KEY = "deal_state_{env}"
_DEAL_STATE_MAX = 200        # tickers kept; the least recently seen goes first
_DEAL_STATE_ITEM_IDS = 50    # recent item ids per ticker (repeat screens count once)
_DEAL_STATE_WAIT_S = 1.0     # the screener worker waits at most this for the lock


def _votes_value(votes: dict, mixed: str = "mixed") -> str | None:
    kinds = [k for k, n in (votes or {}).items() if isinstance(n, int) and n > 0]
    return kinds[0] if len(kinds) == 1 else (mixed if kinds else None)


def deal_state_apply(book: dict, ticker: str, item: dict, *, item_id, title, signal: str,
                     actual_form, source, at_iso: str, et_date: str) -> dict | None:
    """book with ticker's deal record updated by one deal item, or None when
    the item was already counted (its id among the recent ones). Pure.
    The record: first_seen_at / announce_date (the first deal item's time
    and ET date: when this record first saw the deal, not a parsed
    announcement date), first_item_id, first_title, last_seen_at,
    last_item_id, items (deal items counted), screened_in / screened_out,
    via (counts by deal_item via), role / role_votes (one role seen, or
    "mixed"), consideration / consideration_votes, terms_seen, forms_seen,
    sources_seen, item_ids (the recent ones), terms_sha and deal_sha."""
    old = book.get(ticker) if isinstance(book.get(ticker), dict) else {}
    ids = [i for i in (old.get("item_ids") or []) if isinstance(i, str)]
    iid = str(item_id)[:300] if item_id else None
    if iid and iid in ids:
        return None
    rec = dict(old)
    rec.setdefault("first_seen_at", at_iso)
    rec.setdefault("announce_date", et_date)
    rec.setdefault("first_item_id", iid)
    rec.setdefault("first_title", str(title or "")[:200] or None)
    rec["last_seen_at"], rec["last_item_id"] = at_iso, iid
    rec["items"] = int(old.get("items") or 0) + 1
    side = "screened_in" if signal == "SCREENED_IN" else "screened_out"
    rec[side] = int(old.get(side) or 0) + 1
    rec.setdefault("screened_in" if side == "screened_out" else "screened_out", 0)
    via = dict(old.get("via") or {})
    via[item["via"]] = int(via.get(item["via"]) or 0) + 1
    rec["via"] = via
    for field, value in (("role", item.get("role")), ("consideration", item.get("consideration"))):
        votes = dict(old.get(f"{field}_votes") or {})
        if value:
            votes[value] = int(votes.get(value) or 0) + 1
        rec[f"{field}_votes"] = votes
        rec[field] = _votes_value(votes)
    for field, values in (("terms_seen", item.get("terms") or []),
                          ("forms_seen", [actual_form] if actual_form else []),
                          ("sources_seen", [source] if source else [])):
        rec[field] = sorted(set(old.get(field) or []) | {str(v) for v in values})
    if iid:
        ids.append(iid)
    rec["item_ids"] = ids[-_DEAL_STATE_ITEM_IDS:]
    rec["terms_sha"], rec["deal_sha"] = TERMS_SHA, DEAL_SHA
    new = dict(book)
    new[ticker] = rec
    if len(new) > _DEAL_STATE_MAX:
        keep = sorted(new, key=lambda t: str((new[t] or {}).get("last_seen_at") or ""),
                      reverse=True)[:_DEAL_STATE_MAX]
        new = {t: new[t] for t in keep}
    return new


def deal_state_note(ticker: str, event: dict, signal: str, provenance: dict | None,
                    now=None) -> bool:
    """Record only: after a SCREENED_IN or SCREENED_OUT row, fold the item
    into ticker's deal record when it is a deal item (deal_item). Edits the
    system_state record under a short Redis lock (waits at most
    _DEAL_STATE_WAIT_S; a busy lock skips the edit; without Redis it edits
    unlocked); a failed read writes nothing, and a stored value that is not
    a JSON object is reset to {} with one warning. Read by nothing that
    decides. Cost on the screener worker: nothing beyond the title regexes
    for an item that is not a deal item; a deal item (165 in the
    11-10..11-23 replay of screened rows: 3 to 20 a session, 84 on the
    11-23 deal day) waits at most 1 s for the lock and makes two
    system_state round trips (read, write). True when written. Never
    raises."""
    lock = None
    try:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        event = event if isinstance(event, dict) else {}
        prov = provenance if isinstance(provenance, dict) else {}
        title, actual_form = event.get("title"), event.get("actual_form")
        terms = prov.get("material_event_terms")
        if terms is None:
            terms = material_event_terms(title)
        title_names = prov.get("title_names_ticker")
        company = ""
        if title_names is None or terms:
            try:
                from knowledge_loader import load_watchlist
                company = (load_watchlist().get(ticker) or {}).get("company", "") or ""
            except Exception:
                company = ""
        if title_names is None and terms:
            try:
                from ingestion.news import mentions_company
                title_names = mentions_company(ticker, company, title or "")
            except Exception:
                title_names = None
        item = deal_item(ticker, company, title, actual_form, terms, title_names)
        if item is None:
            return False
        import json
        import db
        import stop_lock
        from config import TRADING_ENV
        now = now or datetime.now(timezone.utc)
        key = DEAL_STATE_KEY.format(env=TRADING_ENV)
        try:
            lock = stop_lock._redis().lock(f"helios:{key}", timeout=30,
                                           blocking_timeout=_DEAL_STATE_WAIT_S)
            if not lock.acquire():
                log.warning(f"{key}: lock busy; deal record edit skipped")
                return False
        except Exception as e:
            lock = None
            log.debug(f"{key}: lock unavailable ({type(e).__name__}); editing without it")
        raw = db.get_state_strict(key, "")
        try:
            book = json.loads(raw or "{}")
        except (TypeError, ValueError):
            book = None
        if not isinstance(book, dict):
            log.warning(f"{key}: stored value is not a JSON object; deal record book reset to {{}}")
            book = {}
        item_id = event.get("id") or event.get("url") or event.get("title")
        new = deal_state_apply(book, ticker, item, item_id=item_id, title=title, signal=signal,
                               actual_form=actual_form, source=event.get("source"),
                               at_iso=now.isoformat(),
                               et_date=now.astimezone(ZoneInfo("America/New_York")).date().isoformat())
        if new is None:
            return False
        db.set_state(key, json.dumps(new, default=str))
        return True
    except Exception as e:
        log.debug(f"[{ticker}] deal record not written: {type(e).__name__}: {e}")
        return False
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass
