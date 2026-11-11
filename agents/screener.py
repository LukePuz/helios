"""
7B Screener Agent — fast relevance filter.

Reads every incoming event and decides:
  1. Is this relevant to any of our holdings?
  2. How urgent is it? (immediate / eod / low)
  3. Which specific tickers are affected?

Uses Ollama (qwen2.5:7b) when available.
Falls back to keyword heuristics during development (before new build arrives).

Speed target: < 5 seconds per event.
"""

import json
import logging
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Dedicated CPU-only Ollama for the small models (deploy/systemd/ollama-cpu.service).
# Found 2026-10-25: num_gpu=0 kept the 7B off the GPUs but not out of the main
# server's scheduler. Its unbroken stream of screening calls (86 in 10 min
# after an outage backlog) meant the 72B analyst never got a turn to load:
# zero 72B loads in 6+ hours, every analysis a stub. On an idle server the
# same 72B loads in 11s. A separate server means screening load can't block
# the analyst at all.
OLLAMA_CPU_HOST = os.getenv("OLLAMA_CPU_HOST", "http://localhost:11435")
SCREENER_MODEL = os.getenv("SCREENER_MODEL", "qwen2.5:7b")

from config import SYMBOLS

SCREENER_CONTENT_CAP = 500   # characters of event text the screener prompt includes

_SCREENER_PROMPT = """\
You are a financial analyst screening events for a trading system that holds \
tech, AI, semiconductor, and quantum computing stocks. Every event you mark \
relevant costs several minutes of expensive 72B-model analysis downstream — \
be a real filter, not a rubber stamp. On a busy news day this system sees \
hundreds of events; most should NOT pass.

WATCHLIST: {symbols}

EVENT TO SCREEN:
Source: {source}
Ticker: {ticker}
Watchlist tier: {tier}
We currently hold this stock: {held}
Title: {title}
Content (first 500 chars): {content}

Mark relevant ONLY if BOTH are true:
1. It's materially relevant to a specific watchlist stock — including \
   INDIRECT effects (e.g., a TSMC supply issue affects NVDA even though \
   TSMC isn't in our list) — AND
2. It contains genuinely NEW, specific factual information that could change \
   a position or thesis (an earnings result, a filing, a contract, a guidance \
   change, a specific analyst price-target change with a stated reason).

Treat bad news exactly like good news: an analyst price-target CUT or a \
downgrade is as much new information as a raise or an upgrade, and a guidance \
cut, lost contract, lawsuit or probe is as relevant as a win. For a stock we \
HOLD, a specific negative development is relevant even if the headline is \
phrased as a recap.

Mark NOT relevant for: opinion/recap pieces ("Is X a buy?", "Why is X stock \
falling/rising today?"), multi-stock roundup or "stocks that explain the \
market today" listicles, restatements of information already obviously \
priced in (a headline just describing today's price move with no new cause), \
and generic sector commentary with no ticker-specific new fact. These are \
real content, just not something worth a deep trading analysis every time \
one is published — being mentioned in a roundup is not the same as being \
the subject of new information.

If relevant, set urgency, weighted by watchlist tier:
- Core: this is a large, high-conviction position — bias toward "immediate" \
  for anything materially relevant; the position size means fast reaction \
  matters most here.
- Watch: default to "eod" unless the news is a genuine, position-changing \
  catalyst (earnings surprise, guidance change, major contract/partnership).
- Speculative: these are small, thesis-validation-stage positions — reserve \
  "immediate" for binary/thesis-defining catalysts only (a contract award, \
  a regulatory decision, an FDA/DoD/major-partner announcement); routine \
  news should be "eod" even if it clears the relevance bar.

If relevant and you'd mark urgency "low" (background context, no action \
needed), mark the event NOT relevant instead — there's no reason to pay for \
a deep analysis of something you've just said doesn't need action.

Separately from relevance, set "new_fact": true when the event itself reports \
new factual information (an event, a filing, guidance, data, a deal, or a \
rating change with a stated reason); false when it is opinion, a recap, a \
listicle or roundup, or commentary on a price move.

Respond ONLY with valid JSON, no other text:
{{
  "relevant": true or false,
  "urgency": "immediate" or "eod",
  "affected_tickers": ["TICK1", "TICK2"],
  "reason": "one sentence explanation — if relevant, name the specific new fact",
  "indirect_effects": "brief description of any second-order effects, or null",
  "new_fact": true or false
}}
"""


def parse_new_fact(value) -> bool | None:
    """The model's new_fact: a JSON bool, or the strings "true"/"false".
    Anything else (missing, null, a number, other text) is None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _call_ollama(prompt: str) -> str | None:
    """Calls the local Ollama 7B model. Returns raw text or None if unavailable.

    Tries the dedicated CPU server first. Falls back to the main server only
    when the CPU one isn't running at all (connection refused), which is the
    pre-2026-10-25 behavior. A timeout or error does NOT fall back, so a slow
    CPU server can never double the latency or push load onto the 72B's
    server."""
    for host in (OLLAMA_CPU_HOST, OLLAMA_HOST):
        try:
            return _post_generate(host, prompt)
        except requests.exceptions.ConnectionError:
            continue  # this server isn't running; try the next
        except Exception as e:
            log.warning(f"Ollama 7B call failed ({host}): {e}")
            return None
    return None  # no Ollama running yet


def _post_generate(host: str, prompt: str) -> str:
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model":  SCREENER_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256, "num_gpu": 0},
            # Screening now runs on its own queue/worker at higher
            # concurrency than the 72B analyst (see tasks.screen_event) —
            # both GPUs already sit at ~93% VRAM just holding the 72B
            # model, so a concurrent GPU-bound 7B call would either fail
            # or force the 72B further onto CPU. Forcing CPU here (same
            # fix as chat_assistant.py) means screening never contends
            # with the analyst for VRAM at all.
            "keep_alive": "30m",
        },
        # First CPU call after a cold start can take longer than the old
        # 30s bound (measured ~30s+ cold vs ~28s once warm) — generous
        # enough to avoid an unnecessary keyword-fallback on every
        # worker restart, still bounded since screening must stay fast.
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _get_tier(ticker: str) -> str:
    """Watchlist tier for a ticker (Core | Watch | Speculative), used to
    weight urgency — see _SCREENER_PROMPT's tier guidance. Defaults to
    "Watch" for tickers not on the watchlist (indirect-effect events) so
    they get the middle-ground urgency treatment rather than either
    extreme."""
    try:
        from knowledge_loader import load_watchlist
        return load_watchlist().get(ticker.upper(), {}).get("tier", "Watch")
    except Exception:
        return "Watch"


# Keywords strong enough to count as a binary/thesis-defining catalyst even
# for a Speculative-tier position — everything else on a Speculative ticker
# gets downgraded to "eod" in the keyword fallback (mirrors the tier
# guidance given to the 7B model in _SCREENER_PROMPT).
_MAJOR_CATALYST_KEYWORDS = [
    "contract", "fda", "doj", "antitrust", "acquisition", "merger",
    "quantum advantage", "breakthrough", "sec investigation", "bankruptcy",
]


def _keyword_fallback(event: dict, tier: str = "Watch") -> dict:
    """
    Heuristic fallback when Ollama is not available.
    Less accurate than the model but keeps the pipeline running.
    """
    text = f"{event.get('title','')} {event.get('text','')}".lower()
    ticker = event.get("ticker", "").upper()

    # High-urgency keywords
    urgent_keywords = [
        "contract", "partnership", "acquisition", "merger", "earnings",
        "guidance", "recall", "fda", "sec investigation", "ceo resign",
        "bankruptcy", "layoffs", "breakthrough", "quantum advantage",
        "data breach", "cyberattack", "doj", "antitrust",
    ]

    # Recap/opinion pieces that shouldn't burn a deep-analysis pass even if
    # they mention a watchlist ticker by name.
    low_signal_keywords = [
        "is it a buy", "should you buy", "why is", "why did", "stocks that explain",
        "stocks to watch", "top gainers", "top losers", "here's why", "explained",
    ]
    if any(kw in text for kw in low_signal_keywords):
        return {
            "relevant": False, "urgency": "eod", "affected_tickers": [],
            "reason": "Keyword heuristic: looks like a recap/opinion piece, not new information",
            "indirect_effects": None, "model_used": "keyword_fallback",
        }

    is_urgent = any(kw in text for kw in urgent_keywords)

    # Find affected tickers by mention
    affected = [ticker] if ticker in SYMBOLS else []
    for sym in SYMBOLS:
        if sym.lower() in text and sym not in affected:
            affected.append(sym)

    # Without the model to judge "genuinely new information," the fallback
    # only fires on an explicit urgent keyword hit — matches the tightened
    # philosophy of being a real filter, not a rubber stamp for anything
    # that merely mentions a watchlist ticker.
    relevant = is_urgent and len(affected) > 0

    # Tier-weighted urgency: Speculative positions are small and thesis-
    # validation-stage, so only a binary/thesis-defining catalyst warrants
    # "immediate" — routine urgent-keyword hits (e.g. "layoffs",
    # "partnership") still pass as relevant but downgrade to "eod".
    if relevant and tier == "Speculative" and not any(kw in text for kw in _MAJOR_CATALYST_KEYWORDS):
        urgency = "eod"
    else:
        urgency = "immediate" if relevant else "eod"

    return {
        "relevant":         relevant,
        "urgency":          urgency,
        "affected_tickers": affected if relevant else [],
        "reason":           "Keyword heuristic (Ollama not available)",
        "indirect_effects": None,
        "model_used":       "keyword_fallback",
    }



_DIGEST_CACHE: dict = {}


def _screener_digest() -> str | None:
    """Digest of SCREENER_MODEL on the Ollama that serves it (CPU host first,
    same order as _call_ollama), cached per process. SCREENED_OUT rows had
    no digest (SEF run 8)."""
    if "d" not in _DIGEST_CACHE:
        digest = None
        for host in (OLLAMA_CPU_HOST, OLLAMA_HOST):
            try:
                tags = requests.get(f"{host}/api/tags", timeout=5).json().get("models", [])
                digest = next((m.get("digest") for m in tags if m.get("name") == SCREENER_MODEL), None)
            except Exception:
                continue
            if digest:
                break
        _DIGEST_CACHE["d"] = digest[:12] if digest else None
    return _DIGEST_CACHE["d"]

# Crude keyword tone of a headline, log-only (audit A1853-1): the screener
# passed price-target raises 93/171 and cuts 3/105, so its pass rate is
# tracked by tone. It never feeds the screener prompt or any gate.
_TONE_POS = re.compile(
    r"\b(beats?|soars?|surges?|jumps?|rall(?:y|ies)|raises?|upgrades?|tops|climbs?|"
    r"gains?|record high|outperforms?|boosts?|wins?)\b", re.I)
_TONE_NEG = re.compile(
    r"\b(falls?|plunges?|drops?|slumps?|tumbles?|sinks?|slides?|cuts?|lowers?|"
    r"misses?|downgrades?|warns?|lawsuit|probe|halts?|crash(?:es)?|why is .* falling)\b", re.I)


def title_tone(title: str) -> str:
    """'pos', 'neg', 'mixed' (both word lists hit) or 'neutral'."""
    pos, neg = bool(_TONE_POS.search(title or "")), bool(_TONE_NEG.search(title or ""))
    return "mixed" if pos and neg else "pos" if pos else "neg" if neg else "neutral"


# Opinion/listicle headline patterns, log-only (opinion_shadow in provenance):
# measures how often the 7B passes pieces its prompt calls not relevant.
# Separate from _keyword_fallback's low_signal_keywords, which stay as they are.
# First match in this order wins.
_OPINION_PATTERNS = [
    ("is_x_a_buy", re.compile(
        r"(?:^|[:?!.\-–—|]\s*)is\s+(?:[\w.&'’-]+\s+){0,5}?an?\s+"
        r"(?:(?:good|great|smart|strong|top)\s+)?(?:buy|sell|investment)\b", re.I)),
    ("should_you_buy", re.compile(
        r"\bshould\s+(?:you|investors|i)\s+(?:buy|sell|own|hold|invest)\b", re.I)),
    ("stocks_to_buy", re.compile(
        r"\bstocks?\s+to\s+(?:buy|watch|own|sell|avoid|consider)\b", re.I)),
    ("best_stocks", re.compile(r"\bbest\s+(?:[\w-]+\s+){0,2}?stocks?\b", re.I)),
    ("buy_the_dip", re.compile(r"\bbuy(?:ing)?\s+the\s+dip\b", re.I)),
    ("too_late_to_buy", re.compile(r"\btoo\s+late\s+to\s+buy\b", re.I)),
    ("time_to_buy", re.compile(r"\bis\s+(?:it|now)\s+(?:the\s+)?(?:right\s+)?time\s+to\s+buy\b", re.I)),
    ("prediction", re.compile(r"\bprediction\s*:", re.I)),
    ("could_soar", re.compile(
        r"\bcould\s+(?:\w+\s+){0,2}?(?:soar|double|triple|skyrocket|10x)\b", re.I)),
    ("heres_why", re.compile(r"\bhere(?:['’]s|\s+is)\s+why\b", re.I)),
    ("what_investors_should_know", re.compile(
        r"\bwhat\s+investors\s+(?:should|need\s+to)\s+know\b", re.I)),
    ("im_buying", re.compile(r"\bi(?:['’]m|\s+am)\s+buying\b|\bi\s+just\s+bought\b", re.I)),
    ("millionaire_maker", re.compile(r"\bmillionaire[-\s]+makers?\b", re.I)),
    ("forever_stock", re.compile(r"\bforever\s+stocks?\b", re.I)),
    # "X Is Betting Big on AI: Is Its Growth ..." (2026-11-03).
    ("colon_question", re.compile(
        r":\s+(?:is|are|can|could|should|will|does)\s+(?:it|its|this|now)\b", re.I)),
]


def _opinion_patterns_sha(patterns) -> str:
    """First 12 hex of sha256 over the ordered (name, pattern, flags) list, so
    opinion_shadow rows from different pattern lists can be told apart."""
    import hashlib
    spec = json.dumps([[name, pat.pattern, int(pat.flags)] for name, pat in patterns])
    return hashlib.sha256(spec.encode()).hexdigest()[:12]


try:
    _OPINION_PATTERNS_SHA = _opinion_patterns_sha(_OPINION_PATTERNS)
except Exception:
    _OPINION_PATTERNS_SHA = None

# The same hash over title_tone's two word lists, so tone tags from
# different lists can be told apart (professor m583).
try:
    TONE_SHA = _opinion_patterns_sha([("pos", _TONE_POS), ("neg", _TONE_NEG)])
except Exception:
    TONE_SHA = None

# Log-only (audit 2026-11-10 10:53 #5): headline forms of a hard company
# catalyst. The 7B dropped RKLB/ASTS/QUBT Q3 results and TER guidance on
# 11-10. catalyst_shadow records whether a deterministic rule would pass the
# item; nothing reads it back. All matches are recorded, in this order.
_CATALYST_PATTERNS = [
    ("earnings_results", re.compile(
        r"\b(?:q[1-4]|(?:first|second|third|fourth)[-\s]quarter|quarterly|fiscal)\b[^.]{0,40}?"
        r"\b(?:results?|earnings|loss(?:es)?|profits?|revenues?|sales|eps)\b"
        r"|\b(?:earnings|eps|revenues?|sales)\s+(?:beats?|miss(?:es)?|tops?|tops|falls?\s+short)\b"
        r"|\b(?:beats?|miss(?:es)?|tops)\s+(?:[\w-]+\s+){0,3}?(?:estimates|expectations|consensus)\b", re.I)),
    ("guidance", re.compile(
        r"\b(?:raises?|raised|cuts?|lowers?|lowered|boosts?|lifts?|reaffirms?|projects?|issues?|"
        r"withdraws?|hikes?|trims?|ups)\s+(?:[\w$%.-]+\s+){0,4}?(?:guidance|outlook|forecasts?)\b"
        r"|\b(?:guidance|outlook|forecast)\s+(?:raise|cut|hike|boost)\b", re.I)),
    ("contract", re.compile(
        r"\b(?:wins?|won|awarded|secures?|secured|lands?|landed|signs?|signed|inks?|inked)\b"
        r"[^.]{0,50}?\b(?:contracts?|awards?|deals?|orders?|agreements?)\b", re.I)),
    ("rating_change", re.compile(
        r"\b(?:upgrades?|upgraded|downgrades?|downgraded|initiates?|initiated)\b"
        r"|\bprice\s+targets?\b|\bPT\s+(?:raised|cut|lowered|to)\b", re.I)),
]

# An earnings preview names the results it is waiting for; not a catalyst yet.
_CATALYST_PREVIEW = re.compile(
    r"\b(?:preview|what\s+to\s+expect|ahead\s+of|on\s+deck|upcoming|before\s+(?:its\s+|the\s+)?"
    r"(?:q[1-4]\s+)?(?:results|earnings|report)|(?:set|due|scheduled|expected)\s+to\s+report|"
    r"to\s+report\s+(?:q[1-4]|earnings|results))\b", re.I)

try:
    _CATALYST_PATTERNS_SHA = _opinion_patterns_sha(_CATALYST_PATTERNS + [("preview", _CATALYST_PREVIEW)])
except Exception:
    _CATALYST_PATTERNS_SHA = None


def _catalyst_shadow(ticker: str, title: str, relevant, opinion_match) -> dict:
    """Log-only record of what a deterministic catalyst rule would do: pass
    when the title names the ticker or its company and matches a catalyst
    form, unless it is an opinion headline or an earnings preview. would_rescue: the rule would
    pass what the screener did not. Never raises."""
    rec = {"matches": [], "preview": False, "names_ticker": None, "would_pass": False,
           "would_rescue": False, "patterns_sha": _CATALYST_PATTERNS_SHA, "error": None}
    try:
        text = title or ""
        rec["matches"] = [name for name, pat in _CATALYST_PATTERNS if pat.search(text)]
        rec["preview"] = bool(_CATALYST_PREVIEW.search(text))
        rec["names_ticker"] = _title_names_ticker(ticker, text)
        rec["would_pass"] = (bool(rec["matches"]) and rec["names_ticker"] is True
                             and not opinion_match and not rec["preview"])
        rec["would_rescue"] = rec["would_pass"] and relevant is not True
    except Exception as e:
        rec["error"] = type(e).__name__
    return rec


def _opinion_headlines(title) -> list[str]:
    """Names of every _OPINION_PATTERNS entry the title matches, in list order."""
    t = title or ""
    return [name for name, pat in _OPINION_PATTERNS if pat.search(t)]


def _opinion_headline(title) -> str | None:
    """Name of the first _OPINION_PATTERNS entry the title matches, or None."""
    names = _opinion_headlines(title)
    return names[0] if names else None


def _readthrough_record(result: dict, ticker: str) -> dict:
    """Log-only (audit 2026-11-07 15:07 #5): the reply's affected_tickers
    and indirect_effects, which were only in the log line, so read-through
    to other names can be counted from the signal rows. Changes nothing in
    result; never raises."""
    try:
        raw = result.get("affected_tickers")
        names = None
        if isinstance(raw, list):
            names = []
            for t in raw:
                t = str(t or "").strip().upper()[:12]
                if t and t not in names:
                    names.append(t)
            names = names[:20]
        others = [t for t in names if t != str(ticker).upper()] if names is not None else None
        ie = result.get("indirect_effects")
        ie = None if ie is None or str(ie).strip().lower() in ("", "null", "none") else str(ie)[:300]
        rec = {"affected_tickers": names,
               "affected_others": others,
               "affected_others_on_watchlist": [t for t in others if t in SYMBOLS] if others is not None else None,
               "indirect_effects": ie}
        if raw is not None and names is None:
            # Not a list (e.g. "NVDA, AMD"): the reply's type, so how often
            # can be counted (SEF 154d). Not parsed.
            rec["affected_tickers_type"] = type(raw).__name__
        return rec
    except Exception as e:
        log.debug(f"[{ticker}] readthrough record failed: {e}")
        return {"affected_tickers": None, "affected_others": None,
                "affected_others_on_watchlist": None, "indirect_effects": None,
                "readthrough_error": type(e).__name__}


_HELD_OVERRIDE_COOLDOWN_S = 4 * 3600


def _is_held(ticker: str) -> bool:
    try:
        import db
        return db.get_position_stop(ticker) is not None
    except Exception:
        return False


def _title_names_ticker(ticker: str, title: str) -> bool | None:
    """Log-only (audit 11-06 09:31 #3): does the title name the ticker or its
    watchlist company, by the news fetcher's own test (mentions_company)?
    The held-negative override fires on tone alone, so a market-wide recap
    tagged to a held ticker reaches a full review. None when it can't be
    read. Never raises."""
    try:
        from ingestion.news import mentions_company
        from knowledge_loader import load_watchlist
        company = (load_watchlist().get(ticker) or {}).get("company", "")
        return mentions_company(ticker, company, title or "")
    except Exception:
        return None


def _held_negative_override(result: dict, ticker: str, source: str, title: str, held: bool) -> None:
    """Held stock + analyst rating change or bad-news headline: send it to the
    analyst even if the 7B said no (Luke D7, 10-29). The screener passed PT
    raises 93/171 but cuts 3/105, so held names got no review on bad news.
    One override per ticker per 4 h for headlines, kept apart from rating
    changes so a recap can't use up a downgrade's slot (review 4 #3), keeps
    recaps from flooding the 72B."""
    if result.get("relevant") or not held:
        return
    tone = title_tone(title)
    if source != "analyst_ratings" and tone not in ("neg", "mixed"):
        return
    try:
        import stop_lock
        kind = "rating" if source == "analyst_ratings" else "headline"
        if not stop_lock._redis().set(f"helios:held_override:{ticker}:{kind}", "1",
                                      nx=True, ex=_HELD_OVERRIDE_COOLDOWN_S):
            result["override_skipped"] = "cooldown"
            return
    except Exception as e:
        log.warning(f"[{ticker}] held-override cooldown unavailable, not overriding: {e}")
        return
    result.update(relevant=True, urgency="eod", override="held_negative_or_rating",
                  reason=f"Held position: {'rating change' if source == 'analyst_ratings' else 'negative headline'} "
                         f"(screener said: {str(result.get('reason', ''))[:120]})")
    log.info(f"[{ticker}] Screener override: held position, {source}, tone={tone}")


_RATING_DOWN = re.compile(r"\bdown\s+[A-Z][A-Z0-9.\-]*:")          # "<Firm> down NVDA: ..."
_PT_LOWERED = re.compile(r"price target (?:lowers|lowered|cuts?)\b", re.I)


def _would_pass_held_negative(result: dict, event: dict, source: str, title: str,
                              held: bool) -> dict | None:
    """Log-only (audit 2026-11-02 19:27 #4, roadmap 01(c)): a held name's
    event that is still screened out (after _held_negative_override) and
    reads negative on the fields the screener already has: the title tone,
    a rating "down" action or a lowered price target in an analyst_ratings
    title, or a BEARISH sentiment/direction field (options flow). Records
    that a held-aware screener would pass it. None unless held and screened
    out. The screener's decision never reads it. Never raises."""
    try:
        if not held or result.get("relevant"):
            return None
        hits = []
        tone = title_tone(title)
        if tone in ("neg", "mixed"):
            hits.append(f"tone_{tone}")
        if source == "analyst_ratings" and _RATING_DOWN.search(title or ""):
            hits.append("rating_down")
        if source == "analyst_ratings" and _PT_LOWERED.search(title or ""):
            hits.append("pt_lowered")
        for f in ("sentiment", "direction"):
            if str((event or {}).get(f) or "").upper() == "BEARISH":
                hits.append(f"{f}_bearish")
        return {"would_pass": bool(hits), "negative_signals": hits, "source": source,
                "override_skipped": result.get("override_skipped")}
    except Exception as e:
        return {"error": str(e)[:120]}


_EARNINGS_TITLE = re.compile(
    r"\b(earnings|results|Q[1-4]|guidance|outlook|beats?|miss(?:es)?|EPS|revenues?)\b", re.I)
_ITEM_202 = re.compile(r"Item\s+2\.02|8-K items:[^\n]*\b2\.02\b", re.I)
_POST_PRINT_KEY = "would_bypass_post_print_{env}"
# Unheld watchlist names keep their own book, so the held book's counts mean
# what they meant before unheld names were logged.
_POST_PRINT_UNHELD_KEY = "would_bypass_post_print_unheld_{env}"
_POST_PRINT_KEEP_DAYS = 14


def _earnings_like(event: dict, source: str, title: str) -> str | None:
    """'8k_item_2_02' for an 8-K carrying Item 2.02 (results of operations),
    'title' for a title that reads like a print, else None."""
    form = str((event or {}).get("form") or (event or {}).get("type") or "").upper()
    if source == "sec_filing" and form.startswith("8-K") and _ITEM_202.search(
            str((event or {}).get("text") or "")):
        return "8k_item_2_02"
    return "title" if _EARNINGS_TITLE.search(title or "") else None


def _next_trading_day(day):
    """The first NYSE session after `day` (weekends and holidays skipped)."""
    from datetime import timedelta
    from market_calendar import NYSE_HOLIDAYS
    d = day + timedelta(days=1)
    while d.weekday() >= 5 or d in NYSE_HOLIDAYS:
        d += timedelta(days=1)
    return d


def _log_would_bypass_post_print(result: dict, event: dict, ticker: str, source: str,
                                 title: str, held: bool, now=None) -> None:
    """Log-only: a held or watchlist name's earnings item that the screener
    drops in the post-print window: from 16:00 ET on the report day until
    09:30 ET on the next trading day, weekends and holidays included. A
    name's earnings items can all be screened out; _held_negative_override
    covers only negative tone and ratings, and only held names. Rows and book
    entries carry "held"; unheld names (watchlist only, any other ticker is
    skipped) count in _POST_PRINT_UNHELD_KEY, held names in _POST_PRINT_KEY. The earnings calendar has no before/after-open
    timing, so a before-open report's own morning is not covered. One activity_feed row (stage screener_shadow) for the first
    such item per (ticker, earnings date); later ones only raise the count
    kept in system_state. The screener's decision never reads it. DB reads
    only outside 09:30-16:00 on a session day, on a dropped, earnings-like
    item of a held or watchlist name. Never raises."""
    try:
        if result.get("relevant"):
            return
        kind = _earnings_like(event, source, title)
        if kind is None:
            return
        from datetime import date, datetime, time as dtime, timedelta, timezone
        from zoneinfo import ZoneInfo
        et_now = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
        today = et_now.date()
        from market_calendar import NYSE_HOLIDAYS
        in_session_day = today.weekday() < 5 and today not in NYSE_HOLIDAYS
        if in_session_day and dtime(9, 30) <= et_now.time() < dtime(16, 0):
            return
        held = bool(held)
        if not held:
            from knowledge_loader import load_watchlist
            if str(ticker or "").upper() not in load_watchlist():
                return
        import db
        ed = db.get_earnings_around([ticker], days_ahead=1, days_back=5).get(ticker)
        if ed is None:
            return
        ed = ed if isinstance(ed, date) else date.fromisoformat(str(ed)[:10])
        nxt = _next_trading_day(ed)
        if ed == today and et_now.time() >= dtime(16, 0):
            window = "after_close"
        elif ed < today < nxt or (today == nxt and et_now.time() < dtime(9, 30)):
            window = "pre_open"
        else:
            return
        import json as _json
        from config import TRADING_ENV
        from agents.analyst import shadow_versions
        key = (_POST_PRINT_KEY if held else _POST_PRINT_UNHELD_KEY).format(env=TRADING_ENV)
        try:
            book = _json.loads(db.get_state_strict(key, "") or "{}")
        except ValueError:
            log.warning(f"{key} is not valid JSON; starting a new post-print book")
            book = {}
        if not isinstance(book, dict):
            book = {}
        keep_from = (today - timedelta(days=_POST_PRINT_KEEP_DAYS)).isoformat()
        book = {k: v for k, v in book.items()
                if isinstance(v, dict) and str(v.get("earnings_date") or "") >= keep_from}
        sk = f"{ticker}:{ed.isoformat()}"
        entry = book.get(sk) or {"earnings_date": ed.isoformat(), "count": 0, "logged": False}
        entry["held"] = held
        entry["count"] = int(entry.get("count") or 0) + 1
        entry["last_at"] = et_now.isoformat()
        if not entry.get("logged"):
            item_id = event.get("_seen_key") or event.get("url") or event.get("id")
            entry["logged"] = bool(db.log_activity(
                "screener_shadow", ticker,
                f"would bypass (post-print): {'held' if held else 'watchlist'} {ticker} reported {ed}, screener dropped "
                f"{source} item: {str(title)[:80]}", "neutral",
                {"would_bypass_post_print": True, "held": held, "shadow_key": sk,
                 "item_id": str(item_id)[:300] if item_id else None,
                 "title": str(title)[:200], "source": source,
                 "form": event.get("form") or event.get("type"),
                 "actual_form": event.get("actual_form"),
                 "earnings_like": kind, "earnings_date": ed.isoformat(), "window": window,
                 "screener_reason": str(result.get("reason") or "")[:200],
                 "screener_model": result.get("model_used"),
                 "override_skipped": result.get("override_skipped"),
                 "count_at_log": entry["count"], "count_state_key": key,
                 "et_date": today.isoformat(), **shadow_versions()}))
            if entry["logged"]:
                entry["first_item_id"] = str(item_id)[:300] if item_id else None
        book[sk] = entry
        db.set_state(key, _json.dumps(book, default=str))
    except Exception as e:
        log.debug(f"[{ticker}] would_bypass_post_print log failed: {e}")


# Outage alert for the keyword fallback (M275). Its new_fact is None, which
# the crew new-fact gate counts as false, so a 7B outage blocks every
# news-path buy. Alert only: nothing here changes a screen or a route.
_FALLBACK_ALERT_COUNT = 3            # the 3rd fallback ...
_FALLBACK_ALERT_WINDOW_S = 15 * 60   # ... within 15 minutes ...
_FALLBACK_ALERT_THROTTLE_S = 3600    # ... alerts, at most once an hour


def _new_fact_window_phrase() -> str:
    """BUY_NEW_FACT_WINDOW as the fallback alert words it: "24h" -> "the
    last 24 h", "3d" -> "the last 3 d", same_run -> "the same run". A value
    that doesn't parse is shown as set."""
    import config
    raw = getattr(config, "BUY_NEW_FACT_WINDOW", "")
    m = re.fullmatch(r"\s*(\d+)\s*([hd])\s*", str(raw))
    if m:
        return f"the last {int(m.group(1))} {m.group(2)}"
    if str(raw).strip() == "same_run":
        return "the same run"
    return f"the last {raw}"


def _note_keyword_fallback(reason: str, now=None) -> bool:
    """Count a keyword-fallback screen (system_state, shared across worker
    processes) and send one Telegram on the _FALLBACK_ALERT_COUNT-th within
    _FALLBACK_ALERT_WINDOW_S, at most one per _FALLBACK_ALERT_THROTTLE_S
    (that key is written only after a successful send). True when sent.
    Never raises."""
    try:
        from datetime import datetime, timezone
        import alerts
        import db
        from config import TRADING_ENV
        now = now or datetime.now(timezone.utc)
        times_key = f"screener_fallback_times:{TRADING_ENV}"
        alert_key = f"screener_fallback_alert:{TRADING_ENV}"
        try:
            stamps = json.loads(db.get_state(times_key) or "[]")
        except (TypeError, ValueError):
            stamps = []
        recent = []
        for st in stamps if isinstance(stamps, list) else []:
            try:
                t = datetime.fromisoformat(st)
            except (TypeError, ValueError):
                continue
            if 0 <= (now - t).total_seconds() < _FALLBACK_ALERT_WINDOW_S:
                recent.append(st)
        recent.append(now.isoformat())
        recent = recent[-_FALLBACK_ALERT_COUNT:]
        db.set_state(times_key, json.dumps(recent))
        if len(recent) < _FALLBACK_ALERT_COUNT:
            return False
        last = db.get_state(alert_key)
        if last:
            try:
                if (now - datetime.fromisoformat(last)).total_seconds() < _FALLBACK_ALERT_THROTTLE_S:
                    return False
            except (TypeError, ValueError):
                pass   # unreadable stamp: send
        if alerts._send(f"Screener on the keyword fallback: {len(recent)} screens in "
                        f"{_FALLBACK_ALERT_WINDOW_S // 60} min (last: {reason}). Its new_fact "
                        f"is empty, so the new-fact gate blocks news-path OPEN/ADD unless a "
                        f"new-fact item from {_new_fact_window_phrase()} exists; "
                        f"check the 7B model server") is None:
            return False
        db.set_state(alert_key, now.isoformat())
        return True
    except Exception as e:
        log.warning(f"screener fallback alert failed: {e}")
        return False


def screen(event: dict) -> dict:
    """
    Main entry point. Returns screening result dict.

    Always returns a valid dict regardless of model availability.
    """
    ticker = event.get("ticker", "?")
    source = event.get("source", "unknown")
    title  = event.get("title", "")
    tier   = _get_tier(ticker)

    try:
        from feed import emit
        emit("screener", ticker,
             f"Reading {source.upper()} — {title[:70]}{'…' if len(title) > 70 else ''}",
             "neutral")
    except Exception:
        pass

    held = _is_held(ticker)
    prompt = _SCREENER_PROMPT.format(
        symbols=", ".join(SYMBOLS),
        source=source,
        ticker=ticker,
        tier=tier,
        held="yes" if held else "no",
        title=title,
        content=(event.get("text") or "")[:SCREENER_CONTENT_CAP],
    )

    raw = _call_ollama(prompt)

    fallback_reason = None
    urgency_missing = False
    reply_keys = None   # the 7B reply's own keys, before any defaults (professor m569)
    if raw is None:
        log.warning("Ollama unavailable — using keyword fallback for screening.")
        result = _keyword_fallback(event, tier=tier)
        fallback_reason = "ollama_unavailable"
    else:
        try:
            # Extract JSON from model response (model sometimes adds commentary)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if isinstance(result, dict):
                    reply_keys = sorted(str(k) for k in result)
                result["model_used"] = SCREENER_MODEL
            else:
                # The fallback judged it, not the 7B: model_used says so.
                log.warning("7B model returned no JSON — using fallback.")
                result = _keyword_fallback(event, tier=tier)
                fallback_reason = "no_json"
        except (json.JSONDecodeError, AttributeError):
            log.warning("7B model returned invalid JSON — using fallback.")
            result = _keyword_fallback(event, tier=tier)
            fallback_reason = "invalid_json"
    if fallback_reason:
        _note_keyword_fallback(fallback_reason)   # alert only (M275)
    # A 7B reply without urgency raised KeyError below and lost the event.
    # "eod" is the fallback's non-urgent value and the prompt's default.
    if result.get("urgency") is None:
        urgency_missing = True
        result["urgency"] = "eod"
    # "false" (a string) read as relevant. Parsed like new_fact; anything
    # else is dropped, which every reader takes as not relevant, as for a
    # missing value.
    relevant_invalid = False
    if "relevant" in result:
        relevant = parse_new_fact(result["relevant"])
        if relevant is None:
            relevant_invalid = True
            result.pop("relevant")
        else:
            result["relevant"] = relevant
    # The keyword fallback doesn't judge it: None there, as for a missing value.
    result["new_fact"] = (parse_new_fact(result.get("new_fact"))
                          if result.get("model_used") == SCREENER_MODEL else None)

    _held_negative_override(result, ticker, source, title, held)
    held_negative_shadow = _would_pass_held_negative(result, event, source, title, held)
    _log_would_bypass_post_print(result, event, ticker, source, title, held)   # log-only
    # Log-only: would a deterministic opinion-headline check have dropped
    # what the screener passed? Changes nothing in result.
    try:
        _ops = _opinion_headlines(title)
        _op = _ops[0] if _ops else None
        opinion_shadow = {"match": _op, "matches": _ops,
                          "would_drop": result.get("relevant") is True and _op is not None,
                          "patterns_sha": _OPINION_PATTERNS_SHA, "error": None}
    except Exception as e:
        log.debug(f"[{ticker}] opinion_shadow failed: {e}")
        opinion_shadow = {"match": None, "matches": [], "would_drop": False,
                          "patterns_sha": _OPINION_PATTERNS_SHA, "error": type(e).__name__}
    catalyst_shadow = _catalyst_shadow(ticker, title, result.get("relevant"),
                                       opinion_shadow.get("match"))

    # Provenance for the SCREENED_OUT signal row (SEF run 7, 2026-10-28):
    # those rows had no model or code version, so screener rejections couldn't
    # be tied to what made them.
    try:
        import hashlib
        from agents.analyst import _CODE_SHA, _code_mixed_check, _config_env_record
        result["provenance"] = {
            "code_sha": _CODE_SHA["code_sha"],
            "head_sha": _CODE_SHA["head_sha"],
            **_code_mixed_check(),  # professor M75
            # The screener's settings, and a hash of every env-overridable
            # config.py setting, as in the analyst's provenance (M182).
            "config": {"SCREENER_MODEL": SCREENER_MODEL, "env": _config_env_record()},
            "model": result.get("model_used") or "keyword_fallback",
            "model_digest": _screener_digest() if result.get("model_used") == SCREENER_MODEL else None,
            "prompt_template_sha": hashlib.sha256(_SCREENER_PROMPT.encode()).hexdigest()[:12],
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
            "title_tone": title_tone(title),
            # The event's source and form, so pass rates can be counted by
            # source from the rows (SEF, packet G54).
            "source": source,
            "form": event.get("form") or event.get("type"),
            # EDGAR only: the filing's own form; form is the matched prefix.
            "actual_form": event.get("actual_form"),
            "new_fact": result.get("new_fact"),
            "override": result.get("override"),
            "title_names_ticker": _title_names_ticker(ticker, title),
            "would_pass_held_negative": held_negative_shadow,
            "opinion_shadow": opinion_shadow,
            "catalyst_shadow": catalyst_shadow,
            # How much of the event text the 7B saw (audit 2026-11-05-0725 #10;
            # G54). Log-only.
            "content_len": len(event.get("text") or ""),
            "content_cut": len(event.get("text") or "") > SCREENER_CONTENT_CAP,
            # The tier and held flag the prompt was given, so pass rates can
            # be split by them as of the screen (audit 2026-11-07 15:07 #1).
            "tier": tier,
            "held": held,
            **_readthrough_record(result, ticker),
        }
        if fallback_reason:
            result["provenance"]["fallback_reason"] = fallback_reason
        if urgency_missing:
            result["provenance"]["urgency_missing"] = True
        if relevant_invalid:
            result["provenance"]["relevant_invalid"] = True
        if reply_keys is not None:
            # Record-only: whether the 7B left "reason" out, null or blank.
            rv = result.get("reason")
            result["provenance"]["reply_keys"] = reply_keys
            result["provenance"]["reason_missing"] = (
                "absent" if "reason" not in reply_keys else
                "null" if rv is None else
                "blank" if not str(rv).strip() else None)
    except Exception as e:
        log.warning(f"screener provenance failed: {e}")

    log.info(
        f"Screened [{source}] {ticker}: "
        f"relevant={result.get('relevant')} urgency={result.get('urgency')} tone={title_tone(title)} "
        f"new_fact={result.get('new_fact')} "
        f"tickers={result.get('affected_tickers',[])} "
        f"model={result.get('model_used','?')}"
    )

    try:
        from feed import emit
        # A null reason from the model used to raise on the slice, and the
        # except below dropped the feed row with no trace (49 on 11-05).
        reason = str(result.get("reason") or "")[:100]
        if result.get("relevant"):
            urgency = result.get("urgency") or "low"
            sentiment = "alert" if urgency == "immediate" else "neutral"
            emit("screener", ticker,
                 f"Relevant ({urgency}) — {reason}",
                 sentiment)
        else:
            emit("screener", ticker, f"Filtered out — {reason}", "neutral")
    except Exception:
        pass

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_event = {
        "source": "sec_filing",
        "ticker": "IONQ",
        "title":  "IonQ Announces $54.5M DoD Contract for Quantum Computing Services",
        "text":   "IonQ Inc. has been awarded a contract by the US Department of Defense...",
    }
    result = screen(test_event)
    print(f"\nScreening result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
