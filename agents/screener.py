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

Respond ONLY with valid JSON, no other text:
{{
  "relevant": true or false,
  "urgency": "immediate" or "eod",
  "affected_tickers": ["TICK1", "TICK2"],
  "reason": "one sentence explanation — if relevant, name the specific new fact",
  "indirect_effects": "brief description of any second-order effects, or null"
}}
"""


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


_HELD_OVERRIDE_COOLDOWN_S = 4 * 3600


def _is_held(ticker: str) -> bool:
    try:
        import db
        return db.get_position_stop(ticker) is not None
    except Exception:
        return False


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
        content=event.get("text", "")[:500],
    )

    raw = _call_ollama(prompt)

    if raw is None:
        log.debug("Ollama unavailable — using keyword fallback for screening.")
        result = _keyword_fallback(event, tier=tier)
    else:
        try:
            # Extract JSON from model response (model sometimes adds commentary)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(json_match.group()) if json_match else _keyword_fallback(event, tier=tier)
            result["model_used"] = SCREENER_MODEL
        except (json.JSONDecodeError, AttributeError):
            log.warning("7B model returned invalid JSON — using fallback.")
            result = _keyword_fallback(event, tier=tier)

    _held_negative_override(result, ticker, source, title, held)

    # Provenance for the SCREENED_OUT signal row (SEF run 7, 2026-10-28):
    # those rows had no model or code version, so screener rejections couldn't
    # be tied to what made them.
    try:
        import hashlib
        from agents.analyst import _CODE_SHA, _code_mixed_check
        result["provenance"] = {
            "code_sha": _CODE_SHA["code_sha"],
            "head_sha": _CODE_SHA["head_sha"],
            **_code_mixed_check(),  # professor M75
            "model": result.get("model_used") or "keyword_fallback",
            "model_digest": _screener_digest() if result.get("model_used") else None,
            "prompt_template_sha": hashlib.sha256(_SCREENER_PROMPT.encode()).hexdigest()[:12],
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
            "title_tone": title_tone(title),
            "override": result.get("override"),
        }
    except Exception as e:
        log.warning(f"screener provenance failed: {e}")

    log.info(
        f"Screened [{source}] {ticker}: "
        f"relevant={result['relevant']} urgency={result['urgency']} tone={title_tone(title)} "
        f"tickers={result.get('affected_tickers',[])} "
        f"model={result.get('model_used','?')}"
    )

    try:
        from feed import emit
        reason = result.get("reason", "")[:100]
        if result.get("relevant"):
            urgency = result.get("urgency", "low")
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
