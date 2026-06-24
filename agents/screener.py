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
SCREENER_MODEL = os.getenv("SCREENER_MODEL", "qwen2.5:7b")

from config import SYMBOLS

_SCREENER_PROMPT = """\
You are a financial analyst screening events for a trading system that holds \
tech, AI, semiconductor, and quantum computing stocks.

WATCHLIST: {symbols}

EVENT TO SCREEN:
Source: {source}
Ticker: {ticker}
Title: {title}
Content (first 500 chars): {content}

Your job is to determine if this event is materially relevant to ANY stock \
in our watchlist — including INDIRECT effects (e.g., a TSMC supply issue \
affects NVDA even though TSMC isn't in our list).

Respond ONLY with valid JSON, no other text:
{{
  "relevant": true or false,
  "urgency": "immediate" or "eod" or "low",
  "affected_tickers": ["TICK1", "TICK2"],
  "reason": "one sentence explanation",
  "indirect_effects": "brief description of any second-order effects, or null"
}}

Urgency definitions:
- immediate: market-moving news, act within the hour
- eod:       relevant for end-of-day review
- low:       background context, no urgent action needed
"""


def _call_ollama(prompt: str) -> str | None:
    """Calls the local Ollama 7B model. Returns raw text or None if unavailable."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  SCREENER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return None  # Ollama not running yet
    except Exception as e:
        log.warning(f"Ollama 7B call failed: {e}")
        return None


def _keyword_fallback(event: dict) -> dict:
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

    # Low-signal keywords
    low_keywords = ["routine", "form 4", "scheduled", "annual report"]

    urgency = "low"
    for kw in urgent_keywords:
        if kw in text:
            urgency = "immediate"
            break
    if urgency == "low":
        urgency = "eod"

    # Find affected tickers by mention
    affected = [ticker] if ticker in SYMBOLS else []
    for sym in SYMBOLS:
        if sym.lower() in text and sym not in affected:
            affected.append(sym)

    relevant = len(affected) > 0 or any(kw in text for kw in urgent_keywords)

    return {
        "relevant":         relevant,
        "urgency":          urgency if relevant else "low",
        "affected_tickers": affected,
        "reason":           "Keyword heuristic (Ollama not available)",
        "indirect_effects": None,
        "model_used":       "keyword_fallback",
    }


def screen(event: dict) -> dict:
    """
    Main entry point. Returns screening result dict.

    Always returns a valid dict regardless of model availability.
    """
    ticker = event.get("ticker", "?")
    source = event.get("source", "unknown")
    title  = event.get("title", "")

    try:
        from feed import emit
        emit("screener", ticker,
             f"Reading {source.upper()} — {title[:70]}{'…' if len(title) > 70 else ''}",
             "neutral")
    except Exception:
        pass

    prompt = _SCREENER_PROMPT.format(
        symbols=", ".join(SYMBOLS),
        source=source,
        ticker=ticker,
        title=title,
        content=event.get("text", "")[:500],
    )

    raw = _call_ollama(prompt)

    if raw is None:
        log.debug("Ollama unavailable — using keyword fallback for screening.")
        result = _keyword_fallback(event)
    else:
        try:
            # Extract JSON from model response (model sometimes adds commentary)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(json_match.group()) if json_match else _keyword_fallback(event)
            result["model_used"] = SCREENER_MODEL
        except (json.JSONDecodeError, AttributeError):
            log.warning("7B model returned invalid JSON — using fallback.")
            result = _keyword_fallback(event)

    log.info(
        f"Screened [{source}] {ticker}: "
        f"relevant={result['relevant']} urgency={result['urgency']} "
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
