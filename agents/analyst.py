"""
72B Analyst Agent — deep reasoning engine.

Only called when the 7B screener flags an event as relevant.
Receives the full assembled situation brief and produces a structured
investment thesis with conviction score, position sizing, and
explicit invalidation conditions.

Uses Ollama (qwen2.5:72b) when available.
Returns a stub thesis during development so the rest of the
pipeline can be tested before the new build arrives.

Reasoning target: 30-90 seconds per call.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANALYST_MODEL  = os.getenv("ANALYST_MODEL", "qwen2.5:72b")

_ANALYST_SYSTEM = """\
You are a senior quantitative analyst and portfolio manager specializing in \
technology, artificial intelligence, semiconductor, and quantum computing stocks. \
You combine deep technical knowledge of these industries with rigorous \
quantitative analysis.

Your role is to analyze events and produce actionable investment theses. \
You think in terms of:
- Narrative shifts: is the market's story about this company changing?
- Competitive moats: does this event widen or narrow the moat?
- Second-order effects: who else benefits or suffers indirectly?
- Timing: when will this be fully priced in?
- Risk asymmetry: what's the upside vs downside scenario?

You are disciplined and never recommend a position size larger than the \
risk manager allows. You explicitly state what would invalidate your thesis \
so positions can be exited cleanly if wrong.
"""

_ANALYST_PROMPT = """\
{situation_brief}

=== YOUR TASK ===
Based on the complete situation brief above, produce a rigorous investment \
thesis for how to respond to this event.

Consider:
1. What narrative shift does this represent, if any?
2. What is the direct price impact on the triggering ticker?
3. What are the indirect effects on related holdings?
4. What does the macro environment mean for this trade?
5. What does the options flow tell us about smart money positioning?
6. What historical parallels exist from our memory?
7. What would make you wrong?

Respond ONLY with valid JSON, no other text:
{{
  "signal": "BUY" or "SELL" or "HOLD" or "WATCH",
  "ticker": "primary ticker to act on",
  "conviction": 0.0 to 1.0,
  "thesis": "2-3 sentence investment thesis explaining the narrative shift",
  "time_horizon": "intraday" or "1-3 days" or "1-2 weeks" or "1+ month",
  "suggested_position_size_pct": 0.01 to 0.04,
  "price_target": null or numeric,
  "stop_loss": null or numeric,
  "invalidation_conditions": ["condition 1", "condition 2"],
  "indirect_effects": {{
    "TICKER": "bullish/bearish/neutral — reason"
  }},
  "risks": ["risk 1", "risk 2", "risk 3"],
  "confidence_factors": ["what makes you confident"],
  "uncertainty_factors": ["what makes you uncertain"]
}}

IMPORTANT: Only recommend conviction > 0.7 if multiple data sources confirm \
the thesis. Be conservative — missing a trade is better than a bad trade.
"""


def _call_ollama(situation_brief: str) -> str | None:
    """Calls the local Ollama 72B model. Returns raw text or None."""
    prompt = _ANALYST_PROMPT.format(situation_brief=situation_brief)
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  ANALYST_MODEL,
                "system": _ANALYST_SYSTEM,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature":    0.2,   # low temp for analytical consistency
                    "num_predict":    1024,
                    "num_ctx":        8192,  # 8k context window
                    "repeat_penalty": 1.1,
                },
            },
            timeout=180,  # 72B can take up to 3 minutes
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return None  # Ollama not running yet
    except Exception as e:
        log.warning(f"Ollama 72B call failed: {e}")
        return None


def _stub_thesis(event: dict, screening_result: dict) -> dict:
    """
    Returns a conservative HOLD stub when Ollama isn't available.
    The pipeline stays functional — just no real AI reasoning yet.
    Logs clearly so we know the stub was used.
    """
    log.info("72B model unavailable — returning HOLD stub thesis.")
    return {
        "signal":                    "HOLD",
        "ticker":                    event.get("ticker", "UNKNOWN"),
        "conviction":                0.0,
        "thesis":                    "Model unavailable — no thesis generated. "
                                     "Defaulting to HOLD to avoid uninformed trades.",
        "time_horizon":              "N/A",
        "suggested_position_size_pct": 0.0,
        "price_target":              None,
        "stop_loss":                 None,
        "invalidation_conditions":   [],
        "indirect_effects":          {},
        "risks":                     ["Model unavailable"],
        "confidence_factors":        [],
        "uncertainty_factors":       ["72B model not running — install Ollama and pull qwen2.5:72b"],
        "model_used":                "stub",
        "timestamp":                 datetime.now(timezone.utc).isoformat(),
    }


def analyze(situation_brief: str, event: dict, screening_result: dict) -> dict:
    """
    Main entry point. Takes assembled situation brief, returns thesis dict.
    Always returns a valid dict — never raises.
    """
    raw = _call_ollama(situation_brief)

    if raw is None:
        return _stub_thesis(event, screening_result)

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in model response")
        thesis = json.loads(json_match.group())
        thesis["model_used"] = ANALYST_MODEL
        thesis["timestamp"]  = datetime.now(timezone.utc).isoformat()

        # Sanity clamp conviction to [0, 1]
        thesis["conviction"] = max(0.0, min(1.0, float(thesis.get("conviction", 0))))

        log.info(
            f"Thesis generated: {thesis.get('signal')} {thesis.get('ticker')} "
            f"conviction={thesis.get('conviction'):.2f} "
            f"horizon={thesis.get('time_horizon')}"
        )
        return thesis

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        log.error(f"72B model returned unparseable response: {e}\nRaw: {raw[:500]}")
        return _stub_thesis(event, screening_result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a fake situation brief
    test_brief = """=== SITUATION BRIEF ===
Generated: Friday June 20, 2026 10:30 AM ET
Triggering Event: SEC_FILING — IONQ
Event Type: 8-K
Event Title: IonQ Announces $54.5M DoD Contract

=== EVENT CONTENT ===
IonQ Inc. has been awarded a $54.5 million contract by the US Department of Defense
to provide quantum computing services across multiple branches of the armed forces.
This represents IonQ's largest government contract to date.

=== CURRENT PORTFOLIO ===
  IONQ    100 shares | Entry: $14.20 | Current: $15.10 | P&L: +6.3%

=== CURRENT MACRO ENVIRONMENT ===
Fed Funds Rate: 5.25
VIX: 17.2 — low volatility, risk-on environment"""

    test_event = {"ticker": "IONQ", "source": "sec_filing"}
    test_screen = {"relevant": True, "urgency": "immediate"}

    thesis = analyze(test_brief, test_event, test_screen)
    print("\nThesis:")
    for k, v in thesis.items():
        print(f"  {k}: {v}")
