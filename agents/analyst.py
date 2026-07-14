"""
Analyst Agent — deep thesis evaluation using the local LLM.

Evaluates scanner candidates and event-triggered opportunities.
Produces a structured recommendation matching analyst_agent.md spec:
  action: OPEN | ADD | HOLD | TRIM | EXIT | WATCH | PASS
  conviction: 1-10 integer
  thesis_bucket: Compounder | Accelerator | Moonshot
  expectation_gap, catalyst, valuation, upside_downside_ratio, key_risks

Designed for 72B model reasoning quality; degrades gracefully on smaller models.
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

OLLAMA_HOST   = os.getenv("OLLAMA_HOST",    "http://localhost:11434")
ANALYST_MODEL = os.getenv("ANALYST_MODEL",  "qwen2.5:72b")

# qwen2.5 3b/7b/72b all support this context length natively — raised from
# the previous 8192, which risked silent truncation once briefs got large.
_NUM_CTX = 32768


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_track_record_text() -> str:
    """
    Realized win-rate and alpha-vs-SPY by conviction level, so the model can
    weigh its own stated confidence against how it has actually performed.
    Returns "" when there isn't enough closed-trade history yet.
    """
    try:
        import db
        calibration = db.get_conviction_calibration()
    except Exception:
        return ""

    rows = [c for c in calibration if c.get("total")]
    if not rows:
        return ""

    lines = ["Historical calibration — realized outcome by conviction you stated at entry:"]
    for c in rows:
        alpha = c.get("avg_alpha_pct")
        alpha_str = f"{alpha:+.1%} alpha vs SPY" if alpha is not None else "alpha unknown"
        lines.append(
            f"  Conviction {c['conviction']}/10: {c['total']} closed trades, "
            f"{c['win_rate']:.0%} win rate, {alpha_str}"
        )
    lines.append(
        "Weigh this record when you state your own conviction — if past conviction-9 calls "
        "realized a much lower win rate than 9/10 implies, that overconfidence should be corrected now."
    )
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough heuristic (~4 chars/token) — good enough for a budget guard, not exact."""
    return len(text) // 4


# Reserved for the model's own output (num_predict) plus a safety margin so
# the budget check doesn't cut it razor-thin.
_OUTPUT_TOKEN_RESERVE = 1200 + 300


def _build_prompt(
    candidate: dict,
    situation_brief: str,
    thesis_text: str,
    regime_packet: dict,
    philosophy_text: str,
    num_ctx: int = _NUM_CTX,
) -> str:
    ticker = candidate.get("ticker", "UNKNOWN")
    role   = candidate.get("role",   "Accelerator")
    tier   = candidate.get("tier",   "Watch")
    sector = candidate.get("sector", "")

    regime    = regime_packet.get("regime",   "Neutral")
    reg_score = regime_packet.get("score",    50)

    _markov = regime_packet.get("signals", {}).get("markov", {})
    markov_line = ""
    if _markov.get("state") is not None and _markov.get("stickiness") is not None and _markov.get("signal") is not None:
        markov_line = (
            f"\nSPY Markov state: {_markov['state']} "
            f"(stickiness {_markov['stickiness']:.0%}, directional signal {_markov['signal']:+.2f})"
        )

    # Generous caps — num_ctx was raised from 8192, and the budget loop below
    # trims further if the assembled prompt still doesn't fit.
    thesis_excerpt      = thesis_text[:6000]
    philosophy_excerpt  = philosophy_text[:2000]
    brief_text          = situation_brief[:8000] if situation_brief else "No additional context available."
    track_record        = _build_track_record_text()

    def _render() -> str:
        track_record_section = f"\n=== YOUR TRACK RECORD ===\n{track_record}\n" if track_record else ""
        return f"""You are a disciplined investment analyst evaluating whether to open, add, hold, trim, exit, or pass on a stock position.

=== PORTFOLIO PHILOSOPHY (excerpt) ===
{philosophy_excerpt}

=== SECTOR THESIS ===
{thesis_excerpt}

=== MARKET REGIME ===
Current Regime: {regime} (score: {reg_score}/100){markov_line}
{regime_packet.get("reasoning", "")[:500]}
{track_record_section}
=== CANDIDATE DETAILS ===
Ticker: {ticker}
Portfolio Role: {role}
Conviction Tier: {tier}
Sector: {sector}
Scanner composite score: {candidate.get("composite_score", 0)}/100

=== SITUATION BRIEF ===
{brief_text}

=== YOUR TASK ===
Evaluate {ticker} and produce a structured investment recommendation.

Answer these questions in order:
1. Which sector thesis does this fit? Is the thesis intact or has it weakened?
2. What does the current price IMPLY the market expects? What is MORE LIKELY to happen?
3. What is the specific catalyst? When? What happens if it disappoints?
4. Is the upside/downside ratio at least 2:1?
5. What are the top 2-3 risks?

Valuation grounding: if the situation brief includes a FUNDAMENTALS section
with real numbers, derive "current_multiple", "peer_range", and the
upside/downside estimates FROM THOSE FIGURES — do not invent a multiple.
If the FUNDAMENTALS section says data isn't cached yet, say so explicitly
in your reasoning and cap conviction at 6 for OPEN/ADD, since the valuation
call would otherwise be an unsupported guess.

Then output ONLY valid JSON in exactly this format (no markdown, no explanation outside the JSON):

{{
  "action": "OPEN or ADD or HOLD or TRIM or EXIT or WATCH or PASS",
  "conviction": 7,
  "thesis_bucket": "Compounder or Accelerator or Moonshot",
  "thesis_summary": "One sentence why this position exists.",
  "expectation_gap": {{
    "bull_case": "What needs to happen for upside",
    "current_price_implies": "What growth/outcome is already priced in",
    "upside_exists_if": "Condition under which upside materializes",
    "primary_risk": "Single most likely thing that makes this wrong"
  }},
  "catalyst": {{
    "event": "Description of catalyst",
    "date": "YYYY-MM or ongoing",
    "expected_upside_if_positive": "+X to +Y%",
    "expected_downside_if_negative": "-X to -Y%",
    "is_priced_in": false
  }},
  "valuation": {{
    "current_multiple": "Xx NTM earnings or revenue",
    "peer_range": "Range for comparable peers",
    "bull_case_target": "$X (12 months)",
    "downside_scenario": "$X"
  }},
  "upside_downside_ratio": 2.5,
  "key_risks": ["Risk 1", "Risk 2", "Risk 3"],
  "options_opportunity": false,
  "options_notes": "Brief note or null",
  "pass_to_risk": true,
  "reasoning": "2-3 sentence narrative explaining the recommendation."
}}

Rules:
- Set pass_to_risk to true ONLY if action is OPEN, ADD, TRIM, or EXIT AND conviction >= 5 AND upside_downside_ratio >= 2.0
- For Moonshot tickers: conviction >= 7 required to recommend OPEN; never exceed 2% position
- In {regime} regime: {'be selective, raise threshold to conviction 7+ for new positions' if regime in ('Neutral', 'Risk-Off', 'Strong Risk-Off') else 'normal conviction thresholds apply'}
{'- Your stated conviction has a track record above — calibrate against it, not against how confident you feel' if track_record else ''}
- Output ONLY the JSON object, nothing else
"""

    prompt = _render()

    # Token-budget guard: if the assembled prompt won't fit num_ctx minus the
    # output reserve, trim the lowest-priority sections first — the situation
    # brief's tail (context_engine builds it with the most important sections
    # first, so its tail is the least critical), then the thesis excerpt,
    # then the philosophy excerpt. Every trim is logged; nothing is silently
    # truncated by Ollama without us knowing.
    budget = num_ctx - _OUTPUT_TOKEN_RESERVE
    est = _estimate_tokens(prompt)
    if est > budget:
        original_est = est
        for _ in range(20):
            if est <= budget:
                break
            if len(brief_text) > 500:
                brief_text = brief_text[: int(len(brief_text) * 0.75)]
            elif len(thesis_excerpt) > 500:
                thesis_excerpt = thesis_excerpt[: int(len(thesis_excerpt) * 0.75)]
            elif len(philosophy_excerpt) > 300:
                philosophy_excerpt = philosophy_excerpt[: int(len(philosophy_excerpt) * 0.75)]
            else:
                break
            prompt = _render()
            est = _estimate_tokens(prompt)
        log.warning(
            f"[{ticker}] Prompt trimmed to fit context budget: "
            f"~{original_est} -> ~{est} tokens (budget {budget})"
        )

    return prompt


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, num_ctx: int = _NUM_CTX) -> str | None:
    payload = {
        "model":   ANALYST_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature":    0.2,
            "num_predict":    1200,
            "num_ctx":        num_ctx,
            "repeat_penalty": 1.1,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        log.warning(f"Ollama not reachable at {OLLAMA_HOST}")
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# JSON extraction (robust to 3B model quirks)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: pull individual fields from malformed output
    log.warning("JSON parse failed — attempting field extraction")
    extracted: dict = {}
    for field, pattern in [
        ("action",        r'"action"\s*:\s*"(\w+)"'),
        ("conviction",    r'"conviction"\s*:\s*(\d+)'),
        ("thesis_bucket", r'"thesis_bucket"\s*:\s*"([^"]+)"'),
        ("reasoning",     r'"reasoning"\s*:\s*"([^"]+)"'),
        ("pass_to_risk",  r'"pass_to_risk"\s*:\s*(true|false)'),
        ("upside_downside_ratio", r'"upside_downside_ratio"\s*:\s*([0-9.]+)'),
    ]:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            val = m.group(1)
            if field == "conviction":
                extracted[field] = int(val)
            elif field == "pass_to_risk":
                extracted[field] = val.lower() == "true"
            elif field == "upside_downside_ratio":
                extracted[field] = float(val)
            else:
                extracted[field] = val

    return extracted if extracted else None


# ---------------------------------------------------------------------------
# Stub for unavailable model
# ---------------------------------------------------------------------------

def _stub_recommendation(candidate: dict, reason: str) -> dict:
    ticker = candidate.get("ticker", "UNKNOWN")
    log.warning(f"[{ticker}] Stub recommendation — {reason}")
    try:
        from feed import emit
        emit("analyst", ticker, f"Model unavailable ({reason}) — skipping", "neutral")
    except Exception:
        pass
    return {
        "ticker":         ticker,
        "action":         "WATCH",
        "conviction":     1,
        "thesis_bucket":  candidate.get("role", "Accelerator"),
        "thesis_summary": f"{ticker} on watchlist — model unavailable.",
        "expectation_gap": {
            "bull_case": "Unknown", "current_price_implies": "Unknown",
            "upside_exists_if": "Unknown", "primary_risk": "Unknown",
        },
        "catalyst": {
            "event": "Unknown", "date": "Unknown",
            "expected_upside_if_positive": "Unknown",
            "expected_downside_if_negative": "Unknown",
            "is_priced_in": False,
        },
        "valuation": {
            "current_multiple": "Unknown", "peer_range": "Unknown",
            "bull_case_target": "Unknown", "downside_scenario": "Unknown",
        },
        "upside_downside_ratio": 0.0,
        "key_risks":       [f"Model unavailable: {reason}"],
        "options_opportunity": False,
        "options_notes":   None,
        "pass_to_risk":    False,
        "reasoning":       f"Stub: {reason}. Model ({ANALYST_MODEL}) not available.",
        "model_used":      f"stub",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main analyst function
# ---------------------------------------------------------------------------

def analyze(
    candidate: dict,
    situation_brief: str = "",
    regime_packet: dict | None = None,
) -> dict:
    """
    Evaluate a scanner candidate and produce an investment recommendation.

    Args:
        candidate:       Scanner candidate dict (must have 'ticker', 'role', 'tier').
        situation_brief: Assembled context string from context_engine.
        regime_packet:   Market regime output packet.

    Returns:
        Full analyst recommendation dict matching analyst_agent.md spec.
    """
    ticker = candidate.get("ticker", "UNKNOWN")
    log.info(f"[{ticker}] Analyzing with {ANALYST_MODEL}...")

    try:
        from feed import emit
        emit("analyst", ticker, f"Starting deep analysis with {ANALYST_MODEL}…", "neutral")
    except Exception:
        pass

    if regime_packet is None:
        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
        except Exception:
            regime_packet = {"regime": "Neutral", "score": 50, "reasoning": ""}

    try:
        from knowledge_loader import load_thesis, load_philosophy
        thesis_text     = load_thesis(ticker)
        philosophy_text = load_philosophy()
    except Exception as e:
        log.warning(f"[{ticker}] Knowledge load failed: {e}")
        thesis_text     = ""
        philosophy_text = ""

    prompt = _build_prompt(candidate, situation_brief, thesis_text, regime_packet, philosophy_text)

    raw_response = _call_ollama(prompt)
    if not raw_response:
        return _stub_recommendation(candidate, "Ollama unreachable")

    parsed = _extract_json(raw_response)
    if not parsed:
        log.warning(f"[{ticker}] Could not parse model response")
        return _stub_recommendation(candidate, "JSON parse failed")

    action     = str(parsed.get("action", "WATCH")).upper()
    conviction = max(1, min(10, int(parsed.get("conviction", 1) or 1)))
    udr        = float(parsed.get("upside_downside_ratio", 0) or 0)

    # Bucket is watchlist-authoritative, not an LLM call — strategy ("strategy
    # is data, not code") lives in knowledge/watchlist.md, not in whatever a
    # given inference happens to output. A DDOG recommendation was sized as
    # Compounder (wider caps, wider ATR stop) purely because the model said so
    # in its JSON, despite being told the watchlist role was Accelerator —
    # silently accepting that let a weak/misconfigured model quietly loosen
    # risk limits. The LLM's suggestion is kept for audit but never used for
    # sizing; update the watchlist by hand if a reclassification is deliberate.
    watchlist_role = candidate.get("role", "Accelerator")
    llm_bucket     = parsed.get("thesis_bucket", watchlist_role)
    bucket_mismatch = llm_bucket != watchlist_role
    if bucket_mismatch:
        log.warning(
            f"[{ticker}] Analyst suggested bucket '{llm_bucket}' but watchlist role "
            f"is '{watchlist_role}' — using the watchlist role for sizing. Update "
            f"knowledge/watchlist.md if this reclassification should stick."
        )
    bucket = watchlist_role

    # Enforce minimum UDR for actionable recommendations
    if udr < 2.0 and action in ("OPEN", "ADD"):
        log.info(f"[{ticker}] UDR {udr:.1f} < 2.0 — downgrading to WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False
        parsed["reasoning"] = (parsed.get("reasoning", "") +
                               f" [Downgraded: UDR {udr:.1f} < 2.0 minimum]")

    # Moonshots need conviction >= 7 for OPEN
    if bucket == "Moonshot" and action == "OPEN" and conviction < 7:
        log.info(f"[{ticker}] Moonshot conviction {conviction} < 7 — downgrading to WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False

    # Deterministic guardrail for ungrounded valuation: the prompt instructs
    # the model to cap its own conviction at 6 when fundamentals aren't
    # cached, but a prompt instruction isn't reliable enough on its own —
    # enforce it here too so an OPEN/ADD can't ride on an invented multiple.
    try:
        import db
        fund = db.get_fundamentals(ticker)
        fundamentals_missing = not fund or fund.get("market_cap") is None
    except Exception:
        fundamentals_missing = True
    if fundamentals_missing and action in ("OPEN", "ADD") and conviction > 6:
        log.info(f"[{ticker}] No cached fundamentals — capping conviction {conviction} -> 6")
        conviction = 6

    # Validate pass_to_risk
    actionable = action in ("OPEN", "ADD", "TRIM", "EXIT")
    if parsed.get("pass_to_risk") and not (actionable and conviction >= 5):
        parsed["pass_to_risk"] = False

    try:
        import db
        db.log_signal(
            symbol=ticker,
            signal=action,
            confidence=conviction / 10.0,
            reasoning=parsed.get("reasoning", "")[:500],
        )
    except Exception as e:
        log.debug(f"[{ticker}] Signal log failed: {e}")

    result = {
        "ticker":    ticker,
        "action":    action,
        "conviction": conviction,
        "thesis_bucket": bucket,
        "llm_suggested_bucket": llm_bucket if bucket_mismatch else None,
        "thesis_summary": parsed.get("thesis_summary", ""),
        "expectation_gap": parsed.get("expectation_gap", {}),
        "catalyst":  parsed.get("catalyst", {}),
        "valuation": parsed.get("valuation", {}),
        "upside_downside_ratio": udr,
        "key_risks": parsed.get("key_risks", []),
        "options_opportunity": bool(parsed.get("options_opportunity", False)),
        "options_notes": parsed.get("options_notes"),
        "pass_to_risk": bool(parsed.get("pass_to_risk", False)),
        "reasoning": parsed.get("reasoning", ""),
        "model_used": ANALYST_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        f"[{ticker}] → {action} | conviction={conviction}/10 | "
        f"UDR={udr:.1f} | pass_to_risk={result['pass_to_risk']}"
    )

    try:
        from feed import emit
        reasoning_snip = result.get("reasoning", "")[:140]
        if action in ("OPEN", "ADD"):
            sentiment = "bullish"
        elif action in ("TRIM", "EXIT"):
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        pass_flag = " — sending to Risk Manager" if result["pass_to_risk"] else ""
        emit("analyst", ticker,
             f"{action} | conviction {conviction}/10 | UDR {udr:.1f}x{pass_flag} — {reasoning_snip}",
             sentiment)
    except Exception:
        pass

    return result


def analyze_batch(
    candidates: list[dict],
    regime_packet: dict | None = None,
    max_candidates: int = 10,
    positions: dict[str, dict] | None = None,
    trading_client=None,
) -> list[dict]:
    """
    Analyze a batch of scanner candidates.
    Limits to max_candidates to control inference time on smaller models.

    Each candidate gets a full situation brief (context_engine.build_scan_brief)
    — macro, technicals, options flow, fundamentals, and vector-memory
    retrieval — the same context quality the rare event-triggered path gets.
    Previously this called analyze() with no brief at all, so the daily scan
    (the main decision flow) ran on scanner scores alone.
    """
    top = sorted(
        candidates[:max_candidates],
        key=lambda c: (c.get("priority") == "HIGH", c.get("composite_score", 0)),
        reverse=True,
    )

    results = []
    for candidate in top:
        ticker = candidate.get("ticker")
        brief = ""
        try:
            from context_engine import build_scan_brief
            brief = build_scan_brief(ticker, positions, trading_client)
        except Exception as e:
            log.warning(f"[{ticker}] Scan brief build failed, analyzing without it: {e}")

        try:
            result = analyze(candidate, situation_brief=brief, regime_packet=regime_packet)
            results.append(result)
        except Exception as e:
            log.error(f"[{ticker}] Analyze failed: {e}")
            results.append(_stub_recommendation(candidate, f"Exception: {e}"))

    return results
