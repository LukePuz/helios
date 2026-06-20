"""
Options flow ingestion via Unusual Whales API.
Unusual options activity (large call sweeps, high IV, out-of-the-money bets)
is one of the strongest leading indicators for short-term price moves.
"Smart money" often buys options before news becomes public.

Requires UNUSUAL_WHALES_API_KEY in .env
Sign up at unusualwhales.com (~$50/mo)
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

API_KEY  = os.getenv("UNUSUAL_WHALES_API_KEY", "")
BASE_URL = "https://api.unusualwhales.com/api"

from config import SYMBOLS


def _get_flow(ticker: str) -> list[dict]:
    """Fetches recent unusual options flow for a ticker."""
    if not API_KEY:
        return []
    try:
        resp = requests.get(
            f"{BASE_URL}/stock/{ticker}/flow",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        log.warning(f"Unusual Whales fetch failed for {ticker}: {e}")
        return []


def _is_unusual(flow: dict) -> bool:
    """
    Flags an options contract as unusual if it meets any of these:
    - Premium > $500k (institutional size)
    - OTM (out of the money) — speculative bet on a move
    - Expiry < 30 days (near-term conviction)
    - Ask-side fill (aggressor paid the ask = urgency)
    """
    try:
        premium   = float(flow.get("premium", 0))
        otm       = flow.get("otm", False)
        ask_side  = flow.get("side", "").lower() == "ask"
        return premium > 500_000 and (otm or ask_side)
    except (ValueError, TypeError):
        return False


def get_flow_summary(ticker: str) -> dict:
    """
    Returns a structured summary of current options flow for a ticker.
    Used by context_engine to populate situation brief.
    """
    flows = _get_flow(ticker)
    if not flows:
        return {"ticker": ticker, "unusual": False, "summary": "No options flow data."}

    unusual = [f for f in flows if _is_unusual(f)]
    call_premium = sum(float(f.get("premium", 0)) for f in flows if f.get("type") == "call")
    put_premium  = sum(float(f.get("premium", 0)) for f in flows if f.get("type") == "put")
    total        = call_premium + put_premium
    call_ratio   = call_premium / total if total > 0 else 0.5

    sentiment = "BULLISH" if call_ratio > 0.65 else "BEARISH" if call_ratio < 0.35 else "NEUTRAL"

    summary_text = (
        f"{ticker} options flow: {sentiment} "
        f"(call/put ratio {call_ratio:.0%}). "
        f"Total premium: ${total:,.0f}. "
        f"Unusual contracts detected: {len(unusual)}."
    )
    if unusual:
        top = unusual[0]
        summary_text += (
            f" Largest unusual: ${float(top.get('premium', 0)):,.0f} "
            f"{top.get('type','').upper()} "
            f"@ ${top.get('strike', '?')} expiring {top.get('expiry', '?')}."
        )

    return {
        "ticker":      ticker,
        "unusual":     len(unusual) > 0,
        "sentiment":   sentiment,
        "call_ratio":  round(call_ratio, 3),
        "total_premium": total,
        "unusual_count": len(unusual),
        "summary":     summary_text,
    }


def get_all_flow_summaries() -> dict[str, dict]:
    """Returns flow summary for every symbol. Used in context assembly."""
    if not API_KEY:
        log.debug("UNUSUAL_WHALES_API_KEY not set — options flow skipped.")
        return {s: {"ticker": s, "unusual": False, "summary": "API key not configured."} for s in SYMBOLS}

    summaries = {}
    for ticker in SYMBOLS:
        summaries[ticker] = get_flow_summary(ticker)
        time.sleep(0.2)
    return summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not API_KEY:
        print("UNUSUAL_WHALES_API_KEY not set in .env — add it to activate options flow.")
    else:
        summaries = get_all_flow_summaries()
        for ticker, s in summaries.items():
            flag = "⚠️ UNUSUAL" if s["unusual"] else "  normal  "
            print(f"[{flag}] {ticker}: {s['summary']}")
