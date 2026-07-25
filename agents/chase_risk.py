"""
Chase risk — a deterministic (not LLM-estimated) sub-score answering a
narrower question than the existing conviction sub-scores: not "is this a
good company/thesis" but "would buying at today's price mean chasing an
already-extended move". A great thesis can still score high conviction
while chase_risk flags that right now is a bad entry.

Computed from the Ichimoku Kijun-sen (26-period midpoint of high/low) --
price's distance from that line is a standard, well-established measure of
how far a stock has moved from its own recent equilibrium. Deliberately
computed from real cached price history (ingestion/price_history.py's
nightly-refreshed `price_history` table) rather than asked of the LLM: the
model has a known failure mode of inventing plausible-sounding numbers when
it lacks a real anchor (see context_engine.assemble_brief's docstring on
the 2026-07-17 SNDK price hallucination) -- this is exactly that kind of
numeric-extension judgment, so it's computed in code instead.

2026-07-25: added informational-only per user decision -- this does not
change position sizing or gating, only surfaces the score in trade_outcomes
and Telegram alerts. A future pass may act on it (e.g. staged entries).
"""
import logging

log = logging.getLogger(__name__)

KIJUN_PERIOD = 26

# % distance above Kijun-sen -> 1-5 score. Same polarity as the other
# conviction sub-scores (5 = supportive of buying now, 1 = working against
# it) -- at or below equilibrium is never a "chase", so it floors at 5.
_THRESHOLDS = [
    (0,  5),
    (5,  4),
    (10, 3),
    (20, 2),
]


def _score_from_pct(pct_from_kijun: float) -> int:
    if pct_from_kijun <= 0:
        return 5
    for threshold, score in _THRESHOLDS[1:]:
        if pct_from_kijun <= threshold:
            return score
    return 1


def compute_chase_risk(ticker: str, period: int = KIJUN_PERIOD) -> dict | None:
    """
    Returns {"score": 1-5, "pct_from_kijun": float, "kijun_sen": float,
    "reference_price": float} or None if there isn't enough cached price
    history yet (new tickers, or before the nightly refresh has run twice).
    `reference_price` is the most recent cached daily close, not a live
    quote -- Ichimoku is inherently a daily-chart concept, and this is
    meant to flag medium-term extension, not intraday timing.
    """
    try:
        import db
        bars = db.get_price_history(ticker, days=period)
    except Exception as e:
        log.debug(f"[{ticker}] chase_risk: price history fetch failed: {e}")
        return None

    if len(bars) < period:
        return None

    window = bars[-period:]
    highest_high = max(b["high"] for b in window)
    lowest_low = min(b["low"] for b in window)
    kijun_sen = (highest_high + lowest_low) / 2
    if kijun_sen <= 0:
        return None

    reference_price = window[-1]["close"]
    pct_from_kijun = (reference_price - kijun_sen) / kijun_sen * 100

    return {
        "score": _score_from_pct(pct_from_kijun),
        "pct_from_kijun": round(pct_from_kijun, 2),
        "kijun_sen": round(kijun_sen, 2),
        "reference_price": reference_price,
    }
