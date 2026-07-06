"""
Free options-flow proxy via yfinance option chains.

Unusual Whales (~$50/mo) gives real sweep/premium data; this gives a much
weaker but free substitute — put/call volume ratio and open-interest deltas
from the nearest-dated chain — so the scanner and regime signals aren't
silently zeroed out just because UNUSUAL_WHALES_API_KEY is a placeholder.
Per the improvement roadmap: only pay for Unusual Whales once the signal
calibration table shows this component actually predicts outcomes.

Same return shape as ingestion.options_flow so callers can't tell which
backend answered — see options_flow.py's fallback dispatch.
"""

import logging
import time

import yfinance as yf

log = logging.getLogger(__name__)


def _nearest_chain(ticker: str):
    t = yf.Ticker(ticker)
    expirations = t.options
    if not expirations:
        return None, None
    calls_df = t.option_chain(expirations[0]).calls
    puts_df  = t.option_chain(expirations[0]).puts
    return calls_df, puts_df


def get_flow_summary(ticker: str) -> dict:
    """
    Free proxy for options_flow.get_flow_summary(). Uses call/put VOLUME
    (not premium — that data isn't reliably available for free) from the
    nearest expiry as a sentiment proxy.
    """
    try:
        calls, puts = _nearest_chain(ticker)
    except Exception as e:
        log.debug(f"[{ticker}] yfinance option chain fetch failed: {e}")
        calls, puts = None, None

    if calls is None or puts is None or (calls.empty and puts.empty):
        return {"ticker": ticker, "unusual": False,
                "summary": "No free options data available (yfinance chain empty)."}

    call_vol = float(calls["volume"].fillna(0).sum())
    put_vol  = float(puts["volume"].fillna(0).sum())
    total_vol = call_vol + put_vol
    call_ratio = call_vol / total_vol if total_vol > 0 else 0.5

    # OI concentration as a rough "unusual" proxy: any single strike holding
    # a large share of total OI suggests a concentrated bet.
    call_oi = calls["openInterest"].fillna(0)
    put_oi  = puts["openInterest"].fillna(0)
    total_oi = float(call_oi.sum() + put_oi.sum())
    max_strike_oi = float(max(call_oi.max() if len(call_oi) else 0,
                              put_oi.max() if len(put_oi) else 0))
    concentrated = total_oi > 0 and (max_strike_oi / total_oi) > 0.25

    sentiment = "BULLISH" if call_ratio > 0.65 else "BEARISH" if call_ratio < 0.35 else "NEUTRAL"
    unusual = concentrated or call_ratio > 0.70 or call_ratio < 0.30

    summary_text = (
        f"{ticker} options (free proxy): {sentiment} "
        f"(call/put volume ratio {call_ratio:.0%}). "
        f"Total contract volume: {total_vol:,.0f}."
    )
    if concentrated:
        summary_text += " Notable open-interest concentration at a single strike."

    return {
        "ticker":        ticker,
        "unusual":       unusual,
        "sentiment":     sentiment,
        "call_ratio":    round(call_ratio, 3),
        "total_premium": None,  # not available for free — do not fabricate
        "unusual_count": 1 if unusual else 0,
        "summary":       summary_text,
    }


def get_all_flow_summaries() -> dict[str, dict]:
    from knowledge_loader import get_all_tickers
    summaries = {}
    for ticker in get_all_tickers():
        summaries[ticker] = get_flow_summary(ticker)
        time.sleep(0.2)
    return summaries


def get_darkpool_activity(ticker: str) -> dict:
    """No free reliable dark-pool data source — explicit stub, not a guess."""
    return {"ticker": ticker, "has_significant_print": False, "total_value": 0,
            "largest_print": 0, "summary": "Dark pool data unavailable (no free source)."}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from knowledge_loader import get_all_tickers
    for t in get_all_tickers()[:5]:
        print(get_flow_summary(t)["summary"])
