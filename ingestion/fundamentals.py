"""
Fundamentals ingestion — free, via yfinance.

No fundamentals existed anywhere in this codebase before this module: the
scanner's momentum score was explicitly a "price proxy for fundamentals"
and the analyst LLM self-reported valuation multiples with nothing to
ground them. This fetches real revenue growth, margins, FCF, and valuation
multiples for the watchlist so both can be grounded in actual numbers.
"""

import logging
import time

import yfinance as yf

log = logging.getLogger(__name__)


def get_fundamentals(ticker: str) -> dict | None:
    """
    Returns a metrics dict for a single ticker, or None if data is unavailable.
    All fields are best-effort — yfinance's `.info` dict has inconsistent
    coverage across tickers (especially newer/smaller names), so every key
    is read defensively and left None rather than guessed.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        log.warning(f"[{ticker}] yfinance fundamentals fetch failed: {e}")
        return None

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        return None

    revenue_growth = info.get("revenueGrowth")
    gross_margins  = info.get("grossMargins")

    # Free-cash-flow margin: freeCashflow / totalRevenue (both in absolute $)
    fcf_margin = None
    fcf = info.get("freeCashflow")
    revenue = info.get("totalRevenue")
    if fcf is not None and revenue:
        fcf_margin = fcf / revenue

    metrics = {
        "revenue_growth_yoy": revenue_growth,
        "gross_margin":       gross_margins,
        "fcf_margin":         fcf_margin,
        "price_to_sales":     info.get("priceToSalesTrailing12Months"),
        "ev_to_ebitda":       info.get("enterpriseToEbitda"),
        "forward_pe":         info.get("forwardPE"),
        "market_cap":         info.get("marketCap"),
        "raw": {
            k: info.get(k) for k in (
                "trailingPE", "pegRatio", "profitMargins", "operatingMargins",
                "returnOnEquity", "debtToEquity", "targetMeanPrice",
                "recommendationKey", "earningsGrowth",
            )
        },
    }
    return metrics


def refresh_all_fundamentals() -> int:
    """
    Refreshes fundamentals for every watchlist ticker. Returns count updated.
    Meant to run off-hours (nightly) — yfinance's .info call is slow
    (~0.5-1s per ticker) and this fetches ~75 tickers sequentially.
    """
    import db
    from knowledge_loader import get_all_tickers

    tickers = get_all_tickers()
    updated = 0
    for ticker in tickers:
        metrics = get_fundamentals(ticker)
        if metrics:
            db.upsert_fundamentals(ticker, metrics)
            updated += 1
        time.sleep(0.3)  # be polite to Yahoo's unofficial endpoint
    return updated


def get_peer_percentile(ticker: str, metric: str) -> float | None:
    """
    Returns what percentile `ticker`'s `metric` falls at among all watchlist
    tickers with data for that metric (0=lowest, 1=highest). Used to give the
    analyst a peer-relative valuation frame instead of an absolute number.
    """
    import db

    rows = db.get_fundamentals_all()
    values = sorted(
        (r[metric] for r in rows if r.get(metric) is not None)
    )
    if not values:
        return None

    target_row = next((r for r in rows if r["ticker"] == ticker), None)
    if not target_row or target_row.get(metric) is None:
        return None

    target = target_row[metric]
    below = sum(1 for v in values if v < target)
    return round(below / len(values), 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from knowledge_loader import get_all_tickers
    for t in get_all_tickers()[:5]:
        m = get_fundamentals(t)
        print(f"{t}: {m}")
