"""
Price history cache — nightly yfinance batch download of daily bars for the
full watchlist plus SPY/QQQ, stored in the price_history table.

This is what closes two gaps at once:
  - risk_manager.check_correlation was a sector-string count, not a real
    correlation; it needs a return series per ticker to compute one.
  - market_regime's multi-ticker breadth branch (_score_breadth with a real
    sample_tickers argument) was dead code — always called with [] because
    fetching 75 tickers of bars live on every regime check was too slow/costly.
A cached local copy makes both cheap.
"""

import logging
import time

import yfinance as yf

log = logging.getLogger(__name__)

BENCHMARK_TICKERS = ["SPY", "QQQ"]


def _fetch_bars(ticker: str, period: str = "1y") -> list[dict]:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
    except Exception as e:
        log.warning(f"[{ticker}] price history fetch failed: {e}")
        return []
    if hist.empty:
        return []

    bars = []
    for idx, row in hist.iterrows():
        try:
            bars.append({
                "date":   idx.strftime("%Y-%m-%d"),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        except (ValueError, KeyError):
            continue
    return bars


def refresh_all_price_history() -> int:
    """
    Refreshes 1y daily bars for every watchlist ticker plus SPY/QQQ.
    Returns count of tickers successfully updated.
    """
    import db
    from knowledge_loader import get_all_tickers

    tickers = get_all_tickers() + BENCHMARK_TICKERS
    updated = 0
    for ticker in tickers:
        bars = _fetch_bars(ticker)
        if bars:
            n = db.upsert_price_bars(ticker, bars)
            if n:
                updated += 1
        time.sleep(0.2)  # be polite to Yahoo's unofficial endpoint
    return updated


def refresh_sp500_price_history(skip_watchlist: bool = True) -> int:
    """
    Refreshes 1y daily bars for the full S&P 500 universe (added 2026-07-26
    for two new consumers, both of which need real per-ticker history
    outside the AI/tech watchlist):
      - market_regime's GICS-sector breadth (a real "% of Financials above
        its 200DMA" needs actual Financials constituents, not the AI/tech
        watchlist -- using the watchlist there would just repeat the same
        tech-breadth number under every unrelated sector's name)
      - candidate_scanner's new-ticker/watchlist-candidate movement screen

    Weekly cadence (not nightly like refresh_all_price_history): breadth is
    inherently a slow-moving background signal, and candidate discovery is
    exploratory, not time-sensitive -- a week-old bar set is never
    meaningfully wrong for either. skip_watchlist avoids re-fetching ~101
    tickers the nightly job already refreshes.
    """
    import db
    from ingestion.sp500_universe import get_sp500_tickers

    tickers = get_sp500_tickers()
    if skip_watchlist:
        try:
            from knowledge_loader import get_all_tickers
            watchlist = set(get_all_tickers())
            tickers = [t for t in tickers if t not in watchlist]
        except Exception as e:
            log.debug(f"Could not load watchlist to dedupe S&P 500 refresh: {e}")

    updated = 0
    for ticker in tickers:
        bars = _fetch_bars(ticker)
        if bars:
            n = db.upsert_price_bars(ticker, bars)
            if n:
                updated += 1
        time.sleep(0.2)  # be polite to Yahoo's unofficial endpoint
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Refreshed price history: {refresh_all_price_history()} tickers")
