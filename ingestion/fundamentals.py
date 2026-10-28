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
        # Promoted to top-level, queryable columns (2026-09-13, swing-trading
        # redesign) -- shares_outstanding lets a peer multiple be converted
        # back into an implied dollar price (agents/trade_ladder.py's
        # peer-implied fair value / DCF). target_mean_price/low/high are
        # Wall Street's consensus range -- kept as a DISPLAYED reference
        # only per 2026-09-13 user decision (Street targets skewed 60% of
        # the watchlist into "Deep Value" when blended as a primary
        # valuation input; see agents/valuation.py), never fed into the
        # fair-value ensemble itself. free_cash_flow/net_cash/beta/
        # earnings_growth/forward_eps feed the DCF and PEG methods there.
        "shares_outstanding": info.get("sharesOutstanding"),
        "target_mean_price":  info.get("targetMeanPrice"),
        "target_low_price":   info.get("targetLowPrice"),
        "target_high_price":  info.get("targetHighPrice"),
        "free_cash_flow":     info.get("freeCashflow"),
        "net_cash": (
            info.get("totalCash") - info.get("totalDebt")
            if info.get("totalCash") is not None and info.get("totalDebt") is not None else None
        ),
        "beta":              info.get("beta"),
        "earnings_growth":   info.get("earningsGrowth"),
        "forward_eps":       info.get("forwardEps"),
        # Residual Income (Ohlson) + Justified Multiple inputs (2026-09-13)
        "book_value_per_share": info.get("bookValue"),
        "return_on_equity":     info.get("returnOnEquity"),
        "net_profit_margin":    info.get("profitMargins"),
        # yfinance's own sector/industry classification (2026-09-13,
        # broadened Comps) -- works for ANY ticker regardless of S&P 500
        # membership (confirmed: RGTI/QBTS/LUNR all have real values here
        # despite being too small for index inclusion), finer-grained than
        # GICS sector, and the join key against sp500_fundamentals for the
        # peer search in get_industry_peer_median below.
        "sector":            info.get("sector"),
        "industry":          info.get("industry"),
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
            db.log_fundamentals_history(ticker, metrics)
            updated += 1
        time.sleep(0.3)  # be polite to Yahoo's unofficial endpoint
    return updated


def refresh_sp500_fundamentals(skip_watchlist: bool = True) -> int:
    """
    Broadened Comps peer universe (2026-09-13) -- refreshes the lighter
    sp500_fundamentals table for the full S&P 500 (minus the watchlist,
    already covered by refresh_all_fundamentals). Weekly cadence, same
    reasoning as ingestion.price_history.refresh_sp500_price_history:
    peer multiples are a slow-moving background signal, not time-
    sensitive, and ~500 sequential yfinance calls at ~0.5-1s each isn't
    something to run nightly on top of the existing 75-ticker refresh.
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
            log.debug(f"Could not load watchlist to dedupe S&P 500 fundamentals refresh: {e}")

    updated = 0
    for ticker in tickers:
        metrics = get_fundamentals(ticker)
        if metrics:
            db.upsert_sp500_fundamentals(ticker, metrics)
            updated += 1
        time.sleep(0.3)
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



# Found 2026-09-13 building the fair-value layer: LUNR's declared sector
# ("Space Data Centers") peer group is [LUNR 4.7x, RKLB 52.3x, ASTS 202.0x
# P/S] -- a real 3-member group per the >=3 threshold below, but spanning
# a 43x range. Applying that median (52.3x, RKLB) to LUNR's own revenue
# produced a $159.85 "fair value" against a $14.35 stock -- not a bug in
# the median math, a real methodology failure: three wildly-different-
# stage businesses sharing a sector label isn't a coherent comp set no
# matter how many of them there are. A max/min ratio cap catches this
# class of case; count alone (the original >=3 check) does not.
_MAX_PEER_DISPERSION_RATIO = 8.0


def get_sector_peer_median(ticker: str, metric: str) -> float | None:
    """
    Median `metric` among watchlist tickers sharing `ticker`'s sector (per
    knowledge/watchlist.md), not the whole watchlist -- unlike
    get_peer_percentile above, which deliberately compares cross-sector
    (a quick "how does this rank overall" signal). This one backs the
    peer-implied fair-value calc (agents/trade_ladder.py), where comparing
    a quantum-computing name's P/S against a semiconductor name's would be
    a meaningless multiple to apply.

    Returns None (fails closed, not "fall back to a number anyway") when
    the sector group is too small (<3 members with data) OR too
    dispersed (max/min ratio > _MAX_PEER_DISPERSION_RATIO) -- either case
    means there isn't a real, coherent comp set to derive a multiple from,
    and compute_fair_value falls back to the Street-consensus anchor
    alone rather than blending in a number this unreliable.
    """
    import db
    from knowledge_loader import load_watchlist

    watchlist = load_watchlist()
    sector = (watchlist.get(ticker.upper()) or {}).get("sector")
    if not sector:
        return None
    rows = db.get_fundamentals_all()

    sector_tickers = {t for t, e in watchlist.items() if e.get("sector") == sector}
    sector_values = sorted(r[metric] for r in rows if r["ticker"] in sector_tickers and r.get(metric) is not None)
    if len(sector_values) < 3:
        return None

    if sector_values[0] > 0 and sector_values[-1] / sector_values[0] > _MAX_PEER_DISPERSION_RATIO:
        log.debug(
            f"[{ticker}] sector '{sector}' peer group too dispersed for a coherent {metric} "
            f"median ({sector_values[0]:.2f}x - {sector_values[-1]:.2f}x, "
            f"ratio {sector_values[-1] / sector_values[0]:.1f}x > {_MAX_PEER_DISPERSION_RATIO}x cap)"
        )
        return None

    n = len(sector_values)
    mid = n // 2
    median = sector_values[mid] if n % 2 else (sector_values[mid - 1] + sector_values[mid]) / 2
    return round(median, 3)


# Minimum peers required from the BROAD (S&P 500 + watchlist) pool before
# trusting it over the narrower watchlist-only version -- higher than
# get_sector_peer_median's own 3, since the whole point of this function
# is a statistically sturdier comparison; a broad pool that still can't
# clear a handful of members isn't meaningfully better than the narrow one.
_INDUSTRY_MIN_PEERS = 5


def get_industry_peer_median(ticker: str, metric: str) -> float | None:
    """
    Broadened Comps peer search (2026-09-13) -- groups by yfinance's own
    `industry` field (finer-grained than GICS sector, and works for ANY
    ticker regardless of S&P 500 membership: confirmed RGTI/QBTS/LUNR all
    carry real industry labels despite being too small for index
    inclusion) across BOTH the watchlist's own fundamentals AND the
    broader sp500_fundamentals pool (~500 names) -- a real, sturdier peer
    set than get_sector_peer_median's own ~75-ticker-only search, though
    still not a fix for a ticker whose business model genuinely has no
    large-cap public comp yet (quantum computing hardware, for instance
    -- see this function's own dispersion guard, which still exists to
    catch exactly that case rather than force a number through).

    Same fail-closed posture as get_sector_peer_median: returns None on
    <_INDUSTRY_MIN_PEERS members or >_MAX_PEER_DISPERSION_RATIO spread,
    never falls back to a less coherent grouping just to produce a number.
    """
    import db

    fund = db.get_fundamentals(ticker)
    industry = (fund or {}).get("industry")
    if not industry:
        return None

    watchlist_rows = db.get_fundamentals_all()
    sp500_rows = db.get_sp500_fundamentals_by_industry(industry)

    peer_values = sorted(
        r[metric] for r in watchlist_rows
        if r["ticker"] != ticker and r.get("industry") == industry and r.get(metric) is not None
    ) + sorted(r[metric] for r in sp500_rows if r.get(metric) is not None)
    peer_values.sort()

    if len(peer_values) < _INDUSTRY_MIN_PEERS:
        return None
    if peer_values[0] > 0 and peer_values[-1] / peer_values[0] > _MAX_PEER_DISPERSION_RATIO:
        log.debug(
            f"[{ticker}] industry '{industry}' broad peer group too dispersed for a coherent "
            f"{metric} median ({peer_values[0]:.2f}x - {peer_values[-1]:.2f}x)"
        )
        return None

    n = len(peer_values)
    mid = n // 2
    median = peer_values[mid] if n % 2 else (peer_values[mid - 1] + peer_values[mid]) / 2
    return round(median, 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from knowledge_loader import get_all_tickers
    for t in get_all_tickers()[:5]:
        m = get_fundamentals(t)
        print(f"{t}: {m}")
