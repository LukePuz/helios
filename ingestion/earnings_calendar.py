"""
Earnings calendar ingestion via yfinance (free, no API key required).

Fetches next earnings dates for all watchlist tickers and caches them in the
earnings_calendar DB table. The scanner reads from DB (fast) rather than
calling yfinance on every scan.

Refresh every weekday at 8 AM ET via Celery beat.
"""
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)


def get_next_earnings(ticker: str) -> date | None:
    """
    Fetch the next upcoming earnings date for a single ticker via yfinance.
    Returns None if unavailable or no upcoming date found.
    """
    return get_next_earnings_detail(ticker)[0]


def get_next_earnings_detail(ticker: str) -> tuple[date | None, bool]:
    """(next date, is_estimate). yfinance gives two dates for an estimated
    window and one for a set date; the window's start is used, flagged as an
    estimate rather than stored as confirmed (A1453-2)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).calendar
        if info is None:
            return None, False
        # yfinance returns a dict with "Earnings Date" key containing a list of timestamps
        earnings_dates = info.get("Earnings Date", [])
        if not earnings_dates:
            return None, False
        # Filter to future dates only, return the nearest
        today = date.today()
        future = [d.date() if hasattr(d, "date") else d for d in earnings_dates if (d.date() if hasattr(d, "date") else d) >= today]
        return (min(future), len(earnings_dates) > 1) if future else (None, False)
    except Exception as e:
        log.debug(f"[{ticker}] get_next_earnings failed: {e}")
        return None, False


def refresh_all_earnings() -> int:
    """
    Refresh earnings dates for all 75 watchlist tickers.
    Writes results to the earnings_calendar DB table.
    Returns count of tickers updated.
    """
    import db
    from knowledge_loader import get_all_tickers

    tickers = get_all_tickers()
    count = 0
    no_date = []
    for ticker in tickers:
        try:
            d, is_estimate = get_next_earnings_detail(ticker)
            if not d:
                no_date.append(ticker)
            if d:
                db.upsert_earnings_date(ticker, d, confirmed=not is_estimate, replace_future=True)
                try:
                    from catalyst_service.storage import upsert_earnings_from_sources
                    from knowledge_loader import get_watchlist_entry

                    entry = get_watchlist_entry(ticker) or {}
                    tier = entry.get("tier")
                    role = entry.get("role")
                    bucket_relevance = "tail" if tier == "Speculative" or role == "Moonshot" else "core"
                    upsert_earnings_from_sources(
                        ticker=ticker,
                        aggregator_dates=[d],
                        sources=[
                            {
                                "name": "yfinance",
                                "url": f"https://finance.yahoo.com/quote/{ticker}",
                                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                "reported_date": d.isoformat(),
                            }
                        ],
                        bucket_relevance=bucket_relevance,
                    )
                except Exception as e:
                    log.warning(f"[{ticker}] catalyst calendar write failed: {e}")
                count += 1
                log.debug(f"[{ticker}] Next earnings: {d}")
        except Exception as e:
            log.warning(f"[{ticker}] refresh failed: {e}")
        time.sleep(0.25)  # yfinance rate limit

    log.info(f"Earnings calendar refreshed: {count}/{len(tickers)} tickers updated"
             + (f"; no upcoming date for {', '.join(no_date)}" if no_date else ""))
    return count
