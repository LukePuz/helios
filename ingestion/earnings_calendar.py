"""
Earnings calendar ingestion via yfinance (free, no API key required).

Fetches next earnings dates for all watchlist tickers and caches them in the
earnings_calendar DB table. The scanner reads from DB (fast) rather than
calling yfinance on every scan.

Refresh weekly (Monday 8 AM ET via Celery beat).
"""
import logging
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)


def get_next_earnings(ticker: str) -> date | None:
    """
    Fetch the next upcoming earnings date for a single ticker via yfinance.
    Returns None if unavailable or no upcoming date found.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).calendar
        if info is None:
            return None
        # yfinance returns a dict with "Earnings Date" key containing a list of timestamps
        earnings_dates = info.get("Earnings Date", [])
        if not earnings_dates:
            return None
        # Filter to future dates only, return the nearest
        today = date.today()
        future = [d.date() if hasattr(d, "date") else d for d in earnings_dates if (d.date() if hasattr(d, "date") else d) >= today]
        return min(future) if future else None
    except Exception as e:
        log.debug(f"[{ticker}] get_next_earnings failed: {e}")
        return None


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
    for ticker in tickers:
        try:
            d = get_next_earnings(ticker)
            if d:
                db.upsert_earnings_date(ticker, d, confirmed=True)
                count += 1
                log.debug(f"[{ticker}] Next earnings: {d}")
        except Exception as e:
            log.warning(f"[{ticker}] refresh failed: {e}")
        time.sleep(0.25)  # yfinance rate limit

    log.info(f"Earnings calendar refreshed: {count}/{len(tickers)} tickers updated")
    return count
