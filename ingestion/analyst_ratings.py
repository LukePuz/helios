"""
Analyst rating-change ingestion — free, via yfinance (Ticker.upgrades_downgrades).

Added 2026-07-26 to close a real gap: no source in this codebase tracked
analyst upgrades/downgrades or price-target changes at all, despite
_SCREENER_PROMPT explicitly calling out "a specific analyst price-target
change with a stated reason" as a relevance criterion it could never
actually see fire. No paid press-release-wire API key is configured
(Benzinga/similar), so this stays in the same "no API key required"
posture as news.py/edgar.py/insider.py/arxiv_watcher.py rather than adding
a new paid dependency.

yfinance's upgrades_downgrades returns full multi-year history per
ticker -- only genuinely new (last N days) rows are ever emitted, and
db.seen_items handles cross-run dedup on top of that.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

_SOURCE = "analyst_ratings"
_LOOKBACK_DAYS = 3  # matches edgar.py's "only recent filings" posture


def _is_seen(item_id: str) -> bool:
    try:
        import db
        return db.is_seen(_SOURCE, item_id)
    except Exception:
        return False


def _fetch_for_ticker(ticker: str) -> list[dict]:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).upgrades_downgrades
    except Exception as e:
        log.debug(f"[{ticker}] upgrades_downgrades fetch failed: {e}")
        return []

    if df is None or df.empty:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)
    results = []
    for grade_date, row in df.iterrows():
        try:
            grade_dt = grade_date.tz_localize("UTC") if grade_date.tzinfo is None else grade_date
        except Exception:
            grade_dt = grade_date
        if grade_dt < cutoff:
            continue  # yfinance returns full history — only recent rows are new information

        firm       = str(row.get("Firm", "")).strip()
        to_grade   = str(row.get("ToGrade", "")).strip()
        from_grade = str(row.get("FromGrade", "")).strip()
        action     = str(row.get("Action", "")).strip()
        pt_action  = str(row.get("priceTargetAction", "")).strip()
        pt_current = row.get("currentPriceTarget")
        pt_prior   = row.get("priorPriceTarget")

        item_id = f"{ticker}_{grade_date.isoformat()}_{firm}"
        if not firm or _is_seen(item_id):
            continue

        grade_change = f"{from_grade} → {to_grade}" if from_grade and from_grade != to_grade else to_grade
        pt_text = ""
        if pt_action and pt_current:
            pt_text = f", price target {pt_action.lower()} to ${pt_current:.0f}"
            if pt_prior:
                pt_text += f" (from ${pt_prior:.0f})"

        title = f"{firm} {action or 'rates'} {ticker}: {grade_change}{pt_text}"

        results.append({
            "source": "analyst_ratings",
            "ticker": ticker,
            "title":  title,
            "url":    "",
            "text":   title,
            "_seen_keys": [(_SOURCE, item_id)],
        })
    return results


def poll_once() -> list[dict]:
    """Fetches new analyst rating changes for every watchlist ticker."""
    from knowledge_loader import get_all_tickers

    all_items = []
    for ticker in get_all_tickers():
        all_items.extend(_fetch_for_ticker(ticker))
        time.sleep(0.15)  # be polite to Yahoo's unofficial endpoint
    return all_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent analyst rating changes for the watchlist...\n")
    items = poll_once()
    for i in items[:10]:
        print(f"[{i['ticker']}] {i['title']}")
