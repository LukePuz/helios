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


def _clean(value: str) -> str | None:
    """A yfinance string cell, or None for an empty or NaN one."""
    return None if value.strip().lower() in ("", "nan", "none") else value.strip()


# Log-only, per poll (audit 2026-11-07 03:08 #5, as news.py's _FEED_STATS):
# a fetch failure was a debug line returning [], and yfinance can answer a
# block with an empty frame, so a dead feed looked like a quiet one. Tickers
# polled, failures (with the first error), empty frames, rows inside the
# lookback window, and items returned.
_STATS: dict = {}


def _reset_stats() -> None:
    _STATS.clear()
    _STATS.update({"tickers": 0, "failed": 0, "empty": 0, "recent": 0, "returned": 0,
                   "first_error": None})


def _log_stats() -> None:
    """One INFO line per poll, and a WARNING when every ticker failed or
    every frame was empty. Never raises."""
    try:
        st = _STATS
        if not st.get("tickers"):
            return
        log.info(f"Analyst ratings this poll (log-only): tickers {st['tickers']} failed {st['failed']} "
                 f"empty {st['empty']} recent {st['recent']} returned {st['returned']}")
        if st["failed"] == st["tickers"]:
            log.warning(f"Analyst ratings feed failed for all {st['tickers']} ticker(s) this poll "
                        f"(first: {st['first_error']})")
        elif st["failed"] + st["empty"] == st["tickers"]:
            log.warning(f"Analyst ratings feed read no rows for any of {st['tickers']} ticker(s) "
                        f"this poll ({st['failed']} failed)")
    except Exception as e:
        log.debug(f"analyst ratings stats failed: {e}")


_reset_stats()


def _fetch_for_ticker(ticker: str) -> list[dict]:
    _STATS["tickers"] = _STATS.get("tickers", 0) + 1
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).upgrades_downgrades
    except Exception as e:
        log.debug(f"[{ticker}] upgrades_downgrades fetch failed: {e}")
        _STATS["failed"] = _STATS.get("failed", 0) + 1
        if _STATS.get("first_error") is None:
            _STATS["first_error"] = f"{ticker}: {type(e).__name__}: {str(e)[:120]}"
        return []

    if df is None or df.empty:
        _STATS["empty"] = _STATS.get("empty", 0) + 1
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
        _STATS["recent"] = _STATS.get("recent", 0) + 1

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
            "event_time": grade_dt.isoformat() if hasattr(grade_dt, "isoformat") else str(grade_dt),
            # Structured fields for the verdict's trigger record (log-only,
            # audit 2026-11-05-0725 #2); the prompt sees only title/text.
            # A NaN cell reads as "nan" here (m544); the title above keeps
            # its existing text, the record maps it to None.
            "rating": {"action": _clean(action),
                       "grade_changed": bool(_clean(from_grade)) and _clean(from_grade) != _clean(to_grade),
                       "pt_action": _clean(pt_action)},
            "_seen_keys": [(_SOURCE, item_id)],
        })
    _STATS["returned"] = _STATS.get("returned", 0) + len(results)
    return results


def poll_once() -> list[dict]:
    """Fetches new analyst rating changes for every watchlist ticker."""
    from knowledge_loader import get_all_tickers, get_etf_tickers

    # ETFs have no analyst ratings; Yahoo answers 404 and yfinance logs an
    # ERROR for each one on every poll (2026-10-28).
    etfs = get_etf_tickers()
    all_items = []
    _reset_stats()
    try:
        for ticker in get_all_tickers():
            if ticker in etfs:
                continue
            all_items.extend(_fetch_for_ticker(ticker))
            time.sleep(0.15)  # be polite to Yahoo's unofficial endpoint
    finally:
        # Logged when a ticker's rows raise too, so every poll has its
        # stats line (SEF 148, 09fe1e0).
        _log_stats()
    return all_items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent analyst rating changes for the watchlist...\n")
    items = poll_once()
    for i in items[:10]:
        print(f"[{i['ticker']}] {i['title']}")
