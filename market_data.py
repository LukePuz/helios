"""
Market data access via Interactive Brokers (ib_insync).
TWS must be running on port 7497 (paper) or 7496 (live).

Central price authority: every quote carries a bar timestamp and staleness
metadata so consumers (drift guards, stop monitors) can refuse to act on
last-RTH prices hours after the close.
"""
import logging
from datetime import datetime, time as dtime, timedelta, timezone

from ibkr import get_latest_price, get_historical_bars
from config import SYMBOLS

log = logging.getLogger(__name__)

# A 1-min bar older than this is considered stale for execution decisions
STALE_AFTER_MINUTES = 20


def is_market_hours(now_utc: datetime | None = None) -> bool:
    """Rough US regular-trading-hours check (Mon-Fri 13:30-20:00 UTC).

    Promoted from module-private 2026-07-27 so ibkr.place_market_order and
    execution.execute_approved can gate new BUY entries on it directly --
    see place_market_order's docstring for the incident this closes (a
    Sunday-evening auto-executed ADD submitted a DAY market order while the
    market was closed; IBKR queued it for the next session instead of
    filling or rejecting it, and the 30s fill-wait then declared the trade
    FAILED without ever cancelling the now-orphaned, still-live order)."""
    now = now_utc or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return dtime(13, 30) <= now.time() <= dtime(20, 0)


# Old name kept as an alias -- this module's own get_priced_quote already
# called the private name before the rename.
_is_market_hours = is_market_hours


def next_market_open(now_utc: datetime | None = None) -> datetime:
    """Next regular-session open (13:30 UTC / 9:30 ET on a weekday) at or
    after `now`. Added 2026-07-28 alongside the move to a 5:00 AM ET wake --
    early ingestion/analysis now often finishes a new-entry recommendation
    well before the market opens, and execution.execute_approved's
    market-hours gate defers it rather than firing blind at a stale price.
    tasks.retry_deferred_entry uses this to schedule a single retry right
    at the open, instead of the recommendation just sitting FAILED until
    something else happens to re-trigger analysis."""
    now = now_utc or datetime.now(timezone.utc)
    candidate = now.replace(hour=13, minute=30, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def get_priced_quote(ticker: str) -> dict | None:
    """
    Returns {price, bar_time, age_minutes, is_stale, market_open} or None.
    price comes from the last 1-min bar (RTH); is_stale is True whenever the
    bar is older than STALE_AFTER_MINUTES — i.e. always true outside RTH.
    """
    try:
        bars = get_historical_bars(ticker, duration="1 D", bar_size="1 min")
        if not bars:
            bars = get_historical_bars(ticker, duration="2 D", bar_size="1 day")
        if not bars:
            return None
        last = bars[-1]
        price = float(last["close"])

        bar_time = None
        age_min: float | None = None
        try:
            raw = str(last["time"])
            # Current ib_insync intraday bars stringify as
            # '2026-07-29 13:04:00-04:00' (real UTC offset, tz-aware) --
            # fromisoformat handles that natively and gives a correct age
            # directly, no UTC-approximation needed.
            try:
                bar_time = datetime.fromisoformat(raw)
                if bar_time.tzinfo is not None:
                    age_min = (datetime.now(timezone.utc) - bar_time.astimezone(timezone.utc)).total_seconds() / 60
                    bar_time = bar_time.replace(tzinfo=None)
            except ValueError:
                bar_time = None
            if age_min is None:
                # Older/alternate formats: '20260706 15:59:00 US/Eastern' or
                # date-only for daily bars -- naive, treat as UTC-approx
                # (error is bounded by tz offset and only makes the check
                # MORE conservative during RTH).
                stripped = raw.split(" US/")[0]
                for fmt in ("%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y-%m-%d"):
                    try:
                        bar_time = datetime.strptime(stripped, fmt)
                        break
                    except ValueError:
                        continue
                if bar_time:
                    age_min = (datetime.now(timezone.utc)
                               - bar_time.replace(tzinfo=timezone.utc)).total_seconds() / 60
        except Exception:
            pass

        market_open = _is_market_hours()
        is_stale = (age_min is None) or (age_min > STALE_AFTER_MINUTES) or not market_open

        return {
            "price":       price,
            "bar_time":    bar_time,
            "age_minutes": round(age_min, 1) if age_min is not None else None,
            "is_stale":    is_stale,
            "market_open": market_open,
        }
    except Exception as e:
        log.warning(f"get_priced_quote({ticker}) failed: {e}")
        return None


def check_gap_confirmation(ticker: str) -> dict:
    """
    Judges whether an opening gap (today's open vs. the prior day's close)
    is holding or already fading. Added 2026-07-29 after TER: a stock that
    gaps up hard on an overnight catalyst can give the whole move back
    within the first half hour once real regular-session volume shows up --
    premarket/opening price discovery runs on a fraction of normal volume
    and isn't a reliable "this is the new price" signal on its own.

    Returns {"status": "no_gap"|"confirmed"|"pending"|"failed",
             "gap_pct": float|None, "retracement_pct": float|None,
             "open_price": float|None, "prior_close": float|None,
             "current_price": float|None}

    - "no_gap": the open didn't move enough from the prior close (below
      config.GAP_CONFIRMATION_THRESHOLD_PCT) for this check to apply --
      the ordinary drift guard governs.
    - "pending": a real gap, but not enough 1-min bars yet since the open
      to tell whether it's holding or fading.
    - "confirmed": a real gap, and price hasn't given back
      config.GAP_FAIL_RETRACEMENT_PCT of it after enough bars to judge.
    - "failed": a real gap that has already given back that fraction or
      more -- judged to be reversing, not holding.
    """
    from config import GAP_CONFIRMATION_THRESHOLD_PCT, GAP_FAIL_RETRACEMENT_PCT

    empty = {"status": "no_gap", "gap_pct": None, "retracement_pct": None,
             "open_price": None, "prior_close": None, "current_price": None}
    try:
        daily = get_historical_bars(ticker, duration="2 D", bar_size="1 day")
        intraday = get_historical_bars(ticker, duration="1 D", bar_size="1 min")
        if not daily or not intraday:
            return empty

        prior_close = float(daily[-1]["close"])
        open_price = float(intraday[0]["open"])
        current_price = float(intraday[-1]["close"])
        bars_since_open = len(intraday)

        if not prior_close:
            return empty

        gap_pct = (open_price - prior_close) / prior_close
        if abs(gap_pct) < GAP_CONFIRMATION_THRESHOLD_PCT:
            return {**empty, "gap_pct": gap_pct, "open_price": open_price,
                    "prior_close": prior_close, "current_price": current_price}

        gap_size = open_price - prior_close
        retracement = ((open_price - current_price) / gap_size) if gap_size else 0.0

        result = {
            "gap_pct": gap_pct, "retracement_pct": retracement,
            "open_price": open_price, "prior_close": prior_close,
            "current_price": current_price,
        }
        if retracement >= GAP_FAIL_RETRACEMENT_PCT:
            result["status"] = "failed"
        elif bars_since_open < 10:
            result["status"] = "pending"
        else:
            result["status"] = "confirmed"
        return result
    except Exception as e:
        log.warning(f"check_gap_confirmation({ticker}) failed: {e}")
        return empty


def get_latest_quotes(symbols: list[str]) -> dict[str, dict]:
    """
    Returns latest known price per symbol with staleness metadata.
    NOTE: bid/ask are NOT available from bar data — do not fabricate them.
    """
    result = {}
    for ticker in symbols:
        q = get_priced_quote(ticker)
        if q:
            result[ticker] = {
                "mid":       q["price"],
                "is_stale":  q["is_stale"],
                "timestamp": datetime.now(timezone.utc),
            }
    return result


def get_bars(
    ticker: str,
    days: int = 20,
    bar_size: str = "1 day",
) -> list[dict]:
    """Returns OHLCV bars. Used by scanner and risk manager."""
    duration = f"{days} D"
    return get_historical_bars(ticker, duration=duration, bar_size=bar_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Fetching quotes via IBKR at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    quotes = get_latest_quotes(SYMBOLS[:5])
    print(f"{'Symbol':<8} {'Price':>10} {'Stale':>6}")
    print("-" * 28)
    for symbol, q in quotes.items():
        print(f"{symbol:<8} ${q['mid']:>9.2f} {str(q['is_stale']):>6}")
