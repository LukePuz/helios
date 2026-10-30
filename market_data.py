"""
Market data access via Interactive Brokers (ib_insync).
TWS must be running on port 7497 (paper) or 7496 (live).

Central price authority: every quote carries a bar timestamp and staleness
metadata so consumers (drift guards, stop monitors) can refuse to act on
last-RTH prices hours after the close.
"""
import logging
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

from ibkr import get_latest_price, get_historical_bars
from config import SYMBOLS

log = logging.getLogger(__name__)

# A 1-min bar older than this is considered stale for execution decisions
STALE_AFTER_MINUTES = 20


_ET = ZoneInfo("America/New_York")
_RTH_OPEN = dtime(9, 30)
_RTH_CLOSE = dtime(16, 0)
_EARLY_CLOSE = dtime(13, 0)
# NYSE full-day holidays and 13:00 ET early closes, 2026-2027 (published NYSE
# calendar). Extend before 2028; is_market_hours logs a warning past the end.
_NYSE_HOLIDAYS = {date(*d) for d in (
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19),
    (2026, 7, 3), (2026, 9, 7), (2026, 11, 26), (2026, 12, 25),
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26), (2027, 5, 31), (2027, 6, 18),
    (2027, 7, 5), (2027, 9, 6), (2027, 11, 25), (2027, 12, 24),
)}
_NYSE_EARLY_CLOSES = {date(*d) for d in ((2026, 11, 27), (2026, 12, 24), (2027, 11, 26))}
_CALENDAR_LAST_YEAR = 2027


def _session(day: date) -> tuple[dtime, dtime] | None:
    """(open, close) in ET for a trading day, None for weekends/holidays."""
    if day.weekday() >= 5 or day in _NYSE_HOLIDAYS:
        return None
    if day.year > _CALENDAR_LAST_YEAR:
        log.warning(f"NYSE holiday calendar ends {_CALENDAR_LAST_YEAR}; extend market_data._NYSE_HOLIDAYS")
    return (_RTH_OPEN, _EARLY_CLOSE if day in _NYSE_EARLY_CLOSES else _RTH_CLOSE)


def is_market_hours(now_utc: datetime | None = None) -> bool:
    """US regular trading hours, 09:30-16:00 America/New_York, NYSE holidays
    and early closes excluded.

    Promoted from module-private 2026-07-27 so ibkr.place_market_order and
    execution.execute_approved can gate new BUY entries on it directly --
    see place_market_order's docstring for the incident this closes (a
    Sunday-evening auto-executed ADD submitted a DAY market order while the
    market was closed; IBKR queued it for the next session instead of
    filling or rejecting it, and the 30s fill-wait then declared the trade
    FAILED without ever cancelling the now-orphaned, still-live order).

    Before 2026-10-28 this was fixed 13:30-20:00 UTC, which is only right
    during daylight time: from 11-02 it would have called 08:30-09:30 ET
    open and 15:00-16:00 ET closed, and it had no holidays (professor C14)."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(_ET)
    sess = _session(now.date())
    if sess is None:
        return False
    return sess[0] <= now.time() <= sess[1]


# Old name kept as an alias -- this module's own get_priced_quote already
# called the private name before the rename.
_is_market_hours = is_market_hours


def next_market_open(now_utc: datetime | None = None) -> datetime:
    """Next regular-session open (09:30 ET on a trading day) at or after
    `now`, in UTC. Added 2026-07-28 alongside the move to a 5:00 AM ET wake --
    early ingestion/analysis now often finishes a new-entry recommendation
    well before the market opens, and execution.execute_approved's
    market-hours gate defers it rather than firing blind at a stale price.
    tasks.retry_deferred_entry uses this to schedule a single retry right
    at the open, instead of the recommendation just sitting FAILED until
    something else happens to re-trigger analysis. DST- and holiday-aware
    since 2026-10-28 (professor C14)."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(_ET)
    day = now.date()
    for _ in range(15):
        if _session(day) is not None:
            candidate = datetime.combine(day, _RTH_OPEN, tzinfo=_ET)
            if candidate > now:
                return candidate.astimezone(timezone.utc)
        day += timedelta(days=1)
    raise RuntimeError("no trading day in the next 15 days")


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


_SESSION_QUOTE_CACHE: dict[str, tuple[float, dict | None]] = {}
_SESSION_QUOTE_TTL_S = 60   # price and brief label share one fetch (review 5 #2)


def get_session_quote(ticker: str, now_utc: datetime | None = None) -> dict | None:
    """Outside regular hours: the last pre-market/after-hours trade, with its
    time and the last regular-session close. None during regular hours or
    when IBKR has no extended-hours bar. The regular-hours quote above is the
    prior close all night, so a post-earnings morning priced the pre-print
    close (audit A1853-4; Luke D6, 10-29)."""
    import time
    cached = now_utc is None and _SESSION_QUOTE_CACHE.get(ticker)
    if cached and time.monotonic() - cached[0] < _SESSION_QUOTE_TTL_S:
        return cached[1]
    q = _session_quote_uncached(ticker, now_utc or datetime.now(timezone.utc))
    if now_utc is None:
        _SESSION_QUOTE_CACHE[ticker] = (time.monotonic(), q)
    return q


def _session_quote_uncached(ticker: str, now_utc: datetime) -> dict | None:
    if is_market_hours(now_utc):
        return None
    bars = get_historical_bars(ticker, duration="1 D", bar_size="1 min", use_rth=False)
    if not bars:
        return None
    rth = get_priced_quote(ticker)
    if not rth:
        return None
    last = bars[-1]
    try:
        bar_time = datetime.fromisoformat(str(last["time"]))
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=_ET)
    except ValueError:
        return None
    et = bar_time.astimezone(_ET)
    if et.time() >= dtime(9, 30) and et.time() < dtime(16, 0):
        return None   # the last trade is a regular-hours one: nothing newer
    session = "pre-market" if et.time() < dtime(9, 30) else "after-hours"
    return {"price": float(last["close"]), "bar_time": bar_time, "session": session,
            "prior_close": float(rth["price"])}


def session_price_label(ticker: str, price: float | None) -> str:
    """' (pre-market, last trade 07:52 ET; prior close $X, +Y%)' for the
    brief, or '' during regular hours / when unknown."""
    try:
        q = get_session_quote(ticker)
        if not q or not price or abs(q["price"] - price) > 0.01 * price:
            return ""
        chg = (q["price"] - q["prior_close"]) / q["prior_close"] if q["prior_close"] else 0.0
        bar_et = q["bar_time"].astimezone(_ET)
        when = (f"{bar_et:%H:%M}" if bar_et.date() == datetime.now(_ET).date()
                else f"{bar_et:%b %d %H:%M}")
        return (f" ({q['session']}, last trade {when} ET; "
                f"prior close ${q['prior_close']:.2f}, {chg:+.1%})")
    except Exception as e:
        log.debug(f"[{ticker}] session price label unavailable: {e}")
        return ""


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
