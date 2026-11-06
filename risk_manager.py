"""
Risk Manager — bucket-aware position sizing, ATR-based stops, and drawdown protection.

Implements the full risk framework from knowledge/risk_rules.md:
  - Sizing by bucket (Compounder/Accelerator/Moonshot) × conviction × regime multiplier
  - ATR-based stop calculation
  - Concentration and correlation checks
  - Portfolio-level drawdown halt

Maintains backward-compatible interface for any modules still calling
shares_affordable() and check_sell_allowed().
"""
import logging
import threading

from stop_lock import TickerBusy, locked_by_ticker, ticker_lock
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

try:
    from config import (
        DRAWDOWN_HALT_PCT,
        BUCKET_SIZE_TABLE,
        REGIME_MULTIPLIER,
        ATR_STOP_MULTIPLIER,
        MOONSHOT_MAX_LOSS_PCT,
        RISK_PCT_PER_TRADE,
        TRAILING_STOP_ACTIVATE,
        TRAILING_STOP_DISTANCE,
        MAX_POSITION_PCT_COMPOUNDER,
        MAX_POSITION_PCT_ACCELERATOR,
        MAX_POSITION_PCT_MOONSHOT,
        MAX_SINGLE_THEME_PCT,
        MAX_AI_INFRA_PCT,
        MAX_MOONSHOT_TOTAL_PCT,
        MAX_TRADE_PCT_OF_CASH_RESERVE,
        SWAP_MIN_CONVICTION_GAP,
        SWAP_MAX_CANDIDATE_CONVICTION,
        get_active_limits,
        TRADING_ENV,
        PROFIT_TARGET_PCT_BY_STYLE,
        PARTIAL_SELL_PCT_BY_STYLE,
        STALE_WINDOW_DAYS_BY_STYLE,
        STALE_MOVE_THRESHOLD_PCT,
        SETUP_GATES_ENABLED,
        CATEGORY_RISK_PCT_PER_TRADE,
        CATEGORY_SOFT_CEILING_PCT,
        PORTFOLIO_HEAT_CAP_PCT,
        ALLOW_ONE_SHARE_ABOVE_ALLOCATION,
        ONE_SHARE_MAX_PCT_OF_CASH,
    )
except ImportError:
    ALLOW_ONE_SHARE_ABOVE_ALLOCATION = False
    ONE_SHARE_MAX_PCT_OF_CASH = 0.20
    TRADING_ENV                   = "paper"
    DRAWDOWN_HALT_PCT             = 0.10
    BUCKET_SIZE_TABLE             = {
        "Compounder":  {9: (0.08, 0.12), 7: (0.05, 0.08), 5: (0.03, 0.05), 3: (0.02, 0.03)},
        "Accelerator": {9: (0.04, 0.07), 7: (0.03, 0.05), 5: (0.02, 0.03), 3: (0.01, 0.02)},
        "Moonshot":    {9: (0.01, 0.02), 7: (0.005, 0.01), 5: (0.0025, 0.005), 3: (None, None)},
    }
    REGIME_MULTIPLIER             = {"Strong Risk-On": 1.2, "Risk-On": 1.0, "Neutral": 0.75,
                                     "Risk-Off": 0.5, "Strong Risk-Off": 0.25}
    ATR_STOP_MULTIPLIER           = {"Compounder": (2.0, 3.0), "Accelerator": (1.5, 2.0), "Moonshot": (None, None)}
    MOONSHOT_MAX_LOSS_PCT         = 0.20
    RISK_PCT_PER_TRADE            = 0.02
    TRAILING_STOP_ACTIVATE        = {"Compounder": 0.30, "Accelerator": 0.20}
    TRAILING_STOP_DISTANCE        = {"Compounder": (0.15, 0.20), "Accelerator": (0.10, 0.15)}
    MAX_POSITION_PCT_COMPOUNDER   = 0.12
    MAX_POSITION_PCT_ACCELERATOR  = 0.07
    MAX_POSITION_PCT_MOONSHOT     = 0.02
    MAX_SINGLE_THEME_PCT          = 0.35
    MAX_AI_INFRA_PCT              = 0.60
    MAX_MOONSHOT_TOTAL_PCT        = 0.10
    MAX_TRADE_PCT_OF_CASH_RESERVE = 0.20
    SWAP_MIN_CONVICTION_GAP       = 2
    SWAP_MAX_CANDIDATE_CONVICTION = 6
    PROFIT_TARGET_PCT_BY_STYLE    = {
        "Trending/Expanding":  {"Accelerator": (0.20, 0.30), "Moonshot": (0.40, 0.60)},
        "Choppy/Distribution": {"Accelerator": (0.10, 0.15), "Moonshot": (0.20, 0.30)},
    }
    PARTIAL_SELL_PCT_BY_STYLE     = {"Trending/Expanding": 0.25, "Choppy/Distribution": 0.45}
    STALE_WINDOW_DAYS_BY_STYLE    = {
        "Trending/Expanding":  {"Accelerator": 28, "Moonshot": 21},
        "Choppy/Distribution": {"Accelerator": 14, "Moonshot": 10},
    }
    STALE_MOVE_THRESHOLD_PCT      = 0.05
    SETUP_GATES_ENABLED           = False
    CATEGORY_RISK_PCT_PER_TRADE   = {"Breakout": 0.015, "Catalyst": 0.010, "Continuation": 0.010}
    CATEGORY_SOFT_CEILING_PCT     = {"Breakout": (0.06, 0.08), "Catalyst": (0.04, 0.06), "Continuation": (0.03, 0.05)}
    PORTFOLIO_HEAT_CAP_PCT        = 0.07

    def get_active_limits(balance: float) -> dict:
        return {"tier": "standard", "max_concurrent_positions": None,
                "min_conviction_to_queue": 5,
                "max_position_pct_moonshot": MAX_POSITION_PCT_MOONSHOT,
                "max_position_pct_compounder": MAX_POSITION_PCT_COMPOUNDER,
                "max_position_pct_accelerator": MAX_POSITION_PCT_ACCELERATOR,
                "single_share_cap_pct": None}


# ---------------------------------------------------------------------------
# Helpers: Alpaca bar fetch for ATR
# ---------------------------------------------------------------------------

# Why the last get_atr() on this thread had no ATR: {"ticker", "n_bars",
# "error"}. Read only by the log-only stop_atr_fallback row.
_ATR_DIAG = threading.local()


def _fetch_bars_for_atr(ticker: str, days: int = 20) -> list[dict]:
    try:
        from ibkr import get_historical_bars
        bars = get_historical_bars(ticker, duration=f"{days + 5} D", bar_size="1 day")
        return [{"h": b["high"], "l": b["low"], "c": b["close"]} for b in bars]
    except Exception as e:
        log.debug(f"ATR bar fetch failed for {ticker}: {e}")
        try:
            _ATR_DIAG.error = type(e).__name__
        except Exception:
            pass
        return []


def _cached_bars_for_atr(ticker: str, days: int = 20) -> list[dict]:
    """_fetch_bars_for_atr's bars from the price_history cache: no broker
    call (simulations)."""
    try:
        import db
        bars = db.get_price_history(ticker, days=days + 5)
        return [{"h": b["high"], "l": b["low"], "c": b["close"]} for b in bars]
    except Exception as e:
        log.debug(f"Cached ATR bar read failed for {ticker}: {e}")
        return []


def _compute_atr(bars: list[dict]) -> float | None:
    """True Range ATR over available bars."""
    if len(bars) < 2:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, prev_c = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return sum(trs) / len(trs)


_ATR_FALLBACK_MARKER = "helios:stop_atr_fallback:{env}:{day}:{caller}:{ticker}"
_ATR_FALLBACK_TICKERS = "helios:stop_atr_fallback_tickers:{env}:{day}"
_ATR_FALLBACK_TICKERS_TTL_S = 3 * 86400
_ATR_FALLBACK_REDIS: dict = {}
# At most one fallback logger thread at a time; a fallback while one runs
# is dropped (log.debug), so these threads can never pile up.
_ATR_FALLBACK_SEM = threading.BoundedSemaphore(1)
_ATR_FALLBACK_CALLERS = ("size_position", "post_fill", "manual", "other")
# Fallbacks that arrive while the logger thread runs wait here, and that
# thread logs them in turn (professor m308: they used to be dropped). Past
# _ATR_FALLBACK_PENDING_MAX they are dropped and counted; the next written
# row carries the count (dropped_before).
_ATR_FALLBACK_PENDING: list = []
_ATR_FALLBACK_PENDING_MAX = 50
_ATR_FALLBACK_PENDING_LOCK = threading.Lock()
_ATR_FALLBACK_DROPPED = {"n": 0}


def _atr_fallback_redis():
    """The fallback logger's own Redis client, with short socket timeouts so
    a slow Redis can't hold its thread for long. Cached per process."""
    client = _ATR_FALLBACK_REDIS.get("client")
    if client is None:
        import redis
        client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                      socket_timeout=0.5, socket_connect_timeout=0.5)
        _ATR_FALLBACK_REDIS["client"] = client
    return client


def _spawn(target, name: str, args: tuple = ()) -> None:
    """Run target on a daemon thread. Tests replace this to run it inline."""
    threading.Thread(target=target, args=args, name=name, daemon=True).start()


def _atr_fallback_reason(ticker: str, atr) -> tuple[str, dict]:
    """Why get_atr() gave no ATR, from the thread's _ATR_DIAG: exception,
    no_bars, too_few_bars, or other (a zero ATR, or no diagnostics for this
    ticker, e.g. get_atr replaced in a test)."""
    diag = getattr(_ATR_DIAG, "last", None) or {}
    if diag.get("ticker") != ticker:
        return "other", {"n_bars": None, "error": None}
    n_bars, error = diag.get("n_bars"), diag.get("error")
    if error:
        reason = "exception"
    elif n_bars == 0:
        reason = "no_bars"
    elif n_bars is not None and n_bars < 2:
        reason = "too_few_bars"
    else:
        reason = "other"
    return reason, {"n_bars": n_bars, "error": error}


def _log_atr_fallback(ticker: str, entry_price, bucket, stop, fallback_pct, atr,
                      reason: str, diag: dict, caller: str = "other", queued: bool = False) -> None:
    """Log-only, run on its own thread for calculate_stop: one activity_feed
    row (stage stop_atr_fallback) per ticker per caller per ET day when the
    flat-percent fallback stop was used. The row goes through
    db.log_activity_direct (its own connection, never the shared pool). Dedup is a Redis SET NX marker that expires an
    hour after ET midnight; if that call fails the row is written anyway and
    no other Redis call is made. After a written row the ticker is added to
    the day's Redis set for the daily digest; after a failed write the marker
    is deleted so a later fallback that day retries. No DB read. Never
    raises, and nothing here touches the stop."""
    try:
        import sys
        from zoneinfo import ZoneInfo
        et_now = datetime.now(ZoneInfo("America/New_York"))
        day = et_now.date().isoformat()
        env = TRADING_ENV
        marker = _ATR_FALLBACK_MARKER.format(env=env, day=day, caller=caller, ticker=ticker)
        r, dedup = None, "unavailable"
        try:
            client = _atr_fallback_redis()
            midnight = (et_now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            ttl = max(60, int((midnight - et_now).total_seconds()) + 3600)
            if not client.set(marker, "1", nx=True, ex=ttl):
                return      # already logged for this ticker and caller today
            r, dedup = client, "ok"
        except Exception as e:
            log.debug(f"[{ticker}] ATR fallback dedup unavailable ({e}); logging anyway")
        versions = {"code_sha": None, "env_sha": None}
        analyst = sys.modules.get("agents.analyst")
        if analyst is not None:
            try:
                versions = analyst.shadow_versions()
            except Exception:
                pass
        import db
        with _ATR_FALLBACK_PENDING_LOCK:
            dropped, _ATR_FALLBACK_DROPPED["n"] = _ATR_FALLBACK_DROPPED["n"], 0
        written = db.log_activity_direct(
            "stop_atr_fallback", ticker,
            f"ATR unavailable ({reason}): flat {fallback_pct:.0%} fallback stop ${stop:.2f}",
            "neutral",
            {"ticker": ticker, "caller": caller, "reason": reason, "n_bars": diag.get("n_bars"),
             "error": diag.get("error"), "atr": atr, "stop_basis": "flat_pct_fallback",
             "fallback_pct": fallback_pct, "stop_price": stop, "price": entry_price,
             "bucket": bucket, "env": env, "et_date": day, "dedup": dedup,
             "queued_behind_logger": bool(queued), "dropped_before": dropped,
             "code_sha": versions.get("code_sha"), "env_sha": versions.get("env_sha")})
        if not written and dropped:
            with _ATR_FALLBACK_PENDING_LOCK:
                _ATR_FALLBACK_DROPPED["n"] += dropped
        if r is None:
            return
        try:
            if written:
                tkey = _ATR_FALLBACK_TICKERS.format(env=env, day=day)
                r.pipeline(transaction=False).sadd(tkey, ticker).expire(
                    tkey, _ATR_FALLBACK_TICKERS_TTL_S).execute()
            else:
                r.delete(marker)
        except Exception as e:
            log.debug(f"[{ticker}] ATR fallback marker/set update failed: {e}")
    except Exception as e:
        log.debug(f"[{ticker}] stop_atr_fallback log failed: {e}")


def _run_atr_fallback_log(*args) -> None:
    """Thread body: the logger, then the fallbacks queued while it ran, then
    release the slot taken by _queue_atr_fallback_log. A fallback queued
    just after the last check is picked up by retaking the slot; if another
    logger already holds it, that one drains it."""
    try:
        _log_atr_fallback(*args)
    finally:
        _drain_atr_fallback_and_release()


def _drain_atr_fallback_and_release() -> None:
    """Thread body for a holder of the logger slot: log the queued
    fallbacks, release the slot, and retake it if more were queued in the
    meantime (if another thread took it, that one drains them)."""
    while True:
        try:
            _drain_atr_fallback_pending()
        finally:
            _ATR_FALLBACK_SEM.release()
        with _ATR_FALLBACK_PENDING_LOCK:
            if not _ATR_FALLBACK_PENDING:
                return
        if not _ATR_FALLBACK_SEM.acquire(blocking=False):
            return


def _drain_atr_fallback_pending() -> None:
    """Log each queued fallback in turn, on the logger thread. Never raises."""
    while True:
        with _ATR_FALLBACK_PENDING_LOCK:
            if not _ATR_FALLBACK_PENDING:
                return
            args = _ATR_FALLBACK_PENDING.pop(0)
        try:
            _log_atr_fallback(*args, queued=True)
        except Exception as e:
            log.debug(f"stop_atr_fallback queued log failed: {e}")


def _queue_atr_fallback_log(ticker: str, entry_price, bucket, stop, fallback_pct, atr,
                            caller: str = "other") -> None:
    """calculate_stop's side: read the thread-local diagnostics, then hand
    every Redis and DB call to a daemon thread. Only one logger thread runs
    at a time; a fallback while one runs is queued for that thread (up to
    _ATR_FALLBACK_PENDING_MAX, then dropped and counted). Never raises."""
    try:
        if not _ATR_FALLBACK_SEM.acquire(blocking=False):
            caller = caller if caller in _ATR_FALLBACK_CALLERS else "other"
            reason, diag = _atr_fallback_reason(ticker, atr)
            with _ATR_FALLBACK_PENDING_LOCK:
                queued = len(_ATR_FALLBACK_PENDING) < _ATR_FALLBACK_PENDING_MAX
                if queued:
                    _ATR_FALLBACK_PENDING.append(
                        (ticker, entry_price, bucket, stop, fallback_pct, atr, reason, diag, caller))
                else:
                    _ATR_FALLBACK_DROPPED["n"] += 1
            if not queued:
                log.debug(f"[{ticker}] stop_atr_fallback log dropped: the logger queue is full")
            # The running logger may have finished its last queue check
            # between our failed acquire and the append: if the slot is free
            # now, drain here, or the item waits for the next fallback.
            if _ATR_FALLBACK_SEM.acquire(blocking=False):
                try:
                    _spawn(_drain_atr_fallback_and_release, f"atr-fallback-log-{ticker}")
                except Exception as e:
                    log.debug(f"[{ticker}] stop_atr_fallback drain not started: {e}")
                    _ATR_FALLBACK_SEM.release()
            return
    except Exception as e:
        log.debug(f"[{ticker}] stop_atr_fallback log not started: {e}")
        return
    try:
        caller = caller if caller in _ATR_FALLBACK_CALLERS else "other"
        reason, diag = _atr_fallback_reason(ticker, atr)
        _spawn(_run_atr_fallback_log, f"atr-fallback-log-{ticker}",
               (ticker, entry_price, bucket, stop, fallback_pct, atr, reason, diag, caller))
    except Exception as e:
        log.debug(f"[{ticker}] stop_atr_fallback log not started: {e}")
        try:
            _ATR_FALLBACK_SEM.release()
        except Exception:
            pass


def atr_fallback_tickers_today(now=None) -> list[str] | None:
    """Sorted tickers that had a logged ATR fallback stop on today's ET date,
    from the Redis set _log_atr_fallback keeps; None if Redis can't be read.
    For the daily digest only."""
    try:
        from zoneinfo import ZoneInfo
        now = now or datetime.now(timezone.utc)
        day = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        raw = _atr_fallback_redis().smembers(_ATR_FALLBACK_TICKERS.format(env=TRADING_ENV, day=day))
        return sorted(k.decode() if isinstance(k, bytes) else str(k) for k in raw)
    except Exception as e:
        log.debug(f"atr_fallback_tickers_today unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Conviction band lookup
# ---------------------------------------------------------------------------

def _price_correlation(bars_a: list[dict], bars_b: list[dict]) -> float | None:
    """
    Pearson correlation of daily returns between two {date, close} bar series,
    aligned by date (only dates present in both series are used).
    Returns None if fewer than 10 overlapping return observations exist.
    """
    a_by_date = {b["date"]: b["close"] for b in bars_a}
    b_by_date = {b["date"]: b["close"] for b in bars_b}
    common_dates = sorted(set(a_by_date) & set(b_by_date))
    if len(common_dates) < 15:  # need >= 15 prices for >= 14 daily returns
        return None

    a_prices = [a_by_date[d] for d in common_dates]
    b_prices = [b_by_date[d] for d in common_dates]

    a_returns = [(a_prices[i] - a_prices[i - 1]) / a_prices[i - 1]
                 for i in range(1, len(a_prices)) if a_prices[i - 1]]
    b_returns = [(b_prices[i] - b_prices[i - 1]) / b_prices[i - 1]
                 for i in range(1, len(b_prices)) if b_prices[i - 1]]
    if len(a_returns) < 10 or len(a_returns) != len(b_returns):
        return None

    try:
        import numpy as np
        corr = np.corrcoef(a_returns, b_returns)[0, 1]
        return float(corr) if corr == corr else None  # NaN check
    except Exception as e:
        log.debug(f"Correlation computation failed: {e}")
        return None


def _conviction_band(conviction: int) -> int:
    """Maps conviction (1-10) to the nearest lower band key (9,7,5,3)."""
    if conviction >= 9:
        return 9
    elif conviction >= 7:
        return 7
    elif conviction >= 5:
        return 5
    else:
        return 3


def _read_resting_stop_floor(ticker: str, new_level: float, what: str,
                             floor_only: bool = False) -> tuple[bool, float | None]:
    """Body of RiskManager._resting_stop_floor (see there), module level so
    execution.py's ADD path can use it without a RiskManager.

    floor_only=True is for the paths that must replace the stop anyway (ADD,
    breakeven, profit-target partial): the caller raises its level to the
    highest resting stop instead of skipping. Then ok is False only when the
    read failed or a price was unreadable, and the alert says the change went
    ahead without the floor. A lower level is not alerted: with a native
    TRAIL the record is expected to lag the broker's trigger."""
    import tasks

    def alert(kind: str, body: str) -> None:
        # Throttled per ticker and kind, so a ticker skipped run after
        # run is never skipped in silence, nor alerted every run.
        try:
            if tasks._reconcile_alert_due(f"stop_no_loosen_{kind}_{ticker}"):
                import alerts
                head = "STOP FLOOR UNREAD" if floor_only else "STOP CHANGE SKIPPED"
                alerts._send(f"⚠️ {head} — <b>{ticker}</b>\n{body}")
        except Exception as e:
            log.warning(f"[{ticker}] no-loosen alert failed: {e}")

    try:
        resting = tasks._resting_sell_stops(ticker)
    except Exception as e:
        if floor_only:
            at = f" at ${float(new_level):.2f}" if new_level else ""
            log.warning(f"[{ticker}] {what}: could not read the resting stops ({e}); going ahead"
                        f"{at} without the no-loosen floor")
            alert("floor_read", f"{what} went ahead{at}: the broker's resting stops could not "
                                f"be read, so its stop may sit below a resting TRAIL it "
                                f"replaced. Check the stop in TWS.")
            return False, None
        log.warning(f"[{ticker}] {what} skipped: could not read the resting stops ({e}); "
                    f"nothing cancelled")
        alert("read", f"{what} to ${float(new_level):.2f} was skipped: the broker's resting "
                      f"stops could not be read, so nothing was cancelled. It retries next run.")
        return False, None
    prices = [o.get("stop_price") for o in resting]
    if any(p is None for p in prices):
        if floor_only:
            at = f" at ${float(new_level):.2f}" if new_level else ""
            log.warning(f"[{ticker}] {what}: a resting stop has no readable price; going ahead"
                        f"{at} without the no-loosen floor")
            alert("floor_price", f"{what} went ahead{at}: a resting stop has no readable price, "
                                 f"so its stop may sit below a resting TRAIL it replaced. Check "
                                 f"the stop in TWS.")
            return False, None
        log.warning(f"[{ticker}] {what} skipped: a resting stop has no readable price; "
                    f"nothing cancelled")
        alert("price", f"{what} to ${float(new_level):.2f} was skipped: a resting stop has no "
                       f"readable price, so nothing was cancelled. Check the stops in TWS.")
        return False, None
    highest = max((float(p) for p in prices), default=None)
    if floor_only:
        return True, highest
    # Compared on the tick the broker would place it at (rounded up), so a
    # same-cent level isn't read as lower. The placed value is unchanged.
    try:
        import ibkr
        on_tick = float(ibkr.round_sell_stop_up_to_tick(new_level))
    except Exception:
        on_tick = float(new_level)
    if highest is not None and on_tick < highest - 1e-6:
        log.warning(f"[{ticker}] {what} skipped: ${float(new_level):.2f} would replace a "
                    f"resting stop at ${highest:.2f} (the record is lower than the broker); "
                    f"resting stops kept")
        alert("lower", f"{what} to ${float(new_level):.2f} would have replaced a resting stop at "
                       f"${highest:.2f}, so nothing was changed. The recorded stop is lower "
                       f"than the broker's; check the stops in TWS.")
        return False, highest
    return True, highest


def _floor_at_resting_stop(ticker: str, level: float, what: str,
                           market_price: float | None = None) -> tuple[float, float | None]:
    """For a cancel-all-and-replace that must go ahead (ADD, breakeven,
    profit-target partial): read the working SELL stops BEFORE the cancel
    and raise the replacement to the highest of them, so a native TRAIL whose
    trigger sits above the DB record is never loosened.

    Returns (level to place, floor), floor being the highest resting stop
    when it is usable (None otherwise), for the caller's restore paths. A
    failed read returns the level unchanged (logged and alerted by
    _read_resting_stop_floor): protection is never blocked by it. A resting
    price at or above market_price is not used, since a SELL stop there
    fills at once."""
    level = float(level)
    try:
        _, highest = _read_resting_stop_floor(ticker, level, what, floor_only=True)
    except Exception as e:   # never block the replacement on the floor read
        log.warning(f"[{ticker}] {what}: no-loosen floor read crashed ({e}); placing at ${level:.2f}")
        return level, None
    if highest is None:
        return level, None
    mp = _plausible_price(market_price)
    if mp is not None and highest >= mp:
        log.warning(f"[{ticker}] {what}: resting stop ${highest:.2f} is at/above the price "
                    f"${mp:.2f}; not used as the floor")
        return level, None
    return _floored_level(ticker, level, highest, None, what), highest


def _plausible_price(price) -> float | None:
    """A market price usable for the floor check, or None: missing, NaN,
    non-positive and absurd (> 1e9) values count as no price."""
    try:
        mp = float(price) if price is not None else None
    except (TypeError, ValueError):
        return None
    if mp is None or not 0 < mp <= 1e9:   # NaN fails both comparisons
        return None
    return mp


def _floored_level(ticker: str, level: float, floor: float | None,
                   market_price: float | None, what: str) -> float:
    """level raised to floor (a resting stop read before the cancel), unless
    the floor is missing, not higher, or at/above a plausible market price
    (a SELL stop there fills at once)."""
    level = float(level)
    if floor is None or float(floor) <= level + 1e-9:
        return level
    mp = _plausible_price(market_price)
    if mp is not None and float(floor) >= mp:
        log.warning(f"[{ticker}] {what}: resting stop ${float(floor):.2f} is at/above the price "
                    f"${mp:.2f}; placing at ${level:.2f}")
        return level
    log.info(f"[{ticker}] {what}: ${level:.2f} raised to the resting stop ${float(floor):.2f} "
             f"(the record is below the broker's)")
    return float(floor)


# Profit targets this process sold but could not mark taken (the mark's pool
# checkout raised): ticker -> {"opened_at", "new_stop", "held_after",
# "entry_price"}. check_profit_targets skips them, so a DB failure can't lead
# to a second partial sale in this process, and retries the mark each run.
# In memory only: a restart or another worker process doesn't see it.
_PT_TAKEN_UNRECORDED: dict[str, dict] = {}


def _mark_profit_target_taken_guarded(ticker: str, new_stop: float, opened_at,
                                      held_after: float | None = None,
                                      entry_price: float | None = None,
                                      held_before: float | None = None) -> bool:
    """db.mark_profit_target_taken after a partial sale went through. Never
    raises: a failure is logged, alerted and kept in _PT_TAKEN_UNRECORDED
    with the shares held after the sale and the entry price, so an ADD
    before the mark lands re-arms the target instead of being marked taken
    (_profit_target_unrecorded). True when the mark was written."""
    import db
    try:
        db.mark_profit_target_taken(ticker, new_stop)
    except Exception as e:
        first = ticker not in _PT_TAKEN_UNRECORDED
        _PT_TAKEN_UNRECORDED[ticker] = {"opened_at": opened_at, "new_stop": new_stop,
                                        "held_after": held_after, "entry_price": entry_price,
                                        "held_before": held_before}
        log.error(f"[{ticker}] profit target sold but not marked taken ({type(e).__name__}: {e}); "
                  f"held in memory so this process doesn't sell it again, retried each run")
        if first:
            try:
                import alerts
                alerts._send(
                    f"⚠️ PROFIT TARGET NOT RECORDED — {ticker}\n"
                    f"The partial sale went through, but the DB could not mark the target taken "
                    f"({type(e).__name__}). This process won't sell it again and retries the "
                    f"mark each run; a restart before the mark lands could sell it again. "
                    f"Check position_stops.profit_target_taken if this repeats."
                )
            except Exception:
                pass
        return False
    _PT_TAKEN_UNRECORDED.pop(ticker, None)
    return True


def _held_after_sale(qty: float, order_result: dict, sell_qty: float) -> float:
    """Shares held right after a profit-target sale, for the marker."""
    try:
        sold = float(order_result.get("filled_qty") or sell_qty)
    except (TypeError, ValueError):
        sold = float(sell_qty)
    return round(max(float(qty) - sold, 0.0), 4)


def _profit_target_unrecorded(ticker: str, stop_record: dict, pos=None) -> bool:
    """True when this process already sold this position's target but the
    mark is missing: the caller skips the ticker. The mark is retried here.

    The entry is dropped (and the target evaluated as usual) for a different
    position (another opened_at), or when the position changed since the
    sale: more shares held than right after it, or another entry price. An
    ADD in between re-arms the target rather than having the retry mark the
    new shares' target taken."""
    pending = _PT_TAKEN_UNRECORDED.get(ticker)
    if pending is None:
        return False
    changed = pending.get("opened_at") != stop_record.get("opened_at")
    held_after, entry = pending.get("held_after"), pending.get("entry_price")
    held_before = pending.get("held_before")
    try:
        if not changed and held_after is not None and pos is not None:
            held = abs(float(pos.qty))
            # Exactly the pre-sale quantity reads as a position cache that
            # hasn't caught up with the sale (it lags fills), not an ADD:
            # re-selling would be the double sale this marker prevents.
            lagging = held_before is not None and abs(held - float(held_before)) <= 1e-6
            changed = held > float(held_after) + 1e-6 and not lagging
        rec_entry = stop_record.get("entry_price")
        if not changed and entry is not None and rec_entry is not None:
            changed = abs(float(rec_entry) - float(entry)) > 1e-6
    except (TypeError, ValueError):
        changed = True   # can't tell: evaluate the target as usual
    if changed:
        log.warning(f"[{ticker}] position changed since the unrecorded profit-target sale; "
                    f"dropping the in-memory marker, target re-armed")
        _PT_TAKEN_UNRECORDED.pop(ticker, None)
        return False
    if stop_record.get("profit_target_taken"):
        _PT_TAKEN_UNRECORDED.pop(ticker, None)
        return True
    _mark_profit_target_taken_guarded(ticker, pending["new_stop"], pending["opened_at"],
                                      held_after, entry, held_before)
    return True


# Profit-target sales held because a stop the cancel didn't take was still
# working: ticker -> identities of those stops (not PendingCancel), see
# _stop_identity. While any
# of them still works the sale is skipped, so the restored stop isn't
# cancelled and put back every run. In memory: a restart clears it, and the
# next run then holds once more and records them again.
_PT_HELD_BY_WORKING_STOP: dict[str, set] = {}


def _working_non_cancelling_stops(ticker: str) -> list[dict] | None:
    """Working SELL stops for ticker other than ones being cancelled, or
    None if the read fails."""
    import execution
    import tasks
    try:
        stops = tasks._resting_sell_stops(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] could not read the working stops: {e}")
        return None
    return [o for o in stops if o.get("status") not in execution._CANCELLING_STATUSES]


def _stop_identity(order: dict) -> tuple:
    """A stop's identity across clients: its permId, or its orderId when the
    permId is missing or 0. orderIds repeat across clientIds, and a manual
    TWS order can show orderId 0, so the orderId is only the fallback."""
    perm = order.get("perm_id")
    try:
        perm = int(perm) if perm is not None else 0
    except (TypeError, ValueError):
        perm = 0
    if perm:
        return ("perm", perm)
    return ("order", order.get("order_id"))


def _identity_text(ids) -> str:
    return ", ".join(sorted(f"{kind} {val}" for kind, val in ids))


def _pt_skip_reminder(ticker: str, detail: str) -> None:
    """Throttled reminder (about every RECONCILE_ALERT_THROTTLE_HOURS) that a
    profit target is still being skipped. If the throttle can't be read, no
    reminder: a DB outage must not send one every run."""
    try:
        import tasks
        if not tasks._reconcile_alert_due(f"pt_sale_skipped_{ticker}"):
            return
    except Exception as e:
        log.warning(f"[{ticker}] profit-target skip reminder throttle unreadable: {e}")
        return
    try:
        import alerts
        alerts._send(
            f"⏸️ PROFIT TARGET STILL SKIPPED — {ticker}\n"
            f"The partial sale stays held: {detail}. Selling on top of it could short the "
            f"account. The target re-arms on its own once that stop is gone; cancel it in TWS "
            f"if it shouldn't be there.")
    except Exception as e:
        log.warning(f"[{ticker}] profit-target skip reminder failed: {e}")


def _note_pt_hold_working_stops(ticker: str) -> None:
    """Remember the non-cancelling stops a held profit-target sale found."""
    stops = _working_non_cancelling_stops(ticker)
    ids = {_stop_identity(o) for o in (stops or [])}
    if ids:
        _PT_HELD_BY_WORKING_STOP[ticker] = ids
        log.warning(f"[{ticker}] profit-target sale skipped while stop(s) {_identity_text(ids)} "
                    f"still work")


def _pt_hold_still_blocked(ticker: str) -> bool:
    """True while a stop that held this ticker's last profit-target sale
    still works: the caller skips the sale. Cleared once none of them works
    (the sale then runs and re-checks after its own cancel). An unreadable
    read keeps the skip: no sale is the safe side."""
    ids = _PT_HELD_BY_WORKING_STOP.get(ticker)
    if not ids:
        return False
    stops = _working_non_cancelling_stops(ticker)
    if stops is None:
        log.info(f"[{ticker}] Profit-target sale skipped: stops unreadable, an earlier hold "
                 f"found a working stop")
        _pt_skip_reminder(ticker, "the broker's stops could not be read, and an earlier hold "
                                  "found a stop the cancel couldn't take")
        return True
    still = ids & {_stop_identity(o) for o in stops}
    if still:
        log.info(f"[{ticker}] Profit-target sale skipped: stop(s) {_identity_text(still)} "
                 f"that held it still work")
        _pt_skip_reminder(ticker, f"stop(s) {_identity_text(still)} that the cancel couldn't "
                                  f"take still work")
        return True
    _PT_HELD_BY_WORKING_STOP.pop(ticker, None)
    log.info(f"[{ticker}] the stop that held the profit-target sale is gone; target re-armed")
    return False


# ---------------------------------------------------------------------------
# RiskManager class
# ---------------------------------------------------------------------------

class _StopRecordNoBroker(ValueError):
    """position_stops lists the ticker but the broker positions don't."""


# Percentage points below the micro-tier name cap that the swap pre-check
# sizes against (would_size_after_swap).
SWAP_NAME_CAP_SLACK_PP = 0.5


class RiskManager:
    simulation = False   # set per instance by __init__
    # The ATR-fallback alert only (the swap pre-check); nothing else changes.
    quiet_fallback_alert = False

    def __init__(self, client: object | None = None, start_value: float | None = None,
                 simulation: bool = False):
        # simulation: a what-if sizing (swap pre-check, c6 bar shadow). The
        # halt check only reads, ATR comes from the price_history cache and
        # no alert is sent.
        self.simulation = simulation
        if client is None:
            from ibkr import get_client
            client = get_client()
        self._client = client
        account = self._get_account()
        self.start_value: float = start_value or float(account.portfolio_value)
        # Set by is_halted(): True when it blocked because the state or the
        # account read was unusable, not because a halt is latched.
        self.halt_unknown: bool = False
        # Set by check_trailing_stops / check_profit_targets when their reads
        # failed, so an empty result isn't mistaken for "nothing triggered".
        self.monitor_read_failed: bool = False

    def _get_account(self) -> object:
        return self._client.get_account()

    def portfolio_value(self) -> float:
        return float(self._get_account().portfolio_value)

    def buying_power(self) -> float:
        return float(self._get_account().buying_power)

    def cash(self) -> float:
        """Raw cash balance (not leveraged buying power) — the actual reserve
        the per-trade cash cap in size_position() is measured against."""
        return float(self._get_account().cash)

    def _alert_cash_blocked(self, cash_reserve: float) -> None:
        """Cash exhaustion used to fail silently — a buy would just come
        back with 0 shares and crew.py would drop it with a debug log line,
        with nobody told why the system stopped opening new positions.
        Throttled to once per calendar day."""
        if self.simulation:
            return
        try:
            import db
            from datetime import date
            today = date.today().isoformat()
            if db.get_state("cash_blocked_alert_date") == today:
                return
            db.set_state("cash_blocked_alert_date", today)
            import alerts
            alerts._send(
                f"💰 ALL BUYS BLOCKED — cash reserve is ${cash_reserve:,.2f}\n"
                f"The cash-reserve cap ({MAX_TRADE_PCT_OF_CASH_RESERVE:.0%}/trade) can no "
                f"longer size any new position. This is expected once the account "
                f"is close to fully deployed; sells/exits still work normally."
            )
        except Exception as e:
            log.warning(f"_alert_cash_blocked failed: {e}")

    # ------------------------------------------------------------------
    # Drawdown halt
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        """
        Halt if (a) the persisted manual/auto halt flag is set, or
        (b) drawdown from the persisted high-water mark exceeds the limit.
        Baselines live in the system_state table so every process agrees.

        Keys are trading_env-scoped (high_water_mark_paper vs
        high_water_mark_live, etc.) -- found 2026-08-14: a shared
        unscoped "high_water_mark" key held paper's ~$1.09M baseline, and
        reconnecting to the live account (~$2k) would have hit the
        implausible-value guard below on EVERY check, permanently
        (not just on one transient bad read, which is what that guard is
        actually for) -- silently disabling the drawdown halt, the single
        most important safety mechanism in the system, for live trading.
        """
        self.halt_unknown = False
        # Persisted halt flag (set manually via /halt or by the drawdown task)
        try:
            import db
            if db.get_state_strict(f"trading_halted_{TRADING_ENV}", "false") == "true":
                log.warning(f"HALT — trading_halted_{TRADING_ENV} flag is set in system_state")
                return True
            hwm_str = db.get_state_strict(f"high_water_mark_{TRADING_ENV}")
        except Exception as e:
            # Fail closed (audit A1031-1053-2): an unread flag may be a /halt,
            # and the fallback baseline is the account value at construction,
            # so the drawdown would read as ~0. Not latched: this call only.
            log.warning(f"Halt state read failed ({e}) — treating as halted for this check")
            self.halt_unknown = True
            return True

        current  = self.portfolio_value()
        baseline = float(hwm_str) if hwm_str else self.start_value
        if baseline <= 0:
            return False

        # Plausibility guard: a transient broker read of $0 (clientId
        # collision, cold-morning connection race) would otherwise compute
        # as a 100% drawdown and latch trading_halted until a manual
        # /resume. An implausible read blocks this one buy without latching
        # (A1031-1053-2): sizing on it would be wrong anyway, and a
        # persistent one halts via check_portfolio_drawdown.
        if current <= 0 or current < 0.5 * baseline:
            log.warning(f"Implausible portfolio value ${current:,.2f} "
                       f"(baseline ${baseline:,.2f}) — blocking this buy, not latching a halt")
            self.halt_unknown = True
            return True

        drawdown = (baseline - current) / baseline
        if drawdown >= DRAWDOWN_HALT_PCT and self.simulation:
            return True
        if drawdown >= DRAWDOWN_HALT_PCT:
            log.warning(
                f"DRAWDOWN HALT — drawdown {drawdown:.1%} exceeds {DRAWDOWN_HALT_PCT:.0%} "
                f"(high-water=${baseline:,.2f}, current=${current:,.2f})"
            )
            try:
                import db
                not_saved = db.set_state_verified([
                    (f"trading_halted_{TRADING_ENV}", "true"),
                    (f"halt_reason_{TRADING_ENV}", f"Auto: drawdown {drawdown:.1%} from HWM ${baseline:,.2f}")])
                # Marked when the flag couldn't be confirmed saved (M149); this buy is blocked either way.
                db.log_halt_event("auto_halt", "per_buy_check", "drawdown",
                                  drawdown_pct=round(drawdown, 4),
                                  verified=f"trading_halted_{TRADING_ENV}" not in not_saved)
            except Exception:
                pass
            return True

        # Ratchet the high-water mark up as the portfolio grows
        try:
            import db
            if current > baseline and not self.simulation:
                db.set_state(f"high_water_mark_{TRADING_ENV}", str(current))
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Bucket-aware position sizing
    # ------------------------------------------------------------------

    def size_position(
        self,
        ticker: str,
        conviction: int,
        bucket: str,
        regime: str,
        price: float,
        log_fallback: bool = True,
        conviction_floor: int | None = None,
    ) -> dict:
        """
        Calculate full sizing parameters for a new position.
        log_fallback is passed to calculate_stop (False for simulations).
        conviction_floor replaces the tier's conviction minimum (the c6 bar
        shadow only; None, the live value, keeps the tier's).

        Returns a dict with:
            position_size_pct    — target allocation as % of portfolio
            dollar_amount        — target dollar value
            initial_entry_pct   — 50-60% staged entry first tranche
            initial_dollar       — dollar value for first tranche
            shares_initial       — whole shares for first tranche
            stop_loss_price      — computed stop level
            stop_method          — description of stop methodology
            trailing_stop_activate_at_gain_pct
            trailing_stop_distance_pct
            max_loss_if_stopped  — dollars at risk if stop hits from entry
            regime_multiplier    — applied multiplier
            flags                — list of warning strings
        """
        if self.is_halted():
            flag = ("HALT CHECK BLOCKED — halt state or account value unreadable (no halt latched)"
                    if self.halt_unknown else "HALTED — portfolio drawdown limit reached")
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False, "halt_unknown": self.halt_unknown, "flags": [flag]}

        pv     = self.portfolio_value()
        limits = get_active_limits(pv)

        # Account-tier conviction floor — added 2026-08-13 ahead of the live
        # rollout. Below $25k the whole-shares-only constraint below makes
        # low-conviction ideas structurally more likely to waste a scarce
        # position slot on an unexecutable or marginal size; Reduced/Micro
        # require a higher bar before sizing even begins. No effect at
        # Standard (min_conviction_to_queue == today's MIN_CONVICTION_TO_QUEUE).
        floor = limits["min_conviction_to_queue"] if conviction_floor is None else conviction_floor
        if conviction < floor:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Conviction {conviction} below the {limits['tier']}-tier minimum "
                              f"({floor}) — skipping"]}

        # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_GATES_
        # ENABLED and a live setup-gate classification + trade ladder exist
        # for this ticker, risk-first sizing off the gate's structural stop
        # REPLACES the legacy %-of-account allocation-table method below
        # entirely for this call. Returns None (not a sizing dict) when
        # there's no live category/ladder data for this ticker (e.g. a
        # legacy position still carrying an old bucket, or the gates
        # haven't classified it) -- same "insufficient data -> fall back"
        # convention compute_trend_template already uses elsewhere in this
        # codebase, so the pre-cutover method directly below is always the
        # fallback, never bypassed outright.
        if SETUP_GATES_ENABLED:
            new_path = self._size_position_setup_gates(ticker, conviction, bucket, regime, price, pv)
            if new_path is not None:
                return new_path

        band  = _conviction_band(conviction)
        sizes = BUCKET_SIZE_TABLE.get(bucket, BUCKET_SIZE_TABLE["Accelerator"])
        lo, hi = sizes.get(band, (0.01, 0.02))

        if lo is None:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Conviction {conviction} too low for {bucket} — WATCH only"]}

        # BUCKET_SIZE_TABLE's bands are calibrated against each bucket's
        # Standard-tier hard cap — raising the hard cap alone doesn't change
        # anything because these bands, not the cap, are what actually binds
        # at Standard. At Reduced/Micro the tier cap IS the intended scale,
        # so scale the bands themselves by the same ratio, preserving
        # conviction-band shape instead of flattening it. Originally
        # Moonshot-only (2026-08-13 sign-off); extended to Compounder/
        # Accelerator 2026-08-21 after the same silent-drop failure mode
        # (sub-1-share allocations, no alert) turned out to affect them too
        # — see TIER_MAX_POSITION_PCT_COMPOUNDER/ACCELERATOR in config.py.
        standard_cap = {
            "Compounder":  MAX_POSITION_PCT_COMPOUNDER,
            "Accelerator": MAX_POSITION_PCT_ACCELERATOR,
            "Moonshot":    MAX_POSITION_PCT_MOONSHOT,
        }.get(bucket)
        tier_cap = {
            "Compounder":  limits.get("max_position_pct_compounder"),
            "Accelerator": limits.get("max_position_pct_accelerator"),
            "Moonshot":    limits.get("max_position_pct_moonshot"),
        }.get(bucket)
        if limits["tier"] != "standard" and standard_cap and tier_cap:
            tier_scale = tier_cap / standard_cap
            lo, hi = lo * tier_scale, hi * tier_scale

        # Midpoint of range, then apply regime multiplier
        base_pct  = (lo + hi) / 2
        mult      = REGIME_MULTIPLIER.get(regime, 0.75)
        target    = base_pct * mult

        # Enforce hard caps — tier-dependent for all three buckets (see
        # config.TIER_MAX_POSITION_PCT_*): the Standard-tier caps are
        # structurally too small to clear the whole-share floor below at
        # Reduced/Micro balances, so each is raised per-tier instead.
        hard_cap = tier_cap if tier_cap is not None else MAX_POSITION_PCT_ACCELERATOR

        target = min(target, hard_cap)

        # Check if adding would exceed current position cap
        held = self.current_position_value(ticker)
        held_pct = held / pv if pv > 0 else 0

        # Max-concurrent-positions cap — Reduced/Micro tiers only, and only
        # for a genuinely NEW position (held_pct == 0); adding to something
        # already held doesn't consume a new slot. Checked here rather than
        # upstream in crew.py so it applies uniformly to every caller of
        # size_position, not just the auto-execution path.
        if held_pct == 0 and limits["max_concurrent_positions"] is not None:
            try:
                open_count = len(self._client.get_all_positions())
            except Exception as e:
                log.error(f"Could not fetch position count for max-concurrent check: {e}")
                open_count = None
            if open_count is not None and open_count >= limits["max_concurrent_positions"]:
                return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                        "cash_blocked": False,
                        "flags": [f"At the {limits['tier']}-tier max of "
                                  f"{limits['max_concurrent_positions']} concurrent positions "
                                  f"({open_count} held) — skipping new {ticker} position"]}

        remaining = max(0, target - held_pct)
        if remaining < 0.005:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Position already at/near target ({held_pct:.1%} held)"]}

        if not price or price <= 0:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Invalid price ({price!r}) — cannot size position"]}

        # Single-share cap — Reduced/Micro tiers only (see
        # config.TIER_SINGLE_SHARE_CAP_PCT). One share alone can't already
        # represent more than this fraction of the account; guards against
        # one expensive share structurally dominating a small portfolio in
        # a way the % sizing bands above don't otherwise catch, since those
        # operate on target allocation, not on what a single indivisible
        # share costs.
        share_cap_pct = limits["single_share_cap_pct"]
        if share_cap_pct is not None and price > pv * share_cap_pct:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"{ticker} share price ${price:,.2f} exceeds the {limits['tier']}-tier "
                              f"single-share cap of {share_cap_pct:.0%} portfolio "
                              f"(${pv * share_cap_pct:,.2f}) — skipping"]}

        dollar_amount  = pv * remaining
        initial_pct    = 0.55  # 55% initial tranche (midpoint of 50-60%)
        initial_dollar = dollar_amount * initial_pct

        flags = []

        # Cash-reserve cap — this trade's actual spend may not exceed a fixed
        # fraction of *current cash*, not portfolio value. Since the cap
        # shrinks along with cash, remaining cash decays geometrically toward
        # zero instead of hitting it outright after a burst of auto-executes.
        cash_reserve = self.cash()
        cash_cap = max(cash_reserve, 0) * MAX_TRADE_PCT_OF_CASH_RESERVE
        cash_capped = initial_dollar > cash_cap
        if cash_capped:
            initial_dollar = cash_cap
            flags.append(
                f"Capped at {MAX_TRADE_PCT_OF_CASH_RESERVE:.0%} of cash reserve "
                f"(${cash_reserve:,.0f} available)"
            )

        # Whole shares only — IBKR rejects fractional stock orders on most tickers
        shares_initial = int(initial_dollar / price)

        # One share above the allocation (2026-10-27). On a micro account the
        # 55% first tranche is often smaller than one share of a $400-600
        # name, so every pass since Oct 13 sized to 0 and nothing traded.
        # Allow exactly 1 share when it still fits the hard per-position cap
        # and ONE_SHARE_MAX_PCT_OF_CASH of cash (50% from 10-30; the 20%
        # per-trade cap blocked every expensive share); the single-share cap above and the
        # per-trade risk cap below still apply. Decided by the lead session
        # after Luke delegated the call (Jarvis, bus #36).
        # Held value counts toward the cap: on an ADD, checking the new share
        # alone let the position reach target + one share, up to about twice
        # the hard cap (limitations audit 2026-10-28 09:00, finding 5).
        if (shares_initial < 1 and ALLOW_ONE_SHARE_ABOVE_ALLOCATION
                and held + price <= pv * hard_cap
                and price <= max(cash_reserve, 0) * ONE_SHARE_MAX_PCT_OF_CASH):
            flags.append(
                f"Allocation ${initial_dollar:,.2f} is under 1 share; buying 1 share "
                f"(${price:,.2f} = {price / pv:.1%} of portfolio, within the "
                f"{hard_cap:.0%} position cap, {price / cash_reserve:.0%} of cash)"
            )
            shares_initial = 1
            initial_dollar = price

        if shares_initial < 1:
            cash_blocked = cash_reserve <= 0 or cash_capped
            if cash_blocked:
                # This is buys silently vanishing at crew.py's "0 shares"
                # drop with no one told why — surface it, throttled to once
                # a day so a busy morning doesn't spam the same alert on
                # every subsequent buy attempt.
                self._alert_cash_blocked(cash_reserve)
            # cash_blocked distinguishes "genuinely out of room" from every
            # other reason size_position can return 0 shares (halted,
            # conviction too low for the bucket, already at target, bad
            # price) — 2026-08-13, this is what tells the conviction-swap
            # check in crew.py whether it's even worth looking for a
            # position to rotate out of.
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": cash_blocked,
                    "flags": flags + [f"Allocation ${initial_dollar:,.2f} buys less than 1 share "
                              f"of {ticker} at ${price:,.2f} — skipping"]}

        # Stop loss
        stop_price, stop_method = self.calculate_stop(ticker, price, bucket,
                                                      log_fallback=log_fallback,
                                                      log_caller="size_position")

        # Fixed dollar-risk sizing cap (2026-09-11, user request, Minervini
        # SEPA methodology): shares_initial above came purely from the
        # bucket's %-of-account allocation table -- a wide-stop trade and a
        # tight-stop trade in the same conviction band get the SAME share
        # count, with the dollar risk simply whatever falls out of that
        # allocation times the stop distance. This recomputes shares so a
        # stop-out costs at most RISK_PCT_PER_TRADE of equity and takes
        # whichever of the two share counts is SMALLER -- it only ever
        # tightens sizing on a wide-stop trade, never loosens beyond the
        # existing % cap. Applied after the whole-shares/cash-cap checks
        # above (which already guarantee shares_initial >= 1) since the
        # stop distance -- and therefore this cap -- isn't known until
        # calculate_stop() runs.
        if stop_price and stop_price < price:
            risk_dollars = pv * RISK_PCT_PER_TRADE
            risk_based_shares = int(risk_dollars / (price - stop_price))
            if risk_based_shares < shares_initial:
                if risk_based_shares < 1:
                    return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                            "cash_blocked": False,
                            "flags": flags + [
                                f"Stop distance (${price - stop_price:.2f}, {stop_method}) is too wide to "
                                f"buy even 1 share of {ticker} within the {RISK_PCT_PER_TRADE:.1%} "
                                f"per-trade risk cap (${risk_dollars:,.2f}) — skipping"]}
                flags.append(
                    f"Risk-based sizing: ${price - stop_price:.2f} stop distance limits to "
                    f"{risk_based_shares} shares (vs {shares_initial} from the {bucket} allocation "
                    f"cap) to keep max loss at/under {RISK_PCT_PER_TRADE:.1%} of equity"
                )
                shares_initial = risk_based_shares
                initial_dollar = shares_initial * price

        # Trailing stop params
        trail_activate = TRAILING_STOP_ACTIVATE.get(bucket)
        trail_dist_lo, trail_dist_hi = TRAILING_STOP_DISTANCE.get(bucket, (None, None))
        trail_distance = (trail_dist_lo + trail_dist_hi) / 2 if trail_dist_lo else None

        # Max loss at stop
        initial_cost = shares_initial * price
        max_loss = (initial_cost - shares_initial * stop_price) if stop_price else initial_cost * 0.35

        if mult < 1.0:
            flags.append(f"Regime multiplier {mult}x applied ({regime})")
        if target == hard_cap:
            flags.append(f"Size capped at {hard_cap:.0%} bucket maximum")

        return {
            "ticker":              ticker,
            "position_size_pct":   round(remaining, 4),
            "dollar_amount":       round(dollar_amount, 2),
            "initial_entry_pct":   initial_pct,
            "initial_dollar":      round(initial_dollar, 2),
            "shares_initial":      shares_initial,
            # What this order alone buys, as a share of the portfolio. Can
            # exceed position_size_pct when one whole share is above target.
            "order_pct":           round(shares_initial * price / pv, 4) if pv else 0,
            "stop_loss_price":     stop_price,
            "stop_method":         stop_method,
            "trailing_stop_activate_at_gain_pct": trail_activate,
            "trailing_stop_distance_pct": trail_distance,
            "max_loss_if_stopped": round(max_loss, 2),
            "regime_multiplier":   mult,
            "bucket":              bucket,
            "conviction":          conviction,
            "flags":               flags,
            "heat_shadow":         self._heat_shadow(ticker, shares_initial, price, stop_price, pv, mult),
        }

    def _heat_shadow(self, ticker: str, shares: int, price: float, stop_price, pv: float,
                     mult: float) -> dict | None:
        """What the portfolio heat cap would do on this (legacy) path, where
        it isn't enforced: it only runs in the setup-gate path, which is off
        (audit A2114-9). Log-only, never raises."""
        try:
            if not (pv and stop_price and stop_price < price):
                return None
            import db
            positions_map = {p.symbol: {"qty": float(p.qty)} for p in self._client.get_all_positions()}
            current = float(db.get_portfolio_heat_pct(pv, positions_map).get("heat_pct", 0.0))
            cap = PORTFOLIO_HEAT_CAP_PCT * mult
            new = shares * (price - stop_price) / pv
            fits = max(0, int(max(0.0, cap - current) * pv / (price - stop_price)))
            return {"current_heat_pct": round(current, 4), "new_trade_risk_pct": round(new, 4),
                    "cap_pct": round(cap, 4), "would_cut": current + new > cap,
                    "shares_within_cap": min(shares, fits)}
        except Exception as e:
            log.debug(f"[{ticker}] heat shadow failed: {e}")
            return {"error": str(e)[:200]}

    # ------------------------------------------------------------------
    # Swing-trading cutover, Phase 4/5 (2026-10-11): setup-gate sizing path
    # ------------------------------------------------------------------

    def _size_position_setup_gates(
        self, ticker: str, conviction: int, bucket: str, regime: str, price: float, pv: float,
    ) -> dict | None:
        """
        SETUP_GATES_ENABLED sizing path, called from size_position() above.
        Risk-first: get the gate's own structural stop_price FIRST (from
        agents.setup_gates.classify_setup, which reads agents.trade_ladder's
        nightly rungs), THEN derive risk_dollars = equity x
        CATEGORY_RISK_PCT_PER_TRADE[category] x REGIME_MULTIPLIER[regime],
        THEN shares = risk_dollars / (entry - stop) -- the reverse order
        from the legacy path, which sizes off a %-of-account table first and
        only derives dollar risk as a side effect. CATEGORY_SOFT_CEILING_PCT
        is then applied as a pure TRIM (never raises shares), same posture
        as the cash-reserve and portfolio-heat caps below.

        Returns None (never a zero-share dict) when there's no live
        category+ladder match for this ticker -- the caller must fall back
        to the legacy allocation-table method in that case. Every actual
        "can't size this" outcome below (stop too wide, at/above ceiling,
        cash-blocked, heat-blocked) returns a real 0-share sizing dict
        instead, matching the legacy path's own convention so callers
        (agents/crew.py's swap-candidate logic, alerts) see a consistent
        shape either way.
        """
        if bucket not in CATEGORY_RISK_PCT_PER_TRADE:
            return None
        try:
            import db
            from agents.setup_gates import classify_setup
        except Exception as e:
            log.warning(f"[{ticker}] setup-gate sizing unavailable ({e}) — falling back to legacy sizing")
            return None

        ladder = db.get_trade_ladder(ticker)
        if not ladder:
            return None

        try:
            best, _all_gates = classify_setup(ticker)
        except Exception as e:
            log.warning(f"[{ticker}] classify_setup failed ({e}) — falling back to legacy sizing")
            return None

        if not best or best.get("category") != bucket or best.get("stop_price") is None:
            return None

        stop_price = float(best["stop_price"])
        if not price or price <= 0 or stop_price >= price:
            # A structurally invalid stop (at/above entry) can't be risk-sized
            # at all — fall back rather than divide by zero/negative.
            log.warning(f"[{ticker}] setup-gate stop ${stop_price:.2f} is not below "
                        f"price ${price:.2f} — falling back to legacy sizing")
            return None

        risk_pct     = CATEGORY_RISK_PCT_PER_TRADE[bucket]
        mult         = REGIME_MULTIPLIER.get(regime, 0.75)
        risk_dollars = pv * risk_pct * mult
        shares       = int(risk_dollars / (price - stop_price))

        flags: list[str] = []
        if shares < 1:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False, "heat_blocked": False,
                    "flags": [f"Stop distance (${price - stop_price:.2f}, structural/{bucket}) is too "
                              f"wide to buy even 1 share of {ticker} within the {risk_pct:.1%} "
                              f"per-trade risk cap (${risk_dollars:,.2f}) — skipping"]}

        # Soft ceiling — trims, never raises (see config.CATEGORY_SOFT_CEILING_PCT).
        held     = self.current_position_value(ticker)
        _lo, ceiling_hi = CATEGORY_SOFT_CEILING_PCT.get(bucket, (None, None))
        if ceiling_hi is not None:
            ceiling_dollar = max(0.0, pv * ceiling_hi - held)
            ceiling_shares = int(ceiling_dollar / price) if price else 0
            if ceiling_shares < shares:
                flags.append(f"Soft ceiling ({ceiling_hi:.0%} of portfolio) trims {shares} -> {ceiling_shares} shares")
                shares = ceiling_shares
        if shares < 1:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False, "heat_blocked": False,
                    "flags": flags + [f"{ticker} already at/above the {bucket} soft ceiling — skipping"]}

        # Cash-reserve cap — identical convention to the legacy path.
        cash_reserve  = self.cash()
        cash_cap      = max(cash_reserve, 0) * MAX_TRADE_PCT_OF_CASH_RESERVE
        initial_dollar = shares * price
        cash_capped   = initial_dollar > cash_cap
        if cash_capped:
            shares = int(cash_cap / price) if price else 0
            initial_dollar = shares * price
            flags.append(f"Capped at {MAX_TRADE_PCT_OF_CASH_RESERVE:.0%} of cash reserve "
                         f"(${cash_reserve:,.0f} available)")
        cash_blocked = False
        if shares < 1:
            cash_blocked = cash_reserve <= 0 or cash_capped
            if cash_blocked:
                self._alert_cash_blocked(cash_reserve)
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": cash_blocked, "heat_blocked": False,
                    "flags": flags + [f"Allocation ${initial_dollar:,.2f} buys less than 1 share "
                              f"of {ticker} at ${price:,.2f} — skipping"]}

        # Portfolio heat cap (config.PORTFOLIO_HEAT_CAP_PCT) — partial-sizes
        # down to fit the remaining budget, the same "scale down, don't
        # hard-block" posture the cash-reserve cap above already uses, not
        # a hard block. See db.get_portfolio_heat_pct's docstring for the
        # exact dollar-risk-to-CURRENT-stop definition.
        heat_blocked = False
        try:
            positions_map = {p.symbol: {"qty": float(p.qty)} for p in self._client.get_all_positions()}
            heat = db.get_portfolio_heat_pct(pv, positions_map)
            current_heat_pct = heat.get("heat_pct", 0.0)
        except Exception as e:
            log.warning(f"[{ticker}] portfolio heat lookup failed ({e}) — treating current heat as 0")
            current_heat_pct = 0.0

        heat_cap = PORTFOLIO_HEAT_CAP_PCT * mult
        new_trade_risk_pct = (shares * (price - stop_price)) / pv if pv else 0.0
        if current_heat_pct + new_trade_risk_pct > heat_cap:
            remaining_budget_dollars = max(0.0, heat_cap - current_heat_pct) * pv
            heat_capped_shares = int(remaining_budget_dollars / (price - stop_price))
            if heat_capped_shares < shares:
                flags.append(
                    f"Portfolio heat cap ({heat_cap:.1%} of equity, {current_heat_pct:.1%} already used) "
                    f"trims {shares} -> {heat_capped_shares} shares"
                )
                shares = heat_capped_shares
                initial_dollar = shares * price
            if shares < 1:
                heat_blocked = True
                return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                        "cash_blocked": False, "heat_blocked": True,
                        "flags": flags + [f"Portfolio heat already at/above the {heat_cap:.1%} cap "
                                  f"({current_heat_pct:.1%} used) — skipping"]}

        max_loss = shares * (price - stop_price)
        if mult < 1.0:
            flags.append(f"Regime multiplier {mult}x applied ({regime})")

        return {
            "ticker":              ticker,
            "position_size_pct":   round(initial_dollar / pv, 4) if pv else 0,
            "dollar_amount":       round(initial_dollar, 2),
            "initial_entry_pct":   1.0,   # risk-sized directly, no staged tranche
            "initial_dollar":      round(initial_dollar, 2),
            "shares_initial":      shares,
            "order_pct":           round(initial_dollar / pv, 4) if pv else 0,
            "stop_loss_price":     stop_price,
            "stop_method":         f"Setup-gate structural stop ({bucket})",
            "trailing_stop_activate_at_gain_pct": None,  # see check_trailing_stops' ladder-ratchet path
            "trailing_stop_distance_pct": None,
            "max_loss_if_stopped": round(max_loss, 2),
            "regime_multiplier":   mult,
            "bucket":              bucket,
            "conviction":          conviction,
            "cash_blocked":        cash_blocked,
            "heat_blocked":        heat_blocked,
            "target_prices":       best.get("target_prices", []),
            "flags":               flags,
        }

    # ------------------------------------------------------------------
    # ATR-based stop calculation
    # ------------------------------------------------------------------

    def get_atr(self, ticker: str, days: int = 14) -> float | None:
        try:
            _ATR_DIAG.error = None
        except Exception:
            pass
        bars = (_cached_bars_for_atr if self.simulation else _fetch_bars_for_atr)(ticker, days=days)
        try:
            _ATR_DIAG.last = {"ticker": ticker, "n_bars": len(bars),
                              "error": getattr(_ATR_DIAG, "error", None)}
        except Exception:
            pass
        return _compute_atr(bars)

    def calculate_stop(
        self, ticker: str, entry_price: float, bucket: str, log_fallback: bool = True,
        log_caller: str = "other",
    ) -> tuple[float | None, str]:
        """
        Returns (stop_price, method_description). log_fallback=False skips the
        log-only stop_atr_fallback row (simulations and dry runs); log_caller
        (size_position, post_fill, manual, other) is recorded on that row.
        The stop is the same either way.
        """
        # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_GATES_
        # ENABLED and a live gate classification with a real stop_price
        # exists for this ticker under this exact category, use it directly
        # instead of the ATR-multiple calc below. Falls back to the
        # existing ATR logic otherwise (bad classify_setup call, no live
        # match, or no stop_price on the match) -- same "insufficient data
        # -> fall back" convention compute_trend_template already uses
        # elsewhere in this codebase.
        if SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE:
            try:
                from agents.setup_gates import classify_setup
                best, _all_gates = classify_setup(ticker)
            except Exception as e:
                log.warning(f"[{ticker}] setup-gate stop lookup failed ({e}) — falling back to ATR stop")
                best = None
            if best and best.get("category") == bucket and best.get("stop_price") is not None:
                return float(best["stop_price"]), f"Setup-gate structural stop ({bucket})"

        if bucket == "Moonshot":
            stop = round(entry_price * (1 - MOONSHOT_MAX_LOSS_PCT), 2)
            return stop, f"Moonshot hard stop: {MOONSHOT_MAX_LOSS_PCT:.0%} max loss from entry"

        try:
            _ATR_DIAG.last = None
        except Exception:
            pass
        atr = self.get_atr(ticker)
        if atr is None or atr == 0:
            fallback_pct = 0.08 if bucket == "Compounder" else 0.05
            stop = round(entry_price * (1 - fallback_pct), 2)
            if log_fallback:
                try:
                    _queue_atr_fallback_log(ticker, entry_price, bucket, stop, fallback_pct, atr,
                                            log_caller)
                except Exception:
                    pass
            log.warning(
                f"[{ticker}] ATR unavailable — using flat {fallback_pct:.0%} fallback stop ${stop:.2f}. "
                "Stop accuracy may be poor; verify manually."
            )
            if not (self.simulation or self.quiet_fallback_alert):
                try:
                    import alerts as _al
                    _al._send(
                        f"⚠️ ATR Fallback Stop — {ticker}\n"
                        f"Could not compute ATR-based stop. Using flat <b>{fallback_pct:.0%}</b> stop "
                        f"(${stop:.2f}). Verify this is appropriate before approving."
                    )
                except Exception:
                    pass
            return stop, f"Fallback stop ({fallback_pct:.0%} below entry — ATR unavailable)"

        mult_lo, mult_hi = ATR_STOP_MULTIPLIER.get(bucket, (1.5, 2.0))
        mult = (mult_lo + mult_hi) / 2
        stop = round(entry_price - mult * atr, 2)
        return stop, f"{mult:.1f}× ATR ({atr:.2f}) below entry = ${stop:.2f}"

    # ------------------------------------------------------------------
    # Concentration and correlation checks
    # ------------------------------------------------------------------

    def check_concentration(self, ticker: str, proposed_size_pct: float,
                            bucket: str | None = None) -> dict:
        """
        Checks if adding ticker at proposed_size_pct would breach any limit.
        Returns {passed, flags, ai_infrastructure_total, ai_infra_size_cap_pct}.
        FAIL-CLOSED: an errored check counts as failed so a broken data feed
        can never wave a trade through unchecked.

        2026-08-11: this used to be advisory only — a "failed" check just
        appended text to entry_strategy's flags field, which nobody reviews
        in real time since auto-execution has no human-in-the-loop tap
        (2026-07-14 decision). The portfolio sat over its own stated 60% AI
        infra limit for a week-plus (65.3%-65.8% observed) while every new
        ADD to an AI-infra name kept sailing through anyway. ai_infra_total
        exceeding the limit is still only informational for the *other*
        sub-checks below (theme/moonshot/single-position) — but for AI
        infra specifically, the caller now gets ai_infra_size_cap_pct: the
        largest proposed_size_pct for THIS ticker that would bring the
        total back to exactly the limit, so sizing can be capped instead of
        just flagged. Only meaningful when ticker itself is an AI-infra
        name (only then does its own size actually drive the total) — None
        otherwise, since capping an unrelated ticker wouldn't address an
        AI-infra breach caused entirely by other, pre-existing positions.
        """
        try:
            positions = self._client.get_all_positions()
            pv = self.portfolio_value()
        except Exception as e:
            # Every cap is 0 on an error: callers only cut size when a cap
            # is set, so with None here the BUY went ahead at full size
            # despite passed=False (audit 2026-11-02 #3). A 0 cap leaves no
            # room, so crew logs ROUTE_CONCENTRATION_ZERO and skips, and
            # shares_after_concentration returns 0.
            log.error(f"Concentration check ERRORED for {ticker}: {e}; blocking the buy (0 room)")
            return {"passed": False,
                    "flags": [f"RISK CHECK FAILED — could not verify concentration: {e}. "
                              "Buy blocked."],
                    "errored": True,
                    "ai_infrastructure_total": 0,
                    "ai_infra_size_cap_pct": 0.0,
                    "theme_size_cap_pct": 0.0,
                    "moonshot_size_cap_pct": 0.0}

        flags: list[str] = []

        # Current position weights
        pos_values: dict[str, float] = {}
        for p in positions:
            pct = float(p.market_value) / pv if pv else 0
            pos_values[p.symbol] = pct

        # Add proposed
        pos_values[ticker] = pos_values.get(ticker, 0) + proposed_size_pct

        # AI infrastructure tickers (approximate)
        ai_infra_keywords = {"NVDA", "AMD", "ARM", "INTC", "MSFT", "AMZN", "GOOG", "GOOGL",
                             "META", "AVGO", "ANET", "CRDO", "MRVL", "ASML", "SNPS", "KLAC",
                             "LRCX", "AMAT", "MU", "CRWV", "ORCL", "CLS", "PLTR", "SNOW",
                             "DDOG", "VRT", "PWR", "GEV", "CEG"}
        ai_infra_total = sum(v for t, v in pos_values.items() if t in ai_infra_keywords)

        ai_infra_size_cap_pct = None
        if ticker in ai_infra_keywords:
            ai_infra_total_without_this = ai_infra_total - proposed_size_pct
            ai_infra_size_cap_pct = max(0.0, MAX_AI_INFRA_PCT - ai_infra_total_without_this)

        if ai_infra_total > MAX_AI_INFRA_PCT:
            flags.append(f"AI infrastructure exposure {ai_infra_total:.1%} exceeds {MAX_AI_INFRA_PCT:.0%} limit")

        # The account tier's Compounder cap, the one sizing uses: the
        # standard 12% flagged every buy on a smaller account as a breach.
        name_flag_cap = get_active_limits(pv)["max_position_pct_compounder"]
        if pos_values.get(ticker, 0) > name_flag_cap:
            flags.append(f"{ticker} position {pos_values[ticker]:.1%} exceeds Compounder cap {name_flag_cap:.0%}")

        # Theme concentration — positions sharing the proposed ticker's sector.
        # theme_size_cap_pct mirrors ai_infra_size_cap_pct's shape below --
        # found 2026-08-18 that this breach, like moonshot's, was flagged in
        # text but never actually capped (only AI-infra got that fix on
        # 2026-08-11). Only meaningful for a ticker that's actually in the
        # breaching sector; None otherwise.
        theme_size_cap_pct = None
        try:
            from knowledge_loader import load_watchlist
            watchlist = load_watchlist()
            sector = watchlist.get(ticker, {}).get("sector", "")
            if sector:
                theme_total = sum(
                    v for t, v in pos_values.items()
                    if watchlist.get(t, {}).get("sector", "") == sector
                )
                theme_total_without_this = theme_total - proposed_size_pct
                theme_size_cap_pct = max(0.0, MAX_SINGLE_THEME_PCT - theme_total_without_this)
                if theme_total > MAX_SINGLE_THEME_PCT:
                    flags.append(f"'{sector}' theme exposure {theme_total:.1%} exceeds "
                                 f"{MAX_SINGLE_THEME_PCT:.0%} limit")
        except Exception as e:
            log.warning(f"Theme concentration sub-check unavailable: {e}")

        # Total moonshot exposure — RGTI incident 2026-08-18: flagged
        # correctly ("13.7% exceeds 10% limit") but nothing capped it,
        # since only ai_infra_size_cap_pct existed at the time. moonshot_
        # size_cap_pct only computed when bucket=="Moonshot", since that's
        # the only case this ticker's own size drives the total.
        moonshot_size_cap_pct = None
        if bucket == "Moonshot":
            try:
                from knowledge_loader import load_watchlist
                watchlist = load_watchlist()
                moonshot_total = sum(
                    v for t, v in pos_values.items()
                    if watchlist.get(t, {}).get("role", "") == "Moonshot"
                )
                moonshot_total_without_this = moonshot_total - proposed_size_pct
                moonshot_size_cap_pct = max(0.0, MAX_MOONSHOT_TOTAL_PCT - moonshot_total_without_this)
                if moonshot_total > MAX_MOONSHOT_TOTAL_PCT:
                    flags.append(f"Total moonshot exposure {moonshot_total:.1%} exceeds "
                                 f"{MAX_MOONSHOT_TOTAL_PCT:.0%} limit")
            except Exception as e:
                log.warning(f"Moonshot concentration sub-check unavailable: {e}")

        return {
            "passed": len(flags) == 0,
            "flags":  flags,
            "ai_infrastructure_total": round(ai_infra_total, 4),
            "ai_infra_size_cap_pct": (round(ai_infra_size_cap_pct, 4)
                                      if ai_infra_size_cap_pct is not None else None),
            "theme_size_cap_pct": (round(theme_size_cap_pct, 4)
                                    if theme_size_cap_pct is not None else None),
            "moonshot_size_cap_pct": (round(moonshot_size_cap_pct, 4)
                                       if moonshot_size_cap_pct is not None else None),
        }

    def check_correlation(self, ticker: str, corr_threshold: float = 0.7,
                          min_history_days: int = 30) -> dict:
        """
        Real return-series correlation against currently held positions, using
        the price_history cache (ingestion/price_history.py, refreshed
        nightly). Previously this only counted positions sharing the same
        watchlist sector STRING — no actual price relationship was ever
        computed, so e.g. two stocks in different "sectors" that move in
        lockstep were invisible to it.

        Falls back to the old sector-heuristic when cached price history isn't
        available yet (e.g. immediately after a fresh install, before the
        first nightly refresh has run) so this never breaks sizing outright.
        """
        try:
            positions = self._client.get_all_positions()
        except Exception as e:
            log.error(f"Correlation check ERRORED for {ticker}: {e}")
            return {"shared_catalyst_count": 0,
                    "note": f"RISK CHECK FAILED — correlation unverifiable ({e}); "
                            "size reduced 25% as a precaution",
                    "reduce_by": 0.25}

        held_symbols = [p.symbol for p in positions]
        if not held_symbols:
            return {"shared_catalyst_count": 0, "note": "No open positions", "reduce_by": 0}

        try:
            import db
            bulk = db.get_price_history_bulk([ticker] + held_symbols, days=90)
        except Exception as e:
            log.warning(f"Price history unavailable for correlation check: {e}")
            bulk = {}

        if ticker not in bulk or len(bulk[ticker]) < min_history_days:
            return self._check_correlation_sector_fallback(ticker, positions)

        correlated: list[tuple[str, float]] = []
        for sym in held_symbols:
            if sym not in bulk or len(bulk[sym]) < min_history_days:
                continue
            corr = _price_correlation(bulk[ticker], bulk[sym])
            if corr is not None and corr >= corr_threshold:
                correlated.append((sym, round(corr, 2)))

        reduce_by = 0.25 if len(correlated) >= 2 else 0
        note = (
            f"Return-correlated (r>={corr_threshold}) with {len(correlated)} held "
            f"position(s): {correlated}" if correlated
            else f"No high return-correlation (r>={corr_threshold}) with current holdings"
        )
        return {
            "shared_catalyst_count": len(correlated),
            "note": note,
            "reduce_by": reduce_by,
            "correlations": dict(correlated),
        }

    def _check_correlation_sector_fallback(self, ticker: str, positions: list) -> dict:
        """Sector-string fallback used only when price history isn't cached yet."""
        try:
            from knowledge_loader import load_watchlist
            watchlist = load_watchlist()
        except Exception as e:
            return {"shared_catalyst_count": 0,
                    "note": f"Correlation check skipped: {e}", "reduce_by": 0}

        ticker_sector = watchlist.get(ticker, {}).get("sector", "")
        shared = sum(
            1 for p in positions
            if watchlist.get(p.symbol, {}).get("sector", "") == ticker_sector
        )
        reduce_by = 0.25 if shared >= 2 else 0
        return {
            "shared_catalyst_count": shared,
            "note": (f"[price history not yet cached — sector fallback] "
                     f"{shared} existing positions in {ticker_sector}") if ticker_sector
                    else "Sector unknown",
            "reduce_by": reduce_by,
        }

    def _latest_conviction(self, ticker: str) -> int | None:
        """Freshest available conviction reading for a held position — the
        most recent analyst signal (logged on every review, including
        routine HOLDs, see db.get_latest_signal_for's docstring) if one
        exists, otherwise the conviction the position was last entered/
        added at. Returns None only if neither exists (shouldn't happen
        for a genuinely held position)."""
        import db
        signal = db.get_latest_signal_for(ticker)
        if signal is not None:
            return round(signal["confidence"] * 10)
        stop_rec = db.get_position_stop(ticker)
        if stop_rec and stop_rec.get("conviction") is not None:
            return int(stop_rec["conviction"])
        return None

    def find_swap_candidate(self, new_ticker: str, new_conviction: int) -> dict | None:
        """
        Added 2026-08-13, ahead of a live-money switch: when cash is too
        tight to size a high-conviction new idea, the old behavior was to
        just cap or block it outright, even if the portfolio was sitting on
        a position nobody's excited about anymore. This looks for an
        existing Accelerator/Moonshot holding that has independently fallen
        weak enough to be worth exiting anyway, so the new idea can be
        funded by rotation instead of by force.

        Deliberately opportunistic, never forced — by explicit design
        decision (2026-08-13), not just an implementation detail:
          - Compounder-bucket holdings (the long-term core) are never
            eligible, regardless of conviction gap.
          - A candidate needs BOTH new_conviction - candidate_conviction >=
            SWAP_MIN_CONVICTION_GAP AND the candidate's own fresh conviction
            <= SWAP_MAX_CANDIDATE_CONVICTION. The second condition is the
            one that actually matters: a held position at 7+/10 is never
            touched by this no matter how much higher the new idea scores,
            because it hasn't independently earned an exit — this isn't a
            relative ranking that always finds a "weakest" position to
            sacrifice, it's a check for whether anything already looks like
            it deserves to go.
          - If multiple positions qualify, prefers realizing a LOSS (or the
            smallest gain) over a bigger gain — this is a real, deliberate
            tax consideration, not an afterthought: in a taxable account,
            selling a position that's up triggers a capital gain (almost
            certainly short-term given how this system holds positions,
            taxed at ordinary income rates, not the lower long-term rate).
            Preferring losses here both minimizes that drag and tends to
            align with "this one probably isn't working out anyway."
          - Returns None if nothing qualifies — callers must fall through
            to the existing cash-cap/block behavior unchanged, never invent
            a sale just because cash is tight.

        Returns {"ticker", "conviction", "qty", "market_value",
        "unrealized_pl"} for the single best candidate, or None.
        """
        import db
        try:
            positions = self._client.get_all_positions()
        except Exception as e:
            log.error(f"find_swap_candidate: could not fetch positions: {e}")
            return None

        candidates = []
        for p in positions:
            ticker = p.symbol
            if ticker == new_ticker:
                continue
            stop_rec = db.get_position_stop(ticker)
            bucket = (stop_rec or {}).get("thesis_bucket")
            if bucket not in ("Accelerator", "Moonshot"):
                continue
            conviction = self._latest_conviction(ticker)
            if conviction is None:
                continue
            if new_conviction - conviction < SWAP_MIN_CONVICTION_GAP:
                continue
            if conviction > SWAP_MAX_CANDIDATE_CONVICTION:
                continue
            try:
                candidates.append({
                    "ticker":         ticker,
                    "conviction":     conviction,
                    "qty":            abs(float(p.qty)),
                    "market_value":   float(p.market_value),
                    "unrealized_pl":  float(p.unrealized_pl),
                })
            except (TypeError, ValueError) as e:
                log.warning(f"[{ticker}] find_swap_candidate: skipping, bad position data: {e}")

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c["unrealized_pl"], c["conviction"]))
        return candidates[0]

    def _held_value_strict(self, ticker: str) -> float:
        """Market value held in ticker (0.0 when not held). Raises when the
        positions can't be read or the value is NaN or not positive, unlike
        current_position_value, which reads any error as 0. Also raises when
        the broker list lacks the ticker but its stop record says it's held
        (an empty portfolio right after a reconnect), or that read fails."""
        import math
        for p in self._client.get_all_positions():
            if p.symbol == ticker:
                mv = float(p.market_value)
                if not math.isfinite(mv) or mv <= 0:
                    raise ValueError(f"{ticker} market value unusable")
                return mv
        import db
        if db.get_position_stop(ticker, strict=True) is not None:
            raise _StopRecordNoBroker(f"{ticker} has a stop record but no broker position")
        return 0.0

    def _alert_stop_record_no_broker(self, ticker: str) -> None:
        """One LOW Telegram per ticker per ET day when the name cap blocked a
        buy because position_stops lists the ticker and the broker doesn't.
        The day is stored after a successful send. None from simulations.
        Never raises."""
        if self.simulation or self.quiet_fallback_alert:
            return
        try:
            from zoneinfo import ZoneInfo
            import alert_policy
            import alerts
            import db
            key = f"micro_cap_stop_record_alert_{ticker}"
            today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
            if db.get_state(key) == today:
                return
            if alerts._send(f"{ticker}: buy blocked by the name cap: a stop record exists but "
                            f"the broker shows no position (stale record or a cache gap)",
                            severity=alert_policy.LOW) is not None:
                db.set_state(key, today)
        except Exception as e:
            log.warning(f"[{ticker}] name-cap stop-record alert failed: {e}")

    def micro_name_cap(self, ticker: str, shares: int, price: float,
                       slack_pp: float = 0.0) -> dict | None:
        """Micro-tier per-name cap on a new buy: the most shares that keep
        (held value + shares x price) / equity at or under
        MICRO_TIER_NAME_CAP_PCT (less slack_pp percentage points: the swap
        pre-check's margin). None when it doesn't apply (flag off, not the
        micro tier). Never sells: a name already over the cap gets 0 new
        shares, the holding is left alone. Fails closed: an unreadable
        account or position, or a NaN/non-positive equity, price or market
        value gives 0 (outcome "errored"); so a held position whose market
        value reads 0 (no market data) blocks an ADD as errored.
        held_source: broker (the broker list, checked against position_stops
        when it lacks the ticker), stop_record_no_broker_position (errored,
        alerted once a day per ticker), error (the held read failed), or None
        (not reached). The record holds no account dollar values (equity,
        held value): percentages, the share price and share counts only."""
        import math
        from config import MICRO_TIER_NAME_CAP_ENABLED, MICRO_TIER_NAME_CAP_PCT
        if not MICRO_TIER_NAME_CAP_ENABLED:
            return None
        shares = max(int(shares or 0), 0)
        rec = {"cap_pct": MICRO_TIER_NAME_CAP_PCT, "tier": None, "held_pct": None,
               "price": price, "shares_before": shares, "max_shares": 0, "shares_after": 0,
               "outcome": "errored", "held_source": None}
        if slack_pp:
            rec["slack_pp"] = slack_pp
        try:
            pv = float(self.portfolio_value())
            if not math.isfinite(pv) or pv <= 0:
                raise ValueError("equity not positive")
            rec["tier"] = get_active_limits(pv)["tier"]
            if rec["tier"] != "micro":
                return None
            px = float(price)
            if not math.isfinite(px) or px <= 0:
                raise ValueError("price not positive")
            try:
                held = self._held_value_strict(ticker)
            except _StopRecordNoBroker:
                rec["held_source"] = "stop_record_no_broker_position"
                self._alert_stop_record_no_broker(ticker)
                raise
            except Exception:
                rec["held_source"] = "error"
                raise
            rec["held_source"] = "broker"
        except Exception as e:
            log.error(f"[{ticker}] micro name cap: account, position or price unusable "
                      f"({type(e).__name__}); blocking the buy")
            return rec
        room = pv * (MICRO_TIER_NAME_CAP_PCT - slack_pp) / 100 - held
        max_shares = max(int(room / px + 1e-9), 0)
        after = min(shares, max_shares)
        rec.update(held_pct=round(held / pv * 100, 2), max_shares=max_shares, shares_after=after,
                   outcome=("blocked" if after <= 0 < shares
                            else "shrunk" if after < shares else "within_cap"))
        return rec

    def shares_after_concentration(self, ticker: str, sizing: dict, bucket: str,
                                   name_cap: bool = False, price: float | None = None,
                                   name_cap_slack_pp: float = 0.0) -> int:
        """shares_initial after the concentration caps, checked at the larger
        of target and order weight (same rule as crew._route_to_approval).
        With name_cap, then the micro-tier name cap at price (default the
        sizing's price_at_sizing; none known gives 0 on the micro tier); its
        record is kept on self.last_name_cap (None when it didn't apply)."""
        shares = int(sizing.get("shares_initial", 0) or 0)
        self.last_concentration_errored = False
        self.last_name_cap = None
        if shares <= 0:
            return 0
        proposed = max(sizing.get("position_size_pct") or 0, sizing.get("order_pct") or 0)
        conc = self.check_concentration(ticker, proposed, bucket=bucket)
        # Lets a caller tell "no room" from "couldn't check" (both return 0).
        self.last_concentration_errored = conc.get("errored") is True
        if not conc.get("passed", False):
            caps = [c for c in (conc.get("ai_infra_size_cap_pct"), conc.get("theme_size_cap_pct"),
                                conc.get("moonshot_size_cap_pct")) if c is not None]
            cap = min(caps) if caps else None
            if cap is not None and cap < proposed:
                base = (sizing.get("order_pct") or 0) or proposed
                shares = min(shares, int(shares * cap / base + 1e-9)) if base > 0 else 0
        if name_cap and shares > 0:
            price = price or sizing.get("price_at_sizing")
            self.last_name_cap = self.micro_name_cap(ticker, shares, float(price or 0),
                                                     slack_pp=name_cap_slack_pp)
            if self.last_name_cap is not None:
                shares = self.last_name_cap["shares_after"]
        return shares

    def would_size_after_swap(self, ticker: str, conviction: int, bucket: str,
                              regime: str, price: float, swap: dict) -> int:
        """Shares the new buy would get if `swap` were sold first: sizing run
        against the account with the swap's market value added to cash and
        its position removed. crew used to sell first and re-size after, so a
        buy that still sized to 0 left the holding sold and nothing bought
        (limitations audit 2026-10-28 09:00, finding 9). Returns 0 on error.
        No ATR-fallback alert from this pre-check."""
        from types import SimpleNamespace
        real = self._client
        try:
            acct = real.get_account()
            freed = float(swap.get("market_value") or 0)
            view_acct = SimpleNamespace(portfolio_value=float(acct.portfolio_value),
                                        cash=float(acct.cash) + freed,
                                        buying_power=float(acct.buying_power) + freed)
            remaining = [p for p in real.get_all_positions() if p.symbol != swap.get("ticker")]
        except Exception as e:
            log.warning(f"[{ticker}] swap pre-check could not read the account: {e}")
            return 0

        class _View:
            def get_account(self):
                return view_acct

            def get_all_positions(self):
                return remaining

            def get_open_position(self, symbol):
                for p in remaining:
                    if p.symbol == symbol:
                        return p
                raise Exception(f"No open position in {symbol}")

        try:
            sim = RiskManager(_View(), start_value=self.start_value)
            sim.quiet_fallback_alert = True   # the real sizing after the swap still alerts
            sizing = sim.size_position(ticker, conviction, bucket, regime, price,
                                       log_fallback=False)
            # Same concentration cap crew applies after sizing (professor M36):
            # a swap whose buy the caps cut to 0 must not sell the holding.
            # And the micro-tier name cap, as on the route after the swap,
            # with SWAP_NAME_CAP_SLACK_PP of margin so a one-share boundary
            # case doesn't sell a holding the post-sale cap then won't use.
            return sim.shares_after_concentration(ticker, sizing, bucket, name_cap=True,
                                                  price=price,
                                                  name_cap_slack_pp=SWAP_NAME_CAP_SLACK_PP)
        except Exception as e:
            log.warning(f"[{ticker}] swap pre-check sizing failed: {e}")
            return 0

    # ------------------------------------------------------------------
    # Backward-compatible methods (used by remaining code)
    # ------------------------------------------------------------------

    def current_position_value(self, symbol: str) -> float:
        try:
            position = self._client.get_open_position(symbol)
            return float(position.market_value)
        except Exception:
            return 0.0

    def shares_affordable(self, symbol: str, price: float) -> int:
        """Legacy method — uses flat 2%/20% caps. Use size_position() for new code."""
        if self.is_halted():
            return 0
        pv = self.portfolio_value()
        trade_cap    = pv * 0.02
        position_cap = pv * 0.20
        held         = self.current_position_value(symbol)
        budget       = min(trade_cap, max(0, position_cap - held))
        return round(budget / price, 4) if price else 0.0

    def check_sell_allowed(self, symbol: str, shares: float) -> bool:
        try:
            position = self._client.get_open_position(symbol)
            owned = float(position.qty)
            if shares > owned:
                log.warning(f"Sell rejected — only {owned} shares of {symbol} owned")
                return False
            return True
        except Exception:
            log.warning(f"Sell rejected — no position in {symbol}")
            return False

    # ------------------------------------------------------------------
    # Trailing stop monitor
    # ------------------------------------------------------------------

    def _changed_while_waiting(self, ticker: str, qty: float | None = None,
                               stop_px: float | None = None) -> str | None:
        """Called once a ticker's stop lock is held. The caller read the
        position and stop before taking the lock; if another stop change ran
        in between (a partial sale, TRIM, restore), acting on that snapshot
        could cancel the new stop and place one for the old quantity
        (professor C18, 2026-10-28). Returns why to skip, or None."""
        import db
        if stop_px is not None:
            now = (db.get_position_stop(ticker) or {}).get("stop_price")
            if now is None or abs(float(now) - float(stop_px)) > 1e-6:
                return f"recorded stop moved {stop_px} -> {now}"
        if qty is not None:
            try:
                pos = next((p for p in self._client.get_all_positions() if p.symbol == ticker), None)
                live = round(abs(float(pos.qty)), 4) if pos is not None else 0.0
            except Exception:
                return None  # can't tell: don't block a protective change on a read error
            if abs(live - float(qty)) > 1e-6:
                return f"position changed {qty} -> {live}"
        return None

    def _resting_stop_floor(self, ticker: str, new_level: float,
                            what: str) -> tuple[bool, float | None]:
        """Read the working SELL stops for ticker BEFORE a cancel-all and
        replace. Returns (ok, highest resting stop price or None).

        ok is False when the replacement at new_level would sit below the
        highest resting stop (the record can be lower than what rests: a
        manual TWS change, an unrecorded older tranche, adoption), when the
        read fails, or when a resting stop's price can't be read. The caller
        then changes nothing, so the resting stops stay. A TRAIL counts at
        its current trigger price (ibkr.get_open_orders' trailStopPrice)."""
        return _read_resting_stop_floor(ticker, new_level, what)

    def check_trailing_stops(self) -> list[dict]:
        """
        Review all open positions against their logged stops.
        Returns list of {ticker, action, current_price, stop_price, reason}
        for any positions that should be exited or have trailing stop updated.
        """
        self.monitor_read_failed = False
        try:
            import db
            stops    = db.get_all_position_stops(strict=True)
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"Trailing stop check failed: {e}")
            self.monitor_read_failed = True
            return []
        # Stops move only on regular-hours prices: an after-hours print (an
        # earnings gap, say) would arm the trail or ratchet the ladder from a
        # price the open may not confirm (audit A1031-0053-8). Unknown counts
        # as closed. The stop-breach check below still runs.
        try:
            import market_data
            rth = bool(market_data.is_market_hours())
        except Exception:
            rth = False

        alerts: list[dict] = []
        for stop_record in stops:
            ticker = stop_record["ticker"]
            pos    = positions.get(ticker)
            if not pos:
                continue

            current = float(pos.current_price)
            stop_px = stop_record["stop_price"]
            bucket  = stop_record.get("thesis_bucket", "Accelerator")

            # Check if stop hit
            if current <= stop_px:
                alerts.append({
                    "ticker":        ticker,
                    "action":        "EXIT",
                    "current_price": current,
                    "stop_price":    stop_px,
                    "rth":           rth,
                    "reason":        (f"Stop triggered at ${stop_px:.2f} (current ${current:.2f})" if rth else
                                      f"Price ${current:.2f} is below the ${stop_px:.2f} stop outside "
                                      f"regular hours"),   # tasks adds what covers it
                })
                continue

            if not rth:
                continue   # stops move on regular-hours prices only (see above)

            # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_
            # GATES_ENABLED, a position carrying a live setup category uses
            # the ladder-rung ratchet below INSTEAD OF the fixed ATR-
            # multiple trail-activation logic that follows — skip it
            # entirely for this ticker rather than letting both paths touch
            # the same stop in the same run.
            if SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE:
                self._ratchet_ladder_stop(ticker, pos, current, stop_px, alerts)
                continue

            # Check if trailing stop should be activated
            entry    = stop_record.get("entry_price", current)
            gain_pct = (current - entry) / entry if entry else 0
            activate_at = TRAILING_STOP_ACTIVATE.get(bucket)
            trail_lo, trail_hi = TRAILING_STOP_DISTANCE.get(bucket, (None, None))

            if activate_at and gain_pct >= activate_at and trail_lo:
                trail_dist = (trail_lo + trail_hi) / 2
                already_trailing = bool(stop_record.get("trailing_active"))

                if not already_trailing:
                    # ONE-TIME conversion to a native IBKR TRAIL order. After
                    # this, the broker ratchets the stop itself on every tick
                    # — no more cancel/replace round-trips, and no window
                    # where the position is briefly unprotected between
                    # cancelling the old stop and placing the new one.
                    #
                    # Trail distance is ATR-based (N x ATR, same multiplier
                    # table as the initial stop) rather than a flat percent —
                    # a flat % gives every stock in a bucket the same trail
                    # regardless of how it individually trades, the same
                    # one-size-fits-all problem the initial stop already
                    # solved by using ATR instead of a flat percent. Falls
                    # back to the flat-% distance only if ATR is unavailable.
                    atr = self.get_atr(ticker)
                    trail_amount = None
                    trail_pct = None
                    if atr:
                        mult_lo, mult_hi = ATR_STOP_MULTIPLIER.get(bucket, (1.5, 2.0))
                        mult = (mult_lo + mult_hi) / 2
                        trail_amount = round(mult * atr, 2)
                        new_stop = round(current - trail_amount, 2)
                        method_label = f"{mult:.1f}x ATR (${atr:.2f}) = ${trail_amount:.2f}"
                    else:
                        trail_pct = round(trail_dist * 100, 2)
                        new_stop = round(current * (1 - trail_dist), 2)
                        method_label = f"{trail_pct}% (ATR unavailable, flat-% fallback)"

                    # A re-arm after a partial sale or TRIM starts the trail
                    # from today's price; don't convert while that would sit
                    # below the fixed stop already in place (audit 12:53 #3).
                    if stop_px and new_stop < float(stop_px):
                        log.info(f"[{ticker}] Trail not armed yet: ${new_stop:.2f} would be below "
                                 f"the current stop ${float(stop_px):.2f}")
                        continue

                    try:
                        # One stop change per ticker at a time (stop_lock).
                        with ticker_lock(ticker):
                          changed = self._changed_while_waiting(
                              ticker, qty=round(abs(float(pos.qty)), 4), stop_px=stop_px)
                          # Never cancel a higher resting stop for a lower
                          # trail start; read before the cancel (no-loosen).
                          floor_ok, resting_high = (False, None) if changed else \
                              self._resting_stop_floor(ticker, new_stop, "TRAIL conversion")
                          if changed:
                            log.info(f"[{ticker}] TRAIL conversion skipped this run: {changed}")
                          elif not floor_ok:
                            pass   # logged (and alerted) by _resting_stop_floor
                          else:
                            # The fallback restore never goes below what was
                            # cancelled. Equal to the record when consistent.
                            restore_px = float(stop_px) if resting_high is None or \
                                resting_high <= float(stop_px) else resting_high
                            cancelled = False
                            # Set once the broker accepts the TRAIL: from then
                            # on no fallback below may place another stop, or
                            # two full-size stops rest (professor M227).
                            placed = False
                            try:
                                import tasks
                                import execution
                                qty = round(abs(float(pos.qty)), 4)
                                tasks.manage_cancel_stop_orders_for(ticker)
                                cancelled = True
                                if trail_amount is not None:
                                    result = tasks.manage_place_trailing_stop(ticker, qty, trail_amount=trail_amount)
                                else:
                                    result = tasks.manage_place_trailing_stop(ticker, qty, trail_percent=trail_pct)
                                placed = True
                                execution._record_placed_stop(ticker, "TRAIL conversion", result.get("order_id"), [
                                    ("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"])),
                                    ("trailing stop", lambda: db.update_trailing_stop(ticker, new_stop, current)),
                                    ("trail params", lambda: db.set_trail_params(ticker, trail_amount, trail_pct)),
                                ])
                                log.info(f"[{ticker}] Converted to native TRAIL stop ({method_label}), "
                                        f"orderId={result.get('order_id')}")
                            except Exception as e:
                                if placed:
                                    # Unreachable unless the log line itself
                                    # raised: the TRAIL rests, so no fallback.
                                    log.error(f"[{ticker}] Error after the TRAIL was placed ({e}); "
                                              f"no fallback stop placed")
                                    continue
                                log.error(f"[{ticker}] Native TRAIL conversion failed — falling back to "
                                         f"one-shot cancel/replace: {e}")
                                prev_high = stop_record.get("trailing_high") or entry
                                if not cancelled:
                                    # The cancel itself failed: the old stop is
                                    # still resting, so placing another would
                                    # stack two (professor C19).
                                    log.error(f"[{ticker}] Stop cancel failed before the TRAIL "
                                              f"conversion; existing stop left in place")
                                elif isinstance(e, tasks.OrdersCallUnknown):
                                    # The TRAIL may be resting; a fixed stop on
                                    # top would stack. Reconcile fixes coverage
                                    # (professor C24).
                                    log.error(f"[{ticker}] TRAIL placement result unknown; no "
                                              f"fallback stop placed, reconcile will fix coverage")
                                    # The old stop is cancelled, so say so (m38).
                                    try:
                                        import execution
                                        execution._alert_stop_unknown(
                                            ticker, round(abs(float(pos.qty)), 4), float(new_stop))
                                    except Exception:
                                        pass
                                elif not (current > prev_high and new_stop > stop_px):
                                    # The stops were already cancelled above: put the
                                    # fixed stop back rather than leave it unprotected
                                    # (professor C16, 2026-10-28).
                                    try:
                                        import tasks
                                        import execution
                                        qty = round(abs(float(pos.qty)), 4)
                                        # Sized to what no working stop covers:
                                        # the failed call may have placed the
                                        # TRAIL anyway (M227 follow-up).
                                        outcome, result = execution._restore_sized_stop(
                                            ticker, qty, restore_px, "TRAIL conversion",
                                            market_price=current)
                                        if outcome == "placed":
                                            # The restore rests: a write failure
                                            # must not reach the queue below (M227).
                                            execution._record_placed_stop(
                                                ticker, "Stop restore after a failed TRAIL",
                                                result.get("order_id"),
                                                [("order id", lambda: db.set_stop_broker_order_id(
                                                    ticker, result["order_id"]))])
                                            log.info(f"[{ticker}] Fixed stop restored @ ${restore_px:.2f} "
                                                     f"after the failed TRAIL conversion")
                                        if outcome == "placed" and restore_px > float(stop_px):
                                            # Restored at a higher cancelled
                                            # stop: raise the record to match
                                            # so the next run works from it.
                                            # The write logs its own SQL errors,
                                            # but its pool checkout (_get_conn)
                                            # sits outside that try: a raise here
                                            # must not reach the restore-failed
                                            # path below while the stop rests.
                                            try:
                                                db.update_trailing_stop(
                                                    ticker, restore_px,
                                                    stop_record.get("trailing_high"),
                                                    native_trail=False)
                                            except Exception as e4:
                                                log.warning(f"[{ticker}] Stop restored @ ${restore_px:.2f} "
                                                            f"but the record update failed: {e4}")
                                    except Exception as e2:
                                        log.error(f"[{ticker}] Stop restore after TRAIL failure failed: {e2}")
                                        try:
                                            # The stops are cancelled: hand the
                                            # restore to the retry queue (m39).
                                            tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
                                        except Exception as e3:
                                            log.error(f"[{ticker}] Could not queue the stop restore: {e3}")
                                        alerts.append({
                                            "ticker":        ticker,
                                            "action":        "FIX_STOP",
                                            "current_price": current,
                                            "stop_price":    stop_px if restore_px == float(stop_px) else restore_px,
                                            "reason":        f"TRAIL conversion failed and the ${restore_px:.2f} "
                                                             f"stop could not be restored — place it manually in TWS",
                                        })
                                        tasks.open_manual_action(
                                            ticker, f"restore the stop @ ${restore_px:.2f} after a "
                                                    f"failed TRAIL conversion")
                                elif current > prev_high and new_stop > stop_px:
                                    try:
                                        import tasks
                                        import execution
                                        tasks.manage_cancel_stop_orders_for(ticker)
                                        qty = round(abs(float(pos.qty)), 4)
                                        result = tasks.manage_stop_place(ticker, qty, new_stop,
                                                                         market_price=current)
                                        execution._record_placed_stop(
                                            ticker, "Fallback stop move", result.get("order_id"), [
                                                ("order id", lambda: db.set_stop_broker_order_id(
                                                    ticker, result["order_id"])),
                                                ("trailing stop", lambda: db.update_trailing_stop(
                                                    ticker, new_stop, current, native_trail=False)),
                                            ])
                                        log.info(f"[{ticker}] Broker stop moved to ${new_stop:.2f} (fallback path)")
                                    except Exception as e2:
                                        log.error(f"[{ticker}] Fallback stop move also failed: {e2}")
                                        alerts.append({
                                            "ticker":        ticker,
                                            "action":        "FIX_STOP",
                                            "current_price": current,
                                            "stop_price":    new_stop,
                                            "reason":        f"Trailing stop should be ${new_stop:.2f} but "
                                                             f"broker order update FAILED — fix manually in TWS",
                                        })
                    except TickerBusy:
                        log.warning(f"[{ticker}] TRAIL conversion skipped this run: "
                                    f"another stop change holds the ticker")
                else:
                    # Already converted to a native TRAIL order — the broker
                    # is ratcheting it automatically. Just track the new high
                    # in DB for dashboard visibility; no order action needed.
                    # The recorded stop follows the trail as a lower bound
                    # (new high - trail), so a later partial sale, TRIM or
                    # restore that replaces the TRAIL with a fixed STP starts
                    # from about where the trail was, not the conversion value
                    # (limitations audit 12:53, #3 follow-up).
                    prev_high = stop_record.get("trailing_high") or entry
                    if current > prev_high:
                        est = None
                        if stop_record.get("trail_amount"):
                            est = round(current - stop_record["trail_amount"], 2)
                        elif stop_record.get("trail_percent"):
                            est = round(current * (1 - stop_record["trail_percent"] / 100), 2)
                        new_db_stop = max(stop_px, est) if est is not None else stop_px
                        try:
                            db.update_trailing_stop(ticker, new_db_stop, current)
                        except Exception as e:
                            log.warning(f"Failed to update trailing high for {ticker}: {e}")

        return alerts

    def _restore_after_failed_ratchet(self, ticker: str, qty: float, stop_px: float,
                                      resting_high: float | None, current: float,
                                      err: Exception) -> None:
        """The ratchet's cancel went through and its placement failed: put
        the previous stop back (never below what was cancelled), sized to the
        shares no working stop covers, or queue it for the reconcile retry.
        A placement with no reply may be resting: no restore then."""
        import execution
        import db
        import tasks
        if isinstance(err, tasks.OrdersCallUnknown):
            execution._alert_stop_unknown(ticker, qty, float(stop_px))
            return
        restore_px = float(stop_px) if resting_high is None or resting_high <= float(stop_px) \
            else float(resting_high)
        try:
            outcome, result = execution._restore_sized_stop(
                ticker, qty, restore_px, "Ladder ratchet", market_price=current)
        except Exception as e2:
            log.error(f"[{ticker}] Stop restore after the failed ratchet failed: {e2}")
            try:
                tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
            except Exception as e3:
                log.error(f"[{ticker}] Could not queue the stop restore: {e3}")
            try:
                tasks.open_manual_action(
                    ticker, f"restore the stop @ ${restore_px:.2f} after a failed ladder ratchet")
            except Exception as e4:
                log.error(f"[{ticker}] could not open the manual action: {e4}")
            return
        if outcome == "placed":
            writes = [("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"]))]
            if restore_px > float(stop_px):
                writes.append(("record stop", lambda: db.update_trailing_stop(
                    ticker, restore_px, current, native_trail=False)))
            execution._record_placed_stop(ticker, "Stop restore after a failed ladder ratchet",
                                          result.get("order_id"), writes)

    @locked_by_ticker(lambda self, ticker, *a, **k: ticker)
    def _ratchet_ladder_stop(self, ticker: str, pos: object, current: float,
                              stop_px: float, alerts: list[dict]) -> None:
        """
        SETUP_GATES_ENABLED trailing-stop path (2026-10-11): ratchets the
        stop to the highest ladder rung (agents.trade_ladder's nightly
        rungs, confluence_score >= 1 -- i.e. every real rung, single-type
        levels included, not just multi-confluence clusters) strictly
        below the current price, refreshed from whatever the ladder's
        latest nightly build computed -- instead of a fixed ATR-multiple
        trail distance. Ratchet-only: never moves the stop down, same
        guarantee the legacy ATR-trail path above already provides (a new
        candidate rung below the existing stop is simply ignored).
        """
        import db
        changed = self._changed_while_waiting(ticker, stop_px=stop_px)
        if changed:
            log.info(f"[{ticker}] Ladder ratchet skipped this run: {changed}")
            return
        ladder = db.get_trade_ladder(ticker)
        if not ladder:
            return
        candidates = [
            r["price"] for r in (ladder.get("rungs") or [])
            if r.get("confluence_score", 0) >= 1 and r["price"] < current
        ]
        if not candidates:
            return
        new_stop = max(candidates)
        if new_stop <= stop_px:
            return  # not an improvement -- never move down
        # Nor below a higher stop resting at the broker; read before the
        # cancel, and a failed read changes nothing (no-loosen).
        floor_ok, resting_high = self._resting_stop_floor(ticker, new_stop, "Ladder ratchet")
        if not floor_ok:
            return

        import tasks
        import execution
        qty = round(abs(float(pos.qty)), 4)
        cancelled = False
        try:
            tasks.manage_cancel_stop_orders_for(ticker)
            cancelled = True
            result = tasks.manage_stop_place(ticker, qty, new_stop, market_price=current)
        except Exception as e:
            log.error(f"[{ticker}] Ladder trailing stop ratchet FAILED: {e}")
            alerts.append({
                "ticker":        ticker,
                "action":        "FIX_STOP",
                "current_price": current,
                "stop_price":    new_stop,
                "reason":        f"Ladder trailing stop should be ${new_stop:.2f} but broker "
                                 f"order update FAILED — fix manually in TWS",
            })
            if cancelled:
                self._restore_after_failed_ratchet(ticker, qty, stop_px, resting_high, current, e)
            return
        # Placed: a record write failure is logged and alerted, not read as
        # a failed ratchet (professor M227).
        execution._record_placed_stop(
            ticker, "Ladder ratchet", result.get("order_id"), [
                ("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"])),
                ("trailing stop", lambda: db.update_trailing_stop(
                    ticker, new_stop, current, native_trail=False)),
            ])
        log.info(f"[{ticker}] Ladder trailing stop ratcheted ${stop_px:.2f} -> ${new_stop:.2f}")

    def _restore_previous_stop(self, ticker: str, held: float, old_stop: float,
                               floor: float | None, market_price: float | None, what: str,
                               trailing_high) -> str:
        """After a failed breakeven move or profit-target sale (stops already
        cancelled): put the old stop back at max(record, resting floor), sized
        to the shares no working stop covers. A restore above the record
        raises the record to match. Returns the outcome: "placed", "covered"
        (nothing uncovered), "unread" (coverage unreadable: nothing placed,
        queued with an alert by _restore_sized_stop), "sign_flip" (the
        position reads disagree on the sign: nothing placed) or "failed" (the
        placement raised, for the caller's alert)."""
        import db
        import execution
        restore_px = max(float(old_stop), float(floor)) if floor is not None else float(old_stop)
        try:
            outcome, result = execution._restore_sized_stop(
                ticker, round(held, 4), restore_px, f"Stop restore after a failed {what}",
                market_price=market_price)
        except Exception as e:
            log.error(f"[{ticker}] Could not restore stop after a failed {what}: {e}")
            return "failed"
        if outcome == "placed":
            writes = [("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"]))]
            if restore_px > float(old_stop):
                # Restored above the record (a resting TRAIL): the next run
                # works from what rests.
                writes.append(("record stop", lambda: db.update_trailing_stop(
                    ticker, restore_px, trailing_high, native_trail=False)))
            execution._record_placed_stop(ticker, f"Stop restore after a failed {what}",
                                          result.get("order_id"), writes)
        return outcome

    @locked_by_ticker(lambda self, ticker, *a, **k: ticker, on_busy=False)
    def _breakeven_without_sale(self, ticker: str, qty: float, new_stop: float,
                                current_price: float | None = None) -> bool:
        """Profit target hit on a position too small to sell part of (1-3
        shares x 25-45% rounds to 0). Before 2026-10-28 that just continued,
        and the breakeven stop move lived inside the partial sale, so it never
        happened: a 1-share position could rise past target, round-trip and
        stop out at a loss (limitations audit 07:00, finding 3). Now the stop
        moves to breakeven for the whole position and the target is marked
        taken. Cancel first, then place; if the new stop fails, the old one
        goes back on. Returns True when the target is marked taken."""
        import alerts
        import db
        import execution
        import tasks

        changed = self._changed_while_waiting(ticker, qty=qty)
        if changed:
            log.info(f"[{ticker}] Breakeven move skipped this run: {changed}")
            return False
        import ibkr
        new_stop = ibkr.round_sell_stop_up_to_tick(new_stop, current_price)   # raw fill price (audit 2026-11-02 #9)

        stop_rec = db.get_position_stop(ticker) or {}
        old_stop = stop_rec.get("stop_price")
        if old_stop is not None and float(old_stop) >= new_stop:
            try:
                db.mark_profit_target_taken(ticker, float(old_stop))
            except Exception as e:
                # Nothing was sold or moved: the next run tries again.
                log.error(f"[{ticker}] Breakeven: could not mark the target taken: {e}")
                return False
            return True
        # A native TRAIL's trigger can sit above the record: never replace it
        # lower. Read before the cancel; a failed read keeps new_stop (M228).
        new_stop, floor = _floor_at_resting_stop(ticker, new_stop, "Breakeven move",
                                                 market_price=current_price)
        try:
            tasks.manage_cancel_stop_orders_for(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Breakeven move: could not cancel broker stop: {e}")
            return False
        try:
            result = tasks.manage_stop_place(ticker, round(qty, 4), new_stop,
                                              market_price=current_price)
        except Exception as e:
            log.error(f"[{ticker}] Breakeven stop placement failed: {e}")
            restored = False
            if isinstance(e, tasks.OrdersCallUnknown):
                # The new stop may be resting; restoring the old one would
                # stack two. Reconcile fixes coverage (professor C24).
                alerts._send(f"⚠️ STOP STATUS UNKNOWN — {ticker}\nThe breakeven stop got no "
                             f"reply; not restoring the old one. The stop check fixes coverage "
                             f"within ~30 min.")
                return False
            if old_stop:
                # Back at the highest cancelled stop, sized to what no working
                # stop covers (the failed call may have placed its stop).
                restored = self._restore_previous_stop(
                    ticker, qty, float(old_stop), floor, current_price, "breakeven move",
                    stop_rec.get("trailing_high")) != "failed"
            if not restored:
                alerts._send(
                    f"🚨 UNPROTECTED POSITION — {ticker}\n"
                    f"Breakeven stop move failed and the old stop could not be restored. "
                    f"Place a stop for {round(qty, 4)} shares manually."
                )
            return False
        # Placed: a record write failure must not propagate (professor M227).
        # If the target can't be marked taken, the next run moves the stop
        # again (cancel, then place), so no second stop rests.
        execution._record_placed_stop(ticker, "Breakeven move", result.get("order_id"), [
            ("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"])),
            ("profit target taken", lambda: db.mark_profit_target_taken(ticker, new_stop)),
        ])
        log.info(f"[{ticker}] Profit target hit on {qty:g} shares (too few to sell part): "
                 f"stop moved to breakeven ${new_stop:.2f}, no sale")
        return True

    def _hold_profit_target_sale(self, ticker: str, qty: float, old_stop, floor: float | None,
                                 decision_price: float | None, trailing_high,
                                 resting: float | None) -> None:
        """The profit-target sale is held because a stop still works (or the
        stops can't be read) after the cancel. Nothing is sold and the target
        is not marked taken, so the next run tries again. Protection: the old
        stop goes back for the shares no non-cancelling stop covers
        (_restore_previous_stop), the restore is queued for reconcile, and
        one alert is sent."""
        import alerts
        import tasks
        why = ("the resting stops could not be read" if resting is None
               else f"stop(s) for {resting:g} shares were still working after the cancel")
        log.error(f"[{ticker}] Profit-target sale held: {why}; nothing sold")
        # Stops still working that aren't being cancelled (another client's,
        # or a cancel the broker refused): read before our restore, so it
        # isn't among them. While any of them works, the target is skipped
        # rather than cancelling the restored stop every run.
        if resting:
            _note_pt_hold_working_stops(ticker)
        outcome = "no_record"   # no previous stop to put back
        if old_stop:
            outcome = self._restore_previous_stop(
                ticker, qty, float(old_stop), floor, decision_price,
                "profit-target sale hold", trailing_high)
        queued = False
        try:
            tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
            queued = True
        except Exception as e:
            log.error(f"[{ticker}] Could not queue the stop restore: {e}")
        try:
            if outcome != "failed":
                # Throttled: a stop the cancel can't take holds the sale run
                # after run (the skip above stops most of those runs).
                try:
                    due = tasks._reconcile_alert_due(f"pt_sale_held_{ticker}")
                except Exception as e:
                    log.warning(f"[{ticker}] held-sale alert throttle unreadable ({e}); sending")
                    due = True
                if due:
                    if outcome == "unread":
                        protection = ("Protection not restored yet; queued for the reconcile "
                                      "retry, which places it once the stops can be read")
                    elif outcome == "no_record":
                        protection = ("No previous stop was recorded, so none was put back"
                                      + ("; the restore is queued for the reconcile check"
                                         if queued else "") + ". Check the stop in TWS")
                    elif outcome == "sign_flip":
                        protection = ("No stop was placed: the position reads disagree on "
                                      "whether it is long or short. Check the position in TWS")
                    elif outcome == "covered":
                        protection = ("No stop was placed: working stops already cover the "
                                      "position"
                                      + (", and the restore is queued for the reconcile check"
                                         if queued else ""))
                    else:
                        protection = ("Protection was put back"
                                      + (" and the restore is queued for the reconcile check"
                                         if queued else ""))
                    alerts._send(
                        f"⚠️ PROFIT-TARGET SALE HELD — {ticker}\n"
                        f"Nothing was sold: {why}, and selling on top of a working stop could "
                        f"short the account. {protection}. The target is retried once the "
                        f"other stop is gone; check the stops in TWS if this repeats.")
            else:
                alerts._send(
                    f"🚨 UNPROTECTED POSITION — {ticker}\n"
                    f"The profit-target sale was held ({why}) and the stop could not be "
                    f"restored. "
                    + ("It is queued and retried every reconcile cycle; " if queued else "")
                    + f"place a stop for {round(qty, 4)} shares manually if it isn't back "
                    f"within ~15 min.")
        except Exception:
            pass

    @locked_by_ticker(lambda self, ticker, *a, **k: ticker)
    def _partial_sell_and_resize_stop(self, ticker: str, qty: float, sell_qty: float,
                                      new_stop: float,
                                      decision_price: float | None = None) -> dict | None:
        """Profit-target partial sale with the broker stop kept in step.

        Before 2026-10-27 both profit-target paths sold part of the position and
        only wrote the breakeven stop to the DB. The broker STP stayed at the old
        price for the ORIGINAL quantity, so it could oversell into a short, and
        the 30-min backstop saw the size mismatch as uncovered. Same pattern as
        execution.py's TRIM path: cancel the stop, sell, then place a stop for
        what's left. If the sale fails, the old stop goes back on.
        Returns the order result, or None if the sale failed."""
        import alerts
        import db
        import execution
        import ibkr
        import tasks

        # Outside regular hours a market SELL is cancelled after the fill
        # wait (ibkr.place_market_order), and the stop would already be gone:
        # leave everything as is and let the next in-hours check take the
        # target (professor C15, 2026-10-28).
        import market_data
        if not market_data.is_market_hours():
            log.info(f"[{ticker}] Profit target reached outside regular hours: sale deferred to RTH")
            return None
        changed = self._changed_while_waiting(ticker, qty=qty)
        if changed:
            log.info(f"[{ticker}] Profit-target sale skipped this run: {changed}")
            return None
        if _pt_hold_still_blocked(ticker):
            return None   # logged there; nothing cancelled

        stop_rec = db.get_position_stop(ticker) or {}
        old_stop = stop_rec.get("stop_price")
        breakeven = new_stop
        # Never move the stop down: after a trail conversion the recorded stop
        # can already be above breakeven (limitations audit 12:53, finding 3).
        if old_stop is not None:
            new_stop = max(new_stop, float(old_stop))
        # A SELL STP at or above the market fills at once. Use the highest
        # candidate below the price that triggered the target (professor M-1).
        if decision_price and new_stop >= decision_price:
            below = [c for c in (breakeven, float(old_stop) if old_stop is not None else None)
                     if c is not None and c < decision_price]
            new_stop = max(below) if below else breakeven
            log.warning(f"[{ticker}] Recorded stop is at/above the price; using ${new_stop:.2f}")
        # Breakeven is the raw average fill: put it on the tick, rounding
        # up (never wider) unless that reaches the price, so the broker
        # can't reject it (audit 2026-11-02 #9).
        new_stop = ibkr.round_sell_stop_up_to_tick(new_stop, decision_price)
        # A native TRAIL's trigger can sit above the record: the stop for
        # the rest never goes below it, nor does a restore. Read before the
        # cancel; a failed read keeps new_stop (professor M228).
        new_stop, floor = _floor_at_resting_stop(ticker, new_stop, "Profit-target stop",
                                                 market_price=decision_price)
        try:
            tasks.manage_cancel_stop_orders_for(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Profit-target: could not cancel broker stop, not selling: {e}")
            return None
        # The cancel can return with a stop still working (PendingCancel
        # after its poll, or a stop another client placed). Selling on top
        # of it, then re-placing the remainder's stop, leaves two stops that
        # could both fill into a short: hold the sale, put protection back
        # and queue the reprotect, as execute_approved does (A2253-1).
        resting = execution._stops_still_resting(ticker)
        if resting or resting is None:
            self._hold_profit_target_sale(ticker, qty, old_stop, floor, decision_price,
                                          stop_rec.get("trailing_high"), resting)
            return None

        try:
            import stop_lock
            stop_lock.assert_held(ticker)  # professor M69
            order_result = ibkr.place_market_order(ticker, "SELL", sell_qty)
        except Exception as e:
            log.error(f"[{ticker}] Profit-target partial sell FAILED: {e}")
            if old_stop:
                restore_px = max(float(old_stop), floor) if floor is not None else float(old_stop)
                if self._restore_previous_stop(
                        ticker, qty, float(old_stop), floor, decision_price,
                        "profit-target sale", stop_rec.get("trailing_high")) == "failed":
                    alerts._send(
                        f"🚨 UNPROTECTED POSITION — {ticker}\n"
                        f"Profit-target sale failed and the stop could not be restored. "
                        f"Place stop @ ${restore_px:.2f} for {round(qty, 4)} shares manually."
                    )
            return None

        if float(order_result.get("filled_qty") or 0) <= 0:
            # Not restored automatically: a late fill plus a full-size stop
            # could short. Not marked taken, so the target can fire again.
            log.error(f"[{ticker}] Profit-target sale not confirmed filled: {order_result}")
            # Logged unconfirmed so reconcile_unconfirmed_orders sees a late
            # fill and re-protects the rest (professor C12, 2026-10-28).
            import execution
            execution._log_unconfirmed_order(
                ticker=ticker, side="SELL", order_type="MARKET", qty=sell_qty,
                broker_order_id=order_result.get("order_id"),
                perm_id=order_result.get("perm_id"),
                decision_price=decision_price, status="unconfirmed",
                intent="PROFIT_TARGET", stop_price=new_stop)
            alerts._send(
                f"⚠️ PROFIT-TARGET SALE UNCONFIRMED — {ticker}\n"
                f"SELL {sell_qty} submitted but no fill confirmed. The broker stop was "
                f"cancelled and NOT re-placed. Check the position in TWS."
            )
            return None

        # Record the sale like every other fill (audit 2026-10-28 10:53 #3):
        # before this, partial sales never reached trades or orders.
        filled = float(order_result.get("filled_qty") or 0)
        fill_price = order_result.get("fill_price") or decision_price or 0.0
        try:
            db.log_trade(symbol=ticker, side="SELL", quantity=filled, price=fill_price,
                         reason="Profit target partial sell",
                         price_source="fill" if order_result.get("fill_price") else
                         ("decision" if decision_price else None))
            local_id = db.log_order(ticker=ticker, side="SELL", order_type="MARKET",
                                    qty=sell_qty, broker_order_id=order_result.get("order_id"),
                                    perm_id=order_result.get("perm_id"),
                                    decision_price=decision_price, status="submitted",
                                    intent="PROFIT_TARGET", stop_price=new_stop)
            if local_id and local_id > 0 and order_result.get("fill_price"):
                # A remainder may still fill: reconcile picks up 'partial' (M53).
                if not db.update_order_fill(local_id, order_result["fill_price"], filled,
                                            "filled" if filled >= float(sell_qty) - 1e-6 else "partial",
                                            filled_at=order_result.get("fill_time")):
                    db.set_order_reconcile_note(local_id, "fill_write_failed")
            elif local_id and local_id > 0:
                # Filled with no price from the broker: the row stays
                # 'submitted' (status unchanged); the note says why.
                db.set_order_reconcile_note(local_id, "filled_no_price")
        except Exception as e:
            log.warning(f"[{ticker}] Profit-target sale logging failed: {e}")

        try:
            remaining = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
        except Exception:
            remaining = round(qty - float(order_result.get("filled_qty") or sell_qty), 4)
        if remaining > 0:
            try:
                result = tasks.manage_stop_place(ticker, remaining, new_stop,
                                                market_price=order_result.get("fill_price") or decision_price)
            except Exception as e:
                log.error(f"[{ticker}] Failed to re-place stop after profit-target sale: {e}")
                alerts._send(
                    f"🚨 UNPROTECTED POSITION — {ticker}\n"
                    f"Profit-target sale filled but stop re-placement failed. "
                    f"Place stop @ ${new_stop:.2f} for {remaining} shares manually."
                )
            else:
                # Placed: a record write failure is not "unprotected" (M227).
                execution._record_placed_stop(
                    ticker, "Stop re-place after a profit-target sale", result.get("order_id"),
                    [("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"]))])
                log.info(f"[{ticker}] Stop re-placed @ ${new_stop:.2f} for remaining {remaining} shares")
        # The stop actually used (may be above breakeven).
        return {**order_result, "new_stop": new_stop}

    def check_profit_targets(self) -> list[dict]:
        """
        Regime-adaptive profit-taking for Accelerator/Moonshot positions
        (2026-08-29, user request) -- the piece missing from the
        protective-only stop/trailing-stop path above: nothing there ever
        proactively banks a gain, it only ever reacts to price falling
        back. See config.py's PROFIT_TARGET_PCT_BY_STYLE docstring for
        the full rationale (a concrete Q2->Q3 regime-shift example where
        the same buy-and-hold posture rode gains up and gave them back).

        For each Accelerator/Moonshot position that hasn't already taken
        its target: reads that TICKER'S OWN sector trade_style (2026-08-29
        -- agents.market_regime.get_trade_style_for_ticker(), falling back
        to the whole-market get_cached_regime()["trade_style"] if the
        ticker isn't a current S&P 500 constituent or its sector hasn't
        been assessed yet; falls back further to the more conservative
        "Choppy/Distribution" table if no regime data is available at
        all, rather than silently skipping every position). A healthcare
        holding is evaluated against healthcare's own regime, not
        whatever the broad market or an unrelated sector happens to be
        doing -- see agents.market_regime.get_trade_style_for_ticker's
        docstring for why that distinction matters (a sector can trend
        while the broad market chops, or vice versa). Computes that
        style+bucket's profit target (midpoint of the configured range,
        same convention check_trailing_stops uses for ATR_STOP_MULTIPLIER
        above), and if unrealized gain has crossed it, sells
        PARTIAL_SELL_PCT_BY_STYLE of the position and moves the
        remainder's stop to breakeven in one atomic DB write
        (db.mark_profit_target_taken) -- so a crash between the sell and
        the stop update can't leave the remainder both unmarked (re-sold
        again next run) and unprotected (no breakeven floor recorded).

        Compounder positions are never touched -- PROFIT_TARGET_PCT_BY_STYLE
        has no Compounder entry at all, so this loop skips them by
        construction, not a special-cased check.

        Returns [{"ticker", "action": "PARTIAL_SELL", "shares_sold",
        "new_stop", "trade_style", "gain_pct", "order_result"}] for every
        position actually acted on this run.
        """
        import db

        self.monitor_read_failed = False
        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
            market_trade_style = regime_packet.get("trade_style") if regime_packet else None
        except Exception as e:
            log.warning(f"check_profit_targets: market regime lookup failed, defaulting cautious: {e}")
            market_trade_style = None
        if market_trade_style not in PROFIT_TARGET_PCT_BY_STYLE:
            market_trade_style = "Choppy/Distribution"

        def _trade_style_for(ticker: str) -> str:
            try:
                from agents.market_regime import get_trade_style_for_ticker
                style = get_trade_style_for_ticker(ticker)
            except Exception as e:
                log.debug(f"[{ticker}] sector trade_style lookup failed: {e}")
                style = None
            return style if style in PROFIT_TARGET_PCT_BY_STYLE else market_trade_style

        try:
            stops = db.get_all_position_stops(strict=True)
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"check_profit_targets: could not fetch positions: {e}")
            self.monitor_read_failed = True
            return []

        results: list[dict] = []

        for stop_record in stops:
            ticker = stop_record["ticker"]
            bucket = stop_record.get("thesis_bucket")
            if stop_record.get("profit_target_taken"):
                continue
            if _profit_target_unrecorded(ticker, stop_record, positions.get(ticker)):
                continue   # sold this run or earlier, mark not written yet

            # Moved up from below the old bucket-table lookup (unchanged
            # computation, just earlier) so the SETUP_GATES_ENABLED branch
            # immediately below can use pos/entry/current/gain_pct too —
            # behaviorally identical for every legacy bucket, which simply
            # falls through to the unchanged code further down.
            pos = positions.get(ticker)
            entry = stop_record.get("entry_price")
            if not pos or not entry:
                continue
            current = float(pos.current_price)
            gain_pct = (current - entry) / entry

            # Swing-trading cutover, Phase 4/5 (2026-10-11): a position
            # carrying a live setup category reads its scale-out schedule
            # from the gate/ladder instead of the flat PROFIT_TARGET_PCT_
            # BY_STYLE table — handled entirely, never falls through to the
            # legacy table-driven logic below.
            if SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE:
                trade_style = _trade_style_for(ticker)
                partial_pct = PARTIAL_SELL_PCT_BY_STYLE[trade_style]
                self._check_profit_target_setup_gates(
                    ticker, bucket, entry, current, gain_pct, partial_pct, trade_style, results,
                    opened_at=stop_record.get("opened_at"),
                )
                continue

            trade_style = _trade_style_for(ticker)
            targets_by_bucket = PROFIT_TARGET_PCT_BY_STYLE[trade_style]
            if bucket not in targets_by_bucket:
                continue  # Compounder, or any bucket not in this framework
            partial_pct = PARTIAL_SELL_PCT_BY_STYLE[trade_style]

            target_lo, target_hi = targets_by_bucket[bucket]
            target = (target_lo + target_hi) / 2
            if gain_pct < target:
                continue

            qty = abs(float(pos.qty))
            sell_qty = int(qty * partial_pct)  # whole shares; IBKR rejects fractional orders
            if sell_qty <= 0:
                if self._breakeven_without_sale(ticker, qty, entry, current_price=current):
                    results.append({"ticker": ticker, "action": "BREAKEVEN_STOP", "shares_sold": 0,
                                    "new_stop": entry, "trade_style": trade_style,
                                    "gain_pct": gain_pct})
                continue

            new_stop = entry  # breakeven on the remainder
            order_result = self._partial_sell_and_resize_stop(ticker, qty, sell_qty, new_stop,
                                                              decision_price=current)
            if order_result is None:
                continue
            new_stop = order_result.get("new_stop", new_stop)
            # A failed mark must not abort the loop or allow a second sale.
            _mark_profit_target_taken_guarded(
                ticker, new_stop, stop_record.get("opened_at"),
                held_after=_held_after_sale(qty, order_result, sell_qty), entry_price=entry,
                held_before=qty)
            log.info(f"[{ticker}] Profit target hit ({gain_pct:.1%} >= {target:.1%}, "
                     f"style={trade_style}): sold {sell_qty} shares, "
                     f"remainder stop -> breakeven ${new_stop:.2f}")
            results.append({
                "ticker": ticker, "action": "PARTIAL_SELL", "shares_sold": sell_qty,
                "new_stop": new_stop, "trade_style": trade_style, "gain_pct": gain_pct,
                "order_result": order_result,
            })

        return results

    def _check_profit_target_setup_gates(
        self, ticker: str, bucket: str, entry: float, current: float, gain_pct: float,
        partial_pct: float, trade_style: str, results: list[dict],
        opened_at=None,
    ) -> bool:
        """
        SETUP_GATES_ENABLED profit-taking path (2026-10-11): the scale-out
        schedule comes from the live setup-gate's own target_prices
        (agents/setup_gates.py's classify_setup, which projects them off
        agents.trade_ladder's rungs at classification time) instead of the
        flat PROFIT_TARGET_PCT_BY_STYLE %-of-gain table. Falls back to the
        nearest ladder rungs above entry if the ticker's live category no
        longer matches this position's stored bucket (e.g. the setup has
        since stopped qualifying) rather than skipping target-taking
        outright.

        Layers the EXISTING valuation-tier override on top UNCONDITIONALLY:
        an "Overextended" fair-value read (agents/valuation.py, via the
        same ladder) forces a partial sell even if price hasn't reached a
        ladder target yet -- same override tasks.py's digest code already
        surfaces for held positions, just now acted on here instead of
        only displayed.

        This method issues the SAME kind of real partial-sell market order
        the pre-existing (pre-cutover) check_profit_targets loop already
        places for Accelerator/Moonshot above -- it is NOT a new class of
        live-order behavior, just a different trigger (ladder target /
        valuation tier instead of the flat %-of-gain table) reached only
        when SETUP_GATES_ENABLED is explicitly set True. Never reached in
        this task's deliverable, which ships with that flag False.

        Returns True if a sell was actually executed this run.
        """
        import db

        ladder = db.get_trade_ladder(ticker)
        valuation_tier = ladder.get("valuation_tier") if ladder else None
        overextended = valuation_tier == "Overextended"

        best = None
        try:
            from agents.setup_gates import classify_setup
            best, _all_gates = classify_setup(ticker)
        except Exception as e:
            log.debug(f"[{ticker}] profit-target classify_setup failed: {e}")

        target_prices = (best.get("target_prices") or []) if (best and best.get("category") == bucket) else []
        if not target_prices and ladder:
            target_prices = sorted(
                r["price"] for r in (ladder.get("rungs") or []) if r["price"] > entry
            )[:3]

        nearest_target = min(target_prices) if target_prices else None
        target_hit = nearest_target is not None and current >= nearest_target

        if not (target_hit or overextended):
            return False

        try:
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"[{ticker}] setup-gate profit-target: could not fetch positions: {e}")
            return False
        pos = positions.get(ticker)
        if not pos:
            return False

        qty = abs(float(pos.qty))
        sell_qty = int(qty * partial_pct)  # whole shares; IBKR rejects fractional orders
        if sell_qty <= 0:
            return self._breakeven_without_sale(ticker, qty, entry, current_price=current)

        new_stop = entry  # breakeven on the remainder, same convention as the legacy path
        order_result = self._partial_sell_and_resize_stop(ticker, qty, sell_qty, new_stop,
                                                          decision_price=current)
        if order_result is None:
            return False
        new_stop = order_result.get("new_stop", new_stop)
        # A failed mark must not abort the caller's loop or allow a second sale.
        _mark_profit_target_taken_guarded(ticker, new_stop, opened_at,
                                          held_after=_held_after_sale(qty, order_result, sell_qty),
                                          entry_price=entry, held_before=qty)
        if overextended and target_hit:
            reason = f"ladder target ${nearest_target:.2f} hit + Overextended valuation"
        elif overextended:
            reason = "Overextended valuation override"
        else:
            reason = f"ladder target ${nearest_target:.2f} hit"
        log.info(f"[{ticker}] Setup-gate profit target ({reason}, gain {gain_pct:.1%}): "
                 f"sold {sell_qty} shares, remainder stop -> breakeven ${new_stop:.2f}")
        results.append({
            "ticker": ticker, "action": "PARTIAL_SELL", "shares_sold": sell_qty,
            "new_stop": new_stop, "trade_style": trade_style, "gain_pct": gain_pct,
            "setup_category": bucket, "reason": reason, "order_result": order_result,
        })
        return True

    def flag_stale_positions(self) -> list[dict]:
        """
        Identifies Accelerator/Moonshot positions that have gone nowhere
        (less than STALE_MOVE_THRESHOLD_PCT move, either direction) for
        longer than their style+bucket's stale window -- capital sitting
        idle with no momentum, independent of whether it happens to be up
        or down slightly. See config.py's STALE_WINDOW_DAYS_BY_STYLE
        docstring (calendar days, not trading days -- computed straight
        from position_stops.opened_at with no trading-calendar
        dependency).

        Deliberately does NOT exit anything itself -- returns the flagged
        list for the caller (tasks.review_positions, once wired) to bump
        to the FRONT of review_all_held_positions's oldest-reviewed-first
        rotation, so a stale position gets a fresh LLM look promptly
        instead of possibly waiting weeks behind other holdings in the
        queue. The actual TRIM/EXIT/HOLD call stays the LLM's -- same
        division of labor the rest of this system already uses
        (risk_manager computes hard numbers, the LLM makes the
        qualitative call).

        Returns [{"ticker", "days_held", "pct_move", "trade_style",
        "thesis_bucket"}] for every position past its stale window.
        """
        import db

        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
            market_trade_style = regime_packet.get("trade_style") if regime_packet else None
        except Exception as e:
            log.warning(f"flag_stale_positions: market regime lookup failed, defaulting cautious: {e}")
            market_trade_style = None
        if market_trade_style not in STALE_WINDOW_DAYS_BY_STYLE:
            market_trade_style = "Choppy/Distribution"

        def _trade_style_for(ticker: str) -> str:
            try:
                from agents.market_regime import get_trade_style_for_ticker
                style = get_trade_style_for_ticker(ticker)
            except Exception as e:
                log.debug(f"[{ticker}] sector trade_style lookup failed: {e}")
                style = None
            return style if style in STALE_WINDOW_DAYS_BY_STYLE else market_trade_style

        try:
            stops = db.get_all_position_stops()
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"flag_stale_positions: could not fetch positions: {e}")
            return []

        now = datetime.now(timezone.utc)
        flagged: list[dict] = []

        for stop_record in stops:
            ticker = stop_record["ticker"]
            bucket = stop_record.get("thesis_bucket")
            trade_style = _trade_style_for(ticker)
            windows = STALE_WINDOW_DAYS_BY_STYLE[trade_style]
            if bucket not in windows:
                continue
            pos = positions.get(ticker)
            opened_at = stop_record.get("opened_at")
            entry = stop_record.get("entry_price")
            if not pos or not opened_at or not entry:
                continue

            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            days_held = (now - opened_at).days
            if days_held < windows[bucket]:
                continue

            pct_move = abs((float(pos.current_price) - entry) / entry)
            if pct_move >= STALE_MOVE_THRESHOLD_PCT:
                continue

            flagged.append({
                "ticker": ticker, "days_held": days_held, "pct_move": pct_move,
                "trade_style": trade_style, "thesis_bucket": bucket,
            })
            log.info(f"[{ticker}] Flagged stale: held {days_held}d (window {windows[bucket]}d), "
                     f"moved only {pct_move:.1%} (threshold {STALE_MOVE_THRESHOLD_PCT:.0%})")

        return flagged


def exit_routes_for_bucket(bucket: str | None) -> dict:
    """Which automatic exit and discipline routes apply to a held position of
    this bucket, read from the same tables the checks use (audit
    A1031-1453 N3). Log-only: nothing acts on it."""
    targets = (SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE) or any(
        bucket in t for t in PROFIT_TARGET_PCT_BY_STYLE.values())
    return {
        "profit_target": bool(targets),
        "stale_flag": any(bucket in w for w in STALE_WINDOW_DAYS_BY_STYLE.values()),
        "swap_eligible": bucket in ("Accelerator", "Moonshot"),
        "trailing_stop": bucket in TRAILING_STOP_ACTIVATE,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    rm = RiskManager()

    pv = rm.portfolio_value()
    print(f"Portfolio value: ${pv:,.2f}")
    print(f"Halted: {rm.is_halted()}")
    print()

    sizing = rm.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On", price=130.0)
    print("NVDA sizing (Compounder, conviction=8, Risk-On):")
    for k, v in sizing.items():
        print(f"  {k}: {v}")
