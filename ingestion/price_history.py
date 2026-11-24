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
from fractions import Fraction

import yfinance as yf

log = logging.getLogger(__name__)

BENCHMARK_TICKERS = ["SPY", "QQQ"]


def _fetch_bars(ticker: str, period: str = "1y") -> list[dict]:
    """Daily bars from yfinance. An empty result ("possibly delisted") is
    usually a transient Yahoo failure, so it's retried with backoff; for the
    benchmarks, Alpaca is the fallback. SPY failed this way on 10-15, 10-26
    and 10-27 (2026), and SPY is the alpha benchmark."""
    bars = []
    for attempt in range(3):
        bars = _fetch_bars_yf(ticker, period)
        if bars:
            return bars
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    if ticker in BENCHMARK_TICKERS:
        try:
            from agents.market_regime import _fetch_bars as _alpaca_bars
            bars = _alpaca_bars(ticker, days=260)
            if bars:
                log.warning(f"[{ticker}] yfinance returned no bars; used Alpaca ({len(bars)} bars)")
        except Exception as e:
            log.warning(f"[{ticker}] Alpaca fallback failed: {e}")
    return bars


def _fetch_bars_yf(ticker: str, period: str) -> list[dict]:
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


# Stale-head rescale (_scale_check / _store_bars, config
# PRICE_CACHE_RESCALE_ENABLED). k = median(fetched close / stored close) over
# the dates both have. |k - 1| <= _RESCALE_MIN_GAP: nothing (dividend
# adjustments are far smaller). Beyond it, a rescale needs both: k within
# _RESCALE_FACTOR_TOL (relative) of f or 1/f for f in _RESCALE_FACTORS, and at
# least _RESCALE_UNIFORM_SHARE of the shared dates' ratios within
# _RESCALE_FACTOR_TOL of k (a partly adjusted series, e.g. 60% of dates at 1/8
# and 40% at 1, is not one basis change). Otherwise an "unexplained scale gap"
# activity row, no rescale. Fewer shared dates than max(_RESCALE_MIN_OVERLAP,
# _RESCALE_MIN_OVERLAP_SHARE of the fetched bars): nothing.
_RESCALE_MIN_GAP = 0.10
_RESCALE_FACTORS = tuple(Fraction(f) for f in (2, 3, 4, 5, 8, 10, 15, 20, 50)) + (
    Fraction(3, 2), Fraction(4, 3), Fraction(5, 4))
_RESCALE_FACTOR_TOL = 0.02
_RESCALE_UNIFORM_SHARE = 0.90
_RESCALE_MIN_OVERLAP = 20
_RESCALE_MIN_OVERLAP_SHARE = 0.5


def _scale_check(ticker: str, bars: list[dict]) -> dict:
    """Fetched bars against the stored rows on the dates both have (one read
    of the cache). Returns:
      overlap        shared dates
      k              median fetched/stored close (None under the minimum overlap)
      uniform_share  share of shared-date ratios within _RESCALE_FACTOR_TOL of k
      factor         the split factor k matches: "1/8" for an 8-for-1 split,
                     "2" for a 1-for-2 reverse split, "2/3" for 3-for-2; None
      exact          that factor as a number (None without one)
      first_fetched, last_fetched   the oldest and newest fetched dates
      check_date, check_close       the oldest shared date and its stored close,
                     re-read under the write lock (db.rescale_and_upsert_price_bars)
      action         "none", "rescale", "unexplained" or "too_little_overlap"
    Raises when the cache can't be read."""
    import math
    import statistics
    import db
    fetched = {}
    for b in bars:
        try:
            c = float(b["close"])
        except (TypeError, ValueError, KeyError):
            continue
        if c > 0:
            fetched[str(b["date"])[:10]] = c
    out = {"overlap": 0, "k": None, "uniform_share": None, "factor": None, "exact": None,
           "first_fetched": min(fetched) if fetched else None,
           "last_fetched": max(fetched) if fetched else None,
           "check_date": None, "check_close": None, "action": "none"}
    if not fetched:
        return out
    stored = db.get_price_history(ticker, days=len(bars) + 10, strict=True)
    shared = []
    for r in stored or []:
        d = str(r.get("date"))[:10]
        try:
            sc = float(r.get("close"))
        except (TypeError, ValueError):
            continue
        if d in fetched and sc > 0:
            shared.append((d, sc, fetched[d] / sc))
    out["overlap"] = len(shared)
    need = max(_RESCALE_MIN_OVERLAP, math.ceil(_RESCALE_MIN_OVERLAP_SHARE * len(fetched)))
    if len(shared) < need:
        out["action"] = "too_little_overlap"
        return out
    shared.sort()
    out["check_date"], out["check_close"] = shared[0][0], shared[0][1]
    ratios = [r for _, _, r in shared]
    k = statistics.median(ratios)
    out["k"] = round(k, 6)
    share = sum(1 for r in ratios if abs(r / k - 1) <= _RESCALE_FACTOR_TOL) / len(ratios)
    out["uniform_share"] = round(share, 4)
    if abs(k - 1) <= _RESCALE_MIN_GAP:
        return out
    for f in _RESCALE_FACTORS:
        for target in (1 / f, f):
            if abs(k / float(target) - 1) <= _RESCALE_FACTOR_TOL:
                out["factor"], out["exact"] = str(target), float(target)
    out["action"] = ("rescale" if out["factor"] and share >= _RESCALE_UNIFORM_SHARE
                     else "unexplained")
    return out


def _store_bars(ticker: str, bars: list[dict]) -> int:
    """db.upsert_price_bars, after _scale_check. A uniform split-factor gap
    with PRICE_CACHE_RESCALE_ENABLED rescales every stored row this upsert
    won't overwrite (dated before the newest fetched date and not a fetched
    date): prices by k, volume by the exact factor. That and the upsert are
    one transaction (db.rescale_and_upsert_price_bars), with one
    price_cache_rescale activity row (ticker, k, factor, rows, date range). If
    that write fails or raises, nothing changes and the upsert is skipped this
    run (an upsert alone would leave the old rows behind with no gap left to
    detect), with a price_cache_rescale_failed row. Switched off, a split gap
    gets a price_cache_rescale row with applied false and the plain upsert. A
    gap that is no split factor, or not uniform, gets a price_cache_scale_gap
    row ("unexplained scale gap") and the plain upsert. A check that raises:
    the plain upsert. Returns the count upserted, as upsert_price_bars. Never
    raises."""
    import db
    try:
        chk = _scale_check(ticker, bars)
    except Exception as e:
        log.warning(f"[{ticker}] price cache scale check failed ({type(e).__name__}: {e}); plain upsert")
        return db.upsert_price_bars(ticker, bars)
    rec = {k: v for k, v in chk.items() if k != "check_close"}

    def note(stage, msg, extra):
        try:
            db.log_activity(stage, ticker, msg, "alert", extra=extra)
        except Exception as e:
            log.warning(f"[{ticker}] {stage} activity row not written: {type(e).__name__}: {e}")

    try:
        if chk["action"] == "rescale":
            try:
                from config import PRICE_CACHE_RESCALE_ENABLED as enabled
            except ImportError:
                enabled = False
            if enabled:
                try:
                    res = db.rescale_and_upsert_price_bars(
                        ticker, chk["k"], bars, volume_k=chk["exact"],
                        check=(chk["check_date"], chk["check_close"]))
                except Exception as e:
                    log.error(f"[{ticker}] price cache rescale raised ({type(e).__name__}: {e})")
                    res = None
                if res is None:
                    log.error(f"[{ticker}] price cache rescale by {chk['k']} failed; upsert skipped this run")
                    note("price_cache_rescale_failed",
                         f"{ticker}: split-factor gap {chk['factor']} found, rescale write failed; "
                         f"nothing changed, retried next run", rec)
                    return 0
                log.warning(f"[{ticker}] price cache: {res.get('rows')} rows not in the fetch rescaled "
                            f"by {chk['k']} ({chk['factor']})")
                note("price_cache_rescale",
                     f"{ticker}: {res.get('rows')} cached rows not in the fetched window rescaled by "
                     f"{chk['k']} ({chk['factor']} split basis)",
                     {**rec, "applied": True, "rows": res.get("rows"), "rows_first": res.get("first"),
                      "rows_last": res.get("last")})
                return res.get("upserted") or 0
            note("price_cache_rescale",
                 f"{ticker}: split-factor gap {chk['factor']} (k {chk['k']}); rescale off, record only",
                 {**rec, "applied": False})
        elif chk["action"] == "unexplained":
            log.warning(f"[{ticker}] price cache: unexplained scale gap k={chk['k']}; not rescaled")
            note("price_cache_scale_gap",
                 f"{ticker}: unexplained scale gap, fetched/stored {chk['k']} over {chk['overlap']} "
                 f"shared dates ({chk['uniform_share']} of them within 2% of it); nothing rescaled, "
                 f"needs a look", rec)
    except Exception as e:
        log.warning(f"[{ticker}] price cache rescale step failed ({type(e).__name__}: {e}); plain upsert")
    return db.upsert_price_bars(ticker, bars)


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
            n = _store_bars(ticker, bars)
            if n:
                updated += 1
        time.sleep(0.2)  # be polite to Yahoo's unofficial endpoint
    return updated


def _last_weekday_before(today):
    """The last weekday strictly before today (holidays not excluded)."""
    from datetime import timedelta
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def refresh_sp500_price_history(skip_watchlist: bool = True, max_age_days: int | None = None,
                                today=None) -> int:
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

    # Skip tickers already current, so a run killed partway (the 19:00 stop
    # on 10-24 left 423 of 551 at 10-16) resumes instead of restarting the
    # ~2.6 h pass. max_age_days set: only tickers older than that (the
    # weekday catch-up), so it never becomes a daily full refresh.
    from datetime import date, timedelta
    today = today or date.today()
    cutoff = (today - timedelta(days=max_age_days) if max_age_days is not None
              else _last_weekday_before(today))
    newest = db.get_newest_bar_dates(tickers)
    due = [t for t in tickers if newest.get(t) is None or newest[t] < cutoff]
    log.info(f"S&P 500 price history: {len(due)} of {len(tickers)} tickers due (bars before {cutoff})")

    # Counted only when the newest bar moved forward: the upsert also counts
    # re-written old bars, so a ticker Yahoo still serves stale would count
    # every day and keep the heartbeat green (professor M141).
    updated = 0
    for ticker in due:
        bars = _fetch_bars(ticker)
        if bars and _store_bars(ticker, bars):
            latest = max(str(b["date"])[:10] for b in bars)
            prev = newest.get(ticker)
            if prev is None or latest > str(prev)[:10]:
                updated += 1
        time.sleep(0.2)  # be polite to Yahoo's unofficial endpoint
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Refreshed price history: {refresh_all_price_history()} tickers")
