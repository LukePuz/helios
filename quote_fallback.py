"""Yahoo 1-minute bars as a second quote source for a BUY whose IBKR quote is
stale. RECORD ONLY: nothing here licenses an order. tasks.
stale_quote_fallback_shadow logs what a fallback would have said (price,
drift against the sizing price, the opening-gap check on these bars), so
whether to use one, and the age bound N for its bar, can be decided from
data. No key here applies N.

Kept out of market_data on purpose: that module's whole source is hashed
into every verdict's brief_template_full_sha (agents/analyst._BRIEF_MODULES),
and this adds nothing to the brief.
"""
import logging
from datetime import datetime, time, timezone

log = logging.getLogger(__name__)

SOURCE = "yahoo_1m"
_RTH_OPEN, _RTH_CLOSE = time(9, 30), time(16, 0)
# execution.execute_approved's price-drift abort, a literal there: a BUY
# whose |price - price_at_sizing| / price_at_sizing is above this aborts.
DRIFT_ABORT_FRAC = 0.08


def _et():
    from zoneinfo import ZoneInfo
    return ZoneInfo("America/New_York")


def _in_rth(ts) -> bool:
    """A bar that STARTED 09:30-16:00 ET (an early close's last hours read
    as in RTH: known limit)."""
    return _RTH_OPEN <= ts.astimezone(_et()).time() < _RTH_CLOSE


def minute_bars(ticker: str):
    """Today's Yahoo 1-minute bars, pre- and post-market rows included."""
    import yfinance as yf
    return yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True)


def last_minute_close(df, at=None) -> tuple[float | None, str | None]:
    """The last close in df whose bar started at or before `at` (all bars
    when None) and that bar's START time (ISO); (None, None) when none."""
    if df is not None and not df.empty and at is not None:
        df = df[df.index <= at]   # a bar that arrived after `at` doesn't count
    if df is None or df.empty:
        return None, None
    return float(df["Close"].iloc[-1]), df.index[-1].isoformat()


def get_fallback_quote(ticker: str, at=None, bars=None) -> dict:
    """The quote a Yahoo fallback would have used at `at` (now when None):
    {"price", "bar_at" (bar start, ISO), "age_min" (bar start to `at`),
    "in_rth" (see _in_rth), "source", "error"}. bars: a frame from
    minute_bars, so one fetch can serve the gap check too. No freshness rule
    is applied. Never raises."""
    out = {"price": None, "bar_at": None, "age_min": None, "in_rth": None,
           "source": SOURCE, "error": None}
    try:
        ref = at or datetime.now(timezone.utc)
        price, bar_at = last_minute_close(minute_bars(ticker) if bars is None else bars, ref)
        bar = datetime.fromisoformat(bar_at) if bar_at else None
        out.update(price=price, bar_at=bar_at,
                   age_min=round((ref - bar).total_seconds() / 60, 1) if bar else None,
                   in_rth=_in_rth(bar) if bar else None)
    except Exception as e:
        out["error"] = type(e).__name__
    return out


def drift_record(price, price_at_sizing) -> dict:
    """drift_pct (signed, percent) of price against the sizing price, and
    would_pass_drift under execution's abort (|drift| at most
    DRIFT_ABORT_FRAC passes); both None when either price is unknown.
    Never raises."""
    try:
        p, s = float(price), float(price_at_sizing)
        if not (p > 0 and s > 0):
            raise ValueError("non-positive price")
    except (TypeError, ValueError):
        return {"drift_pct": None, "would_pass_drift": None}
    frac = (p - s) / s
    return {"drift_pct": round(frac * 100, 4), "would_pass_drift": abs(frac) <= DRIFT_ABORT_FRAC}


def daily_bars(ticker: str):
    """Yahoo daily bars for the last 5 sessions (today's partial one too)."""
    import yfinance as yf
    return yf.Ticker(ticker).history(period="5d", interval="1d")


def fallback_gap_status(ticker: str, at=None, bars=None, daily=None, today=None) -> str:
    """market_data.check_gap_confirmation's rule run on Yahoo bars instead of
    IBKR's: today's regular-hours 1-minute bars that started at or before
    `at`, and the prior close from the last Yahoo daily bar before today.
    Same thresholds and labels (no_gap, pending, confirmed, failed,
    unreadable); "unreadable" also when Yahoo has no regular-hours bar yet.
    Record only. Returns "error:<class>" instead of raising."""
    try:
        from config import GAP_CONFIRMATION_THRESHOLD_PCT, GAP_FAIL_RETRACEMENT_PCT
        ref = at or datetime.now(timezone.utc)
        today = today or ref.astimezone(_et()).date()
        df = minute_bars(ticker) if bars is None else bars
        if df is None or df.empty:
            return "unreadable"
        rth = df[[ts <= ref and _in_rth(ts) and ts.astimezone(_et()).date() == today
                  for ts in df.index]]
        if rth.empty:
            return "unreadable"
        d = daily_bars(ticker) if daily is None else daily
        if d is None or d.empty:
            return "unreadable"
        before = d[[ts.date() < today for ts in d.index]]
        if before.empty:
            return "unreadable"
        prior_close = float(before["Close"].iloc[-1])
        if not prior_close:
            return "unreadable"
        open_price = float(rth["Open"].iloc[0])
        current_price = float(rth["Close"].iloc[-1])
        gap_pct = (open_price - prior_close) / prior_close
        if abs(gap_pct) < GAP_CONFIRMATION_THRESHOLD_PCT:
            return "no_gap"
        gap_size = open_price - prior_close
        retracement = ((open_price - current_price) / gap_size) if gap_size else 0.0
        if retracement >= GAP_FAIL_RETRACEMENT_PCT:
            return "failed"
        return "pending" if len(rth) < 10 else "confirmed"
    except Exception as e:
        return f"error:{type(e).__name__}"
