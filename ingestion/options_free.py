"""
Free options-flow proxy via yfinance option chains.

Unusual Whales (~$50/mo) gives real sweep/premium data; this gives a much
weaker but free substitute — put/call volume ratio and open-interest deltas
from the nearest-dated chain — so the scanner and regime signals aren't
silently zeroed out just because UNUSUAL_WHALES_API_KEY is a placeholder.
Per the improvement roadmap: only pay for Unusual Whales once the signal
calibration table shows this component actually predicts outcomes.

Same return shape as ingestion.options_flow so callers can't tell which
backend answered — see options_flow.py's fallback dispatch.
"""

import logging
import math
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import yfinance as yf

log = logging.getLogger(__name__)


def _nearest_chain(ticker: str, meta: dict | None = None):
    """meta, when given, is filled with the expiry used and the chain's
    underlying quote (part of the same option_chain response, so no extra
    request)."""
    t = yf.Ticker(ticker)
    expirations = t.options
    if not expirations:
        return None, None
    # Was two separate option_chain() calls for the same expiry (one for
    # .calls, one for .puts) — each is its own network request, doubling
    # yfinance load across 75 tickers every 5 minutes. Both come from one
    # call.
    chain = t.option_chain(expirations[0])
    if meta is not None:
        meta["expiry"] = expirations[0]
        meta["underlying"] = getattr(chain, "underlying", None)
    return chain.calls, chain.puts


def _finite_pos(v):
    """v as a float when it is a finite number above zero, else None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _leg_price(row) -> tuple[float | None, str | None]:
    """(price, source) of one option leg: the bid/ask mid when both sides
    are quoted and not crossed, else lastPrice, else (None, None)."""
    bid, ask = _finite_pos(row.get("bid")), _finite_pos(row.get("ask"))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2, "mid"
    last = _finite_pos(row.get("lastPrice"))
    return (last, "last") if last is not None else (None, None)


def _implied_move(calls, puts, underlying_quote, expiry, today) -> dict | None:
    """Log-only: the nearest-expiry ATM straddle from a chain already
    downloaded. ATM is the strike quoted on both sides nearest the
    underlying's regularMarketPrice (ties to the lower strike).
    implied_move_pct = straddle / underlying * 100. None when the
    underlying price, a common strike or either leg's price is missing.
    Never raises."""
    try:
        if calls is None or puts is None or calls.empty or puts.empty:
            return None
        u = _finite_pos((underlying_quote or {}).get("regularMarketPrice"))
        if u is None:
            return None
        common = set(calls["strike"].dropna()) & set(puts["strike"].dropna())
        if not common:
            return None
        strike = min(common, key=lambda k: (abs(float(k) - u), float(k)))
        c = calls[calls["strike"] == strike].iloc[0].to_dict()
        p = puts[puts["strike"] == strike].iloc[0].to_dict()
        cp, cs = _leg_price(c)
        pp, ps = _leg_price(p)
        if cp is None or pp is None:
            return None
        straddle = cp + pp
        exp = date.fromisoformat(str(expiry)[:10]) if expiry else None

        def iv(r):
            try:
                f = float(r.get("impliedVolatility"))
                return round(f, 6) if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None
        return {
            "implied_move_pct": round(straddle / u * 100, 4),
            "straddle": round(straddle, 4),
            "price_source": cs if cs == ps else "mixed",
            "call_price_source": cs, "put_price_source": ps,
            "underlying": round(u, 4), "atm_strike": float(strike),
            "expiry": exp.isoformat() if exp else None,
            "dte": (exp - today).days if exp else None,
            "call_iv": iv(c), "put_iv": iv(p),
        }
    except Exception as e:
        log.debug(f"implied move calc failed: {e}")
        return None


def get_flow_summary(ticker: str, implied_sink: dict | None = None) -> dict:
    """
    Free proxy for options_flow.get_flow_summary(). Uses call/put VOLUME
    (not premium — that data isn't reliably available for free) from the
    nearest expiry as a sentiment proxy.

    implied_sink (log-only shadow): when a dict is passed, the ATM straddle
    implied move from the same chain is stored in it under the ticker. The
    return value is the same either way.
    """
    meta = {} if implied_sink is not None else None
    try:
        calls, puts = _nearest_chain(ticker, meta) if meta is not None else _nearest_chain(ticker)
    except Exception as e:
        log.debug(f"[{ticker}] yfinance option chain fetch failed: {e}")
        calls, puts = None, None

    if meta is not None:
        try:
            today = datetime.now(ZoneInfo("America/New_York")).date()
            mv = _implied_move(calls, puts, meta.get("underlying"), meta.get("expiry"), today)
            if mv is not None:
                implied_sink[ticker] = mv
        except Exception as e:
            log.debug(f"[{ticker}] implied move shadow failed: {e}")

    if calls is None or puts is None or (calls.empty and puts.empty):
        return {"ticker": ticker, "unusual": False,
                "summary": "No free options data available (yfinance chain empty)."}

    call_vol = float(calls["volume"].fillna(0).sum())
    put_vol  = float(puts["volume"].fillna(0).sum())
    total_vol = call_vol + put_vol
    call_ratio = call_vol / total_vol if total_vol > 0 else 0.5

    # OI concentration as a rough "unusual" proxy: any single strike holding
    # a large share of total OI suggests a concentrated bet.
    call_oi = calls["openInterest"].fillna(0)
    put_oi  = puts["openInterest"].fillna(0)
    total_oi = float(call_oi.sum() + put_oi.sum())
    max_strike_oi = float(max(call_oi.max() if len(call_oi) else 0,
                              put_oi.max() if len(put_oi) else 0))
    concentrated = total_oi > 0 and (max_strike_oi / total_oi) > 0.25

    sentiment = "BULLISH" if call_ratio > 0.65 else "BEARISH" if call_ratio < 0.35 else "NEUTRAL"
    unusual = concentrated or call_ratio > 0.70 or call_ratio < 0.30

    summary_text = (
        f"{ticker} options (free proxy): {sentiment} "
        f"(call/put volume ratio {call_ratio:.0%}). "
        f"Total contract volume: {total_vol:,.0f}."
    )
    if concentrated:
        summary_text += " Notable open-interest concentration at a single strike."

    return {
        "ticker":        ticker,
        "unusual":       unusual,
        "sentiment":     sentiment,
        "call_ratio":    round(call_ratio, 3),
        "total_premium": None,  # not available for free — do not fabricate
        "unusual_count": 1 if unusual else 0,
        "summary":       summary_text,
    }


def get_all_flow_summaries(implied_sink: dict | None = None) -> dict[str, dict]:
    """implied_sink: see get_flow_summary (log-only shadow)."""
    from knowledge_loader import get_all_tickers
    summaries = {}
    for ticker in get_all_tickers():
        summaries[ticker] = (get_flow_summary(ticker, implied_sink) if implied_sink is not None
                             else get_flow_summary(ticker))
        time.sleep(0.2)
    return summaries


def get_darkpool_activity(ticker: str) -> dict:
    """No free reliable dark-pool data source — explicit stub, not a guess."""
    return {"ticker": ticker, "has_significant_print": False, "total_value": 0,
            "largest_print": 0, "summary": "Dark pool data unavailable (no free source)."}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from knowledge_loader import get_all_tickers
    for t in get_all_tickers()[:5]:
        print(get_flow_summary(t)["summary"])
