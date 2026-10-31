"""
Polygon.io market data — real-time quotes, trades, and aggregates.
Significantly better than Alpaca's free data:
  - True real-time (not 15-min delayed)
  - Full L2 order book depth
  - Options chain data
  - Short interest
  - Institutional ownership

Requires POLYGON_API_KEY in .env
Sign up at polygon.io (free tier available, paid for real-time)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

API_KEY  = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"

from config import SYMBOLS

# Endpoint path templates (ticker stripped) that have already 403'd with
# NOT_AUTHORIZED this process's lifetime -- found 2026-09-02: the current
# Polygon plan doesn't include real-time snapshot data ("You are not
# entitled to this data. Please upgrade your plan"), a per-ENDPOINT plan
# restriction, not a per-ticker or transient failure, so every single
# get_snapshot() call across the whole day logged an identical WARNING.
# Since it's plan-tier (not the key itself -- /v3/reference/* works fine
# on the same key), retrying it is pointless until the plan changes; this
# just downgrades the noise to DEBUG after the first hit per endpoint
# template, it doesn't stop calling it (a plan upgrade should start
# working again without a restart).
_ENTITLEMENT_BLOCKED: set[str] = set()


def _safe_err(e: Exception) -> str:
    """Type and HTTP status only: the exception text carries the request URL,
    apiKey included (audit A1031-0253-5)."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    return type(e).__name__ + (f" (HTTP {status})" if status else "")


def _get(endpoint: str, params: dict | None = None) -> dict:
    if not API_KEY:
        return {}
    # Strip the ticker (last path segment) to get a template key shared
    # across all tickers hitting the same restricted endpoint.
    endpoint_template = endpoint.rsplit("/", 1)[0]
    try:
        resp = requests.get(
            f"{BASE_URL}{endpoint}",
            params={"apiKey": API_KEY, **(params or {})},
            timeout=10,
        )
        resp.raise_for_status()
        _ENTITLEMENT_BLOCKED.discard(endpoint_template)  # started working again (plan upgraded)
        return resp.json()
    except requests.exceptions.HTTPError as e:
        is_entitlement = (
            e.response is not None and e.response.status_code == 403
            and "NOT_AUTHORIZED" in e.response.text
        )
        if is_entitlement:
            if endpoint_template not in _ENTITLEMENT_BLOCKED:
                _ENTITLEMENT_BLOCKED.add(endpoint_template)
                log.warning(
                    f"Polygon plan doesn't include {endpoint_template} data "
                    f"(403 NOT_AUTHORIZED) — suppressing repeat warnings for "
                    f"this endpoint until the plan changes or the process "
                    f"restarts. Upgrade at polygon.io/pricing to restore it."
                )
            else:
                log.debug(f"Polygon fetch failed ({endpoint}): {_safe_err(e)}")
        else:
            log.warning(f"Polygon fetch failed ({endpoint}): {_safe_err(e)}")
        return {}
    except Exception as e:
        log.warning(f"Polygon fetch failed ({endpoint}): {_safe_err(e)}")
        return {}


def get_snapshot(ticker: str) -> dict:
    """
    Returns full market snapshot for a ticker:
    current price, day's range, volume, VWAP, after-hours price.
    """
    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    ticker_data = data.get("ticker", {})
    day  = ticker_data.get("day", {})
    prev = ticker_data.get("prevDay", {})

    # after_hours previously read lastQuote.P, which is the current ASK
    # price, not a trade — mislabeling a quote as an after-hours print.
    # The last TRADE price is the correct value to show as "after hours"
    # when the market is closed; during RTH the concept doesn't apply.
    last_trade_price = ticker_data.get("lastTrade", {}).get("p")
    after_hours = None
    try:
        from market_data import _is_market_hours
        if not _is_market_hours():
            after_hours = last_trade_price
    except Exception:
        pass

    return {
        "ticker":        ticker,
        "price":         last_trade_price,
        "vwap":          day.get("vw"),
        "volume":        day.get("v"),
        "open":          day.get("o"),
        "high":          day.get("h"),
        "low":           day.get("l"),
        "prev_close":    prev.get("c"),
        "change_pct":    ticker_data.get("todaysChangePerc"),
        "after_hours":   after_hours,
    }


def get_short_interest(ticker: str) -> dict:
    """
    Short interest is NOT available from this endpoint — the previous
    implementation hit /v2/reference/financials/{ticker}, which returns
    fundamentals data, not short interest, and silently produced unusable
    results. Polygon's paid tiers don't expose short interest either; FINRA
    publishes it for free on a ~2-week lag via a fixed-width text file, but
    without a validated working URL/format for that feed this returns an
    explicit "unavailable" stub rather than guessing at a wrong endpoint.
    """
    return {
        "ticker": ticker,
        "available": False,
        "reason": "No validated free short-interest source configured "
                  "(FINRA biweekly file integration not yet implemented).",
    }


def get_rsi(ticker: str, window: int = 14) -> float | None:
    """Calculates RSI from Polygon's aggregate bars."""
    data = _get(
        f"/v1/indicators/rsi/{ticker}",
        {"timespan": "day", "window": window, "series_type": "close", "limit": 1},
    )
    results = data.get("results", {}).get("values", [])
    return float(results[0]["value"]) if results else None


def get_macd(ticker: str) -> dict | None:
    """Returns MACD signal for trend confirmation."""
    data = _get(
        f"/v1/indicators/macd/{ticker}",
        {"timespan": "day", "series_type": "close", "limit": 1},
    )
    results = data.get("results", {}).get("values", [])
    if not results:
        return None
    r = results[0]
    return {
        "value":    r.get("value"),
        "signal":   r.get("signal"),
        "histogram": r.get("histogram"),
        "trend":    "BULLISH" if (r.get("value", 0) or 0) > (r.get("signal", 0) or 0) else "BEARISH",
    }


def get_technicals_summary(ticker: str) -> str:
    """
    Returns a plain-English technical summary suitable for model context.
    """
    if not API_KEY:
        return f"{ticker}: Polygon API key not configured."

    snap = get_snapshot(ticker)
    rsi  = get_rsi(ticker)
    macd = get_macd(ticker)

    lines = [f"=== {ticker} TECHNICALS ==="]
    if snap.get("price"):
        lines.append(f"Price: ${snap['price']:.2f}  ({snap.get('change_pct', 0):+.2f}% today)")
        lines.append(f"VWAP: ${snap['vwap']:.2f}" if snap.get("vwap") else "")
        lines.append(f"Volume: {snap['volume']:,.0f}" if snap.get("volume") else "")

    if rsi is not None:
        label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "neutral"
        lines.append(f"RSI(14): {rsi:.1f} — {label}")

    if macd:
        lines.append(f"MACD: {macd['trend']} (histogram: {macd['histogram']:.3f})")

    return "\n".join(l for l in lines if l)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not API_KEY:
        print("POLYGON_API_KEY not set in .env — add it to activate technical data.")
    else:
        for sym in SYMBOLS[:5]:
            print(get_technicals_summary(sym))
            print()
