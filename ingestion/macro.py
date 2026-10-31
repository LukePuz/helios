"""
FRED (Federal Reserve Economic Data) macro ingestion.
Fetches key economic indicators that provide market regime context
for the 72B model's reasoning.

API key required — sign up free at fred.stlouisfed.org
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs and human-readable names for our model context
SERIES: dict[str, str] = {
    "FEDFUNDS":  "Fed Funds Rate",
    "DGS10":     "10-Year Treasury Yield",
    "DGS2":      "2-Year Treasury Yield",
    "T10Y2Y":    "Yield Curve (10Y-2Y spread) — negative = recession signal",
    "VIXCLS":    "VIX Volatility Index",
    "CPIAUCSL":  "CPI Index (level)",   # overridden to "CPI Inflation YoY %" below when computable
    "UNRATE":    "Unemployment Rate",
    "DCOILWTICO":"WTI Crude Oil Price",
    "DEXUSEU":   "USD/EUR Exchange Rate",
    "NASDAQCOM": "NASDAQ Composite",
}


def _fetch_series(series_id: str, lookback_days: int = 30, limit: int = 5) -> list[dict]:
    """Returns recent observations for a FRED series."""
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — macro data unavailable.")
        return []

    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    params = {
        "series_id":       series_id,
        "api_key":         FRED_API_KEY,
        "file_type":       "json",
        "observation_start": start,
        "sort_order":      "desc",
        "limit":           limit,
    }
    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception as e:
        # Not the exception text: it carries the request URL, api_key included
        # (audit A1031-0253-5).
        status = getattr(getattr(e, "response", None), "status_code", None)
        log.warning(f"FRED fetch failed for {series_id}: {type(e).__name__}"
                    + (f" (HTTP {status})" if status else ""))
        return []


def _get_cpi_yoy() -> dict[str, str | float | None]:
    """CPIAUCSL is a raw index level (~315), not a rate — the old code
    labeled it "CPI Inflation (YoY)" and handed the LLM a nonsense number
    like "315" as an inflation rate. Compute the true YoY % from the latest
    observation vs. the one ~12 months prior; if that comparison isn't
    available, fall back to reporting the honest index level instead of a
    mislabeled rate."""
    obs = _fetch_series("CPIAUCSL", lookback_days=400, limit=14)
    obs = [o for o in obs if o.get("value") not in (".", None, "")]
    if not obs:
        return {"label": "CPI Index (level)", "value": None, "date": None}

    obs.sort(key=lambda o: o["date"], reverse=True)
    latest = obs[0]
    try:
        latest_val  = float(latest["value"])
        latest_date = datetime.strptime(latest["date"], "%Y-%m-%d")
    except (ValueError, KeyError):
        return {"label": "CPI Index (level)", "value": None, "date": None}

    year_ago_target = latest_date - timedelta(days=365)
    prior = min(
        obs,
        key=lambda o: abs(datetime.strptime(o["date"], "%Y-%m-%d") - year_ago_target),
        default=None,
    )
    if prior:
        prior_date = datetime.strptime(prior["date"], "%Y-%m-%d")
        # Reject if the "closest" observation isn't actually close to a
        # year ago (e.g. history got trimmed) — a bad YoY comparison is
        # worse than none.
        if abs((prior_date - year_ago_target).days) <= 45:
            try:
                prior_val = float(prior["value"])
                yoy = round((latest_val / prior_val - 1) * 100, 2)
                return {"label": "CPI Inflation YoY %", "value": yoy, "date": latest["date"]}
            except (ValueError, ZeroDivisionError):
                pass

    return {"label": "CPI Index (level)", "value": latest_val, "date": latest["date"]}


# Series that FRED only updates monthly -- a 10-day lookback (fine for the
# daily series below) will reliably find NOTHING for these, since the most
# recently published reading is routinely 30-55 days old by the time the
# next one posts. Found 2026-07-26: get_macro_snapshot() used the same
# 10-day window for every series, so FEDFUNDS/UNRATE silently returned
# "unavailable" essentially always (confirmed live: FEDFUNDS's actual latest
# reading was dated 2026-06-01, 55 days before this was caught) -- which
# defaulted market_regime's liquidity signal to Neutral on every assessment,
# not just occasionally. _get_cpi_yoy already used a 400-day window for the
# same reason; these two just got missed.
_MONTHLY_SERIES = {"FEDFUNDS", "UNRATE"}
_MONTHLY_LOOKBACK_DAYS = 60


def get_macro_snapshot() -> dict[str, str | float | None]:
    """
    Returns the latest reading for every macro series.
    Used by context_engine.py to populate the situation brief.
    """
    snapshot: dict[str, str | float | None] = {}
    for series_id, label in SERIES.items():
        if series_id == "CPIAUCSL":
            snapshot[series_id] = _get_cpi_yoy()
            continue
        lookback = _MONTHLY_LOOKBACK_DAYS if series_id in _MONTHLY_SERIES else 10
        obs = _fetch_series(series_id, lookback_days=lookback)
        # Find most recent non-null observation
        latest = next((o for o in obs if o.get("value") not in (".", None, "")), None)
        if latest:
            try:
                snapshot[series_id] = {
                    "label": label,
                    "value": float(latest["value"]),
                    "date":  latest["date"],
                }
            except (ValueError, KeyError):
                snapshot[series_id] = {"label": label, "value": None, "date": None}
        else:
            snapshot[series_id] = {"label": label, "value": None, "date": None}

    return snapshot


def get_macro_summary_text() -> str:
    """
    Returns a plain-English macro summary suitable for inclusion
    directly in a model prompt.
    """
    snap = get_macro_snapshot()
    lines = ["=== CURRENT MACRO ENVIRONMENT ==="]
    for series_id, data in snap.items():
        if data["value"] is not None:
            lines.append(f"{data['label']}: {data['value']} (as of {data['date']})")
        else:
            lines.append(f"{data['label']}: data unavailable")

    # Derived signals
    t10 = snap.get("DGS10", {}).get("value")
    t2  = snap.get("DGS2",  {}).get("value")
    if t10 and t2:
        spread = round(t10 - t2, 3)
        if spread < 0:
            lines.append(f"\n⚠️  YIELD CURVE INVERTED (spread: {spread}%) — historically precedes recession")
        else:
            lines.append(f"\nYield curve spread: +{spread}% (normal)")

    vix = snap.get("VIXCLS", {}).get("value")
    if vix:
        if vix > 30:
            lines.append(f"⚠️  VIX at {vix} — HIGH FEAR regime, reduce position sizing")
        elif vix > 20:
            lines.append(f"VIX at {vix} — elevated uncertainty")
        else:
            lines.append(f"VIX at {vix} — low volatility, risk-on environment")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_macro_summary_text())
