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
    "CPIAUCSL":  "CPI Inflation (YoY)",
    "UNRATE":    "Unemployment Rate",
    "DCOILWTICO":"WTI Crude Oil Price",
    "DEXUSEU":   "USD/EUR Exchange Rate",
    "NASDAQCOM": "NASDAQ Composite",
}


def _fetch_series(series_id: str, lookback_days: int = 30) -> list[dict]:
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
        "limit":           5,
    }
    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception as e:
        log.warning(f"FRED fetch failed for {series_id}: {e}")
        return []


def get_macro_snapshot() -> dict[str, str | float | None]:
    """
    Returns the latest reading for every macro series.
    Used by context_engine.py to populate the situation brief.
    """
    snapshot: dict[str, str | float | None] = {}
    for series_id, label in SERIES.items():
        obs = _fetch_series(series_id, lookback_days=10)
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
