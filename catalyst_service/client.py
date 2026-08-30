"""
Small read client for catalyst data.

Uses HTTP when CATALYST_SERVICE_URL is configured, otherwise reads the local
SQLite store directly. That keeps today's single-machine workflow moving while
preserving the service boundary for later.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # local-storage mode does not need HTTP dependencies
    requests = None

from . import storage

SERVICE_URL = os.getenv("CATALYST_SERVICE_URL", "").rstrip("/")


def query_events(**params) -> list[dict[str, Any]]:
    if SERVICE_URL and requests is not None:
        try:
            resp = requests.get(f"{SERVICE_URL}/catalysts", params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass
    return storage.query_events(**params)


def upcoming_events(
    ticker: str | None = None,
    days_ahead: int = 45,
    include_sector: bool = True,
    include_macro: bool = True,
) -> list[dict[str, Any]]:
    """
    The single combined view (2026-08-30, user request): everything that
    could plausibly move `ticker` over the horizon -- its own ticker-scope
    events, its sector's sector-scope events, and market-wide macro-scope
    events -- sorted by date. This is what context_engine.assemble_brief()
    calls so the analyst sees "NVDA earnings in 9 days, FOMC in 16 days,
    Info Tech sector earnings cluster in 20 days" as one list instead of
    three separate queries the caller has to know to make.

    Pass ticker=None to get sector-agnostic output (macro events only, plus
    whatever include_sector leaves out since there's no ticker to resolve a
    sector from).
    """
    start = date.today().isoformat()
    end = (date.today() + timedelta(days=days_ahead)).isoformat()
    events: list[dict[str, Any]] = []

    if ticker:
        events.extend(query_events(ticker=ticker, start=start, end=end, scope="ticker"))

    if ticker and include_sector:
        try:
            from ingestion.sp500_universe import SECTOR_ETF_MAP, get_sp500_constituents
            constituents = get_sp500_constituents()
            gics_sector = constituents.get(ticker.upper(), {}).get("gics_sector")
            etf = SECTOR_ETF_MAP.get(gics_sector) if gics_sector else None
            if etf:
                events.extend(query_events(ticker=etf, start=start, end=end, scope="sector"))
        except Exception:
            pass  # not an S&P constituent, or sector map unavailable -- skip silently

    if include_macro:
        events.extend(query_events(start=start, end=end, scope="macro"))

    events.sort(key=lambda e: e.get("event_date") or "")
    return events


def blocking_events_for_horizon(ticker: str, days_ahead: int = 45) -> list[dict[str, Any]]:
    start = date.today().isoformat()
    end = (date.today() + timedelta(days=days_ahead)).isoformat()
    if SERVICE_URL:
        # start=today so a stale past-dated event can't block trades forever
        events = query_events(ticker=ticker, start=start, end=end)
        return [
            e for e in events
            if (e.get("action_flags") or {}).get("halt_auto_execution")
            or e.get("confidence") in {"conflicting", "rumored"}
        ]
    return storage.blocking_events(ticker, end=end, start=start)


def nearest_event(ticker: str, days_ahead: int = 120) -> dict[str, Any] | None:
    events = query_events(
        ticker=ticker,
        start=date.today().isoformat(),
        end=(date.today() + timedelta(days=days_ahead)).isoformat(),
    )
    return events[0] if events else None
