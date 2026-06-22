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


def blocking_events_for_horizon(ticker: str, days_ahead: int = 45) -> list[dict[str, Any]]:
    end = (date.today() + timedelta(days=days_ahead)).isoformat()
    if SERVICE_URL:
        events = query_events(ticker=ticker, end=end)
        return [
            e for e in events
            if (e.get("action_flags") or {}).get("halt_auto_execution")
            or e.get("confidence") in {"conflicting", "rumored"}
        ]
    return storage.blocking_events(ticker, end=end)


def nearest_event(ticker: str, days_ahead: int = 120) -> dict[str, Any] | None:
    events = query_events(
        ticker=ticker,
        start=date.today().isoformat(),
        end=(date.today() + timedelta(days=days_ahead)).isoformat(),
    )
    return events[0] if events else None
