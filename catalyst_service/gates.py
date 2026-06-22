from __future__ import annotations

from .client import blocking_events_for_horizon


def catalyst_gate_check(ticker: str, days_ahead: int = 45) -> dict:
    blocking = blocking_events_for_horizon(ticker, days_ahead=days_ahead)
    if not blocking:
        return {"approved": True, "blocking_events": [], "reason": ""}

    first = blocking[0]
    window = first.get("window_start") and first.get("window_end")
    date_text = (
        f"{first['window_start']} to {first['window_end']}"
        if window else first.get("event_date", "unknown date")
    )
    return {
        "approved": False,
        "blocking_events": blocking,
        "reason": (
            f"{first.get('ticker')} {first.get('event_type')} has "
            f"{first.get('confidence')} catalyst timing ({date_text})"
        ),
    }

