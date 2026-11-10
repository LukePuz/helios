"""Sizing records the calendar's tighten_sizing events, log-only (audit
11-06 11:34 #6): the flag had writers and no reader."""
from unittest.mock import patch

from agents import crew


def test_records_only_flagged_events():
    events = [{"event_date": "2026-11-06", "scope": "macro", "event_type": "jobs_report",
               "ticker": "MACRO", "action_flags": {"tighten_sizing": True}},
              {"event_date": "2026-11-07", "scope": "ticker", "event_type": "earnings",
               "ticker": "NVDA", "action_flags": {}}]
    with patch("catalyst_service.client.upcoming_events", return_value=events) as up:
        out = crew._tighten_sizing_shadow("NVDA")
    assert up.call_args.kwargs["days_ahead"] == crew._TIGHTEN_SIZING_LOOKAHEAD_DAYS
    assert out["n"] == 1 and out["events"][0]["event_type"] == "jobs_report"


def test_never_raises():
    with patch("catalyst_service.client.upcoming_events", side_effect=RuntimeError("db")):
        out = crew._tighten_sizing_shadow("NVDA")
    assert out["n"] is None and out["error"] == "RuntimeError"
