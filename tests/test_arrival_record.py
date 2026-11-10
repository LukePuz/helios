"""Sizing carries an arrival record: the brief's price and time, the
triggering item's time, and the move and minutes to the sizing quote (audit
2026-11-07 17:17 #1). Log-only."""
import inspect
from datetime import datetime, timedelta, timezone

from agents import crew


def _result(**trig):
    return {"provenance": {"trigger": trig}}


def test_move_and_minutes_from_the_brief_and_the_event():
    now = datetime.now(timezone.utc)
    r = crew._arrival_record(_result(brief_price=100.0,
                                     brief_built_at=(now - timedelta(minutes=6)).isoformat(),
                                     event_time=(now - timedelta(minutes=218)).isoformat()), 101.5)
    assert r["move_brief_to_sizing_pct"] == 1.5 and r["sizing_price"] == 101.5
    assert 5.9 <= r["minutes_brief_to_sizing"] <= 6.2
    assert 217.9 <= r["minutes_event_to_sizing"] <= 218.2


def test_rfc2822_event_time_and_missing_fields():
    r = crew._arrival_record(_result(event_time="Thu, 05 Nov 2026 12:31:00 GMT"), 50.0)
    assert r["minutes_event_to_sizing"] is not None and r["brief_price"] is None
    assert r["move_brief_to_sizing_pct"] is None and r["minutes_brief_to_sizing"] is None
    r = crew._arrival_record({}, 50.0)
    assert r["event_time"] is None and "error" not in r
    assert crew._arrival_record(_result(event_time="not a date"), 1.0)["minutes_event_to_sizing"] is None


def test_attach_shadows_adds_arrival_and_the_brief_price_reaches_the_trigger():
    sizing = {"stop_loss_price": 90.0}
    crew._attach_shadows(sizing, "NVDA", _result(brief_price=100.0), 100.0)
    assert sizing["arrival"]["move_brief_to_sizing_pct"] == 0.0
    assert sizing["shadow_status"]["arrival"] == "ok"
    src = inspect.getsource(crew.deep_analyze)
    assert 'trigger["brief_price"] = brief_price' in src and "brief_price = price" in src


def test_a_date_only_event_time_gives_no_minutes():
    r = crew._arrival_record(_result(event_time="2026-11-05", event_time_source="filing_date"), 1.0)
    assert r["event_time_date_only"] is True and r["minutes_event_to_sizing"] is None
    assert r["event_time_source"] == "filing_date"
    r = crew._arrival_record(_result(event_time="2026-11-05T12:00:00+00:00", event_time_source="published_at"), 1.0)
    assert r["event_time_date_only"] is False and r["minutes_event_to_sizing"] is not None
