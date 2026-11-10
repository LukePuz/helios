"""Default-off analyst add-ons: scenarios, earnings event_plan (Luke 11-08)."""
from datetime import date
from unittest.mock import patch

import config
from agents import analyst
from agents.crew import _earnings_review_event

PRE = {"path": "earnings_review", "earnings_phase": "pre"}


def test_all_off_appends_nothing():
    for trig in (None, PRE, {"path": "event"}):
        text, rec = analyst._addon_instructions(trig)
        assert text == "" and rec == {"scenarios_requested": False, "event_plan_requested": False}


def test_scenarios_on_every_verdict(monkeypatch):
    monkeypatch.setattr(config, "ANALYST_SCENARIOS_ENABLED", True)
    text, rec = analyst._addon_instructions({"path": "daily_scan"})
    assert '"scenarios"' in text and "event_plan" not in text
    assert text.rstrip().endswith("Output ONLY the JSON object, nothing else")
    assert rec == {"scenarios_requested": True, "event_plan_requested": False}


def test_event_plan_only_before_a_held_print(monkeypatch):
    monkeypatch.setattr(config, "ANALYST_EVENT_PLAN_ENABLED", True)
    assert '"event_plan"' in analyst._addon_instructions(PRE)[0]
    for trig in ({"path": "earnings_review", "earnings_phase": "post"},
                 {"path": "earnings_review", "earnings_phase": "day"},
                 {"path": "event", "earnings_phase": "pre"}, None):
        assert analyst._addon_instructions(trig)[0] == ""


def test_earnings_phase_on_the_event_and_the_trigger():
    d = date(2026, 11, 17)
    assert _earnings_review_event("NVDA", d, today=date(2026, 11, 10))["earnings_phase"] == "pre"
    assert _earnings_review_event("NVDA", d, today=d)["earnings_phase"] == "day"
    ev = _earnings_review_event("NVDA", d, today=date(2026, 11, 19))
    assert ev["earnings_phase"] == "post"
    assert analyst.trigger_record("earnings_review", ev)["earnings_phase"] == "post"
    assert "earnings_phase" not in analyst.trigger_record("event", {"title": "x"})


def test_parse_record():
    rec = {"scenarios_requested": True, "event_plan_requested": True}
    parsed = {"action": "TRIM",
              "scenarios": [{"if": "new export limits", "then_action": "trim"},
                            {"if": "", "then_action": "HOLD"}, "junk"],
              "event_plan": {"pre_print": "trim_before", "expected_move_pct": 8.0,
                             "breaks_thesis_if": "guidance cut"}}
    out = analyst._addon_parse_record(parsed, rec)
    assert out["scenarios_n"] == 3 and out["scenarios_valid"] == 1 and out["scenarios_actions"] == ["TRIM"]
    assert out["event_plan_valid"] is True and out["action_matches_pre_print"] is True
    out = analyst._addon_parse_record({"action": "HOLD", "event_plan": {"pre_print": "exit_before",
                                                                        "expected_move_pct": True}}, rec)
    assert out["scenarios_present"] is False and out["event_plan_valid"] is False
    assert out["action_matches_pre_print"] is False
    assert analyst._addon_parse_record({}, {"scenarios_requested": False,
                                            "event_plan_requested": False}) == {}


def test_addon_text_has_its_own_hash_and_the_full_hash_is_unchanged():
    assert "ADDITIONAL REQUIRED FIELD" not in analyst._prompt_template_full_text()
    shas = analyst._template_shas()
    assert isinstance(shas["addon_template_sha"], str) and len(shas["addon_template_sha"]) == 12


def test_flags_in_treatment_config():
    tc = analyst._treatment_config()
    for k in ("ANALYST_SCENARIOS_ENABLED", "ANALYST_EVENT_PLAN_ENABLED", "BRIEF_PRICE_HISTORY",
              "DIP_REVIEW_ENABLED", "DIP_REVIEW_MIN_DROP_PCT", "DIP_REVIEW_MIN_ATR"):
        assert k in tc


def test_amc_today_is_still_pre_print():
    from agents import crew
    d = date(2026, 11, 17)
    ev = {}
    with patch("agents.crew._earnings_review_event",
               side_effect=lambda t, e: {**_earnings_review_event(t, e, today=d)}), \
         patch("agents.crew._earnings_timing", return_value={"time_of_day": "AMC"}), \
         patch("agents.crew._analyze_held_position", side_effect=lambda *a, **k: ev.update(k["event"]) or {}), \
         patch("agents.crew._capture_latest_thesis"):
        crew.review_held_position("NVDA", None, {}, {}, earnings_date=d)
    assert ev["earnings_phase"] == "pre"
    with patch("agents.crew._earnings_review_event",
               side_effect=lambda t, e: {**_earnings_review_event(t, e, today=d)}), \
         patch("agents.crew._earnings_timing", return_value={"time_of_day": "BMO"}), \
         patch("agents.crew._analyze_held_position", side_effect=lambda *a, **k: ev.update(k["event"]) or {}), \
         patch("agents.crew._capture_latest_thesis"):
        crew.review_held_position("NVDA", None, {}, {}, earnings_date=d)
    assert ev["earnings_phase"] == "day"


def test_routing_switches_in_treatment_config():
    tc = analyst._treatment_config()
    for k in ("BUY_REQUIRES_NEW_FACT", "BUY_NEW_FACT_SCOPE", "BUY_NEW_FACT_WINDOW",
              "TRIM_FRACTION", "REBUY_COOLDOWN_HOURS", "DIP_REVIEW_COUNTS_AS_NEW_FACT"):
        assert k in tc


def test_invalidate_records_a_buy_below_it():
    from datetime import date as _d
    from agents import plans
    plan = {"id": 1, "invalidation_price": 100.0, "target_price": 150.0, "horizon_end": _d(2027, 1, 1)}
    out = plans.classify(plan, "ADD", None, 95.0, _d(2026, 11, 9), True)
    assert out["relation"] == "INVALIDATE" and out["buy_below_invalidation"] is True
    out = plans.classify(plan, "HOLD", None, 95.0, _d(2026, 11, 9), True)
    assert out["buy_below_invalidation"] is False and out["action"] == "HOLD"
