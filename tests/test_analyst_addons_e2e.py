"""analyze() with the add-ons off sends the same prompt as before; on, it
appends the block and records what came back."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import config
from agents import analyst

_BRIEF = "=== SITUATION BRIEF ===\nCurrent Price: $100.00\n"


def _run(extra=None):
    parsed = {
        "action": "HOLD", "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {"bull_case_target": "$130", "downside_scenario": "$90"},
        "expected_return_pct": 8.0, "key_risks": [], "pass_to_risk": False,
        "upside_downside_ratio": 2.4, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}], **(extra or {})}
    call = MagicMock(return_value=json.dumps(parsed))
    log_signal = MagicMock()
    with ExitStack() as st:
        for p in (patch("agents.analyst._call_ollama", call),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1}),
                  patch("db.get_latest_signal_for", return_value=None),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.get_trade_ladder", return_value=None),
                  patch("db.log_signal", log_signal),
                  patch("feed.emit")):
            st.enter_context(p)
        result = analyst.analyze({"ticker": "MSFT", "role": "Compounder", "tier": "Core",
                                  "sector": "Software", "held": True}, situation_brief=_BRIEF,
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""},
                                 trigger={"path": "earnings_review", "earnings_phase": "pre"})
    return call.call_args[0][0], result


def test_off_prompt_is_the_built_prompt_unchanged():
    prompt, result = _run()
    assert "ADDITIONAL REQUIRED FIELD" not in prompt
    assert prompt.rstrip().endswith("Output ONLY the JSON object, nothing else")
    assert result["provenance"]["analyst_addons"] == {"scenarios_requested": False,
                                                      "event_plan_requested": False}
    assert result["provenance"]["prompt_variant"] == "base"


def test_on_prompt_carries_both_blocks_and_the_record(monkeypatch):
    monkeypatch.setattr(config, "ANALYST_SCENARIOS_ENABLED", True)
    monkeypatch.setattr(config, "ANALYST_EVENT_PLAN_ENABLED", True)
    prompt, result = _run({"scenarios": [{"if": "x", "then_action": "HOLD"}],
                           "event_plan": {"pre_print": "hold_through", "expected_move_pct": 7,
                                          "breaks_thesis_if": "miss"}})
    assert "scenarios" in prompt and "event_plan" in prompt
    rec = result["provenance"]["analyst_addons"]
    assert rec["scenarios_valid"] == 1 and rec["event_plan_valid"] is True
    assert rec["action_matches_pre_print"] is True
    assert result["provenance"]["prompt_variant"] == "base+scenarios+event_plan"
