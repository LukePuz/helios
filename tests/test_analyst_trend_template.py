"""
Tests for the Trend Template gate wired into agents/analyst.py (2026-09-11,
user request): Accelerator OPEN/ADD must clear Mark Minervini's 8-point
Trend Template (see agents/price_action.compute_trend_template) on top of
the LLM's own judgment. Mirrors test_analyst_return_rate.py's pattern.
"""
import json
from unittest.mock import patch

from agents.analyst import analyze


def _mock_analyze_deps(json_response: str):
    return [
        patch("agents.analyst._call_ollama", return_value=json_response),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("agents.market_regime.get_cached_regime",
              return_value={"regime": "Neutral", "score": 50, "reasoning": ""}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}),
        patch("db.log_signal"),
        patch("feed.emit"),
    ]


def _response(action, bucket="Accelerator"):
    return json.dumps({
        "action": action, "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": bucket,
        "thesis_summary": "test", "expectation_gap": {},
        "catalyst": {"event": "earnings", "date": "2026-09-20", "is_priced_in": False},
        "valuation": {}, "expected_return_pct": 5.0, "expected_downside_pct": -10.0,
        "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": True,
        "reasoning": "per Q2 fundamentals, real evidence",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })


_CANDIDATE = {"ticker": "NVDA", "role": "Accelerator", "tier": "Watch", "sector": "Software"}


def _run(candidate, response, trend_template_return):
    patches = _mock_analyze_deps(response)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
         patches[6], patches[7], patches[8], \
         patch("agents.price_action.compute_trend_template", return_value=trend_template_return):
        return analyze(candidate, situation_brief="brief")


class TestTrendTemplateGateIntegration:
    def test_open_with_trend_template_pass_is_not_downgraded(self):
        result = _run(_CANDIDATE, _response("OPEN"),
                       {"pass": True, "criteria": {}, "reasoning": "All 8 Trend Template criteria met."})
        assert result["action"] == "OPEN"
        assert result["trend_template"]["pass"] is True

    def test_open_with_trend_template_fail_is_downgraded_to_watch(self):
        result = _run(_CANDIDATE, _response("OPEN"),
                       {"pass": False, "criteria": {"price_above_50dma": False},
                        "reasoning": "Failed 1/8 Trend Template criteria: price_above_50dma."})
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert "Trend Template" in result["reasoning"]

    def test_add_action_is_also_gated(self):
        result = _run(_CANDIDATE, _response("ADD"),
                       {"pass": False, "criteria": {}, "reasoning": "Failed 8/8."})
        assert result["action"] == "WATCH"

    def test_hold_action_is_never_gated_regardless_of_trend_template(self):
        response = json.loads(_response("HOLD"))
        response["pass_to_risk"] = False
        result = _run(_CANDIDATE, json.dumps(response),
                       {"pass": False, "criteria": {}, "reasoning": "Failed 8/8."})
        assert result["action"] == "HOLD"

    def test_moonshot_bucket_is_never_gated(self):
        moonshot_candidate = {**_CANDIDATE, "role": "Moonshot"}
        result = _run(moonshot_candidate, _response("OPEN", bucket="Moonshot"),
                       {"pass": False, "criteria": {}, "reasoning": "Failed 8/8."})
        assert result["action"] == "OPEN"

    def test_compounder_bucket_is_never_gated(self):
        compounder_candidate = {**_CANDIDATE, "role": "Compounder"}
        result = _run(compounder_candidate, _response("OPEN", bucket="Compounder"),
                       {"pass": False, "criteria": {}, "reasoning": "Failed 8/8."})
        assert result["action"] == "OPEN"

    def test_insufficient_history_none_passes_through_ungated(self):
        # compute_trend_template returns None when there isn't enough
        # price history yet -- must NOT fail closed the way the expected-
        # return-rate gate does for "no catalyst timeframe".
        result = _run(_CANDIDATE, _response("OPEN"), None)
        assert result["action"] == "OPEN"
        assert result["trend_template"] is None

    def test_exception_in_trend_template_check_fails_safe_not_crash(self):
        patches = _mock_analyze_deps(_response("OPEN"))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8], \
             patch("agents.price_action.compute_trend_template", side_effect=Exception("db down")):
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "OPEN"
        assert result["trend_template"] is None
