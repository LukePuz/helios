"""
Tests for the multi-source corroboration gate added to agents/analyst.py
2026-08-30 (user request): a thesis resting on a single piece of evidence
-- one article's opinion, one filing, one options-flow blip -- isn't
validated, it's a guess with conviction attached. OPEN/ADD recommendations
now require >=2 distinct sources or get deterministically downgraded to
WATCH, mirroring the existing UDR<2.0 and fundamentals-missing guardrails.

count_distinct_sources is a pure function (no live DB/Ollama required);
the analyze() integration tests mock the LLM call and its supporting
lookups to confirm the guardrail actually wires through end-to-end.
"""
from unittest.mock import MagicMock, patch

from agents.analyst import count_distinct_sources


class TestCountDistinctSources:
    def test_two_distinct_types_counts_as_two(self):
        sources = [
            {"type": "news", "detail": "Reuters reports Q3 guidance raise"},
            {"type": "edgar_filing", "detail": "8-K filed confirming the raise"},
        ]
        assert count_distinct_sources(sources) == 2

    def test_two_news_articles_reporting_different_facts_count_as_two(self):
        # Same type, but genuinely distinct detail -- must still count,
        # per the prompt rule that same-type sources are fine if the
        # underlying facts are actually different reports.
        sources = [
            {"type": "news", "detail": "Bloomberg: CEO announces new contract"},
            {"type": "news", "detail": "Reuters: contract value estimated at $50M"},
        ]
        assert count_distinct_sources(sources) == 2

    def test_same_fact_restated_twice_counts_as_one(self):
        sources = [
            {"type": "news", "detail": "Company announces new AI partnership"},
            {"type": "news", "detail": "Company announces new AI partnership"},
        ]
        assert count_distinct_sources(sources) == 1

    def test_empty_or_missing_list_counts_as_zero(self):
        assert count_distinct_sources([]) == 0
        assert count_distinct_sources(None) == 0

    def test_malformed_entries_are_ignored_not_crashed_on(self):
        sources = [
            {"type": "news", "detail": "real source"},
            "not a dict",
            {"detail": "missing type field"},
            {},
        ]
        assert count_distinct_sources(sources) == 1

    def test_single_source_counts_as_one(self):
        sources = [{"type": "analyst_rating", "detail": "Morgan Stanley upgrade to Overweight"}]
        assert count_distinct_sources(sources) == 1


def _mock_analyze_deps(json_response: str):
    """Mocks everything analyze() touches besides the LLM call and the new
    guardrail, so the test exercises the real wiring without needing a
    live Ollama instance, DB, or knowledge base."""
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


_CANDIDATE = {"ticker": "SNOW", "role": "Accelerator", "tier": "Watch", "sector": "Software"}


class TestSourceGateIntegration:
    def test_open_with_two_sources_is_not_downgraded(self):
        import json as _json
        response = _json.dumps({
            "action": "OPEN", "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
            "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Accelerator",
            "thesis_summary": "test", "expectation_gap": {}, "catalyst": {}, "valuation": {},
            "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": True,
            "reasoning": "per Q2 fundamentals, revenue grew 40%",
            "sources": [
                {"type": "news", "detail": "Reuters: new contract announced"},
                {"type": "fundamentals_data", "detail": "Q2 revenue +40% YoY"},
            ],
        })
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "OPEN"
        assert result["source_count"] == 2

    def test_open_with_one_source_is_downgraded_to_watch(self):
        import json as _json
        response = _json.dumps({
            "action": "OPEN", "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
            "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Accelerator",
            "thesis_summary": "test", "expectation_gap": {}, "catalyst": {}, "valuation": {},
            "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": True,
            "reasoning": "per one article",
            "sources": [{"type": "news", "detail": "One article's opinion"}],
        })
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert result["source_count"] == 1
        assert "distinct source" in result["reasoning"]

    def test_hold_with_one_source_is_not_downgraded(self):
        # The gate only applies to OPEN/ADD -- an existing HOLD shouldn't
        # be force-exited just because this particular review cycle's
        # brief happened to cite only one source.
        import json as _json
        response = _json.dumps({
            "action": "HOLD", "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
            "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Accelerator",
            "thesis_summary": "test", "expectation_gap": {}, "catalyst": {}, "valuation": {},
            "upside_downside_ratio": 2.5, "key_risks": [], "pass_to_risk": False,
            "reasoning": "no change",
            "sources": [{"type": "price_action", "detail": "still above 50DMA"}],
        })
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "HOLD"

    def test_no_sources_field_at_all_still_downgrades_open(self):
        # Older/degraded model output that omits "sources" entirely must
        # fail safe (0 sources -> downgrade), not crash or silently pass.
        import json as _json
        response = _json.dumps({
            "action": "OPEN", "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
            "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Accelerator",
            "thesis_summary": "test", "expectation_gap": {}, "catalyst": {}, "valuation": {},
            "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": True,
            "reasoning": "no sources field",
        })
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "WATCH"
        assert result["source_count"] == 0
