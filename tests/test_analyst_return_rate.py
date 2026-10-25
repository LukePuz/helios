"""
Tests for the minimum expected-return-RATE gate added to agents/analyst.py
2026-08-30 (user request): expected_return_pct / weeks_to_catalyst must
clear MIN_EXPECTED_RETURN_PCT_PER_WEEK (1%/week) for an Accelerator/
Moonshot OPEN/ADD to be honored. Worked examples from the user's own
request, all of which must clear the bar:
  - 5% expected run-up into an earnings call 3 weeks out (5/3 = 1.67%/wk)
  - 3% expected move the next session, reading an earnings report right
    now (huge implied rate once the same-day catalyst is floored to 1 day)
  - 20% expected move over the next quarter, ~13 weeks (20/13 = 1.54%/wk)

compute_weeks_to_catalyst and expected_return_per_week are pure functions
(no live DB/Ollama required); the analyze() integration tests mock the
LLM call end-to-end, same pattern as test_analyst_sources.py.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.analyst import compute_weeks_to_catalyst, expected_return_per_week


_AS_OF = datetime(2026, 8, 30, tzinfo=timezone.utc)


class TestComputeWeeksToCatalyst:
    def test_specific_date_is_used_exactly(self):
        # 2026-09-20 is 21 days after 2026-08-30 -> 3.0 weeks
        weeks = compute_weeks_to_catalyst("2026-09-20", as_of=_AS_OF)
        assert abs(weeks - 3.0) < 1e-9

    def test_month_only_defaults_to_the_1st(self):
        # "2026-09" -> strptime("%Y-%m") defaults the day to 1 ->
        # 2026-09-01, 2 days after 2026-08-30 -> 2/7 weeks
        weeks = compute_weeks_to_catalyst("2026-09", as_of=_AS_OF)
        assert abs(weeks - 2 / 7) < 1e-9

    def test_ongoing_returns_none(self):
        assert compute_weeks_to_catalyst("ongoing", as_of=_AS_OF) is None
        assert compute_weeks_to_catalyst("Ongoing", as_of=_AS_OF) is None  # case-insensitive

    def test_none_or_empty_returns_none(self):
        assert compute_weeks_to_catalyst(None, as_of=_AS_OF) is None
        assert compute_weeks_to_catalyst("", as_of=_AS_OF) is None

    def test_malformed_string_returns_none(self):
        assert compute_weeks_to_catalyst("sometime next year", as_of=_AS_OF) is None

    def test_next_day_catalyst_is_floored_to_one_day_not_zero(self):
        # "The next session" reading a report today -> catalyst date is
        # effectively today/tomorrow -- must floor to 1 day, not go to
        # zero or negative.
        weeks = compute_weeks_to_catalyst("2026-08-30", as_of=_AS_OF)
        assert abs(weeks - 1 / 7) < 1e-9

    def test_past_date_is_also_floored_not_negative(self):
        weeks = compute_weeks_to_catalyst("2026-01-01", as_of=_AS_OF)
        assert abs(weeks - 1 / 7) < 1e-9


class TestExpectedReturnPerWeek:
    def test_normal_division(self):
        # 5% over 3 weeks = 1.67%/week
        assert abs(expected_return_per_week(5.0, 3.0) - 5.0 / 3.0) < 1e-9

    def test_worked_example_quarter(self):
        # 20% over ~13 weeks = 1.54%/week
        assert abs(expected_return_per_week(20.0, 13.0) - 20.0 / 13.0) < 1e-9

    def test_none_expected_return_returns_none(self):
        assert expected_return_per_week(None, 3.0) is None

    def test_none_weeks_returns_none(self):
        assert expected_return_per_week(5.0, None) is None

    def test_zero_or_negative_weeks_returns_none(self):
        assert expected_return_per_week(5.0, 0.0) is None
        assert expected_return_per_week(5.0, -1.0) is None


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


def _response(action, expected_return_pct, catalyst_date, bucket="Accelerator"):
    import json
    return json.dumps({
        "action": action, "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": bucket,
        "thesis_summary": "test", "expectation_gap": {},
        "catalyst": {"event": "earnings", "date": catalyst_date, "is_priced_in": False},
        "valuation": {}, "expected_return_pct": expected_return_pct, "expected_downside_pct": -10.0,
        "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": True,
        "reasoning": "per Q2 fundamentals, real evidence",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })


# analyze() measures weeks_to_catalyst from the real clock, so the
# integration tests put the catalyst 3 weeks after today, not on a fixed date.
_THREE_WEEKS_OUT = (datetime.now(timezone.utc) + timedelta(days=21)).strftime("%Y-%m-%d")

_CANDIDATE = {"ticker": "SNOW", "role": "Accelerator", "tier": "Watch", "sector": "Software"}


class TestExpectedReturnRateGateIntegration:
    def test_five_pct_in_three_weeks_clears_the_bar(self):
        response = _response("OPEN", 5.0, _THREE_WEEKS_OUT)
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "OPEN"
        assert result["expected_return_per_week"] is not None
        assert result["expected_return_per_week"] >= 1.0

    def test_two_pct_in_three_weeks_fails_the_bar(self):
        response = _response("OPEN", 2.0, _THREE_WEEKS_OUT)  # 2/3 = 0.67%/wk < 1.0
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert "week" in result["reasoning"]

    def test_ongoing_catalyst_fails_closed(self):
        response = _response("OPEN", 20.0, "ongoing")
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "WATCH"
        assert result["weeks_to_catalyst"] is None

    def test_compounder_bucket_is_exempt(self):
        # Same "ongoing"/low-implied-rate shape that would fail for
        # Accelerator must NOT be touched for Compounder -- the long-term
        # core bucket by explicit prior design, no defined catalyst
        # timeline to rate-check against. Bucket is watchlist-role-
        # authoritative (not the LLM's stated thesis_bucket), so the
        # candidate itself must carry role="Compounder".
        compounder_candidate = {**_CANDIDATE, "role": "Compounder"}
        response = _response("OPEN", 20.0, "ongoing", bucket="Compounder")
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(compounder_candidate, situation_brief="brief")
        assert result["action"] == "OPEN"

    def test_hold_action_is_never_gated_regardless_of_rate(self):
        response = _response("HOLD", 2.0, "ongoing")
        # HOLD's pass_to_risk is False in the source JSON already, matching
        # a routine review -- the gate must not even evaluate/alter it.
        import json
        payload = json.loads(response)
        payload["pass_to_risk"] = False
        response = json.dumps(payload)
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "HOLD"

    def test_next_session_catalyst_with_a_small_move_still_clears_easily(self):
        # The user's own "reading an earnings report, expecting 3% the
        # next session" example -- today's date as the catalyst date,
        # floored to 1 day, produces a huge implied weekly rate.
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        response = _response("OPEN", 3.0, today_str)
        patches = _mock_analyze_deps(response)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        assert result["action"] == "OPEN"
        assert result["expected_return_per_week"] > 1.0
