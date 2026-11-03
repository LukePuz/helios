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

import pytest

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
        weeks = compute_weeks_to_catalyst("2026-09", as_of=_AS_OF, month_midpoint=False)
        assert abs(weeks - 2 / 7) < 1e-9

    def test_month_only_reads_the_15th_with_the_midpoint_switch(self):
        # 2026-09-15 is 16 days after 2026-08-30 (audit A1031-0253-1)
        weeks = compute_weeks_to_catalyst("2026-09", as_of=_AS_OF, month_midpoint=True)
        assert abs(weeks - 16 / 7) < 1e-9
        assert abs(compute_weeks_to_catalyst("2026-9", as_of=_AS_OF, month_midpoint=True) - 16 / 7) < 1e-9
        # a full date is unaffected by the switch
        assert abs(compute_weeks_to_catalyst("2026-09-20", as_of=_AS_OF, month_midpoint=True) - 3.0) < 1e-9

    def test_the_switch_ships_off(self):
        import config
        assert config.CATALYST_MONTH_MIDPOINT_ENABLED is False
        assert abs(compute_weeks_to_catalyst("2026-09", as_of=_AS_OF) - 2 / 7) < 1e-9

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



class TestNullExpectedReturnWithDatedCatalyst:
    """A dated catalyst (weeks_to_catalyst computable) with a null
    expected_return_pct leaves the rate uncomputable. The gate's None
    branch already decides: OPEN/ADD fail closed to WATCH, a HOLD is
    flagged stale. The reason text used to format the None rate and
    raise TypeError, so analyze() logged ANALYST_FAILED instead."""

    @pytest.fixture(autouse=True)
    def _tolerate_malformed_catalyst(self):
        with patch("config.ANALYST_TOLERATE_MALFORMED_CATALYST", True):
            yield

    def _run(self, action, extra_payload=None):
        import json
        payload = json.loads(_response(action, None, _THREE_WEEKS_OUT))
        payload.update(extra_payload or {})
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8]:
            from agents.analyst import analyze
            result = analyze(_CANDIDATE, situation_brief="brief")
        gates = log_signal.call_args.kwargs["decision"]["gates"]
        return result, gates

    def test_open_fails_closed_to_watch(self):
        result, gates = self._run("OPEN")
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert result["weeks_to_catalyst"] is not None
        assert result["expected_return_per_week"] is None
        assert "[Downgraded: no expected_return_pct]" in result["reasoning"]
        assert any(g.startswith("return rate (no expected_return_pct)") for g in gates)

    def test_add_fails_closed_to_watch(self):
        result, _ = self._run("ADD")
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False

    def test_hold_is_flagged_stale_not_downgraded(self):
        result, _ = self._run("HOLD", {"pass_to_risk": False})
        assert result["action"] == "HOLD"
        assert result["catalyst_stale"] is True
        assert "[STALE CATALYST ON HOLD: no expected_return_pct]" in result["reasoning"]


def test_the_midpoint_switch_is_in_the_provenance_config():
    from agents.analyst import _treatment_config
    assert _treatment_config()["CATALYST_MONTH_MIDPOINT_ENABLED"] is False


class TestNonObjectCatalyst:
    """A catalyst the model returns as a string or a list (the schema asks
    for an object) is read as no catalyst. Before, a Compounder OPEN lost
    its signals row (_catalyst_check raised inside log_signal's arguments)
    and an Accelerator/Moonshot buy or HOLD raised AttributeError, ending
    as ANALYST_FAILED."""

    @pytest.fixture(autouse=True)
    def _tolerate_malformed_catalyst(self):
        with patch("config.ANALYST_TOLERATE_MALFORMED_CATALYST", True):
            yield

    def _run(self, action, catalyst, role, extra_payload=None):
        import json
        payload = json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket=role))
        payload["catalyst"] = catalyst
        payload.update(extra_payload or {})
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8]:
            from agents.analyst import analyze
            result = analyze({**_CANDIDATE, "role": role}, situation_brief="brief")
        assert log_signal.call_count == 1
        decision = log_signal.call_args.kwargs["decision"]
        return result, decision

    @pytest.mark.parametrize("catalyst", ["earnings next month", [], ["earnings"]])
    def test_compounder_open_still_writes_its_signal(self, catalyst):
        result, decision = self._run("OPEN", catalyst, "Compounder")
        assert result["action"] == "OPEN"
        assert decision["final_action"] == "OPEN"
        assert decision["catalyst_check"]["model_date"] is None
        assert decision["model_output"]["catalyst"] == catalyst

    @pytest.mark.parametrize("catalyst", ["earnings next month", [], ["earnings"]])
    def test_accelerator_open_fails_closed(self, catalyst):
        result, decision = self._run("OPEN", catalyst, "Accelerator")
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert result["weeks_to_catalyst"] is None
        assert any(g.startswith("return rate (no computable catalyst timeframe)")
                   for g in decision["gates"])

    @pytest.mark.parametrize("catalyst", ["earnings next month", [], ["earnings"]])
    def test_moonshot_hold_is_flagged_stale(self, catalyst):
        result, _ = self._run("HOLD", catalyst, "Moonshot", {"pass_to_risk": False})
        assert result["action"] == "HOLD"
        assert result["catalyst_stale"] is True
        assert "[STALE CATALYST ON HOLD: no computable catalyst timeframe]" in result["reasoning"]


class TestNonStringCatalystDate:
    """A catalyst.date that isn't a string: a number is read as its text,
    anything else as no date. Before, compute_weeks_to_catalyst called
    .strip() on it and analyze() raised (ANALYST_FAILED)."""

    @pytest.fixture(autouse=True)
    def _tolerate_malformed_catalyst(self):
        with patch("config.ANALYST_TOLERATE_MALFORMED_CATALYST", True):
            yield

    def _run(self, action, date, role, extra_payload=None):
        import json
        payload = json.loads(_response(action, 20.0, date, bucket=role))
        payload.update(extra_payload or {})
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8]:
            from agents.analyst import analyze
            result = analyze({**_CANDIDATE, "role": role}, situation_brief="brief")
        assert log_signal.call_count == 1
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["model_output"]["catalyst"]["date"] == date
        return result, decision

    @pytest.mark.parametrize("date, read_as", [(20261201, "20261201"), (["2026-12"], None)])
    def test_accelerator_open_fails_closed(self, date, read_as):
        result, decision = self._run("OPEN", date, "Accelerator")
        assert result["catalyst"]["date"] == read_as
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert result["weeks_to_catalyst"] is None
        assert any(g.startswith("return rate (no computable catalyst timeframe)")
                   for g in decision["gates"])

    @pytest.mark.parametrize("date", [20261201, ["2026-12"]])
    def test_moonshot_hold_is_flagged_stale(self, date):
        result, _ = self._run("HOLD", date, "Moonshot", {"pass_to_risk": False})
        assert result["action"] == "HOLD"
        assert result["catalyst_stale"] is True
        assert "[STALE CATALYST ON HOLD: no computable catalyst timeframe]" in result["reasoning"]


def test_the_malformed_catalyst_switch_ships_off():
    import config
    from agents.analyst import _treatment_config
    assert config.ANALYST_TOLERATE_MALFORMED_CATALYST is False
    assert _treatment_config()["ANALYST_TOLERATE_MALFORMED_CATALYST"] is False


class TestMalformedCatalystNotTolerated:
    """ANALYST_TOLERATE_MALFORMED_CATALYST off: the three cases above end as
    they did before they were tolerated. On the Accelerator/Moonshot rate
    path analyze() raises the same exception, writes one ANALYST_FAILED
    row (kind "exception") and no verdict row; a Compounder OPEN with a
    non-object catalyst still loses its verdict row (_catalyst_check raises
    inside log_signal's arguments)."""

    @pytest.fixture(autouse=True)
    def _not_tolerated(self):
        with patch("config.ANALYST_TOLERATE_MALFORMED_CATALYST", False):
            yield

    def _run_failed(self, payload, role, error_type, caplog):
        import json
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8], caplog.at_level("WARNING"):
            from agents.analyst import analyze
            with pytest.raises(error_type) as exc:
                analyze({**_CANDIDATE, "role": role}, situation_brief="brief")
        assert log_signal.call_count == 1
        args, kwargs = log_signal.call_args
        assert args[:3] == ("SNOW", "ANALYST_FAILED", None)
        assert args[3] == f"No verdict (exception): {error_type.__name__}: {exc.value}"
        assert kwargs["model"] is None
        failure = kwargs["decision"]["failure"]
        assert failure["kind"] == "exception"
        assert failure["error_type"] == error_type.__name__
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelname == "WARNING" and "not tolerated" in r.getMessage()]
        assert len(warnings) == 1
        return str(exc.value), warnings[0]

    @pytest.mark.parametrize("action, extra", [("OPEN", {}), ("ADD", {}),
                                               ("HOLD", {"pass_to_risk": False})])
    def test_null_expected_return_raises_type_error(self, action, extra, caplog):
        import json
        payload = {**json.loads(_response(action, None, _THREE_WEEKS_OUT)), **extra}
        msg, warning = self._run_failed(payload, "Accelerator", TypeError, caplog)
        assert msg == "unsupported format string passed to NoneType.__format__"
        assert "expected_return_pct" in warning

    @pytest.mark.parametrize("catalyst", ["earnings next month", ["earnings"]])
    @pytest.mark.parametrize("action, role, extra", [("OPEN", "Accelerator", {}),
                                                     ("HOLD", "Moonshot", {"pass_to_risk": False})])
    def test_non_object_catalyst_raises_attribute_error(self, catalyst, action, role, extra, caplog):
        import json
        payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket=role)),
                   "catalyst": catalyst, **extra}
        msg, warning = self._run_failed(payload, role, AttributeError, caplog)
        assert msg == f"'{type(catalyst).__name__}' object has no attribute 'get'"
        assert "Malformed catalyst:" in warning

    @pytest.mark.parametrize("date", [20261201, ["2026-12"]])
    @pytest.mark.parametrize("action, role, extra", [("OPEN", "Accelerator", {}),
                                                     ("HOLD", "Moonshot", {"pass_to_risk": False})])
    def test_non_string_date_raises_attribute_error(self, date, action, role, extra, caplog):
        import json
        payload = {**json.loads(_response(action, 20.0, date, bucket=role)), **extra}
        msg, warning = self._run_failed(payload, role, AttributeError, caplog)
        assert msg == f"'{type(date).__name__}' object has no attribute 'strip'"
        assert "Malformed catalyst.date:" in warning

    def test_compounder_open_with_non_object_catalyst_writes_no_verdict_row(self):
        import json
        payload = {**json.loads(_response("OPEN", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8]:
            from agents.analyst import analyze
            result = analyze({**_CANDIDATE, "role": "Compounder"}, situation_brief="brief")
        assert log_signal.call_count == 0
        assert result["action"] == "OPEN"
        assert result["catalyst"] == "earnings next month"
