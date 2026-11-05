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


def test_the_midpoint_switch_is_in_the_provenance_config():
    from agents.analyst import _treatment_config
    assert _treatment_config()["CATALYST_MONTH_MIDPOINT_ENABLED"] is False


def test_the_malformed_catalyst_policy_is_a_fixed_provenance_marker():
    import config
    from agents.analyst import _treatment_config
    cfg = _treatment_config()
    assert cfg["malformed_catalyst_policy"] == "watch_rate_scoped"
    assert "ANALYST_TOLERATE_MALFORMED_CATALYST" not in cfg
    assert not hasattr(config, "ANALYST_TOLERATE_MALFORMED_CATALYST")


@pytest.mark.parametrize("parsed, want", [
    ({"catalyst": "earnings"}, ["catalyst"]),
    ({"catalyst": []}, ["catalyst"]),
    ({"catalyst": ["earnings"]}, ["catalyst"]),
    ({"catalyst": 5}, ["catalyst"]),
    ({"catalyst": {"date": 20261201}}, ["catalyst.date"]),
    ({"catalyst": {"date": 0}}, ["catalyst.date"]),
    ({"catalyst": {"date": False}}, ["catalyst.date"]),
    ({"catalyst": {"date": ["2026-12"]}}, ["catalyst.date"]),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": None}, ["expected_return_pct"]),
    ({"catalyst": {"date": "2026-12"}}, ["expected_return_pct"]),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": "n/a"}, ["expected_return_pct"]),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": True}, ["expected_return_pct"]),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": "nan"}, ["expected_return_pct"]),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": {}}, ["expected_return_pct"]),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": 10 ** 400}, ["expected_return_pct"]),
    # well-formed, or no date to rate against
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": 20}, []),
    ({"catalyst": {"date": "2026-12-01"}, "expected_return_pct": "20"}, []),
    ({"catalyst": {"date": "ongoing"}, "expected_return_pct": None}, []),
    ({"catalyst": {"date": "next quarter"}, "expected_return_pct": None}, []),
    ({"catalyst": {"date": None}, "expected_return_pct": None}, []),
    ({"catalyst": {}}, []),
    ({"catalyst": None}, []),
    ({}, []),
])
def test_malformed_catalyst_fields(parsed, want):
    from agents.analyst import _malformed_catalyst_fields
    assert _malformed_catalyst_fields(parsed) == want


# (payload changes, the malformed fields) for each malformed shape that
# makes any verdict WATCH.
_MALFORMED_SHAPES = {
    "catalyst_str": ({"catalyst": "earnings next month"}, ["catalyst"]),
    "catalyst_empty_list": ({"catalyst": []}, ["catalyst"]),
    "catalyst_list": ({"catalyst": ["earnings"]}, ["catalyst"]),
    "date_int": ({"catalyst": {"event": "earnings", "date": 20261201}}, ["catalyst.date"]),
    "date_list": ({"catalyst": {"event": "earnings", "date": ["2026-12"]}}, ["catalyst.date"]),
    "date_bool": ({"catalyst": {"event": "earnings", "date": True}}, ["catalyst.date"]),
}
# A dated catalyst with an expected return that is null, non-numeric,
# NaN/+-inf or a bool: malformed only at the rate gate (Accelerator/Moonshot
# OPEN/ADD); an Accelerator/Moonshot HOLD stays HOLD, flagged stale.
_RETURN_SHAPES = {
    "erp_null": None,
    "erp_text": "n/a",
    "erp_bool": True,
    "erp_nan": "nan",
}
_DATED = "2026-12-01"


def _run_analyze(payload, role, caplog, held=None):
    import json
    patches = _mock_analyze_deps(json.dumps(payload))
    cand = {**_CANDIDATE, "role": role}
    if held is not None:
        cand["held"] = held
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
         patches[6], patches[7] as log_signal, patches[8], caplog.at_level("WARNING"):
        from agents.analyst import analyze
        result = analyze(cand, situation_brief="brief")
    warnings = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING" and "Malformed" in r.getMessage()]
    return result, log_signal, warnings


class TestMalformedCatalystIsWatch:
    """A malformed catalyst or catalyst.date makes any verdict WATCH with
    pass_to_risk False, in every bucket: one WARNING, one signals row (the
    WATCH verdict, not ANALYST_FAILED), the raw model_output kept and the
    reason in the decision record."""

    @pytest.mark.parametrize("shape", sorted(_MALFORMED_SHAPES))
    @pytest.mark.parametrize("action", ["OPEN", "ADD", "HOLD", "TRIM", "EXIT", "WATCH"])
    @pytest.mark.parametrize("role", ["Compounder", "Accelerator", "Moonshot"])
    def test_any_verdict_becomes_an_unrouted_watch(self, shape, action, role, caplog):
        import json
        changes, fields = _MALFORMED_SHAPES[shape]
        payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket=role)), **changes}
        result, log_signal, warnings = _run_analyze(payload, role, caplog)
        assert result["action"] == "WATCH"
        assert result["pass_to_risk"] is False
        assert len(warnings) == 1 and ", ".join(fields) in warnings[0]
        assert log_signal.call_count == 1
        kwargs = log_signal.call_args.kwargs
        assert kwargs["signal"] == "WATCH"
        decision = kwargs["decision"]
        assert decision["model_action"] == action
        assert decision["final_action"] == "WATCH"
        assert decision["final_pass_to_risk"] is False
        assert decision["downgraded_reason"] == "malformed_catalyst"
        assert decision["malformed_fields"] == fields
        assert decision["gates"][0] == f"malformed catalyst ({', '.join(fields)}): {action} -> WATCH"
        for key, raw in changes.items():
            assert decision["model_output"][key] == raw

    @pytest.mark.parametrize("shape", sorted(_MALFORMED_SHAPES))
    @pytest.mark.parametrize("action", ["ADD", "HOLD"])
    def test_held_flag_does_not_change_it(self, shape, action, caplog):
        import json
        changes, fields = _MALFORMED_SHAPES[shape]
        payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT)), **changes,
                   "pass_to_risk": False}
        for held in (True, False):
            caplog.clear()
            result, log_signal, warnings = _run_analyze(payload, "Accelerator", caplog, held=held)
            assert result["action"] == "WATCH" and result["pass_to_risk"] is False
            assert log_signal.call_args.kwargs["decision"]["gates"][0].startswith("malformed catalyst")


def _run_with_alert_state(payload, role, caplog, held, state):
    """_run_analyze with system_state in a dict and alerts._send mocked
    (a send that went out returns a message id)."""
    with patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("alerts._send", return_value=101) as send:
        result, log_signal, _ = _run_analyze(payload, role, caplog, held=held)
    return result, log_signal, send


class TestHeldSellLoweredToWatchAlerts:
    """A held TRIM/EXIT lowered to WATCH by the malformed-catalyst rule sends
    one Telegram per ticker and action per day; nothing else does."""

    @pytest.mark.parametrize("shape", ["catalyst_str", "date_int"])
    @pytest.mark.parametrize("action", ["TRIM", "EXIT"])
    def test_held_trim_or_exit_alerts(self, shape, action, caplog):
        import json
        changes, fields = _MALFORMED_SHAPES[shape]
        payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   **changes}
        state = {}
        result, _, send = _run_with_alert_state(payload, "Compounder", caplog, True, state)
        assert result["action"] == "WATCH"
        assert send.call_count == 1
        assert send.call_args.args[0] == (
            f"⚠️ <b>SNOW: model said {action} on a held position</b>\n"
            f"The verdict was held as WATCH because of malformed field(s): "
            f"{', '.join(fields)}. The stop is unchanged.")
        assert f"malformed_sell_alert_at_SNOW_{action}" in state

    @pytest.mark.parametrize("action, held", [
        ("TRIM", False), ("EXIT", False), ("EXIT", None),
        ("OPEN", True), ("ADD", True), ("HOLD", True), ("WATCH", True),
    ])
    def test_no_alert_otherwise(self, action, held, caplog):
        import json
        payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        result, _, send = _run_with_alert_state(payload, "Compounder", caplog, held, {})
        assert result["action"] == "WATCH"
        send.assert_not_called()

    @pytest.mark.parametrize("action", ["TRIM", "EXIT"])
    def test_held_sell_with_a_malformed_return_is_not_lowered(self, action, caplog):
        import json
        payload = json.loads(_response(action, None, _DATED, bucket="Accelerator"))
        result, _, send = _run_with_alert_state(payload, "Accelerator", caplog, True, {})
        assert result["action"] == action
        send.assert_not_called()

    def test_well_formed_held_exit_does_not_alert(self, caplog):
        import json
        payload = json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder"))
        result, _, send = _run_with_alert_state(payload, "Compounder", caplog, True, {})
        assert result["action"] == "EXIT"
        send.assert_not_called()

    def test_throttled_to_one_a_day_per_ticker(self, caplog):
        import json
        from datetime import datetime, timedelta, timezone
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        state = {}
        _, _, first = _run_with_alert_state(payload, "Compounder", caplog, True, state)
        _, _, second = _run_with_alert_state(payload, "Compounder", caplog, True, state)
        assert first.call_count == 1 and second.call_count == 0
        state["malformed_sell_alert_at_SNOW_EXIT"] = (
            datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        _, _, later = _run_with_alert_state(payload, "Compounder", caplog, True, state)
        assert later.call_count == 1

    def test_alert_failure_keeps_the_verdict(self, caplog):
        import json
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        with patch("db.get_state_strict", side_effect=RuntimeError("db down")), \
             patch("db.set_state") as set_state, patch("alerts._send") as send:
            result, log_signal, _ = _run_analyze(payload, "Compounder", caplog, held=True)
        send.assert_not_called()
        set_state.assert_not_called()
        assert any("sell alert skipped: state read failed" in r.getMessage()
                   for r in caplog.records if r.levelname == "WARNING")
        assert result["action"] == "WATCH"
        assert log_signal.call_args.kwargs["decision"]["downgraded_reason"] == "malformed_catalyst"

    def test_throttle_is_per_action(self, caplog):
        import json
        state = {}
        for action, sends in (("EXIT", 1), ("TRIM", 1), ("EXIT", 0)):
            payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                       "catalyst": "earnings next month"}
            _, _, send = _run_with_alert_state(payload, "Compounder", caplog, True, state)
            assert send.call_count == sends
        assert set(state) == {"malformed_sell_alert_at_SNOW_EXIT", "malformed_sell_alert_at_SNOW_TRIM"}

    def test_state_is_written_only_after_a_send_that_went_out(self, caplog):
        import json
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        state = {}
        with patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("alerts._send", return_value=None) as send:
            _run_analyze(payload, "Compounder", caplog, held=True)
            _run_analyze(payload, "Compounder", caplog, held=True)
        assert send.call_count == 2 and state == {}

    @pytest.mark.parametrize("hours_ago, sends", [(1, 0), (25, 1), (None, 1)])
    def test_stored_time_is_read_robustly(self, hours_ago, sends, caplog):
        """A naive stored time is UTC; an unreadable one sends."""
        import json
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        naive = (datetime.now(timezone.utc) - timedelta(hours=hours_ago or 0)).replace(tzinfo=None)
        state = {"malformed_sell_alert_at_SNOW_EXIT":
                 naive.isoformat() if hours_ago is not None else "not a time"}
        _, _, send = _run_with_alert_state(payload, "Compounder", caplog, True, state)
        assert send.call_count == sends

    def test_a_non_string_stored_time_sends(self, caplog):
        import json
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        state = {"malformed_sell_alert_at_SNOW_EXIT": 12345}     # fromisoformat: TypeError
        _, _, send = _run_with_alert_state(payload, "Compounder", caplog, True, state)
        assert send.call_count == 1

    @pytest.mark.parametrize("earnings_date", [None, "2026-11-02"])
    def test_review_held_position_alerts_on_a_malformed_exit(self, earnings_date, caplog):
        """The scheduled and earnings reviews build a candidate without
        "held"; held_review makes the alert fire, and the held flag recorded
        for the gates stays None."""
        import json
        from datetime import date
        from agents import crew
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month", "pass_to_risk": True}
        state = {}
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8], \
             patch("context_engine.assemble_brief", return_value="brief"), \
             patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}), \
             patch("agents.crew._capture_latest_thesis"), \
             patch("agents.crew._route_to_approval") as route, \
             patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("alerts._send", return_value=7) as send:
            result = crew.review_held_position(
                "SNOW", None, {"SNOW": {"current_price": 100.0}}, {"regime": "Neutral"},
                earnings_date=date.fromisoformat(earnings_date) if earnings_date else None)
        assert result["action"] == "WATCH" and result["pass_to_risk"] is False
        route.assert_not_called()
        assert send.call_count == 1
        assert "SNOW: model said EXIT on a held position" in send.call_args.args[0]
        assert "malformed_sell_alert_at_SNOW_EXIT" in state
        exposure = log_signal.call_args.kwargs["provenance"]["brief_exposure"]
        assert exposure["held_at_gate"] is None
        assert log_signal.call_args.kwargs["decision"]["held_review"] is True

    def test_backfill_does_not_alert(self, caplog):
        import json
        from agents import crew
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        patches = _mock_analyze_deps(json.dumps(payload))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7] as log_signal, patches[8], \
             patch("context_engine.assemble_brief", return_value="brief"), \
             patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}), \
             patch("db.get_state_strict", return_value=None), patch("db.set_state"), \
             patch("alerts._send", return_value=7) as send:
            result = crew._analyze_held_position("SNOW", None, {"SNOW": {"current_price": 100.0}},
                                                 {"regime": "Neutral"}, path="backfill")
        assert result["action"] == "WATCH"
        send.assert_not_called()
        assert log_signal.call_args.kwargs["decision"]["held_review"] is False

    @pytest.mark.parametrize("held, stored, read_fails, send_id, want", [
        (True, None, False, 7, "sent"),
        (True, "recent", False, 7, "throttled"),
        (True, None, True, 7, "skipped_state_error"),
        (True, None, False, None, "send_failed"),
        (True, None, False, RuntimeError("telegram down"), "send_failed"),
        (False, None, False, 7, None),
    ])
    def test_decision_records_the_alert_outcome(self, held, stored, read_fails, send_id, want,
                                                caplog):
        import json
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        state = {}
        if stored == "recent":
            state["malformed_sell_alert_at_SNOW_EXIT"] = datetime.now(timezone.utc).isoformat()
        read = (patch("db.get_state_strict", side_effect=RuntimeError("db down")) if read_fails
                else patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)))
        send = (patch("alerts._send", side_effect=send_id) if isinstance(send_id, Exception)
                else patch("alerts._send", return_value=send_id))
        with read, send, patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
            result, log_signal, _ = _run_analyze(payload, "Compounder", caplog, held=held)
        assert result["action"] == "WATCH"
        assert log_signal.call_args.kwargs["decision"]["malformed_sell_alert"] == want

    def test_an_error_before_the_send_is_error_not_send_failed(self, caplog):
        import json
        import sys
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        with patch("db.get_state_strict", return_value=None), patch("db.set_state") as set_state, \
             patch.dict(sys.modules, {"alerts": None}):     # import alerts raises
            result, log_signal, _ = _run_analyze(payload, "Compounder", caplog, held=True)
        assert result["action"] == "WATCH"
        assert log_signal.call_args.kwargs["decision"]["malformed_sell_alert"] == "error"
        set_state.assert_not_called()

    def test_well_formed_verdict_records_no_alert(self, caplog):
        import json
        payload = json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder"))
        _, log_signal, _ = _run_with_alert_state(payload, "Compounder", caplog, True, {})
        assert log_signal.call_args.kwargs["decision"]["malformed_sell_alert"] is None

    def test_scan_alerts_on_a_held_name_only_with_the_same_prompt(self, caplog):
        """analyze_batch marks a held scan name held_review from the positions
        it is given; the prompt, the recorded held flag and the decision are
        as for an unheld one."""
        import json
        from agents import analyst
        payload = {**json.loads(_response("EXIT", 20.0, _THREE_WEEKS_OUT, bucket="Compounder")),
                   "catalyst": "earnings next month"}
        runs = {}
        for name, positions in (("held", {"SNOW": {"current_price": 100.0}}),
                                ("unheld", {"NVDA": {"current_price": 100.0}})):
            state = {}
            patches = _mock_analyze_deps(json.dumps(payload))
            cand = {**_CANDIDATE, "role": "Compounder"}
            with patches[0] as call, patches[1], patches[2], patches[3], patches[4], \
                 patches[5], patches[6], patches[7] as log_signal, patches[8], \
                 patch("context_engine.build_scan_brief", return_value="brief"), \
                 patch("agents.crew._get_price", return_value=100.0), \
                 patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
                 patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
                 patch("alerts._send", return_value=7) as send:
                results = analyst.analyze_batch([cand], positions=positions)
            assert "held_review" not in cand
            assert results[0]["action"] == "WATCH"
            kwargs = log_signal.call_args.kwargs
            runs[name] = (send.call_count, call.call_args, kwargs["prompt"],
                          kwargs["provenance"]["brief_exposure"]["held_at_gate"],
                          {k: v for k, v in kwargs["decision"].items()
                           if k not in ("malformed_sell_alert", "held_review", "latency_ms")},
                          kwargs["decision"]["held_review"])
        assert runs["held"][0] == 1 and runs["unheld"][0] == 0
        assert runs["held"][1] == runs["unheld"][1]
        assert runs["held"][2] == runs["unheld"][2]
        assert runs["held"][3] is None and runs["unheld"][3] is None
        assert runs["held"][4] == runs["unheld"][4]
        assert runs["held"][5] is True and runs["unheld"][5] is False


@pytest.mark.parametrize("action", ["HOLD", "OPEN"])
def test_catalyst_stale_is_not_computed_on_a_lowered_verdict(action, caplog):
    import json
    payload = {**json.loads(_response(action, 20.0, _THREE_WEEKS_OUT, bucket="Accelerator")),
               "catalyst": "earnings next month", "pass_to_risk": action == "OPEN"}
    result, _, _ = _run_analyze(payload, "Accelerator", caplog, held=True)
    assert result["action"] == "WATCH" and result["catalyst_stale"] is None
    ok = json.loads(_response("HOLD", 20.0, _THREE_WEEKS_OUT, bucket="Accelerator"))
    ok["pass_to_risk"] = False
    result, _, _ = _run_analyze(ok, "Accelerator", caplog, held=True)
    assert result["action"] == "HOLD" and result["catalyst_stale"] is False


@pytest.mark.parametrize("shape", sorted(_RETURN_SHAPES))
@pytest.mark.parametrize("role", ["Accelerator", "Moonshot"])
def test_held_hold_with_a_malformed_return_stays_hold_flagged_stale(shape, role, caplog):
    """Not lowered to WATCH: flagged catalyst_stale with the stale-HOLD note,
    no malformed gate or WARNING; return_malformed records the flag."""
    import json
    from agents import analyst
    payload = {**json.loads(_response("HOLD", _RETURN_SHAPES[shape], _DATED, bucket=role)),
               "pass_to_risk": False}
    with patch.object(analyst.log, "info") as info:
        result, log_signal, warnings = _run_analyze(payload, role, caplog, held=True)
    assert result["action"] == "HOLD" and result["catalyst_stale"] is True
    assert result["expected_return_per_week"] is None
    assert "[STALE CATALYST ON HOLD: no expected_return_pct]" in result["reasoning"]
    assert any("HOLD carries a stale catalyst (no expected_return_pct)" in c.args[0]
               for c in info.call_args_list)
    assert warnings == []
    decision = log_signal.call_args.kwargs["decision"]
    assert decision["final_action"] == "HOLD" and decision["gates"] == []
    assert decision["return_malformed"] is True
    assert decision["malformed_fields"] is None and decision["downgraded_reason"] is None


def test_failed_analysis_rows_carry_the_malformed_keys_as_none():
    from agents import analyst
    with patch("db.log_signal") as log_signal:
        analyst._log_failed_analysis({"ticker": "NVDA"}, "timeout", "Ollama timed out", prompt="p")
    decision = log_signal.call_args.kwargs["decision"]
    assert decision["downgraded_reason"] is None and decision["malformed_fields"] is None
    assert decision["return_malformed"] is None and decision["malformed_sell_alert"] is None
    assert decision["held_review"] is False
    with patch("db.log_signal") as log_signal:
        analyst._log_failed_analysis({"ticker": "NVDA", "held_review": True}, "timeout", "t")
    assert log_signal.call_args.kwargs["decision"]["held_review"] is True
    assert decision["failure"]["kind"] == "timeout"


class TestMalformedReturnOnlyAtTheRateGate:
    """A dated catalyst with an expected_return_pct that is null,
    non-numeric, NaN/+-inf or a bool is malformed only where the rate gate
    reads it: Accelerator/Moonshot with OPEN or ADD at that gate.
    Everywhere else the verdict routes as it would with no malformed check,
    and decision.return_malformed (log-only) records the flag."""

    @pytest.mark.parametrize("shape", sorted(_RETURN_SHAPES))
    @pytest.mark.parametrize("action", ["OPEN", "ADD"])
    @pytest.mark.parametrize("role", ["Accelerator", "Moonshot"])
    @pytest.mark.parametrize("held", [None, True])
    def test_rate_gated_verdict_becomes_an_unrouted_watch(self, shape, action, role, held, caplog):
        import json
        erp = _RETURN_SHAPES[shape]
        payload = json.loads(_response(action, erp, _DATED, bucket=role))
        result, log_signal, warnings = _run_analyze(payload, role, caplog, held=held)
        assert result["action"] == "WATCH" and result["pass_to_risk"] is False
        assert len(warnings) == 1 and "expected_return_pct" in warnings[0]
        assert log_signal.call_count == 1
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["final_action"] == "WATCH" and decision["final_pass_to_risk"] is False
        assert decision["downgraded_reason"] == "malformed_catalyst"
        assert decision["malformed_fields"] == ["expected_return_pct"]
        assert decision["gates"] == [f"malformed catalyst (expected_return_pct): {action} -> WATCH"]
        assert decision["model_output"]["expected_return_pct"] == erp
        assert decision["return_malformed"] is True

    @pytest.mark.parametrize("shape", sorted(_RETURN_SHAPES))
    @pytest.mark.parametrize("action, role", [
        *[(a, r) for a in ("EXIT", "TRIM", "WATCH")
          for r in ("Compounder", "Accelerator", "Moonshot")],
        ("OPEN", "Compounder"), ("ADD", "Compounder"), ("HOLD", "Compounder"),
    ])
    def test_other_verdicts_route_normally(self, shape, action, role, caplog):
        import json
        erp = _RETURN_SHAPES[shape]
        payload = json.loads(_response(action, erp, _DATED, bucket=role))
        if action in ("HOLD", "WATCH"):
            payload["pass_to_risk"] = False
        result, log_signal, warnings = _run_analyze(payload, role, caplog)
        assert result["action"] == action
        assert result["pass_to_risk"] is payload["pass_to_risk"]
        assert warnings == []
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["downgraded_reason"] is None
        assert decision["malformed_fields"] is None
        assert decision["gates"] == []
        assert decision["return_malformed"] is True

    @pytest.mark.parametrize("held, gates", [
        (True, ["malformed catalyst (expected_return_pct): ADD -> WATCH"]),
        (False, ["unheld ADD -> OPEN", "malformed catalyst (expected_return_pct): OPEN -> WATCH"]),
    ])
    def test_add_uses_the_action_after_the_unheld_coercion(self, held, gates, caplog):
        import json
        payload = json.loads(_response("ADD", None, _DATED))
        result, log_signal, warnings = _run_analyze(payload, "Accelerator", caplog, held=held)
        assert result["action"] == "WATCH" and len(warnings) == 1
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["gates"] == gates
        assert decision["malformed_fields"] == ["expected_return_pct"]

    def test_unheld_hold_is_a_watch_not_malformed(self, caplog):
        import json
        payload = {**json.loads(_response("HOLD", None, _DATED)), "pass_to_risk": False}
        result, log_signal, warnings = _run_analyze(payload, "Accelerator", caplog, held=False)
        assert result["action"] == "WATCH" and warnings == []
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["gates"] == ["unheld HOLD -> WATCH"]
        assert decision["downgraded_reason"] is None and decision["malformed_fields"] is None

    def test_a_buy_stopped_before_the_rate_gate_is_not_malformed(self, caplog):
        import json
        payload = {**json.loads(_response("OPEN", None, _DATED)), "upside_downside_ratio": 1.5}
        result, log_signal, warnings = _run_analyze(payload, "Accelerator", caplog)
        assert result["action"] == "WATCH" and warnings == []
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["gates"] == ["udr 1.5 < 2.0: OPEN -> WATCH"]
        assert decision["downgraded_reason"] is None and decision["malformed_fields"] is None
        assert decision["return_malformed"] is True


class TestWellFormedCatalystUnchanged:
    """No malformed field: no WARNING, no malformed gate, the decision
    record's downgraded_reason and malformed_fields are None and
    return_malformed is False."""

    @pytest.mark.parametrize("action, erp, date, role, want", [
        ("OPEN", 20.0, _THREE_WEEKS_OUT, "Accelerator", "OPEN"),
        ("OPEN", "20", _THREE_WEEKS_OUT, "Accelerator", "OPEN"),
        ("OPEN", 2.0, _THREE_WEEKS_OUT, "Accelerator", "WATCH"),     # rate gate
        ("OPEN", None, "ongoing", "Accelerator", "WATCH"),          # no timeframe
        ("OPEN", None, "ongoing", "Compounder", "OPEN"),
        ("HOLD", 2.0, "ongoing", "Moonshot", "HOLD"),
        ("TRIM", 20.0, _THREE_WEEKS_OUT, "Compounder", "TRIM"),
    ])
    def test_verdict_as_before(self, action, erp, date, role, want, caplog):
        import json
        payload = json.loads(_response(action, erp, date, bucket=role))
        if action in ("HOLD", "TRIM"):
            payload["pass_to_risk"] = action == "TRIM"
        result, log_signal, warnings = _run_analyze(payload, role, caplog)
        assert result["action"] == want
        assert warnings == []
        decision = log_signal.call_args.kwargs["decision"]
        assert decision["downgraded_reason"] is None
        assert decision["malformed_fields"] is None
        assert decision["return_malformed"] is False
        assert not any(g.startswith("malformed catalyst") for g in decision["gates"])
