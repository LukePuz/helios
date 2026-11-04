"""Parity of helios3d.replay.gates with the live analyst gates.

The test imports agents.analyst (the package never does): it runs the live
analyze() with the model call and every I/O read mocked, captures the
decision dict passed to db.log_signal, and checks that replay_gates() on the
same model output and inputs gives the same gate list (text for text),
final action, conviction and pass_to_risk. No DB, no Ollama.
"""
import json
import random
from contextlib import ExitStack
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import agents.analyst as live
from helios3d.replay import gates as G


# ---------------------------------------------------------------------------
# Lifted helpers
# ---------------------------------------------------------------------------

def test_literals_match_live():
    assert G.LOW_CONVICTION_SELL_MAX == live.LOW_CONVICTION_SELL_MAX
    assert G.EXIT_CONVICTION_MAX == live.EXIT_CONVICTION_MAX
    assert G.LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS == live.LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS


@pytest.mark.parametrize("sources", [
    None, [], [{"type": "news", "detail": "a"}], [{"type": "news", "detail": "a"}] * 3,
    [{"type": "news", "detail": "a"}, {"type": "news", "detail": "b"}],
    [{"type": "news", "detail": "x" * 90}, {"type": "news", "detail": "x" * 85}],
    [{"type": None, "detail": "a"}, "text", {"detail": "b"}, {"type": "filing", "detail": None}],
])
def test_count_distinct_sources(sources):
    assert G.count_distinct_sources(sources) == live.count_distinct_sources(sources)


_AS_OF = datetime(2026, 10, 30, 14, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw", [None, "", "ongoing", " Ongoing ", "2026-11-20", "2026-10-30",
                                 "2026-10-01", "2026-12", "2027-01", "Q4 2026", "2026-13-01",
                                 " 2026-11-05 "])
@pytest.mark.parametrize("midpoint", [False, True])
def test_weeks_to_catalyst(raw, midpoint):
    assert (G.compute_weeks_to_catalyst(raw, _AS_OF, midpoint)
            == live.compute_weeks_to_catalyst(raw, as_of=_AS_OF, month_midpoint=midpoint))


@pytest.mark.parametrize("ret,weeks", [(None, 1.0), (5.0, None), (5.0, 0), (5.0, -1), (5.0, 2.5),
                                       (-3.0, 1.0)])
def test_expected_return_per_week(ret, weeks):
    assert G.expected_return_per_week(ret, weeks) == live.expected_return_per_week(ret, weeks)


def _prev(conv, days_ago):
    ts = None if days_ago is None else datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {"signal": "TRIM", "confidence": conv / 10, "timestamp": ts}


@pytest.mark.parametrize("action", ["TRIM", "EXIT"])
@pytest.mark.parametrize("conviction", [1, 2, 3, 4])
@pytest.mark.parametrize("prev", [None, _prev(3, 1), _prev(4, None), _prev(5, 1), _prev(2, 10),
                                  _prev(1, 6.5)])
def test_low_conviction_sell(action, conviction, prev):
    with patch("db.get_latest_signal_for", return_value=prev):
        want = live._low_conviction_sell("QBTS", action, conviction)
    assert G.low_conviction_sell(action, conviction, prev, datetime.now(timezone.utc)) == want


def test_low_conviction_age_limit_is_exclusive():
    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    at = {"signal": "TRIM", "confidence": 0.3, "timestamp": now - timedelta(days=7)}
    past = {"signal": "TRIM", "confidence": 0.3, "timestamp": now - timedelta(days=7, seconds=1)}
    assert G.low_conviction_sell("TRIM", 3, at, now)[:2] == ("TRIM", True)
    assert G.low_conviction_sell("TRIM", 3, past, now)[:2] == ("TRIM", False)


def test_role_line_pattern_matches_the_rendered_prompt():
    """The harness reads the bucket with E'\\nPortfolio Role: (...)\\n'; the
    same pattern finds the role in a prompt the live template renders."""
    import re
    from helios3d.replay import harness as H
    m = re.search(r"E'(.*)'\) ", H.SIGNALS_BY_ID_SQL)
    pattern = m.group(1).replace("\\n", "\n")   # the E-string escape, as Postgres reads it
    assert pattern.startswith("\nPortfolio Role: (") and pattern.endswith(")\n")
    with patch("db.get_conviction_calibration", return_value=[]):
        for role in ("Compounder", "Accelerator", "Moonshot", "Speculative"):
            prompt = live._build_prompt({"ticker": "TST", "role": role, "tier": "Watch",
                                         "sector": "Software"}, "brief", "", {"regime": "Neutral"}, "")
            found = re.search(pattern, prompt)
            assert (found.group(1) if found else None) == (role if role != "Speculative" else None)


# ---------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------

def _model_output(rng):
    today = date.today()
    cat = rng.choice([None, {}, {"date": "ongoing"},
                      {"date": (today + timedelta(days=rng.choice([10, 30, 90]))).isoformat()},
                      {"date": (today - timedelta(days=3)).isoformat()},
                      {"date": (today + timedelta(days=60)).strftime("%Y-%m")}])
    out = {
        "action": rng.choice(["OPEN", "ADD", "HOLD", "WATCH", "TRIM", "EXIT", "PASS", "open"]),
        "conviction": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "thesis_strength": 3, "catalyst_quality": 3, "valuation_risk_reward": 3,
        "data_completeness": 3, "thesis_bucket": "Accelerator", "thesis_summary": "t",
        "expectation_gap": {}, "valuation": {},
        "upside_downside_ratio": rng.choice([0, 1.5, 1.95, 2.0, 3.2]),
        "expected_return_pct": rng.choice([None, 2.0, 6.0, 25.0]),
        "key_risks": [], "pass_to_risk": rng.choice([True, False]),
        "reasoning": "r",
        "sources": rng.choice([[], [{"type": "news", "detail": "a"}],
                               [{"type": "news", "detail": "a"}, {"type": "filing", "detail": "b"}]]),
    }
    if cat is not None:
        out["catalyst"] = cat
    return out


def _case(rng):
    return {
        "model_output": _model_output(rng),
        "role": rng.choice(["Accelerator", "Moonshot", "Compounder"]),
        "held": rng.choice([True, False, "absent"]),
        "regime": rng.choice(["Neutral", "Risk-On", "Risk-Off", "Strong Risk-Off", "Strong Risk-On"]),
        "fund": rng.choice(["ok", "missing", "error"]),
        "trend": rng.choice([None, True, False]),
        "prev": rng.choice([None, (3, 1), (4, 2), (6, 1), (2, 9)]),
        "etf": rng.choice([False, True]),
        "etf_exempt": rng.choice([False, True]),
    }


_TARGETED = [
    # OPEN c6 in a cautious regime, Moonshot c6, unheld ADD/HOLD, fundamentals cap,
    # ETF exemption, low-conviction EXIT, pass dropped on WATCH.
    dict(action="OPEN", conviction=6, role="Compounder", regime="Neutral"),
    dict(action="OPEN", conviction=6, role="Moonshot", regime="Risk-On"),
    dict(action="ADD", conviction=8, role="Compounder", held=False),
    dict(action="HOLD", conviction=5, role="Compounder", held=False),
    dict(action="OPEN", conviction=9, role="Compounder", fund="missing"),
    dict(action="OPEN", conviction=9, role="Compounder", fund="missing", etf=True, etf_exempt=True),
    dict(action="OPEN", conviction=9, role="Compounder", fund="error", etf=True),
    dict(action="EXIT", conviction=3, role="Compounder", prev=(3, 1)),
    dict(action="EXIT", conviction=2, role="Compounder", prev=None),
    dict(action="WATCH", conviction=8, role="Compounder"),
    dict(action="OPEN", conviction=8, role="Accelerator", trend=False),
    # Boundaries of the c<7 and c>6 rules.
    dict(action="OPEN", conviction=7, role="Moonshot", regime="Risk-On"),
    dict(action="OPEN", conviction=10, role="Moonshot", regime="Risk-On"),
    dict(action="OPEN", conviction=7, role="Compounder", regime="Neutral"),
    dict(action="OPEN", conviction=10, role="Compounder", regime="Strong Risk-Off"),
    dict(action="OPEN", conviction=6, role="Compounder", regime="Strong Risk-Off"),
    dict(action="OPEN", conviction=7, role="Compounder", fund="missing"),
    dict(action="OPEN", conviction=10, role="Compounder", fund="error"),
    dict(action="ADD", conviction=7, role="Compounder", fund="missing", etf=True, etf_exempt=True),
    dict(action="OPEN", conviction=6, role="Compounder", fund="missing"),
    # The expected-return field and the catalyst in odd shapes.
    dict(action="OPEN", conviction=8, role="Accelerator", ret="high"),
    dict(action="OPEN", conviction=8, role="Moonshot", ret=None),
    dict(action="HOLD", conviction=6, role="Accelerator", ret=None),
    dict(action="OPEN", conviction=8, role="Accelerator", catalyst="Q4 earnings"),
    dict(action="OPEN", conviction=8, role="Accelerator", catalyst=[]),
    dict(action="HOLD", conviction=6, role="Moonshot", catalyst="soon"),
    dict(action="OPEN", conviction=8, role="Compounder", catalyst="Q4 earnings"),
    # Previous verdict exactly at the 7-day confirmation limit.
    dict(action="TRIM", conviction=3, role="Compounder", prev=(3, 7)),
    dict(action="EXIT", conviction=2, role="Compounder", prev=(2, 7)),
    # Passes that go through.
    dict(action="OPEN", conviction=8, role="Compounder"),
    dict(action="ADD", conviction=7, role="Accelerator", trend=None),
    # Non-string catalyst dates: a number is read as its text, a list as no date.
    dict(action="OPEN", conviction=8, role="Accelerator", catalyst={"date": 20261201}),
    dict(action="OPEN", conviction=8, role="Accelerator", catalyst={"date": ["2026-12"]}),
    dict(action="OPEN", conviction=8, role="Moonshot", catalyst={"date": True}),
    # A dated catalyst with no expected return: malformed only at the rate gate.
    dict(action="OPEN", conviction=8, role="Compounder", ret=None),
    dict(action="ADD", conviction=8, role="Compounder", ret=None),
    dict(action="EXIT", conviction=8, role="Accelerator", ret=None),
    dict(action="TRIM", conviction=8, role="Moonshot", ret=None),
    dict(action="ADD", conviction=8, role="Moonshot", held=False, ret=None),
    dict(action="ADD", conviction=8, role="Accelerator", ret=None),
    dict(action="WATCH", conviction=8, role="Moonshot", ret=None),
]


def _targeted_case(spec):
    mo = {"action": spec["action"], "conviction": spec["conviction"], "upside_downside_ratio": 3.0,
          "expected_return_pct": spec.get("ret", 25.0), "pass_to_risk": True,
          "catalyst": spec.get("catalyst", {"date": (date.today() + timedelta(days=14)).isoformat()}),
          "sources": [{"type": "news", "detail": "a"}, {"type": "filing", "detail": "b"}]}
    return {"model_output": mo, "role": spec["role"], "held": spec.get("held", True),
            "regime": spec.get("regime", "Risk-On"), "fund": spec.get("fund", "ok"),
            "trend": spec.get("trend", True), "prev": spec.get("prev"),
            "etf": spec.get("etf", False), "etf_exempt": spec.get("etf_exempt", False)}


def _run_live(case):
    import config
    ticker = "TST"
    cand = {"ticker": ticker, "role": case["role"], "tier": "Watch", "sector": "Software"}
    if case["held"] != "absent":
        cand["held"] = case["held"]
    prev = None
    if case["prev"] is not None:
        c, days = case["prev"]
        prev = {"signal": "TRIM", "confidence": c / 10,
                "timestamp": datetime.now(timezone.utc) - timedelta(days=days)}
    fund_kw = ({"return_value": {"market_cap": 1_000_000_000}} if case["fund"] == "ok" else
               {"return_value": None} if case["fund"] == "missing" else
               {"side_effect": RuntimeError("down")})
    tt = None if case["trend"] is None else {"pass": case["trend"], "criteria": {}, "reasoning": "x"}
    patches = [
        patch("agents.analyst._call_ollama", return_value=json.dumps(case["model_output"])),
        patch("agents.analyst.SETUP_GATES_ENABLED", False),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_etf_tickers", return_value={ticker} if case["etf"] else set()),
        patch.object(config, "ETF_FUNDAMENTALS_CAP_EXEMPT", case["etf_exempt"]),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("agents.price_action.compute_trend_template", return_value=tt),
        patch("db.get_fundamentals", **fund_kw),
        patch("db.get_latest_signal_for", return_value=prev),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("feed.emit"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        log_signal = stack.enter_context(patch("db.log_signal"))
        live.analyze(cand, situation_brief="brief",
                     regime_packet={"regime": case["regime"], "score": 50, "reasoning": ""})
    verdicts = [c for c in log_signal.call_args_list if c.kwargs.get("decision")]
    if not verdicts:
        # The verdict row was never written (building its decision dict
        # raised, e.g. _catalyst_check on a non-dict catalyst; live logs
        # "Signal log failed"): no record exists to replay.
        return None, prev
    assert len(verdicts) == 1
    return verdicts[0].kwargs["decision"], prev


def _replay(case, prev):
    import config
    fund_read = {"ok": "ok", "missing": "missing", "error": "read error: RuntimeError"}[case["fund"]]
    inp = {"held": None if case["held"] == "absent" else case["held"], "bucket": case["role"],
           "regime": case["regime"], "fund_read": fund_read, "is_etf": case["etf"],
           "trend_template_pass": case["trend"], "prev_verdict": prev,
           "as_of": datetime.now(timezone.utc),
           "min_return_per_week": live.MIN_EXPECTED_RETURN_PCT_PER_WEEK,
           "month_midpoint": config.CATALYST_MONTH_MIDPOINT_ENABLED,
           "etf_exempt": case["etf_exempt"],
           "malformed_catalyst_policy": live.MALFORMED_CATALYST_POLICY}
    return G.replay_gates(case["model_output"], inp)


def _assert_same(case):
    try:
        want, prev = _run_live(case)
    except Exception as e:
        # Live analyze() raised (it logs ANALYST_FAILED, no verdict): the
        # replay must raise the same exception type on the same output.
        with pytest.raises(type(e)):
            _replay(case, None)
        return
    if want is None:
        return
    got = _replay(case, prev)
    for k in ("model_action", "model_conviction", "model_pass_to_risk", "final_action",
              "final_conviction", "final_pass_to_risk", "gates"):
        assert got[k] == want[k], (k, case, got[k], want[k])
    assert all(s["status"] != G.ORACLE for s in got["steps"])


_RNG = random.Random(20261103)
_RANDOM_CASES = [_case(_RNG) for _ in range(240)]


@pytest.mark.parametrize("spec", _TARGETED, ids=lambda s: "-".join(f"{k}={v}" for k, v in s.items()))
def test_chain_parity_targeted(spec):
    _assert_same(_targeted_case(spec))


@pytest.mark.parametrize("i", range(len(_RANDOM_CASES)))
def test_chain_parity_grid(i):
    _assert_same(_RANDOM_CASES[i])


def test_grid_reaches_every_gate():
    """The grid is only evidence for gates it actually fires."""
    kinds, finals = set(), set()
    for case in _RANDOM_CASES + [_targeted_case(s) for s in _TARGETED]:
        prev = None
        if case["prev"] is not None:
            c, days = case["prev"]
            prev = {"signal": "TRIM", "confidence": c / 10,
                    "timestamp": datetime.now(timezone.utc) - timedelta(days=days)}
        try:
            r = _replay(case, prev)
        except Exception:
            continue
        kinds |= {G.gate_kind(g) for g in r["gates"]}
        finals.add((r["final_action"], r["final_pass_to_risk"]))
    assert kinds >= {"malformed_catalyst", "unheld", "udr", "sources", "return_rate", "trend_template", "moonshot",
                     "regime", "fundamentals_cap", "fundamentals_etf_exempt", "pass_dropped",
                     "low_conviction_sell"}
    assert {("OPEN", True), ("ADD", True), ("TRIM", True), ("WATCH", False)} <= finals


# ---------------------------------------------------------------------------
# Oracle steps reproduce the record when their input is unknown
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drop", ["held", "bucket", "regime", "fund_read", "is_etf",
                                  "trend_template_pass", "prev_verdict",
                                  "malformed_catalyst_policy"])
def test_oracle_reproduces_live_record(drop):
    for i in range(0, len(_RANDOM_CASES), 3):
        case = _RANDOM_CASES[i]
        try:
            want, prev = _run_live(case)
        except Exception:
            continue
        if want is None:
            continue
        import config
        fund_read = {"ok": "ok", "missing": "missing", "error": "read error: RuntimeError"}[case["fund"]]
        inp = {"held": None if case["held"] == "absent" else case["held"], "bucket": case["role"],
               "regime": case["regime"], "fund_read": fund_read, "is_etf": case["etf"],
               "trend_template_pass": case["trend"], "prev_verdict": prev,
               "as_of": datetime.now(timezone.utc),
               "min_return_per_week": live.MIN_EXPECTED_RETURN_PCT_PER_WEEK,
               "month_midpoint": config.CATALYST_MONTH_MIDPOINT_ENABLED,
               "etf_exempt": case["etf_exempt"],
               "malformed_catalyst_policy": live.MALFORMED_CATALYST_POLICY}
        inp.pop(drop)
        got = G.replay_gates(case["model_output"], inp, want["gates"])
        for k in ("final_action", "final_conviction", "final_pass_to_risk", "gates"):
            assert got[k] == want[k], (drop, k, case)


_BOUNDARY_EXPECT = [
    # (spec index in _TARGETED, expected live outcome)
    (11, ("OPEN", 7, True)), (12, ("OPEN", 10, True)), (13, ("OPEN", 7, True)),
    (14, ("OPEN", 10, True)), (15, ("WATCH", 6, False)), (16, ("OPEN", 6, True)),
    (17, ("OPEN", 6, True)), (18, ("ADD", 7, True)), (19, ("OPEN", 6, True)),
    # 20-21 and 38-39: a malformed expected_return_pct on a rate-gated
    # OPEN/ADD, 23-26 and 31-33: a malformed catalyst or catalyst.date on
    # any verdict make it an unrouted WATCH. 22: an Accelerator HOLD with a
    # null return stays HOLD (flagged catalyst_stale). 34-37 and 40: a null
    # return elsewhere routes as usual.
    (20, ("WATCH", 8, False)), (21, ("WATCH", 8, False)), (22, ("HOLD", 6, False)),
    (23, ("WATCH", 8, False)), (24, ("WATCH", 8, False)), (25, ("WATCH", 6, False)),
    (26, ("WATCH", 8, False)),
    (27, ("TRIM", 3, False)), (28, ("EXIT", 2, False)), (29, ("OPEN", 8, True)),
    (30, ("ADD", 7, True)), (31, ("WATCH", 8, False)), (32, ("WATCH", 8, False)),
    (33, ("WATCH", 8, False)),
    (34, ("OPEN", 8, True)), (35, ("ADD", 8, True)), (36, ("EXIT", 8, True)),
    (37, ("TRIM", 8, True)), (38, ("WATCH", 8, False)), (39, ("WATCH", 8, False)),
    (40, ("WATCH", 8, False)),
]


@pytest.mark.parametrize("idx,expect", _BOUNDARY_EXPECT)
def test_boundary_cases_reach_live_as_intended(idx, expect):
    """The boundary cases are compared, not skipped: each gives the live
    outcome it was written for (and test_chain_parity_targeted checks the
    replay against it)."""
    case = _targeted_case(_TARGETED[idx])
    if isinstance(expect, type):
        with pytest.raises(expect):
            _run_live(case)
        return
    want, _ = _run_live(case)
    if expect == "no row":
        assert want is None
        return
    assert (want["final_action"], want["final_conviction"], want["final_pass_to_risk"]) == expect
    malformed = {20: "expected_return_pct", 21: "expected_return_pct",
                 23: "catalyst", 24: "catalyst", 25: "catalyst", 26: "catalyst",
                 31: "catalyst.date", 32: "catalyst.date", 33: "catalyst.date",
                 38: "expected_return_pct", 39: "expected_return_pct"}
    if idx in malformed:
        at = 1 if idx == 38 else 0          # after "unheld ADD -> OPEN"
        assert want["gates"][at].startswith(f"malformed catalyst ({malformed[idx]}): ")
        assert want["downgraded_reason"] == "malformed_catalyst"
        assert want["malformed_fields"] == [malformed[idx]]
    else:
        assert want["downgraded_reason"] is None and want["malformed_fields"] is None
    if idx == 22:       # the HOLD keeps its action; only the log-only flag is set
        assert want["return_malformed"] is True
        assert not any(g.startswith("malformed") for g in want["gates"])


def test_catalyst_normalisation_keeps_model_output_raw():
    mo = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
          "catalyst": {"date": 20261201}, "expected_return_pct": 25,
          "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
    G.replay_gates(mo, {"bucket": "Accelerator", "as_of": _AS_OF, "min_return_per_week": 1.0,
                        "month_midpoint": False, "trend_template_pass": True})
    assert mo["catalyst"] == {"date": 20261201}
    for raw, want in (("x", {}), (["2026-12"], {}), (None, None), ({"date": 1.5}, {"date": "1.5"}),
                      ({"date": False}, {"date": None}), ({"date": {"d": 1}}, {"date": None})):
        parsed = {"catalyst": raw}
        G.normalise_catalyst(parsed)
        assert parsed["catalyst"] == want


def test_rate_step_not_applicable_for_known_non_rate_bucket_without_config():
    mo = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
          "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
    out = G.replay_gates(mo, {"bucket": "Compounder", "as_of": _AS_OF})
    rr = next(s for s in out["steps"] if s["step"] == "return_rate")
    assert rr["status"] == G.NOT_APPLICABLE and rr["reason"] == "bucket Compounder"
    out = G.replay_gates(mo, {"bucket": "Moonshot", "as_of": _AS_OF})
    assert next(s for s in out["steps"] if s["step"] == "return_rate")["status"] == G.ORACLE


def test_malformed_catalyst_fields_match_live():
    from agents.analyst import _malformed_catalyst_fields
    for case in _RANDOM_CASES + [_targeted_case(t) for t in _TARGETED]:
        mo = case["model_output"]
        assert G.malformed_catalyst_fields(mo) == _malformed_catalyst_fields(mo)
    for extra in ({"catalyst": {"date": "2026-12"}, "expected_return_pct": True},
                  {"catalyst": {"date": "2026-12-01"}, "expected_return_pct": "nan"},
                  {"catalyst": {"date": "ongoing"}, "expected_return_pct": None},
                  {"catalyst": 7}, {"catalyst": {"date": 0}}):
        assert G.malformed_catalyst_fields(extra) == _malformed_catalyst_fields(extra)


def test_pre_policy_malformed_row_replays_as_recorded():
    """A decision recorded before the policy (no malformed_catalyst_policy
    in provenance.config, so not in inp) with a malformed catalyst that
    still wrote a row: a Compounder OPEN with a list catalyst.date was
    routed as OPEN on the normalised catalyst. The step is oracle and does
    not fire without a recorded malformed gate, so the replay matches."""
    mo = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
          "catalyst": {"date": ["2026-12"]}, "expected_return_pct": 25,
          "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
    inp = {"held": None, "bucket": "Compounder", "regime": "Risk-On", "fund_read": "ok",
           "as_of": _AS_OF, "min_return_per_week": 1.0, "month_midpoint": False}
    old = G.replay_gates(mo, inp, recorded_gates=[])
    assert (old["final_action"], old["final_pass_to_risk"], old["gates"]) == ("OPEN", True, [])
    step = next(s for s in old["steps"] if s["step"] == "malformed_catalyst")
    assert step["status"] == G.ORACLE and step["fired"] is False
    # The same output under the policy: WATCH; a policy row's recorded gate
    # drives the oracle step the same way.
    new = G.replay_gates(mo, {**inp, "malformed_catalyst_policy": "watch"})
    assert (new["final_action"], new["final_pass_to_risk"]) == ("WATCH", False)
    assert new["gates"] == ["malformed catalyst (catalyst.date): OPEN -> WATCH"]
    oracle = G.replay_gates(mo, inp, recorded_gates=new["gates"])
    assert (oracle["final_action"], oracle["gates"]) == ("WATCH", new["gates"])


_NULL_RETURN = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
                "catalyst": {"date": "2026-12-01"}, "expected_return_pct": None,
                "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
_RATE_INP = {"held": True, "regime": "Risk-On", "fund_read": "ok", "as_of": _AS_OF,
             "min_return_per_week": 1.0, "month_midpoint": False, "trend_template_pass": True,
             "is_etf": False, "prev_verdict": None}


def _step(out, name):
    return next(s for s in out["steps"] if s["step"] == name)


def test_rate_scoped_return_check_needs_bucket_and_buy():
    scoped = {**_RATE_INP, "malformed_catalyst_policy": "watch_rate_scoped"}
    out = G.replay_gates(_NULL_RETURN, {**scoped, "bucket": "Moonshot"})
    assert out["gates"] == ["malformed catalyst (expected_return_pct): OPEN -> WATCH"]
    assert _step(out, "malformed_catalyst")["status"] == G.NOT_APPLICABLE
    assert _step(out, "malformed_return")["status"] == G.REPLAYED
    out = G.replay_gates(_NULL_RETURN, {**scoped, "bucket": "Compounder"})
    assert (out["final_action"], out["gates"]) == ("OPEN", [])
    rs = _step(out, "malformed_return")
    assert rs["status"] == G.NOT_APPLICABLE and rs["reason"] == "bucket Compounder"
    for action in ("EXIT", "TRIM", "WATCH"):
        out = G.replay_gates({**_NULL_RETURN, "action": action}, {**scoped, "bucket": "Accelerator"})
        assert not any(g.startswith("malformed") for g in out["gates"])
        assert _step(out, "malformed_return")["status"] == G.NOT_APPLICABLE
    # An Accelerator/Moonshot HOLD stays HOLD (live flags catalyst_stale).
    hold = {**_NULL_RETURN, "action": "HOLD", "pass_to_risk": False}
    for bucket in ("Moonshot", "Accelerator"):
        out = G.replay_gates(hold, {**scoped, "bucket": bucket})
        assert (out["final_action"], out["gates"]) == ("HOLD", [])
        assert _step(out, "malformed_return")["status"] == G.NOT_APPLICABLE
    out = G.replay_gates(hold, {**scoped, "bucket": "Compounder"})
    assert (out["final_action"], out["gates"]) == ("HOLD", [])
    out = G.replay_gates(hold, {**scoped, "held": False, "bucket": "Moonshot"})
    assert out["gates"] == ["unheld HOLD -> WATCH"]
    # "watch" (unscoped) still downgrades any verdict at step 0
    out = G.replay_gates({**_NULL_RETURN, "action": "EXIT"},
                         {**_RATE_INP, "bucket": "Compounder", "malformed_catalyst_policy": "watch"})
    assert out["gates"] == ["malformed catalyst (expected_return_pct): EXIT -> WATCH"]
    assert _step(out, "malformed_catalyst")["status"] == G.REPLAYED


def test_rate_scoped_return_check_is_oracle_without_bucket():
    scoped = {**_RATE_INP, "malformed_catalyst_policy": "watch_rate_scoped"}
    rec = ["malformed catalyst (expected_return_pct): OPEN -> WATCH"]
    out = G.replay_gates(_NULL_RETURN, scoped, recorded_gates=rec)
    rs = _step(out, "malformed_return")
    assert rs["status"] == G.ORACLE and rs["fired"] and rs["reason"] == "bucket not recorded"
    assert (out["final_action"], out["gates"]) == ("WATCH", rec)
    out = G.replay_gates(_NULL_RETURN, scoped, recorded_gates=[])
    assert _step(out, "malformed_return")["fired"] is False and out["final_action"] == "OPEN"


def test_unknown_policy_places_a_recorded_return_gate_after_the_unheld_one():
    mo = {**_NULL_RETURN, "action": "ADD"}
    rec = ["unheld ADD -> OPEN", "malformed catalyst (expected_return_pct): OPEN -> WATCH"]
    out = G.replay_gates(mo, {**_RATE_INP, "held": False, "bucket": "Moonshot"}, recorded_gates=rec)
    assert (out["final_action"], out["gates"]) == ("WATCH", rec)
    assert _step(out, "malformed_catalyst")["fired"] is False
    assert _step(out, "malformed_return")["status"] == G.ORACLE


def test_an_unknown_policy_string_makes_the_steps_oracle():
    """A recorded malformed_catalyst_policy that is neither "watch" nor
    "watch_rate_scoped" is not acted on: both malformed steps take the
    recorded gates, with the reason naming the policy."""
    inp = {**_RATE_INP, "bucket": "Moonshot", "malformed_catalyst_policy": "watch_v9"}
    bad_date = {**_NULL_RETURN, "catalyst": {"date": 20261201}, "expected_return_pct": 25}
    rec = ["malformed catalyst (catalyst.date): OPEN -> WATCH"]
    out = G.replay_gates(bad_date, inp, recorded_gates=rec)
    s0 = _step(out, "malformed_catalyst")
    assert s0 == {"step": "malformed_catalyst", "status": G.ORACLE, "fired": True,
                  "reason": "unknown policy watch_v9"}
    assert (out["final_action"], out["gates"]) == ("WATCH", rec)
    out = G.replay_gates(bad_date, {**inp, "bucket": "Compounder"}, recorded_gates=[])
    assert _step(out, "malformed_catalyst")["fired"] is False and out["final_action"] == "OPEN"
    # A return gate recorded after the unheld one is the malformed_return step's.
    add = {**_NULL_RETURN, "action": "ADD"}
    rec = ["unheld ADD -> OPEN", "malformed catalyst (expected_return_pct): OPEN -> WATCH"]
    out = G.replay_gates(add, {**inp, "held": False}, recorded_gates=rec)
    s0, rs = _step(out, "malformed_catalyst"), _step(out, "malformed_return")
    assert (s0["status"], s0["fired"], s0["reason"]) == (G.ORACLE, False, "unknown policy watch_v9")
    assert (rs["status"], rs["fired"], rs["reason"]) == (G.ORACLE, True, "unknown policy watch_v9")
    assert (out["final_action"], out["gates"]) == ("WATCH", rec)
    # With no policy recorded the reason is unchanged.
    out = G.replay_gates(add, {**_RATE_INP, "held": False, "bucket": "Moonshot"}, recorded_gates=rec)
    assert _step(out, "malformed_return")["reason"] == "malformed_catalyst_policy not recorded"


def test_well_formed_output_has_no_malformed_step():
    mo = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
          "catalyst": {"date": "2026-12-01"}, "expected_return_pct": 25,
          "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
    out = G.replay_gates(mo, {"bucket": "Compounder", "as_of": _AS_OF,
                              "malformed_catalyst_policy": "watch"})
    step = next(s for s in out["steps"] if s["step"] == "malformed_catalyst")
    assert step["status"] == G.NOT_APPLICABLE and out["final_action"] == "OPEN"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_catalyst_date_replays_on_the_recorded_gate(bad):
    """Live flags a NaN/inf catalyst.date; the signals row stores it as
    null (db._finite_json), so the stored output shows no malformed field.
    The first step takes the recorded gate (oracle) and the replay matches."""
    import db
    raw = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
           "catalyst": {"date": bad}, "expected_return_pct": 25,
           "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
    assert live._malformed_catalyst_fields(raw) == ["catalyst.date"]
    stored = db._finite_json(raw)
    assert stored["catalyst"]["date"] is None and G.malformed_catalyst_fields(stored) == []
    recorded = ["malformed catalyst (catalyst.date): OPEN -> WATCH"]
    inp = {"held": None, "bucket": "Compounder", "regime": "Risk-On", "fund_read": "ok",
           "as_of": _AS_OF, "min_return_per_week": 1.0, "month_midpoint": False,
           "malformed_catalyst_policy": "watch"}
    out = G.replay_gates(stored, inp, recorded_gates=recorded)
    assert (out["final_action"], out["final_pass_to_risk"], out["gates"]) == ("WATCH", False, recorded)
    step = next(s for s in out["steps"] if s["step"] == "malformed_catalyst")
    assert step == {"step": "malformed_catalyst", "status": G.ORACLE, "fired": True,
                    "reason": "recorded malformed fields (value not stored)"}
    # Without the recorded gate nothing fires.
    plain = G.replay_gates(stored, inp, recorded_gates=[])
    assert plain["final_action"] == "OPEN" and plain["gates"] == []
    # Under the scoped policy too.
    scoped = G.replay_gates(stored, {**inp, "malformed_catalyst_policy": "watch_rate_scoped"},
                            recorded_gates=recorded)
    assert (scoped["final_action"], scoped["gates"]) == ("WATCH", recorded)


def test_a_recorded_return_gate_is_left_to_the_rate_step():
    """A first recorded malformed gate on expected_return_pct, with the
    stored output still showing it, is the malformed_return step's, not
    the value-not-stored oracle."""
    mo = {"action": "OPEN", "conviction": 8, "upside_downside_ratio": 3, "pass_to_risk": True,
          "catalyst": {"date": "2026-12-01"}, "expected_return_pct": None,
          "sources": [{"type": "a", "detail": "1"}, {"type": "b", "detail": "2"}]}
    recorded = ["malformed catalyst (expected_return_pct): OPEN -> WATCH"]
    inp = {"held": None, "bucket": "Accelerator", "regime": "Risk-On", "fund_read": "ok",
           "as_of": _AS_OF, "min_return_per_week": 1.0, "month_midpoint": False,
           "malformed_catalyst_policy": "watch_rate_scoped"}
    out = G.replay_gates(mo, inp, recorded_gates=recorded)
    steps = {s["step"]: s for s in out["steps"]}
    assert steps["malformed_catalyst"]["status"] == G.NOT_APPLICABLE
    assert steps["malformed_return"]["status"] == G.REPLAYED and steps["malformed_return"]["fired"]
    assert (out["final_action"], out["gates"]) == ("WATCH", recorded)
