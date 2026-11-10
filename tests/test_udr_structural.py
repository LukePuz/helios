"""Log-only udr_structural shadow: the reward-to-risk from compile_plan's
structural stop and targets, beside the model's own UDR in the verdict's
decision record. Nothing routes on it and the verdict dict is unchanged."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents import analyst
from agents.analyst import _udr_structural

_LADDER = {"atr": 2.0, "rungs": [
    {"price": 95.0, "types": ["swing_low"]},
    {"price": 110.0, "types": ["swing_high"], "band_high": 111.0},
]}
_PARSED = {"valuation": {"bull_case_target": "$130", "downside_scenario": "$90"},
           "upside_downside_ratio": 2.4, "thesis_bucket": "Compounder"}
_BRIEF = "Current Price: $100.00\nBRIEF"


def _run(parsed=_PARSED, ladder=_LADDER, brief=_BRIEF, udr=2.4):
    with patch("db.get_trade_ladder", return_value=ladder) as gl:
        out = _udr_structural("MSFT", parsed, udr, brief)
    return out, gl


def test_structural_ratios_from_the_ladder_beside_the_model_udr():
    out, gl = _run()
    assert gl.call_count == 1
    assert out["structural_stop"] == 94.0 and out["t1"] == 110.0 and out["t2"] == 130.0
    assert out["udr_structural"] == round(10 / 6, 2) and out["udr_structural_t2"] == 5.0
    assert out["model_udr"] == 2.4 and out["model_udr_stated"] is True
    assert out["plan_source"] == "ladder" and out["stop_from_support"] is True
    assert "why_null" not in out and "error" not in out


def test_invalid_levels_give_null_ratios():
    out, _ = _run(ladder={"atr": 2.0, "rungs": [{"price": 95.0, "types": ["s"]}]})
    assert out["t1"] is None and out["udr_structural"] is None and out["why_null"] == "no t1"
    assert out["udr_structural_t2"] == 5.0           # t2 still from the bull target
    out, _ = _run(parsed={"valuation": {}}, ladder=None)
    assert out["structural_stop"] is None and out["udr_structural"] is None
    assert out["udr_structural_t2"] is None and out["why_null"] == "no structural stop"
    out, _ = _run(brief="no price here")
    assert out["udr_structural"] is None and out["why_null"] == "no price in the brief"
    out, _ = _run(udr=float("nan"), parsed={**_PARSED, "upside_downside_ratio": None})
    assert out["model_udr"] is None and out["model_udr_stated"] is False


def test_swallows_its_own_errors():
    with patch("db.get_trade_ladder", side_effect=RuntimeError("db down")):
        out = _udr_structural("MSFT", _PARSED, 2.4, _BRIEF)
    assert out["udr_structural"] is None and "db down" in out["error"]
    with patch("agents.plans.compile_plan", side_effect=ValueError("bad")), \
         patch("db.get_trade_ladder", return_value=_LADDER):
        assert "error" in _udr_structural("MSFT", _PARSED, 2.4, _BRIEF)
    assert "error" in _udr_structural("MSFT", object(), object(), _BRIEF)


def _analyze(shadow=None):
    parsed = {
        "action": "HOLD", "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {"bull_case_target": "$130", "downside_scenario": "$90"},
        "expected_return_pct": 8.0, "key_risks": [], "pass_to_risk": False,
        "upside_downside_ratio": 2.4, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}],
    }
    log_signal = MagicMock()
    ladder = MagicMock(return_value=_LADDER)
    patches = [
        patch("agents.analyst._call_ollama", return_value=json.dumps(parsed)),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.get_trade_ladder", ladder),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
    ]
    if shadow is not None:
        patches.append(patch.object(analyst, "_udr_structural", shadow))
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "MSFT", "role": "Compounder", "tier": "Core",
                                  "sector": "Software", "held": True}, situation_brief=_BRIEF,
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"], ladder.call_count


def _without_clock(d):
    """Drop *_at times (provenance.brief_exposure.read_at, process.code_read_at
    are whole-second reads, so two runs across a second boundary differ; the
    suite-order flake of 11-07/11-08)."""
    if isinstance(d, dict):
        return {k: _without_clock(v) for k, v in d.items() if not k.endswith("_at")}
    return d


def test_recorded_on_the_verdict_row_and_the_verdict_is_unchanged():
    result, decision, reads_on = _analyze()
    s = decision["udr_structural"]
    assert s["udr_structural"] == round(10 / 6, 2) and s["model_udr"] == 2.4
    off, decision_off, reads_off = _analyze(shadow=lambda *a: None)
    assert reads_on - reads_off == 1                   # one ladder read per verdict
    for r in (result, off):
        r.pop("timestamp")
    assert _without_clock(result) == _without_clock(off)
    assert "udr_structural" not in result
    # catalyst_check reads the clock (weeks to catalyst), so it is left out
    assert {k: v for k, v in decision.items() if k not in ("udr_structural", "catalyst_check")} == \
           {k: v for k, v in decision_off.items() if k not in ("udr_structural", "catalyst_check")}
    _, d, _ = _analyze(shadow=MagicMock(side_effect=None, return_value={"error": "x"}))
    assert d["final_action"] == decision["final_action"]
