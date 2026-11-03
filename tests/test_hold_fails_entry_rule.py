"""Log-only hold_fails_entry_rule shadow (audit 2026-11-02 19:27 #4,
roadmap 01(c)): a HOLD on a held name under the OPEN/ADD UDR floor is
recorded as a would-trim. Nothing routes on it."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents.analyst import _hold_fails_entry_rule


def _analyze(action, udr, held=True):
    parsed = {
        "action": action, "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {}, "expected_return_pct": 8.0, "key_risks": [], "pass_to_risk": False,
        "reasoning": "r", "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}],
    }
    if udr is not None:
        parsed["upside_downside_ratio"] = udr
    log_signal = MagicMock()
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
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from agents.analyst import analyze
        result = analyze({"ticker": "MSFT", "role": "Compounder", "tier": "Core",
                          "sector": "Software", "held": held}, situation_brief="brief",
                         regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"]


def test_held_hold_under_the_floor_is_a_would_trim_and_routes_nothing():
    result, d = _analyze("HOLD", 1.1)
    s = d["hold_fails_entry_rule"]
    assert s["fails"] is True and s["would_trim"] is True
    assert s["udr"] == 1.1 and s["udr_floor"] == 2.0 and s["held_at_gate"] is True
    assert s["checks"] == "udr_floor_only" and isinstance(s["conviction"], int)   # m270
    assert result["action"] == "HOLD" and result["pass_to_risk"] is False
    assert d["final_action"] == "HOLD" and d["gates"] == []


def test_hold_at_or_above_the_floor_is_recorded_as_passing():
    _, d = _analyze("HOLD", 2.0)
    assert d["hold_fails_entry_rule"]["fails"] is False


def test_other_actions_and_unheld_holds_record_none():
    _, d = _analyze("WATCH", 1.1)
    assert d["hold_fails_entry_rule"] is None
    _, d = _analyze("HOLD", 1.1, held=False)          # coerced to WATCH
    assert d["final_action"] == "WATCH" and d["hold_fails_entry_rule"] is None


def test_missing_udr_reads_as_the_gate_reads_it():
    _, d = _analyze("HOLD", None)
    s = d["hold_fails_entry_rule"]
    assert s["fails"] is True and s["udr"] == 0.0 and s["udr_stated"] is False


def test_swallows_its_own_errors():
    assert "error" in _hold_fails_entry_rule("HOLD", object(), {}, {})
    assert _hold_fails_entry_rule("HOLD", float("nan"), {}, {})["fails"] is None
