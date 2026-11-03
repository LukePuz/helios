"""A failed log-only provenance helper leaves {"error": <type>}, not a
missing key (m239)."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents import analyst


def _provenance_with(*extra):
    body = json.dumps({"action": "WATCH", "conviction": 5, "reasoning": "r", "sources": []})
    log_signal = MagicMock()
    with ExitStack() as stack:
        for p in (patch("agents.analyst._call_ollama", return_value=body),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1}),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.log_signal", log_signal), patch("feed.emit"), *extra):
            stack.enter_context(p)
        analyst.analyze({"ticker": "MSFT", "role": "Compounder"}, situation_brief="b",
                        regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})
    return log_signal.call_args.kwargs["provenance"]


def test_failed_helpers_store_the_error_type_and_warn():
    with patch.object(analyst.log, "warning") as warn:
        prov = _provenance_with(
            patch("agents.analyst._brief_exposure", side_effect=KeyError("x")),
            patch("agents.analyst._gate_flags", side_effect=ZeroDivisionError()))
    assert prov["brief_exposure"] == {"error": "KeyError"}
    assert prov["gate_flags"] == {"error": "ZeroDivisionError"}
    msgs = " ".join(str(c.args[0]) for c in warn.call_args_list)
    assert "brief exposure record failed" in msgs and "gate flags record failed" in msgs


def test_held_fields_failing_keeps_the_exposure_record():
    prov = _provenance_with(
        patch("agents.analyst._brief_exposure", return_value={"position_context_shown": False}),
        patch("agents.analyst._held_at_gate", side_effect=TypeError()))
    assert prov["brief_exposure"] == {"position_context_shown": False, "error": "TypeError"}
