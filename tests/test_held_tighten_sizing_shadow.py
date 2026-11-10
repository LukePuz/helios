"""Held reviews record the calendar's tighten_sizing events too; the buy-side
shadow only sees sized OPEN/ADD verdicts (audit 11-06 13:30 #6). Log-only."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents import analyst


def _provenance(candidate, stop_row=None):
    body = json.dumps({"action": "HOLD", "conviction": 6, "reasoning": "r", "sources": [],
                       "thesis_bucket": "Compounder"})
    log_signal = MagicMock()
    shadow = {"lookahead_days": 3, "n": 1, "events": [{"event_type": "jobs_report"}]}
    with ExitStack() as stack:
        for p in (patch("agents.analyst._call_ollama", return_value=body),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1}),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.log_signal", log_signal), patch("feed.emit"),
                  patch("agents.plans.observe", return_value=None),
                  patch("agents.crew._tighten_sizing_shadow", return_value=shadow),
                  patch("db.get_position_stop", return_value=stop_row)):
            stack.enter_context(p)
        analyst.analyze(candidate, situation_brief="b",
                        regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})
    return log_signal.call_args.kwargs["provenance"]


def test_held_review_records_the_shadow():
    prov = _provenance({"ticker": "MSFT", "role": "Compounder", "held": True})
    assert prov["tighten_sizing_shadow"]["n"] == 1
    assert prov["tighten_sizing_shadow"]["basis"] == ["held"]


def test_unheld_verdict_has_no_shadow_key():
    assert "tighten_sizing_shadow" not in _provenance({"ticker": "MSFT", "role": "Compounder"})


def test_stop_row_without_held_flag_still_records_it():
    prov = _provenance({"ticker": "MSFT", "role": "Compounder"}, stop_row={"stop_price": 1.0})
    assert prov["tighten_sizing_shadow"]["n"] == 1
    assert prov["tighten_sizing_shadow"]["basis"] == ["stop_row"]


def test_route_rows_do_not_copy_it():
    from agents import crew
    assert "tighten_sizing_shadow" in crew._VERDICT_ONLY_PROVENANCE
