"""The plan shadow record carries the sizing bucket next to the model's
thesis_bucket, which sets the plan horizon (audit 11-06 07:38 #13)."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents import analyst


def _provenance(plan):
    body = json.dumps({"action": "WATCH", "conviction": 5, "reasoning": "r", "sources": [],
                       "thesis_bucket": "Moonshot"})
    log_signal = MagicMock()
    with ExitStack() as stack:
        for p in (patch("agents.analyst._call_ollama", return_value=body),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1}),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.log_signal", log_signal), patch("feed.emit"),
                  patch("agents.plans.observe", return_value=plan)):
            stack.enter_context(p)
        analyst.analyze({"ticker": "MSFT", "role": "Compounder"}, situation_brief="b",
                        regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})
    return log_signal.call_args.kwargs["provenance"]


def test_plan_record_gets_the_sizing_bucket():
    prov = _provenance({"relation": "NO_PLAN"})
    assert prov["plan"] == {"relation": "NO_PLAN", "sizing_bucket": "Compounder"}


def test_no_plan_record_is_left_alone():
    assert _provenance(None)["plan"] is None
