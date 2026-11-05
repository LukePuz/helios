"""Scores the model left out and a default filled in (conviction 1, a
sub-score 3) are listed in imputed_fields, on the verdict's decision and on
the result (so pending_approvals.analyst_json carries it). Log-only."""
import json
from unittest.mock import patch

import pytest

from agents.analyst import analyze

_CANDIDATE = {"ticker": "NVDA", "role": "Compounder", "tier": "Watch", "sector": "Software"}
_FULL = {"action": "HOLD", "conviction": 6, "thesis_strength": 4, "catalyst_quality": 4,
         "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Compounder",
         "thesis_summary": "t", "expectation_gap": {}, "catalyst": {}, "valuation": {},
         "upside_downside_ratio": 2.0, "key_risks": [], "pass_to_risk": False,
         "reasoning": "r", "sources": []}


def _analyze(reply: dict):
    with patch("agents.analyst._call_ollama", return_value=json.dumps(reply)), \
         patch("knowledge_loader.load_thesis", return_value=""), \
         patch("knowledge_loader.load_philosophy", return_value=""), \
         patch("agents.market_regime.get_cached_regime",
               return_value={"regime": "Neutral", "score": 50, "reasoning": ""}), \
         patch("agents.chase_risk.compute_chase_risk", return_value=None), \
         patch("agents.price_action.compute_price_action", return_value=None), \
         patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}), \
         patch("db.log_signal") as log_signal, patch("feed.emit"):
        result = analyze(dict(_CANDIDATE), situation_brief="brief")
    verdict = next(c for c in log_signal.call_args_list
                   if (c.kwargs.get("decision") or {}).get("model_action") is not None)
    return result, verdict.kwargs["decision"]


@pytest.mark.parametrize("missing, expected", [
    ((), []),
    (("thesis_strength",), ["thesis_strength"]),
    (("conviction", "catalyst_quality", "data_completeness"),
     ["conviction", "catalyst_quality", "data_completeness"]),
])
def test_imputed_fields(missing, expected):
    reply = {k: v for k, v in _FULL.items() if k not in missing}
    result, decision = _analyze(reply)
    assert result["imputed_fields"] == expected and decision["imputed_fields"] == expected
    if "conviction" in missing:
        assert decision["model_conviction"] == 1
    for f in missing:
        if f != "conviction":
            assert result[f] == 3


def test_null_score_is_imputed_but_zero_is_not():
    result, _ = _analyze({**_FULL, "valuation_risk_reward": None, "thesis_strength": 0})
    assert result["imputed_fields"] == ["valuation_risk_reward"]
    assert result["thesis_strength"] == 1                    # floored, not filled
