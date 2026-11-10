"""decision.parse_mode records which _extract_json path read the reply
("json", "json_substring", "salvaged"). Log-only: the model output and the
gates are unchanged. _extract_json itself stays byte-identical, because the
prompt-order A/B tool_sha hashes its source."""
import hashlib
import inspect
import json
from unittest.mock import patch

import pytest

from agents import analyst

_CANDIDATE = {"ticker": "NVDA", "role": "Compounder", "tier": "Watch", "sector": "Software"}
_FULL = {"action": "WATCH", "conviction": 6, "thesis_strength": 4, "catalyst_quality": 4,
         "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Compounder",
         "thesis_summary": "t", "expectation_gap": {}, "catalyst": {}, "valuation": {},
         "upside_downside_ratio": 2.0, "key_risks": [], "pass_to_risk": False,
         "reasoning": "r", "sources": []}
# sha256 of the sources of the analyst functions in prompt_order_ab.TOOL_FUNCS at 9b913d7.
TOOL_FUNCS_SHA_AT_9B913D7 = "4c19cf87efa901044a9ef523e65522c9aabbd915f78d0221cc02d6e4c8f8fa5f"


def _decision(raw: str) -> dict:
    with patch("agents.analyst._call_ollama", return_value=raw), \
         patch("knowledge_loader.load_thesis", return_value=""), \
         patch("knowledge_loader.load_philosophy", return_value=""), \
         patch("agents.market_regime.get_cached_regime",
               return_value={"regime": "Neutral", "score": 50, "reasoning": ""}), \
         patch("agents.chase_risk.compute_chase_risk", return_value=None), \
         patch("agents.price_action.compute_price_action", return_value=None), \
         patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}), \
         patch("db.log_signal") as log_signal, patch("feed.emit"):
        analyst.analyze(dict(_CANDIDATE), situation_brief="brief")
    verdict = next(c for c in log_signal.call_args_list
                   if (c.kwargs.get("decision") or {}).get("model_action") is not None)
    return verdict.kwargs["decision"]


@pytest.mark.parametrize("raw, mode", [
    (json.dumps(_FULL), "json"),
    ("```json\n" + json.dumps(_FULL) + "\n```", "json"),
    ("Here is my answer: " + json.dumps(_FULL) + " done", "json_substring"),
    ('{"action": "WATCH", "conviction": 6, "reasoning": "r" oops', "salvaged"),
])
def test_verdict_records_parse_mode(raw, mode):
    dec = _decision(raw)
    assert dec["parse_mode"] == mode
    assert "_salvaged" not in dec["model_output"] and "parse_mode" not in dec["model_output"]


def test_tool_funcs_source_unchanged():
    h = hashlib.sha256()
    for name in ("_extract_json", "_conviction_distribution", "redact_account_figures"):
        h.update(inspect.getsource(getattr(analyst, name)).encode())
    assert h.hexdigest() == TOOL_FUNCS_SHA_AT_9B913D7
