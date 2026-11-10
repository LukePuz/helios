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


def test_text_sub_score_is_imputed_not_raised(monkeypatch):
    """Flag on: a sub-score written as text ("4/5") is imputed with its raw
    text kept, instead of failing the verdict (audit 2026-11-07 09:07 #9)."""
    import config
    monkeypatch.setattr(config, "SUBSCORE_TEXT_IMPUTE_ENABLED", True)
    result, decision = _analyze({**_FULL, "catalyst_quality": "4/5", "thesis_strength": "4.0"})
    assert result["catalyst_quality"] == 3 and result["thesis_strength"] == 4
    assert result["imputed_fields"] == ["catalyst_quality"]
    assert result["unparsed_scores"] == decision["unparsed_scores"] == {"catalyst_quality": "4/5"}
    assert result["action"] == "HOLD"


def test_no_unparsed_scores_on_clean_reply():
    result, decision = _analyze(dict(_FULL))
    assert result["unparsed_scores"] == {} and decision["unparsed_scores"] == {}


@pytest.mark.parametrize("raw", ["4/5", "4.0"])
def test_flag_off_text_sub_score_still_fails(monkeypatch, raw):
    """Flag off (the default): a text sub-score raises as before, so the
    verdict ends ANALYST_FAILED and never routes."""
    import config
    monkeypatch.setattr(config, "SUBSCORE_TEXT_IMPUTE_ENABLED", False)
    with pytest.raises(ValueError):
        _analyze({**_FULL, "catalyst_quality": raw})


def test_flag_defaults_off():
    import config
    src = open(config.__file__).read()
    assert 'os.getenv("SUBSCORE_TEXT_IMPUTE_ENABLED", "false")' in src


def test_flag_is_in_the_treatment_config(monkeypatch):
    import config
    from agents.analyst import _treatment_config
    monkeypatch.setattr(config, "SUBSCORE_TEXT_IMPUTE_ENABLED", False)
    assert _treatment_config()["SUBSCORE_TEXT_IMPUTE_ENABLED"] is False
    monkeypatch.setattr(config, "SUBSCORE_TEXT_IMPUTE_ENABLED", True)
    assert _treatment_config()["SUBSCORE_TEXT_IMPUTE_ENABLED"] is True


@pytest.mark.parametrize("given, kept, coerced", [
    (7, 7, None), (7.5, 7, "7.5"), (0, 1, "0"), (11, 10, "11"), ("7", 7, "7"), (True, 1, "True")])
def test_conviction_coerced_records_what_the_model_wrote(given, kept, coerced):
    """professor m618: the coerced conviction is recorded as written. Log-only."""
    result, decision = _analyze({**_FULL, "conviction": given})
    assert decision["model_conviction"] == kept
    assert result["conviction_coerced"] == decision["conviction_coerced"] == coerced


def test_missing_conviction_is_imputed_not_coerced():
    result, _ = _analyze({k: v for k, v in _FULL.items() if k != "conviction"})
    assert result["conviction_coerced"] is None and "conviction" in result["imputed_fields"]
