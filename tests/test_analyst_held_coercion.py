"""ADD/HOLD on a ticker that isn't held (2026-10-27, limitations audit
finding 5): ADD becomes OPEN and must pass the OPEN gates; HOLD becomes WATCH."""
import json
from contextlib import ExitStack
from unittest.mock import patch


def _analyze(action, held, conviction=7):
    response = json.dumps({
        "action": action, "conviction": conviction, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Accelerator",
        "thesis_summary": "test", "expectation_gap": {},
        "catalyst": {"event": "earnings", "date": "2026-11-20", "is_priced_in": False},
        "valuation": {}, "expected_return_pct": 8.0,
        "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": True,
        "reasoning": "per Q3 fundamentals, revenue grew 30%",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })
    patches = [
        patch("agents.analyst._call_ollama", return_value=response),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("agents.market_regime.get_cached_regime",
              return_value={"regime": "Neutral", "score": 50, "reasoning": ""}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("agents.price_action.compute_trend_template", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.log_signal"),
        patch("feed.emit"),
    ]
    candidate = {"ticker": "MSFT", "role": "Accelerator", "tier": "Watch", "sector": "Software",
                 "held": held}
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from agents.analyst import analyze
        return analyze(candidate, situation_brief="brief")


def test_add_on_unheld_ticker_becomes_open():
    assert _analyze("ADD", held=False)["action"] == "OPEN"


def test_add_on_held_ticker_stays_add():
    assert _analyze("ADD", held=True)["action"] == "ADD"


def test_hold_on_unheld_ticker_becomes_watch():
    result = _analyze("HOLD", held=False)
    assert result["action"] == "WATCH"


def test_unknown_holding_is_left_alone():
    assert _analyze("ADD", held=None)["action"] == "ADD"


def test_result_carries_provenance():
    prov = _analyze("OPEN", held=False)["provenance"]
    assert set(prov) >= {"code_sha", "model", "model_digest", "prompt_template_sha", "prompt_sha"}
    assert len(prov["prompt_sha"]) == 12
