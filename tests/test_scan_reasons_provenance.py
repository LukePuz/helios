"""Audit 11-07 01:08 #3: scan verdicts record why the scanner picked the
name (the prompt shows only the composite). Record-only."""
import json
from contextlib import ExitStack
from unittest.mock import patch

from agents import analyst

_CAND = {"ticker": "MSFT", "role": "Compounder", "tier": "Core", "sector": "Software",
         "scan_type": "daily", "priority": "MEDIUM", "composite_score": 66,
         "technical_score": 20, "momentum_score": 11, "catalyst_score": 10,
         "thesis_alignment_score": 25, "catalyst_detail": "Earnings in 9 days",
         "technical_notes": "t", "momentum_notes": "m", "thesis_notes": "Core conviction tier"}


def _run(trigger):
    response = json.dumps({
        "action": "WATCH", "conviction": 5, "thesis_strength": 2, "catalyst_quality": 2,
        "valuation_risk_reward": 2, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "x", "expectation_gap": {}, "catalyst": {}, "valuation": {},
        "upside_downside_ratio": 1.0, "key_risks": [], "pass_to_risk": False, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}]})
    with ExitStack() as st:
        for p in (patch("agents.analyst._call_ollama", return_value=response),
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
                  patch("feed.emit")):
            st.enter_context(p)
        log_signal = st.enter_context(patch("db.log_signal"))
        analyst.analyze(_CAND, situation_brief="brief", trigger=trigger)
    return log_signal.call_args.kwargs["provenance"]


def test_scan_verdict_records_the_scanner_reasons():
    prov = _run(analyst.trigger_record("daily_scan", {"source": "scanner"}))
    assert prov["scan_reasons"]["catalyst_detail"] == "Earnings in 9 days"
    assert prov["scan_reasons"]["thesis_alignment_score"] == 25
    assert set(prov["scan_reasons"]) == set(analyst._SCAN_REASON_KEYS)


def test_event_verdict_has_no_scan_reasons():
    prov = _run(analyst.trigger_record("event", {"source": "news", "title": "t"}))
    assert "scan_reasons" not in prov
