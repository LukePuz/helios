"""A failed fundamentals read caps conviction exactly as missing data does,
but carries its own gate label and provenance value."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


def _analyze(fund_patch):
    response = json.dumps({
        "action": "OPEN", "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Compounder",
        "thesis_summary": "test", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {}, "expected_return_pct": 8.0, "upside_downside_ratio": 3.0,
        "key_risks": [], "pass_to_risk": True, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })
    log_signal = MagicMock()
    patches = [
        patch("agents.analyst._call_ollama", return_value=response),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("knowledge_loader.get_etf_tickers", return_value=set()),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        fund_patch,
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from agents.analyst import analyze
        result = analyze({"ticker": "MSFT", "role": "Compounder", "tier": "Core", "sector": "Software",
                          "held": False}, situation_brief="brief",
                         regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    kw = log_signal.call_args.kwargs
    return result, kw["decision"], kw["provenance"]


def test_read_error_caps_like_missing_but_is_labelled_apart(caplog):
    err = patch("db.get_fundamentals", side_effect=OSError("could not translate host name"))
    with caplog.at_level("WARNING"):
        result, d, prov = _analyze(err)
    assert d["final_conviction"] == 6 and result["conviction_raw"] == 8
    assert "fundamentals read error: OSError: conviction 8 -> 6" in d["gates"]
    assert not any(g.startswith("no fundamentals") for g in d["gates"])
    assert prov["fundamentals_read"] == "read error: OSError"
    assert any("Fundamentals read failed" in r.message and r.levelname == "WARNING"
               for r in caplog.records)


def test_missing_keeps_its_label():
    result, d, prov = _analyze(patch("db.get_fundamentals", return_value=None))
    assert d["final_conviction"] == 6
    assert "no fundamentals: conviction 8 -> 6" in d["gates"]
    assert prov["fundamentals_read"] == "missing"


def test_present_fundamentals_record_ok():
    result, d, prov = _analyze(patch("db.get_fundamentals", return_value={"market_cap": 1}))
    assert d["final_conviction"] == 8 and d["gates"] == []
    assert prov["fundamentals_read"] == "ok"
