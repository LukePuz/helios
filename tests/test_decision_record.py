"""signals.decision keeps the model's raw verdict, each gate that changed it,
and the full model output (roadmap 0e #8, professor C11)."""
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


def _analyze(udr, regime="Risk-On", fund={"market_cap": 1}, extra=()):
    response = json.dumps({
        "action": "OPEN", "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Compounder",
        "thesis_summary": "test", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {}, "expected_return_pct": 8.0, "upside_downside_ratio": udr,
        "key_risks": [], "pass_to_risk": True, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })
    log_signal = MagicMock()
    patches = [
        patch("agents.analyst._call_ollama", return_value=response),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value=fund),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
        *extra,
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from agents.analyst import analyze
        analyze({"ticker": "MSFT", "role": "Compounder", "tier": "Core", "sector": "Software",
                 "held": False}, situation_brief="brief",
                regime_packet={"regime": regime, "score": 70, "reasoning": ""})
    assert "brief" in (log_signal.call_args.kwargs["prompt"] or "")
    return log_signal.call_args.kwargs["decision"]


def test_downgraded_open_records_model_verdict_and_gate():
    d = _analyze(udr=1.5)
    assert (d["model_action"], d["model_conviction"], d["model_pass_to_risk"]) == ("OPEN", 8, True)
    assert d["final_action"] == "WATCH" and d["final_pass_to_risk"] is False
    assert any(g.startswith("udr 1.5 < 2.0") for g in d["gates"])
    assert d["model_output"]["upside_downside_ratio"] == 1.5


def test_untouched_verdict_has_no_gates():
    d = _analyze(udr=3.0)
    assert d["final_action"] == "OPEN" and d["gates"] == []


def test_log_signal_writes_decision_column():
    import db
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_signal("NVDA", "WATCH", 0.8, "r", decision={"gates": ["x"]})
    sql, params = cur.execute.call_args[0]
    assert "decision" in sql and params[7].adapted == {"gates": ["x"]}


def test_etf_exemption_from_the_missing_fundamentals_cap_is_off_by_default():
    """Audit A0100-2: ETFs never have fundamentals; the exemption ships off."""
    etf = patch("knowledge_loader.get_etf_tickers", return_value={"MSFT"})
    d = _analyze(udr=3.0, fund=None, extra=(etf,))
    assert d["final_conviction"] == 6 and any("ETF; exemption off" in g for g in d["gates"])
    on = patch("config.ETF_FUNDAMENTALS_CAP_EXEMPT", True)
    d = _analyze(udr=3.0, fund=None, extra=(patch("knowledge_loader.get_etf_tickers", return_value={"MSFT"}), on))
    assert d["final_conviction"] == 8 and any("ETF, cap not applied" in g for g in d["gates"])
