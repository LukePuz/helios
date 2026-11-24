"""Record-only reward-to-risk against the placed stop (risk_json["rr_vs_stop"],
agents/crew.py:_rr_vs_stop) and the read-only readout scripts/rr_vs_stop.py
(at-sizing and at-fill replay, realised R per closed lot)."""
import importlib.util
import os
from contextlib import ExitStack
from unittest.mock import patch

from agents import crew

RESULT = {"valuation": {"bull_case_target": "$130", "downside_scenario": "$85"},
          "upside_downside_ratio": 2.5}
SIZING = {"stop_loss_price": 90.0, "atr_stop": 4.0, "stop_method": "2.5x ATR",
          "plan_compiled": {"t1": 105.0, "t2": 130.0}}


def test_ratio_distances_and_labels():
    r = crew._rr_vs_stop(RESULT, SIZING, 100.0, (1.5, 2.0))
    assert r["r_to_target"] == 3.0 and r["target_source"] == "bull_case_target"
    assert r["r_to_plan_t1"] == 0.5 and r["r_to_plan_t2"] == 3.0
    assert r["stop_pct"] == 10.0 and r["stop_atr"] == 2.5 and r["atr_source"] == "atr_stop"
    assert r["target_pct"] == 30.0 and r["target_atr"] == 7.5
    assert r["model_udr"] == 2.5 and r["model_floor"] == 85.0
    assert r["floor_vs_stop_pct"] == -5.0 and r["stop_above_model_floor"] is True
    assert r["would_block"] == {"r_lt_1.5": False, "r_lt_2": False} and r["reason"] is None


def test_labels_compare_unrounded_r():
    # (119.96 - 100) / 10 = 1.996: rounds to 2.0 but is under 2
    r = crew._rr_vs_stop({"valuation": {"bull_case_target": 119.96}}, SIZING, 100.0, (1.5, 2.0))
    assert r["r_to_target"] == 2.0 and r["would_block"] == {"r_lt_1.5": False, "r_lt_2": True}


def test_fallbacks_expected_return_and_brief_atr():
    res = {"valuation": {"bull_case_target": "n/a"}, "expected_return_pct": 15.0}
    sizing = {"stop_loss_price": 94.0, "atr_brief": {"atr": 3.0}}
    r = crew._rr_vs_stop(res, sizing, 100.0, (1.5, 2.0))
    assert r["target_source"] == "expected_return_pct" and r["r_to_target"] == 2.5
    assert r["atr_source"] == "atr_brief" and r["stop_atr"] == 2.0
    assert r["r_to_plan_t1"] is None and r["stop_above_model_floor"] is None


def test_missing_levels_give_reasons_and_null_labels():
    r = crew._rr_vs_stop(RESULT, {}, 100.0, (1.5, 2.0))
    assert r["r_to_target"] is None and r["reason"] == "no_stop"
    assert r["would_block"] == {"r_lt_1.5": None, "r_lt_2": None}
    assert crew._rr_vs_stop(RESULT, SIZING, None)["reason"] == "no_price"
    assert crew._rr_vs_stop({}, SIZING, 100.0)["reason"] == "no_target"
    r = crew._rr_vs_stop(RESULT, SIZING, 135.0)            # price above the target
    assert r["r_to_target"] is None and r["reason"] == "levels_not_stop_lt_price_lt_target"


def test_default_thresholds_from_config_and_never_raises():
    import config
    with patch.object(config, "RR_VS_STOP_LABEL_THRESHOLDS", (1.0,)):
        assert set(crew._rr_vs_stop(RESULT, SIZING, 100.0)["would_block"]) == {"r_lt_1"}
    r = crew._rr_vs_stop(None, None, "x")
    assert r["r_to_target"] is None and "error" not in r
    r = crew._rr_vs_stop(RESULT, SIZING, 100.0, thresholds="bad")
    assert "error" in r and r["r_to_target"] is None


def _route(enabled: bool, price=100.0):
    sizing = {"shares_initial": 2, "initial_dollar": 2 * price, "position_size_pct": 0.1,
              "order_pct": 0.1, "initial_entry_pct": 0.55, "flags": [],
              "stop_loss_price": price * 0.9, "atr_stop": 4.0}
    result = {"ticker": "ANET", "action": "OPEN", "conviction": 7, "thesis_bucket": "Compounder",
              "reasoning": "r", "valuation": {"bull_case_target": "$130"}}
    with ExitStack() as st:
        st.enter_context(patch("config.RR_VS_STOP_SHADOW_ENABLED", enabled))
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=price))
        st.enter_context(patch("agents.crew._entry_gate", return_value=False))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, rm_cls.return_value)
    return lpa.call_args.kwargs


def test_route_records_rr_vs_stop_and_changes_nothing_else():
    on, off = _route(True), _route(False)
    rec = on["risk_json"]["rr_vs_stop"]
    assert rec["r_to_target"] == 3.0 and rec["stop_atr"] == 2.5
    assert "rr_vs_stop" not in off["risk_json"]
    for k in ("shares_initial", "position_size_pct", "stop_loss_price", "order_pct", "flags"):
        assert on["risk_json"][k] == off["risk_json"][k]
    for k in ("action", "conviction", "size_pct", "stop_price", "entry_strategy"):
        assert on[k] == off[k]


def _script():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "rr_vs_stop.py")
    spec = importlib.util.spec_from_file_location("rr_vs_stop_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readout_entry_rows_at_sizing_and_fill():
    s = _script()
    approvals = [{"id": 1, "trading_env": "live", "ticker": "ANET", "action": "OPEN",
                  "status": "APPROVED", "timestamp": "2026-11-04 10:00", "stop_loss_price": 90.0,
                  "analyst_json": RESULT, "risk_json": {**SIZING, "price_at_sizing": 100.0}},
                 {"id": 2, "trading_env": "live", "ticker": "X", "action": "EXIT", "risk_json": {}}]
    rows = s.entry_rows(approvals, {1: 100.0})
    assert len(rows) == 1 and rows[0]["filled"] and rows[0]["at_sizing"]["r_to_target"] == 3.0
    assert rows[0]["at_fill"]["r_to_target"] == 3.0
    summ = s.summarize(rows, (1.5, 2.0))
    assert summ[0]["n"] == 1 and summ[0]["would_block"]["r_lt_2"] == (0, 1)
    assert "| live | yes | 1 | 1 |" in s.render(summ, rows, [], (1.5, 2.0), "abc")


def test_readout_realised_r_stored_and_derived():
    s = _script()
    outcomes = [{"approval_id": 1, "trading_env": "live", "ticker": "A", "action": "OPEN",
                 "entry_price": 100.0, "exit_price": 90.0, "outcome_pct": -0.1,
                 "planned_risk_per_share": 8.0, "fill_risk_per_share": 10.0},
                {"approval_id": 2, "trading_env": "live", "ticker": "B", "action": "ADD",
                 "entry_price": 100.0, "exit_price": 95.0, "outcome_pct": -0.05},
                {"approval_id": 3, "trading_env": "live", "ticker": "C", "entry_price": 100.0,
                 "exit_price": None}]
    approvals = {2: {"stop_loss_price": 90.0, "risk_json": {"price_at_sizing": 100.0}}}
    lots = s.closed_lots(outcomes, approvals)
    assert len(lots) == 2
    assert lots[0]["r_fill"] == -1.0 and lots[0]["r_planned"] == -1.25 and lots[0]["risk_source"] == "stored"
    assert lots[1]["r_fill"] == -0.5 and lots[1]["risk_source"] == "derived"
    assert lots[0]["outcome_pct"] == -10.0
    text = s.render([], [], lots, (1.5, 2.0), "abc")
    assert "live: 2 closed lots, R (fill risk) on 2, mean -0.75" in text


def test_infinite_target_records_no_ratio():
    from agents import crew
    out = crew._rr_vs_stop({"valuation": {"bull_case_target": float("inf"),
                                          "downside_scenario": float("inf")}},
                           {"stop_loss_price": 90.0}, 100.0)
    assert out["r_to_target"] is None and out["reason"] is not None
    assert out["would_block"] is None or all(v is None for v in out["would_block"].values())
