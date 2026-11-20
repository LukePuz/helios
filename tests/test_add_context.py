"""Log-only add_context on ADD routes (audit 2026-11-10 16:53 #2)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents import crew


def _client(avg=None, err=None):
    c = MagicMock()
    if err:
        c.get_open_position.side_effect = err
    else:
        c.get_open_position.return_value = SimpleNamespace(avg_entry_price=avg)
    return c


def test_nvda_11_10_add():
    sizing = {"stop_loss_price": 146.24, "atr_stop": 8.44}
    with patch("db.get_position_stop", return_value={"stop_price": 163.68}):
        rec = crew._add_context("NVDA", 166.56, sizing, _client(avg=183.39))
    assert rec["kept_stop"] == 163.68 and rec["stop_distance_atr"] == 0.34
    assert rec["below_avg_cost"] is True and rec["pct_vs_avg_cost"] == -9.18
    assert rec["error"] is None


def test_new_stop_above_the_record_is_kept():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}):
        rec = crew._add_context("X", 110.0, {"stop_loss_price": 100.0, "atr_stop": 5.0}, _client(avg=100.0))
    assert rec["kept_stop"] == 100.0 and rec["stop_distance_atr"] == 2.0 and rec["below_avg_cost"] is False


def test_unreadable_record_and_position_are_recorded_not_raised():
    with patch("db.get_position_stop", side_effect=RuntimeError("db")):
        rec = crew._add_context("X", 110.0, {"stop_loss_price": 100.0, "atr_stop": 5.0},
                                _client(err=RuntimeError("ib")))
    assert rec["kept_stop"] == 100.0 and rec["avg_cost"] is None
    assert "stop record: RuntimeError" in rec["error"] and "position: RuntimeError" in rec["error"]


def test_missing_atr_leaves_distance_null():
    with patch("db.get_position_stop", return_value=None):
        rec = crew._add_context("X", 110.0, {"stop_loss_price": 100.0}, _client(avg=0))
    assert rec["stop_distance_atr"] is None and rec["avg_cost"] is None and rec["error"] is None


def test_never_raises_on_garbage():
    rec = crew._add_context("X", "not-a-price", {}, None)
    assert rec["error"] == "ValueError"


# ── exit state an ADD resets (record only) ───────────────────────────────────

def _pos(qty, avg):
    c = MagicMock()
    c.get_open_position.return_value = SimpleNamespace(qty=qty, avg_entry_price=avg)
    return c


def _state(prior, ctx, client, sizing=None, price=130.0, ctx_exc=None):
    sizing = sizing or {"stop_loss_price": 110.0, "atr_stop": 5.0, "shares_initial": 2,
                        "bucket": "Accelerator"}
    with patch("db.get_position_stop", return_value=prior), \
         patch("db.get_position_context",
               **({"side_effect": ctx_exc} if ctx_exc else {"return_value": ctx})):
        return crew._add_context("NVDA", price, sizing, client)


def test_add_to_a_trailing_position_records_the_state_it_resets():
    """10 held at $100, 2 more at $130: post-ADD average $105, gain 23.81%,
    still over the Accelerator's 20% activation gain."""
    rec = _state({"stop_price": 120.0, "trailing_active": True, "conviction": 7},
                 {"original_thesis": {"conviction": 6}, "profit_target_taken": True},
                 _pos(10, 100.0))
    assert rec["trailing_active"] is True and rec["record_conviction"] == 7
    assert rec["entry_conviction"] == 6 and rec["profit_target_taken"] is True
    assert rec["held_qty"] == 10 and rec["post_add_avg_est"] == 105.0
    assert rec["gain_vs_post_add_avg_pct"] == 23.81 and rec["trail_activate_pct"] == 20.0
    assert rec["trail_gain_qualifies_post_add"] is True
    assert rec["error"] is None and rec["state_error"] is None


def test_gain_below_the_bucket_threshold_does_not_qualify():
    sizing = {"stop_loss_price": 110.0, "atr_stop": 5.0, "shares_initial": 2, "bucket": "Compounder"}
    rec = _state({"stop_price": 120.0, "trailing_active": False, "conviction": 8},
                 {"original_thesis": {"conviction": 8}, "profit_target_taken": False},
                 _pos(10, 100.0), sizing=sizing)
    assert rec["trail_activate_pct"] == 30.0 and rec["trail_gain_qualifies_post_add"] is False
    assert rec["trailing_active"] is False and rec["profit_target_taken"] is False


def test_no_gain_rule_for_the_bucket_or_on_the_ladder_path():
    sizing = {"stop_loss_price": 110.0, "shares_initial": 2, "bucket": "Moonshot"}
    rec = _state({"stop_price": 120.0}, {}, _pos(10, 100.0), sizing=sizing)
    assert rec["trail_activate_pct"] is None and rec["trail_gain_qualifies_post_add"] is None
    assert rec["gain_vs_post_add_avg_pct"] == 23.81
    with patch("risk_manager.SETUP_GATES_ENABLED", True), \
         patch("risk_manager.CATEGORY_RISK_PCT_PER_TRADE", {"Accelerator": 0.01}):
        rec = _state({"stop_price": 120.0}, {}, _pos(10, 100.0))
    assert rec["trail_activate_pct"] is None and rec["trail_gain_qualifies_post_add"] is None


def test_failed_state_reads_are_recorded_not_raised():
    rec = _state(None, None, _client(err=RuntimeError("ib")), ctx_exc=RuntimeError("db"))
    assert rec["state_error"] == "position context: RuntimeError"
    assert rec["trailing_active"] is None and rec["entry_conviction"] is None
    assert rec["post_add_avg_est"] is None and rec["trail_gain_qualifies_post_add"] is None
    assert "position: RuntimeError" in rec["error"]


def test_state_never_raises_on_garbage():
    out = crew._add_exit_state("X", "nan?", "not-a-dict", {"trailing_active": 1}, "q", "a")
    assert out["trailing_active"] is True and out["state_error"]


# ── whole-position risk at the kept stop (record only) ───────────────────────

def _rm(equity=None, exc=None):
    rm = MagicMock()
    if exc:
        rm.portfolio_value.side_effect = exc
    else:
        rm.portfolio_value.return_value = equity
    return rm


def _risk(rm, held=10, sizing=None, record_stop=163.68, price=166.56):
    sizing = sizing or {"stop_loss_price": 146.24, "atr_stop": 8.44, "shares_initial": 1,
                        "bucket": "Accelerator"}
    with patch("db.get_position_stop", return_value={"stop_price": record_stop}), \
         patch("db.get_position_context", return_value={}):
        return crew._add_context("NVDA", price, sizing, _pos(held, 183.39), rm=rm)


def test_position_risk_counts_held_and_new_shares_at_the_kept_stop():
    """11-10 shape: 1 new share sized against its own $146.24 stop, the whole
    11-share position kept at $163.68: 11 x $2.88 = $31.68 of a $1,000 account."""
    rec = _risk(_rm(1000.0))
    assert rec["kept_stop"] == 163.68
    assert rec["position_risk_pct_at_kept_stop"] == 3.168
    assert rec["per_trade_cap_pct"] == 2.0 and rec["over_per_trade_cap"] is True
    assert rec["position_risk_status"] == "ok"
    assert "equity" not in rec and 1000.0 not in rec.values()        # equity never stored


def test_position_risk_uses_the_gap_shadow_cap_when_present():
    sizing = {"stop_loss_price": 146.24, "shares_initial": 1, "bucket": "Accelerator",
              "gap_risk_shadow": {"cap_pct": 5.0}}
    rec = _risk(_rm(1000.0), sizing=sizing)
    assert rec["per_trade_cap_pct"] == 5.0 and rec["over_per_trade_cap"] is False


def test_position_risk_missing_inputs_are_named_not_raised():
    assert _risk(None)["position_risk_status"] == "no_equity"
    assert _risk(_rm(exc=RuntimeError("ib")))["position_risk_status"] == "equity: RuntimeError"
    assert _risk(_rm(0.0))["position_risk_status"] == "no_equity"
    assert _risk(_rm(1000.0), held=0)["position_risk_status"] == "no_shares"
    rec = _risk(_rm(1000.0), sizing={"shares_initial": 1}, record_stop=None)
    assert rec["position_risk_status"] == "no_stop" and rec["position_risk_pct_at_kept_stop"] is None
    rec = _risk(_rm("garbage"))
    assert rec["position_risk_status"] == "equity: ValueError"
    assert rec["over_per_trade_cap"] is None


def test_add_route_records_position_risk_in_risk_json():
    from contextlib import ExitStack
    sizing = {"shares_initial": 1, "initial_dollar": 166.56, "position_size_pct": 0.2,
              "order_pct": 0.05, "initial_entry_pct": 0.55, "flags": [],
              "stop_loss_price": 146.24, "atr_stop": 8.44, "bucket": "Accelerator"}
    result = {"ticker": "NVDA", "action": "ADD", "conviction": 7, "thesis_bucket": "Accelerator",
              "reasoning": "r"}
    client = _pos(10, 183.39)
    client.get_all_positions.return_value = []
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=166.56))
        st.enter_context(patch("agents.crew._compiled_plan", return_value=None))
        st.enter_context(patch("agents.crew._rule_shadow_average_down"))
        st.enter_context(patch("agents.crew._attach_gap_risk_shadow"))
        st.enter_context(patch("db.get_position_stop", return_value={"stop_price": 163.68}))
        st.enter_context(patch("db.get_position_context", return_value={}))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        st.enter_context(patch("db.log_signal"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        rm.portfolio_value.return_value = 1000.0
        crew._route_to_approval(result, {"regime": "Neutral"}, client)
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 1                                  # sizing unchanged
    ctx = risk["add_context"]
    assert ctx["position_risk_status"] == "ok" and ctx["position_risk_pct_at_kept_stop"] == 3.168
    assert ctx["over_per_trade_cap"] is True


def test_held_qty_is_recorded_without_a_broker_average():
    """With no broker average cost the post-ADD average can't be estimated,
    but the held share count behind the position risk is still recorded."""
    rec = _state({"stop_price": 120.0}, {}, _pos(10, 0.0))
    assert rec["held_qty"] == 10 and rec["post_add_avg_est"] is None
