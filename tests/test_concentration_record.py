"""The concentration block records which weight it checked and what the cap
left, and rescales order_pct with the shares (the 11-04 ANET row read
"position 16.2% exceeds cap" while the one share bought was 9.5%)."""
from contextlib import ExitStack
from unittest.mock import patch

from agents import crew

PV = 2_000.0


def _route(conc, shares=4, price=100.0):
    sizing = {"shares_initial": shares, "initial_dollar": shares * price, "position_size_pct": 0.25,
              "order_pct": round(shares * price / PV, 4), "initial_entry_pct": 0.55,
              "flags": [], "stop_loss_price": price * 0.9}
    result = {"ticker": "ANET", "action": "OPEN", "conviction": 7,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=price))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = conc
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        client = rm_cls.return_value
        crew._route_to_approval(result, {"regime": "Neutral"}, client)
    return lpa


def test_capped_order_records_weights_and_rescales_order_pct():
    lpa = _route({"passed": False, "flags": ["ANET position 25.0% exceeds Compounder cap 12%"],
                  "ai_infra_size_cap_pct": 0.10})
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 2
    assert risk["order_pct"] == 0.1                      # 2 x $100 / $2,000, not the pre-cap 0.2
    rec = risk["concentration_check"]
    assert rec["checked_pct"] == 0.25 and rec["target_pct"] == 0.25 and rec["order_pct"] == 0.2
    assert rec["cap_pct"] == 0.1 and rec["shares_before"] == 4 and rec["shares_after"] == 2
    assert rec["order_pct_after"] == 0.1


def test_failed_check_without_a_cap_records_no_cap():
    lpa = _route({"passed": False, "flags": ["ANET position 25.0% exceeds Compounder cap 12%"]})
    risk = lpa.call_args.kwargs["risk_json"]
    rec = risk["concentration_check"]
    assert rec["cap_pct"] is None and rec["shares_after"] == 4 and risk["order_pct"] == 0.2


def test_passed_check_records_nothing():
    lpa = _route({"passed": True, "flags": []})
    assert "concentration_check" not in lpa.call_args.kwargs["risk_json"]


def test_route_records_udr_at_sizing():
    lpa = _route({"passed": True, "flags": []})
    rec = lpa.call_args.kwargs["risk_json"]["udr_at_sizing"]
    assert rec["price"] == 100.0 and rec["udr"] is None and rec["reason"] == "no_target_or_floor"


def test_failed_rules_errored_and_cap_binding():
    lpa = _route({"passed": False, "flags": ["AI infrastructure exposure 65.4% exceeds 60% limit",
                                             "ANET position 25.0% exceeds Compounder cap 12%"],
                  "ai_infra_size_cap_pct": 0.10})
    rec = lpa.call_args.kwargs["risk_json"]["concentration_check"]
    assert rec["failed_rules"] == ["ai_infra", "single_name"]
    assert rec["cap_binding"] is True and rec["errored"] is False
    lpa = _route({"passed": False, "flags": ["ANET position 25.0% exceeds Compounder cap 12%"]})
    rec = lpa.call_args.kwargs["risk_json"]["concentration_check"]
    assert rec["failed_rules"] == ["single_name"] and rec["cap_binding"] is False


def test_zero_room_route_row_keeps_the_record():
    from contextlib import ExitStack
    from unittest.mock import patch
    sizing = {"shares_initial": 1, "initial_dollar": 100.0, "position_size_pct": 0.25,
              "order_pct": 0.05, "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0}
    result = {"ticker": "ANET", "action": "OPEN", "conviction": 7, "thesis_bucket": "Compounder", "reasoning": "r"}
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=100.0))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = {"passed": False, "errored": True, "ai_infra_size_cap_pct": 0.0,
                                               "flags": ["RISK CHECK FAILED — could not verify concentration: x. Buy blocked."]}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, rm_cls.return_value)
    assert ls.call_args.args[1] == "ROUTE_CONCENTRATION_ZERO"
    rec = ls.call_args.kwargs["decision"]["concentration_check"]
    assert rec["errored"] is True and rec["failed_rules"] == ["errored"] and rec["shares_after"] == 0
    # an errored check's 0.0 caps cut the order to 0, so the cap binds
    assert rec["cap_binding"] is True and rec["binding_cap"] == "errored"


def test_every_concentration_flag_maps_to_a_named_rule():
    """failed_rules reads the flag text, so pin it: each flag check_concentration
    can emit (its source f-strings, placeholders filled) maps to a named rule."""
    import inspect
    import re
    import risk_manager
    src = inspect.getsource(risk_manager.RiskManager.check_concentration)
    templates = re.findall(r'flags\.append\(\s*f(["\'])(.+?)\1', src)
    templates = [t for _, t in templates]
    templates += re.findall(r'"flags": \[f"([^"]+)"', src)
    assert len(templates) >= 5
    for t in templates:
        text = re.sub(r"\{[^}]*\}", "X", t)
        rules = crew._conc_failed_rules([text])
        assert rules and rules != ["other"], t



def test_cap_between_order_and_target_does_not_bind():
    """A cap above the order's weight but under the target leaves the shares
    as they were: recorded, not binding (professor M295)."""
    lpa = _route({"passed": False, "flags": ["AI infrastructure exposure 61.0% exceeds 60% limit"],
                  "ai_infra_size_cap_pct": 0.22})          # order 0.20 < cap 0.22 < target 0.25
    rec = lpa.call_args.kwargs["risk_json"]["concentration_check"]
    assert rec["cap_pct"] == 0.22 and rec["shares_after"] == 4
    assert rec["cap_binding"] is False and rec["binding_cap"] == "ai_infra"


def test_unknown_flag_text_is_other():
    assert crew._conc_failed_rules(["something new"]) == ["other"]
    assert crew._conc_failed_rules([]) == []
