"""An ADD that dies in routing after the price read (zero shares,
concentration zero, micro name cap) carries the add-to-loser record and the
raw days to the next earnings date on its route row. An OPEN carries
neither. A failed read records an error and never changes routing."""
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents import crew

KEYS = ("add_to_loser", "next_earnings_days", "next_earnings_read")
from zoneinfo import ZoneInfo

TODAY = datetime.now(ZoneInfo("America/New_York")).date()


def _in(days):
    return TODAY + timedelta(days=days)


def _result(action="ADD"):
    return {"ticker": "NVDA", "action": action, "conviction": 7, "thesis_bucket": "Compounder",
            "reasoning": "r", "provenance": {"trigger": {"path": "event", "new_fact": False}}}


def _client(avg=200.0):
    c = MagicMock()
    c.get_all_positions.return_value = [SimpleNamespace(symbol="NVDA", avg_entry_price=avg)]
    return c


def _route(action="ADD", sizing=None, conc=None, micro=None, earnings=None, earnings_exc=None,
           stop_exc=None):
    sizing = sizing or {"shares_initial": 0, "flags": ["Position already at/near target (18.0% held)"],
                        "position_size_pct": 0, "stop_loss_price": 90.0}
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=100.0))
        st.enter_context(patch("agents.crew._compiled_plan", return_value=None))
        st.enter_context(patch("agents.crew._rule_shadow_average_down"))
        st.enter_context(patch("agents.crew._shadow_swap"))
        st.enter_context(patch("agents.crew._c6_bar_tier_floor"))
        st.enter_context(patch("agents.crew._add_context", return_value={}))
        st.enter_context(patch("db.get_position_stop",
                               **({"side_effect": stop_exc} if stop_exc
                                  else {"return_value": {"entry_price": 150.0}})))
        ge = st.enter_context(patch("db.get_next_earnings",
                                    **({"side_effect": earnings_exc} if earnings_exc
                                       else {"return_value": earnings})))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = conc or {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = micro
        crew._route_to_approval(_result(action), {"regime": "Neutral"}, _client())
    return ls, lpa, ge


def _row(ls, kind):
    rows = [c for c in ls.call_args_list if c.args[1] == kind]
    assert len(rows) == 1
    return rows[0].kwargs["decision"]


def test_zero_share_add_row_carries_both_keys():
    ls, lpa, ge = _route(earnings=_in(1))
    lpa.assert_not_called()
    d = _row(ls, "ROUTE_ZERO_SHARES")
    assert d["next_earnings_days"] == 1 and d["next_earnings_read"] == "ok"
    assert d["add_to_loser"]["basis"] == "avg_cost" and d["add_to_loser"]["below_entry"] is True
    assert d["add_to_loser"]["breaks_rule"] is True
    assert ge.call_args.kwargs == {"strict": True}


def test_zero_share_open_row_carries_neither():
    ls, _, ge = _route(action="OPEN", earnings=_in(1))
    d = _row(ls, "ROUTE_ZERO_SHARES")
    assert not any(k in d for k in KEYS)
    ge.assert_not_called()


def test_no_calendar_row_is_told_from_a_failed_read():
    d = _row(_route(earnings=None)[0], "ROUTE_ZERO_SHARES")
    assert d["next_earnings_days"] is None and d["next_earnings_read"] == "no_row"
    d = _row(_route(earnings_exc=RuntimeError("db down"))[0], "ROUTE_ZERO_SHARES")
    assert d["next_earnings_days"] is None and d["next_earnings_read"] == "error: RuntimeError"


def test_raising_reads_never_change_routing():
    """Both reads raise: the same exit, still no approval, an error in each record."""
    ls, lpa, _ = _route(earnings_exc=RuntimeError("db down"), stop_exc=RuntimeError("db down"))
    lpa.assert_not_called()
    d = _row(ls, "ROUTE_ZERO_SHARES")
    assert d["next_earnings_read"] == "error: RuntimeError"
    assert "error" in d["add_to_loser"]


def test_concentration_zero_add_row_carries_both_keys():
    sizing = {"shares_initial": 1, "initial_dollar": 100.0, "position_size_pct": 0.25,
              "order_pct": 0.05, "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0}
    conc = {"passed": False, "errored": True, "ai_infra_size_cap_pct": 0.0,
            "flags": ["RISK CHECK FAILED — could not verify concentration: x. Buy blocked."]}
    for action in ("ADD", "OPEN"):
        ls, lpa, _ = _route(action=action, sizing=dict(sizing), conc=conc,
                            earnings=_in(4))
        lpa.assert_not_called()
        d = _row(ls, "ROUTE_CONCENTRATION_ZERO")
        assert d["concentration_check"]["shares_after"] == 0
        if action == "ADD":
            assert d["next_earnings_days"] == 4 and d["next_earnings_read"] == "ok"
            assert d["add_to_loser"]["basis"] == "avg_cost"
        else:
            assert not any(k in d for k in KEYS)


def test_micro_name_cap_add_row_carries_both_keys():
    sizing = {"shares_initial": 1, "initial_dollar": 100.0, "position_size_pct": 0.25,
              "order_pct": 0.05, "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0}
    micro = {"outcome": "blocked", "cap_pct": 30.0, "held_pct": 29.0, "shares_after": 0}
    for action in ("ADD", "OPEN"):
        ls, lpa, _ = _route(action=action, sizing=dict(sizing, flags=[]), micro=dict(micro),
                            earnings=_in(10))
        lpa.assert_not_called()
        d = _row(ls, "ROUTE_MICRO_NAME_CAP")
        assert d["micro_name_cap"]["outcome"] == "blocked"
        if action == "ADD":
            assert d["next_earnings_days"] == 10 and d["next_earnings_read"] == "ok"
            assert d["add_to_loser"]["breaks_rule"] is True
        else:
            assert not any(k in d for k in KEYS)


def test_passed_add_route_is_unchanged():
    """An ADD that sizes and routes gets none of these keys in risk_json and
    makes no earnings read for them."""
    sizing = {"shares_initial": 1, "initial_dollar": 100.0, "position_size_pct": 0.25,
              "order_pct": 0.05, "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0}
    with patch("agents.crew._attach_gap_risk_shadow"):
        ls, lpa, ge = _route(sizing=sizing, earnings=_in(1))
    lpa.assert_called_once()
    assert not any(k in lpa.call_args.kwargs["risk_json"] for k in ("next_earnings_days",
                                                                    "next_earnings_read"))
    ge.assert_not_called()


@pytest.mark.parametrize("bad", [None, "not a dict"])
def test_helper_never_raises(bad):
    out = crew._add_guard_record("NVDA", {"action": "ADD"}, bad, today=TODAY) \
        if bad is None else crew._add_guard_record("NVDA", bad, {}, today=TODAY)
    assert out is None or isinstance(out, dict)


def test_helper_unit():
    with patch("db.get_next_earnings", return_value=date(2026, 11, 13)):
        rec = crew._add_guard_record("NVDA", {"action": "ADD"}, {"add_to_loser": {"x": 1}},
                                     today=date(2026, 11, 16))
    # a past date the calendar still returns is kept raw (negative), not dropped
    assert rec == {"add_to_loser": {"x": 1}, "next_earnings_days": -3, "next_earnings_read": "ok"}
    assert crew._add_guard_record("NVDA", {"action": "OPEN"}, {}, today=TODAY) is None
