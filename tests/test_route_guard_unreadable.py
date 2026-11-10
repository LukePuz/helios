"""An unreadable duplicate-buy guard skips a buy with a ROUTE_GUARD_UNREADABLE
row instead of reading as "no prior approval" (audit 11-06 21:36 #8); a sell
goes on. No live DB/IBKR."""
from unittest.mock import MagicMock, patch

import pytest

from agents import c6_shadow, crew


def _result(action="OPEN", conviction=7):
    return {"ticker": "NVDA", "action": action, "conviction": conviction,
            "thesis_bucket": "Accelerator", "reasoning": "test",
            "provenance": {"model": "m", "prompt_sha": "p"}}


def _route(result, active, prior):
    log_signal = MagicMock()
    with patch("db.has_active_approval", **active), patch("db.get_last_executed_approval", **prior), \
         patch("db.log_signal", log_signal), patch("alerts._send"), \
         patch("risk_manager.RiskManager", side_effect=RuntimeError("stop here")), \
         patch("agents.crew._get_positions", return_value=None), \
         patch("db.is_near_earnings", return_value=False), patch("db.get_next_earnings", return_value=None), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}):
        try:
            crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
        except Exception:
            pass
    return [c.args[1] for c in log_signal.call_args_list]


_DOWN = {"side_effect": RuntimeError("EMAXCONNSESSION")}


@pytest.mark.parametrize("action", ["OPEN", "ADD"])
def test_an_unreadable_pending_check_skips_the_buy(action):
    kinds = _route(_result(action), _DOWN, {"return_value": None})
    assert kinds[0] == "ROUTE_GUARD_UNREADABLE"


def test_an_unreadable_cooldown_check_skips_the_buy():
    kinds = _route(_result("OPEN"), {"return_value": False}, _DOWN)
    assert kinds[0] == "ROUTE_GUARD_UNREADABLE"


def test_a_sell_goes_on_when_the_pending_check_is_unreadable():
    kinds = _route(_result("TRIM"), _DOWN, {"return_value": None})
    assert "ROUTE_GUARD_UNREADABLE" not in kinds and "SELL_POSITION_UNREADABLE" in kinds   # reached the sell branch


def test_guards_are_read_strictly():
    import db
    conn = MagicMock()
    conn.cursor.side_effect = RuntimeError("pool")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.has_active_approval("NVDA") is False
        assert db.get_last_executed_approval("NVDA", 4) is None
        with pytest.raises(RuntimeError):
            db.has_active_approval("NVDA", strict=True)
        with pytest.raises(RuntimeError):
            db.get_last_executed_approval("NVDA", 4, strict=True)


def test_c6_shadow_mirrors_the_closed_guard_for_buys():
    with patch("db.has_active_approval", **_DOWN), patch("db.get_last_executed_approval", **_DOWN), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("db.is_near_earnings", return_value=False), patch("db.get_next_earnings", return_value=None):
        blocks = c6_shadow._routing_blocks("NVDA", "OPEN", 7)
    assert blocks.count("guard_unreadable") == 1
