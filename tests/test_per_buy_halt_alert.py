"""RiskManager.is_halted's per-buy drawdown latch sends one Telegram alert
(percentages only), after the flag write and the halt row; it never raises
and never changes the return value."""
from unittest.mock import MagicMock, patch

import risk_manager
from risk_manager import RiskManager

ENV = risk_manager.TRADING_ENV


def _rm(pv, simulation=False):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=pv)
    return RiskManager(client, simulation=simulation)


def _run(rm, state, not_saved=(), send_error=None):
    order = MagicMock()
    order.set_state_verified.return_value = list(not_saved)
    if send_error:
        order._send.side_effect = send_error
    with patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state_verified", order.set_state_verified), \
         patch("db.log_halt_event", order.log_halt_event), \
         patch("db.set_state", order.set_state), \
         patch("alerts._send", order._send):
        out = rm.is_halted()
    return out, order


def _sends(order):
    return [c for c in order.mock_calls if c[0] == "_send"]


def test_a_latch_sends_one_alert_after_the_flag_and_the_row():
    out, order = _run(_rm(890.0), {f"high_water_mark_{ENV}": "1000"})
    assert out is True
    names = [c[0] for c in order.mock_calls]
    assert names == ["set_state_verified", "log_halt_event", "_send"]
    text = _sends(order)[0].args[0]
    assert "TRADING HALTED" in text and "per-buy check" in text
    assert "-11.0%" in text and f"limit {risk_manager.DRAWDOWN_HALT_PCT:.0%}" in text
    assert "$" not in text and "890" not in text and "1000" not in text and "1,000" not in text
    assert "could not be confirmed saved" not in text


def test_a_flag_that_did_not_save_says_so():
    _, order = _run(_rm(890.0), {f"high_water_mark_{ENV}": "1000"},
                    not_saved=[f"trading_halted_{ENV}"])
    assert "could not be confirmed saved" in _sends(order)[0].args[0]
    assert order.log_halt_event.call_args.kwargs["verified"] is False


def test_already_halted_sends_nothing():
    out, order = _run(_rm(890.0), {f"trading_halted_{ENV}": "true", f"high_water_mark_{ENV}": "1000"})
    assert out is True and not _sends(order) and not order.set_state_verified.called


def test_simulation_sends_nothing():
    out, order = _run(_rm(890.0, simulation=True), {f"high_water_mark_{ENV}": "1000"})
    assert out is True and order.mock_calls == []


def test_a_raising_alert_still_returns_true():
    out, order = _run(_rm(890.0), {f"high_water_mark_{ENV}": "1000"}, send_error=RuntimeError("telegram down"))
    assert out is True and order.log_halt_event.called


def test_a_failed_state_write_still_alerts_and_returns_true():
    state = {f"high_water_mark_{ENV}": "1000"}
    order_rm = _rm(890.0)
    with patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state_verified", side_effect=RuntimeError("db down")), \
         patch("db.log_halt_event") as ev, patch("alerts._send") as send:
        assert order_rm.is_halted() is True
    assert not ev.called and send.call_count == 1
    assert "could not be confirmed saved" in send.call_args.args[0]


def test_below_the_limit_sends_nothing():
    out, order = _run(_rm(950.0), {f"high_water_mark_{ENV}": "1000"})
    assert out is False and not _sends(order) and not order.set_state_verified.called
