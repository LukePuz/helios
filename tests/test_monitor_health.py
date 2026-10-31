"""Mechanical monitors alert after repeated failures (audit A1031-1053-8)."""
from unittest.mock import MagicMock, patch

import tasks


def _health(state, failed):
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("config.TRADING_ENV", "live"), patch("alerts._send") as send:
        tasks._monitor_health("trailing_stops", failed)
    return send


def test_alerts_once_at_three_and_once_on_recovery():
    state = {}
    for _ in range(2):
        _health(state, True).assert_not_called()
    send = _health(state, True)
    assert "MONITOR BLIND" in send.call_args[0][0]
    _health(state, True).assert_not_called()                    # no repeat at 4
    send = _health(state, False)
    assert "recovered after 4" in send.call_args[0][0]
    assert state["monitor_fail_count_trailing_stops_live"] == "0"


def test_a_short_failure_streak_recovers_silently():
    state = {"monitor_fail_count_trailing_stops_live": "2"}
    _health(state, False).assert_not_called()
    assert state["monitor_fail_count_trailing_stops_live"] == "0"


def test_trailing_stop_read_failure_counts_as_a_failure():
    rm = MagicMock()
    rm.check_trailing_stops.return_value = []
    rm.monitor_read_failed = True
    with patch("tasks._get_client"), patch("risk_manager.RiskManager", return_value=rm), \
         patch("tasks._monitor_health") as health:
        tasks.check_trailing_stops.run()
    health.assert_called_once_with("trailing_stops", True)
    rm.monitor_read_failed = False
    with patch("tasks._get_client"), patch("risk_manager.RiskManager", return_value=rm), \
         patch("tasks._monitor_health") as health:
        tasks.check_trailing_stops.run()
    health.assert_called_once_with("trailing_stops", False)


def test_risk_manager_flags_a_failed_read():
    import risk_manager
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1000.0)
    client.get_all_positions.side_effect = RuntimeError("gateway down")
    with patch("db.get_all_position_stops", return_value=[]):
        rm = risk_manager.RiskManager(client)
        assert rm.check_trailing_stops() == [] and rm.monitor_read_failed is True
