"""A stale high-water mark must not disable the drawdown halt forever
(limitations audit 2026-10-28: live HWM was the $1M paper default)."""
from unittest.mock import MagicMock, patch

import tasks


def _run(state):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1900.0)
    sets = {}

    def set_state(k, v):
        state[k] = v
        sets[k] = v

    with patch("tasks._get_client", return_value=client), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=set_state), \
         patch("alerts._send") as send, \
         patch("config.TRADING_ENV", "live"):
        tasks.check_portfolio_drawdown.run()
    return sets, send


def test_stale_baseline_is_reset_after_three_reads():
    state = {"high_water_mark_live": "1000000"}
    for _ in range(2):
        sets, send = _run(state)
        assert "high_water_mark_live" not in sets
        send.assert_not_called()
    sets, send = _run(state)
    assert float(sets["high_water_mark_live"]) == 1900.0
    send.assert_called_once()


def test_a_plausible_read_resets_the_counter():
    state = {"high_water_mark_live": "2000", "drawdown_implausible_count_live": "2"}
    sets, _ = _run(state)
    assert state["drawdown_implausible_count_live"] == "0"
