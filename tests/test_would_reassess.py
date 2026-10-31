"""Log-only shadow of the dead intraday VIX-shock reassessment (A1031-1453-2)."""
from unittest.mock import patch

import tasks

ROW = {"regime": "Neutral", "regime_source": "overall", "timestamp": "2026-11-02 05:01:00+00",
       "signals_json": {"volatility": {"vix": 17.0}}}


def _run(live, state, row=ROW):
    with patch("db.get_latest_regime", return_value=row), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_activity") as act, patch("config.TRADING_ENV", "live"):
        tasks._log_would_reassess(live)
    return act


def test_logs_once_per_regime_row_when_vix_moves_more_than_5():
    state = {}
    act = _run(23.5, state)
    extra = act.call_args.args[4]
    assert act.call_args.args[0] == "regime_shadow" and extra["would_reassess"] is True
    assert extra["delta"] == 6.5 and extra["regime"] == "Neutral"
    _run(24.0, state).assert_not_called()                  # same stored row: once


def test_small_moves_and_sector_rows_log_nothing():
    _run(21.0, {}).assert_not_called()
    _run(30.0, {}, {**ROW, "regime_source": "gics_sector:XLK"}).assert_not_called()
    _run(30.0, {}, {**ROW, "signals_json": {}}).assert_not_called()


def test_a_nan_vix_logs_nothing():
    state = {}
    _run(float("nan"), state).assert_not_called()
    assert state == {}
