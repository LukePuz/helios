"""Log-only daily-loss halt shadow (audit 2026-11-02 19:27 #10, roadmap
01(c)): logged once per ET day when equity is past DAILY_LOSS_SHADOW_PCT
below the day's first snapshot. The halt flag is never set."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import tasks

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)          # 10:00 ET
FIRST = {"timestamp": datetime(2026, 11, 3, 10, 5, tzinfo=timezone.utc), "total_value": 1000.0}


def _run(pv, state, first=FIRST, written=True, now=NOW):
    with patch("db.get_first_snapshot_since", return_value=first) as snap, \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_activity", return_value=written) as act, patch("config.TRADING_ENV", "live"):
        tasks._log_would_halt_daily_loss(pv, now=now)
    return act, snap


def test_a_loss_past_the_threshold_logs_once_per_et_day():
    state = {}
    act, snap = _run(960.0, state)
    since = snap.call_args.args[0]
    assert since.isoformat() == "2026-11-03T00:00:00-05:00"           # ET midnight
    stage, ticker, _, sentiment, extra = act.call_args.args
    assert (stage, ticker, sentiment) == ("risk_shadow", "", "neutral")   # not counted as an alert
    assert extra["would_halt_daily_loss"] is True and extra["day_pnl_pct"] == -4.0
    assert extra["threshold_pct"] == 3.0 and extra["et_date"] == "2026-11-03"
    assert state == {"would_halt_daily_loss_logged_live": "2026-11-03"}
    _run(950.0, state)[0].assert_not_called()                        # same day: once
    _run(950.0, state, now=datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc))[0].assert_called_once()


def test_small_losses_gains_and_missing_baselines_log_nothing():
    _run(980.0, {})[0].assert_not_called()
    _run(1100.0, {})[0].assert_not_called()
    _run(900.0, {}, first=None)[0].assert_not_called()
    _run(900.0, {}, first={"timestamp": None, "total_value": 0.0})[0].assert_not_called()
    _run(float("nan"), {})[0].assert_not_called()


def test_a_failed_write_leaves_the_key_unset():
    state = {}
    _run(900.0, state, written=False)
    assert state == {}


def test_threshold_comes_from_config():
    with patch("config.DAILY_LOSS_SHADOW_PCT", 0.01):
        _run(985.0, {})[0].assert_called_once()


def test_swallows_its_own_errors():
    with patch("db.get_first_snapshot_since", side_effect=RuntimeError("db down")):
        tasks._log_would_halt_daily_loss(900.0, now=NOW)


def test_drawdown_check_calls_the_shadow_and_never_halts_on_it():
    state = {"high_water_mark_live": "1000"}
    client = MagicMock()
    client.get_account.return_value.portfolio_value = 950.0
    with patch("tasks._get_client", return_value=client), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.set_state_verified") as verified, \
         patch("db.get_first_snapshot_since", return_value=FIRST), \
         patch("db.log_activity", return_value=True) as act, \
         patch("tasks._monitor_health"), patch("config.TRADING_ENV", "live"):
        tasks.check_portfolio_drawdown.run()
    assert act.call_args.args[4]["would_halt_daily_loss"] is True
    verified.assert_not_called()
    assert state.get("trading_halted_live") is None


def test_first_snapshot_query_is_env_scoped_and_ascending():
    import db
    cur = MagicMock()
    cur.fetchone.return_value = (FIRST["timestamp"], 1000, 2)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.get_first_snapshot_since(FIRST["timestamp"])
    sql, params = cur.execute.call_args[0]
    assert "trading_env=%s" in sql and "ORDER BY timestamp ASC LIMIT 1" in sql
    # m266: a $0 cold-start snapshot is never the baseline; skipped ones are counted
    assert "AND total_value > 0" in sql
    assert params[3] == FIRST["timestamp"]
    assert out == {**FIRST, "total_value": 1000.0, "skipped_nonpositive": 2}


def test_the_row_counts_skipped_nonpositive_baselines():
    extra = _run(960.0, {}, first={**FIRST, "skipped_nonpositive": 1})[0].call_args.args[4]
    assert extra["baseline_skipped_nonpositive"] == 1


def test_the_row_names_its_code_and_config():
    """m268: code_sha and env_sha ride on the shadow row itself."""
    from agents import analyst
    extra = _run(960.0, {})[0].call_args.args[4]
    assert extra["code_sha"] == "test"
    assert extra["env_sha"] == analyst._config_env_record().get("env_sha") and extra["env_sha"]
