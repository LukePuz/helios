"""30/60/90-day checkpoint writes: one open row, written once, with an
activity_feed row per checkpoint."""
from datetime import date, datetime, time as dtime, timedelta, timezone
from unittest.mock import MagicMock, patch

import db
import tasks


def _conn(rowcount=1, exc=None):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = rowcount
    if exc:
        cur.execute.side_effect = exc
    return conn, cur


def test_update_by_outcome_id_needs_an_open_row_with_the_checkpoint_unset():
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.update_outcome_price(62, 30, 101.5, outcome_id=88) is True
    sql, params = cur.execute.call_args.args
    assert "WHERE id=%s" in sql and "exit_date IS NULL" in sql and "review_30d_price IS NULL" in sql
    assert params == (101.5, 88)


def test_update_by_approval_is_scoped_to_env_open_rows_and_unset_column(monkeypatch):
    monkeypatch.setattr(db, "TRADING_ENV", "live")
    conn, cur = _conn(rowcount=0)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.update_outcome_price(67, 60, 50.0) is False
    sql, params = cur.execute.call_args.args
    assert "approval_id=%s AND trading_env=%s" in sql and "exit_date IS NULL" in sql
    assert "review_60d_price IS NULL" in sql and params == (50.0, 67, "live")


def test_update_failure_returns_none_and_unknown_horizon_false():
    conn, _ = _conn(exc=RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.update_outcome_price(71, 90, 1.0, outcome_id=3) is None
    assert db.update_outcome_price(71, 45, 1.0) is False


def _outcome(**kw):
    o = {"id": 501, "approval_id": 62, "ticker": "NVDA", "trading_env": "live",
         # Midday UTC on the local date 65 days back, so days_held (local date
         # minus entry_date's UTC date) is 65 at any hour of the day.
         "entry_date": datetime.combine(date.today() - timedelta(days=65), dtime(12),
                                        tzinfo=timezone.utc),
         "review_30d_price": None, "review_60d_price": None, "review_90d_price": None}
    return o | kw


def _review(outcomes, written=True):
    fast = MagicMock()
    fast.fast_info.last_price = 120.0
    with patch("db.get_open_outcomes", return_value=outcomes), \
         patch("db.update_outcome_price", return_value=written) as upd, \
         patch("db.log_activity", return_value=True) as act, \
         patch("yfinance.Ticker", return_value=fast):
        tasks.review_trade_outcomes.run()
    return upd, act


def test_each_written_checkpoint_gets_an_activity_row_by_outcome_id():
    upd, act = _review([_outcome()])
    assert [c.args[1] for c in upd.call_args_list] == [30, 60]
    assert all(c.kwargs["outcome_id"] == 501 for c in upd.call_args_list)
    assert act.call_count == 2
    stage, ticker, _msg, _sent, extra = act.call_args.args
    assert stage == "outcome_checkpoint" and ticker == "NVDA"
    assert extra["checkpoint_days"] == 60 and extra["days_held"] == 65
    assert extra["outcome_id"] == 501 and extra["env"] == "live" and extra["source"]


def test_open_outcomes_all_envs_drops_the_env_filter(monkeypatch):
    monkeypatch.setattr(db, "TRADING_ENV", "live")
    conn, cur = _conn()
    cur.fetchall.return_value = []
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.get_open_outcomes(all_envs=True)
        sql_all, params_all = cur.execute.call_args.args
        db.get_open_outcomes()
        sql_env, params_env = cur.execute.call_args.args
    assert "trading_env=%s" not in sql_all and params_all == ()
    assert "trading_env=%s" in sql_env and params_env == ("live",)


def test_review_covers_paper_rows_by_id_with_market_data_only(monkeypatch):
    """A paper row under a live process: written by its own id, one yfinance
    fetch per ticker, and no account or broker call."""
    monkeypatch.setattr("config.TRADING_ENV", "live")
    rows = [_outcome(), _outcome(id=77, approval_id=12, trading_env="paper")]
    fast = MagicMock()
    fast.fast_info.last_price = 120.0
    with patch("db.get_open_outcomes", return_value=rows) as get, \
         patch("db.update_outcome_price", return_value=True) as upd, \
         patch("db.log_activity", return_value=True) as act, \
         patch("yfinance.Ticker", return_value=fast) as yt, \
         patch("tasks._get_client", side_effect=AssertionError("no account calls")):
        tasks.review_trade_outcomes.run()
    assert get.call_args.kwargs == {"all_envs": True}
    assert yt.call_count == 1                                     # same ticker, one fetch
    assert {c.kwargs["outcome_id"] for c in upd.call_args_list} == {501, 77}
    assert {c.args[4]["env"] for c in act.call_args_list} == {"live", "paper"}


def test_no_activity_row_when_nothing_was_written():
    _, act = _review([_outcome()], written=False)
    act.assert_not_called()
    _, act = _review([_outcome()], written=None)
    act.assert_not_called()
