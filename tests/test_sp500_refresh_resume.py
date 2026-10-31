"""The weekly S&P 500 refresh resumes instead of restarting, and a weekday
catch-up refetches only tickers more than 7 days stale (audit A1031-0053-12)."""
from datetime import date
from unittest.mock import patch

from ingestion import price_history as ph

MON = date(2026, 11, 2)


def _run(newest, **kw):
    with patch("ingestion.sp500_universe.get_sp500_tickers", return_value=list(newest) + ["NEW"]), \
         patch("db.get_newest_bar_dates", return_value={k: v for k, v in newest.items() if v}), \
         patch("ingestion.price_history._fetch_bars", return_value=[{"date": "2026-10-30"}]) as fetch, \
         patch("db.upsert_price_bars", return_value=1), patch("ingestion.price_history.time.sleep"):
        n = ph.refresh_sp500_price_history(skip_watchlist=False, today=MON, **kw)
    return n, [c.args[0] for c in fetch.call_args_list]


def test_weekly_run_skips_tickers_already_current():
    n, fetched = _run({"CUR": date(2026, 10, 30), "OLD": date(2026, 10, 16)})
    assert fetched == ["OLD", "NEW"] and n == 2          # Friday 10-30 is current on Monday


def test_catchup_only_takes_tickers_over_a_week_old():
    n, fetched = _run({"CUR": date(2026, 10, 30), "WEEK": date(2026, 10, 27), "OLD": date(2026, 10, 16)},
                      max_age_days=7)
    assert fetched == ["OLD", "NEW"]


def test_last_weekday_before():
    assert ph._last_weekday_before(MON) == date(2026, 10, 30)
    assert ph._last_weekday_before(date(2026, 11, 4)) == date(2026, 11, 3)


def test_both_refresh_tasks_share_one_run_lock():
    import tasks
    with patch("tasks._run_singleton") as single:
        tasks.refresh_sp500_history.run()
        tasks.refresh_sp500_history_catchup.run()
    assert [c.args[0] for c in single.call_args_list] == ["sp500_history", "sp500_history"]


def test_a_no_op_catchup_records_no_heartbeat():
    import tasks
    with patch("tasks._run_singleton", side_effect=lambda name, fn, **k: fn()), \
         patch("ingestion.price_history.refresh_sp500_price_history", return_value=0), \
         patch("tasks._heartbeat_ok") as ok:
        tasks.refresh_sp500_history_catchup.run()
    ok.assert_not_called()
    with patch("tasks._run_singleton", side_effect=lambda name, fn, **k: fn()), \
         patch("ingestion.price_history.refresh_sp500_price_history", return_value=12), \
         patch("tasks._heartbeat_ok") as ok:
        tasks.refresh_sp500_history_catchup.run()
    ok.assert_called_once_with("sp500_history", 12)
