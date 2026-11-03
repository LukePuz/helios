"""Log-only tranche2 shadow: a held position whose entry plan's add_trigger
was crossed by the latest daily close gets one activity_feed row. Nothing
is queued, sized or traded."""
from datetime import date, datetime, timezone
from unittest.mock import patch

import stop_lock
import tasks

POSITIONS = {"NVDA": {"qty": 3, "market_value": 1950.0}}
OUTCOMES = [{"approval_id": 11, "ticker": "NVDA", "entry_price": 600.0,
             "entry_date": datetime(2026, 11, 2, 15, tzinfo=timezone.utc)}]
APPROVAL = {"id": 11, "action": "OPEN", "timestamp": datetime(2026, 11, 2, 14, tzinfo=timezone.utc),
            "analyst_json": {"provenance": {"signal_id": 42}},
            "risk_json": {"initial_entry_pct": 0.55, "dollar_amount": 4000.0, "shares_initial": 3,
                          "plan_compiled": {"source": "ladder", "t1": 640.0, "add_trigger": 645.0,
                                            "time_stop": "2026-12-15"}}}


def _bar(d, close):
    return {"date": d, "open": close, "high": close, "low": close, "close": close, "volume": 1}


def _run(close=650.0, approval=APPROVAL, keys=None, positions=POSITIONS, written=True,
         bars=None, outcomes=OUTCOMES, approvals=None):
    if bars is None:
        bars = {"NVDA": [_bar("2026-11-02", 600.0), _bar("2026-11-03", close)]}
    get_appr = (lambda i: approvals.get(i)) if approvals is not None else (lambda i: approval)
    with patch("tasks._snapshot_positions_or_none", return_value=positions), \
         patch("db.get_open_outcomes", return_value=outcomes), \
         patch("db.get_approval_by_id", side_effect=get_appr), \
         patch("db.get_price_history_bulk", return_value=bars), \
         patch("db.get_activity_shadow_keys", return_value=keys), \
         patch("db.log_activity", return_value=written) as act, \
         patch("config.TRADING_ENV", "live"):
        n = tasks._check_tranche2_shadow(today=date(2026, 11, 3))
    return n, act


def test_a_crossed_trigger_writes_one_row_with_the_raw_facts():
    n, act = _run()
    assert n == 1 and act.call_count == 1
    stage, ticker, _, _, extra = act.call_args.args
    assert (stage, ticker) == ("tranche2_shadow", "NVDA")
    assert extra["crossed"] is True and extra["add_trigger"] == 645.0 and extra["close"] == 650.0
    assert extra["close_date"] == "2026-11-03" and extra["bar_is_today"] is True
    assert extra["close_minus_trigger"] == 5.0
    assert extra["held_shares"] == 3 and extra["initial_entry_pct"] == 0.55
    assert extra["tranche2_fraction"] == 0.45
    # No initial_dollar recorded: int(4000 / 600) - 3 shares bought
    assert extra["initial_dollar"] is None and extra["tranche2_shares_est"] == 3
    assert extra["approval_id"] == 11 and extra["verdict_signal_id"] == 42
    assert extra["plan_source"] == "ladder" and extra["env"] == "live"
    assert extra["plan_time_stop"] == "2026-12-15"
    assert extra["shadow_key"] == "live:NVDA:a11:tranche2"
    assert "code_sha" in extra and "env_sha" in extra


def test_a_close_under_the_trigger_writes_nothing():
    n, act = _run(close=644.99)
    assert n == 0 and not act.called


def test_a_second_run_writes_nothing():
    assert _run()[0] == 1
    n, act = _run()                                   # Redis marker
    assert n == 0 and not act.called
    stop_lock._redis().store.clear()                  # a Redis flush
    n, act = _run(keys={"live:NVDA:a11:tranche2"})    # durable shadow key
    assert n == 0 and not act.called


def test_a_failed_write_retries_next_run():
    assert _run(written=False)[0] == 0
    assert _run()[0] == 1


def test_missing_plan_or_fields_no_crash_no_row():
    for approval in (None, {"id": 11, "risk_json": None},
                     {"id": 11, "risk_json": {"plan_compiled": {"error": "x"}}},
                     {"id": 11, "risk_json": {"plan_compiled": {"add_trigger": None}}}):
        n, act = _run(approval=approval)
        assert n == 0 and not act.called
    # Plan present, sizing fields absent: the row still goes out with nulls.
    n, act = _run(approval={"id": 11, "risk_json": {"plan_compiled": {"add_trigger": 645.0}}})
    extra = act.call_args.args[4]
    assert n == 1 and extra["initial_entry_pct"] is None and extra["tranche2_shares_est"] is None


def test_a_bar_older_than_the_plan_is_not_counted():
    # The cache's latest bar (bars in date order) predates the 11-02 approval.
    n, act = _run(bars={"NVDA": [_bar("2026-10-30", 600.0), _bar("2026-11-01", 650.0)]})
    assert n == 0 and not act.called


def test_no_snapshot_or_nothing_held_writes_nothing():
    for positions in (None, {}, {"NVDA": {"qty": 0}}):
        n, act = _run(positions=positions)
        assert n == 0 and not act.called


def test_its_own_errors_are_swallowed():
    with patch("tasks._snapshot_positions_or_none", side_effect=RuntimeError("db down")):
        assert tasks._check_tranche2_shadow() == 0


def test_an_exception_does_not_break_the_host_task():
    with patch("ingestion.price_history.refresh_all_price_history", return_value=3), \
         patch("tasks._heartbeat_ok") as ok, patch("tasks._heartbeat_fail") as fail, \
         patch("tasks._check_pending_entries"), patch("tasks._log_plan_cross_section") as xs, \
         patch("tasks._check_tranche2_shadow", side_effect=RuntimeError("boom")) as t2:
        tasks.refresh_price_history.run()
    t2.assert_called_once()
    xs.assert_called_once()
    ok.assert_called_once()
    assert not fail.called


def test_the_host_task_runs_it_after_the_cross_section():
    order = []
    with patch("ingestion.price_history.refresh_all_price_history", return_value=3), \
         patch("tasks._heartbeat_ok"), patch("tasks._heartbeat_fail"), \
         patch("tasks._check_pending_entries", side_effect=lambda: order.append("pending")), \
         patch("tasks._log_plan_cross_section", side_effect=lambda: order.append("xs")), \
         patch("tasks._check_tranche2_shadow", side_effect=lambda: order.append("t2")):
        tasks.refresh_price_history.run()
    assert order == ["pending", "xs", "t2"]


def test_a_capped_first_tranche_counts_what_it_actually_spent():
    # The risk cap cut the 55% tranche (2200) to 2 shares at 600 (1200), so
    # the rest of the 4000 full size is 2800, not 4000 x 0.45 = 1800.
    risk = {**APPROVAL["risk_json"], "initial_dollar": 1200.0, "shares_initial": 2}
    n, act = _run(approval={**APPROVAL, "risk_json": risk})
    extra = act.call_args.args[4]
    assert n == 1 and extra["initial_dollar"] == 1200.0
    assert extra["tranche2_shares_est"] == 4          # int((4000 - 1200) / 600)


def test_a_one_go_sizing_writes_nothing():
    risk = {**APPROVAL["risk_json"], "initial_entry_pct": 1.0}
    n, act = _run(approval={**APPROVAL, "risk_json": risk})
    assert n == 0 and not act.called


def test_the_first_failed_write_stops_the_run(caplog):
    positions = {"AMD": {"qty": 2}, "NVDA": {"qty": 3}}
    outcomes = OUTCOMES + [{**OUTCOMES[0], "approval_id": 12, "ticker": "AMD"}]
    approvals = {11: APPROVAL, 12: {**APPROVAL, "id": 12}}
    bars = {t: [_bar("2026-11-03", 650.0)] for t in ("AMD", "NVDA")}
    n, act = _run(written=False, positions=positions, outcomes=outcomes,
                  approvals=approvals, bars=bars)
    assert n == 0 and act.call_count == 1             # the second ticker isn't tried
    assert "write failed" in caplog.text
    assert not stop_lock._redis().store               # its marker was released


def test_redis_down_falls_back_to_the_db_keys():
    with patch.object(stop_lock, "_redis", side_effect=ConnectionError("down")):
        n, act = _run(keys=set())
        assert n == 1 and act.call_count == 1
        n, act = _run(keys={"live:NVDA:a11:tranche2"})
        assert n == 0 and not act.called
        n, act = _run(keys=None)                      # no dedup record at all: write nothing
        assert n == 0 and not act.called

    class SetFails:
        def set(self, *a, **k):
            raise ConnectionError("down")

        def delete(self, *a):
            raise ConnectionError("down")
    with patch.object(stop_lock, "_redis", return_value=SetFails()):
        n, act = _run(keys=set())
        assert n == 1
        n, act = _run(keys=set(), written=False)      # no marker to release; no crash
        assert n == 0


def test_the_newest_open_outcome_on_a_ticker_wins():
    old = {**OUTCOMES[0], "approval_id": 9,
           "entry_date": datetime(2026, 10, 20, tzinfo=timezone.utc)}
    approvals = {9: {**APPROVAL, "id": 9}, 11: APPROVAL}
    for outcomes in ([old, OUTCOMES[0]], [OUTCOMES[0], old]):
        stop_lock._redis().store.clear()
        n, act = _run(outcomes=outcomes, approvals=approvals)
        assert n == 1 and act.call_args.args[4]["approval_id"] == 11
