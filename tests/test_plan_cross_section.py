"""Log-only plan_cross_section snapshot (audit 2026-11-03 01:29 #5): one
activity_feed row per held or recently-reviewed ticker with the stored
compile_plan components. Raw components only; nothing ranks or trades."""
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import tasks

NOW = datetime(2026, 11, 3, 21, 50, tzinfo=timezone.utc)      # Tue 16:50 ET
VERDICT_TS = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)

AAPL_V = {"signal_id": 11, "timestamp": VERDICT_TS, "signal": "HOLD", "conviction": 7,
          "model_action": "HOLD",
          "udr_structural": {"price": 200.0, "structural_stop": 180.0, "t1": 220.0, "t2": 250.0,
                             "plan_source": "ladder"},
          "pending_entry": None,
          "model_output": {"catalyst": {"date": "2026-11-20"}, "thesis_bucket": "Accelerator"},
          "plan": {"price": 200.0}}
NVDA_V = {"signal_id": 12, "timestamp": VERDICT_TS, "signal": "WATCH", "conviction": 6,
          "model_action": "OPEN",
          "udr_structural": {"price": 600.0, "structural_stop": 560.0, "t1": None, "t2": 700.0,
                             "plan_source": "ladder"},
          "pending_entry": {"price": 600.0, "expires": "2026-11-25"},
          "model_output": {}, "plan": None}


def _run(positions=None, stops=(), recent=None, held_v=None, bars=None, keys=None,
         written=True, now=NOW, stops_error=None, calls=None):
    def latest(days=None, tickers=None):
        if calls is not None:
            calls.append({"days": days, "tickers": tickers})
        return recent if tickers is None else held_v

    with ExitStack() as st:
        st.enter_context(patch("tasks._snapshot_positions_or_none", return_value=positions))
        st.enter_context(patch("db.get_all_position_stops",
                               return_value=[{"ticker": t} for t in stops],
                               side_effect=stops_error))
        st.enter_context(patch("db.get_latest_verdicts", side_effect=latest))
        bulk = st.enter_context(patch("db.get_price_history_bulk", return_value=bars or {}))
        st.enter_context(patch("db.get_activity_shadow_keys", return_value=keys))
        act = st.enter_context(patch("db.log_activity", return_value=written))
        st.enter_context(patch("config.TRADING_ENV", "live"))
        n = tasks._log_plan_cross_section(now=now)
    return n, act, bulk


def _rows(act):
    return {c.args[1]: c.args[4] for c in act.call_args_list}


def test_rows_for_held_and_recent_tickers_with_stored_components():
    n, act, bulk = _run(
        positions={"AAPL": {"qty": 10, "market_value": 2100.0}}, stops=["AAPL"],
        recent={"NVDA": NVDA_V}, held_v={"AAPL": AAPL_V},
        bars={"AAPL": [{"date": "2026-11-03", "close": 210.0}],
              "NVDA": [{"date": "2026-11-03", "close": 620.0}]})
    assert n == 2 and bulk.call_args.args[0] == ["AAPL", "NVDA"]
    rows = _rows(act)
    assert all(c.args[0] == "plan_cross_section" for c in act.call_args_list)
    a, v = rows["AAPL"], rows["NVDA"]
    assert a["held"] is True and a["held_broker"] is True and a["shares"] == 10.0
    assert a["shadow_key"] == "live:2026-11-03:AAPL" and a["snapshot_date"] == "2026-11-03"
    assert a["env"] == "live" and "code_sha" in a and "env_sha" in a
    assert a["price"] == 210.0 and a["price_source"] == "price_history_close"
    assert (a["structural_stop"], a["t1"], a["t2"], a["compile_source"]) == (180.0, 220.0, 250.0, "ladder")
    assert a["dist_t1"] == round((220 - 210) / 210, 6)
    assert a["dist_structural_stop"] == round((180 - 210) / 210, 6)
    assert a["change_since_verdict"] == 0.05
    # time_stop not stored for a HOLD: recomputed from model_output (catalyst + 5 d)
    assert a["time_stop"] == "2026-11-25" and a["time_stop_source"] == "recomputed"
    assert a["plan_source"] == "stored+time_stop_recomputed" and a["days_to_time_stop"] == 22
    assert a["action"] == "HOLD" and a["conviction"] == 7 and a["verdict_age_days"] == 1
    assert v["held"] is False and v["shares"] is None and v["in_recent_window"] is True
    assert v["plan_source"] == "stored" and v["time_stop_source"] == "stored"
    assert v["t1"] is None and v["dist_t1"] is None and v["model_action"] == "OPEN"
    for row in rows.values():       # raw components only
        assert not any(k in row for k in ("rank", "score", "would_rotate"))


def test_missing_plan_gives_nulls_not_a_crash():
    n, act, _ = _run(positions={"MSFT": {"qty": 5, "market_value": 2000.0}},
                     recent={}, held_v={}, bars={})
    assert n == 1
    m = _rows(act)["MSFT"]
    assert m["plan_source"] == "none" and m["action"] is None
    assert m["structural_stop"] is None and m["dist_t2"] is None and m["time_stop"] is None
    assert m["price"] == 400.0 and m["price_source"] == "snapshot_mark"
    # a verdict row with no stored plan output at all
    bare = {"signal_id": 3, "timestamp": VERDICT_TS, "signal": "WATCH", "conviction": 5,
            "model_action": None, "udr_structural": "junk", "pending_entry": None,
            "model_output": None, "plan": None}
    n, act, _ = _run(recent={"TSLA": bare}, held_v={})
    t = _rows(act)["TSLA"]
    assert n == 1 and t["plan_source"] == "none" and t["price"] is None
    assert t["dist_structural_stop"] is None and t["change_since_verdict"] is None


def test_a_second_run_writes_nothing():
    kw = dict(recent={"NVDA": NVDA_V}, held_v={},
              bars={"NVDA": [{"date": "2026-11-03", "close": 620.0}]})
    n, act, _ = _run(keys=set(), **kw)
    assert n == 1
    keys = {c.args[4]["shadow_key"] for c in act.call_args_list}
    n, act, _ = _run(keys=set(), **kw)                       # Redis marker
    assert n == 0 and not act.called
    import stop_lock
    stop_lock._redis().store.clear()                         # a Redis flush
    n, act, _ = _run(keys=keys, **kw)                        # activity_feed shadow_key
    assert n == 0 and not act.called


def test_a_failed_write_retries_and_weekends_and_failed_reads_write_nothing():
    kw = dict(recent={"NVDA": NVDA_V}, held_v={})
    assert _run(written=False, **kw)[0] == 0
    assert _run(**kw)[0] == 1
    assert _run(now=datetime(2026, 11, 7, 21, 0, tzinfo=timezone.utc), **kw)[0] == 0   # Saturday
    assert _run(recent=None, held_v={})[0] == 0


def test_an_exception_inside_does_not_break_the_host_task():
    with patch("ingestion.price_history.refresh_all_price_history", return_value=3), \
         patch("tasks._heartbeat_ok") as ok, patch("tasks._heartbeat_fail") as fail, \
         patch("tasks._check_pending_entries") as chk, \
         patch("db.get_latest_verdicts", side_effect=RuntimeError("boom")):
        tasks.refresh_price_history.run()
    chk.assert_called_once()
    ok.assert_called_once()
    assert not fail.called
    with patch("ingestion.price_history.refresh_all_price_history", return_value=3), \
         patch("tasks._heartbeat_ok"), patch("tasks._heartbeat_fail") as fail, \
         patch("tasks._check_pending_entries"), \
         patch("tasks._log_plan_cross_section", side_effect=RuntimeError("boom")) as xs:
        tasks.refresh_price_history.run()
    xs.assert_called_once()
    assert not fail.called


def test_only_a_stored_time_stop_is_labelled_time_stop_stored():
    v = {**NVDA_V, "udr_structural": None}
    n, act, _ = _run(recent={"NVDA": v}, held_v={})
    row = _rows(act)["NVDA"]
    assert row["plan_source"] == "time_stop_stored" and row["time_stop"] == "2026-11-25"
    assert row["structural_stop"] is None and row["compile_source"] is None and "source" not in row


class _DeadRedis:
    def set(self, *a, **k):
        raise ConnectionError("redis down")

    def delete(self, *a, **k):
        raise ConnectionError("redis down")


def test_redis_down_uses_the_db_keys_alone_and_writes_nothing_without_them():
    import stop_lock
    kw = dict(recent={"NVDA": NVDA_V}, held_v={})
    with patch.object(stop_lock, "_redis", lambda: _DeadRedis()):
        n, act, _ = _run(keys=set(), **kw)
        assert n == 1 and act.call_count == 1
        n, act, _ = _run(keys={"live:2026-11-03:NVDA"}, **kw)
        assert n == 0 and not act.called
        n, act, _ = _run(keys=None, **kw)                    # no dedup source at all
        assert n == 0 and not act.called


def test_both_held_sources_unknown_skips_the_day():
    calls = []
    n, act, _ = _run(positions=None, stops_error=RuntimeError("db down"),
                     recent={"NVDA": NVDA_V}, held_v={}, calls=calls)
    assert n == 0 and not act.called and calls == []


def test_the_loop_stops_at_the_first_failed_write():
    n, act, _ = _run(recent={"AMD": NVDA_V, "MSFT": NVDA_V, "NVDA": NVDA_V}, held_v={},
                     written=False)
    assert n == 0 and act.call_count == 1
    n, act, _ = _run(recent={"AMD": NVDA_V, "MSFT": NVDA_V, "NVDA": NVDA_V}, held_v={})
    assert n == 3                                            # the failed one retries next run


def test_a_held_ticker_with_a_verdict_older_than_the_window():
    old = {**AAPL_V, "timestamp": datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)}
    calls = []
    n, act, _ = _run(positions={"AAPL": {"qty": 3, "market_value": 630.0}}, stops=["AAPL"],
                     recent={}, held_v={"AAPL": old}, calls=calls)
    assert {"days": 365, "tickers": ["AAPL"]} in calls and {"days": 5, "tickers": None} in calls
    row = _rows(act)["AAPL"]
    assert n == 1 and row["held"] is True and row["in_recent_window"] is False
    assert row["verdict_age_days"] == 33 and row["t1"] == 220.0 and row["shares"] == 3.0
    assert row["price"] == 210.0 and row["price_source"] == "snapshot_mark"
