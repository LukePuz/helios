"""scripts/batch_verify.py: OK / WARN / FAIL per record on fixture rows, the
watcher's log counts, and that the script only reads."""
import importlib.util
import inspect
import os
from datetime import datetime, timezone

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "batch_verify.py")
_spec = importlib.util.spec_from_file_location("batch_verify", _PATH)
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)

NY = bv.NY
SINCE = datetime(2026, 11, 24, 5, 0, tzinfo=NY)                 # Tue 05:00 ET
NOW = datetime(2026, 11, 24, 12, 10, tzinfo=NY)                 # Tue 12:10 ET


def _by(checks):
    return {c["name"]: c for c in checks}


# ------------------------------------------------------------ entry_gate

def _gate(path="event", market=True, trend_blocked=False, trend_read="ok", stop_read="ok"):
    g = {"path": path, "stopout": {"days": 10, "read": stop_read, "blocked": False}}
    if path != "deferred_retry":
        g["trend"] = {"read": trend_read, "blocked": trend_blocked,
                      "market": ({"ticker": "SPY", "read": "ok", "downtrend": True}
                                 if market else None)}
    return {"id": 1, "ticker": "AMD", "decision": {"entry_gate": g}}


def test_entry_gate_ok_warn_and_deferred_retry_needs_stopout_only():
    assert bv.check_entry_gate([])["status"] == "WARN"
    c = bv.check_entry_gate([_gate(trend_blocked=True), _gate(path="deferred_retry")])
    assert c["status"] == "OK" and c["rows"] == 2
    assert c["reads"]["path"] == {"event": 1, "deferred_retry": 1}


def test_entry_gate_missing_keys_fail():
    no_market_key = _gate()
    del no_market_key["decision"]["entry_gate"]["trend"]["market"]
    no_stop = _gate()
    del no_stop["decision"]["entry_gate"]["stopout"]
    no_gate = {"id": 9, "ticker": "X", "decision": {}}
    for row in (no_market_key, no_stop, no_gate):
        assert bv.check_entry_gate([row])["status"] == "FAIL"


def test_entry_gate_trend_block_without_market_fails():
    c = bv.check_entry_gate([_gate(trend_blocked=True, market=False)])
    assert c["status"] == "FAIL" and "no market dict" in c["why"]
    # market None is normal when the stock would not block
    assert bv.check_entry_gate([_gate(market=False)])["status"] == "OK"


def test_error_share_over_20_percent_fails_at_20_passes():
    rows = [_gate() for _ in range(4)] + [_gate(trend_read="error: TimeoutError")]
    assert bv.check_entry_gate(rows)["status"] == "OK"                    # 1 of 5 = 20%
    rows.append(_gate(stop_read="error: OperationalError"))
    rows.append(_gate(trend_read="error: TimeoutError"))
    c = bv.check_entry_gate(rows)                                        # trend 2 of 7 = 29%
    assert c["status"] == "FAIL" and "trend.read" in c["why"]


# ------------------------------------------------------------ story_dedup

def test_story_dedup():
    ok = {"id": 1, "ticker": "NVDA", "kind": "HOLD",
          "trigger": {"path": "event", "story_dedup": {"read": "ok", "would_skip_duplicate": False}}}
    miss = {"id": 2, "ticker": "NVDA", "kind": "WATCH", "trigger": {"path": "event"}}
    assert bv.check_story_dedup([])["status"] == "WARN"
    assert bv.check_story_dedup([ok, ok])["status"] == "OK"
    c = bv.check_story_dedup([ok, miss])
    assert c["status"] == "FAIL" and c["missing"] == [2]
    assert c["reads"]["missing by kind"] == {"WATCH": 1}


# ------------------------------------------------------------ approvals

def _approval(i=1, action="OPEN", is_test=False, **over):
    risk = {"rr_vs_stop": {"r_to_target": 2.1, "reason": None},
            "vol_target_shadow": {"read": "ok", "weights_at": "final_order"},
            "stop_vol": {"reason": None, "stop_sigma20": 0.7},
            "book_heat": {"read": "ok"},
            "recheck_contradiction": {"read": "ok", "would_block": False}}
    for k, v in over.items():
        if v is None:
            risk.pop(k)
        else:
            risk[k] = v
    return {"id": i, "ticker": "MSFT", "action": action, "is_test": is_test, "risk_json": risk}


def test_approvals_ok_and_non_buys_and_tests_ignored():
    checks = _by(bv.check_approvals([_approval(1), _approval(2, action="ADD"),
                                     _approval(3, action="SELL", book_heat=None),
                                     _approval(4, is_test=True, rr_vs_stop=None)]))
    assert all(c["status"] == "OK" and c["rows"] == 2 for c in checks.values())
    assert checks["rr_vs_stop"]["reads"]["reason"] == {"ok": 2}


def test_approvals_none_is_warn():
    assert all(c["status"] == "WARN" for c in bv.check_approvals([_approval(action="SELL")]))


@pytest.mark.parametrize("key", ["rr_vs_stop", "vol_target_shadow", "stop_vol", "book_heat",
                                 "recheck_contradiction"])
def test_approval_missing_field_fails_that_record_only(key):
    checks = _by(bv.check_approvals([_approval(1), _approval(2, **{key: None})]))
    assert checks[key]["status"] == "FAIL" and checks[key]["missing"] == [2]
    assert all(c["status"] == "OK" for k, c in checks.items() if k != key)


def test_vol_shadow_weights_not_at_final_order_fails():
    c = _by(bv.check_approvals([_approval(vol_target_shadow={"read": "ok",
                                                              "weights_at": "size_position"})]))
    assert c["vol_target_shadow"]["status"] == "FAIL" and "weights_at" in c["vol_target_shadow"]["why"]


def test_book_heat_error_reads_fail_and_reasons_listed():
    rows = [_approval(i, book_heat={"read": "error: OperationalError"}) for i in range(2)] + [_approval(9)]
    c = _by(bv.check_approvals(rows))["book_heat"]
    assert c["status"] == "FAIL"
    text = bv.readout([c], SINCE, "live", "abc123", NOW)
    assert "error reasons: error: OperationalError 2 (100%)" in text


# ------------------------------------------------------------ TRIM_EXIT_RULE

def test_trim_exit_fields():
    ok = {"id": 1, "ticker": "AMD", "decision": {"held_trend": {"read": "ok"},
                                                  "held_exit_candidates": None,
                                                  "reads": {"held_trend": "ok", "held_exit": "ok"}}}
    miss = {"id": 2, "ticker": "AMD", "decision": {"held_trend": None, "reads": {}}}
    checks = _by(bv.check_trim_exit([ok]))
    assert checks["held_trend"]["status"] == checks["held_exit_candidates"]["status"] == "OK"
    checks = _by(bv.check_trim_exit([ok, miss]))
    assert checks["held_trend"]["status"] == "OK"                        # key present, value None
    assert checks["held_exit_candidates"]["status"] == "FAIL"
    assert _by(bv.check_trim_exit([]))["held_trend"]["status"] == "WARN"


# ------------------------------------------------------------ activity rows

def _act(i, stage, **extra):
    return {"id": i, "stage": stage, "ticker": "AMD", "extra": extra}


def test_activity_rows():
    rows = [_act(1, "add_lot_r", read="ok", path="whole_position", max_lot_r=1.1, lots=[]),
            _act(2, "broker_sell_seen", env="live", first_exec_at="2026-11-24T15:00:00+00:00",
                 seen_lag_s=90.0, held_read="snapshot", exec_time_source="execution"),
            _act(3, "broker_sell_seen", env="paper", first_exec_at=None),          # other env
            _act(4, "entry_gate_split_fail_open", entry_gate={"trend": {"read": "ok"}}, action="OPEN"),
            _act(5, "price_cache_rescale", applied=False, k=0.25, factor="4:1")]
    checks = _by(bv.check_activity(rows, "live"))
    for stage in ("add_lot_r", "broker_sell_seen", "entry_gate_split_fail_open", "price_cache_rescale"):
        assert checks[stage]["status"] == "OK", stage
    assert checks["broker_sell_seen"]["rows"] == 1
    assert checks["price_cache_rescale"]["reads"]["applied"] == {"false": 1}
    assert checks["price_cache_scale_gap"]["status"] == "WARN"
    assert "16:45" in checks["price_cache_scale_gap"]["notes"][0]


def test_activity_missing_key_and_error_reads_fail():
    checks = _by(bv.check_activity([_act(1, "add_lot_r", read="ok", path="stale_record")], "live"))
    assert checks["add_lot_r"]["status"] == "FAIL"
    rows = [_act(i, "add_lot_r", read="error: TimeoutError", path="whole_position",
                 max_lot_r=None, lots=[]) for i in range(3)]
    assert _by(bv.check_activity(rows, "live"))["add_lot_r"]["status"] == "FAIL"


def test_rescale_failed_rows_count_as_error_reads():
    rows = [_act(1, "price_cache_rescale", applied=True, k=0.5, factor="2:1"),
            _act(2, "price_cache_rescale_failed", k=0.5, factor="2:1", action="rescale")]
    c = _by(bv.check_activity(rows, "live"))
    assert c["price_cache_rescale"]["status"] == "FAIL"                  # 1 of 2 = 50% errors
    assert c["price_cache_rescale_failed"]["rows"] == 1


def test_unmatched_broker_exits():
    seen = [_act(1, "broker_sell_seen", env="live", first_exec_at="2026-11-24T15:00:00+00:00")]
    seen[0]["ticker"] = "MSFT"
    exits = [("MSFT", datetime(2026, 11, 24, 15, 10, tzinfo=timezone.utc)),
             ("MSFT", datetime(2026, 11, 24, 15, 10, tzinfo=timezone.utc)),          # second lot
             ("AMD", datetime(2026, 11, 24, 16, 0, tzinfo=timezone.utc))]
    out = bv.unmatched_broker_exits(exits, seen)
    assert [u["ticker"] for u in out] == ["AMD"]


# ------------------------------------------------------------ watcher logs

def _line(ts, outcome="succeeded", task="tasks.watch_broker_sells"):
    if outcome == "discarded":
        return f"[{ts},101: INFO/MainProcess] Discarding revoked task: {task}[ab-12]\n"
    tail = " in 0.21s: []" if outcome == "succeeded" else ": TimeoutError()"
    return f"[{ts},101: INFO/ForkPoolWorker-2] Task {task}[ab-12] {outcome}{tail}\n"


def _full_morning(extra=()):
    lines = [_line(f"2026-11-24 {h:02d}:{m:02d}:00") for h in range(9, 12) for m in (31, 45)]
    return lines + list(extra)


def test_watcher_counts_per_hour_and_ignores_other_tasks_and_old_lines():
    lines = _full_morning([_line("2026-11-24 10:05:00", "raised"),
                           _line("2026-11-24 10:07:00", "discarded"),
                           _line("2026-11-24 10:09:00", task="tasks.reconcile_stop_orders"),
                           _line("2026-11-23 15:00:00"),                                 # before since
                           "[2026-11-24 10:09:00,1: INFO/MainProcess] Task "
                           "tasks.watch_broker_sells[ab] received\n"])
    hours = bv.watcher_runs(lines, SINCE)
    ten = hours[datetime(2026, 11, 24, 10, tzinfo=NY)]
    assert ten == {"succeeded": 2, "raised": 1, "discarded": 1}
    c = bv.check_watcher(hours, SINCE, NOW)
    assert c["status"] == "OK" and c["rows"] == 7
    assert c["reads"]["outcome"]["error: raised"] == 1


def test_watcher_gap_in_an_elapsed_market_hour_fails():
    lines = [l for l in _full_morning() if " 10:" not in l]
    c = bv.check_watcher(bv.watcher_runs(lines, SINCE), SINCE, NOW)
    assert c["status"] == "FAIL" and "10:00-11:00" in c["why"]


def test_watcher_no_runs_after_the_open_fails_before_it_warns():
    assert bv.check_watcher({}, SINCE, NOW)["status"] == "FAIL"
    early = datetime(2026, 11, 24, 9, 50, tzinfo=NY)
    assert bv.check_watcher({}, SINCE, early)["status"] == "WARN"


def test_watcher_raised_over_20_percent_fails():
    lines = _full_morning([_line(f"2026-11-24 11:{m:02d}:00", "raised") for m in range(10, 13)])
    c = bv.check_watcher(bv.watcher_runs(lines, SINCE), SINCE, NOW)
    assert c["status"] == "FAIL" and "outcome" in c["why"]                # 3 of 9 = 33%


def test_log_lines_read_only_dot_log_files(tmp_path, monkeypatch):
    (tmp_path / "celery_worker.log").write_text(_line("2026-11-24 10:00:00"))
    (tmp_path / "db_write_spill.jsonl").write_text("tasks.watch_broker_sells secret")
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: opened.append(p) or real_open(p, *a, **k))
    assert len(list(bv._log_lines(str(tmp_path)))) == 1
    assert not any(str(p).endswith(".jsonl") for p in opened)


# ------------------------------------------------------------ the rest

def test_parse_since():
    assert bv.parse_since(None, datetime(2026, 11, 24, 17, 0, tzinfo=timezone.utc)) == SINCE
    assert bv.parse_since("2026-11-24", NOW) == datetime(2026, 11, 24, tzinfo=NY)
    assert bv.parse_since("2026-11-24 05:00", NOW) == SINCE


def test_run_checks_and_readout_cover_every_record():
    checks, unmatched = bv.run_checks([], [], [], [], [], [], {}, SINCE, NOW, "live")
    names = [c["name"] for c in checks]
    assert names == ["entry_gate", "story_dedup", "source_mix", "rr_vs_stop", "vol_target_shadow", "stop_vol",
                     "book_heat", "recheck_contradiction", "same_day_consensus", "add_loser_gate", "held_trend", "held_exit_candidates", "add_lot_r",
                     "broker_sell_seen", "entry_gate_split_fail_open", "price_cache_rescale",
                     "price_cache_scale_gap", "price_cache_rescale_failed", "add_stop_qty",
                     "add_stale_record", "stop_above_market_guard", "add_loser_gate_shadow",
                     "plan_compiled_hit", "pt_stop_qty", "late_fill_expect",
                     "late_fill_held_before", "entry_basis", "entry_ref_age",
                     "stop_slippage_partial", "book_gap_risk", "book_deployment", "order_quote", "verdict_vote_shadow",
                     "risk_subcheck_fail", "hwm_guard", "fundamentals_changes", "dcf_branch", "feed_health",
                     "scan_tier_funnel", "source_yield",
                     "earnings_event_record", "read_through", "earnings_claim", "catalyst_kind",
                     "target_anchor", "news_timeline", "plan_ledger_record", "own_ticker_rejected",
                     "entry_funnel",
                     "watch_broker_sells"]
    text = bv.readout(checks, SINCE, "live", "abc1234 on main (x)", NOW, unmatched)
    assert "abc1234" in text and "$" not in text
    assert all(c["status"] in ("OK", "WARN", "FAIL") for c in checks)


def test_fetch_only_reads():
    src = inspect.getsource(bv._fetch)
    assert "SET TRANSACTION READ ONLY" in src and "conn.rollback()" in src
    assert "commit" not in src
    for word in ("INSERT", "UPDATE ", "DELETE", "TRUNCATE", "ALTER", "DROP"):
        assert word not in src.upper().replace("UPDATE_", ""), word
    whole = open(_PATH).read()
    for call in ("set_state(", "log_activity", "log_signal(", "placeOrder", "cancelOrder",
                 ".commit(", "send_task(", "apply_async(", ".delay("):
        assert call not in whole, call


def test_recheck_contradiction_read_error_is_counted():
    rows = [_approval(1), _approval(2, recheck_contradiction={"read": "error: RuntimeError"})]
    c = _by(bv.check_approvals(rows))["recheck_contradiction"]
    assert c["rows"] == 2 and c["reads"]["read"] == {"ok": 1, "error: RuntimeError": 1}


# ------------------------------------------------------------ added records

_NEW_ROWS = {
    "add_stop_qty": dict(source="broker", broker_read="ok", used_qty=2.0, undersize_flags=[]),
    "add_stale_record": dict(record="rewritten_at_new_stop", new_stop_vs_prior_stop_pct=-1.2,
                             reclassified_open=False),
    "stop_above_market_guard": dict(mode="shadow", path="add_merge",
                                    outcome="would_placed_below_market", resting_read="ok"),
    "add_loser_gate_shadow": dict(add_loser_gate={"mode": "shadow", "avg_cost_read": "ok",
                                                  "would_block": False, "blocked": False},
                                  path="deferred_retry", approval_id=7),
    "plan_compiled_hit": dict(env="live", plan_id=3, newly_crossed=["t1"], levels={}, held=True,
                              levels_source="compiled"),
    "pt_stop_qty": dict(source="broker", broker_read="ok", used_qty=1.0, capped=False,
                        below_expected=False),
    "late_fill_expect": dict(mode="shadow", path="reconcile_late_buy", source="broker_after_fresh",
                             fresh_read="ok", broker_read="ok", expect_at_least=144.0,
                             would_defer=True),
    "late_fill_held_before": dict(order_id=5, stored_held_before=10.0, held_read="ok",
                                  action_today="OPEN", action_would="ADD", mode="shadow",
                                  applied=False),
    "entry_ref_age": dict(env="live", submitted_at="2026-11-20T15:00:30+00:00",
                          sizing_bar_time="2026-11-20T14:58:00+00:00",
                          submit_bar_time="2026-11-20T14:59:00+00:00", ref_age_s=150.0,
                          submit_ref_age_s=90.0, bid=99.9, ask=100.1, md_type_label="delayed",
                          approval_id=171, action="OPEN", order_type="MARKET",
                          submit_attempted=True, outcome_reason=None, ref_error=None,
                          sizing_price_source="latest_quote"),
    "stop_slippage_partial": dict(env="live", stop_price=146.24, fill_price=152.4,
                                  slippage_pct=4.2, shares=1.0,
                                  fill_time="2026-11-12T14:30:11+00:00",
                                  fill_match="stop_order_perm_id", perm_id=9001,
                                  booked_qty_for_perm=1.0,
                                  fill_price_basis="order_day_vwap_at_booking",
                                  shadow_key="live:NVDA:partial:2026-11-12:9001:1"),
    "book_gap_risk": dict(env="live", read="ok", session_date="2026-11-20", window_gaps=250,
                          definitions={"gap": "open/prior close - 1"}, positions=[],
                          invested_pct=18.9, loss_at_stops_pct_nav=1.66, n_common_nights=250,
                          book_gap_p1_pct_nav=-0.82, book_gap_worst_pct_nav=-1.24,
                          close_dates=["2026-11-20"], close_is_today=True,
                          shadow_key="live:2026-11-20", n_suspect_gaps=0, suspect_names=[],
                          suspect_nights=[], n_common_nights_ex_suspect=250,
                          book_gap_p1_pct_nav_ex_suspect=-0.82,
                          book_gap_worst_pct_nav_ex_suspect=-1.24),
    "book_deployment": dict(env="live", read="ok", session_date="2026-11-20",
                            source="portfolio_snapshots", source_as_of="2026-11-20T22:25:00+00:00",
                            invested_pct=19.0, cash_pct=81.0, other_pct=0.0, cash_target_pct=20.0,
                            gap_pct=61.0, regime="Neutral", n_positions=3,
                            sector_weights_pct={"AI Infrastructure — Compute / Accelerators": 12.0},
                            largest_name_weight_pct=8.0, watchlist_read="ok",
                            definitions={"units": "percent"}, shadow_key="live:2026-11-20"),
    "entry_basis": dict(env="live", read="ok", broker_avg=105.0, fill_avg=100.0,
                        basis_gap_frac=0.05, qty_coverage=1.0, trail_arm_vs_fill=0.365,
                        breakeven_vs_fill=0.05, record_written=True, stop_outcome="placed",
                        truncated=False, approval_id=171, legacy_tables_apply=True),
}
_MODES = {"stop_above_market_guard": "shadow", "late_fill_expect": "shadow",
          "late_fill_held_before": "shadow"}


def test_added_activity_stages_ok_and_warn_when_absent():
    rows = [_act(i, st, **ex) for i, (st, ex) in enumerate(_NEW_ROWS.items())]
    checks = _by(bv.check_activity(rows, "live", _MODES))
    for st in _NEW_ROWS:
        assert checks[st]["status"] == "OK" and checks[st]["rows"] == 1, st
    assert checks["late_fill_expect"]["reads"]["would_defer"] == {"true": 1}
    assert checks["stop_above_market_guard"]["reads"]["mode"] == {"shadow": 1}
    empty = _by(bv.check_activity([], "live", _MODES))
    assert all(empty[st]["status"] == "WARN" for st in _NEW_ROWS)
    assert "lands with" in " ".join(empty["pt_stop_qty"]["notes"])
    assert "lands with" in " ".join(empty["late_fill_held_before"]["notes"])


@pytest.mark.parametrize("stage", list(_NEW_ROWS))
def test_added_activity_stage_missing_key_fails(stage):
    required = bv._ACTIVITY_SPECS[stage][0]
    extra = {k: v for k, v in _NEW_ROWS[stage].items() if k != required[0]}
    c = _by(bv.check_activity([_act(1, stage, **extra)], "live", _MODES))[stage]
    assert c["status"] == "FAIL" and c["missing"] == [1]


@pytest.mark.parametrize("stage", list(bv.MODED_STAGES))
def test_row_in_another_mode_than_config_fails(stage):
    row = _act(1, stage, **{**_NEW_ROWS[stage], "mode": "enforce"})
    c = _by(bv.check_activity([row], "live", _MODES))[stage]
    assert c["status"] == "FAIL" and "config" in c["why"]
    # no expected mode given: not checked
    assert _by(bv.check_activity([row], "live"))[stage]["status"] == "OK"


def test_broker_read_errors_over_20_percent_fail_the_stop_count_rows():
    rows = [_act(i, "add_stop_qty", **{**_NEW_ROWS["add_stop_qty"], "broker_read": "error: TimeoutError",
                                         "source": "position_reads"}) for i in range(2)]
    rows.append(_act(9, "add_stop_qty", **_NEW_ROWS["add_stop_qty"]))
    assert _by(bv.check_activity(rows, "live"))["add_stop_qty"]["status"] == "FAIL"


def test_add_loser_shadow_row_that_blocked_fails():
    ex = {**_NEW_ROWS["add_loser_gate_shadow"],
          "add_loser_gate": {"mode": "shadow", "avg_cost_read": "ok", "would_block": True,
                             "blocked": True}}
    c = _by(bv.check_activity([_act(1, "add_loser_gate_shadow", **ex)], "live"))
    assert c["add_loser_gate_shadow"]["status"] == "FAIL"


def test_plan_compiled_hit_other_env_ignored():
    row = _act(1, "plan_compiled_hit", **{**_NEW_ROWS["plan_compiled_hit"], "env": "paper"})
    assert _by(bv.check_activity([row], "live"))["plan_compiled_hit"]["rows"] == 0


def test_add_loser_on_add_approvals_in_shadow_only():
    gate = {"mode": "shadow", "avg_cost_read": "ok", "trigger_new_fact_source": "event_provenance",
            "would_block": True, "blocked": False}
    add_ok = _approval(1, action="ADD", add_loser_gate=gate)
    add_missing = _approval(2, action="ADD")
    open_row = _approval(3)                                   # an OPEN never carries it
    c = bv.check_add_loser([add_ok, open_row])
    assert c["status"] == "OK" and c["rows"] == 1 and c["reads"]["would_block"] == {"true": 1}
    c = bv.check_add_loser([add_ok, add_missing])
    assert c["status"] == "FAIL" and c["missing"] == [2]
    for mode in ("enforce", "off"):
        c = bv.check_add_loser([add_missing], mode)
        assert c["status"] == "WARN" and mode in c["notes"][0]
    blocked = _approval(4, action="ADD", add_loser_gate={**gate, "blocked": True})
    assert bv.check_add_loser([blocked])["status"] == "FAIL"


def test_entry_funnel_statuses():
    assert bv.check_entry_funnel(None)["status"] == "WARN"
    ok = bv.check_entry_funnel({"line": None, "verdicts": 6, "passed": 2,
                                "sessions": ["2026-11-23", "2026-11-24"]})
    assert ok["status"] == "OK" and "6 OPEN/ADD verdicts, 2 passed" in ok["notes"][0]
    few = bv.check_entry_funnel({"line": None, "verdicts": 1, "passed": 0, "sessions": []})
    assert few["status"] == "WARN"
    fired = bv.check_entry_funnel({"line": "ENTRY FUNNEL: 0 passed buys", "verdicts": 5,
                                   "passed": 0, "sessions": []})
    assert fired["status"] == "FAIL" and "0 passed buys" in fired["why"]
    err = bv.check_entry_funnel({"error": "OperationalError"})
    assert err["status"] == "FAIL" and "OperationalError" in err["why"]


def test_entry_funnel_result_uses_the_funnel_check_and_never_raises(monkeypatch):
    from datetime import date

    class FakeFunnel:
        CHECK_SESSIONS = 2

        @staticmethod
        def default_until(now):
            return date(2026, 11, 24)

        @staticmethod
        def sessions_until(until, n):
            return [date(2026, 11, 23), date(2026, 11, 24)][-n:]

        @staticmethod
        def run(sessions, env):
            assert env == "live"
            return {s: {"verdicts": {"OPEN": 2}, "passed": 0} for s in sessions}

        @staticmethod
        def funnel_check(funnel, sessions):
            return "ENTRY FUNNEL: 0 passed buys"
    monkeypatch.setattr(bv, "_entry_funnel_module", lambda: FakeFunnel)
    out = bv.entry_funnel_result(NOW, "live")
    assert out["line"].startswith("ENTRY FUNNEL") and out["verdicts"] == 4 and out["passed"] == 0

    def boom():
        raise RuntimeError("db")
    monkeypatch.setattr(bv, "_entry_funnel_module", boom)
    assert bv.entry_funnel_result(NOW, "live") == {"error": "RuntimeError"}


def test_entry_funnel_module_loads_the_real_script():
    ef = bv._entry_funnel_module()
    assert callable(ef.funnel_check) and ef.CHECK_SESSIONS >= 1


def test_config_modes(monkeypatch):
    import config
    monkeypatch.setattr(config, "STOP_GUARD_MODE", "Shadow", raising=False)
    monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "typo", raising=False)
    monkeypatch.delattr(config, "LATE_FILL_HELD_BEFORE_MODE", raising=False)
    modes, loser = bv.config_modes()
    assert modes["stop_above_market_guard"] == "shadow"
    assert modes["late_fill_held_before"] is None            # not in this clone: not checked
    assert loser == "shadow"                                  # crew reads a typo as shadow


def test_risk_subcheck_fail_is_ok_when_absent_and_warn_on_any_row():
    ok = bv.check_risk_subcheck_fail([{"id": 1, "stage": "add_lot_r", "extra": {}}], "live")
    assert ok["status"] == "OK" and ok["rows"] == 0
    rows = [{"id": 2, "stage": "risk_subcheck_fail", "ticker": "NVDA",
             "extra": {"check": "theme", "error": "RuntimeError", "mode": "shadow", "applied": False}},
            {"id": 3, "stage": "risk_subcheck_fail", "ticker": "IONQ",
             "extra": {"check": "moonshot_total", "error": "KeyError", "mode": "shadow", "applied": False}},
            {"id": 4, "stage": "risk_subcheck_fail", "ticker": "AMD",
             "extra": {"check": "theme", "mode": "shadow", "env": "paper"}}]      # another env
    chk = bv.check_risk_subcheck_fail(rows, "live")
    assert chk["status"] == "WARN" and chk["rows"] == 2
    assert chk["reads"]["check"] == {"theme": 1, "moonshot_total": 1}
    assert chk["reads"]["applied"] == {"false": 2}
    text = bv.readout([chk], SINCE, "live", "abc1234 on main (x)", NOW, [])
    assert "risk_subcheck_fail" in text and "WARN" in text


def test_fetch_reads_the_alarm_stage():
    src = inspect.getsource(bv._fetch)
    assert "ALARM_STAGES" in src and "risk_subcheck_fail" in bv.ALARM_STAGES
    assert not set(bv.ALARM_STAGES) & set(bv.ACTIVITY_STAGES)




# ------------------------------------------------------------ entry_basis, verdict_vote_shadow

def test_entry_basis_warns_when_absent_and_names_the_add_count():
    none, _ = bv.run_checks([], [], [], [], [], [], {}, SINCE, NOW, "live")
    c = _by(none)["entry_basis"]
    assert c["status"] == "WARN" and c["why"] == "no ADD in the window"
    adds = [_approval(1, action="ADD"), _approval(2, action="ADD", is_test=True), _approval(3)]
    some, _ = bv.run_checks([], [], [], adds, [], [], {}, SINCE, NOW, "live")
    c = _by(some)["entry_basis"]
    assert c["status"] == "WARN" and c["why"].startswith("1 ADD approval(s) but no row")
    assert "1 ADD approval(s) in the window" in c["notes"][0]
    row = _act(9, "entry_basis", **_NEW_ROWS["entry_basis"])
    ok, _ = bv.run_checks([], [], [], adds, [row], [], {}, SINCE, NOW, "live")
    c = _by(ok)["entry_basis"]
    assert c["status"] == "OK" and c["reads"]["record_written"] == {"true": 1}
    assert c["reads"]["truncated"] == {"false": 1}


def test_entry_basis_error_reads_and_other_env():
    rows = [_act(i, "entry_basis", **{**_NEW_ROWS["entry_basis"], "read": "error: RuntimeError"})
            for i in range(3)] + [_act(9, "entry_basis", **_NEW_ROWS["entry_basis"])]
    c = _by(bv.check_activity(rows, "live", _MODES))["entry_basis"]
    assert c["status"] == "FAIL" and "3 of 4 reads are errors" in c["why"]
    paper = _act(5, "entry_basis", **{**_NEW_ROWS["entry_basis"], "env": "paper"})
    assert _by(bv.check_activity([paper], "live", _MODES))["entry_basis"]["rows"] == 0


def _vote(i, **over):
    extra = dict(env="live", signal_id=41, status="ok", draws=[{}, {}, {}],
                 actions=["OPEN", "OPEN", "WATCH"], majority_action="OPEN",
                 model_pass_majority=True, prompt_sha="abc123def456", options_match=True,
                 busy_tries=0)
    extra.update(over)
    return {"id": i, "stage": "verdict_vote_shadow", "ticker": "NVDA", "extra": extra}


def test_verdict_vote_off_ok_without_rows_fail_with_any():
    c = bv.check_verdict_vote([_act(1, "add_lot_r")], "live", False)
    assert c["status"] == "OK" and c["rows"] == 0
    c = bv.check_verdict_vote([_vote(1)], "live", False)
    assert c["status"] == "FAIL" and "VERDICT_VOTE_SHADOW_ENABLED is False" in c["why"]
    other = bv.check_verdict_vote([_vote(1, env="paper")], "live", False)
    assert other["status"] == "OK"


def test_verdict_vote_on():
    assert bv.check_verdict_vote([], "live", True)["status"] == "WARN"
    c = bv.check_verdict_vote([_vote(1), _vote(2, status="gave_up_busy")], "live", True)
    assert c["status"] == "OK" and c["reads"]["status"] == {"ok": 1, "gave_up_busy": 1}
    missing = _vote(3)
    del missing["extra"]["prompt_sha"]
    c = bv.check_verdict_vote([missing], "live", True)
    assert c["status"] == "FAIL" and c["missing"] == [3]
    c = bv.check_verdict_vote([_vote(4, status="error"), _vote(5)], "live", True)
    assert c["status"] == "FAIL" and "status: 1 of 2 reads are errors" in c["why"]


def test_vote_switch_is_read_from_config_and_passed(monkeypatch):
    import config
    monkeypatch.setattr(config, "VERDICT_VOTE_SHADOW_ENABLED", True, raising=False)
    assert bv.vote_enabled() is True
    monkeypatch.delattr(config, "VERDICT_VOTE_SHADOW_ENABLED", raising=False)
    assert bv.vote_enabled() is False
    checks, _ = bv.run_checks([], [], [], [], [_vote(1)], [], {}, SINCE, NOW, "live", vote_on=True)
    assert _by(checks)["verdict_vote_shadow"]["status"] == "OK"
    assert "vote_on=vote_enabled()" in inspect.getsource(bv.main)


def test_fetch_reads_the_new_stages():
    src = inspect.getsource(bv._fetch)
    assert "VOTE_STAGE" in src and "entry_basis" in bv.ACTIVITY_STAGES
    assert bv.VOTE_STAGE not in bv.ACTIVITY_STAGES


# ------------------------------------------------------------ hwm_guard, fundamentals_changes, source_yield

def _writer_row(stage, call):
    """The (stage, extra) the real writer passes to the activity log."""
    for c in call.call_args_list:
        if c.args[0] == stage:
            return c.args[4]
    raise AssertionError(f"no {stage} row written")


def _hwm_guard_extra(hwm="0"):
    from unittest.mock import MagicMock, patch
    import config
    from risk_manager import RiskManager
    import risk_manager
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1000.0)
    rm = RiskManager(client, start_value=1000.0)
    env = risk_manager.TRADING_ENV
    with patch.object(config, "HWM_GUARD_MODE", "shadow"), \
         patch("db.get_state_strict", side_effect=lambda k, d=None: {
             f"high_water_mark_{env}": hwm}.get(k, d)), \
         patch("db.get_state", return_value=None), patch("db.set_state"), \
         patch("db.set_state_verified", return_value=["x"]), \
         patch("db.log_activity_queued") as row:
        rm.is_halted()
    return _writer_row("hwm_guard", row), env


def test_hwm_guard_ok_when_absent_warn_on_rows_fail_on_a_missing_key():
    assert bv.check_hwm_guard([_act(1, "add_lot_r")], "live")["status"] == "OK"
    extra, env = _hwm_guard_extra("0")
    assert set(bv.HWM_GUARD_REQUIRED) <= set(extra)          # the writer's row passes
    other = dict(extra, reason="non_finite")
    rows = [_act(2, "hwm_guard", **extra), _act(3, "hwm_guard", **other),
            _act(4, "hwm_guard", **dict(extra, env="elsewhere"))]          # another env
    chk = bv.check_hwm_guard(rows, env)
    assert chk["status"] == "WARN" and chk["rows"] == 2
    assert "non_finite" in chk["why"] and "non_positive" in chk["why"]
    assert chk["reads"]["mode"] == {"shadow": 2}
    for key in bv.HWM_GUARD_REQUIRED:
        bad = {k: v for k, v in extra.items() if k != key}
        c = bv.check_hwm_guard([_act(5, "hwm_guard", **bad)], env)
        assert c["status"] == "FAIL" and c["missing"] == [5], key
    assert "$" not in bv.readout([chk], SINCE, env, "abc1234 on main (x)", NOW, [])


def _fundamentals_extra(changes):
    from unittest.mock import patch
    import tasks
    with patch("db.count_fundamentals_changes", return_value=changes), \
         patch("db.get_state", return_value=None), patch("db.set_state"), \
         patch("alerts._send", return_value=None), patch("db.log_activity_direct") as row:
        tasks._fundamentals_change_record(datetime(2026, 11, 24, 22, 0, tzinfo=timezone.utc))
    return _writer_row("fundamentals_changes", row)


def test_fundamentals_changes_rule():
    import db
    env = db.TRADING_ENV
    tue_eve = datetime(2026, 11, 24, 18, 30, tzinfo=NY)
    # Before the 17:00 refresh is due: none due, no row is OK.
    ok = bv.check_fundamentals_changes([], env, SINCE, NOW)
    assert ok["status"] == "OK" and "due in the window: 0" in " ".join(ok["notes"])
    # After 18:00 on a session day with no row: WARN.
    warn = bv.check_fundamentals_changes([], env, SINCE, tue_eve)
    assert warn["status"] == "WARN" and "1 session-day refresh" in warn["why"]
    # The writer's row passes; flagged is tallied.
    extra = _fundamentals_extra({"tickers": 75, "changed": 0, "no_prior": 0})
    assert set(bv.FUNDAMENTALS_REQUIRED) <= set(extra)
    flagged = dict(extra, flagged=True)
    c = bv.check_fundamentals_changes([_act(1, "fundamentals_changes", **extra),
                                       _act(2, "fundamentals_changes", **flagged)],
                                      env, SINCE, tue_eve)
    assert c["status"] == "OK" and c["reads"]["flagged"] == {"false": 1, "true": 1}
    assert "1 flagged row" in " ".join(c["notes"])
    bad = {k: v for k, v in extra.items() if k != "judged"}
    c = bv.check_fundamentals_changes([_act(3, "fundamentals_changes", **bad)], env, SINCE, tue_eve)
    assert c["status"] == "FAIL" and c["missing"] == [3]


def test_refresh_days_due_skips_weekends_holidays_and_a_refresh_before_since():
    thanksgiving = datetime(2026, 11, 26, 5, 0, tzinfo=NY)            # Thu, NYSE holiday
    assert bv._refresh_days_due(thanksgiving, datetime(2026, 11, 26, 23, 0, tzinfo=NY)) == []
    sat = datetime(2026, 11, 28, 5, 0, tzinfo=NY)
    assert bv._refresh_days_due(sat, datetime(2026, 11, 29, 23, 0, tzinfo=NY)) == []
    late = datetime(2026, 11, 24, 17, 30, tzinfo=NY)                  # since after the 17:00 start
    assert bv._refresh_days_due(late, datetime(2026, 11, 24, 23, 0, tzinfo=NY)) == []
    span = bv._refresh_days_due(SINCE, datetime(2026, 11, 27, 18, 0, tzinfo=NY))
    assert [d.day for d in span] == [24, 25, 27]


def _feed_health_extra(**over):
    extra = {"day": "2026-11-24", "env": "live", "computed_at": "2026-11-24T23:40:05+00:00",
             "flagged": [], "alerted": [], "excluded": ["congressional"], "sources": {},
             "params": {}}
    extra.update(over)
    return extra


def test_feed_health_rule():
    tue_1845 = datetime(2026, 11, 24, 18, 45, tzinfo=NY)
    tue_eve = datetime(2026, 11, 24, 19, 0, tzinfo=NY)
    # Before 18:50 on the first session day: none due, no row is OK.
    for now in (NOW, tue_1845):
        ok = bv.check_feed_health([], "live", SINCE, now)
        assert ok["status"] == "OK", now
        assert "due in the window: 0" in " ".join(ok["notes"])
    # After 18:50 on a session day with no row: WARN.
    warn = bv.check_feed_health([], "live", SINCE, tue_eve)
    assert warn["status"] == "WARN"
    assert "1 session-day run(s) due" in warn["why"]
    # A full row is OK; flagged sources are tallied.
    c = bv.check_feed_health([_act(1, "feed_health", **_feed_health_extra()),
                              _act(2, "feed_health", **_feed_health_extra(
                                  flagged=["analyst_ratings"]))], "live", SINCE, tue_eve)
    assert c["status"] == "OK"
    assert c["reads"]["any_flagged"] == {"false": 1, "true": 1}
    assert "analyst_ratings x1" in " ".join(c["notes"])
    # A missing key is FAIL.
    for key in bv.FEED_HEALTH_REQUIRED:
        bad = {k: v for k, v in _feed_health_extra().items() if k != key}
        c = bv.check_feed_health([_act(3, "feed_health", **bad)], "live", SINCE, tue_eve)
        assert c["status"] == "FAIL", key
        assert c["missing"] == [3], key
    # Another env's row is not counted.
    other = _act(4, "feed_health", **_feed_health_extra(env="paper"))
    assert bv.check_feed_health([other], "live", SINCE, tue_eve)["rows"] == 0


def test_feed_health_notes_a_due_day_without_its_row():
    wed_eve = datetime(2026, 11, 25, 19, 0, tzinfo=NY)
    c = bv.check_feed_health([_act(1, "feed_health", **_feed_health_extra())], "live", SINCE,
                             wed_eve)
    assert c["status"] == "OK"
    assert "due day(s) with no row: 2026-11-25" in " ".join(c["notes"])


def test_feed_health_skips_the_thanksgiving_holiday():
    thu = datetime(2026, 11, 26, 5, 0, tzinfo=NY)
    c = bv.check_feed_health([], "live", thu, datetime(2026, 11, 26, 23, 0, tzinfo=NY))
    assert c["status"] == "OK"


def test_feed_health_switch_off():
    tue_eve = datetime(2026, 11, 24, 19, 0, tzinfo=NY)
    off = bv.check_feed_health([], "live", SINCE, tue_eve, enabled=False)
    assert off["status"] == "OK"
    row = _act(1, "feed_health", **_feed_health_extra())
    bad = bv.check_feed_health([row], "live", SINCE, tue_eve, enabled=False)
    assert bad["status"] == "FAIL"
    checks, _ = bv.run_checks([], [], [], [], [row], [], {}, SINCE, tue_eve, "live",
                              feed_health_on=False)
    assert _by(checks)["feed_health"]["status"] == "FAIL"
    assert "feed_health_on=feed_health_enabled()" in inspect.getsource(bv.main)


def test_feed_health_writer_row_passes():
    from datetime import date, timedelta
    from unittest.mock import MagicMock, patch
    import tasks
    from market_calendar import NYSE_HOLIDAYS
    days, d = [], date(2026, 11, 24)
    while len(days) < 25:
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            days.append(d)
        d -= timedelta(days=1)
    db = MagicMock()
    db.TRADING_ENV = "live"
    db.get_seen_item_counts.return_value = [{"source": src, "day": x, "n": 30}
                                            for x in days for src in ("news", "edgar")]
    db.get_state.return_value = None

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 24, 18, 40, tzinfo=tz)
    with patch.dict("sys.modules", {"db": db}), patch("datetime.datetime", _DT), \
         patch("alerts._send", return_value=None):
        tasks.feed_health.run()
    extra = _writer_row("feed_health", db.log_activity_direct)
    assert set(bv.FEED_HEALTH_REQUIRED) <= set(extra)
    c = bv.check_feed_health([_act(1, "feed_health", **extra)], "live", SINCE,
                             datetime(2026, 11, 24, 19, 0, tzinfo=NY))
    assert c["status"] == "OK"
    assert c["rows"] == 1


def _yield_row(i, at, **over):
    extra = {"day": "2026-11-23", "shadow_key": "live:2026-11-23", "sources": {}, "drops": [],
             "computed_at": at.isoformat(), "env": "live", "feeds": {"news": {}},
             "feed_drops": []}
    extra.update(over)
    return _act(i, "source_yield", **{k: v for k, v in extra.items() if v is not _DROP})


_DROP = object()


def test_source_yield_feed_keys():
    at = datetime(2026, 11, 24, 18, 20, tzinfo=NY)
    assert bv.check_source_yield_feeds([], "live", SINCE)["status"] == "WARN"
    ok = bv.check_source_yield_feeds([_yield_row(1, at), _yield_row(2, at, feeds=None,
                                                                    feed_drops=None)],
                                     "live", SINCE)
    assert ok["status"] == "OK" and ok["reads"]["feeds"] == {"present": 1, "none": 1}
    drop = bv.check_source_yield_feeds([_yield_row(3, at, feed_drops=["news:yfinance"])],
                                       "live", SINCE)
    assert drop["status"] == "WARN" and "news:yfinance (1)" in drop["why"]
    for key in bv.FEED_REQUIRED:
        bad = bv.check_source_yield_feeds([_yield_row(4, at, **{key: _DROP})], "live", SINCE)
        assert bad["status"] == "FAIL" and bad["missing"] == [4], key
    # A replayed row computed before --since is left out, not a FAIL.
    old = _yield_row(5, datetime(2026, 11, 20, 18, 20, tzinfo=NY), feeds=_DROP, feed_drops=_DROP)
    c = bv.check_source_yield_feeds([old, _yield_row(6, at)], "live", SINCE)
    assert c["status"] == "OK" and c["rows"] == 1 and "replayed" in " ".join(c["notes"])
    assert bv.check_source_yield_feeds([_yield_row(7, at, env="paper")], "live",
                                       SINCE)["rows"] == 0


def test_source_yield_writer_row_passes():
    from datetime import date
    from unittest.mock import patch
    import tasks

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 24, 18, 20, tzinfo=NY)
    with patch("datetime.datetime", _Now), \
         patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_seen_item_counts", return_value=[{"source": "news",
                                                         "day": date(2026, 11, 23), "n": 5}]), \
         patch("db.get_news_feed_counts", return_value=None), \
         patch("db.TRADING_ENV", "live"), patch("db.log_activity_direct") as row:
        tasks.source_yield(days_back=1)
    extra = _writer_row("source_yield", row)
    c = bv.check_source_yield_feeds([_act(1, "source_yield", **extra)], "live", SINCE)
    assert c["status"] == "OK" and c["rows"] == 1


def test_fetch_reads_the_hwm_fundamentals_and_feed_stages():
    src = inspect.getsource(bv._fetch)
    assert "FUNDAMENTALS_STAGE" in src and "FEED_STAGE" in src
    assert "FEED_HEALTH_STAGE" in src
    assert "hwm_guard" in bv.ALARM_STAGES
    assert not {"hwm_guard", bv.FUNDAMENTALS_STAGE, bv.FEED_STAGE} & set(bv.ACTIVITY_STAGES)

def test_entry_ref_age_and_partial_stages():
    full = _act(1, "entry_ref_age", **_NEW_ROWS["entry_ref_age"])
    bad = _act(2, "entry_ref_age", **{**_NEW_ROWS["entry_ref_age"], "ref_error": "TypeError"})
    c = _by(bv.check_activity([full], "live", _MODES))["entry_ref_age"]
    assert c["status"] == "OK" and c["reads"]["ref"] == {"ok": 1}
    c = _by(bv.check_activity([full, bad, bad], "live", _MODES))["entry_ref_age"]
    assert c["status"] == "FAIL" and "2 of 3 reads are errors" in c["why"]
    none = _by(bv.check_activity([], "live", _MODES))
    assert none["entry_ref_age"]["status"] == "WARN"
    assert none["stop_slippage_partial"]["status"] == "WARN"
    part = _act(3, "stop_slippage_partial", **_NEW_ROWS["stop_slippage_partial"])
    assert _by(bv.check_activity([part], "live", _MODES))["stop_slippage_partial"]["status"] == "OK"
    del part["extra"]["perm_id"]
    c = _by(bv.check_activity([part], "live", _MODES))["stop_slippage_partial"]
    assert c["status"] == "FAIL" and c["missing"] == [3]

def test_book_gap_risk_stage_is_shape_only():
    row = _act(1, "book_gap_risk", **_NEW_ROWS["book_gap_risk"])
    assert _by(bv.check_activity([row], "live", _MODES))["book_gap_risk"]["status"] == "OK"
    assert _by(bv.check_activity([], "live", _MODES))["book_gap_risk"]["status"] == "WARN"
    err = _act(2, "book_gap_risk", **{**_NEW_ROWS["book_gap_risk"], "read": "error: RuntimeError"})
    c = _by(bv.check_activity([err], "live", _MODES))["book_gap_risk"]
    assert c["status"] == "FAIL" and "read: 1 of 1 reads are errors" in c["why"]
    # No threshold: a large gap loss is still OK.
    big = _act(3, "book_gap_risk", **{**_NEW_ROWS["book_gap_risk"], "book_gap_worst_pct_nav": -40.0})
    assert _by(bv.check_activity([big], "live", _MODES))["book_gap_risk"]["status"] == "OK"


def test_book_gap_risk_requires_the_corporate_action_guard_keys():
    for key in ("n_suspect_gaps", "suspect_names", "suspect_nights", "n_common_nights_ex_suspect",
                "book_gap_p1_pct_nav_ex_suspect", "book_gap_worst_pct_nav_ex_suspect"):
        fields = {k: v for k, v in _NEW_ROWS["book_gap_risk"].items() if k != key}
        c = _by(bv.check_activity([_act(1, "book_gap_risk", **fields)], "live", _MODES))["book_gap_risk"]
        assert c["status"] == "FAIL", key
        assert "lack the field" in c["why"], key
    full = _act(1, "book_gap_risk", **_NEW_ROWS["book_gap_risk"])
    assert _by(bv.check_activity([full], "live", _MODES))["book_gap_risk"]["status"] == "OK"
    # The writer's own fields carry every required key.
    import tasks
    rec = tasks._book_gap_risk({}, 1.0, {}, {})
    for key in bv._ACTIVITY_SPECS["book_gap_risk"][0]:
        if key not in ("session_date", "window_gaps", "definitions", "close_dates",
                       "close_is_today", "shadow_key"):
            assert key in rec, key
