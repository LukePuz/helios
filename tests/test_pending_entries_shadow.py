"""Log-only pending_entries shadow (audit 2026-11-02 19:27 #2, roadmap
01(c)): c>=6 WATCH and gated-down OPEN/ADD verdicts record their plan entry
levels; the daily-bar check logs would_fill. Nothing routes or trades."""
import json
from contextlib import ExitStack
from datetime import date
from unittest.mock import MagicMock, patch

import tasks
from agents import analyst

PARSED = {"valuation": {"bull_case_target": "$700", "downside_scenario": "$500"},
          "catalyst": {"event": "earnings", "date": "2099-11-20"}, "thesis_bucket": "Accelerator"}
BRIEF = "Current Price: $600.00"
LADDER = {"atr": 10.0, "rungs": [{"price": 560.0, "types": ["swing_low"]},
                                 {"price": 640.0, "types": ["swing_high"], "band_high": 645.0}]}


def _shadow(action="WATCH", conviction=6, model_action="WATCH", plan=None, ladder=LADDER,
            parsed=PARSED, brief=BRIEF):
    with patch("db.get_trade_ladder", return_value=ladder), \
         patch("db.get_active_plan", return_value=plan):
        return analyst._pending_entry_shadow("NVDA", action, conviction, model_action, parsed, brief)


def test_watch_c6_records_every_plan_level_and_the_expiry():
    s = _shadow(plan={"id": 7, "entry_price": 580.0})
    assert s["recorded"] and s["trigger"] == "watch_c6" and s["price"] == 600.0
    # plan_entry (580 x 1.05) is already met at the 600 verdict price (M208)
    assert s["levels"]["plan_entry"] == {"side": "at_or_below", "level": 609.0, "plan_id": 7,
                                         "price_field": "low", "in_zone_at_verdict": True}
    assert s["levels"]["price_for_2r"] == {"side": "at_or_below", "level": 566.67,
                                           "price_field": "low", "in_zone_at_verdict": False}
    assert s["levels"]["add_trigger"] == {"side": "at_or_above", "level": 645.0,
                                          "price_field": "close", "in_zone_at_verdict": False}
    assert s["expires"] == "2099-11-25"                     # catalyst + 5 d grace


def test_gated_down_open_qualifies_at_any_conviction():
    s = _shadow(conviction=5, model_action="OPEN")
    assert s["recorded"] and s["trigger"] == "gated_down" and s["model_action"] == "OPEN"


def test_non_qualifying_verdicts_record_none():
    assert _shadow(conviction=5) is None
    assert _shadow(action="OPEN", model_action="OPEN") is None
    assert _shadow(action="HOLD", model_action="HOLD", conviction=8) is None


def test_no_level_or_no_price_is_recorded_as_not_kept():
    s = _shadow(ladder=None, parsed={"catalyst": {}})
    assert s["recorded"] is False and s["why_not"] == "no entry level in the plans"
    s = _shadow(brief="no price here")
    assert s["recorded"] is False and s["why_not"] == "no price in the brief"


def test_swallows_its_own_errors():
    with patch("db.get_trade_ladder", side_effect=RuntimeError("db down")):
        s = analyst._pending_entry_shadow("NVDA", "WATCH", 7, "WATCH", PARSED, BRIEF)
    assert s["recorded"] is False and "db down" in s["error"]
    with patch("db.get_state", side_effect=RuntimeError("db down")), patch("db.set_state") as st:
        analyst._store_pending_entry("NVDA", {"recorded": True, "et_date": "2026-11-03"}, 1)
    st.assert_not_called()


def test_no_expiry_is_not_recorded():
    for missing in (None, "None"):
        with patch("agents.plans.compile_plan",
                   return_value={"time_stop": missing, "add_trigger": 645.0}):
            s = _shadow()
        assert s["recorded"] is False and s["why_not"] == "no_expiry" and s["expires"] is None


REC = {"recorded": True, "trigger": "watch_c6", "model_action": "WATCH", "conviction": 7,
       "price": 600.0, "et_date": "2026-11-03", "expires": "2026-11-25",
       "levels": {"price_for_2r": {"side": "at_or_below", "level": 566.67}}}


def _store(book, ticker="NVDA", rec=REC, sid=42):
    state = {"pending_entries_shadow_live": json.dumps(book)}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_activity", return_value=True) as act, \
         patch("config.TRADING_ENV", "live"):
        analyst._store_pending_entry(ticker, rec, sid)
    return json.loads(state["pending_entries_shadow_live"]), act


def test_store_keeps_one_per_ticker_and_prunes_only_after_the_grace():
    book, act = _store({
        "OLD": {"expires": "2026-10-19", "at": "a"},       # 15 d past expiry: pruned
        "RECENT": {"expires": "2026-11-01", "at": "a"},    # expired, inside the 14 d grace: kept
        "AMD": {"expires": "2026-12-01", "at": "b"}})
    assert sorted(book) == ["AMD", "NVDA", "RECENT"]
    assert book["NVDA"]["conviction"] == 7 and book["NVDA"]["signal_id"] == 42
    assert book["NVDA"]["levels"] == REC["levels"]
    act.assert_not_called()                              # nothing superseded or evicted
    state = {}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("config.TRADING_ENV", "live"):
        analyst._store_pending_entry("MSFT", {"recorded": False}, 43)    # not kept
    assert state == {}


def test_a_replaced_entry_writes_a_superseded_row():
    old = {"expires": "2026-12-01", "at": "c", "et_date": "2026-11-01", "signal_id": 7,
           "trigger": "gated_down", "levels": {"add_trigger": {"side": "at_or_above", "level": 1}}}
    book, act = _store({"NVDA": old})
    assert book["NVDA"]["signal_id"] == 42
    stage, ticker, _, sentiment, extra = act.call_args.args
    assert (stage, ticker, sentiment) == ("entry_shadow", "NVDA", "neutral")
    assert extra["superseded"] is True and extra["would_fill"] is None
    assert extra["verdict_signal_id"] == 7 and extra["replaced_by_signal_id"] == 42
    assert extra["kinds"] == ["add_trigger"] and extra["censored_et_date"] == "2026-11-03"
    assert extra["code_sha"] == "test" and "env_sha" in extra   # m268


def test_the_cap_evicts_the_oldest_with_an_evicted_row():
    full = {f"T{i:03d}": {"expires": "2026-12-01", "at": f"{i:03d}", "signal_id": i}
            for i in range(analyst._PENDING_ENTRIES_MAX)}
    book, act = _store(full)
    assert len(book) == analyst._PENDING_ENTRIES_MAX and "NVDA" in book and "T000" not in book
    assert act.call_count == 1
    _, ticker, _, _, extra = act.call_args.args
    assert ticker == "T000" and extra["evicted"] is True and extra["verdict_signal_id"] == 0


def test_entry_floor_matches_the_entry_gate():
    """_ENTRY_UDR_FLOOR is a copy of the OPEN/ADD gate's literal; read the
    gate from _analyze_inner()'s source so the two can't drift apart."""
    import inspect
    import re
    src = inspect.getsource(analyst._analyze_inner)
    m = re.findall(r'if udr < ([\d.]+) and action in \("OPEN", "ADD"\):', src)
    assert len(m) == 1 and float(m[0]) == analyst._ENTRY_UDR_FLOOR


def _analyze(model_action, udr, conviction=7, state=None):
    parsed = {
        "action": model_action, "conviction": conviction, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {"bull_case_target": "$700", "downside_scenario": "$500"},
        "expected_return_pct": 8.0, "upside_downside_ratio": udr, "key_risks": [],
        "pass_to_risk": model_action == "OPEN", "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}],
    }
    state = {} if state is None else state
    log_signal = MagicMock(return_value=99)
    patches = [
        patch("agents.analyst._call_ollama", return_value=json.dumps(parsed)),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.get_trade_ladder", return_value=None),
        patch("db.get_active_plan", return_value=None),
        patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)),
        patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
        patch("config.TRADING_ENV", "live"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "MSFT", "role": "Compounder", "tier": "Core",
                                  "sector": "Software", "held": False}, situation_brief=BRIEF,
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs, state


def test_gated_down_open_records_in_the_decision_and_state_and_routes_nothing():
    result, kw, state = _analyze("OPEN", 1.5)
    d = kw["decision"]
    assert d["final_action"] == "WATCH" and result["pass_to_risk"] is False
    assert d["pending_entry"]["recorded"] and d["pending_entry"]["trigger"] == "gated_down"
    book = json.loads(state["pending_entries_shadow_live"])
    assert book["MSFT"]["signal_id"] == 99 and "price_for_2r" in book["MSFT"]["levels"]


def test_the_shadow_changes_no_gate_routing_or_prompt():
    with patch.object(analyst, "_pending_entry_shadow", return_value=None), \
         patch.object(analyst, "_store_pending_entry"):
        base, base_kw, _ = _analyze("OPEN", 1.5)
    result, kw, _ = _analyze("OPEN", 1.5)
    for k in ("action", "conviction", "pass_to_risk", "reasoning", "thesis_bucket"):
        assert result[k] == base[k]
    assert kw["decision"]["gates"] == base_kw["decision"]["gates"]
    assert kw["prompt"] == base_kw["prompt"]


def test_an_untouched_open_records_nothing():
    _, kw, state = _analyze("OPEN", 3.0)
    assert kw["decision"]["final_action"] == "OPEN" and kw["decision"]["pending_entry"] is None
    assert "pending_entries_shadow_live" not in state


BOOK = {"NVDA": {"trigger": "watch_c6", "model_action": "WATCH", "conviction": 7, "price": 600.0,
                 "et_date": "2026-11-03", "expires": "2026-11-25", "signal_id": 42, "at": "x",
                 "levels": {"price_for_2r": {"side": "at_or_below", "level": 566.67},
                            "add_trigger": {"side": "at_or_above", "level": 645.0}}}}


def _bar(d, low, high, open_=None):
    return {"date": d, "open": open_ or (low + high) / 2, "high": high, "low": low, "close": high}


# A bar after the 11-25 expiry: the cache has loaded the whole window (M216).
AFTER = _bar("2026-11-30", 600, 610)


def _check(bars, today, book=BOOK, written=True, keys=None):
    state = {"pending_entries_shadow_live": json.dumps(book)}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.get_activity_shadow_keys", return_value=keys), \
         patch("db.get_price_history_bulk", return_value={"NVDA": bars}) as bulk, \
         patch("db.log_activity", return_value=written) as act, patch("config.TRADING_ENV", "live"):
        n = tasks._check_pending_entries(today=today)
    return n, act, bulk


def test_a_touch_after_the_verdict_logs_would_fill_once():
    bars = [_bar("2026-11-03", 550, 610),            # verdict day: not counted
            _bar("2026-11-04", 580, 600), _bar("2026-11-05", 560, 590, open_=565)]
    n, act, bulk = _check(bars, date(2026, 11, 5))
    assert n == 1 and bulk.call_args.args[0] == ["NVDA"]
    stage, ticker, _, _, extra = act.call_args.args
    assert (stage, ticker) == ("entry_shadow", "NVDA")
    assert extra["would_fill"] is True and extra["kind"] == "price_for_2r"
    assert extra["fill_date"] == "2026-11-05" and extra["fill_price_est"] == 565.0
    assert extra["bars_to_fill"] == 2 and extra["verdict_signal_id"] == 42
    n, act, _ = _check(bars, date(2026, 11, 6))
    assert n == 0 and not act.called                    # already logged


def test_breakout_level_fills_on_the_close_and_failed_writes_retry():
    bars = [_bar("2026-11-04", 600, 650, open_=648)]
    n, act, _ = _check(bars, date(2026, 11, 4), written=False)
    assert n == 0 and act.call_args.args[4]["kind"] == "add_trigger"
    n, act, _ = _check(bars, date(2026, 11, 4))
    extra = act.call_args.args[4]
    assert n == 1 and extra["fill_price_est"] == 650.0 and extra["price_field"] == "close"
    assert extra["fill_bar"] == {"open": 648, "high": 650, "low": 600, "close": 650}


def test_add_trigger_waits_for_a_close_but_records_the_touch():
    """M208: compile_plan says "a close above"; a high that pokes through
    and closes back under is no fill, but its date is kept."""
    bars = [{"date": "2026-11-04", "open": 630, "high": 650, "low": 620, "close": 640},
            {"date": "2026-11-05", "open": 640, "high": 652, "low": 638, "close": 646}]
    book = {"NVDA": {**BOOK["NVDA"], "levels": {"add_trigger": BOOK["NVDA"]["levels"]["add_trigger"]}}}
    n, act, _ = _check(bars, date(2026, 11, 5), book=book)
    extra = act.call_args.args[4]
    assert n == 1 and extra["fill_date"] == "2026-11-05" and extra["bars_to_fill"] == 2
    assert extra["first_touch_date"] == "2026-11-04" and extra["first_close_date"] == "2026-11-05"
    assert extra["in_zone_at_verdict"] is False   # computed from the stored price for an old entry


def test_the_expired_row_keeps_both_fill_tests_per_level():
    bars = [{"date": "2026-11-04", "open": 630, "high": 650, "low": 620, "close": 640}, AFTER]
    n, act, _ = _check(bars, date(2026, 12, 1))
    extra = act.call_args.args[4]
    assert extra["filled_kinds"] == [] and extra["would_fill"] is False
    out = extra["level_outcomes"]
    assert out["add_trigger"]["first_touch_date"] == "2026-11-04"
    assert out["add_trigger"]["first_close_date"] is None
    assert out["price_for_2r"] == {"level": 566.67, "side": "at_or_below", "price_field": "low",
                                   "in_zone_at_verdict": False, "first_touch_date": None,
                                   "first_close_date": None}


def test_expiry_logs_once_and_ignores_bars_after_it():
    bars = [_bar("2026-11-04", 580, 600), _bar("2026-11-30", 400, 700)]
    n, act, _ = _check(bars, date(2026, 12, 1))
    assert n == 1
    extra = act.call_args.args[4]
    assert extra["would_fill"] is False and extra["expired"] and extra["filled_kinds"] == []
    assert extra["bars_in_window"] == 1 and extra["last_bar_date"] == "2026-11-04"
    assert extra["expired_no_data"] is False
    assert _check(bars, date(2026, 12, 2))[0] == 0


def test_an_expired_entry_that_filled_says_would_fill_true():
    """M207: the expired row agrees with the fill rows."""
    bars = [_bar("2026-11-05", 560, 590), AFTER]
    n, act, _ = _check(bars, date(2026, 12, 1))
    assert n == 2
    extra = act.call_args_list[-1].args[4]
    assert extra["expired"] and extra["filled_kinds"] == ["price_for_2r"]
    assert extra["would_fill"] is True and extra["expired_no_data"] is False


def test_an_expired_entry_with_no_bars_is_marked_no_data():
    """M207: a name with no cached bars is not a no-fill. M216: it waits
    for bars, then is logged incomplete before analyst prunes it."""
    n, act, _ = _check([], date(2026, 12, 1))     # the day before the fallback
    assert n == 0 and not act.called
    n, act, _ = _check([], date(2026, 12, 2))     # expiry + 14 d grace - 7 d slack
    assert n == 1
    extra = act.call_args.args[4]
    assert extra["would_fill"] is False and extra["expired_no_data"] is True
    assert extra["window_complete"] is False
    assert extra["bars_in_window"] == 0 and extra["last_bar_date"] is None
    assert "no bars in the window" in act.call_args.args[2]


def test_the_checker_swallows_errors_and_skips_a_bad_entry():
    bad = {**BOOK, "BAD": {"levels": "nonsense"}}
    with patch("db.get_state", side_effect=RuntimeError("db down")):
        assert tasks._check_pending_entries(today=date(2026, 11, 5)) == 0
    n, _, _ = _check([_bar("2026-11-05", 560, 590)], date(2026, 11, 5), book=bad)
    assert n == 1


def test_the_checker_skips_an_entry_with_no_expiry():
    book = {"NVDA": {**BOOK["NVDA"], "expires": "None"}}
    n, act, _ = _check([_bar("2026-11-05", 560, 590)], date(2026, 11, 5), book=book)
    assert n == 0 and not act.called


def test_refresh_price_history_runs_the_check_after_the_refresh():
    with patch("ingestion.price_history.refresh_all_price_history", return_value=3), \
         patch("tasks._heartbeat_ok"), patch("tasks._check_pending_entries") as chk:
        tasks.refresh_price_history.run()
    chk.assert_called_once()


def test_checker_rows_name_their_code_and_config():
    """m268: fill and expired rows carry code_sha and env_sha."""
    n, act, _ = _check([_bar("2026-11-05", 560, 590), AFTER], date(2026, 12, 1))
    assert n == 2
    for c in act.call_args_list:
        assert c.args[4]["code_sha"] == "test" and c.args[4]["env_sha"]


def test_rows_already_in_the_db_are_not_rewritten_after_a_redis_flush():
    """m267: the activity_feed shadow_key, not only the Redis marker, dedups."""
    import stop_lock
    bars = [_bar("2026-11-05", 560, 590), AFTER]
    n, act, _ = _check(bars, date(2026, 12, 1), keys=set())
    keys = {c.args[4]["shadow_key"] for c in act.call_args_list}
    assert n == 2 and keys == {"live:NVDA:42:price_for_2r", "live:NVDA:42:expired"}
    stop_lock._redis().store.clear()                     # a Redis flush
    n, act, _ = _check(bars, date(2026, 12, 2), keys=keys)
    assert n == 0 and not act.called
    stop_lock._redis().store.clear()
    n, _, _ = _check(bars, date(2026, 12, 2), keys=None)   # DB read failed: Redis alone
    assert n == 2


def test_a_later_non_qualifying_verdict_is_noted_on_the_old_entry():
    """m269: the old entry stays (no censoring decided here), but the later
    verdict is recorded, and the checker's rows carry it."""
    state = {"pending_entries_shadow_live": json.dumps(BOOK)}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("config.TRADING_ENV", "live"):
        analyst._store_pending_entry("NVDA", None, 77, action="OPEN")
        analyst._store_pending_entry("MSFT", None, 78, action="OPEN")    # no entry: nothing
    book = json.loads(state["pending_entries_shadow_live"])
    assert sorted(book) == ["NVDA"] and book["NVDA"]["levels"] == BOOK["NVDA"]["levels"]
    (later,) = book["NVDA"]["later_verdicts"]
    assert later["signal_id"] == 77 and later["action"] == "OPEN" and later["et_date"]
    book["NVDA"]["later_verdicts"] = [{"signal_id": 77, "action": "OPEN", "et_date": "2026-11-04"},
                                      {"signal_id": 79, "action": "HOLD", "et_date": "2026-11-09"}]
    n, act, _ = _check([_bar("2026-11-05", 560, 590), AFTER], date(2026, 12, 1), book=book)
    fill, expired = (c.args[4] for c in act.call_args_list)
    assert [v["signal_id"] for v in fill["later_verdicts"]] == [77]         # before the fill
    assert [v["signal_id"] for v in expired["later_verdicts"]] == [77, 79]


def test_the_analyst_passes_the_final_action_to_the_store():
    import inspect
    src = inspect.getsource(analyst)
    assert "_store_pending_entry(ticker, pending_entry, signal_id, action=action)" in src


def test_the_expired_row_waits_for_the_bars_through_the_expiry():
    """M216: bars that stop short of the expiry leave the entry pending; a
    late fill on the last day is then counted, and the row is final with
    window_complete true."""
    early = [_bar("2026-11-04", 580, 600), _bar("2026-11-20", 590, 600)]
    n, act, _ = _check(early, date(2026, 11, 28))      # past the expiry, bars to 11-20
    assert n == 0 and not act.called
    full = early + [_bar("2026-11-25", 560, 590, open_=570)]
    n, act, _ = _check(full, date(2026, 11, 29))
    assert n == 2
    fill, expired = (c.args[4] for c in act.call_args_list)
    assert fill["fill_date"] == "2026-11-25"
    assert expired["expired"] and expired["window_complete"] is True
    assert expired["would_fill"] is True and expired["filled_kinds"] == ["price_for_2r"]
    assert expired["last_bar_date"] == "2026-11-25" and expired["bars_in_window"] == 3
    assert _check(full, date(2026, 11, 30))[0] == 0     # final: not rewritten


def test_a_fallback_row_is_final_even_when_the_bars_arrive_later():
    """The fallback and the complete row share the {ident}:expired marker."""
    early = [_bar("2026-11-04", 580, 600)]
    n, act, _ = _check(early, date(2026, 12, 2))
    assert n == 1 and act.call_args.args[4]["window_complete"] is False
    n, act, _ = _check(early + [AFTER], date(2026, 12, 3))
    assert n == 0 and not act.called


def test_an_unparseable_expiry_writes_the_fallback_row():
    """A bad expires can't be waited on; it is logged, not lost."""
    book = {"NVDA": {**BOOK["NVDA"], "expires": "2026-11-31"}}
    n, act, _ = _check([_bar("2026-11-04", 580, 600), AFTER], date(2026, 12, 1), book=book)
    assert n == 1
    extra = act.call_args.args[4]
    assert extra["expired"] and extra["window_complete"] is False


def test_a_failing_pending_entries_hook_does_not_skip_the_later_hooks(caplog):
    with patch("ingestion.price_history.refresh_all_price_history", return_value=1), \
         patch("tasks._heartbeat_ok"), patch("tasks._heartbeat_fail") as fail, \
         patch("tasks._check_pending_entries", side_effect=RuntimeError("bars")), \
         patch("tasks._log_plan_cross_section") as cross, \
         patch("tasks._check_tranche2_shadow") as t2, \
         patch("tasks._check_c6_bar_outcomes") as c6, caplog.at_level("WARNING"):
        tasks.refresh_price_history.run()
    cross.assert_called_once()
    t2.assert_called_once()
    c6.assert_called_once()
    fail.assert_not_called()
    assert any("pending entries hook failed: bars" in r.getMessage() for r in caplog.records)
