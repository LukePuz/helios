"""c6 bar shadow (agents/c6_shadow.py): a conviction-6 OPEN/ADD stopped only
by a conviction-7 bar writes one c6_bar_shadow row with the size it would
have got at a bar of 6; nothing else changes. No live DB/IBKR/Redis."""
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents import c6_shadow, crew


# ── analyst side ──────────────────────────────────────────────────────────────

def _analyze(conviction=6, udr=3.0, regime="Neutral", action="OPEN", role="Compounder",
             extra=()):
    response = json.dumps({
        "action": action, "conviction": conviction, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": role,
        "thesis_summary": "test", "expectation_gap": {},
        "catalyst": {"event": "e", "date": "2026-12-20"},
        "valuation": {}, "expected_return_pct": 30.0, "upside_downside_ratio": udr,
        "key_risks": [], "pass_to_risk": True, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })
    log_signal = MagicMock(return_value=41)
    patches = [
        patch("agents.analyst._call_ollama", return_value=response),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": role}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
        *extra,
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from agents.analyst import analyze
        result = analyze({"ticker": "MSFT", "role": role, "tier": "Core", "sector": "Software",
                          "held": action != "OPEN"}, situation_brief="brief",
                         regime_packet={"regime": regime, "score": 50, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"]


def _live(result, decision):
    return (result["action"], result["conviction"], result["pass_to_risk"],
            decision["gates"], decision["final_action"], decision["final_conviction"],
            decision["final_pass_to_risk"])


def test_regime_bar_recorded_in_provenance():
    result, d = _analyze()
    assert result["action"] == "WATCH" and result["pass_to_risk"] is False
    assert result["provenance"]["c6_bar"] == {"model_action": "OPEN", "model_conviction": 6,
                                              "action": "OPEN", "c7_gates": ["regime"],
                                              "ref_price": None}   # the test brief has no price


def test_moonshot_bar_recorded_in_provenance():
    result, _ = _analyze(role="Moonshot", regime="Risk-On")
    assert result["action"] == "WATCH"
    assert result["provenance"]["c6_bar"]["c7_gates"] == ["moonshot"]


def test_flag_off_records_nothing():
    with patch("config.C6_BAR_SHADOW_ENABLED", False):
        result, _ = _analyze()
        assert "c6_bar" not in result["provenance"]
        assert c6_shadow.observe_unrouted(_unrouted_result(), {"regime": "Neutral"}) is None


def test_udr_blocked_c6_records_nothing():
    """The UDR gate runs before the c7 bars: the verdict never reaches one,
    so there is no c6_bar record and the shadow writes nothing."""
    result, d = _analyze(udr=1.5)
    assert "c6_bar" not in result["provenance"]
    with patch("db.log_activity") as la:
        assert c6_shadow.observe_unrouted(result, {"regime": "Neutral"}) is None
    la.assert_not_called()


@pytest.mark.parametrize("conviction", [5, 7])
def test_c5_and_c7_record_nothing(conviction):
    result, _ = _analyze(conviction=conviction)
    assert "c6_bar" not in result["provenance"]


@pytest.mark.parametrize("action", ["HOLD", "EXIT", "TRIM", "WATCH"])
def test_non_entry_actions_record_nothing(action):
    assert c6_shadow.analyst_record(action, 6, action, True, [], action, 6, True) is None


def test_c6_open_in_risk_on_records_no_analyst_bar():
    result, _ = _analyze(regime="Risk-On")
    assert result["pass_to_risk"] is True
    assert result["provenance"]["c6_bar"]["c7_gates"] == []


@pytest.mark.parametrize("scenario", [
    dict(),                                  # regime bar
    dict(udr=1.5),                           # UDR gate
    dict(conviction=7),                      # passes
    dict(conviction=5),                      # c5
    dict(regime="Risk-On"),                  # c6 passes to risk
    dict(role="Moonshot", regime="Risk-On"),  # Moonshot bar
])
def test_live_decision_identical_flag_on_and_off(scenario):
    with patch("config.C6_BAR_SHADOW_ENABLED", True):
        on = _live(*_analyze(**scenario))
    with patch("config.C6_BAR_SHADOW_ENABLED", False):
        off = _live(*_analyze(**scenario))
    assert on == off


def test_analyst_record_exception_leaves_verdict_unchanged():
    base = _live(*_analyze())
    boom = patch("agents.c6_shadow.analyst_record", side_effect=RuntimeError("boom"))
    result, d = _analyze(extra=(boom,))
    assert _live(result, d) == base
    assert "c6_bar" not in result["provenance"]


# ── routing: enqueue only ─────────────────────────────────────────────────────

def _client(pv, cash=None):
    c = MagicMock()
    c.get_account.return_value = SimpleNamespace(portfolio_value=pv, cash=cash if cash else pv,
                                                 buying_power=cash if cash else pv)
    c.get_all_positions.return_value = []
    c.get_open_position.side_effect = Exception("none")
    return c


def _unrouted_result(c7_gates=("regime",), sid=41, ref_price=100.0):
    return {"ticker": "MSFT", "action": "WATCH", "conviction": 6, "conviction_raw": 6,
            "pass_to_risk": False, "thesis_bucket": "Compounder",
            "provenance": {"signal_id": sid, "fundamentals_read": "ok",
                           "c6_bar": {"model_action": "OPEN", "model_conviction": 6,
                                      "action": "OPEN", "c7_gates": list(c7_gates),
                                      "ref_price": ref_price}}}


def _routed_result(conviction=6, action="OPEN"):
    return {"ticker": "MSFT", "action": action, "conviction": conviction, "conviction_raw": 7,
            "pass_to_risk": True, "thesis_bucket": "Compounder", "reasoning": "r",
            "provenance": {"signal_id": 52, "fundamentals_read": "missing",
                           "c6_bar": {"model_action": "OPEN", "model_conviction": 7,
                                      "action": action, "c7_gates": []}}}


def test_unrouted_enqueues_plain_payload_on_ops_without_price_or_broker_calls():
    client = _client(100_000.0)
    with patch("tasks.c6_bar_shadow.apply_async") as aa, \
         patch("agents.crew._get_price") as gp, \
         patch("risk_manager._fetch_bars_for_atr") as fb, \
         patch("ibkr.get_client") as gc:
        crew._c6_bar_unrouted(_unrouted_result(), {"regime": "Neutral"}, client)
    gp.assert_not_called()
    fb.assert_not_called()
    gc.assert_not_called()
    assert client.method_calls == []
    payload = aa.call_args.kwargs["args"][0]
    assert aa.call_args.kwargs["queue"] == "ops"
    json.dumps(payload)                                     # plain data only
    assert payload["c7_gates"] == ["regime"] and payload["prechecks"] is True
    assert (payload["entry_ref_price"], payload["ref_price_source"]) == (100.0, "brief")
    assert payload["shadow_key"].endswith(":MSFT:s41:c6")
    assert payload["session"] in ("pre", "regular", "post", "closed")


def _risk_patches(price_history=()):
    return [
        patch("db.get_state_strict", side_effect=lambda k, d=None: d),
        patch("db.get_state", side_effect=lambda k, d=None: d),
        patch("db.set_state"),
        patch("db.get_portfolio_heat_pct", return_value={"heat_pct": 0.0}),
        patch("db.has_active_approval", return_value=False),
        patch("db.get_last_executed_approval", return_value=None),
        patch("db.is_near_earnings", return_value=False),
        patch("db.get_next_earnings", return_value=None),
        patch("db.get_price_history", return_value=list(price_history)),
        patch("db.get_price_history_bulk", return_value={}),
        patch("db.get_position_stop", return_value=None),   # no stop record: not held
        patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}),
        patch("risk_manager.RiskManager.check_concentration",
              return_value={"passed": True, "flags": []}),
        patch("stop_lock._redis", side_effect=RuntimeError("no redis")),
        patch("config.SETUP_GATES_ENABLED", False),
        patch("risk_manager.SETUP_GATES_ENABLED", False),
    ]


def _route(result, pv, flag=True):
    with ExitStack() as st:
        for p in _risk_patches():
            st.enter_context(p)
        st.enter_context(patch("config.C6_BAR_SHADOW_ENABLED", flag))
        gp = st.enter_context(patch("agents.crew._get_price", return_value=50.0))
        st.enter_context(patch("risk_manager.RiskManager.calculate_stop",
                               return_value=(47.5, "test")))
        aa = st.enter_context(patch("tasks.c6_bar_shadow.apply_async"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        crew._route_to_approval(result, {"regime": "Neutral"}, _client(pv))
    return aa, gp, lpa, ls


def test_tier_floor_route_enqueues_only():
    aa, gp, lpa, ls = _route(_routed_result(), 3_000.0)
    lpa.assert_not_called()                                   # live: still no order
    assert ls.call_args.args[1] == "ROUTE_ZERO_SHARES"
    assert gp.call_count == 1                                 # the live quote only
    payload = aa.call_args.kwargs["args"][0]
    assert payload["c7_gates"] == ["tier_floor"] and payload["prechecks"] is False
    assert (payload["entry_ref_price"], payload["ref_price_source"]) == (50.0, "routing_quote")
    assert (payload["conviction_raw"], payload["model_conviction"],
            payload["fundamentals_read"]) == (7, 7, "missing")


@pytest.mark.parametrize("flag", [True, False])
def test_tier_floor_live_route_identical_flag_on_and_off(flag):
    aa, gp, lpa, ls = _route(_routed_result(), 3_000.0, flag=flag)
    lpa.assert_not_called()
    assert ls.call_args.args[1] == "ROUTE_ZERO_SHARES" and gp.call_count == 1
    assert aa.called is flag


@pytest.mark.parametrize("conviction", [5, 7])
def test_tier_floor_c5_c7_enqueue_nothing(conviction):
    sizing = {"flags": [f"Conviction {conviction} below the micro-tier minimum (8) — skipping"]}
    with patch("tasks.c6_bar_shadow.apply_async") as aa:
        assert c6_shadow.observe_tier_floor(_routed_result(conviction), "Neutral", 50.0,
                                            sizing) is None
    aa.assert_not_called()


def test_tier_floor_exit_enqueues_nothing():
    sizing = {"flags": ["Conviction 6 below the micro-tier minimum (7) — skipping"]}
    with patch("tasks.c6_bar_shadow.apply_async") as aa:
        assert c6_shadow.observe_tier_floor(_routed_result(action="EXIT"), "Neutral", 50.0,
                                            sizing) is None
    aa.assert_not_called()


def test_enqueue_failure_never_reaches_routing():
    with patch("tasks.c6_bar_shadow.apply_async", side_effect=RuntimeError("broker down")):
        assert c6_shadow.observe_unrouted(_unrouted_result(), {"regime": "Neutral"}) is None
    with patch("agents.c6_shadow.observe_unrouted", side_effect=RuntimeError("boom")):
        crew._c6_bar_unrouted(_unrouted_result(), {"regime": "Neutral"}, MagicMock())
    with patch("agents.c6_shadow.observe_tier_floor", side_effect=RuntimeError("boom")):
        crew._c6_bar_tier_floor(_routed_result(), "Neutral", 50.0, {})


def test_session_of():
    assert c6_shadow.session_of("2026-11-03T13:00:00+00:00") == "pre"       # 08:00 ET
    assert c6_shadow.session_of("2026-11-03T15:00:00+00:00") == "regular"
    assert c6_shadow.session_of("2026-11-03T21:30:00+00:00") == "post"
    assert c6_shadow.session_of("2026-11-27T19:00:00+00:00") == "post"      # early close
    assert c6_shadow.session_of("2026-11-26T15:00:00+00:00") == "closed"    # Thanksgiving
    assert c6_shadow.session_of("2026-11-07T15:00:00+00:00") == "closed"    # Saturday


# ── the task ──────────────────────────────────────────────────────────────────

def _payload(result=None, regime="Neutral"):
    with patch("tasks.c6_bar_shadow.apply_async") as aa:
        c6_shadow.observe_unrouted(result or _unrouted_result(), {"regime": regime})
    return aa.call_args.kwargs["args"][0]


def _run(payload, client, extra=(), price_history=()):
    with ExitStack() as st:
        for p in _risk_patches(price_history):
            st.enter_context(p)
        for p in extra:
            st.enter_context(p)
        la = st.enter_context(patch("db.log_activity", return_value=True))
        send = st.enter_context(patch("alerts._send"))
        fb = st.enter_context(patch("risk_manager._fetch_bars_for_atr"))
        rec = c6_shadow.run(payload, client)
    return rec, la, send, fb


def test_regime_blocked_c6_open_would_queue_with_size():
    rec, la, send, fb = _run(_payload(), _client(100_000.0))
    assert rec["would_queue"] is True and rec["blocked_also_by"] == []
    assert rec["c7_gates"] == ["regime"] and rec["account_tier"] == "standard"
    assert rec["intended_shares"] >= 1 and rec["intended_notional_pct_of_equity"] > 0
    assert rec["entry_ref_price"] == 100.0 and rec["conviction"] == 6
    stage, ticker, msg, sentiment, extra = la.call_args.args
    assert (stage, ticker, extra["kind"]) == ("c6_bar_shadow", "MSFT", "verdict")
    assert "$" not in msg
    send.assert_not_called()
    fb.assert_not_called()


def test_tier_floor_blocked_c6_open_would_queue_with_size():
    aa, *_ = _route(_routed_result(), 3_000.0)
    rec, la, send, fb = _run(aa.call_args.kwargs["args"][0], _client(3_000.0))
    assert rec["c7_gates"] == ["tier_floor"] and rec["account_tier"] == "micro"
    assert rec["would_queue"] is True and rec["intended_shares"] >= 1
    assert rec["signal_id"] == 52 and rec["conviction_raw"] == 7


def test_regime_blocked_on_micro_account_also_names_tier_floor():
    rec, *_ = _run(_payload(_unrouted_result(ref_price=50.0)), _client(3_000.0))
    assert rec["c7_gates"] == ["regime", "tier_floor"] and rec["would_queue"] is True


def test_routing_block_is_would_queue_false():
    rec, *_ = _run(_payload(), _client(100_000.0),
                   extra=(patch("db.has_active_approval", return_value=True),))
    assert rec["would_queue"] is False and rec["blocked_also_by"] == ["active_approval"]


def test_atr_failure_sends_no_alert_and_uses_no_broker_bars():
    """calculate_stop is not patched: no cached bars, so the ATR fallback
    stop is used, without the Telegram alert or an IBKR bar request."""
    rec, la, send, fb = _run(_payload(), _client(100_000.0))
    assert rec["would_queue"] is True
    send.assert_not_called()
    fb.assert_not_called()


def test_cached_bars_give_the_atr_stop():
    import risk_manager
    bars = [{"date": f"2026-10-{d:02d}", "open": 100, "high": 101, "low": 99, "close": 100.0}
            for d in range(1, 25)]
    with patch("db.get_price_history", return_value=bars), \
         patch("risk_manager._fetch_bars_for_atr") as fb:
        rm = risk_manager.RiskManager(_client(100_000.0), simulation=True)
        stop, method = rm.calculate_stop("MSFT", 100.0, "Compounder", log_fallback=False)
    assert "ATR" in method and "Fallback" not in method
    fb.assert_not_called()


def test_task_sizing_has_no_side_effects():
    """Neither the high-water mark nor the halt flag is written, and no cash
    alert is sent."""
    with patch("db.set_state_verified") as ssv:
        rec, la, send, fb = _run(_payload(), _client(100_000.0, cash=1.0))
    assert rec["would_queue"] is False and rec["blocked_also_by"] == ["cash_blocked"]
    send.assert_not_called()
    ssv.assert_not_called()


def test_task_without_price_uses_cached_close():
    bars = [{"date": "2026-11-02", "open": 1, "high": 1, "low": 1, "close": 80.0}]
    rec, *_ = _run(_payload(_unrouted_result(ref_price=None)), _client(100_000.0),
                   price_history=bars)
    assert (rec["entry_ref_price"], rec["ref_price_source"], rec["ref_close_date"]) == \
        (80.0, "price_history_close", "2026-11-02")


def test_task_exception_recorded_not_raised():
    rec, *_ = _run(_payload(), _client(100_000.0),
                   extra=(patch("risk_manager.RiskManager.size_position",
                                side_effect=RuntimeError("boom")),))
    assert rec["would_queue"] is None and "RuntimeError" in rec["error"]


def test_task_entry_point():
    import tasks
    with patch("agents.c6_shadow.run") as run:
        tasks.c6_bar_shadow.run({"ticker": "MSFT"})
    run.assert_called_once_with({"ticker": "MSFT"})


# ── outcomes ──────────────────────────────────────────────────────────────────

def _bars(n, start_close=100.0):
    """n daily bars, one per NYSE session from 2026-11-02 (the verdict day),
    close start_close + the session's index."""
    import tasks
    days = ["2026-11-02"] + tasks._c6_sessions_after("2026-11-02", max(n - 1, 1))
    return [{"date": d, "open": 1, "high": 1, "low": 1, "close": start_close + i}
            for i, d in enumerate(days[:n])]


_VERDICT = {"kind": "verdict", "ticker": "MSFT", "env": "paper", "signal_id": 41,
            "as_of": "2026-11-02T15:00:00+00:00", "entry_ref_price": 100.0, "action": "OPEN",
            "would_queue": True, "c7_gates": ["regime"], "shadow_key": "paper:MSFT:s41:c6"}


def _outcomes(verdict, bars, done=frozenset(), today=None, actions=(), raw_close=None):
    import tasks
    acts = ({"side_effect": actions} if isinstance(actions, Exception) else
            {"return_value": {"events": list(actions), "raw_close": dict(raw_close or {}),
                              "fetched_at": "2026-12-31T00:00:00+00:00"}})
    with patch("db.get_activity_rows", return_value=[verdict]) as rows, \
         patch("tasks._c6_corporate_actions", **acts), \
         patch("db.get_activity_shadow_keys", return_value=set(done)), \
         patch("db.get_price_history_bulk", return_value={"MSFT": bars}), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
         patch("db.log_activity", return_value=True) as la:
        n = tasks._check_c6_bar_outcomes(today=today)
    return n, [c.args[4] for c in la.call_args_list], rows


def test_outcome_tracker_logs_each_horizon_and_status_once():
    n, recs, rows = _outcomes(_VERDICT, _bars(22))
    assert n == 4
    assert rows.call_args.args[:2] == ("c6_bar_shadow", "verdict")
    assert [r.get("horizon_days") for r in recs[:3]] == [5, 10, 20]
    # bar 0 is the verdict day (2026-11-02, close 100); the 5th bar after it closes 105
    assert recs[0]["close"] == 105.0 and recs[0]["return_pct"] == 5.0
    assert recs[0]["verdict_day_close"] == 100.0
    assert recs[2]["shadow_key"] == "paper:MSFT:s41:c6:r20"
    assert (recs[3]["kind"], recs[3]["status"], recs[3]["horizons_logged"]) == \
        ("outcome_status", "complete", [5, 10, 20])
    n2, recs2, _ = _outcomes(_VERDICT, _bars(22), done={r["shadow_key"] for r in recs})
    assert n2 == 0 and recs2 == []


def test_outcome_horizon_counts_sessions_not_cached_bars():
    """A session missing from the cache doesn't move the horizon to a later
    bar; the outcome says how many sessions inside it were missing."""
    import tasks
    sessions = tasks._c6_sessions_after("2026-11-02", 20)
    assert sessions[:5] == ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
    assert "2026-11-26" not in sessions          # Thanksgiving
    bars = [b for b in _bars(22) if b["date"] != "2026-11-04"]
    n, recs, _ = _outcomes(_VERDICT, bars)
    r5, r10, r20 = recs[:3]
    assert (r5["close_date"], r5["close"]) == ("2026-11-09", 105.0)
    assert (r5["missing_sessions"], r5["bars_gap"]) == (1, True)
    assert (r20["close_date"], r20["close"], r20["bars_gap"]) == (sessions[19], 120.0, True)
    _, full, _ = _outcomes(_VERDICT, _bars(22))
    assert (full[0]["missing_sessions"], full[0]["bars_gap"]) == (0, False)


def test_outcome_horizon_without_its_own_session_bar_is_not_logged():
    bars = [b for b in _bars(22) if b["date"] != "2026-11-09"]
    n, recs, _ = _outcomes(_VERDICT, bars)
    assert [r.get("horizon_days") for r in recs] == [10, 20]
    assert all(r["kind"] == "outcome" for r in recs)


def test_outcome_tracker_waits_for_bars():
    from datetime import date
    n, recs, _ = _outcomes(_VERDICT, _bars(5), today=date(2026, 11, 6))
    assert n == 0 and recs == []


@pytest.mark.parametrize("bars,status", [(5, "missing_bars"), (0, "expired")])
def test_outcome_status_after_40_sessions(bars, status):
    from datetime import date
    n, recs, _ = _outcomes(_VERDICT, _bars(bars), today=date(2027, 1, 15))
    assert recs[-1]["kind"] == "outcome_status" and recs[-1]["status"] == status
    assert recs[-1]["shadow_key"] == "paper:MSFT:s41:c6:status"


def test_outcome_status_no_ref():
    n, recs, _ = _outcomes({**_VERDICT, "entry_ref_price": None}, _bars(22))
    assert n == 1 and recs[0]["status"] == "no_ref"


def test_outcome_tracker_hooked_into_price_refresh():
    import tasks
    with patch("ingestion.price_history.refresh_all_price_history", return_value=1), \
         patch("tasks._heartbeat_ok"), patch("tasks._check_pending_entries"), \
         patch("tasks._log_plan_cross_section"), patch("tasks._check_tranche2_shadow"), \
         patch("tasks._check_c6_bar_outcomes") as c6:
        tasks.refresh_price_history.run()
    c6.assert_called_once()


# ── task hardening ────────────────────────────────────────────────────────────

def test_task_time_limits():
    import tasks
    assert (tasks.c6_bar_shadow.soft_time_limit, tasks.c6_bar_shadow.time_limit) == (20, 30)


def test_enqueue_passes_retry_false():
    with patch("tasks.c6_bar_shadow.apply_async") as aa:
        c6_shadow.observe_unrouted(_unrouted_result(), {"regime": "Neutral"})
    assert aa.call_args.kwargs["retry"] is False and aa.call_args.kwargs["queue"] == "ops"


def test_null_signal_id_keys_on_the_et_date():
    payload = _payload(_unrouted_result(sid=None))
    et = c6_shadow._et_date(payload["as_of"])
    assert payload["shadow_key"].endswith(f":MSFT:d{et}:c6")


def test_broker_not_connected_skips_evaluation_without_connecting():
    import ibkr
    with patch.object(ibkr, "_ib", None), \
         patch("ibkr.get_client") as gc, patch("ibkr.get_ib") as gib:
        rec, la, send, fb = _run(_payload(), None)
    gc.assert_not_called()
    gib.assert_not_called()
    assert rec["would_queue"] is None and rec["blocked_also_by"] == ["broker_unavailable"]
    assert rec["intended_shares"] is None and la.call_args.args[4]["kind"] == "verdict"


def test_broker_connected_uses_the_worker_client():
    import ibkr
    ib = MagicMock()
    ib.isConnected.return_value = True
    with patch.object(ibkr, "_ib", ib), \
         patch("ibkr.get_client", return_value=_client(100_000.0)) as gc:
        rec, *_ = _run(_payload(), None)
    gc.assert_called_once()
    assert rec["would_queue"] is True


@pytest.mark.parametrize("where", ["size_position", "routing"])
def test_soft_time_limit_is_recorded_then_reraised(where):
    from celery.exceptions import SoftTimeLimitExceeded
    target = (patch("risk_manager.RiskManager.size_position", side_effect=SoftTimeLimitExceeded())
              if where == "size_position"
              else patch("db.has_active_approval", side_effect=SoftTimeLimitExceeded()))
    with pytest.raises(SoftTimeLimitExceeded):
        with ExitStack() as st:
            for p in _risk_patches():
                st.enter_context(p)
            st.enter_context(target)
            la = st.enter_context(patch("db.log_activity", return_value=True))
            c6_shadow.run(_payload(), _client(100_000.0))
    rec = la.call_args.args[4]
    assert rec["error"] == "SoftTimeLimitExceeded" and rec["would_queue"] is None


def test_class_level_flags_stay_false():
    import risk_manager
    _run(_payload(), _client(100_000.0))
    client = _client(100_000.0, cash=50_000.0)
    with ExitStack() as st:
        for p in _risk_patches():
            st.enter_context(p)
        st.enter_context(patch("alerts._send"))
        st.enter_context(patch("risk_manager._fetch_bars_for_atr", return_value=[]))
        risk_manager.RiskManager(client).would_size_after_swap(
            "NEW", 7, "Compounder", "Neutral", 100.0, {"ticker": "OLD", "market_value": 0.0})
    assert risk_manager.RiskManager.simulation is False
    assert risk_manager.RiskManager.quiet_fallback_alert is False


def test_flag_not_in_treatment_config():
    from agents.analyst import _treatment_config
    assert "C6_BAR_SHADOW_ENABLED" not in _treatment_config()


def test_session_counts_past_the_holiday_calendar_warn_once(caplog):
    """NYSE_HOLIDAYS ends at CALENDAR_LAST_YEAR: counting past it warns once
    per process (no holidays are made up), and within it stays quiet."""
    import logging
    import tasks
    from market_calendar import CALENDAR_LAST_YEAR
    with patch.object(tasks, "_calendar_end_warned", False, create=True), caplog.at_level(logging.WARNING):
        tasks._c6_sessions_after("2026-11-02", 20)
        tasks._c6_trading_days_after("2026-11-02", "2026-12-31")
        assert not [r for r in caplog.records if "holiday calendar ends" in r.getMessage()]
        last = f"{CALENDAR_LAST_YEAR}-12-20"
        assert len(tasks._c6_sessions_after(last, 20)) == 20
        tasks._c6_trading_days_after(last, f"{CALENDAR_LAST_YEAR + 1}-02-01")
        tasks._c6_sessions_after(last, 5)
    warned = [r for r in caplog.records if "holiday calendar ends" in r.getMessage()]
    assert len(warned) == 1 and str(CALENDAR_LAST_YEAR) in warned[0].getMessage()


def test_outcome_rows_flag_splits_and_dividends_after_the_verdict():
    """m602: events after the verdict day are listed; in_window flags cover
    those up to the horizon's close date."""
    acts = [{"date": "2026-10-30", "kind": "dividend", "value": 0.8},   # before: ignored
            {"date": "2026-11-04", "kind": "dividend", "value": 0.8},   # inside r5
            {"date": "2026-11-20", "kind": "split", "value": 4.0}]      # after r5, inside r20
    n, recs, _ = _outcomes(_VERDICT, _bars(22), today=__import__("datetime").date(2026, 12, 31), actions=acts)
    r5, r20 = recs[0], recs[2]
    assert [a["date"] for a in r5["corp_actions_after_verdict"]] == ["2026-11-04", "2026-11-20"]
    assert (r5["dividend_in_window"], r5["split_in_window"]) == (True, False)
    assert (r20["dividend_in_window"], r20["split_in_window"]) == (True, True)
    # M320: the split after r5's close still biases r5's adjusted return
    assert (r5["split_after_verdict_any"], r5["dividend_after_verdict_any"]) == (True, True)


def test_outcome_rows_carry_the_unadjusted_close_return():
    """63d: yfinance's Close is split-adjusted; a 2:1 split after the r5
    close is multiplied back."""
    d5 = _bars(22)[5]["date"]
    n, recs, _ = _outcomes(_VERDICT, _bars(22), today=__import__("datetime").date(2026, 12, 31),
                           raw_close={d5: 55.0},
                           actions=[{"date": "2026-12-20", "kind": "split", "value": 2.0}])
    assert recs[0]["close_split_adj"] == 55.0 and recs[0]["close_unadjusted"] == 110.0
    assert recs[0]["return_pct_unadjusted"] == 10.0
    assert recs[1]["close_split_adj"] is None and recs[1]["return_pct_unadjusted"] is None
    assert recs[0]["corp_actions_fetched_at"] == "2026-12-31T00:00:00+00:00"


def test_unreadable_corporate_actions_are_unknown_not_none():
    with patch("tasks._c6_corporate_actions", return_value=None):
        import tasks
        with patch("db.get_activity_rows", return_value=[_VERDICT]), \
             patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.get_price_history_bulk", return_value={"MSFT": _bars(22)}), \
             patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
             patch("db.log_activity", return_value=True) as la:
            tasks._check_c6_bar_outcomes(today=__import__("datetime").date(2026, 12, 31))
    r5 = la.call_args_list[0].args[4]
    assert r5["corp_actions_after_verdict"] is None and r5["split_in_window"] is None


def test_corporate_actions_are_parsed_and_cached():
    import pandas as pd
    import tasks
    idx = pd.to_datetime(["2026-11-04", "2026-11-20"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"Close": [1.0, 1.0], "Dividends": [0.8, 0.0], "Stock Splits": [0.0, 4.0]},
                         index=idx)
    r = MagicMock()
    r.get.return_value = None
    with patch("yfinance.Ticker") as t:
        t.return_value.history.return_value = frame
        cache = {}
        out = tasks._c6_corporate_actions("MSFT", r, cache)
        assert tasks._c6_corporate_actions("MSFT", r, cache) is out
    assert out["events"] == [{"date": "2026-11-04", "kind": "dividend", "value": 0.8},
                             {"date": "2026-11-20", "kind": "split", "value": 4.0}]
    assert out["raw_close"] == {"2026-11-04": 1.0, "2026-11-20": 1.0} and out["fetched_at"]
    assert t.call_count == 1 and r.set.call_args.kwargs["ex"] == tasks._C6_ACTIONS_TTL_S
    assert t.return_value.history.call_args.kwargs["timeout"] == 10
    with patch("yfinance.Ticker", side_effect=RuntimeError("rate limited")):
        assert tasks._c6_corporate_actions("AMD", None, {}) is None
    r2 = MagicMock()
    r2.get.return_value = None
    with patch("yfinance.Ticker") as t2:      # an empty frame is unknown, not "none", and not cached
        t2.return_value.history.return_value = pd.DataFrame()
        assert tasks._c6_corporate_actions("AMD", r2, {}) is None
    r2.set.assert_not_called()
    with patch("yfinance.Ticker") as t3:      # bars but no events: none found
        t3.return_value.history.return_value = pd.DataFrame(
            {"Close": [1.0], "Dividends": [0.0], "Stock Splits": [0.0]}, index=idx[:1])
        assert tasks._c6_corporate_actions("AMD", None, {})["events"] == []
