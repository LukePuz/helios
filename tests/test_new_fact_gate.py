"""Buy gate on the screener's new_fact (crew.new_fact_gate): an OPEN/ADD
routes only when its news item was marked a new fact, per
BUY_NEW_FACT_SCOPE and BUY_NEW_FACT_WINDOW; otherwise a ROUTE_NO_NEW_FACT
row. Sells, HOLD and WATCH are never gated; the verdict is untouched. The
conftest turns the gate off for other tests; these turn it on."""
import copy
import logging
import re
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents import c6_shadow, crew


def _cfg(scope="all_buys", window="same_run", on=True):
    return [patch("config.BUY_REQUIRES_NEW_FACT", on), patch("config.BUY_NEW_FACT_SCOPE", scope),
            patch("config.BUY_NEW_FACT_WINDOW", window)]


def _gate(action, trigger, scope="all_buys", window="same_run", items=None, on=True):
    with ExitStack() as st:
        for p in _cfg(scope, window, on):
            st.enter_context(p)
        lookup = st.enter_context(patch("db.recent_new_fact_items", return_value=items))
        return crew.new_fact_gate("NVDA", action, trigger), lookup


EVENT_NEW = {"path": "event", "new_fact": True}
EVENT_OLD = {"path": "event", "new_fact": False}
EVENT_NONE = {"path": "event", "new_fact": None}
SCAN = {"path": "daily_scan", "source": "scanner"}
REVIEW = {"path": "scheduled_review"}
ITEM = [{"signal_id": 9, "timestamp": "2026-11-03 14:00:00+00", "title": "deal", "url": "u"}]


# ── defaults ──────────────────────────────────────────────────────────────────

def test_config_defaults():
    src = (Path(crew.__file__).parent.parent / "config.py").read_text()
    assert re.search(r'getenv\("BUY_REQUIRES_NEW_FACT", "true"\)', src)
    assert re.search(r'getenv\("BUY_NEW_FACT_SCOPE", "news_only"\)', src)
    assert re.search(r'getenv\("BUY_NEW_FACT_WINDOW", "24h"\)', src)


def test_off_or_not_a_buy_is_not_gated():
    assert _gate("OPEN", EVENT_OLD, on=False)[0] is None
    for action in ("TRIM", "EXIT", "HOLD", "WATCH", "PASS", None):
        assert _gate(action, EVENT_OLD)[0] is None


# ── scope all_buys (default) ──────────────────────────────────────────────────

@pytest.mark.parametrize("action", ["OPEN", "ADD"])
def test_all_buys_passes_a_new_fact_item(action):
    g, lookup = _gate(action, EVENT_NEW)
    assert g["passed"] is True and g["applies"] is True and g["item_new_fact"] is True
    lookup.assert_not_called()


def test_all_buys_blocks_an_opinion_item():
    g, lookup = _gate("OPEN", EVENT_OLD)
    assert g["passed"] is False and g["item_new_fact"] is False
    assert g["reason"] == ("no new fact (all_buys, same_run): the triggering item is not "
                           "a new fact (opinion/recap)")
    lookup.assert_not_called()


def test_missing_new_fact_counts_as_false_and_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="agents.crew"):
        g, _ = _gate("ADD", EVENT_NONE)
    assert g["passed"] is False and g["item_new_fact"] is None and g["item_new_fact_missing"]
    assert "gave no new_fact (counted as false)" in g["reason"]
    assert any("new_fact missing" in r.message for r in caplog.records)
    g2, _ = _gate("ADD", {"path": "event", "new_fact": "yes"})     # invalid: same
    assert g2["passed"] is False and g2["item_new_fact"] is None


@pytest.mark.parametrize("trigger, path", [(SCAN, "daily_scan"), (REVIEW, "scheduled_review"),
                                           ({"path": "plan_level_review"}, "plan_level_review"),
                                           (None, "unknown"), ({}, "unknown")])
def test_all_buys_blocks_paths_without_a_news_item(trigger, path):
    g, _ = _gate("ADD", trigger)
    assert g["passed"] is False and g["news_item"] is False
    assert g["reason"] == f"no new fact (all_buys, same_run): a {path} verdict has no news item"


def test_new_fact_on_a_non_event_trigger_is_ignored():
    g, _ = _gate("OPEN", {"path": "daily_scan", "new_fact": True})
    assert g["passed"] is False and g["item_new_fact"] is None


def test_unknown_settings_fall_back_to_the_defaults():
    g, _ = _gate("OPEN", SCAN, scope="bogus", window="1y")
    assert (g["scope"], g["window"], g["passed"]) == ("all_buys", "same_run", False)


# ── scope news_only ───────────────────────────────────────────────────────────

def test_news_only_leaves_non_news_buys_alone():
    for trig in (SCAN, REVIEW, None):
        g, _ = _gate("OPEN", trig, scope="news_only")
        assert g["applies"] is False and g["passed"] is True


def test_news_only_gates_news_buys():
    assert _gate("OPEN", EVENT_OLD, scope="news_only")[0]["passed"] is False
    assert _gate("OPEN", EVENT_NONE, scope="news_only")[0]["passed"] is False
    assert _gate("OPEN", EVENT_NEW, scope="news_only")[0]["passed"] is True


# ── scope all_but_held_review_add ─────────────────────────────────────────────

@pytest.mark.parametrize("path", ["scheduled_review", "earnings_review", "plan_level_review"])
def test_held_review_add_is_exempt(path):
    g, _ = _gate("ADD", {"path": path}, scope="all_but_held_review_add")
    assert g["applies"] is False and g["passed"] is True


def test_all_but_held_review_add_gates_the_rest():
    s = "all_but_held_review_add"
    assert _gate("OPEN", REVIEW, scope=s)[0]["passed"] is False    # only ADDs are exempt
    assert _gate("OPEN", SCAN, scope=s)[0]["passed"] is False
    assert _gate("ADD", SCAN, scope=s)[0]["passed"] is False
    assert _gate("ADD", EVENT_OLD, scope=s)[0]["passed"] is False
    assert _gate("ADD", EVENT_NEW, scope=s)[0]["passed"] is True


# ── windows ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("window, hours", [("24h", 24), ("3d", 72)])
def test_window_finds_a_recent_new_fact_item(window, hours):
    g, lookup = _gate("OPEN", EVENT_OLD, window=window, items=ITEM)
    assert g["passed"] is True and g["window_items"] == ITEM
    assert lookup.call_args.args == ("NVDA", hours)
    g2, _ = _gate("OPEN", SCAN, window=window, items=ITEM)        # a scan buy too
    assert g2["passed"] is True


@pytest.mark.parametrize("window", ["24h", "3d"])
def test_window_with_no_item_blocks(window):
    g, _ = _gate("OPEN", SCAN, window=window, items=[])
    assert g["passed"] is False
    assert g["reason"] == (f"no new fact (all_buys, {window}): a daily_scan verdict has no news "
                           f"item; no new-fact item on NVDA within {window}")


def test_window_lookup_failure_blocks():
    g, _ = _gate("OPEN", EVENT_OLD, window="24h", items=None)
    assert g["passed"] is False and g["window_lookup_error"] is True
    assert g["reason"].endswith("(lookup failed)")


def test_window_own_item_needs_no_lookup():
    g, lookup = _gate("OPEN", EVENT_NEW, window="3d", items=[])
    assert g["passed"] is True
    lookup.assert_not_called()


def test_db_lookup_query_shape():
    import db
    cur = MagicMock()
    cur.fetchall.return_value = [(9, "2026-11-03", "deal", "u")]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.recent_new_fact_items("NVDA", 24)
    sql, params = cur.execute.call_args.args
    assert "provenance->'trigger'->>'new_fact' = 'true'" in sql
    assert "provenance->'trigger'->>'path' = 'event'" in sql
    assert params[0] == "NVDA" and params[-1] == 24
    assert out == [{"signal_id": 9, "timestamp": "2026-11-03", "title": "deal", "url": "u"}]
    with patch("db._get_conn", side_effect=RuntimeError("down")):
        assert db.recent_new_fact_items("NVDA", 24) is None
    conn.cursor.side_effect = RuntimeError("down")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.recent_new_fact_items("NVDA", 24) is None


# ── routing ───────────────────────────────────────────────────────────────────

def _result(action="OPEN", trigger=None, conviction=8):
    return {"ticker": "NVDA", "action": action, "conviction": conviction,
            "thesis_bucket": "Accelerator", "reasoning": "r", "pass_to_risk": True,
            "provenance": {"signal_id": 41, "model": "m", "trigger": trigger}}


def _route(result, scope="all_buys", window="same_run", items=None, held=0.0):
    client = MagicMock()
    client.get_all_positions.return_value = (
        [SimpleNamespace(symbol="NVDA", qty=10.0, avg_entry_price=100.0, current_price=100.0,
                         market_value=1000.0)] if held else [])
    with ExitStack() as st:
        for p in _cfg(scope, window):
            st.enter_context(p)
        st.enter_context(patch("db.recent_new_fact_items", return_value=items))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=100.0))
        st.enter_context(patch("tasks.c6_bar_shadow.apply_async"))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        rm = rm_cls.return_value
        rm.size_position.return_value = {"shares_initial": 2, "initial_dollar": 200.0,
                                         "position_size_pct": 0.02, "initial_entry_pct": 0.55,
                                         "flags": [], "stop_loss_price": 90.0}
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, client)
    return rm_cls, lpa, ls


@pytest.mark.parametrize("trigger", [EVENT_OLD, EVENT_NONE, SCAN, REVIEW, None])
def test_route_blocks_with_a_route_row_before_sizing(trigger):
    result = _result("OPEN", trigger)
    before = copy.deepcopy(result)
    rm_cls, lpa, ls = _route(result)
    lpa.assert_not_called()
    rm_cls.assert_not_called()                     # no sizing, so no swap sale
    assert ls.call_count == 1                      # the route row only
    assert ls.call_args.args[1] == "ROUTE_NO_NEW_FACT"
    assert ls.call_args.args[3].startswith("no new fact (all_buys, same_run): ")
    decision = ls.call_args.kwargs["decision"]
    assert decision["route_exit"] == "ROUTE_NO_NEW_FACT"
    assert decision["new_fact_gate"]["passed"] is False
    assert result == before                        # the verdict is untouched


def test_route_passes_and_records_the_gate():
    rm_cls, lpa, ls = _route(_result("OPEN", EVENT_NEW))
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["new_fact_gate"]["passed"] is True and risk["new_fact_gate"]["item_new_fact"]
    assert not [c for c in ls.call_args_list if c.args[1] == "ROUTE_NO_NEW_FACT"]


@pytest.mark.parametrize("scope, trigger, window, items, routed", [
    ("news_only", SCAN, "same_run", None, True),
    ("news_only", EVENT_OLD, "same_run", None, False),
    ("all_but_held_review_add", REVIEW, "same_run", None, True),
    ("all_but_held_review_add", SCAN, "same_run", None, False),
    ("all_buys", SCAN, "24h", ITEM, True),
    ("all_buys", SCAN, "3d", [], False),
])
def test_route_follows_scope_and_window(scope, trigger, window, items, routed):
    _, lpa, ls = _route(_result("ADD", trigger), scope=scope, window=window, items=items)
    assert lpa.called is routed
    assert any(c.args[1] == "ROUTE_NO_NEW_FACT" for c in ls.call_args_list) is not routed


@pytest.mark.parametrize("action", ["TRIM", "EXIT"])
def test_sells_are_not_gated(action):
    _, lpa, ls = _route(_result(action, SCAN), held=10.0)
    assert lpa.call_args.kwargs["action"] == action
    assert "new_fact_gate" not in lpa.call_args.kwargs["risk_json"]
    assert not [c for c in ls.call_args_list if c.args[1] == "ROUTE_NO_NEW_FACT"]


def test_verdict_inputs_identical_gate_on_and_off():
    """The gate sits in routing only: what the analyst gets (candidate,
    trigger) and what it returns are the same with the gate on or off."""
    seen = []

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        trigger = copy.deepcopy(trigger)
        trigger.pop("brief_built_at", None)   # a clock reading
        seen.append((copy.deepcopy(candidate), trigger))
        return {"action": "WATCH", "ticker": "NVDA", "conviction": 5, "pass_to_risk": False}
    out = []
    for on in (True, False):
        with patch("config.BUY_REQUIRES_NEW_FACT", on), \
             patch("ibkr.IBKRClient", side_effect=RuntimeError("no broker")), \
             patch("agents.analyst.analyze", fake_analyze), \
             patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
             patch("knowledge_loader.get_watchlist_entry", return_value={}), \
             patch("agents.crew._c6_bar_unrouted"):
            out.append(crew.deep_analyze({"ticker": "NVDA", "source": "news", "title": "t",
                                          "text": "x"}, {"relevant": True, "new_fact": False}))
    assert seen[0] == seen[1] and out[0] == out[1]


# ── c6 shadow ─────────────────────────────────────────────────────────────────

def _c6_client(pv):
    c = MagicMock()
    c.get_account.return_value = SimpleNamespace(portfolio_value=pv, cash=pv, buying_power=pv)
    c.get_all_positions.return_value = []
    c.get_open_position.side_effect = Exception("none")
    return c


def _c6_run(payload, client, scope="all_buys", on=True):
    with ExitStack() as st:
        for p in _cfg(scope, on=on):
            st.enter_context(p)
        for p in (patch("config.C6_BAR_SHADOW_ENABLED", True),
                  patch("config.SETUP_GATES_ENABLED", False),
                  patch("risk_manager.SETUP_GATES_ENABLED", False),
                  patch("db.get_state", side_effect=lambda k, d=None: d), patch("db.set_state"),
                  patch("db.get_portfolio_heat_pct", return_value={"heat_pct": 0.0}),
                  patch("db.has_active_approval", return_value=False),
                  patch("db.get_last_executed_approval", return_value=None),
                  patch("db.is_near_earnings", return_value=False),
                  patch("db.get_price_history", return_value=[]),
                  patch("db.get_price_history_bulk", return_value={}),
                  patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}),
                  patch("risk_manager.RiskManager.check_concentration",
                        return_value={"passed": True, "flags": []}),
                  patch("risk_manager.RiskManager.calculate_stop", return_value=(47.5, "test")),
                  patch("risk_manager._fetch_bars_for_atr")):
            st.enter_context(p)
        la = st.enter_context(patch("db.log_activity", return_value=True))
        return c6_shadow.run(payload, client), la


def _unrouted(trigger):
    r = {"ticker": "MSFT", "action": "WATCH", "conviction": 6, "conviction_raw": 6,
         "pass_to_risk": False, "thesis_bucket": "Compounder",
         "provenance": {"signal_id": 41, "trigger": trigger,
                        "c6_bar": {"model_action": "OPEN", "model_conviction": 6, "action": "OPEN",
                                   "c7_gates": ["regime"], "ref_price": 50.0}}}
    with patch("tasks.c6_bar_shadow.apply_async") as aa, patch("config.C6_BAR_SHADOW_ENABLED", True):
        c6_shadow.observe_unrouted(r, {"regime": "Neutral"})
    return aa.call_args.kwargs["args"][0]


def test_c6_shadow_names_no_new_fact():
    payload = _unrouted(SCAN)
    assert payload["trigger"] == {"path": "daily_scan", "new_fact": None}
    rec, _ = _c6_run(payload, _c6_client(100_000.0))
    assert rec["would_queue"] is False and rec["blocked_also_by"] == ["no_new_fact"]
    assert rec["new_fact_gate"]["passed"] is False


def test_c6_shadow_new_fact_item_not_blocked():
    rec, _ = _c6_run(_unrouted(EVENT_NEW), _c6_client(100_000.0))
    assert rec["would_queue"] is True and "no_new_fact" not in rec["blocked_also_by"]
    assert rec["new_fact_gate"]["passed"] is True


def test_c6_shadow_gate_off_records_no_gate():
    rec, _ = _c6_run(_unrouted(SCAN), _c6_client(100_000.0), on=False)
    assert "new_fact_gate" not in rec and rec["would_queue"] is True


def test_route_block_of_a_c6_buy_reaches_the_shadow():
    """A c6 buy the gate stops before sizing never reaches the tier floor:
    it's enqueued, and written with tier_floor on a micro account."""
    result = _result("OPEN", SCAN, conviction=6)
    result["ticker"], result["thesis_bucket"] = "MSFT", "Compounder"
    with ExitStack() as st:
        for p in _cfg():
            st.enter_context(p)
        st.enter_context(patch("config.C6_BAR_SHADOW_ENABLED", True))
        aa = st.enter_context(patch("tasks.c6_bar_shadow.apply_async"))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("db.log_signal"))
        crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
    payload = aa.call_args.kwargs["args"][0]
    assert payload["gate_blocked"] == "no_new_fact" and payload["c7_gates"] == []
    rec, la = _c6_run(payload, _c6_client(3_000.0))
    assert rec["c7_gates"] == ["tier_floor"] and "no_new_fact" in rec["blocked_also_by"]
    assert rec["would_queue"] is False and la.called
    rec2, la2 = _c6_run(payload, _c6_client(100_000.0))           # no c7 bar at standard
    assert rec2 is None and not la2.called


# ── setting validation ────────────────────────────────────────────────────────

@pytest.fixture
def fresh_alerts(monkeypatch):
    monkeypatch.setattr(crew, "_NEW_FACT_SETTING_ALERTED", set())
    with patch("alerts._send") as send:
        yield send


def test_typo_in_buy_requires_new_fact_keeps_the_gate_on(monkeypatch):
    import importlib
    import config
    try:
        monkeypatch.setenv("BUY_REQUIRES_NEW_FACT", "ture")
        importlib.reload(config)
        assert config.BUY_REQUIRES_NEW_FACT is True and config.BUY_REQUIRES_NEW_FACT_RAW == "ture"
        monkeypatch.setenv("BUY_REQUIRES_NEW_FACT", "False")
        importlib.reload(config)
        assert config.BUY_REQUIRES_NEW_FACT is False
    finally:
        monkeypatch.delenv("BUY_REQUIRES_NEW_FACT", raising=False)
        importlib.reload(config)
        monkeypatch.setattr(config, "BUY_REQUIRES_NEW_FACT", False)   # the conftest default


def test_bad_buy_requires_value_alerts_once(fresh_alerts):
    import alert_policy
    with patch("config.BUY_REQUIRES_NEW_FACT_RAW", "ture"):
        g1, _ = _gate("OPEN", SCAN)
        g2, _ = _gate("ADD", SCAN)
    assert g1["passed"] is False and g2["passed"] is False      # gate stays on
    assert fresh_alerts.call_count == 1
    text = fresh_alerts.call_args.args[0]
    assert "BUY_REQUIRES_NEW_FACT='ture'" in text and "true (gate on)" in text
    assert fresh_alerts.call_args.kwargs["severity"] == alert_policy.LOW


def test_bad_scope_and_window_alert_once_each(fresh_alerts):
    for _ in range(3):
        g, _ = _gate("OPEN", SCAN, scope="news_onyl", window="1d")
    assert (g["scope"], g["window"], g["passed"]) == ("all_buys", "same_run", False)
    texts = [c.args[0] for c in fresh_alerts.call_args_list]
    assert len(texts) == 2
    assert any("BUY_NEW_FACT_SCOPE='news_onyl'" in t for t in texts)
    assert any("BUY_NEW_FACT_WINDOW='1d'" in t for t in texts)


def test_valid_settings_send_no_alert(fresh_alerts):
    for scope in ("all_buys", "news_only", "all_but_held_review_add"):
        for window in ("same_run", "24h", "3d"):
            _gate("OPEN", EVENT_NEW, scope=scope, window=window)
    fresh_alerts.assert_not_called()


def test_alert_failure_never_reaches_routing(monkeypatch):
    monkeypatch.setattr(crew, "_NEW_FACT_SETTING_ALERTED", set())
    with patch("alerts._send", side_effect=RuntimeError("telegram down")):
        g, _ = _gate("OPEN", EVENT_NEW, scope="bogus")
    assert g["passed"] is True


# ── more routing ──────────────────────────────────────────────────────────────

def test_gate_on_route_through_a_swap_and_pass():
    result = _result("ADD", EVENT_NEW)
    with ExitStack() as st:
        for p in _cfg("news_only", "24h"):
            st.enter_context(p)
        st.enter_context(patch("db.recent_new_fact_items", return_value=[]))
        for name, value in (("db.has_active_approval", False), ("db.get_last_executed_approval", None),
                            ("db.is_near_earnings", False), ("agents.crew._get_price", 100.0),
                            ("market_data.is_market_hours", True),
                            ("agents.crew._wait_for_swap_cash", True),
                            ("catalyst_service.gates.catalyst_gate_check", {"approved": True})):
            st.enter_context(patch(name, return_value=value))
        sell = st.enter_context(patch("execution.execute_manual",
                                      return_value={"success": True, "filled_qty": 1,
                                                    "fill_price": 300.0}))
        st.enter_context(patch("alerts._send"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        rm = st.enter_context(patch("risk_manager.RiskManager")).return_value
        rm.size_position.side_effect = [
            {"shares_initial": 0, "cash_blocked": True, "flags": ["cash"], "position_size_pct": 0},
            {"shares_initial": 2, "initial_dollar": 200.0, "position_size_pct": 0.02,
             "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 90.0}]
        rm.find_swap_candidate.return_value = {"ticker": "OLD", "conviction": 4, "qty": 1,
                                               "market_value": 300.0, "unrealized_pl": -5.0}
        rm.would_size_after_swap.return_value = 2
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
    sell.assert_called_once()
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 2 and risk["new_fact_gate"]["passed"] is True
    assert not [c for c in ls.call_args_list if c.args[1] == "ROUTE_NO_NEW_FACT"]


@pytest.mark.parametrize("action", ["HOLD", "WATCH"])
def test_hold_and_watch_pass_through_untouched(action):
    """A HOLD/WATCH verdict isn't routed and gets no gate row, gate on."""
    verdict = {"action": action, "ticker": "NVDA", "conviction": 6, "pass_to_risk": False,
               "provenance": {"trigger": SCAN}}
    with ExitStack() as st:
        for p in _cfg():
            st.enter_context(p)
        st.enter_context(patch("ibkr.IBKRClient", side_effect=RuntimeError("no broker")))
        st.enter_context(patch("agents.analyst.analyze", return_value=dict(verdict)))
        st.enter_context(patch("agents.market_regime.get_cached_regime",
                               return_value={"regime": "Neutral"}))
        st.enter_context(patch("knowledge_loader.get_watchlist_entry", return_value={}))
        route = st.enter_context(patch("agents.crew._route_to_approval"))
        unrouted = st.enter_context(patch("agents.crew._c6_bar_unrouted"))
        ls = st.enter_context(patch("db.log_signal"))
        out = crew.deep_analyze({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"},
                                {"relevant": True, "new_fact": False})
    route.assert_not_called()
    unrouted.assert_called_once()
    ls.assert_not_called()
    assert out == verdict


@pytest.mark.parametrize("items, blocked", [(ITEM, False), ([], True), (None, True)])
def test_c6_shadow_uses_the_24h_lookup(items, blocked):
    with patch("db.recent_new_fact_items", return_value=items) as lookup, \
         patch("config.BUY_NEW_FACT_WINDOW", "24h"):
        payload = _unrouted(EVENT_OLD)
        with patch("config.BUY_REQUIRES_NEW_FACT", True):
            rec, _ = _c6_run_window(payload)
    assert lookup.call_args.args == ("MSFT", 24)
    assert ("no_new_fact" in rec["blocked_also_by"]) is blocked
    assert rec["new_fact_gate"]["window"] == "24h"
    if not blocked:
        assert rec["new_fact_gate"]["window_items"] == ITEM


def _c6_run_window(payload):
    with patch("config.BUY_NEW_FACT_SCOPE", "all_buys"):
        with ExitStack() as st:
            for p in (patch("config.C6_BAR_SHADOW_ENABLED", True),
                      patch("config.SETUP_GATES_ENABLED", False),
                      patch("risk_manager.SETUP_GATES_ENABLED", False),
                      patch("db.get_state", side_effect=lambda k, d=None: d), patch("db.set_state"),
                      patch("db.get_portfolio_heat_pct", return_value={"heat_pct": 0.0}),
                      patch("db.get_price_history", return_value=[]),
                      patch("db.get_price_history_bulk", return_value={}),
                      patch("risk_manager.RiskManager.check_concentration",
                            return_value={"passed": True, "flags": []}),
                      patch("risk_manager.RiskManager.calculate_stop", return_value=(47.5, "test")),
                      patch("risk_manager._fetch_bars_for_atr")):
                st.enter_context(p)
            la = st.enter_context(patch("db.log_activity", return_value=True))
            payload = dict(payload, prechecks=False)
            return c6_shadow.run(payload, _c6_client(100_000.0)), la


def test_trigger_carries_the_screener_digest_and_template_sha():
    seen = {}

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        seen["trigger"] = trigger
        return {"action": "HOLD", "ticker": "NVDA", "conviction": 5, "pass_to_risk": False}
    screening = {"relevant": True, "new_fact": True,
                 "provenance": {"model_digest": "abc123def456", "prompt_template_sha": "0123456789ab"}}
    with patch("ibkr.IBKRClient", side_effect=RuntimeError("no broker")), \
         patch("agents.analyst.analyze", fake_analyze), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("agents.crew._c6_bar_unrouted"):
        crew.deep_analyze({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"}, screening)
    t = seen["trigger"]
    assert t["new_fact"] is True
    assert (t["screener_model_digest"], t["screener_prompt_template_sha"]) == \
        ("abc123def456", "0123456789ab")
