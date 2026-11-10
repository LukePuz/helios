"""Micro-tier per-name cap on new buys (config.MICRO_TIER_NAME_CAP_PCT):
an OPEN/ADD is cut so held value + the order stays at or under the cap of
equity, or blocked when even 1 share would exceed it. Never sells. Other
tiers and the flag off keep the old behaviour. No live DB/IBKR/Redis."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import risk_manager
from agents import crew

MICRO_PV = 2_000.0          # under config.ACCOUNT_TIER_REDUCED_MIN


@pytest.fixture(autouse=True)
def _no_stop_records():
    """The cap's held read cross-checks position_stops for a ticker the
    broker list lacks; here no ticker has a stop record unless a test says."""
    with patch("db.get_position_stop", return_value=None):
        yield


def _pos(symbol, value, qty=1.0):
    return SimpleNamespace(symbol=symbol, qty=qty, market_value=value, unrealized_pl=0.0,
                           avg_entry_price=value / qty, current_price=value / qty)


def _client(pv, held=None, cash=None):
    held = held or {}
    c = MagicMock()
    c.get_account.return_value = SimpleNamespace(portfolio_value=pv, cash=cash if cash else pv,
                                                 buying_power=cash if cash else pv)
    c.get_all_positions.return_value = [_pos(s, v) for s, v in held.items()]

    def _open(symbol):
        if symbol in held:
            return _pos(symbol, held[symbol])
        raise Exception(f"No open position in {symbol}")
    c.get_open_position.side_effect = _open
    return c


def _rm(pv, held=None):
    rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
    rm._client = _client(pv, held)
    rm.start_value = pv
    return rm


# ── RiskManager.micro_name_cap ────────────────────────────────────────────────

def test_shrinks_to_the_cap():
    rec = _rm(MICRO_PV, {"MSFT": 400.0}).micro_name_cap("MSFT", 5, 100.0)
    # 30% of $2,000 = $600; $400 held leaves room for 2 shares at $100.
    assert (rec["outcome"], rec["shares_after"], rec["max_shares"]) == ("shrunk", 2, 2)
    assert rec["cap_pct"] == 30.0 and rec["tier"] == "micro" and rec["held_pct"] == 20.0


def test_blocks_when_one_share_exceeds_it():
    rec = _rm(MICRO_PV, {"MSFT": 560.0}).micro_name_cap("MSFT", 1, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("blocked", 0)


def test_new_name_within_the_cap_is_unchanged():
    rec = _rm(MICRO_PV).micro_name_cap("AMD", 3, 150.0)
    assert (rec["outcome"], rec["shares_after"]) == ("within_cap", 3)


def test_existing_position_over_the_cap_gets_no_new_shares_and_is_not_sold():
    rm = _rm(MICRO_PV, {"MSFT": 700.0})              # 35%, already over
    rec = rm.micro_name_cap("MSFT", 2, 100.0)
    assert (rec["outcome"], rec["shares_after"], rec["max_shares"]) == ("blocked", 0, 0)
    assert not [c for c in rm._client.method_calls if "order" in c[0] or "sell" in c[0]]


def test_other_tiers_are_not_capped():
    assert _rm(10_000.0, {"MSFT": 5_000.0}).micro_name_cap("MSFT", 5, 100.0) is None
    assert _rm(100_000.0, {"MSFT": 50_000.0}).micro_name_cap("MSFT", 5, 100.0) is None


def test_flag_off_is_the_old_behaviour():
    with patch("config.MICRO_TIER_NAME_CAP_ENABLED", False):
        assert _rm(MICRO_PV, {"MSFT": 700.0}).micro_name_cap("MSFT", 2, 100.0) is None


def test_cap_pct_comes_from_config():
    with patch("config.MICRO_TIER_NAME_CAP_PCT", 50.0):
        rec = _rm(MICRO_PV, {"MSFT": 700.0}).micro_name_cap("MSFT", 5, 100.0)
    assert rec["shares_after"] == 3 and rec["cap_pct"] == 50.0


def test_unreadable_account_blocks():
    rm = _rm(MICRO_PV)
    rm._client.get_account.side_effect = RuntimeError("IBKR down")
    rec = rm.micro_name_cap("MSFT", 2, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)


# ── crew routing ──────────────────────────────────────────────────────────────

def _sizing(shares=5, price=100.0):
    return {"shares_initial": shares, "initial_dollar": shares * price, "position_size_pct": 0.25,
            "order_pct": round(shares * price / MICRO_PV, 4), "initial_entry_pct": 0.55,
            "flags": [], "stop_loss_price": price * 0.9}


def _route(pv, held, sizing, action="ADD", flag=True):
    """crew._route_to_approval with sizing/concentration mocked and the real
    micro_name_cap on an account worth pv holding `held`."""
    real = _rm(pv, held)
    result = {"ticker": "MSFT", "action": action, "conviction": 8,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", flag))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=100.0))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        sell = st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.side_effect = real.micro_name_cap
        crew._route_to_approval(result, {"regime": "Neutral"}, _client(pv, held))
    return lpa, ls, sell


def test_route_shrinks_the_order_and_records_the_cap():
    lpa, ls, _ = _route(MICRO_PV, {"MSFT": 400.0}, _sizing(5))
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 2 and risk["initial_dollar"] == 200.0
    assert risk["order_pct"] == 0.1
    assert lpa.call_args.kwargs["size_pct"] == pytest.approx(0.10)    # 30% - 20% held
    assert risk["micro_name_cap"]["outcome"] == "shrunk"
    assert any("micro-tier 30% name cap" in f for f in risk["flags"])


def test_route_blocks_with_a_route_row_and_never_sells():
    lpa, ls, sell = _route(MICRO_PV, {"MSFT": 558.0}, _sizing(1))     # the 11-03 MSFT case
    lpa.assert_not_called()
    sell.assert_not_called()
    assert ls.call_args.args[1] == "ROUTE_MICRO_NAME_CAP"
    assert "micro-tier name cap 30%" in ls.call_args.args[3]
    decision = ls.call_args.kwargs["decision"]
    assert decision["route_exit"] == "ROUTE_MICRO_NAME_CAP"
    assert decision["micro_name_cap"]["outcome"] == "blocked"


def test_route_within_the_cap_records_and_keeps_size():
    lpa, _, _ = _route(MICRO_PV, {}, _sizing(3))
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 3 and risk["micro_name_cap"]["outcome"] == "within_cap"


def test_route_on_a_standard_account_is_unchanged():
    lpa, _, _ = _route(100_000.0, {"MSFT": 50_000.0}, _sizing(5))
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 5 and "micro_name_cap" not in risk


def test_route_flag_off_is_unchanged():
    lpa, _, _ = _route(MICRO_PV, {"MSFT": 700.0}, _sizing(5), flag=False)
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 5 and "micro_name_cap" not in risk


def test_route_sells_are_not_capped():
    real = _rm(MICRO_PV, {"MSFT": 1_000.0})
    with patch("db.has_active_approval", return_value=False), \
         patch("db.log_pending_approval", return_value=-1) as lpa, \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("risk_manager.RiskManager") as rm_cls:
        rm_cls.return_value.micro_name_cap.side_effect = real.micro_name_cap
        crew._route_to_approval({"ticker": "MSFT", "action": "TRIM", "conviction": 5,
                                 "thesis_bucket": "Compounder", "reasoning": "r"},
                                {"regime": "Neutral"}, _client(MICRO_PV, {"MSFT": 1_000.0}))
    rm_cls.return_value.micro_name_cap.assert_not_called()
    assert lpa.call_args.kwargs["action"] == "TRIM"


# ── swap path ─────────────────────────────────────────────────────────────────

def _swap_precheck(held, flag=True):
    rm = _rm(MICRO_PV, held)
    with patch("config.MICRO_TIER_NAME_CAP_ENABLED", flag), \
         patch.object(risk_manager.RiskManager, "size_position",
                      return_value={"shares_initial": 4, "position_size_pct": 0.2,
                                    "order_pct": 0.2}), \
         patch.object(risk_manager.RiskManager, "check_concentration",
                      return_value={"passed": True, "flags": []}), \
         patch("db.get_state", return_value=None), patch("db.set_state"):
        return rm.would_size_after_swap("MSFT", 8, "Compounder", "Neutral", 100.0,
                                        {"ticker": "OLD", "market_value": 300.0})


def test_swap_precheck_applies_the_cap():
    # The swap sells OLD; MSFT stays at $400 (20%): 1 more share fits under 29.5%.
    assert _swap_precheck({"MSFT": 400.0, "OLD": 300.0}) == 1


def test_swap_precheck_keeps_half_a_point_of_slack():
    """A buy that lands exactly on the cap (25% + one $100 share = 30%) passes
    the route's cap, but the pre-check sizes at 29.5%, so no holding is sold
    for it."""
    assert _swap_precheck({"MSFT": 500.0, "OLD": 300.0}) == 0
    rec = _rm(MICRO_PV, {"MSFT": 500.0}).micro_name_cap("MSFT", 4, 100.0)
    assert rec["shares_after"] == 1


def test_swap_precheck_blocks_when_the_cap_leaves_no_room():
    # No holding is sold for a buy the cap would block.
    assert _swap_precheck({"MSFT": 560.0, "OLD": 300.0}) == 0


def test_swap_precheck_flag_off_is_unchanged():
    assert _swap_precheck({"MSFT": 560.0, "OLD": 300.0}, flag=False) == 4


def test_post_swap_resize_is_capped_too():
    """After the swap sale crew re-sizes and the same tail (concentration,
    correlation, name cap) bounds the order."""
    real = _rm(MICRO_PV, {"MSFT": 400.0})
    result = {"ticker": "MSFT", "action": "ADD", "conviction": 8,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    with patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("db.is_near_earnings", return_value=False), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("agents.crew._wait_for_swap_cash", return_value=True), \
         patch("execution.execute_manual",
               return_value={"success": True, "filled_qty": 1, "fill_price": 300.0}) as sell, \
         patch("alerts._send"), \
         patch("db.log_pending_approval", return_value=-1) as lpa, \
         patch("db.log_signal"), \
         patch("risk_manager.RiskManager") as rm_cls:
        rm = rm_cls.return_value
        rm.size_position.side_effect = [
            {"shares_initial": 0, "cash_blocked": True, "flags": ["cash"], "position_size_pct": 0},
            _sizing(5)]
        rm.find_swap_candidate.return_value = {"ticker": "OLD", "conviction": 4, "qty": 1,
                                               "market_value": 300.0, "unrealized_pl": -5.0}
        rm.would_size_after_swap.return_value = 2
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.side_effect = real.micro_name_cap
        crew._route_to_approval(result, {"regime": "Neutral"}, _client(MICRO_PV))
    sell.assert_called_once()
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 2 and risk["micro_name_cap"]["outcome"] == "shrunk"


# ── deferred entry re-size ────────────────────────────────────────────────────

def test_deferred_resize_applies_the_cap():
    import tasks
    real = _rm(MICRO_PV, {"SNOW": 500.0})
    real.size_position = MagicMock(return_value={"shares_initial": 4, "position_size_pct": 0.2,
                                                 "order_pct": 0.2})
    real.check_concentration = MagicMock(return_value={"passed": True, "flags": []})
    approval = {"id": 7, "ticker": "SNOW", "action": "ADD", "thesis_bucket": "Accelerator",
                "conviction": 7, "risk_json": {"shares_initial": 4, "price_at_sizing": 50.0}}
    with patch("ibkr.get_client"), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("risk_manager.RiskManager", return_value=real):
        assert tasks._deferred_buy_shares_now(approval) == 2      # ($600 - $500) / $50


# ── c6 shadow ─────────────────────────────────────────────────────────────────

def _c6_run(held, flag=True):
    from agents import c6_shadow
    payload = {"ticker": "MSFT", "env": "paper", "signal_id": 41, "as_of": "2026-11-03T15:00:00+00:00",
               "session": "regular", "bucket": "Compounder", "regime": "Risk-On",
               "model_action": "OPEN", "model_conviction": 6, "conviction_raw": 6,
               "fundamentals_read": "ok", "action": "ADD", "conviction": 6, "c7_gates": ["regime"],
               "entry_ref_price": 50.0, "ref_price_source": "brief", "prechecks": False,
               "shadow_key": "paper:MSFT:s41:c6"}
    with patch("config.MICRO_TIER_NAME_CAP_ENABLED", flag), \
         patch("config.C6_BAR_SHADOW_ENABLED", True), \
         patch("config.SETUP_GATES_ENABLED", False), \
         patch("risk_manager.SETUP_GATES_ENABLED", False), \
         patch("db.get_state_strict", side_effect=lambda k, d=None: d), \
         patch("db.get_state", side_effect=lambda k, d=None: d), \
         patch("db.set_state"), \
         patch("db.get_portfolio_heat_pct", return_value={"heat_pct": 0.0}), \
         patch("db.get_price_history_bulk", return_value={}), \
         patch("risk_manager.RiskManager.check_concentration",
               return_value={"passed": True, "flags": []}), \
         patch("risk_manager.RiskManager.size_position",
               return_value={"shares_initial": 3, "position_size_pct": 0.05, "order_pct": 0.05,
                             "flags": []}), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
         patch("db.log_activity", return_value=True), \
         patch("risk_manager._fetch_bars_for_atr"):
        return c6_shadow.run(payload, _client(3_000.0, held))


def test_c6_shadow_names_the_cap_when_it_blocks():
    rec = _c6_run({"MSFT": 870.0})                       # 29%: one $50 share goes over
    assert rec["would_queue"] is False and "micro_name_cap" in rec["blocked_also_by"]
    assert rec["micro_name_cap"]["outcome"] == "blocked" and rec["intended_shares"] == 0


def test_c6_shadow_flag_off_unchanged():
    rec = _c6_run({"MSFT": 870.0}, flag=False)
    assert "micro_name_cap" not in rec and "micro_name_cap" not in rec["blocked_also_by"]
    assert rec["would_queue"] is True and rec["intended_shares"] >= 1


# ── no account dollar values recorded ─────────────────────────────────────────

_RECORD_KEYS = {"cap_pct", "tier", "held_pct", "price", "shares_before", "max_shares",
                "shares_after", "outcome", "held_source"}


def _no_dollars(rec):
    assert set(rec) == _RECORD_KEYS, set(rec) ^ _RECORD_KEYS
    assert "equity" not in rec and "held_value" not in rec


def test_cap_records_hold_no_account_dollar_values(caplog):
    import logging
    for held, shares in (({"MSFT": 400.0}, 5), ({"MSFT": 560.0}, 1), ({}, 3)):
        _no_dollars(_rm(MICRO_PV, held).micro_name_cap("MSFT", shares, 100.0))
    rm = _rm(MICRO_PV)
    rm._client.get_account.side_effect = RuntimeError("account U1234 net liq $1,999.00")
    with caplog.at_level(logging.DEBUG):
        _no_dollars(rm.micro_name_cap("MSFT", 2, 100.0))
    assert "1,999" not in caplog.text and "U1234" not in caplog.text


def test_route_records_hold_no_account_dollar_values(caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        lpa, _, _ = _route(MICRO_PV, {"MSFT": 400.0}, _sizing(5))
        _, ls, _ = _route(MICRO_PV, {"MSFT": 558.0}, _sizing(1))
    _no_dollars(lpa.call_args.kwargs["risk_json"]["micro_name_cap"])
    _no_dollars(ls.call_args.kwargs["decision"]["micro_name_cap"])
    reason = ls.call_args.args[3]
    assert "2,000" not in reason and "558" not in reason and "27.9% of equity" in reason
    assert "2,000" not in caplog.text and "558" not in caplog.text


def test_c6_shadow_record_holds_no_account_dollar_values():
    _no_dollars(_c6_run({"MSFT": 870.0})["micro_name_cap"])


def test_approval_spill_keeps_dropping_account_keys():
    import db
    lpa, _, _ = _route(MICRO_PV, {"MSFT": 400.0}, _sizing(5))
    risk = dict(lpa.call_args.kwargs["risk_json"], cash_blocked=False, equity_now=1.0)
    rec = db._approval_spill_record("MSFT", "ADD", 8, "Compounder", 0.1, 90.0, "e", "r",
                                    {}, risk, None, None, False)
    assert "cash_blocked" not in rec["risk_json"] and "equity_now" not in rec["risk_json"]
    assert "flags" not in rec["risk_json"]
    _no_dollars(rec["risk_json"]["micro_name_cap"])
    assert "held_value" not in str(rec) and "'equity'" not in str(rec)


# ── fail closed on unusable reads ─────────────────────────────────────────────

def test_held_value_read_error_is_errored():
    rm = _rm(MICRO_PV, {"MSFT": 400.0})
    rm._client.get_all_positions.side_effect = RuntimeError("positions unreadable")
    rec = rm.micro_name_cap("MSFT", 3, 100.0)
    assert (rec["outcome"], rec["shares_after"], rec["max_shares"]) == ("errored", 0, 0)


def test_current_position_value_still_reads_errors_as_zero():
    rm = _rm(MICRO_PV)
    rm._client.get_open_position.side_effect = RuntimeError("down")
    assert rm.current_position_value("MSFT") == 0.0


@pytest.mark.parametrize("price", [float("nan"), 0.0, -5.0, float("inf")])
def test_unusable_price_is_errored(price):
    rec = _rm(MICRO_PV).micro_name_cap("MSFT", 3, price)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)


@pytest.mark.parametrize("value", [float("nan"), 0.0, -100.0])
def test_unusable_market_value_is_errored(value):
    rec = _rm(MICRO_PV, {"MSFT": value}).micro_name_cap("MSFT", 3, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)


@pytest.mark.parametrize("pv", [0.0, -50.0, float("nan")])
def test_unusable_equity_is_errored_not_blocked(pv):
    rec = _rm(pv).micro_name_cap("MSFT", 3, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)


def test_errored_cap_defers_the_deferred_entry():
    """An errored cap neither executes nor expires the deferred buy: it waits
    for a retry."""
    import tasks
    real = _rm(MICRO_PV, {"SNOW": 500.0})
    real._client.get_all_positions.side_effect = RuntimeError("positions unreadable")
    real.size_position = MagicMock(return_value={"shares_initial": 4, "position_size_pct": 0.2,
                                                 "order_pct": 0.2})
    real.check_concentration = MagicMock(return_value={"passed": True, "flags": []})
    approval = {"id": 7, "ticker": "SNOW", "action": "ADD", "thesis_bucket": "Accelerator",
                "conviction": 7, "status": "FAILED", "timestamp": None,
                "risk_json": {"shares_initial": 4, "price_at_sizing": 50.0}}
    with patch("ibkr.get_client"), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("risk_manager.RiskManager", return_value=real):
        shares_now = tasks._deferred_buy_shares_now(approval)
    assert shares_now == tasks._RESIZE_UNREADABLE
    from datetime import datetime, timedelta, timezone
    approval["timestamp"] = datetime.now(timezone.utc) - timedelta(hours=3)
    with patch("db.claim_deferred_entry_for_retry", return_value=True), \
         patch("db.get_approval_by_id", return_value=approval), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=shares_now), \
         patch("db.update_approval_status") as status, \
         patch.object(tasks.retry_deferred_entry, "apply_async") as retry, \
         patch("execution.execute_approved", return_value=True) as execute:
        tasks.retry_deferred_entry.run(7)
    status.assert_not_called()
    execute.assert_not_called()
    retry.assert_called_once()


def test_post_swap_cap_block_sends_swap_incomplete():
    """The swap sold a holding, the post-sale cap leaves 0 shares: the same
    SWAP INCOMPLETE alert as a post-sale 0-share sizing."""
    real = _rm(MICRO_PV, {"MSFT": 600.0})                      # at the cap
    result = {"ticker": "MSFT", "action": "ADD", "conviction": 8,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    with patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("db.is_near_earnings", return_value=False), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("agents.crew._wait_for_swap_cash", return_value=True), \
         patch("execution.execute_manual",
               return_value={"success": True, "filled_qty": 1, "fill_price": 300.0}) as sell, \
         patch("alerts._send") as send, \
         patch("db.log_pending_approval", return_value=-1) as lpa, \
         patch("db.log_signal") as ls, \
         patch("risk_manager.RiskManager") as rm_cls:
        rm = rm_cls.return_value
        rm.size_position.side_effect = [
            {"shares_initial": 0, "cash_blocked": True, "flags": ["cash"], "position_size_pct": 0},
            _sizing(2)]
        rm.find_swap_candidate.return_value = {"ticker": "OLD", "conviction": 4, "qty": 1,
                                               "market_value": 300.0, "unrealized_pl": -5.0}
        rm.would_size_after_swap.return_value = 2
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.side_effect = real.micro_name_cap
        crew._route_to_approval(result, {"regime": "Neutral"}, _client(MICRO_PV))
    sell.assert_called_once()
    lpa.assert_not_called()
    assert ls.call_args.args[1] == "ROUTE_MICRO_NAME_CAP"
    texts = [c.args[0] for c in send.call_args_list]
    assert any("SWAP INCOMPLETE" in t and "OLD" in t and "name cap" in t for t in texts)


def test_no_swap_no_swap_incomplete_alert():
    with patch("alerts._send") as send:
        _route(MICRO_PV, {"MSFT": 600.0}, _sizing(1))
    assert not [c for c in send.call_args_list if "SWAP INCOMPLETE" in c.args[0]]


def test_crew_does_not_skip_a_non_dict_cap_record():
    """No silent branch: a cap record that isn't None must be a real record."""
    sizing = _sizing(3)
    rm = MagicMock()
    with pytest.raises(TypeError):
        crew._apply_micro_name_cap(rm, "MSFT", sizing, 100.0, {"ticker": "MSFT"})


def test_stop_record_without_a_broker_position_is_errored():
    """An empty broker portfolio right after a reconnect: position_stops
    still lists the name, so the held value is unknown, not 0."""
    rm = _rm(MICRO_PV)                                     # broker lists nothing
    with patch("db.get_position_stop", return_value={"ticker": "MSFT", "entry_price": 500.0}) as ps:
        rec = rm.micro_name_cap("MSFT", 1, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)
    assert ps.call_args.args == ("MSFT",) and ps.call_args.kwargs == {"strict": True}


def test_stop_record_read_failure_is_errored():
    rm = _rm(MICRO_PV)
    with patch("db.get_position_stop", side_effect=RuntimeError("db down")):
        rec = rm.micro_name_cap("MSFT", 1, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)


def test_broker_listed_position_needs_no_stop_record_read():
    rm = _rm(MICRO_PV, {"MSFT": 400.0})
    with patch("db.get_position_stop", side_effect=AssertionError("not read")):
        assert rm.micro_name_cap("MSFT", 5, 100.0)["outcome"] == "shrunk"


# ── held_source and the stop-record alert ─────────────────────────────────────

_STOP = {"ticker": "MSFT", "entry_price": 500.0}


def test_held_source_is_recorded():
    assert _rm(MICRO_PV, {"MSFT": 400.0}).micro_name_cap("MSFT", 1, 100.0)["held_source"] == "broker"
    assert _rm(MICRO_PV).micro_name_cap("MSFT", 1, 100.0)["held_source"] == "broker"
    with patch("db.get_position_stop", return_value=_STOP), patch("alerts._send"), \
         patch("db.get_state", return_value=None), patch("db.set_state"):
        rec = _rm(MICRO_PV).micro_name_cap("MSFT", 1, 100.0)
    assert (rec["held_source"], rec["outcome"]) == ("stop_record_no_broker_position", "errored")
    rm = _rm(MICRO_PV)
    rm._client.get_all_positions.side_effect = RuntimeError("down")
    assert rm.micro_name_cap("MSFT", 1, 100.0)["held_source"] == "error"
    with patch("db.get_position_stop", side_effect=RuntimeError("db down")):
        assert _rm(MICRO_PV).micro_name_cap("MSFT", 1, 100.0)["held_source"] == "error"
    assert _rm(MICRO_PV).micro_name_cap("MSFT", 1, float("nan"))["held_source"] is None


def _stop_record_case(state=None, sent=1, rm=None):
    import alert_policy
    rm = rm or _rm(MICRO_PV)
    with patch("db.get_position_stop", return_value=_STOP), \
         patch("alerts._send", return_value=sent) as send, \
         patch("db.get_state", return_value=state) as gs, \
         patch("db.set_state") as ss:
        rec = rm.micro_name_cap("MSFT", 1, 100.0)
    return rec, send, gs, ss, alert_policy


def test_stop_record_case_alerts_once_per_et_day():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    rec, send, gs, ss, alert_policy = _stop_record_case()
    assert rec["outcome"] == "errored"
    assert send.call_args.args[0] == ("MSFT: buy blocked by the name cap: a stop record exists but "
                                      "the broker shows no position (stale record or a cache gap)")
    assert send.call_args.kwargs["severity"] == alert_policy.LOW
    assert gs.call_args.args[0] == "micro_cap_stop_record_alert_MSFT"
    ss.assert_called_once_with("micro_cap_stop_record_alert_MSFT", today)
    _, send2, _, ss2, _ = _stop_record_case(state=today)          # already sent today
    send2.assert_not_called()
    ss2.assert_not_called()


def test_throttle_written_only_after_a_successful_send():
    rec, send, _, ss, _ = _stop_record_case(sent=None)
    send.assert_called_once()
    ss.assert_not_called()
    assert rec["outcome"] == "errored"


def test_alert_failure_never_changes_the_outcome():
    import alert_policy
    with patch("db.get_position_stop", return_value=_STOP), \
         patch("alerts._send", side_effect=RuntimeError("telegram down")), \
         patch("db.get_state", side_effect=RuntimeError("db down")), patch("db.set_state"):
        rec = _rm(MICRO_PV).micro_name_cap("MSFT", 1, 100.0)
    assert (rec["outcome"], rec["shares_after"]) == ("errored", 0)
    assert rec["held_source"] == "stop_record_no_broker_position"


def test_simulations_send_no_stop_record_alert():
    for attr in ("simulation", "quiet_fallback_alert"):
        rm = _rm(MICRO_PV)
        setattr(rm, attr, True)
        rec, send, _, _, _ = _stop_record_case(rm=rm)
        assert rec["held_source"] == "stop_record_no_broker_position"
        send.assert_not_called()


def test_other_errored_cases_send_no_stop_record_alert():
    rm = _rm(MICRO_PV)
    rm._client.get_all_positions.side_effect = RuntimeError("down")
    with patch("alerts._send") as send:
        assert rm.micro_name_cap("MSFT", 1, 100.0)["held_source"] == "error"
    send.assert_not_called()


def test_bucket_maximum_flag_gets_a_binding_limit_note_on_the_micro_tier():
    """Audit 11-06 21:36 #9: the 40% bucket flag can't describe the 30% name cap."""
    from unittest.mock import MagicMock
    from agents import crew
    rec = {"cap_pct": 30.0, "held_pct": 0.0, "shares_after": 5, "outcome": "within_cap"}
    rm = MagicMock()
    rm.micro_name_cap.return_value = rec
    sizing = {"shares_initial": 5, "flags": ["Size capped at 40% bucket maximum"], "position_size_pct": 0.2}
    assert crew._apply_micro_name_cap(rm, "NVDA", sizing, 100.0, {}) is False
    assert sizing["flags"][0] == "Size capped at 40% bucket maximum"
    assert sizing["flags"][1] == ("Micro tier: the 30% name cap is below the 40% bucket maximum, "
                                  "so the bucket maximum can't bind")
    rm.micro_name_cap.return_value = None   # not micro: no note
    sizing = {"shares_initial": 5, "flags": ["Size capped at 40% bucket maximum"]}
    crew._apply_micro_name_cap(rm, "NVDA", sizing, 100.0, {})
    assert sizing["flags"] == ["Size capped at 40% bucket maximum"]
