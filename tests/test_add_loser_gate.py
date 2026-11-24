"""Add-to-loser gate (agents/crew.py:_add_loser_check, _entry_gate;
tasks._deferred_entry_gate): an ADD below the broker's average cost is skipped
with one ENTRY_GATE_BLOCKED row (entry_gate.add_loser) unless the triggering
news item itself is a new fact; a failed read does not block; OPEN and sells
are untouched; switched off, nothing is read and the rows are as before.
Synthetic data only."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import config
import tasks
from agents import crew


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch):
    # conftest turns the entry gates off elsewhere; only this gate on here.
    # enforce here unless a test says otherwise; production default is shadow
    monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "enforce")
    monkeypatch.setattr(config, "ADD_LOSER_GATE_MIN_PCT", 0.0)
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 0)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", False)
    monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", False)


def _trigger(new_fact=False, path="event", source="news"):
    t = {"path": path, "source": source, "title": "t"}
    if path == "event" and new_fact is not None:
        t["new_fact"] = new_fact
    return t


def _result(action="ADD", new_fact=False, path="event", source="news"):
    return {"ticker": "NVDA", "action": action, "conviction": 7, "thesis_bucket": "Accelerator",
            "reasoning": "r", "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m",
                           "trigger": _trigger(new_fact, path, source)}}


def _dip_result():
    # what tasks.dip_watch queues: deep_analyze_event, so path "event", source
    # "dip_watch", and no screening, so new_fact None
    return _result(new_fact=None, path="event", source="dip_watch")


def _client(avg=100.0, current=72.5, held=True, raises=None):
    c = MagicMock()
    if raises:
        c.get_all_positions.side_effect = raises
    else:
        c.get_all_positions.return_value = (
            [SimpleNamespace(symbol="NVDA", avg_entry_price=avg, current_price=current, qty=1)]
            if held else [SimpleNamespace(symbol="AMD", avg_entry_price=10.0, current_price=9.0, qty=1)])
    return c


def _check(result, client, price=72.5):
    with patch("agents.crew._get_price", return_value=price):
        return crew._add_loser_check("NVDA", result, client)


class TestCheck:
    def test_loser_add_without_new_fact_blocks(self):
        # the 11-13 shape: an event verdict, new_fact False, ~27.5% under the average cost
        rec = _check(_result(new_fact=False), _client())
        assert rec == {"mode": "enforce", "min_pct": 0.0, "avg_cost_read": "ok", "price_read": "quote",
                       "pct_vs_avg_cost": -27.5, "trigger_path": "event", "trigger_source": "news",
                       "trigger_new_fact": False, "trigger_new_fact_source": "event_provenance",
                       "screener_fallback_reason": None, "earnings_phase": None,
                       "would_block": True, "blocked": True}

    def test_triggering_new_fact_passes(self):
        rec = _check(_result(new_fact=True), _client())
        assert rec["blocked"] is False and rec["trigger_new_fact"] is True and rec["pct_vs_avg_cost"] == -27.5

    def test_missing_new_fact_is_unjudged_and_does_not_block(self):
        rec = _check(_result(new_fact=None), _client())
        assert rec["trigger_new_fact"] is None and rec["trigger_new_fact_source"] == "unjudged"
        assert rec["would_block"] is False and rec["blocked"] is False

    @pytest.mark.parametrize("path", ["held_review", "dip_watch", None])
    def test_no_news_item_is_no_new_fact(self, path):
        res = _result(path=path)
        res["provenance"]["trigger"]["new_fact"] = True      # ignored off the news path
        rec = _check(res, _client())
        assert rec["trigger_new_fact"] is None and rec["blocked"] is True
        assert rec["trigger_new_fact_source"] == "none_on_path"

    def test_dip_review_below_cost_blocks_with_the_switch_off(self):
        rec = _check(_dip_result(), _client())
        assert rec["trigger_path"] == "event" and rec["trigger_source"] == "dip_watch"
        assert rec["trigger_new_fact"] is None and rec["trigger_new_fact_source"] == "no_news_item"
        assert rec["blocked"] is True

    def test_dip_review_below_cost_passes_with_the_switch_on(self, monkeypatch):
        monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", True)
        rec = _check(_dip_result(), _client())
        assert rec["trigger_new_fact"] is True
        assert rec["trigger_new_fact_source"] == "dip_review_counts_as_new_fact"
        assert rec["blocked"] is False and rec["pct_vs_avg_cost"] == -27.5

    @pytest.mark.parametrize("new_fact,path", [(False, "event"), (None, "held_review")])
    def test_switch_on_leaves_other_triggers_alone(self, monkeypatch, new_fact, path):
        monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", True)
        rec = _check(_result(new_fact=new_fact, path=path), _client())
        assert rec["blocked"] is True
        assert rec["trigger_new_fact_source"] == ("event_provenance" if path == "event" else "none_on_path")

    def test_dip_source_off_the_event_path_is_not_a_dip_review(self, monkeypatch):
        monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", True)
        rec = _check(_result(new_fact=None, path="held_review", source="dip_watch"), _client())
        assert rec["trigger_new_fact_source"] == "none_on_path" and rec["blocked"] is True

    @pytest.mark.parametrize("price,blocked", [(100.0, False), (130.0, False), (99.99, True)])
    def test_at_or_above_cost_passes(self, price, blocked):
        assert _check(_result(), _client(), price=price)["blocked"] is blocked

    @pytest.mark.parametrize("pct,blocked", [(-4.9, False), (-5.0, False), (-5.1, True)])
    def test_min_pct(self, monkeypatch, pct, blocked):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MIN_PCT", 5.0)
        rec = _check(_result(), _client(), price=100.0 * (1 + pct / 100))
        assert rec["blocked"] is blocked and rec["min_pct"] == 5.0

    def test_position_read_error_fails_open(self):
        rec = _check(_result(), _client(raises=ConnectionError("gateway down")))
        assert rec["blocked"] is False and rec["avg_cost_read"] == "error: ConnectionError"
        assert rec["pct_vs_avg_cost"] is None

    def test_not_held_fails_open(self):
        rec = _check(_result(), _client(held=False))
        assert rec["blocked"] is False and rec["avg_cost_read"] == "not_held"

    @pytest.mark.parametrize("avg", [None, 0, "x"])
    def test_no_average_cost_fails_open(self, avg):
        rec = _check(_result(), _client(avg=avg))
        assert rec["blocked"] is False and rec["avg_cost_read"] == "no_avg_cost"

    def test_no_client_fails_open(self):
        rec = _check(_result(), None)
        assert rec["blocked"] is False and rec["avg_cost_read"] == "no_client"

    def test_price_falls_back_to_the_position(self):
        rec = _check(_result(), _client(current=80.0), price=None)
        assert rec["price_read"] == "position" and rec["pct_vs_avg_cost"] == -20.0 and rec["blocked"]

    def test_no_price_fails_open(self):
        rec = _check(_result(), _client(current=None), price=None)
        assert rec["blocked"] is False and rec["price_read"] == "none"

    def test_raising_quote_falls_back(self):
        with patch("agents.crew._get_price", side_effect=RuntimeError("feed")):
            rec = crew._add_loser_check("NVDA", _result(), _client(current=80.0))
        assert rec["price_read"] == "position" and rec["blocked"] is True

    def test_open_and_off_are_none(self, monkeypatch):
        client = _client()
        assert _check(_result(action="OPEN"), client) is None
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        assert _check(_result(), client) is None
        client.get_all_positions.assert_not_called()

    def test_record_has_no_prices(self):
        rec = _check(_result(), _client())
        assert set(rec) == {"mode", "min_pct", "avg_cost_read", "price_read", "pct_vs_avg_cost",
                            "trigger_path", "trigger_source", "trigger_new_fact",
                            "trigger_new_fact_source", "screener_fallback_reason", "earnings_phase",
                            "would_block", "blocked"}


def _gate(add_loser, result=None):
    with patch("db.log_signal") as ls:
        out = crew._entry_gate("NVDA", result or _result(), None, None, add_loser=add_loser)
    return out, [c for c in ls.call_args_list if c[0][1] == "ENTRY_GATE_BLOCKED"]


class TestEntryGate:
    def test_blocked_record_writes_one_row(self):
        rec = _check(_result(), _client())
        out, rows = _gate(rec)
        assert out is True and len(rows) == 1
        assert rows[0][1]["decision"]["entry_gate"]["add_loser"] == rec
        assert "-27.5% vs the average cost" in rows[0][0][3]
        assert "the triggering item not a new fact" in rows[0][0][3]

    def test_passing_record_writes_nothing(self):
        out, rows = _gate(_check(_result(new_fact=True), _client()))
        assert out is False and rows == []

    def test_no_record_is_the_old_gate(self):
        out, rows = _gate(None)
        assert out is False and rows == []


def _route(action="ADD", new_fact=False, client=None, price=72.5):
    result = _result(action=action, new_fact=new_fact)
    client = client or _client()
    with patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("db.get_last_broker_exit", return_value=None), \
         patch("agents.crew._get_price", return_value=price), \
         patch("agents.crew.new_fact_gate", return_value=None), \
         patch("db.log_signal") as ls, \
         patch("risk_manager.RiskManager") as rm:
        rm.return_value.size_position.return_value = {
            "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
        crew._route_to_approval(result, {"regime": "Neutral"}, client)
    return rm.return_value.size_position.call_count, ls.call_args_list


def _kinds(calls):
    return [c[0][1] for c in calls]


class TestRouting:
    def test_loser_add_is_skipped_before_sizing(self):
        sized, calls = _route()
        assert sized == 0
        rows = [c for c in calls if c[0][1] == "ENTRY_GATE_BLOCKED"]
        assert len(rows) == 1
        al = rows[0][1]["decision"]["entry_gate"]["add_loser"]
        assert al["blocked"] is True and al["pct_vs_avg_cost"] == -27.5 and al["trigger_new_fact"] is False

    def test_same_add_with_a_new_fact_trigger_is_sized(self):
        sized, calls = _route(new_fact=True)
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in _kinds(calls)

    def test_dip_review_add_follows_the_switch(self, monkeypatch):
        def go():
            res = _dip_result()
            with patch("db.has_active_approval", return_value=False), \
                 patch("db.get_last_executed_approval", return_value=None), \
                 patch("db.get_last_broker_exit", return_value=None), \
                 patch("agents.crew._get_price", return_value=72.5), \
                 patch("agents.crew.new_fact_gate", return_value=None), \
                 patch("db.log_signal") as ls, \
                 patch("risk_manager.RiskManager") as rm:
                rm.return_value.size_position.return_value = {
                    "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
                crew._route_to_approval(res, {"regime": "Neutral"}, _client())
            return rm.return_value.size_position.call_count, ls.call_args_list
        sized, calls = go()
        rows = [c for c in calls if c[0][1] == "ENTRY_GATE_BLOCKED"]
        assert sized == 0 and len(rows) == 1
        assert rows[0][1]["decision"]["entry_gate"]["add_loser"]["trigger_source"] == "dip_watch"
        monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", True)
        sized, calls = go()
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in _kinds(calls)

    def test_add_above_cost_is_sized(self):
        sized, calls = _route(price=110.0)
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in _kinds(calls)

    def test_read_error_is_sized(self):
        sized, calls = _route(client=_client(raises=ConnectionError("down")))
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in _kinds(calls)

    def test_open_is_not_checked(self):
        with patch("agents.crew._add_loser_check") as chk:
            sized, calls = _route(action="OPEN")
        chk.assert_not_called()
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in _kinds(calls)

    @pytest.mark.parametrize("action", ["TRIM", "EXIT"])
    def test_sells_are_not_checked(self, action):
        with patch("agents.crew._add_loser_check") as chk, \
             patch("db.has_active_approval", return_value=False), \
             patch("agents.crew._get_positions", return_value=None), \
             patch("db.log_signal"), patch("alerts._send"):
            crew._route_to_approval(_result(action=action), {"regime": "Neutral"}, _client())
        chk.assert_not_called()

    def test_off_leaves_the_verdict_time_rows_unchanged(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        client = _client()
        sized_off, calls_off = _route(client=client)
        # the same route with the check removed altogether
        with patch("agents.crew._add_loser_check", return_value=None):
            sized_ref, calls_ref = _route(client=_client())
        assert sized_off == sized_ref == 1
        strip = lambda calls: [json.loads(json.dumps([c.args[1:], c.kwargs], default=str)) for c in calls]
        assert strip(calls_off) == strip(calls_ref)
        assert "ENTRY_GATE_BLOCKED" not in _kinds(calls_off)

    def test_off_with_another_block_has_no_add_loser_field(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
        monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", True)
        ex = {"exit_at": "x", "hours_since": 30.0, "lots": 1, "lot_outcome_pct": [-20.0],
              "lot_alpha_pct": [-15.0], "lot_convictions": [7], "entries_since_exit": 0}
        with patch("db.get_last_broker_exit", return_value=ex), \
             patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=72.5), \
             patch("db.log_signal") as ls:
            crew._route_to_approval(_result(), {"regime": "Neutral"}, _client())
        rows = [c for c in ls.call_args_list if c[0][1] == "ENTRY_GATE_BLOCKED"]
        assert len(rows) == 1 and "add_loser" not in rows[0][1]["decision"]["entry_gate"]


APPROVED_AT = datetime.now(timezone.utc) - timedelta(hours=3)


def _approval(action="ADD", new_fact=False):
    aj = {"ticker": "NVDA", "action": action, "conviction": 7,
          "provenance": {"prompt_sha": "abc123", "model": "m", "trigger": _trigger(new_fact)}}
    return {"id": 7, "ticker": "NVDA", "action": action, "status": "FAILED", "conviction": 7,
            "timestamp": APPROVED_AT, "thesis_bucket": "Accelerator", "analyst_json": aj,
            "risk_json": {"shares_initial": 1, "price_at_sizing": 100.0}}


def _retry(approval, client=None, client_error=None, price=72.5):
    gc = (patch("ibkr.get_client", side_effect=client_error) if client_error
          else patch("ibkr.get_client", return_value=client or _client()))
    with patch("db.get_approval_by_id", return_value=approval), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=None), \
         patch("db.claim_deferred_entry_for_retry", return_value=True), \
         patch("db.get_last_broker_exit", return_value=None), \
         patch("db.expire_failed_entry", return_value=True) as expire, \
         patch("db.update_approval_status"), \
         patch("db.log_signal", return_value=1) as ls, \
         patch("agents.crew._get_price", return_value=price), \
         gc as get_client, \
         patch("execution.execute_approved", return_value=True) as execute:
        tasks.retry_deferred_entry.run(7)
    return {"execute": execute, "expire": expire, "log_signal": ls, "get_client": get_client}


class TestDeferredRetry:
    def test_loser_add_at_retry_is_expired(self):
        m = _retry(_approval())
        m["execute"].assert_not_called()
        m["expire"].assert_called_once()
        reason = m["expire"].call_args.args[1]
        assert reason.startswith("entry gate at retry: ADD -27.5% vs the average cost")
        rows = [c for c in m["log_signal"].call_args_list if c.args[1] == "ENTRY_GATE_BLOCKED"]
        assert len(rows) == 1
        gate = rows[0].kwargs["decision"]["entry_gate"]
        assert gate["path"] == "deferred_retry" and gate["approval_id"] == 7
        assert gate["add_loser"]["blocked"] is True and "stopout" not in gate

    def test_dip_review_add_at_retry_follows_the_switch(self, monkeypatch):
        approval = _approval(new_fact=None)
        approval["analyst_json"]["provenance"]["trigger"]["source"] = "dip_watch"
        m = _retry(approval)
        m["execute"].assert_not_called()
        monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", True)
        m = _retry(approval)
        m["execute"].assert_called_once()

    def test_new_fact_trigger_at_retry_executes(self):
        m = _retry(_approval(new_fact=True))
        m["execute"].assert_called_once()
        m["expire"].assert_not_called()

    def test_above_cost_at_retry_executes(self):
        m = _retry(_approval(), price=101.0)
        m["execute"].assert_called_once()

    def test_client_error_fails_open(self):
        m = _retry(_approval(), client_error=RuntimeError("gateway down"))
        m["execute"].assert_called_once()
        assert not [c for c in m["log_signal"].call_args_list if c.args[1] == "ENTRY_GATE_BLOCKED"]

    def test_open_at_retry_is_not_checked(self):
        m = _retry(_approval(action="OPEN"))
        m["get_client"].assert_not_called()
        m["execute"].assert_called_once()

    def test_off_skips_it(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        m = _retry(_approval())
        m["get_client"].assert_not_called()
        m["execute"].assert_called_once()

    def test_both_halves_block_and_name_both(self, monkeypatch):
        monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
        monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", True)
        ex = {"exit_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
              "hours_since": 2.0, "lots": 1, "lot_outcome_pct": [-6.1], "lot_alpha_pct": [0.6],
              "lot_convictions": [7], "entries_since_exit": 0}
        with patch("db.get_last_broker_exit", return_value=ex):
            approval = _approval()
            with patch("db.get_approval_by_id", return_value=approval), \
                 patch("db.has_newer_buy_approval", return_value=False), \
                 patch("db.get_latest_signal_for", return_value=None), \
                 patch("tasks._deferred_buy_shares_now", return_value=None), \
                 patch("db.claim_deferred_entry_for_retry", return_value=True), \
                 patch("db.expire_failed_entry", return_value=True) as expire, \
                 patch("db.update_approval_status"), \
                 patch("db.log_signal", return_value=1), \
                 patch("agents.crew._get_price", return_value=72.5), \
                 patch("ibkr.get_client", return_value=_client()), \
                 patch("execution.execute_approved", return_value=True) as execute:
                tasks.retry_deferred_entry.run(7)
        execute.assert_not_called()
        reason = expire.call_args.args[1]
        assert "exit the bot did not place 2.0 h ago (gate 10 d)" in reason
        assert "ADD -27.5% vs the average cost" in reason


# ---- modes, unjudged screens, earnings phase, reason text -------------------------------------

class TestJudgement:
    @pytest.mark.parametrize("nf", [True, False])
    def test_keyword_fallback_screen_is_unjudged(self, nf):
        res = _result(new_fact=nf)
        res["provenance"]["trigger"]["screener_fallback_reason"] = "ollama_unavailable"
        rec = _check(res, _client())
        assert rec["screener_fallback_reason"] == "ollama_unavailable"
        assert rec["trigger_new_fact"] is nf and rec["trigger_new_fact_source"] == "unjudged"
        assert rec["would_block"] is False and rec["blocked"] is False

    def test_stale_entry_recheck_is_unjudged(self):
        rec = _check(_result(new_fact=None, source="stale_entry_recheck"), _client())
        assert rec["trigger_new_fact_source"] == "unjudged" and rec["would_block"] is False

    def test_judged_false_still_blocks(self):
        res = _result(new_fact=False)
        res["provenance"]["trigger"]["screener_fallback_reason"] = None     # the 7B judged it
        assert _check(res, _client())["blocked"] is True

    @pytest.mark.parametrize("phase", ["pre", "day", "post"])
    def test_earnings_review_records_the_phase_and_is_not_exempt(self, phase):
        res = _result(path="earnings_review")
        res["provenance"]["trigger"]["earnings_phase"] = phase
        rec = _check(res, _client())
        assert rec["earnings_phase"] == phase and rec["trigger_new_fact_source"] == "none_on_path"
        assert rec["blocked"] is True

    def test_phase_only_on_earnings_review(self):
        res = _result()
        res["provenance"]["trigger"]["earnings_phase"] = "post"
        assert _check(res, _client())["earnings_phase"] is None


class TestModes:
    def test_production_default_is_shadow(self):
        import re
        from pathlib import Path
        src = (Path(config.__file__)).read_text()
        assert re.search(r'^ADD_LOSER_GATE_MODE = "shadow"$', src, re.M)

    def test_shadow_record_would_block_but_does_not_block(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "shadow")
        rec = _check(_result(), _client())
        assert rec["mode"] == "shadow" and rec["would_block"] is True and rec["blocked"] is False
        out, rows = _gate(rec)
        assert out is False and rows == []

    @pytest.mark.parametrize("bad", ["Enforce", "on", "", None, True])
    def test_invalid_mode_reads_as_shadow(self, monkeypatch, bad):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", bad)
        rec = _check(_result(), _client())
        assert rec["mode"] == "shadow" and rec["blocked"] is False and rec["would_block"] is True

    def test_off_reads_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        client = _client()
        assert _check(_result(), client) is None
        client.get_all_positions.assert_not_called()


class TestReasonText:
    def test_at_the_average_cost(self):
        rec = _check(_result(), _client(), price=99.999)
        assert rec["blocked"] is True and rec["pct_vs_avg_cost"] == -0.0
        assert crew._add_loser_reason(rec).startswith("ADD at the average cost with ")
        assert "-0.0%" not in crew._add_loser_reason(rec)

    def test_dip_watch_reason(self):
        rec = _check(_dip_result(), _client())
        assert "price-only drop-watch review" in crew._add_loser_reason(rec)
        assert "DIP_REVIEW_COUNTS_AS_NEW_FACT is off" in crew._add_loser_reason(rec)

    def test_no_news_item_reason_names_the_path(self):
        rec = _check(_result(path="scheduled_review"), _client())
        assert "(a scheduled_review verdict)" in crew._add_loser_reason(rec)


def _to_approval(price=72.5):
    sizing = {"shares_initial": 2, "initial_dollar": 2 * price, "position_size_pct": 0.1,
              "order_pct": 0.1, "initial_entry_pct": 0.55, "flags": [],
              "stop_loss_price": price * 0.9, "atr_stop": 4.0}
    with patch("config.MICRO_TIER_NAME_CAP_ENABLED", False), \
         patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("db.get_last_broker_exit", return_value=None), \
         patch("db.is_near_earnings", return_value=False), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("agents.crew._get_price", return_value=price), \
         patch("agents.crew.new_fact_gate", return_value=None), \
         patch("db.get_position_stop", return_value=None), \
         patch("risk_manager.RiskManager") as rm_cls, \
         patch("db.log_pending_approval", return_value=-1) as lpa, \
         patch("db.log_signal") as ls, \
         patch("execution.execute_manual"):
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(_result(), {"regime": "Neutral"}, _client())
    return (lpa.call_args.kwargs if lpa.called else None), [c.args[1] for c in ls.call_args_list]


class TestShadowRouting:
    @pytest.fixture(autouse=True)
    def _shadow(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "shadow")

    def test_shadow_loser_add_reaches_the_approval_with_the_record(self):
        kw, kinds = _to_approval()
        assert kw is not None and "ENTRY_GATE_BLOCKED" not in kinds
        rec = kw["risk_json"]["add_loser_gate"]
        assert rec["mode"] == "shadow" and rec["would_block"] is True and rec["blocked"] is False
        assert rec["pct_vs_avg_cost"] == -27.5

    def test_shadow_changes_nothing_else(self, monkeypatch):
        on, kinds_on = _to_approval()
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        off, kinds_off = _to_approval()
        assert "add_loser_gate" not in off["risk_json"] and kinds_on == kinds_off
        def strip(kw):      # the arrival record carries the wall-clock sizing time
            rj = {k: v for k, v in kw["risk_json"].items() if k != "add_loser_gate"}
            rj["arrival"] = {k: v for k, v in (rj.get("arrival") or {}).items() if k != "sizing_at"}
            return {**kw, "risk_json": rj}
        assert strip(on) == strip(off)

    def test_enforce_has_no_risk_json_record(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "enforce")
        kw, _ = _to_approval(price=110.0)            # above cost: passes
        assert kw is not None and "add_loser_gate" not in kw["risk_json"]

    def test_old_shadow_keeps_its_meaning(self):
        kw, _ = _to_approval()
        old = kw["risk_json"]["add_to_loser"]
        assert old["breaks_rule"] is True and old["new_fact"] is False


class TestShadowDeferred:
    def test_shadow_retry_writes_one_activity_row_and_executes(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "shadow")
        with patch("db.log_activity_queued") as act:
            m = _retry(_approval())
        m["execute"].assert_called_once()
        m["expire"].assert_not_called()
        rows = [c for c in act.call_args_list if c.args[0] == "add_loser_gate_shadow"]
        assert len(rows) == 1
        extra = rows[0].kwargs["extra"]
        assert extra["path"] == "deferred_retry" and extra["approval_id"] == 7
        assert extra["add_loser_gate"]["would_block"] is True and extra["add_loser_gate"]["blocked"] is False
        assert "would block: ADD -27.5% vs the average cost" in rows[0].args[2]
        assert not [c for c in m["log_signal"].call_args_list if c.args[1] == "ENTRY_GATE_BLOCKED"]

    def test_shadow_retry_above_cost_says_would_not_block(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "shadow")
        with patch("db.log_activity_queued") as act:
            m = _retry(_approval(), price=105.0)
        m["execute"].assert_called_once()
        row = next(c for c in act.call_args_list if c.args[0] == "add_loser_gate_shadow")
        assert "would not block (ok, event_provenance)" in row.args[2]

    def test_shadow_retry_with_stopout_on_still_lets_stopout_block(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "shadow")
        monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
        monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", True)
        ex = {"exit_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
              "hours_since": 2.0, "lots": 1, "lot_outcome_pct": [-6.1], "lot_alpha_pct": [0.6],
              "lot_convictions": [7], "entries_since_exit": 0}
        approval = _approval()
        with patch("db.log_activity_queued") as act, \
             patch("db.get_last_broker_exit", return_value=ex), \
             patch("db.get_approval_by_id", return_value=approval), \
             patch("db.has_newer_buy_approval", return_value=False), \
             patch("db.get_latest_signal_for", return_value=None), \
             patch("tasks._deferred_buy_shares_now", return_value=None), \
             patch("db.claim_deferred_entry_for_retry", return_value=True), \
             patch("db.expire_failed_entry", return_value=True) as expire, \
             patch("db.update_approval_status"), \
             patch("db.log_signal", return_value=1) as ls, \
             patch("agents.crew._get_price", return_value=72.5), \
             patch("ibkr.get_client", return_value=_client()), \
             patch("execution.execute_approved", return_value=True) as execute:
            tasks.retry_deferred_entry.run(7)
        execute.assert_not_called()
        reason = expire.call_args.args[1]
        assert "exit the bot did not place" in reason and "average cost" not in reason
        gate = next(c for c in ls.call_args_list
                    if c.args[1] == "ENTRY_GATE_BLOCKED").kwargs["decision"]["entry_gate"]
        assert "add_loser" not in gate
        assert [c.args[0] for c in act.call_args_list].count("add_loser_gate_shadow") == 1

    @pytest.mark.parametrize("mode", ["enforce", "shadow"])
    def test_stopout_shadow_not_called_on_a_deferred_add_with_its_switch_off(self, monkeypatch, mode):
        monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
        monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", False)
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", mode)
        with patch("agents.crew._stopout_reentry_shadow") as sr, patch("db.log_activity_queued"):
            m = _retry(_approval())
        sr.assert_not_called()
        if mode == "enforce":
            m["execute"].assert_not_called()
        else:
            m["execute"].assert_called_once()

    def test_enforce_retry_at_the_average_cost_reason(self):
        m = _retry(_approval(), price=99.999)
        assert m["expire"].call_args.args[1].startswith("entry gate at retry: ADD at the average cost with")
