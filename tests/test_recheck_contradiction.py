"""Recheck contradiction (record only; agents/crew.py:_recheck_contradiction_record):
an OPEN/ADD past the rebuy cooldown records the most recent non-buy verdict on
the ticker within RECHECK_CONTRADICTION_WINDOW_HOURS, and would_block when
that verdict was a stale-entry recheck or a held review and the current
trigger is not a new fact. Written to risk_json or the routing row that ended
the route; nothing blocks on it. Synthetic data only."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
import db
from agents import crew

NOW = datetime(2026, 11, 20, 15, 28, tzinfo=timezone.utc)      # 10:28 ET


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(config, "RECHECK_CONTRADICTION_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "RECHECK_CONTRADICTION_WINDOW_HOURS", 4)


def _v(kind, hours_ago, path="event", source="news", conviction=6, sid=1, model_action=None):
    return {"timestamp": NOW - timedelta(hours=hours_ago), "signal": kind, "conviction": conviction,
            "udr": 1.2, "signal_id": sid, "trigger_path": path, "trigger_source": source,
            "model_action": model_action if model_action is not None else kind}


def _result(action="OPEN", new_fact=False, path="event"):
    trig = {"path": path, "source": "news", "title": "t"}
    if path == "event" and new_fact is not None:
        trig["new_fact"] = new_fact
    return {"ticker": "NVDA", "action": action, "conviction": 7, "thesis_bucket": "Accelerator",
            "reasoning": "r", "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m", "trigger": trig}}


def _rec(rows, result=None, **kw):
    with patch("db.get_recent_verdicts", return_value=rows, **kw) as gv:
        out = crew._recheck_contradiction_record("NVDA", result or _result(), now=NOW)
    return out, gv


RECHECK = dict(path="event", source="stale_entry_recheck")


class TestRecord:
    def test_the_11_20_shape(self):
        # 09:35 ET recheck WATCH c6, then a 10:28 headline OPEN with no new fact
        rows = [_v("OPEN", 0.0, sid=3), _v("WATCH", 0.88, conviction=6, sid=2, **RECHECK)]
        rec, gv = _rec(rows)
        assert gv.call_args.args == ("NVDA",)
        assert gv.call_args.kwargs == {"hours": 4, "limit": 50, "strict": True, "with_trigger": True}
        assert rec["read"] == "ok" and rec["n_non_buy_in_window"] == 1
        assert rec["prior"] == {"signal_id": 2, "kind": "WATCH", "model_action": "WATCH",
                                "source": "stale_entry_recheck",
                                "trigger_path": "event", "trigger_source": "stale_entry_recheck",
                                "age_hours": 0.88, "conviction": 6, "is_recheck": True,
                                "at": (NOW - timedelta(hours=0.88)).isoformat()}
        assert rec["current_trigger_path"] == "event" and rec["current_new_fact"] is False
        assert rec["would_block"] is True and rec["window_hours"] == 4.0

    def test_new_fact_on_the_current_trigger_clears_would_block(self):
        rec, _ = _rec([_v("WATCH", 0.88, **RECHECK)], _result(new_fact=True))
        assert rec["prior"]["is_recheck"] is True and rec["would_block"] is False

    @pytest.mark.parametrize("path,source,cls,blocks", [
        ("scheduled_review", "portfolio_review", "held_review", True),
        ("earnings_review", "portfolio_review", "held_review", True),
        ("plan_level_review", "plan_watcher", "plan_level", False),
        ("event", "news", "event", False),
        ("event", "dip_watch", "event", False),
        ("daily_scan", "scanner", "daily_scan", False),
        (None, None, "unknown", False)])
    def test_source_classes(self, path, source, cls, blocks):
        rec, _ = _rec([_v("HOLD", 1.0, path=path, source=source)])
        assert rec["prior"]["source"] == cls and rec["prior"]["is_recheck"] is False
        assert rec["would_block"] is blocks

    def test_no_news_item_on_the_current_verdict_is_no_new_fact(self):
        rec, _ = _rec([_v("HOLD", 1.0, path="scheduled_review", source="portfolio_review")],
                      _result(action="ADD", path="scheduled_review"))
        assert rec["current_new_fact"] is None and rec["would_block"] is True

    def test_most_recent_non_buy_wins_and_buys_are_skipped(self):
        rows = [_v("ADD", 0.1, sid=9), _v("HOLD", 0.5, sid=8), _v("OPEN", 1.0, sid=7),
                _v("WATCH", 2.0, sid=6, **RECHECK)]
        rec, _ = _rec(rows)
        assert rec["prior"]["signal_id"] == 8 and rec["n_non_buy_in_window"] == 2
        assert rec["would_block"] is False          # the newest one is a plain event HOLD

    @pytest.mark.parametrize("kind", ["HOLD", "WATCH", "PASS", "TRIM", "EXIT", "SELL"])
    def test_non_buy_kinds(self, kind):
        assert _rec([_v(kind, 1.0, **RECHECK)])[0]["prior"]["kind"] == kind

    def test_window_edges(self):
        rec, _ = _rec([_v("WATCH", 4.01, **RECHECK)])
        assert rec["prior"] is None and rec["n_non_buy_in_window"] == 0 and rec["would_block"] is False
        rec, _ = _rec([_v("WATCH", 4.0, **RECHECK)])
        assert rec["prior"]["age_hours"] == 4.0 and rec["would_block"] is True
        rec, _ = _rec([_v("WATCH", -0.2, **RECHECK)])           # well ahead of now: not a prior
        assert rec["prior"] is None
        rec, _ = _rec([_v("WATCH", -0.04, **RECHECK)])          # within the clock-skew tolerance
        assert rec["prior"]["age_hours"] == -0.04 and rec["would_block"] is True
        rec, _ = _rec([_v("WATCH", -0.06, **RECHECK)])
        assert rec["prior"] is None

    def test_window_constant(self, monkeypatch):
        monkeypatch.setattr(config, "RECHECK_CONTRADICTION_WINDOW_HOURS", 1.5)
        rec, gv = _rec([_v("WATCH", 1.6, **RECHECK)])
        assert gv.call_args.kwargs["hours"] == 2 and rec["prior"] is None and rec["window_hours"] == 1.5

    def test_naive_timestamp_is_utc(self):
        row = _v("WATCH", 1.0, **RECHECK)
        row["timestamp"] = row["timestamp"].replace(tzinfo=None)
        assert _rec([row])[0]["prior"]["age_hours"] == 1.0

    def test_no_verdicts(self):
        rec, _ = _rec([])
        assert rec["read"] == "ok" and rec["prior"] is None and rec["would_block"] is False

    def test_read_error_is_recorded_not_raised(self):
        rec, _ = _rec(None, side_effect=ConnectionError("db down"))
        assert rec["read"] == "error: ConnectionError"
        assert rec["prior"] is None and rec["would_block"] is None and rec["n_non_buy_in_window"] is None

    def test_bad_rows_do_not_raise(self):
        rec, _ = _rec([{"signal": "WATCH", "timestamp": "not a time"}])
        assert rec["read"] == "ok" and rec["prior"] is None
        rec, _ = _rec([None])
        assert rec["read"] == "error: AttributeError" and rec["would_block"] is None

    @pytest.mark.parametrize("action", ["TRIM", "EXIT", "HOLD", "WATCH"])
    def test_only_buys(self, action):
        rec, gv = _rec([_v("WATCH", 1.0, **RECHECK)], _result(action=action))
        assert rec is None
        gv.assert_not_called()

    def test_off_reads_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "RECHECK_CONTRADICTION_SHADOW_ENABLED", False)
        rec, gv = _rec([_v("WATCH", 1.0, **RECHECK)])
        assert rec is None
        gv.assert_not_called()


class TestDbRead:
    def _run(self, **kw):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        row = (NOW, "WATCH", 0.6, "1.2") + ((42, "event", "stale_entry_recheck", "OPEN")
                                            if kw.get("with_trigger") else ())
        cur.fetchall.return_value = [row]
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            out = db.get_recent_verdicts("NVDA", hours=4, limit=50, **kw)
        return out, cur.execute.call_args.args[0]

    def test_default_query_and_rows_unchanged(self):
        out, sql = self._run()
        # rows picked by id first, the jsonb read for those rows (signals-index)
        assert sql.startswith("SELECT s.timestamp, s.signal, s.confidence, "
                              "s.decision->'model_output'->>'upside_downside_ratio' FROM (SELECT id, ")
        assert ") k JOIN signals s ON s.id = k.id ORDER BY k.timestamp DESC" in sql
        assert out == [{"timestamp": NOW, "signal": "WATCH", "conviction": 6, "udr": 1.2}]

    def test_with_trigger(self):
        out, sql = self._run(with_trigger=True)
        assert (", s.id, s.provenance->'trigger'->>'path', s.provenance->'trigger'->>'source', "
                "s.decision->>'model_action' FROM (SELECT id, ") in sql
        assert out == [{"timestamp": NOW, "signal": "WATCH", "conviction": 6, "udr": 1.2,
                        "signal_id": 42, "trigger_path": "event", "trigger_source": "stale_entry_recheck",
                        "model_action": "OPEN"}]


def _live_verdicts():
    # the route reads the real clock, so the prior verdict is 0.88 h before it
    return [{"timestamp": datetime.now(timezone.utc) - timedelta(hours=0.88), "signal": "WATCH",
             "conviction": 6, "udr": 1.2, "signal_id": 2, "trigger_path": "event",
             "trigger_source": "stale_entry_recheck"}]


def _route(shares=2, entry_gate=False, new_fact_gate=None, enabled=True, price=100.0, read_error=None):
    sizing = {"shares_initial": shares, "initial_dollar": shares * price, "position_size_pct": 0.1,
              "order_pct": 0.1, "initial_entry_pct": 0.55, "flags": [] if shares else ["too small"],
              "stop_loss_price": price * 0.9, "atr_stop": 4.0}
    result = _result()
    with ExitStack() as st:
        st.enter_context(patch("config.RECHECK_CONTRADICTION_SHADOW_ENABLED", enabled))
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=price))
        st.enter_context(patch("agents.crew.new_fact_gate", return_value=new_fact_gate))
        gv = st.enter_context(patch("db.get_recent_verdicts", side_effect=read_error)
                              if read_error else patch("db.get_recent_verdicts", return_value=_live_verdicts()))
        if entry_gate:
            st.enter_context(patch("db.get_last_broker_exit", return_value={
                "exit_at": "x", "hours_since": 30.0, "lots": 1, "lot_outcome_pct": [-20.0],
                "lot_alpha_pct": [-15.0], "lot_convictions": [7], "entries_since_exit": 0}))
            st.enter_context(patch("config.STOPOUT_REENTRY_GATE_DAYS", 10))
            st.enter_context(patch("config.STOPOUT_REENTRY_SHADOW_ENABLED", True))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, rm_cls.return_value)
    rows = {c.args[1]: c.kwargs for c in ls.call_args_list}
    return lpa.call_args.kwargs if lpa.called else None, rows, gv


class TestRouting:
    def test_risk_json_carries_the_record_and_nothing_else_changes(self):
        on, _, gv = _route()
        off, _, gv_off = _route(enabled=False)
        rec = on["risk_json"]["recheck_contradiction"]
        assert rec["read"] == "ok" and rec["prior"]["source"] == "stale_entry_recheck"
        assert rec["prior"]["kind"] == "WATCH" and rec["prior"]["conviction"] == 6
        assert "recheck_contradiction" not in off["risk_json"]
        gv_off.assert_not_called()
        for k in ("shares_initial", "position_size_pct", "stop_loss_price", "order_pct", "flags"):
            assert on["risk_json"][k] == off["risk_json"][k]
        for k in ("action", "conviction", "size_pct", "stop_price", "entry_strategy", "analyst_json"):
            assert on[k] == off[k]

    def test_would_block_does_not_block(self):
        on, _, _ = _route()
        assert on["risk_json"]["recheck_contradiction"]["would_block"] is True
        assert on["action"] == "OPEN"                       # it still reached the approval

    def test_zero_shares_row_carries_it(self):
        lpa, rows, _ = _route(shares=0)
        assert lpa is None
        assert rows["ROUTE_ZERO_SHARES"]["decision"]["recheck_contradiction"]["would_block"] is True

    def test_entry_gate_row_carries_it(self):
        lpa, rows, _ = _route(entry_gate=True)
        assert lpa is None
        d = rows["ENTRY_GATE_BLOCKED"]["decision"]
        assert d["entry_gate"]["stopout"]["blocked"] is True
        assert d["recheck_contradiction"]["prior"]["is_recheck"] is True

    def test_new_fact_gate_row_carries_it(self):
        lpa, rows, _ = _route(new_fact_gate={"passed": False, "reason": "no new fact"})
        assert lpa is None
        d = rows["ROUTE_NO_NEW_FACT"]["decision"]
        assert d["new_fact_gate"]["passed"] is False and d["recheck_contradiction"]["read"] == "ok"

    def test_off_rows_have_no_field(self):
        _, rows, _ = _route(shares=0, enabled=False)
        assert "recheck_contradiction" not in rows["ROUTE_ZERO_SHARES"]["decision"]

    def test_read_error_still_routes(self):
        on, _, _ = _route(read_error=RuntimeError("down"))
        off, _, _ = _route(enabled=False)
        rec = on["risk_json"]["recheck_contradiction"]
        assert rec["read"] == "error: RuntimeError" and rec["would_block"] is None
        assert on["risk_json"]["shares_initial"] == off["risk_json"]["shares_initial"]

    @pytest.mark.parametrize("action", ["TRIM", "EXIT"])
    def test_sells_are_not_recorded(self, action):
        with patch("agents.crew._recheck_contradiction_record") as rc, \
             patch("db.has_active_approval", return_value=False), \
             patch("agents.crew._get_positions", return_value=None), \
             patch("db.log_signal"), patch("alerts._send"):
            crew._route_to_approval(_result(action=action), {"regime": "Neutral"}, MagicMock())
        rc.assert_not_called()


def test_route_row_fields_extra_never_overrides():
    f = crew._route_row_fields(_result(), "X", {"route_exit": "nope", "recheck_contradiction": 1})
    assert f["decision"]["route_exit"] == "X" and f["decision"]["recheck_contradiction"] == 1
    assert "recheck_contradiction" not in crew._route_row_fields(_result(), "X")["decision"]


def test_demoted_watch_is_told_apart_by_model_action():
    rec, _ = _rec([_v("WATCH", 1.0, model_action="OPEN", **RECHECK)])
    assert rec["prior"]["kind"] == "WATCH" and rec["prior"]["model_action"] == "OPEN"


@pytest.mark.parametrize("rows", [
    [_v("WATCH", 0.88, **RECHECK)], [], None])
def test_record_is_plain_json(rows):
    rec, _ = (_rec(rows) if rows is not None else _rec(None, side_effect=RuntimeError("down")))
    import json
    json.dumps(db._finite_json(rec))            # no default=: every value is plain JSON
    assert db._finite_json(rec) == rec


def _route_rows(**patches):
    """The route on an OPEN with one record-carrying exit forced by patches."""
    result = _result()
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.get_recent_verdicts", return_value=_live_verdicts()))
        st.enter_context(patch("agents.crew.new_fact_gate", return_value=None))
        for target, kw in patches.items():
            st.enter_context(patch(target.replace("__", "."), **kw))
        ls = st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("alerts._send"))
        crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
    return {c.args[1]: c.kwargs for c in ls.call_args_list}


class TestOtherExitRows:
    def test_earnings_blocked_row(self):
        rows = _route_rows(config__EARNINGS_ENTRY_BLACKOUT_DAYS={"new": 7},
                           db__is_near_earnings={"return_value": True},
                           db__get_next_earnings={"return_value": "2026-11-25"})
        assert rows["EARNINGS_BLOCKED"]["decision"]["recheck_contradiction"]["prior"]["is_recheck"] is True

    def test_catalyst_blocked_row(self):
        rows = _route_rows(db__is_near_earnings={"return_value": False},
                           catalyst_service__gates__catalyst_gate_check={
                               "return_value": {"approved": False, "reason": "no catalyst"}})
        assert "CATALYST_BLOCKED" in rows
        assert rows["CATALYST_BLOCKED"]["decision"]["recheck_contradiction"]["read"] == "ok"

    def test_no_price_row(self):
        rows = _route_rows(db__is_near_earnings={"return_value": False},
                           catalyst_service__gates__catalyst_gate_check={"return_value": {"approved": True}},
                           agents__crew___get_price={"return_value": None},
                           risk_manager__RiskManager={})
        assert rows["ROUTE_NO_PRICE"]["decision"]["recheck_contradiction"]["would_block"] is True

    def test_risk_init_failed_row(self):
        rows = _route_rows(db__is_near_earnings={"return_value": False},
                           catalyst_service__gates__catalyst_gate_check={"return_value": {"approved": True}},
                           risk_manager__RiskManager={"side_effect": RuntimeError("no broker")})
        d = rows["ROUTE_RISK_INIT_FAILED"]["decision"]
        assert d["error_class"] == "RuntimeError" and d["recheck_contradiction"]["read"] == "ok"

    def test_micro_name_cap_row_carries_it_from_sizing(self):
        rm = MagicMock()
        rm.micro_name_cap.return_value = {"cap_pct": 5.0, "held_pct": 6.0, "shares_after": 0,
                                          "outcome": "capped"}
        sizing = {"shares_initial": 1, "flags": [], "recheck_contradiction": {"read": "ok", "would_block": True}}
        with patch("db.log_signal") as ls:
            assert crew._apply_micro_name_cap(rm, "NVDA", sizing, 100.0, _result()) is True
        d = next(c.kwargs for c in ls.call_args_list if c.args[1] == "ROUTE_MICRO_NAME_CAP")["decision"]
        assert d["recheck_contradiction"] == {"read": "ok", "would_block": True}
        sizing.pop("recheck_contradiction")
        with patch("db.log_signal") as ls:
            crew._apply_micro_name_cap(rm, "NVDA", sizing, 100.0, _result())
        d = next(c.kwargs for c in ls.call_args_list if c.args[1] == "ROUTE_MICRO_NAME_CAP")["decision"]
        assert "recheck_contradiction" not in d
