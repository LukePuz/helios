"""Same-day consensus (record only; agents/crew.py:_same_day_consensus_record):
an OPEN/ADD past the rebuy cooldown records the ticker's earlier analyst
verdicts since ET midnight and over the last 24 h, from one bounded read. Routing, sizing and the
analyst's input are unchanged: the route is compared with the record on and
off, and the prompt and brief template hashes do not read the changed code.
Synthetic data only."""
import copy
import json
import re
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
import db
from agents import crew

NOW = datetime(2026, 11, 20, 15, 28, tzinfo=timezone.utc)      # 10:28 ET
ET_MIDNIGHT = datetime(2026, 11, 20, 5, 0, tzinfo=timezone.utc)
DAY_FIELDS = ("window", "read", "limit_hit", "current_signal_id", "n_earlier", "n_buy_grade",
              "buy_grade_share", "n_buy_grade_unknown", "n_model_buy", "median_conviction",
              "n_conviction", "median_udr", "n_udr", "n_story_keys", "n_distinct_stories",
              "buy_grade_definition")


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_RECORD_ENABLED", True)
    monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_READ_TIMEOUT_MS", 500)
    monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_LIMIT", 200)


def _v(sid, kind="WATCH", conviction=6, udr=1.5, final_pass=False, model_action=None,
       model_pass=False, story="gn:reuters|nvidia china scrutiny", minutes_ago=60):
    return {"signal_id": sid, "timestamp": NOW - timedelta(minutes=minutes_ago), "signal": kind,
            "conviction": conviction, "udr": udr, "final_pass_to_risk": final_pass,
            "model_action": model_action or kind, "model_pass_to_risk": model_pass,
            "story_key": story}


# The 11-20 NVDA shape: mostly non-buy earlier verdicts, one earlier buy.
DAY = [_v(11, "WATCH", 6, 1.5), _v(12, "HOLD", 5, 1.2, story="gn:cnbc|nvidia earnings preview"),
       _v(13, "OPEN", 7, 2.4, final_pass=True, model_pass=True),
       _v(14, "WATCH", 7, 1.8, model_action="OPEN", model_pass=True),     # demoted by a gate
       _v(15, "PASS", 3, None, story=None)]


def _result(action="OPEN", signal_id=99):
    return {"ticker": "NVDA", "action": action, "conviction": 7, "thesis_bucket": "Accelerator",
            "reasoning": "r", "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m", "signal_id": signal_id,
                           "trigger": {"path": "event", "source": "news", "title": "t",
                                       "new_fact": True}}}


def _rec(rows, result=None, **kw):
    with patch("db.get_same_day_verdicts", return_value=rows, **kw) as gv:
        out = crew._same_day_consensus_record("NVDA", result or _result(), now=NOW)
    return out, gv


class TestRecord:
    def test_the_11_20_shape(self):
        rec, gv = _rec(DAY)
        assert rec["read"] == "ok" and rec["n_earlier"] == 5
        assert rec["n_buy_grade"] == 1 and rec["buy_grade_share"] == 0.2
        assert rec["n_model_buy"] == 2                     # the demoted WATCH counts here only
        assert rec["n_buy_grade_unknown"] == 0
        assert rec["median_conviction"] == 6 and rec["median_udr"] == 1.65 and rec["n_udr"] == 4
        assert rec["n_story_keys"] == 4 and rec["n_distinct_stories"] == 2
        assert rec["limit_hit"] is False and "read_ms" not in rec
        assert set(rec["window"]) == {"start", "basis"}         # no wall-clock value stored
        assert rec["n_conviction"] == 5
        assert rec["window_24h"] == {"hours": 24, "basis": "the 24 h before this route"}
        assert rec["n_earlier_24h"] == 5 and rec["n_buy_grade_24h"] == 1
        assert rec["buy_grade_share_24h"] == 0.2 and rec["limit_hit_24h"] is False
        args, kw = gv.call_args                                 # one read, the last 24 h
        assert gv.call_count == 1
        assert args[0] == "NVDA" and args[1] == NOW - timedelta(hours=24) and args[2] == NOW
        assert kw == {"limit": 200, "timeout_ms": 500}

    def test_the_et_day_fields_ignore_the_rows_before_et_midnight(self):
        """The ET-day fields are what an ET-midnight read gave; the rows
        before midnight count in the 24 h fields only."""
        night = [_v(21, "OPEN", 9, 3.0, final_pass=True, minutes_ago=700),    # 22:48 ET 11-19
                 _v(22, "WATCH", 2, 0.5, story="gn:x|other", minutes_ago=1000),
                 _v(23, "OPEN", 8, 2.0, final_pass=True, minutes_ago=1439)]
        day_only, _ = _rec(DAY)
        both, _ = _rec(DAY + night)
        assert {k: both[k] for k in DAY_FIELDS} == {k: day_only[k] for k in DAY_FIELDS}
        assert both["n_earlier_24h"] == 8 and both["n_buy_grade_24h"] == 3
        assert both["buy_grade_share_24h"] == 0.375
        early, _ = _rec(night)                                  # a pre-open route's view
        assert early["n_earlier"] == 0 and early["buy_grade_share"] is None
        assert early["n_earlier_24h"] == 3 and early["n_buy_grade_24h"] == 2

    def test_the_24h_window_starts_at_route_minus_24h(self):
        rec, _ = _rec(DAY + [_v(31, "OPEN", final_pass=True, minutes_ago=1440),
                             _v(32, "OPEN", final_pass=True, minutes_ago=1441)])
        assert rec["n_earlier_24h"] == 6 and rec["n_buy_grade_24h"] == 2

    def test_a_25h_fall_back_day_reads_from_et_midnight(self):
        """2026-11-01 23:30 EST: ET midnight (04:00 UTC) is 24.5 h back, so
        the read starts there; the 24 h fields still start at route - 24 h."""
        now = datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc)
        midnight = datetime(2026, 11, 1, 4, 0, tzinfo=timezone.utc)
        rows = [_v(1, "OPEN", final_pass=True, minutes_ago=0),
                _v(2, "OPEN", final_pass=True, minutes_ago=0)]
        rows[0]["timestamp"] = now - timedelta(hours=24, minutes=15)   # ET day only
        rows[1]["timestamp"] = now - timedelta(hours=1)
        with patch("db.get_same_day_verdicts", return_value=rows) as gv:
            rec = crew._same_day_consensus_record("NVDA", _result(), now=now)
        assert gv.call_args.args[1] == midnight
        assert rec["window"]["start"].startswith("2026-11-01T00:00:00-04:00")
        assert rec["n_earlier"] == 2 and rec["n_earlier_24h"] == 1

    def test_a_null_confidence_is_left_out_of_the_median(self):
        rec, _ = _rec([_v(1, conviction=None), _v(2, conviction=None), _v(3, conviction=8)])
        assert rec["median_conviction"] == 8 and rec["n_conviction"] == 1
        assert rec["n_earlier"] == 3

    def test_buy_grade_is_the_stored_flag_not_recomputed(self):
        """c9 UDR 4 with a stored False (a later gate) is not buy-grade; a
        stored True on a non-buy kind is not either."""
        rec, _ = _rec([_v(1, "OPEN", 9, 4.0, final_pass=False),
                       _v(2, "WATCH", 8, 3.0, final_pass=True),
                       _v(3, "ADD", 5, 2.0, final_pass=True)])
        assert rec["n_buy_grade"] == 1 and rec["buy_grade_share"] == pytest.approx(1 / 3, abs=1e-4)

    def test_rows_without_the_stored_flag_are_counted_unknown(self):
        rec, _ = _rec([_v(1, "OPEN", 8, 3.0, final_pass=None), _v(2, "HOLD")])
        assert rec["n_buy_grade"] == 0 and rec["n_buy_grade_unknown"] == 1

    def test_the_current_verdict_is_excluded(self):
        rec, _ = _rec(DAY + [_v(99, "OPEN", 7, 3.0, final_pass=True, minutes_ago=0)])
        assert rec["n_earlier"] == 5 and rec["current_signal_id"] == 99
        rec, _ = _rec(DAY, result=_result(signal_id=None))
        assert rec["current_signal_id"] is None and rec["n_earlier"] == 5

    def test_no_earlier_verdicts(self):
        rec, _ = _rec([])
        assert rec["n_earlier"] == 0 and rec["buy_grade_share"] is None
        assert rec["median_conviction"] is None and rec["n_distinct_stories"] is None

    def test_the_window_is_the_et_day(self):
        late = datetime(2026, 11, 21, 3, 30, tzinfo=timezone.utc)      # 22:30 ET on 11-20
        with patch("db.get_same_day_verdicts", return_value=[]) as gv:
            rec = crew._same_day_consensus_record("NVDA", _result(), now=late)
        assert gv.call_args.args[1] == late - timedelta(hours=24)       # 22:30 ET 11-19
        assert rec["window"]["basis"] == "same ET day, before this route"
        assert rec["window"]["start"].startswith("2026-11-20T00:00:00-05:00")

    def test_a_slow_read_is_skipped_and_an_error_is_recorded(self):
        class QueryCanceled(Exception):
            pass
        rec, _ = _rec(None, side_effect=QueryCanceled("canceling statement due to statement timeout"))
        assert rec["read"] == "skipped_budget" and rec["n_earlier"] is None
        assert rec["n_earlier_24h"] is None and isinstance(rec["window_24h"], dict)
        rec, _ = _rec(None, side_effect=RuntimeError("down"))
        assert rec["read"] == "error: RuntimeError" and rec["buy_grade_share"] is None
        assert rec["buy_grade_share_24h"] is None

    def test_the_limit_is_flagged(self, monkeypatch):
        monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_LIMIT", 3)
        rec, _ = _rec(DAY[:3])
        assert rec["limit_hit"] is True and rec["limit_hit_24h"] is True

    def test_a_full_read_with_rows_before_midnight_leaves_the_et_day_whole(self, monkeypatch):
        """Newest first: a full 24 h read holding rows from before ET
        midnight holds every ET-day row, so only limit_hit_24h is set."""
        monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_LIMIT", 3)
        rec, _ = _rec(DAY[:2] + [_v(21, minutes_ago=700)])
        assert rec["limit_hit"] is False and rec["limit_hit_24h"] is True
        assert rec["n_earlier"] == 2 and rec["n_earlier_24h"] == 3

    def test_limit_hit_counts_the_raw_rows_before_the_current_verdict_is_dropped(self, monkeypatch):
        """A full read that includes the routed verdict still hit the limit."""
        monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_LIMIT", 3)
        rec, _ = _rec(DAY[:2] + [_v(99, "OPEN", 7, 3.0, final_pass=True, minutes_ago=0)])
        assert rec["n_earlier"] == 2 and rec["limit_hit"] is True
        assert rec["n_earlier_24h"] == 2 and rec["limit_hit_24h"] is True

    def test_non_finite_udr_is_left_out_of_the_median(self):
        rec, _ = _rec([_v(1, udr=float("nan")), _v(2, udr=float("inf")), _v(3, udr=2.0),
                       _v(4, udr=True)])
        assert rec["median_udr"] == 2.0 and rec["n_udr"] == 1

    def test_fire_test_verdicts_are_counted(self):
        """fire_test's analyst row carries no marker (only its approval's
        reasoning is "[TEST RUN] ..."), so the read has no test filter."""
        import inspect
        src = inspect.getsource(db.get_same_day_verdicts)
        assert "TEST RUN" not in src.split('"""')[2]          # the code, not the docstring

    def test_bad_rows_do_not_raise(self):
        rec, _ = _rec([None, "x", {"signal": "OPEN", "conviction": "high", "udr": "n/a",
                                   "timestamp": NOW - timedelta(hours=1)},
                       {"signal": "OPEN", "conviction": True, "timestamp": "today"}])
        assert rec["read"] == "ok" and rec["n_earlier"] == 1 and rec["median_conviction"] is None
        assert rec["n_conviction"] == 0 and rec["n_earlier_24h"] == 1     # undated: in neither

    def test_a_naive_timestamp_reads_as_utc(self):
        row = _v(1, minutes_ago=30)
        row["timestamp"] = row["timestamp"].replace(tzinfo=None)
        rec, _ = _rec([row])
        assert rec["n_earlier"] == 1 and rec["n_earlier_24h"] == 1

    def test_off_reads_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "SAME_DAY_CONSENSUS_RECORD_ENABLED", False)
        rec, gv = _rec(DAY)
        assert rec is None and not gv.called

    @pytest.mark.parametrize("action", ["TRIM", "EXIT", "HOLD", "WATCH"])
    def test_only_buys(self, action):
        rec, gv = _rec(DAY, result=_result(action=action))
        assert rec is None and not gv.called

    @pytest.mark.parametrize("rows", [DAY, [], None])
    def test_record_is_plain_json(self, rows):
        rec, _ = (_rec(rows) if rows is not None else _rec(None, side_effect=RuntimeError("x")))
        json.dumps(db._finite_json(rec))
        assert db._finite_json(rec) == rec


class TestRead:
    def _run(self, rows=(), error=None):
        seen = {}

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, args=None):
                seen.setdefault("sql", []).append((sql, args))
                if error and "SELECT" in sql:
                    raise error

            def fetchall(self):
                return list(rows)
        conn = MagicMock()
        conn.cursor.return_value = Cur()
        with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
            try:
                out = db.get_same_day_verdicts("NVDA", ET_MIDNIGHT, NOW, limit=50, timeout_ms=1500)
            except Exception as e:
                out = e
        return out, seen["sql"], conn, put

    def test_one_bounded_read_only_round_trip(self):
        rows = [(13, NOW, "OPEN", 0.7, "2.4", "true", "OPEN", "true", "gn:x|y"),
                (11, NOW, "WATCH", 0.6, None, None, "WATCH", "false", None)]
        out, sql, conn, put = self._run(rows)
        (q, args), = sql                                        # one execute
        assert q.startswith("SET LOCAL statement_timeout = %s; SELECT ")
        assert args[0] == 1500 and args[-3:] == (ET_MIDNIGHT, NOW, 50)
        assert not any(isinstance(a, str) and "TEST RUN" in a for a in args)
        assert "signal NOT IN %s" in q and db._NON_ANALYST_SIGNALS in args
        assert "trading_env=%s" in q and "LIMIT %s" in q
        assert not any(w in q.upper() for w in ("INSERT", "UPDATE ", "DELETE"))
        conn.rollback.assert_called() and put.assert_called_once()
        assert out[0] == {"signal_id": 13, "timestamp": NOW, "signal": "OPEN", "conviction": 7,
                          "udr": 2.4, "final_pass_to_risk": True, "model_action": "OPEN",
                          "model_pass_to_risk": True, "story_key": "gn:x|y"}
        assert out[1]["final_pass_to_risk"] is None and out[1]["udr"] is None

    def test_a_null_confidence_is_none_not_zero(self):
        out, *_ = self._run([(13, NOW, "WATCH", None, None, None, "WATCH", None, None),
                             (14, NOW, "WATCH", 0, None, None, "WATCH", None, None)])
        assert out[0]["conviction"] is None and out[1]["conviction"] == 0

    def test_an_error_raises_for_the_caller_and_returns_the_conn(self):
        out, _, conn, put = self._run(error=RuntimeError("timeout"))
        assert isinstance(out, RuntimeError) and put.called


# ---- routing is unchanged ----

def _route(shares=2, enabled=True, read=None, read_error=None, entry_gate=False,
           new_fact_gate=None, price=100.0):
    sizing = {"shares_initial": shares, "initial_dollar": shares * price, "position_size_pct": 0.1,
              "order_pct": 0.1, "initial_entry_pct": 0.55, "flags": [] if shares else ["too small"],
              "stop_loss_price": price * 0.9, "atr_stop": 4.0}
    result = _result()
    record = crew._same_day_consensus_record
    with ExitStack() as st:
        st.enter_context(patch("agents.crew._same_day_consensus_record",       # clock at NOW
                               side_effect=lambda t, r, now=None: record(t, r, now=NOW)))
        st.enter_context(patch("config.SAME_DAY_CONSENSUS_RECORD_ENABLED", enabled))
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=price))
        st.enter_context(patch("agents.crew.new_fact_gate", return_value=new_fact_gate))
        gv = st.enter_context(patch("db.get_same_day_verdicts", side_effect=read_error)
                              if read_error else
                              patch("db.get_same_day_verdicts",
                                    return_value=DAY if read is None else read))
        if entry_gate:
            st.enter_context(patch("db.get_last_broker_exit", return_value={
                "exit_at": "x", "hours_since": 30.0, "lots": 1, "lot_outcome_pct": [-20.0],
                "lot_alpha_pct": [-15.0], "lot_convictions": [7], "entries_since_exit": 0}))
            st.enter_context(patch("config.STOPOUT_REENTRY_GATE_DAYS", 10))
            st.enter_context(patch("config.STOPOUT_REENTRY_SHADOW_ENABLED", True))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        ls = st.enter_context(patch("db.log_signal"))
        ex = st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = copy.deepcopy(sizing)
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, rm_cls.return_value)
    return {"approval": lpa.call_args.kwargs if lpa.called else None,
            "signals": [(c.args, c.kwargs) for c in ls.call_args_list],
            "sized": [(c.args, c.kwargs) for c in rm.size_position.call_args_list],
            "executed": [(c.args, c.kwargs) for c in ex.call_args_list], "read": gv}


_WALL_CLOCK = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00$")


def _clockless(v):
    """v with wall-clock ISO times (e.g. arrival.sizing_at, written at the
    moment of the run) replaced by a marker, so two runs compare."""
    if isinstance(v, dict):
        return {k: _clockless(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_clockless(x) for x in v)
    if isinstance(v, str) and _WALL_CLOCK.match(v):
        return "<wall clock>"
    return v


def _without_record(out):
    """The route's outputs with the record removed: what it decides."""
    out = copy.deepcopy({k: v for k, v in out.items() if k != "read"})
    if out["approval"]:
        out["approval"]["risk_json"].pop("same_day_consensus", None)
    for _, kw in out["signals"]:
        if isinstance(kw.get("decision"), dict):
            kw["decision"].pop("same_day_consensus", None)
    return _clockless(out)


class TestRoutingUnchanged:
    @pytest.mark.parametrize("case", [
        {}, {"shares": 0}, {"entry_gate": True},
        {"new_fact_gate": {"passed": False, "reason": "no new fact"}},
        {"read": []}, {"read_error": RuntimeError("down")},
        {"read": DAY + [_v(21, "OPEN", 9, 3.0, final_pass=True, minutes_ago=700)]},
        {"read": [_v(21, "OPEN", 9, 3.0, final_pass=True, minutes_ago=700)]}])
    def test_the_route_decides_the_same_with_and_without_the_record(self, case):
        on, off = _route(enabled=True, **case), _route(enabled=False, **case)
        assert not off["read"].called
        assert _without_record(on) == _without_record(off)

    def test_the_approval_carries_it(self):
        on = _route()
        rec = on["approval"]["risk_json"]["same_day_consensus"]
        assert rec["read"] == "ok" and rec["n_earlier"] == 5 and rec["n_buy_grade"] == 1
        assert rec["n_earlier_24h"] == 5
        assert on["approval"]["action"] == "OPEN"
        assert "same_day_consensus" not in _route(enabled=False)["approval"]["risk_json"]

    @pytest.mark.parametrize("case, kind", [({"shares": 0}, "ROUTE_ZERO_SHARES"),
                                            ({"entry_gate": True}, "ENTRY_GATE_BLOCKED"),
                                            ({"new_fact_gate": {"passed": False, "reason": "x"}},
                                             "ROUTE_NO_NEW_FACT")])
    def test_the_row_that_ends_the_route_carries_it(self, case, kind):
        on = _route(**case)
        rows = {a[1]: kw for a, kw in on["signals"]}
        assert rows[kind]["decision"]["same_day_consensus"]["n_earlier"] == 5

    def test_the_micro_name_cap_row_carries_it_from_sizing(self):
        rm = MagicMock()
        rm.micro_name_cap.return_value = {"cap_pct": 5.0, "held_pct": 6.0, "shares_after": 0,
                                          "outcome": "capped"}
        sizing = {"shares_initial": 1, "flags": [], "same_day_consensus": {"read": "ok"}}
        with patch("db.log_signal") as ls:
            assert crew._apply_micro_name_cap(rm, "NVDA", sizing, 100.0, _result()) is True
        d = next(c.kwargs for c in ls.call_args_list if c.args[1] == "ROUTE_MICRO_NAME_CAP")
        assert d["decision"]["same_day_consensus"] == {"read": "ok"}

    def test_a_record_that_raises_inside_does_not_stop_the_route(self):
        with patch("statistics.median", side_effect=RuntimeError("x")):
            on = _route()
        assert on["approval"]["risk_json"]["same_day_consensus"]["read"] == "error: RuntimeError"
        assert _without_record(on) == _without_record(_route(enabled=False))


def test_sells_are_not_recorded():
    with patch("agents.crew._same_day_consensus_record") as rc, \
         patch("db.has_active_approval", return_value=False), \
         patch("agents.crew._get_positions", return_value=None), \
         patch("db.log_signal"), patch("alerts._send"):
        crew._route_to_approval(_result(action="EXIT"), {"regime": "Neutral"}, MagicMock())
    rc.assert_not_called()


# ---- the analyst's input and its hashes ----

def test_the_template_hashes_do_not_read_the_changed_code():
    from agents import analyst
    assert not {"agents.crew", "crew", "db", "config"} & set(analyst._BRIEF_MODULES)
    assert "get_same_day_verdicts" not in analyst._PROMPT_DB_FUNCS
    for text in (analyst._prompt_template_full_text(), analyst._prompt_db_queries_text()):
        assert "same_day_consensus" not in text and "get_same_day_verdicts" not in text
    for m in analyst._BRIEF_MODULES:
        assert "same_day_consensus" not in (analyst._module_source(m) or "")


def test_nothing_the_analyst_or_brief_reads_names_it():
    import pathlib
    root = pathlib.Path(crew.__file__).resolve().parent.parent
    named = set()
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel.parts[0] in ("tests", "venv", ".venv", ".git", ".claude"):
            continue
        if "same_day_consensus" in path.read_text(encoding="utf-8"):
            named.add(rel.as_posix())
    assert named == {"agents/crew.py", "config.py", "scripts/batch_verify.py",
                     "scripts/shadow_registry_check.py"}, named      # readouts


# ---- batch_verify ----

def _bv():
    import importlib.util
    import pathlib
    path = pathlib.Path(crew.__file__).resolve().parent.parent / "scripts" / "batch_verify.py"
    spec = importlib.util.spec_from_file_location("batch_verify_sdc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_verify_stage():
    bv = _bv()
    rec, _ = _rec(DAY)
    ok = [{"id": 1, "action": "OPEN", "is_test": False, "risk_json": {"same_day_consensus": rec}},
          {"id": 2, "action": "EXIT", "is_test": False, "risk_json": {}},
          {"id": 3, "action": "OPEN", "is_test": True, "risk_json": {}}]
    gate = [{"id": 9, "decision": {"same_day_consensus": rec}}]
    c = bv.check_same_day_consensus(ok, gate, True)
    assert c["status"] == "OK" and c["rows"] == 1 and c["reads"]["read"] == {"ok": 1}
    assert "carrying it: 1" in " ".join(c["notes"])
    assert bv.check_same_day_consensus([], [], True)["status"] == "WARN"
    missing = [{"id": 4, "action": "ADD", "risk_json": {}}]
    assert bv.check_same_day_consensus(missing, [], True)["missing"] == [4]
    bad = dict(rec)
    bad.pop("n_buy_grade")
    c = bv.check_same_day_consensus([{"id": 5, "action": "OPEN", "risk_json": {
        "same_day_consensus": bad}}], [], True)
    assert c["status"] == "FAIL" and c["missing"] == [5]
    skipped = dict(rec, read="skipped_budget")
    c = bv.check_same_day_consensus([{"id": 6, "action": "OPEN", "risk_json": {
        "same_day_consensus": skipped}}], [], True)
    assert c["status"] == "OK" and c["reads"]["read"] == {"skipped_budget": 1}
    off = bv.check_same_day_consensus(missing, [], False)
    assert off["status"] == "OK" and off["rows"] == 0
    assert set(bv.SAME_DAY_REQUIRED) == set(rec)
