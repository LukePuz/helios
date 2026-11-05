"""An analysis that produces no verdict writes an ANALYST_FAILED signals row
with provenance and the failure kind (M166). Log-only."""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import requests

import db
from agents import analyst

_REGIME = {"regime": "Neutral", "score": 50, "reasoning": ""}
_TRIGGER = {"path": "event", "source": "news", "title": "t"}


def _run(*extra, raises=None):
    log_signal = MagicMock()
    with ExitStack() as stack:
        for p in (patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("agents.analyst._stub_recommendation", return_value={"stub": True}),
                  patch("db.log_signal", log_signal), patch("feed.emit"), *extra):
            stack.enter_context(p)
        if raises:
            with pytest.raises(raises):
                analyst.analyze({"ticker": "NVDA"}, situation_brief="Portfolio Value: $1,234.00",
                                regime_packet=_REGIME, trigger=_TRIGGER)
        else:
            out = analyst.analyze({"ticker": "NVDA"}, situation_brief="Portfolio Value: $1,234.00",
                                  regime_packet=_REGIME, trigger=_TRIGGER)
            assert out == {"stub": True}
    assert log_signal.call_count == 1
    return log_signal.call_args


def _check_row(call, kind):
    args, kw = call
    assert args[:3] == ("NVDA", "ANALYST_FAILED", None)
    assert kw["model"] is None and "prompt" not in kw
    prov, failure = kw["provenance"], kw["decision"]["failure"]
    assert failure["kind"] == kind
    assert prov["trigger"] == _TRIGGER and prov["model"] == "test"
    assert "code_sha" in prov and "config_env" in prov
    return prov, failure


def test_timeout_writes_a_failed_row():
    call = _run(patch("agents.analyst.requests.post", side_effect=requests.exceptions.Timeout()))
    prov, failure = _check_row(call, "timeout")
    assert len(prov["prompt_sha"]) == 12
    assert prov["llm_call"]["options"]["temperature"] == 0.2
    assert failure["ollama_failure"] == "timeout" and "raw_reply_sha" not in failure


def test_unreachable_writes_a_failed_row():
    call = _run(patch("agents.analyst.requests.post", side_effect=requests.exceptions.ConnectionError()))
    _check_row(call, "unreachable")


def test_parse_failure_keeps_a_redacted_head_of_the_reply():
    raw = "not json at all. Portfolio Value: $9,876.00"
    call = _run(patch("agents.analyst._call_ollama", return_value=raw))
    _, failure = _check_row(call, "parse_failed")
    assert failure["raw_reply_len"] == len(raw) and len(failure["raw_reply_sha"]) == 12
    assert "9,876" not in failure["raw_reply_head"]


def test_salvaged_reply_writes_a_failed_row():
    broken = '{"action": "EXIT", "conviction": 2, "reasoning": "x" oops'
    call = _run(patch("agents.analyst._call_ollama", return_value=broken))
    _check_row(call, "salvaged_not_routed")


def test_exception_is_recorded_and_reraised():
    call = _run(patch("agents.analyst._build_prompt", side_effect=RuntimeError("boom")),
                raises=RuntimeError)
    prov, failure = _check_row(call, "exception")
    assert failure["error_type"] == "RuntimeError" and failure["error"] == "boom"
    # Never reached the model: no prompt, and no stale call stats.
    assert prov["prompt_sha"] is None and prov["llm_call"] is None


def test_analyze_batch_records_an_exception_once():
    log_signal = MagicMock()
    with patch("agents.analyst._analyze_inner", side_effect=ValueError("x")), \
         patch("agents.analyst._stub_recommendation", return_value={"stub": True}), \
         patch("context_engine.build_scan_brief", return_value=""), \
         patch("agents.crew._get_price", return_value=None), \
         patch("db.log_signal", log_signal):
        assert analyst.analyze_batch([{"ticker": "AMD"}], regime_packet=_REGIME) == [{"stub": True}]
    assert [c.args[1] for c in log_signal.call_args_list] == ["ANALYST_FAILED"]


def test_stub_that_raises_after_its_row_is_not_recorded_twice():
    log_signal = MagicMock()
    with patch("knowledge_loader.load_thesis", return_value=""), \
         patch("knowledge_loader.load_philosophy", return_value=""), \
         patch("db.get_conviction_calibration", return_value=[]), \
         patch("agents.analyst._call_ollama", return_value="not json"), \
         patch("agents.analyst._stub_recommendation", side_effect=RuntimeError("stub broke")), \
         patch("db.log_signal", log_signal), patch("feed.emit"), pytest.raises(RuntimeError):
        analyst.analyze({"ticker": "NVDA"}, situation_brief="b", regime_packet=_REGIME)
    assert log_signal.call_count == 1
    assert log_signal.call_args.kwargs["decision"]["failure"]["kind"] == "parse_failed"


def test_a_failing_writer_never_raises():
    with patch("db.log_signal", side_effect=RuntimeError("db down")):
        analyst._log_failed_analysis({"ticker": "NVDA"}, "timeout", "Ollama timed out", prompt="p")


def test_failed_rows_are_not_verdicts_for_readers():
    assert "ANALYST_FAILED" in db._NON_ANALYST_SIGNALS
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    cur.fetchall.return_value = []
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.get_recent_signals(10)
        db.recent_event_verdicts("NVDA", 15)
    (recent_sql, recent_params), (event_sql, event_params) = [c.args for c in cur.execute.call_args_list]
    assert "signal NOT IN %s" in recent_sql
    assert set(recent_params[0]) == {"ANALYST_FAILED", "NO_VERDICT", "ROUTE_NO_BROKER",
                                     "ROUTE_RISK_INIT_FAILED", "ROUTE_ERROR",
                                     "SCREENED_IN", "EVENT_LOST", "HELD_MOVE_SHADOW"}
    assert recent_params[1] == 10
    # Routing rows carry the verdict's trigger; they are not extra event verdicts.
    assert "signal NOT IN %s" in event_sql
    assert event_params[2] == db._NON_ANALYST_SIGNALS


def test_setup_gate_corroboration_ignores_record_only_rows():
    """With only a record-only row for the ticker in the window, the catalyst
    gate's corroboration reads as before (no signal), end to end via SQL."""
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    rows = [("NVDA", "ANALYST_FAILED", None, None, ""), ("NVDA", "NO_VERDICT", 0, None, ""),
            ("AMD", "HOLD", 0.6, None, "")]

    def execute(sql, params):
        excluded = params[0]
        cur.fetchall.return_value = [r for r in rows if r[1] not in excluded]
    cur.execute.side_effect = execute
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        got = db.get_recent_signals(50)
    assert [r["symbol"] for r in got] == ["AMD"]
