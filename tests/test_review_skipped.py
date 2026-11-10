"""A held-position review skipped because broker positions could not be read
leaves a log-only REVIEW_SKIPPED row, which is not a verdict."""
from unittest.mock import patch

import db
from agents import crew
from helios3d.replay import harness


def test_skipped_review_writes_a_row():
    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=None), \
         patch("agents.analyst.shadow_versions", return_value={"code_sha": "abc"}), \
         patch("agents.crew.review_held_position") as review, \
         patch("db.log_signal") as log_signal:
        assert crew.review_all_held_positions(priority_tickers=["SNOW"]) == []
    review.assert_not_called()
    (row,) = [c for c in log_signal.call_args_list if c.args[1] == "REVIEW_SKIPPED"]
    assert row.args == ("PORTFOLIO", "REVIEW_SKIPPED", 0.0,
                        "held review skipped: broker positions unavailable")
    assert row.kwargs["decision"] == {"task": "review_all_held_positions",
                                      "reason": "positions_unavailable",
                                      "priority_tickers": ["SNOW"]}
    assert row.kwargs["provenance"] == {"code_sha": "abc"}


def test_no_priority_tickers_is_an_empty_list():
    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=None), \
         patch("db.log_signal") as log_signal:
        crew.review_all_held_positions()
    assert log_signal.call_args.kwargs["decision"]["priority_tickers"] == []


def test_failing_log_does_not_raise():
    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=None), \
         patch("db.log_signal", side_effect=RuntimeError("db down")):
        assert crew.review_all_held_positions() == []


def test_kind_is_not_a_verdict():
    assert "REVIEW_SKIPPED" in db._NON_ANALYST_SIGNALS
    assert "REVIEW_SKIPPED" in db._RECORD_ONLY_SIGNALS
    assert "REVIEW_SKIPPED" not in harness.ROUTE_KINDS


def _rows(log_signal):
    return [c for c in log_signal.call_args_list if c.args[1] == "REVIEW_SKIPPED"]


def test_failed_review_task_writes_a_row_on_the_final_attempt_only():
    import pytest
    import tasks
    task = tasks.review_positions
    with patch("tasks._get_client", side_effect=RuntimeError("no client")), \
         patch("tasks._log_exit_routes"), \
         patch("agents.crew.review_all_held_positions", side_effect=ValueError("boom")), \
         patch("agents.analyst.shadow_versions", return_value={"code_sha": "abc"}), \
         patch("db.log_signal") as log_signal:
        for retries in range(task.max_retries):
            task.push_request(retries=retries)
            try:
                with pytest.raises(Exception):
                    task.run()
            finally:
                task.pop_request()
        assert _rows(log_signal) == []
        task.push_request(retries=task.max_retries)
        try:
            with pytest.raises(ValueError):
                task.run()
        finally:
            task.pop_request()
    (row,) = _rows(log_signal)
    assert row.args == ("PORTFOLIO", "REVIEW_SKIPPED", 0.0, "held review failed: ValueError")
    assert row.kwargs["decision"] == {"task": "review_positions", "reason": "review_failed",
                                      "error_class": "ValueError",
                                      "attempts": task.max_retries + 1}
    assert row.kwargs["provenance"] == {"code_sha": "abc"}


def test_failed_review_task_still_raises_when_the_log_fails():
    import pytest
    import tasks
    task = tasks.review_positions
    with patch("tasks._get_client", side_effect=RuntimeError("no client")), \
         patch("tasks._log_exit_routes"), \
         patch("agents.crew.review_all_held_positions", side_effect=ValueError("boom")), \
         patch("db.log_signal", side_effect=RuntimeError("db down")):
        task.push_request(retries=task.max_retries)
        try:
            with pytest.raises(ValueError):
                task.run()
        finally:
            task.pop_request()


def test_failed_ticker_review_writes_a_row_and_the_loop_continues():
    positions = {"AAA": {}, "BBB": {}}
    reviewed = []

    def review(ticker, *a, **k):
        reviewed.append(ticker)
        if ticker == "AAA":
            raise KeyError("x")
        return {"ticker": ticker}

    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=positions), \
         patch("agents.market_regime.get_cached_regime", return_value={}), \
         patch("db.get_state", return_value='{"AAA": "1", "BBB": "2"}'), \
         patch("db.get_earnings_around", return_value={}), \
         patch("db.set_state") as set_state, \
         patch("agents.analyst.shadow_versions", return_value={"code_sha": "abc"}), \
         patch("agents.crew.review_held_position", side_effect=review), \
         patch("db.log_signal") as log_signal:
        assert crew.review_all_held_positions() == [{"ticker": "BBB"}]
    assert reviewed == ["AAA", "BBB"]
    (row,) = _rows(log_signal)
    assert row.args == ("AAA", "REVIEW_SKIPPED", 0.0, "held review failed: KeyError")
    assert row.kwargs["decision"] == {"task": "review_all_held_positions",
                                      "reason": "ticker_review_failed", "error_class": "KeyError"}
    assert row.kwargs["provenance"] == {"code_sha": "abc"}
    import json
    cursor = json.loads(set_state.call_args.args[1])
    assert set(cursor) == {"AAA", "BBB"}            # the cursor still advances past AAA


def test_helper_never_raises():
    with patch("db.log_signal", side_effect=RuntimeError("db down")):
        assert crew.log_review_skipped("PORTFOLIO", "x", {}) is None
    with patch("agents.analyst.shadow_versions", side_effect=RuntimeError("no sha")), \
         patch("db.log_signal") as log_signal:
        assert crew.log_review_skipped("AAA", "x", {}) is None
    log_signal.assert_not_called()


def _review_run(result):
    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value={"NVDA": object()}), \
         patch("agents.crew.get_cached_regime", create=True), \
         patch("agents.market_regime.get_cached_regime", return_value={}), \
         patch("db.get_state", return_value="{}"), patch("db.set_state") as set_state, \
         patch("db.get_earnings_around", return_value={}), \
         patch("agents.crew.review_held_position", return_value=result), \
         patch("agents.analyst.shadow_versions", return_value={"code_sha": "abc"}), \
         patch("db.log_signal") as log_signal:
        out = crew.review_all_held_positions()
    return out, log_signal, set_state


def test_a_stub_review_also_writes_a_skip_row():
    """Audit 2026-11-07 05:07 #2: a stub (no model reply) is not a review."""
    out, log_signal, set_state = _review_run({"model_used": "stub", "signal": "HOLD"})
    (row,) = _rows(log_signal)
    assert row.args == ("NVDA", "REVIEW_SKIPPED", 0.0, "held review skipped: stub verdict (no model reply)")
    assert row.kwargs["decision"] == {"task": "review_all_held_positions", "reason": "stub"}
    assert out == [{"model_used": "stub", "signal": "HOLD"}]
    assert '"NVDA"' in set_state.call_args.args[1]   # the cursor still advances


def test_a_real_review_writes_no_skip_row():
    _, log_signal, _ = _review_run({"model_used": "qwen2.5:72b", "signal": "HOLD"})
    assert _rows(log_signal) == []



def test_an_analyze_exception_also_writes_a_skip_row():
    """SEF 148: _analyze_held_position turns an analyze() raise into a SKIP
    dict, which wrote ANALYST_FAILED only."""
    _, log_signal, _ = _review_run({"action": "SKIP", "review_error": "TimeoutError"})
    (row,) = _rows(log_signal)
    assert row.args == ("NVDA", "REVIEW_SKIPPED", 0.0, "held review skipped: analyze raised TimeoutError")
    assert row.kwargs["decision"] == {"task": "review_all_held_positions", "reason": "analyze_raised",
                                      "error_class": "TimeoutError"}



def test_analyze_raise_is_marked_in_the_skip_dict():
    with patch("context_engine.assemble_brief", return_value="brief"), \
         patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}), \
         patch("agents.analyst.analyze", side_effect=TimeoutError("slow")):
        res = crew._analyze_held_position("SNOW", None, {"SNOW": {"current_price": 100.0}},
                                          {"regime": "Neutral"})
    assert res["action"] == "SKIP" and res["review_error"] == "TimeoutError"
    assert res["pass_to_risk"] is False
