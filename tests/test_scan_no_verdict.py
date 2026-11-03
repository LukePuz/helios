"""Scan candidates whose analysis was lost get a NO_VERDICT signals row."""
from unittest.mock import MagicMock, patch

import pytest

from agents import crew, scan_guard

_CANDS = [{"ticker": "AAA", "priority": "HIGH", "composite_score": 80},
          {"ticker": "BBB", "priority": "HIGH", "composite_score": 70},
          {"ticker": "CCC", "priority": "MEDIUM", "composite_score": 60}]


def _stub(ticker, reason):
    return {"ticker": ticker, "action": "WATCH", "conviction": 1, "pass_to_risk": False,
            "model_used": "stub", "reasoning": f"Stub: {reason}. Model (m) not available."}


def _run(batch):
    log_signal = MagicMock()
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst.analyze_batch", batch), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch.object(crew, "_get_positions", MagicMock(return_value={})), \
         patch.object(crew, "_route_to_approval"), \
         patch("db.log_signal", log_signal):
        crew.process_candidates(_CANDS)
    return log_signal


def _rows(log_signal):
    return [(c.args[0], c.args[1], c.args[3], c.kwargs["decision"]) for c in log_signal.call_args_list]


def test_missing_result_gets_a_no_verdict_row_and_an_exception_stub_does_not():
    """An analyze() exception already has its ANALYST_FAILED row (written by
    analyze() itself), so its stub gets no second row here."""
    batch = MagicMock(return_value=[
        {"ticker": "AAA", "action": "WATCH", "conviction": 5, "pass_to_risk": False, "model_used": "m",
         "reasoning": "fine"},
        _stub("BBB", "Exception: boom"),
    ])
    rows = _rows(_run(batch))
    assert [(t, k) for t, k, _, _ in rows] == [("CCC", "NO_VERDICT")]
    assert rows[0][2] == "no result returned for candidate"
    assert rows[0][3]["priority"] == "MEDIUM" and rows[0][3]["composite_score"] == 60


def test_analyze_exception_in_a_scan_writes_exactly_one_row():
    """End to end through the real analyze_batch: one ANALYST_FAILED, no NO_VERDICT."""
    log_signal = MagicMock()
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst._analyze_inner", side_effect=ValueError("boom")), \
         patch("agents.analyst._stub_recommendation",
               side_effect=lambda c, r: _stub(c["ticker"], r)), \
         patch("context_engine.build_scan_brief", return_value=""), \
         patch("agents.crew._get_price", return_value=None), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch.object(crew, "_get_positions", MagicMock(return_value={})), \
         patch.object(crew, "_route_to_approval"), \
         patch("db.log_signal", log_signal):
        crew.process_candidates(_CANDS[:1])
    assert [c.args[1] for c in log_signal.call_args_list] == ["ANALYST_FAILED"]


def test_model_failure_stub_is_not_a_no_verdict():
    """Ollama/parse stubs are the analyst's own failure path, not lost crew work."""
    batch = MagicMock(return_value=[_stub("AAA", "Ollama timed out"),
                                    _stub("BBB", "JSON parse failed"),
                                    {"ticker": "CCC", "action": "WATCH", "model_used": "m", "reasoning": ""}])
    assert _rows(_run(batch)) == []


def test_batch_exception_writes_a_row_per_candidate_and_reraises():
    batch = MagicMock(side_effect=RuntimeError("worker died"))
    log_signal = MagicMock()
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst.analyze_batch", batch), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch.object(crew, "_get_positions", MagicMock(return_value={})), \
         patch("db.log_signal", log_signal), pytest.raises(RuntimeError):
        crew.process_candidates(_CANDS)
    rows = _rows(log_signal)
    assert [t for t, _, _, _ in rows] == ["AAA", "BBB", "CCC"]
    assert all(k == "NO_VERDICT" and r == "analyze_batch raised: RuntimeError: worker died"
               for _, k, r, _ in rows)


def test_no_verdict_log_never_raises():
    with patch("db.log_signal", side_effect=RuntimeError("db down")):
        scan_guard.log_no_verdict("AAA", "x")


def test_no_verdict_is_not_a_verdict():
    import db
    assert "NO_VERDICT" in db._NON_ANALYST_SIGNALS


def test_a_stub_that_raises_in_analyze_batch_keeps_one_row_per_lost_candidate():
    """Roadmap 00 (f): if the fallback stub raised, analyze_batch raised mid-batch
    and the crew logged a NO_VERDICT for every candidate, a second row for the
    one analyze() had already recorded and a false one for the one analysed."""
    ok = {"ticker": "AAA", "action": "WATCH", "conviction": 5, "pass_to_risk": False,
          "model_used": "m", "reasoning": "fine"}

    def inner(candidate, *a, **k):
        if candidate["ticker"] == "AAA":
            return ok
        raise ValueError("boom")

    log_signal = MagicMock()
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst._analyze_inner", side_effect=inner), \
         patch("agents.analyst._stub_recommendation", side_effect=RuntimeError("stub broke")), \
         patch("context_engine.build_scan_brief", return_value=""), \
         patch("agents.crew._get_price", return_value=None), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch.object(crew, "_get_positions", MagicMock(return_value={})), \
         patch.object(crew, "_route_to_approval") as route, \
         patch("db.log_signal", log_signal):
        results = crew.process_candidates(_CANDS[:2])
    assert [(c.args[0], c.args[1]) for c in log_signal.call_args_list] == [("BBB", "ANALYST_FAILED")]
    assert [r["ticker"] for r in results] == ["AAA", "BBB"]
    assert results[1]["pass_to_risk"] is False
    route.assert_not_called()


def test_a_failing_batch_error_log_still_reraises_the_batch_exception():
    """Roadmap 00 (g): an exception from log_lost_candidates replaced the batch's
    own, so the caller saw the logging error instead of what killed the batch."""
    batch = MagicMock(side_effect=RuntimeError("worker died"))
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst.analyze_batch", batch), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch.object(crew, "_get_positions", MagicMock(return_value={})), \
         patch("agents.scan_guard.log_lost_candidates", side_effect=ValueError("db down")), \
         patch.object(crew.log, "warning") as warn, \
         pytest.raises(RuntimeError, match="worker died"):
        crew.process_candidates(_CANDS)
    assert any("NO_VERDICT logging failed" in c.args[0] for c in warn.call_args_list)
