"""A raise inside _route_to_approval, after the verdict row is written, must
not reach the Celery task: its retry re-ran the 72B analysis (a second
verdict, a second route). It gets a ROUTE_ERROR row instead."""
from unittest.mock import MagicMock, patch

import pytest

import db
import tasks
from agents import crew
from helios3d.replay import harness

_EVENT = {"ticker": "NVDA", "source": "news", "title": "NVIDIA signs supply deal", "text": "x"}
_PASS = {"action": "OPEN", "ticker": "NVDA", "conviction": 8, "pass_to_risk": True,
         "provenance": {"model": "qwen72b", "prompt_sha": "abc"}}


def _route_raises(*a, **k):
    raise KeyError("portfolio_value")


def test_deep_analyze_event_routing_raise_is_not_retried():
    analyze_calls, verdicts = [], []

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        analyze_calls.append(candidate["ticker"])
        verdicts.append("OPEN")                       # analyze() writes the verdict row
        return dict(_PASS)
    with patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch("agents.crew._get_positions", return_value={}), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("context_engine.assemble_brief", return_value="brief"), \
         patch("agents.analyst.analyze", fake_analyze), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("agents.crew._route_to_approval", side_effect=_route_raises), \
         patch("db.log_signal") as log_signal, \
         patch.object(tasks.deep_analyze_event, "retry", side_effect=AssertionError("retried")) as retry:
        tasks.deep_analyze_event(dict(_EVENT), {"relevant": True, "new_fact": True})
    retry.assert_not_called()
    assert analyze_calls == ["NVDA"] and verdicts == ["OPEN"]
    rows = [c for c in log_signal.call_args_list if c.args[1] == "ROUTE_ERROR"]
    assert len(rows) == 1
    kw = rows[0].kwargs
    assert kw["decision"]["route_exit"] == "ROUTE_ERROR"
    assert kw["decision"]["error_class"] == "KeyError"
    assert kw["model"] == "qwen72b" and kw["provenance"]["prompt_sha"] == "abc"


def test_process_candidates_routing_raise_routes_the_rest_and_is_not_retried():
    results = [dict(_PASS, ticker="NVDA"), dict(_PASS, ticker="AMD")]
    routed = []

    def route(result, regime, client):
        routed.append(result["ticker"])
        if result["ticker"] == "NVDA":
            raise KeyError("portfolio_value")
    with patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch("agents.crew._get_positions", return_value={}), \
         patch("agents.analyst.analyze_batch", return_value=results) as batch, \
         patch("agents.scan_guard.log_lost_candidates"), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.crew._route_to_approval", side_effect=route), \
         patch("db.log_signal") as log_signal, \
         patch.object(tasks.analyze_candidates, "retry", side_effect=AssertionError("retried")) as retry:
        tasks.analyze_candidates([{"ticker": "NVDA", "priority": "HIGH"},
                                  {"ticker": "AMD", "priority": "HIGH"}])
    retry.assert_not_called()
    assert batch.call_count == 1 and routed == ["NVDA", "AMD"]
    rows = [c for c in log_signal.call_args_list if c.args[1] == "ROUTE_ERROR"]
    assert [c.args[0] for c in rows] == ["NVDA"]


@pytest.mark.parametrize("fn", ["review_held_position", "review_plan_level"])
def test_held_reviews_write_route_error(fn):
    with patch("agents.crew._analyze_held_position", return_value=dict(_PASS)), \
         patch("agents.crew._capture_latest_thesis"), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch("agents.crew._get_positions", return_value={"NVDA": {}}), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("db.get_state", return_value="{}"), patch("db.set_state"), \
         patch("agents.crew._route_to_approval", side_effect=_route_raises), \
         patch("db.log_signal") as log_signal:
        if fn == "review_held_position":
            out = crew.review_held_position("NVDA", MagicMock(), {"NVDA": {}}, {"regime": "Neutral"})
        else:
            out = crew.review_plan_level("NVDA", "stop", "90", 91.0)
    assert out["action"] == "OPEN"
    assert [c.args[1] for c in log_signal.call_args_list] == ["ROUTE_ERROR"]


def test_route_error_registered():
    assert "ROUTE_ERROR" in db._NON_ANALYST_SIGNALS
    assert "ROUTE_ERROR" in db._RECORD_ONLY_SIGNALS
    assert "ROUTE_ERROR" in harness.ROUTE_TERMINAL


# ── alert ────────────────────────────────────────────────────────────────────

def _state_patches(state):
    return (patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)),
            patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)))


def test_route_error_sends_one_alert_per_ticker_per_hour():
    import alert_policy
    from datetime import datetime, timedelta, timezone
    state = {}
    t0 = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    get, put = _state_patches(state)
    with patch("alerts._send", return_value=11) as send, get, put:
        assert crew._alert_route_error("NVDA", "KeyError", now=t0) is True
        assert crew._alert_route_error("NVDA", "KeyError", now=t0 + timedelta(minutes=59)) is False
        assert crew._alert_route_error("AMD", "KeyError", now=t0 + timedelta(minutes=1)) is True
        assert crew._alert_route_error("NVDA", "KeyError", now=t0 + timedelta(minutes=61)) is True
    assert send.call_count == 3
    text = send.call_args_list[0].args[0]
    assert text == ("NVDA: routing raised KeyError after the verdict; "
                    "check for a partial swap or order")
    assert send.call_args_list[0].kwargs.get("severity") != alert_policy.LOW
    assert alert_policy.classify(text)[0] == alert_policy.CRITICAL


def test_failed_send_does_not_start_the_throttle():
    state = {}
    get, put = _state_patches(state)
    with patch("alerts._send", return_value=None), get, put:
        assert crew._alert_route_error("NVDA", "KeyError") is False
    assert state == {}


def test_alert_failure_never_raises():
    with patch("db.get_state", side_effect=RuntimeError("db down")), patch("alerts._send") as send:
        assert crew._alert_route_error("NVDA", "KeyError") is False
    send.assert_not_called()
    with patch("db.get_state", return_value=None), \
         patch("alerts._send", side_effect=RuntimeError("telegram down")):
        assert crew._alert_route_error("NVDA", "KeyError") is False


def test_routing_raise_sends_the_alert():
    with patch("agents.crew._route_to_approval", side_effect=_route_raises), \
         patch("db.log_signal"), patch("db.get_state", return_value=None), patch("db.set_state"), \
         patch("alerts._send", return_value=1) as send:
        crew._route_guarded(dict(_PASS), {"regime": "Neutral"}, MagicMock())
    assert "NVDA: routing raised KeyError after the verdict" in send.call_args.args[0]
