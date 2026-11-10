"""Verdict provenance puts live ^VIX next to the regime row's VIX (audit
2026-11-07 17:17 #2), read from the Redis value tasks.cache_live_vix keeps
fresh (no Yahoo call or thread on the verdict path; roadmap 04e). Log-only."""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from agents import analyst
import tasks


def _cached(vix, age_s=30, error=None):
    at = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
    return lambda: {"vix": vix, "at": at, "error": error}


def test_gap_between_live_and_regime_vix(monkeypatch):
    monkeypatch.setattr(analyst, "_vix_cache_get", _cached(25.4))
    rec = analyst._vix_live_record(16.61)
    assert rec["vix_regime"] == 16.61 and rec["vix_live"] == 25.4 and rec["vix_gap"] == 8.79
    assert rec["vix_live_at"] and rec["vix_live_status"] == "ok"


def test_stale_missing_failed_and_read_errors(monkeypatch):
    monkeypatch.setattr(analyst, "_vix_cache_get", _cached(25.4, age_s=400))
    rec = analyst._vix_live_record(16.0)
    assert rec["vix_live"] is None and rec["vix_live_status"] == "stale" and rec["vix_live_at"]
    monkeypatch.setattr(analyst, "_vix_cache_get", _cached(None, error="RuntimeError"))
    assert analyst._vix_live_record(16.0)["vix_live_status"] == "RuntimeError"
    monkeypatch.setattr(analyst, "_vix_cache_get", lambda: (_ for _ in ()).throw(ConnectionError()))
    assert analyst._vix_live_record(16.0)["vix_live_status"] == "read_error:ConnectionError"


def test_no_cache_is_countable():
    rec = analyst._vix_live_record(16.0)          # conftest: no Redis value
    assert rec["vix_live_status"] == "no_cache" and "error" not in rec and rec["vix_regime"] == 16.0


def test_decision_inputs_carry_it(monkeypatch):
    monkeypatch.setattr(analyst, "_vix_cache_get", _cached(28.18))
    rp = {"regime": "Neutral", "score": 43, "signals": {"volatility": {"vix": 16.61}}}
    with patch("db.get_trade_ladder", return_value={}):
        rec = analyst._decision_inputs_record("NVDA", rp)
    assert rec["regime_used"]["vix_gap"] == 11.57 and rec["regime_used"]["label"] == "Neutral"


def test_the_task_writes_the_cache():
    client = MagicMock()
    with patch("redis.Redis.from_url", return_value=client), \
         patch("agents.market_regime.get_intraday_vix", return_value=float("nan")):
        rec = tasks.cache_live_vix()
    assert rec["vix"] is None and rec["error"] == "non_finite"
    key, raw = client.set.call_args.args
    assert key == analyst.VIX_LIVE_KEY and json.loads(raw)["error"] == "non_finite"
    assert client.set.call_args.kwargs["ex"] == 900
    with patch("redis.Redis.from_url", return_value=client), \
         patch("agents.market_regime.get_intraday_vix", return_value=24.1):
        assert tasks.cache_live_vix()["vix"] == 24.1
    with patch("redis.Redis.from_url", side_effect=ConnectionError()):
        assert tasks.cache_live_vix() is None


def test_beat_entry():
    e = tasks.app.conf.beat_schedule["cache-live-vix"]
    assert e["task"] == "tasks.cache_live_vix" and e["options"]["queue"] == tasks.INGESTION_QUEUE
