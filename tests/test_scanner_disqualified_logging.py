"""The scanner logs each disqualified ticker with its reason, and its bar
fetch, earnings check and DB log failures at WARNING (they were debug or
silent, so a dead scan looked like a quiet one)."""
import logging
from unittest.mock import MagicMock, patch

from agents import scanner

_REGIME = {"regime": "Neutral", "score": 50}
_BARS = [{"date": "2026-11-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]


def _scan(caplog, bars_by_ticker, near_earnings=None, log_candidate=None):
    with patch("knowledge_loader.load_watchlist",
               return_value={t: {"role": "Accelerator"} for t in bars_by_ticker}), \
         patch("agents.scanner._get_recent_filings", return_value=set()), \
         patch("agents.scanner._fetch_bars", side_effect=lambda t, days: bars_by_ticker[t]), \
         patch("agents.market_regime.completed_bars", side_effect=lambda b: b), \
         patch("config.EARNINGS_ENTRY_BLACKOUT_DAYS", 5), \
         patch("db.is_near_earnings", **(near_earnings or {"return_value": False})), \
         patch("agents.scanner._score_technical", return_value=(1, "")), \
         patch("agents.scanner._score_momentum_with_flow", return_value=(1, "")), \
         patch("agents.scanner._score_catalyst", return_value=(1, "", "event")), \
         patch("agents.scanner._score_thesis_alignment", return_value=(1, "")), \
         patch("agents.setup_gates.classify_setup", return_value=(None, {})), \
         patch("db.log_scanner_candidate", **(log_candidate or {})), \
         caplog.at_level(logging.INFO, logger="agents.scanner"):
        return scanner.run_scan(dict(_REGIME))


def test_disqualified_tickers_are_logged_with_reason(caplog):
    out = _scan(caplog, {"AAA": [], "BBB": _BARS})
    assert {d["ticker"] for d in out["disqualified"]} == {"AAA", "BBB"}
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("[AAA] Disqualified: no price data" in m for m in warn)
    assert any("[BBB] Disqualified: Score 4/100 below threshold" in m for m in info)


def test_earnings_check_and_db_log_failures_warn(caplog):
    _scan(caplog, {"BBB": _BARS}, near_earnings={"side_effect": RuntimeError("db down")},
          log_candidate={"side_effect": RuntimeError("db down")})
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("[BBB] Earnings blackout check failed" in m for m in warn)
    assert any("[BBB] Earnings catalyst cap check failed" in m for m in warn)
    assert any("Scanner DB log failed for BBB" in m for m in warn)


def test_bar_fetch_failure_warns(caplog, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    with patch("agents.scanner.requests.get", side_effect=RuntimeError("timeout")), \
         caplog.at_level(logging.WARNING, logger="agents.scanner"):
        assert scanner._fetch_bars("AAA") == []
    assert any("Bar fetch failed for AAA" in r.getMessage() for r in caplog.records)
