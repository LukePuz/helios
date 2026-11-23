"""Live entry gates (agents/crew.py:_entry_gate): an OPEN/ADD that passed the
rebuy cooldown is skipped with one ENTRY_GATE_BLOCKED row after a recent exit
the bot did not place, or below both the 50- and 200-day averages. Failed
reads, missing averages and stale bars do not block. Synthetic data only."""
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import config
import db
from agents import crew


@pytest.fixture(autouse=True)
def _gates_on(monkeypatch):
    # conftest turns the gates and the entry-quality shadow off elsewhere.
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS", 5)
    monkeypatch.setattr(config, "ENTRY_QUALITY_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", True)
    # Stock-only by default here; TestMarketCondition turns the market read on.
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "")


def _result(action="OPEN", conviction=9):
    return {"ticker": "NVDA", "action": action, "conviction": conviction,
            "thesis_bucket": "Moonshot", "reasoning": "r", "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m"}}


def _trend(below50=True, below200=True, read="ok", age=1):
    return {"read": read, "bar_age_days": age, "below_sma50": below50, "below_sma200": below200,
            "pct_vs_sma50": -30.1 if below50 else 4.0, "pct_vs_sma200": -28.2 if below200 else 9.0}


def _stop(hours=17.5, read="ok"):
    return {"read": read, "hours_since": hours, "exit_at": "2026-11-19T20:56:32+00:00"}


def _blocked_rows(ls):
    return [c for c in ls.call_args_list if c[0][1] == "ENTRY_GATE_BLOCKED"]


def _gate(stop_rec, trend, result=None):
    with patch("db.log_signal") as ls:
        out = crew._entry_gate("NVDA", result or _result(), stop_rec, trend)
    return out, _blocked_rows(ls)


class TestGateDecision:
    @pytest.mark.parametrize("hours,blocked", [(0.5, True), (239.9, True), (240.0, False), (500.0, False)])
    def test_stopout_window(self, hours, blocked):
        out, rows = _gate(_stop(hours), _trend(False, False))
        assert out is blocked and len(rows) == (1 if blocked else 0)

    def test_negative_age_does_not_block(self):
        out, rows = _gate(_stop(-0.5), _trend(False, False))
        assert out is False and not rows

    def test_lookback_covers_a_wider_gate(self, monkeypatch):
        assert crew._stopout_lookback_days() == 10
        monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 14)
        assert crew._stopout_lookback_days() == 14
        with patch("db.get_last_broker_exit", return_value=None) as gx:
            crew._stopout_reentry_shadow("NVDA", _result())
        assert gx.call_args[0] == ("NVDA", 14)

    def test_no_recent_exit_does_not_block(self):
        out, rows = _gate(None, _trend(False, False))
        assert out is False and not rows

    def test_failed_stopout_read_does_not_block(self):
        out, rows = _gate({"read": "error: ConnectionError", "hours_since": None}, _trend(False, False))
        assert out is False and not rows

    @pytest.mark.parametrize("b50,b200,blocked", [(True, True, True), (True, False, False),
                                                  (False, True, False), (None, True, False),
                                                  (True, None, False)])
    def test_trend_needs_both_averages(self, b50, b200, blocked):
        out, rows = _gate(None, _trend(b50, b200))
        assert out is blocked

    @pytest.mark.parametrize("read,age", [("insufficient_bars", 1), ("no_cached_bars", 1),
                                          ("error: RuntimeError", 1), ("ok", 6), ("ok", None),
                                          ("ok", -1), ("ok", True)])
    def test_unusable_trend_read_does_not_block(self, read, age):
        out, rows = _gate(None, _trend(True, True, read=read, age=age))
        assert out is False and not rows

    def test_missing_trend_does_not_block(self):
        assert _gate(None, None)[0] is False

    def test_conviction_does_not_override(self):
        out, _ = _gate(_stop(1.0), _trend(False, False), result=_result(conviction=10))
        assert out is True

    def test_gates_off(self, monkeypatch):
        monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 0)
        monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", False)
        out, rows = _gate(_stop(1.0), _trend(True, True))
        assert out is False and not rows

    def test_row_records_both_gates(self):
        out, rows = _gate(_stop(17.5), _trend(True, True))
        (call,) = rows
        args, kw = call
        assert args[0] == "NVDA" and args[2] == pytest.approx(0.9)
        assert "17.5 h" in args[3] and "-30.1%" in args[3]
        gate = kw["decision"]["entry_gate"]
        assert gate["stopout"]["blocked"] is True and gate["stopout"]["days"] == 10
        assert gate["trend"]["blocked"] is True and gate["trend"]["bar_age_days"] == 1
        assert kw["decision"][db._VERDICT_COPY_KEY] == "ENTRY_GATE_BLOCKED"
        assert kw["decision"]["verdict_action"] == "OPEN"

    def test_log_failure_still_blocks(self):
        # _log_route_exit swallows a failed write; the buy is still skipped.
        with patch("db.log_signal", side_effect=RuntimeError("db down")):
            assert crew._entry_gate("NVDA", _result(), _stop(1.0), None) is True

    def test_internal_error_does_not_block(self):
        class Bad(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")
        assert crew._entry_gate("NVDA", _result(), Bad(), None) is False

    def test_kind_is_routing_not_verdict(self):
        assert "ENTRY_GATE_BLOCKED" in db._NON_ANALYST_SIGNALS
        assert "ENTRY_GATE_BLOCKED" not in db._RECORD_ONLY_SIGNALS

    def test_switches_are_plain_constants(self):
        src = Path(config.__file__).read_text()
        assert re.search(r"^STOPOUT_REENTRY_GATE_DAYS = 10$", src, re.M)
        assert re.search(r"^ENTRY_TREND_GATE_ENABLED = True$", src, re.M)
        assert re.search(r"^ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS = 5$", src, re.M)


class TestMarketCondition:
    @pytest.fixture(autouse=True)
    def _market_on(self, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "SPY")

    def _market(self, below200=True, read="ok", age=1, pct=-5.5):
        return {"read": read, "bar_age_days": age, "below_sma200": below200, "pct_vs_sma200": pct}

    def test_market_downtrend_blocks(self):
        with patch("agents.crew._entry_trend_check", return_value=self._market(True)) as tc:
            out, rows = _gate(None, _trend(True, True))
        assert out is True and tc.call_args[0] == ("SPY",)
        gate = rows[0][1]["decision"]["entry_gate"]["trend"]
        assert gate["market"]["downtrend"] is True and gate["blocked"] is True
        assert "SPY -5.5% vs its 200-day" in rows[0][0][3]

    def test_market_uptrend_does_not_block(self):
        with patch("agents.crew._entry_trend_check", return_value=self._market(False, pct=3.0)):
            out, rows = _gate(None, _trend(True, True))
        assert out is False and not rows

    @pytest.mark.parametrize("read,age,below", [("error: RuntimeError", 1, None), ("ok", 9, True),
                                                ("insufficient_bars", 1, None), ("ok", None, True)])
    def test_unusable_market_read_does_not_block(self, read, age, below):
        with patch("agents.crew._entry_trend_check", return_value=self._market(below, read=read, age=age)):
            assert _gate(None, _trend(True, True))[0] is False

    def test_market_not_read_when_stock_passes(self):
        with patch("agents.crew._entry_trend_check") as tc:
            assert _gate(None, _trend(False, True))[0] is False
        tc.assert_not_called()

    def test_stopout_block_independent_of_market(self):
        with patch("agents.crew._entry_trend_check", return_value=self._market(False)):
            out, rows = _gate(_stop(5.0), _trend(True, True))
        assert out is True
        gate = rows[0][1]["decision"]["entry_gate"]
        assert gate["stopout"]["blocked"] is True and gate["trend"]["blocked"] is False

    def test_market_read_raising_keeps_the_stopout_block(self):
        with patch("agents.crew._entry_trend_check", side_effect=RuntimeError("x")):
            out, rows = _gate(_stop(5.0), _trend(True, True))
        assert out is True
        gate = rows[0][1]["decision"]["entry_gate"]
        assert gate["stopout"]["blocked"] is True and gate["trend"]["blocked"] is False
        assert gate["trend"]["market"]["read"] == "error: RuntimeError"

    @pytest.mark.parametrize("age,below", [(True, True), (1, None)])
    def test_bool_age_or_missing_average_does_not_block(self, age, below):
        with patch("agents.crew._entry_trend_check", return_value=self._market(below, age=age)):
            assert _gate(None, _trend(True, True))[0] is False

    def test_empty_ticker_skips_the_market_read(self, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "")
        with patch("agents.crew._entry_trend_check") as tc:
            assert _gate(None, _trend(True, True))[0] is True
        tc.assert_not_called()

    def test_market_ticker_constant(self):
        src = Path(config.__file__).read_text()
        assert re.search(r'^ENTRY_TREND_GATE_MARKET_TICKER = "SPY"$', src, re.M)


def _bars(closes):
    last = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    n = len(closes)
    return [{"date": (last - timedelta(days=n - 1 - i)).isoformat(), "open": float(c),
             "high": float(c) + 1, "low": float(c) - 1, "close": float(c), "volume": 1000}
            for i, c in enumerate(closes)]


class TestRouting:
    def _route(self, action="OPEN", exit_rec=None, closes=None):
        result = _result(action=action, conviction=7)
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("db.get_last_broker_exit", return_value=exit_rec), \
             patch("db.get_price_history", return_value=_bars(closes or list(range(1, 251)))), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("db.log_signal") as ls, \
             patch("risk_manager.RiskManager") as rm:
            rm.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
        kinds = [c[0][1] for c in ls.call_args_list]
        return rm.return_value.size_position.call_count, kinds

    @pytest.mark.parametrize("action", ["OPEN", "ADD"])
    def test_downtrend_is_skipped_before_sizing(self, action):
        sized, kinds = self._route(action=action, closes=list(range(300, 50, -1)))
        assert sized == 0
        assert kinds.count("ENTRY_GATE_BLOCKED") == 1 and "ENTRY_QUALITY_SHADOW" in kinds

    def test_recent_stopout_is_skipped_before_sizing(self):
        ex = {"exit_at": "x", "hours_since": 30.0, "lots": 1, "lot_outcome_pct": [-20.0],
              "lot_alpha_pct": [-15.0], "lot_convictions": [7], "entries_since_exit": 0}
        sized, kinds = self._route(exit_rec=ex)
        assert sized == 0
        assert kinds.index("STOPOUT_REENTRY_SHADOW") < kinds.index("ENTRY_GATE_BLOCKED")

    def test_uptrend_without_exit_is_sized(self):
        sized, kinds = self._route()
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in kinds

    def test_exit_read_error_still_sized(self):
        result = _result(action="OPEN", conviction=7)
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("db.get_last_broker_exit", side_effect=ConnectionError("down")), \
             patch("db.get_price_history", side_effect=RuntimeError("down")), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("db.log_signal") as ls, \
             patch("risk_manager.RiskManager") as rm:
            rm.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
        assert rm.return_value.size_position.call_count == 1
        assert "ENTRY_GATE_BLOCKED" not in [c[0][1] for c in ls.call_args_list]

    def test_shadow_off_turns_its_gate_off(self, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_QUALITY_SHADOW_ENABLED", False)
        sized, kinds = self._route(closes=list(range(300, 50, -1)))
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in kinds
        monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", False)
        ex = {"exit_at": "x", "hours_since": 30.0, "lots": 1, "lot_outcome_pct": [-20.0],
              "lot_alpha_pct": [-15.0], "lot_convictions": [7], "entries_since_exit": 0}
        sized, kinds = self._route(exit_rec=ex)
        assert sized == 1 and "ENTRY_GATE_BLOCKED" not in kinds

    @pytest.mark.parametrize("action", ["TRIM", "EXIT"])
    def test_sells_are_not_gated(self, action):
        with patch("agents.crew._entry_gate") as g, \
             patch("db.has_active_approval", return_value=False), \
             patch("agents.crew._get_positions", return_value=None), \
             patch("db.log_signal"), patch("alerts._send"):
            crew._route_to_approval(_result(action=action), {"regime": "Neutral"}, MagicMock())
        g.assert_not_called()

    def test_trim_is_not_gated(self):
        with patch("agents.crew._entry_gate") as g, \
             patch("db.has_active_approval", return_value=False), \
             patch("agents.crew._get_positions", return_value=None), \
             patch("db.log_signal"), patch("alerts._send"):
            crew._route_to_approval(_result(action="TRIM"), {"regime": "Neutral"}, MagicMock())
        g.assert_not_called()
