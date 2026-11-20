"""Entry-quality shadow (agents/crew.py:_entry_quality_shadow): one record-only
ENTRY_QUALITY_SHADOW row per OPEN/ADD that passed the rebuy cooldown, with
the trigger's screener new_fact and price structure from the cached daily
bars. Synthetic bars only: no database, no network. Routing must not change."""
import copy
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
import db
from agents import crew

# 10:00 ET on 2026-11-19, before the close: a bar dated that day is unfinished.
_NOW_OPEN = datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)
_NOW_CLOSED = datetime(2026, 11, 19, 21, 30, tzinfo=timezone.utc)   # 16:30 ET
_LAST_DAY = date(2026, 11, 18)


@pytest.fixture(autouse=True)
def _shadow_on(monkeypatch):
    # conftest turns the shadow off for every other test.
    monkeypatch.setattr(config, "ENTRY_QUALITY_SHADOW_ENABLED", True)


def _bars(closes, spread=1.0, last_day=_LAST_DAY):
    n = len(closes)
    return [{"date": (last_day - timedelta(days=n - 1 - i)).isoformat(),
             "open": float(c), "high": float(c) + spread, "low": float(c) - spread,
             "close": float(c), "volume": 1000} for i, c in enumerate(closes)]


def _path(points):
    """Closes along straight lines through (index, price) points."""
    out = []
    for (i0, p0), (i1, p1) in zip(points, points[1:]):
        for i in range(i0, i1):
            out.append(p0 + (p1 - p0) * (i - i0) / (i1 - i0))
    out.append(points[-1][1])
    return out


def _result(action="OPEN", conviction=8, trigger=None):
    prov = {"prompt_sha": "abc", "model": "m"}
    if trigger is not None:
        prov["trigger"] = trigger
    return {"ticker": "NVDA", "action": action, "conviction": conviction,
            "thesis_bucket": "Moonshot", "reasoning": "r", "pass_to_risk": True,
            "provenance": prov}


def _rows(mock_log):
    return [c for c in mock_log.call_args_list if c[0][1] == "ENTRY_QUALITY_SHADOW"]


def _one(bars=None, now=_NOW_OPEN, result=None, **patch_kw):
    if not patch_kw:
        patch_kw = {"return_value": bars}
    with patch("db.get_price_history", **patch_kw) as gp, patch("db.log_signal") as ls:
        crew._entry_quality_shadow("NVDA", result or _result(), now=now)
    rows = _rows(ls)
    assert len(rows) == 1
    return rows[0], gp


class TestTrendCheck:
    def test_sma_math_on_completed_bars(self):
        closes = list(range(1, 251))   # 250 bars, rising
        today = {"date": "2026-11-19", "open": 9999.0, "high": 9999.0, "low": 9999.0,
                 "close": 9999.0, "volume": 1}
        (args, kw), gp = _one(_bars(closes) + [today])
        assert args[0] == "NVDA" and args[2] == 0.0
        assert gp.call_args[0] == ("NVDA",) and gp.call_args[1] == {"days": 260, "strict": True}
        tc = kw["decision"]["trend_check"]
        assert tc["read"] == "ok" and tc["partial_bar_dropped"] is True
        assert tc["bars"] == 250 and tc["rows_read"] == 251
        assert tc["last_bar_date"] == "2026-11-18" and tc["last_close"] == 250
        assert tc["sma50"] == pytest.approx(sum(range(201, 251)) / 50)     # 225.5
        assert tc["sma200"] == pytest.approx(sum(range(51, 251)) / 200)    # 150.5
        assert tc["above_sma50"] is True and tc["below_sma50"] is False
        assert tc["above_sma200"] is True and tc["below_sma200"] is False
        assert tc["pct_vs_sma50"] == round((250 / 225.5 - 1) * 100, 2)
        assert tc["pct_vs_sma200"] == round((250 / 150.5 - 1) * 100, 2)
        assert kw["decision"]["would_block_trend"] == {
            "below_200dma": False, "below_50dma": False, "drawdown_gt_3atr": False}

    def test_bar_age_in_calendar_days(self):
        closes = list(range(1, 251))
        (_, kw), _ = _one(_bars(closes, last_day=date(2026, 11, 12)))
        tc = kw["decision"]["trend_check"]
        assert tc["read"] == "ok" and tc["bar_age_days"] == 7   # 11-12 to 11-19 ET

    def test_inf_bar_is_dropped_not_used(self):
        bars = _bars(list(range(1, 251)))
        bars[-5] = {**bars[-5], "high": float("inf")}
        (_, kw), _ = _one(bars)
        tc = kw["decision"]["trend_check"]
        assert tc["nonfinite_bars"] == 1 and tc["nonfinite_dropped"] == 1
        assert tc["bars"] == 249
        # The 20-day high comes from a finite bar, so the label is computed.
        assert tc["high_20d"] != float("inf")
        assert kw["decision"]["would_block_trend"]["drawdown_gt_3atr"] is False

    def test_after_the_close_todays_bar_counts(self):
        closes = list(range(1, 251))
        (_, kw), _ = _one(_bars(closes, last_day=date(2026, 11, 19)), now=_NOW_CLOSED)
        tc = kw["decision"]["trend_check"]
        assert tc["partial_bar_dropped"] is False and tc["bars"] == 250
        assert tc["last_bar_date"] == "2026-11-19"

    def test_below_both_averages(self):
        closes = list(range(250, 0, -1))   # falling to 1
        (_, kw), _ = _one(_bars(closes, spread=0.25))
        tc = kw["decision"]["trend_check"]
        assert tc["sma50"] == pytest.approx(sum(range(1, 51)) / 50)
        assert tc["sma200"] == pytest.approx(sum(range(1, 201)) / 200)
        wb = kw["decision"]["would_block_trend"]
        assert wb["below_200dma"] is True and wb["below_50dma"] is True

    @pytest.mark.parametrize("last_close,gt3", [(93.0, True), (94.0, False)])
    def test_drawdown_in_atr_units(self, last_close, gt3):
        closes = [100.0] * 40 + [last_close]
        (_, kw), _ = _one(_bars(closes))
        tc = kw["decision"]["trend_check"]
        # 13 true ranges of 2, then the last bar's gap down from 100.
        last_tr = 100.0 - (last_close - 1.0)
        atr = (13 * 2.0 + last_tr) / 14
        assert tc["atr14"] == round(atr, 4)
        assert tc["high_20d"] == 101.0
        assert tc["drawdown_from_20d_high_atr"] == pytest.approx((101.0 - last_close) / atr)
        assert tc["drawdown_from_20d_high_pct"] == round((last_close / 101.0 - 1) * 100, 2)
        assert kw["decision"]["would_block_trend"]["drawdown_gt_3atr"] is gt3
        # 41 bars: the averages are not computed on a partial window.
        assert tc["read"] == "insufficient_bars"
        assert tc["sma50"] is None and tc["sma50_reason"] == "41 completed bars, 50 needed"
        assert tc["sma200"] is None and tc["sma200_reason"] == "41 completed bars, 200 needed"
        assert kw["decision"]["would_block_trend"]["below_50dma"] is None

    def test_insufficient_bars(self):
        (_, kw), _ = _one(_bars([100.0] * 10))
        tc = kw["decision"]["trend_check"]
        assert tc["read"] == "insufficient_bars" and tc["bars"] == 10
        assert tc["drawdown_from_20d_high_atr"] is None
        assert tc["drawdown_reason"] == "10 completed bars, 20 needed"
        assert tc["swing_structure"] is None and tc["last_swing_high_lower"] is None
        assert tc["swing_reason"] == "10 completed bars, 60 needed"
        assert kw["decision"]["would_block_trend"] == {
            "below_200dma": None, "below_50dma": None, "drawdown_gt_3atr": None}

    def test_empty_cache_is_no_cached_bars_and_fetches_nothing(self):
        with patch("requests.get") as rg, \
             patch("agents.market_regime._fetch_bars") as mf, \
             patch("ingestion.price_history._fetch_bars") as pf:
            (_, kw), _ = _one([])
        assert kw["decision"]["trend_check"]["read"] == "no_cached_bars"
        rg.assert_not_called()
        mf.assert_not_called()
        pf.assert_not_called()

    def test_read_error_records_the_error(self):
        (_, kw), _ = _one(side_effect=ConnectionError("down"))
        tc = kw["decision"]["trend_check"]
        assert tc["read"] == "error: ConnectionError"
        assert tc["bars"] == 0 and tc["sma200"] is None
        assert kw["decision"]["would_block_trend"]["below_200dma"] is None

    def test_lower_swing_high(self):
        closes = _path([(0, 100), (10, 120), (20, 105), (30, 115), (40, 98),
                        (50, 110), (65, 100)])
        (_, kw), _ = _one(_bars(closes, spread=0.5))
        tc = kw["decision"]["trend_check"]
        assert tc["swing_highs"] == [115.5, 110.5]
        assert tc["last_swing_high_lower"] is True
        assert tc["swing_structure"] == "downtrend"

    def test_higher_swing_high(self):
        closes = [300 - c for c in _path([(0, 100), (10, 120), (20, 105), (30, 115),
                                          (40, 98), (50, 110), (65, 100)])]
        (_, kw), _ = _one(_bars(closes, spread=0.5))
        tc = kw["decision"]["trend_check"]
        assert tc["last_swing_high_lower"] is False
        assert tc["swing_structure"] == "uptrend"


class TestOpinionTrigger:
    @pytest.mark.parametrize("nf,false_flag", [(False, True), (True, False), (None, None),
                                               ("false", None)])
    def test_new_fact_from_the_trigger(self, nf, false_flag):
        trig = {"path": "event", "source": "news", "type": "headline", "new_fact": nf}
        (_, kw), _ = _one([], result=_result(trigger=trig))
        ot = kw["decision"]["opinion_trigger"]
        assert ot["trigger_present"] is True and ot["path"] == "event"
        assert ot["source"] == "news" and ot["type"] == "headline"
        assert ot["new_fact"] == (nf if isinstance(nf, bool) else None)
        assert ot["new_fact_false"] is false_flag

    def test_no_trigger_records_null(self):
        (_, kw), _ = _one([])
        ot = kw["decision"]["opinion_trigger"]
        assert ot["trigger_present"] is False
        assert ot["new_fact"] is None and ot["new_fact_false"] is None


class TestShadowRow:
    @pytest.mark.parametrize("action", ["TRIM", "EXIT", "HOLD", "WATCH", "SKIP"])
    def test_non_buy_actions_are_not_read(self, action):
        with patch("db.get_price_history") as gp, patch("db.log_signal") as ls:
            crew._entry_quality_shadow("NVDA", _result(action=action))
        gp.assert_not_called()
        ls.assert_not_called()

    def test_flag_off_skips_the_read(self, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_QUALITY_SHADOW_ENABLED", False)
        with patch("db.get_price_history") as gp, patch("db.log_signal") as ls:
            crew._entry_quality_shadow("NVDA", _result())
        gp.assert_not_called()
        ls.assert_not_called()

    def test_switch_is_a_plain_constant(self):
        src = (Path(__file__).resolve().parent.parent / "config.py").read_text()
        assert re.search(r"^ENTRY_QUALITY_SHADOW_ENABLED = True$", src, re.M)

    @pytest.mark.parametrize("action", ["OPEN", "ADD"])
    def test_copy_marker_and_verdict_fields(self, action):
        (_, kw), _ = _one(_bars(list(range(1, 251))), result=_result(action=action))
        d = kw["decision"]
        assert d["shadow"] == "ENTRY_QUALITY_SHADOW"
        # The provenance is a copy of the verdict's, so the copy marker stays.
        assert d[db._VERDICT_COPY_KEY] == "ENTRY_QUALITY_SHADOW"
        assert d["verdict_action"] == action and d["verdict_conviction"] == 8
        assert kw["model"] == "m" and kw["provenance"]["prompt_sha"] == "abc"

    def test_result_is_not_changed(self):
        result = _result(trigger={"path": "event", "new_fact": False})
        before = copy.deepcopy(result)
        _one(_bars(list(range(1, 251))), result=result)
        assert result == before

    def test_log_failure_never_raises(self):
        with patch("db.get_price_history", return_value=_bars([100.0] * 30)), \
             patch("db.log_signal", side_effect=RuntimeError("db down")):
            crew._entry_quality_shadow("NVDA", _result())   # no exception

    def test_bad_bar_never_raises(self):
        bad = _bars([100.0] * 30)
        bad[-1] = {"date": "2026-11-18", "close": None}
        with patch("db.get_price_history", return_value=bad), patch("db.log_signal") as ls:
            crew._entry_quality_shadow("NVDA", _result(), now=_NOW_OPEN)
        tc = _rows(ls)[0][1]["decision"]["trend_check"]
        # Carried forward from the bar before, as the trade ladder does.
        assert tc["read"] == "insufficient_bars" and tc["nonfinite_bars"] == 1
        assert tc["bars"] == 30 and tc["last_close"] == 100.0

    def test_kind_is_record_only_and_not_a_verdict(self):
        assert "ENTRY_QUALITY_SHADOW" in db._RECORD_ONLY_SIGNALS
        assert "ENTRY_QUALITY_SHADOW" in db._NON_ANALYST_SIGNALS


class TestRoutingUnchanged:
    """The buy reaches sizing the same way with the shadow writing a row, with
    its read failing, and with it off."""

    def _route(self, monkeypatch, enabled=True, **patches):
        monkeypatch.setattr(config, "ENTRY_QUALITY_SHADOW_ENABLED", enabled)
        result = _result(action="OPEN", conviction=7, trigger={"path": "event", "new_fact": False})
        before = copy.deepcopy(result)
        client = MagicMock()
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("db.log_signal") as ls, \
             patch("risk_manager.RiskManager") as rm, \
             patch("db.get_price_history", **patches):
            rm.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, client)
        assert result == before
        others = [(c[0][1], c[0][2]) for c in ls.call_args_list
                  if c[0][1] != "ENTRY_QUALITY_SHADOW"]
        sized = rm.return_value.size_position.call_args_list
        return rm.return_value.size_position.call_count, others, sized, len(_rows(ls))

    def test_same_route_with_row_error_and_off(self, monkeypatch):
        with_row = self._route(monkeypatch, return_value=_bars(list(range(1, 251))))
        with_err = self._route(monkeypatch, side_effect=RuntimeError("x"))
        off = self._route(monkeypatch, enabled=False, return_value=_bars(list(range(1, 251))))
        assert with_row[3] == 1 and with_err[3] == 1 and off[3] == 0
        assert with_row[:3] == with_err[:3] == off[:3]
        assert with_row[0] == 1

    def test_cooldown_skip_writes_no_row(self):
        result = _result(action="ADD", conviction=6)
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval",
                   return_value={"conviction": 6, "decided_at": "2026-11-19T12:00:00Z"}), \
             patch("db.get_price_history", return_value=_bars([100.0] * 30)) as gp, \
             patch("db.log_signal") as ls, \
             patch("risk_manager.RiskManager"):
            crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
        gp.assert_not_called()
        assert [c[0][1] for c in ls.call_args_list] == ["COOLDOWN_SKIPPED"]
