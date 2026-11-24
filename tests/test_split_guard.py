"""Split-discontinuity guard: _entry_trend_check records split_suspect when
the SMA200's closes look like two price scales mixed by an unadjusted split,
corroborated by a persistent volume step (agents/crew.py:_split_discontinuity),
and _entry_gate does not block on such a read. Real crashes and pops must not
flag. Synthetic data only."""
import math
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import config
from agents import crew

N = 250
K = 150          # first bar on the new scale


@pytest.fixture(autouse=True)
def _gates_on(monkeypatch):
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS", 5)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "")


def _wiggle(n=N, base=100.0, amp=0.01):
    """Closes with ~0.5-1% daily moves."""
    return [base * (1 + amp * math.sin(i * 1.7)) for i in range(n)]


def _vols(n=N, base=1e6):
    return [base * (1 + 0.1 * math.sin(i * 2.3)) for i in range(n)]


def _dates(n=N):
    last = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    return [(last - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


def _split(price_step, vol_step, k=K, n=N):
    """Closes and volumes as an unadjusted split leaves them: the new scale
    from bar k, price x price_step and volume x vol_step against the old."""
    closes = [c / price_step if i < k else c for i, c in enumerate(_wiggle(n))]
    vols = [v / vol_step if i < k else v for i, v in enumerate(_vols(n))]
    return closes, vols


def _crash(r, spike, after, spike_bars=3, k=K, n=N):
    """A real one-day move by r at bar k: volume x spike for spike_bars bars,
    then x after for good."""
    closes = [c * r if i >= k else c for i, c in enumerate(_wiggle(n))]
    vols = [v * (spike if k <= i < k + spike_bars else after if i >= k else 1.0)
            for i, v in enumerate(_vols(n))]
    return closes, vols


def _check(closes, vols):
    return crew._split_discontinuity(_dates(len(closes)), closes, vols)


def _bars(closes, vols=None):
    vols = vols or [1e6] * len(closes)
    return [{"date": d, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}
            for d, c, v in zip(_dates(len(closes)), closes, vols)]


class TestDetector:
    def test_clean_series(self):
        out = _check(_wiggle(), _vols())
        assert out["split_suspect"] is False and out["split_suspect_count"] == 0
        assert out["split_candidates"] == 0 and out["split_window_bars"] == 200

    def test_two_for_one_split(self):
        closes, vols = _split(0.5, 2.0)
        out = _check(closes, vols)
        assert out["split_suspect"] is True and out["split_suspect_rule"] == "factor"
        assert out["split_suspect_factor"] == "1/2" and out["split_suspect_date"] == _dates()[K]
        assert abs(out["split_suspect_ratio"] - 0.5) < 0.02
        assert 1.6 < out["split_suspect_volume_ratio"] < 2.5

    @pytest.mark.parametrize("step,label", [(0.1, "1/10"), (1 / 3, "1/3"), (4.0, "4"), (5.0, "5")])
    def test_other_factors(self, step, label):
        closes, vols = _split(step, 1 / step, k=200)
        out = _check(closes, vols)
        assert out["split_suspect"] is True and out["split_suspect_factor"] == label

    def test_price_step_without_a_volume_step_is_not_flagged(self):
        closes, vols = _split(0.5, 1.0)
        out = _check(closes, vols)
        assert out["split_suspect"] is False
        assert out["split_candidates"] == 1 and out["split_uncorroborated"] == 1

    def test_missing_or_zero_volume_does_not_flag(self):
        closes, _ = _split(0.5, 2.0)
        assert _check(closes, None)["split_suspect"] is False
        zero = [0.0] * N
        assert _check(closes, zero)["split_suspect"] is False
        vols = [v if i >= K else 0 for i, v in enumerate(_vols())]
        assert _check(closes, vols)["split_suspect"] is False

    def test_step_too_recent_to_corroborate_does_not_flag(self):
        closes, vols = _split(0.5, 2.0, k=N - 4)     # under 5 bars after the skip
        assert _check(closes, vols)["split_suspect"] is False

    def test_scale_break_sndk_like(self):
        # the SNDK cache's step: x0.1277 (2.2% from 1/8), volume up about 6-8x
        closes, vols = _split(0.1277, 6.5)
        out = _check(closes, vols)
        assert out["split_suspect"] is True and out["split_suspect_rule"] == "scale_break"
        assert out["split_suspect_factor"] is None
        closes, vols = _split(0.1277, 2.0)            # volume step under 2.5x
        assert _check(closes, vols)["split_suspect"] is False

    def test_step_day_spike_is_not_a_split(self):
        # a clean 2:1 volume step, but day 0 trades 3x the new level
        closes, vols = _split(0.5, 2.0)
        vols[K] *= 3.0
        out = _check(closes, vols)
        assert out["split_suspect"] is False and out["split_uncorroborated_reasons"] == {"spike": 1}

    def test_spike_limit_is_inclusive_of_two(self):
        closes, vols = _split(0.5, 2.0)
        vols[K] *= 1.9
        assert _check(closes, vols)["split_suspect"] is True

    def test_missing_step_day_volume_is_not_corroborated(self):
        closes, vols = _split(0.5, 2.0)
        vols[K + 1] = 0
        out = _check(closes, vols)
        assert out["split_suspect"] is False and out["split_uncorroborated_reasons"] == {"no_volume": 1}

    def test_scale_break_dollar_volume_bound(self):
        # -85% with volume x3 from day 0, no spike: v x r = 0.45, outside a factor of 2
        closes, vols = _split(0.15, 3.0)
        out = _check(closes, vols)
        assert out["split_suspect"] is False and out["split_uncorroborated_reasons"] == {"dollar_volume": 1}

    def test_flagged_day_records_spike_ratio(self):
        closes, vols = _split(0.5, 2.0)
        out = _check(closes, vols)
        assert out["split_suspect_spike_ratio"] is not None and out["split_suspect_spike_ratio"] <= 2.0

    def test_ratio_outside_tolerance_is_not_a_split(self):
        closes, vols = _split(0.53, 1 / 0.53)          # -47%: 6% away from 1/2
        assert _check(closes, vols)["split_suspect"] is False

    def test_flat_window_with_volume_step(self):
        closes = [100.0] * 100 + [50.0] * 150
        vols = [1e6] * 100 + [2e6] * 150
        out = _check(closes, vols)
        assert out["split_suspect"] is True and out["split_suspect_move_vs_typical"] is None

    def test_step_into_the_window_is_left_out(self):
        closes, vols = _split(0.5, 2.0, k=50)          # step between bar 49 and 50 = index -200
        assert _check(closes, vols)["split_suspect"] is False
        closes, vols = _split(0.5, 2.0, k=51)
        assert _check(closes, vols)["split_suspect"] is True

    def test_short_and_bad_input(self):
        assert crew._split_discontinuity(_dates(1), [100.0], [1.0])["split_suspect"] is None
        out = crew._split_discontinuity(_dates(5), [100.0, 0.0, 101.0, -1.0, 102.0], [1.0] * 5)
        assert out["split_suspect"] is False
        out = crew._split_discontinuity(None, None)
        assert out["split_suspect"] is None and "split_error" in out


class TestRealMovesDoNotFlag:
    def test_crash_minus_50(self):
        # a halving with a 10x volume spike, then 1.5x share volume for good
        closes, vols = _crash(0.5, 10.0, 1.5)
        out = _check(closes, vols)
        assert out["split_candidates"] == 1 and out["split_suspect"] is False

    def test_crash_minus_70(self):
        # x0.30 is a scale-break candidate; volume settles at 1.8x, under 2.5x
        closes, vols = _crash(0.30, 12.0, 1.8)
        out = _check(closes, vols)
        assert out["split_candidates"] == 1 and out["split_suspect"] is False

    def test_pop_plus_100(self):
        # a doubling day with a volume spike, then 1.3x (a 1:2 reverse split would halve it)
        closes, vols = _crash(2.0, 8.0, 1.3)
        out = _check(closes, vols)
        assert out["split_candidates"] == 1 and out["split_suspect"] is False

    @pytest.mark.parametrize("r,hold", [(0.5, 2.0), (0.5, 2.5), (0.30, 3.0), (0.30, 6.0), (0.15, 3.0)])
    def test_crash_holding_elevated_volume(self, r, hold):
        # a real crash: days 0-2 spike to 3x the level volume then holds at
        # (`hold` x normal, at least 8x normal), then volume stays at `hold` x.
        # A crash whose first days sit within _SPLIT_SPIKE_MAX of its held
        # level, with v x r near 1, still reads as a split: the rule's limit.
        closes, vols = _crash(r, max(8.0, 3.0 * hold), hold)
        out = _check(closes, vols)
        assert out["split_candidates"] == 1 and out["split_suspect"] is False
        assert out["split_uncorroborated"] == 1

    def test_half_crash_spike_only(self):
        # -50% with doubled volume only on the first 3 days, then normal
        closes, vols = _crash(0.5, 2.0, 1.0)
        assert _check(closes, vols)["split_suspect"] is False


class TestTrendCheck:
    def test_records_split_and_still_computes_averages(self):
        closes, vols = _split(0.5, 2.0)
        with patch("db.get_price_history", return_value=_bars(closes, vols)):
            rec = crew._entry_trend_check("SNDK")
        assert rec["read"] == "ok" and rec["sma200"] is not None
        assert rec["split_suspect"] is True and rec["split_suspect_factor"] == "1/2"
        # the old scale (2x) inside the 200-bar window drags the close under the SMA200
        assert rec["below_sma200"] is True and rec["pct_vs_sma200"] < -20

    def test_crash_read_is_not_suspect(self):
        closes, vols = _crash(0.5, 10.0, 1.5)
        with patch("db.get_price_history", return_value=_bars(closes, vols)):
            rec = crew._entry_trend_check("XYZ")
        assert rec["split_suspect"] is False and rec["below_sma200"] is True

    def test_failed_read_has_the_keys(self):
        with patch("db.get_price_history", side_effect=RuntimeError("down")):
            rec = crew._entry_trend_check("NVDA")
        assert "split_suspect" in rec and rec["split_suspect"] is None
        assert rec["split_window_bars"] is None


def _result():
    return {"ticker": "SNDK", "action": "OPEN", "conviction": 8, "thesis_bucket": "Accelerator",
            "reasoning": "r", "pass_to_risk": True, "provenance": {"prompt_sha": "abc", "model": "m"}}


def _trend(suspect=True, below=True):
    return {"read": "ok", "bar_age_days": 1, "below_sma50": below, "below_sma200": below,
            "pct_vs_sma50": -30.0, "pct_vs_sma200": -40.0, "split_suspect": suspect,
            "split_suspect_date": "2025-11-20", "split_suspect_ratio": 0.5, "split_suspect_factor": "1/2"}


def _market(below=True, suspect=False):
    return {"read": "ok", "bar_age_days": 1, "below_sma200": below, "pct_vs_sma200": -6.0 if below else 4.0,
            "split_suspect": suspect, "split_suspect_date": "2026-06-01" if suspect else None,
            "split_suspect_ratio": 0.25 if suspect else None}


def _gate(stop_rec, trend):
    with patch("db.log_signal") as ls, patch("db.log_activity_queued") as la:
        out = crew._entry_gate("SNDK", _result(), stop_rec, trend)
    blocked = [c for c in ls.call_args_list if c[0][1] == "ENTRY_GATE_BLOCKED"]
    acts = [c for c in la.call_args_list if c[0][0] == "entry_gate_split_fail_open"]
    return out, blocked, acts


class TestGateStockOnly:
    def test_split_suspect_trend_fails_open_with_a_record(self):
        out, blocked, acts = _gate(None, _trend(True))
        assert out is False and not blocked and len(acts) == 1
        gate = acts[0][1]["extra"]["entry_gate"]["trend"]
        assert gate["split_fail_open"] is True and gate["blocked"] is False and gate["market"] is None
        assert gate["split_suspect_date"] == "2025-11-20" and acts[0][0][1] == "SNDK"

    def test_clean_trend_still_blocks(self):
        out, blocked, acts = _gate(None, _trend(False))
        assert out is True and len(blocked) == 1 and not acts
        assert blocked[0][1]["decision"]["entry_gate"]["trend"]["split_fail_open"] is False

    def test_suspect_but_not_below_writes_nothing(self):
        out, blocked, acts = _gate(None, _trend(True, below=False))
        assert out is False and not blocked and not acts

    def test_stopout_block_carries_the_split_record(self):
        out, blocked, acts = _gate({"read": "ok", "hours_since": 5.0, "exit_at": "x"}, _trend(True))
        assert out is True and len(blocked) == 1 and not acts
        tr = blocked[0][1]["decision"]["entry_gate"]["trend"]
        assert tr["split_fail_open"] is True and tr["blocked"] is False

    def test_record_failure_does_not_block(self):
        with patch("db.log_signal"), patch("db.log_activity_queued", side_effect=RuntimeError("x")):
            assert crew._entry_gate("SNDK", _result(), None, _trend(True)) is False


class TestGateProductionConfig:
    """The market condition as deployed: ENTRY_TREND_GATE_MARKET_TICKER = "SPY"."""

    @pytest.fixture(autouse=True)
    def _spy(self, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "SPY")

    def test_suspect_stock_spy_up_no_fail_open_row(self):
        with patch("agents.crew._entry_trend_check", return_value=_market(False)) as tc:
            out, blocked, acts = _gate(None, _trend(True))
        assert tc.call_args[0] == ("SPY",)
        assert out is False and not blocked and not acts      # SPY up: no block was withheld

    def test_suspect_stock_spy_down_writes_a_row(self):
        with patch("agents.crew._entry_trend_check", return_value=_market(True)) as tc:
            out, blocked, acts = _gate(None, _trend(True))
        assert tc.call_args[0] == ("SPY",)
        assert out is False and not blocked and len(acts) == 1
        tr = acts[0][1]["extra"]["entry_gate"]["trend"]
        assert tr["split_fail_open"] is True and tr["market"]["downtrend"] is True
        assert tr["market"]["split_fail_open"] is False

    def test_suspect_stock_market_read_raising(self):
        with patch("agents.crew._entry_trend_check", side_effect=RuntimeError("db")) as tc:
            out, blocked, acts = _gate(None, _trend(True))
        assert tc.call_args[0] == ("SPY",)
        assert out is False and not blocked and not acts
        with patch("agents.crew._entry_trend_check", side_effect=RuntimeError("db")):
            out, blocked, acts = _gate({"read": "ok", "hours_since": 5.0, "exit_at": "x"}, _trend(True))
        assert out is True and len(blocked) == 1             # the stop-out block stands
        m = blocked[0][1]["decision"]["entry_gate"]["trend"]["market"]
        assert m["read"] == "error: RuntimeError" and m["split_fail_open"] is False
        assert m["split_suspect"] is None and m["split_suspect_date"] is None
        assert m["split_suspect_ratio"] is None

    def test_clean_stock_suspect_spy_down_fails_open(self):
        with patch("agents.crew._entry_trend_check", return_value=_market(True, suspect=True)):
            out, blocked, acts = _gate(None, _trend(False))
        assert out is False and not blocked and len(acts) == 1
        m = acts[0][1]["extra"]["entry_gate"]["trend"]["market"]
        assert m["split_fail_open"] is True and m["downtrend"] is False

    def test_both_suspect_spy_down_leaves_a_record(self):
        with patch("agents.crew._entry_trend_check", return_value=_market(True, suspect=True)) as tc:
            out, blocked, acts = _gate(None, _trend(True))
        assert tc.call_args[0] == ("SPY",)
        assert out is False and not blocked and len(acts) == 1
        tr = acts[0][1]["extra"]["entry_gate"]["trend"]
        assert tr["split_fail_open"] is True and tr["market"]["split_fail_open"] is False

    def test_both_suspect_spy_up_writes_nothing(self):
        with patch("agents.crew._entry_trend_check", return_value=_market(False, suspect=True)):
            out, blocked, acts = _gate(None, _trend(True))
        assert out is False and not blocked and not acts

    def test_clean_stock_clean_spy_down_still_blocks(self):
        with patch("agents.crew._entry_trend_check", return_value=_market(True)) as tc:
            out, blocked, acts = _gate(None, _trend(False))
        assert tc.call_args[0] == ("SPY",)
        assert out is True and len(blocked) == 1 and not acts
