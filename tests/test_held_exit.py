"""Held-exit candidates (agents/held_exit.py): pure, record only."""
from datetime import date, timedelta

import pytest

from agents import held_exit as he

LAST = date(2026, 11, 13)


def _bars(closes, last=LAST, spread=1.0):
    n = len(closes)
    return [{"date": (last - timedelta(days=n - 1 - i)).isoformat(), "open": float(c),
             "high": float(c) + spread, "low": float(c) - spread, "close": float(c)}
            for i, c in enumerate(closes)]


def _run(closes, entry_price=100.0, entry_days_ago=30, plan=None, **kw):
    bars = _bars(closes)
    edate = LAST - timedelta(days=entry_days_ago - 1) if entry_days_ago is not None else None
    return he.candidates(bars, entry_price=entry_price, entry_date=edate, plan=plan, **kw)


def test_no_bars_all_unknown_nothing_fires():
    rec = he.candidates([], entry_price=100, entry_date=LAST, plan=None)
    assert all(c == {"would_exit": None, "reason": "no_cached_bars"} for c in rec["candidates"].values())
    assert rec["firing"] == [] and rec["any_would_exit"] is False
    assert set(rec["candidates"]) == set(he.CANDIDATES)


def test_downtrend_fires_sma_time_and_drawdown():
    closes = [300 - i for i in range(250)]          # 300 .. 51, entry 30 bars ago at 80
    rec = _run(closes, entry_price=80.0, entry_days_ago=30)
    c = rec["candidates"]
    assert c["below_both_sma"]["would_exit"] is True
    assert rec["sessions_held"] == 30
    assert c["time_stop"]["would_exit"] is True             # 30 >= 20 and 51 <= 80
    # highest close since entry is the entry day's close (80); ATR14 is 2, so 14.5 ATR
    assert rec["highest_close_since_entry"] == 80.0 and rec["drawdown_from_high_atr"] >= 3
    assert c["atr_drawdown"]["would_exit"] is True
    assert rec["firing"] == ["below_both_sma", "time_stop", "atr_drawdown"]
    assert rec["close_vs_entry_pct"] == pytest.approx((51 / 80 - 1) * 100, abs=1e-3)


def test_uptrend_fires_nothing():
    closes = [50 + i for i in range(250)]
    rec = _run(closes, entry_price=200.0, entry_days_ago=40)
    c = rec["candidates"]
    assert c["below_both_sma"]["would_exit"] is False
    assert c["time_stop"]["would_exit"] is False            # close above entry
    assert c["atr_drawdown"]["would_exit"] is False and rec["drawdown_from_high_atr"] == 0
    assert rec["any_would_exit"] is False


def test_time_stop_needs_n_sessions():
    closes = [100.0] * 250
    assert _run(closes, entry_days_ago=19)["candidates"]["time_stop"]["would_exit"] is False
    assert _run(closes, entry_days_ago=20)["candidates"]["time_stop"]["would_exit"] is True  # flat = <= 0
    assert _run(closes, entry_days_ago=10, time_stop_sessions=10)["candidates"]["time_stop"]["would_exit"] is True
    assert _run(closes, entry_price=99.0, entry_days_ago=40)["candidates"]["time_stop"]["would_exit"] is False


def test_drawdown_threshold_and_raw_numbers():
    closes = [100.0] * 240 + [110.0] + [110.0 - 0.5 * i for i in range(1, 10)]   # last 105.5
    rec = _run(closes, entry_price=100.0, entry_days_ago=20, drawdown_atr=3.0)
    assert rec["highest_close_since_entry"] == 110.0
    dd = rec["drawdown_from_high_atr"]
    assert dd == pytest.approx((110 - 105.5) / rec["atr14"], abs=1e-3)
    assert rec["candidates"]["atr_drawdown"]["would_exit"] is (dd >= 3)
    loose = _run(closes, entry_price=100.0, entry_days_ago=20, drawdown_atr=100.0)
    assert loose["candidates"]["atr_drawdown"]["would_exit"] is False


def test_short_history_sma_unknown_but_others_work():
    rec = _run([100.0] * 60, entry_days_ago=30)
    assert rec["candidates"]["below_both_sma"] == {"would_exit": None, "reason": "insufficient_bars"}
    assert rec["sma50"] == 100.0 and rec["sma200"] is None
    assert rec["candidates"]["time_stop"]["would_exit"] is True


def test_plan_levels():
    closes = [100.0] * 250
    plan = {"id": 7, "source": "verdict", "revision": 2, "structural_stop": 101.0,
            "invalidation_price": 90.0, "levels_source": "ladder"}
    rec = _run(closes, plan=plan)
    assert rec["candidates"]["below_plan_structural"]["would_exit"] is True
    assert rec["candidates"]["below_plan_invalidation"]["would_exit"] is False
    assert rec["plan"]["plan_id"] == 7 and rec["plan"]["structural_stop"] == 101.0
    no_lvl = _run(closes, plan={"id": 8, "structural_stop": None, "invalidation_price": None})
    assert no_lvl["candidates"]["below_plan_structural"]["reason"] == "no_structural_stop"
    assert no_lvl["candidates"]["below_plan_invalidation"]["reason"] == "no_invalidation"
    assert _run(closes)["candidates"]["below_plan_structural"]["reason"] == "no_plan"
    err = _run(closes, plan_read="error:RuntimeError")
    assert err["candidates"]["below_plan_structural"]["reason"] == "plan_read_error"


def test_no_entry_date_or_price():
    closes = [100.0] * 250
    rec = _run(closes, entry_days_ago=None)
    assert rec["candidates"]["time_stop"]["reason"] == "no_entry_date"
    assert rec["candidates"]["atr_drawdown"]["reason"] == "no_entry_date"
    rec = _run(closes, entry_price=None)
    assert rec["candidates"]["time_stop"]["reason"] == "no_entry_price"
    assert rec["candidates"]["atr_drawdown"]["would_exit"] is False
    rec = he.candidates(_bars(closes), entry_price=100, entry_date=LAST + timedelta(days=3), plan=None)
    assert rec["candidates"]["time_stop"]["reason"] == "no_bars_since_entry"


def test_truncation_flag():
    closes = [100.0] * 30
    rec = he.candidates(_bars(closes), entry_price=100, entry_date=LAST - timedelta(days=100),
                        plan=None)
    assert rec["since_entry_truncated"] is True and rec["sessions_held"] == 30
    # One row short of the read limit (today's partial bar dropped) still flags.
    rec = he.candidates(_bars(closes[:29]), entry_price=100, entry_date=LAST - timedelta(days=100),
                        plan=None)
    assert rec["since_entry_truncated"] is True
    rec = he.candidates(_bars(closes), entry_price=100, entry_date=LAST - timedelta(days=10),
                        plan=None)
    assert rec["since_entry_truncated"] is False


def test_bad_values_never_raise():
    bars = _bars([100.0] * 30) + [{"date": "2026-11-14", "close": "x", "high": None, "low": 1}]
    rec = he.candidates(bars, entry_price="bad", entry_date="not a date", plan="junk")
    assert rec["bars"] == 30 and rec["entry_price"] is None and rec["entry_date"] is None
    assert rec["candidates"]["below_plan_structural"]["reason"] == "no_plan"
    assert he.candidates(bars, entry_price=-5, entry_date=None, plan=None)["entry_price"] is None
