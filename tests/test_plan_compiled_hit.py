"""Record-only compiled plan level hits (agents.plans.compiled_level_state,
tasks._plan_compiled_hit_record): one plan_compiled_hit activity row the first
time a compiled level is crossed; PLAN_LEVEL_HIT unchanged, nothing queued."""
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
import db
import stop_lock
import tasks
from agents import plans

TODAY = date(2026, 11, 24)
NOW = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)


def _plan(**kw):
    p = {"id": 7, "ticker": "NVDA", "source": "verdict", "levels_source": "ladder",
         "structural_stop": 495.0, "t1": 575.0, "t2": 650.0, "add_trigger": 578.0,
         "invalidation_price": 480.0, "target_price": 650.0, "catalyst_date": None,
         "horizon_end": date(2026, 12, 10), "created_at": NOW - timedelta(days=3), "revision": 0}
    p.update(kw)
    return p


def test_state_per_level():
    s = plans.compiled_level_state(_plan(), 540.0, TODAY)
    assert set(s) == {"structural_stop", "t1", "t2", "add_trigger", "time_stop"}
    assert all(s[k]["crossed"] is False for k in s)
    assert s["structural_stop"]["direction"] == "down" and s["t1"]["direction"] == "up"
    assert s["t1"]["pct_vs_level"] == round((540 / 575 - 1) * 100, 3)
    s = plans.compiled_level_state(_plan(), 580.0, TODAY)
    assert s["t1"]["crossed"] and s["add_trigger"]["crossed"] and not s["t2"]["crossed"]
    s = plans.compiled_level_state(_plan(), 495.0, TODAY)              # at the level counts
    assert s["structural_stop"]["crossed"] is True
    s = plans.compiled_level_state(_plan(), None, date(2026, 12, 11))
    assert s["t1"] == {"crossed": None, "direction": "up", "pct_vs_level": None, "reason": "no_price"}
    assert s["time_stop"]["crossed"] is True and s["time_stop"]["date"] == "2026-12-10"
    s = plans.compiled_level_state(_plan(t1=None, horizon_end=None), 540.0, TODAY)
    assert s["t1"]["reason"] == "no_level" and s["time_stop"]["reason"] == "no_level"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(config, "PLAN_COMPILED_HIT_SHADOW_ENABLED", True)
    rows = []
    monkeypatch.setattr(db, "log_activity_direct",
                        lambda stage, ticker, msg, sent="neutral", extra=None: rows.append(
                            (stage, ticker, msg, extra)) or True)
    return rows


def test_first_crossing_writes_one_row_then_nothing(env):
    r = stop_lock._redis()
    out = tasks._plan_compiled_hit_record(_plan(), 580.0, TODAY, r, {"NVDA": 580.0}, now=NOW)
    assert out["newly_crossed"] == ["t1", "add_trigger"] and len(env) == 1
    stage, ticker, msg, extra = env[0]
    assert stage == "plan_compiled_hit" and ticker == "NVDA" and "record only" in msg
    assert extra["levels"]["t1"]["first_crossed_at"] == NOW.isoformat(timespec="seconds")
    assert extra["levels"]["t2"]["first_crossed_at"] is None and extra["held"] is True
    assert extra["price_source"] == "snapshot" and extra["plan_id"] == 7
    # next tick, still above: no new row
    assert tasks._plan_compiled_hit_record(_plan(), 585.0, TODAY, r, {"NVDA": 585.0},
                                           now=NOW + timedelta(minutes=15)) is None
    # later t2 crosses: one row, carrying t1's first time
    later = NOW + timedelta(days=2)
    out = tasks._plan_compiled_hit_record(_plan(), 655.0, TODAY, r, {"NVDA": 655.0}, now=later)
    assert out["newly_crossed"] == ["t2"]
    assert out["levels"]["t1"]["first_crossed_at"] == NOW.isoformat(timespec="seconds")


def test_row_carries_no_price(env):
    tasks._plan_compiled_hit_record(_plan(), 580.0, TODAY, stop_lock._redis(), {"NVDA": 580.0}, now=NOW)
    _, _, msg, extra = env[0]
    text = msg + json.dumps(extra)
    for bad in ("580", "575.0", "\"price\":"):
        assert bad not in text


def test_failed_write_keeps_the_crossing_for_next_tick(env, monkeypatch):
    monkeypatch.setattr(db, "log_activity_direct", lambda *a, **k: False)
    r = stop_lock._redis()
    assert tasks._plan_compiled_hit_record(_plan(), 580.0, TODAY, r, {}, now=NOW) is None
    monkeypatch.setattr(db, "log_activity_direct", lambda *a, **k: True)
    assert tasks._plan_compiled_hit_record(_plan(), 580.0, TODAY, r, {}, now=NOW)["newly_crossed"]


def test_unheld_plan_gets_the_time_stop_only(env):
    out = tasks._plan_compiled_hit_record(_plan(), None, date(2026, 12, 11), stop_lock._redis(), {},
                                          now=NOW)
    assert out["newly_crossed"] == ["time_stop"] and out["held"] is False
    assert out["price_source"] is None


def test_plans_without_compiled_levels_and_switch_off_write_nothing(env, monkeypatch):
    r = stop_lock._redis()
    assert tasks._plan_compiled_hit_record(_plan(levels_source=None), 580.0, TODAY, r, {}) is None
    monkeypatch.setattr(config, "PLAN_COMPILED_HIT_SHADOW_ENABLED", False)
    assert tasks._plan_compiled_hit_record(_plan(), 580.0, TODAY, r, {}) is None
    assert env == []


def test_never_raises(env, monkeypatch):
    class Broken:
        def get(self, k):
            raise ConnectionError()
    assert tasks._plan_compiled_hit_record(_plan(), 580.0, TODAY, Broken(), {}) is None
    assert tasks._plan_compiled_hit_record({"junk": 1}, 580.0, TODAY, stop_lock._redis(), {}) is None
    assert env == []


def _watch(plan, held_prices, enabled):
    with patch("config.PLAN_COMPILED_HIT_SHADOW_ENABLED", enabled), \
         patch("config.PLAN_LEVEL_REVIEW_ENABLED", False), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value=held_prices), \
         patch("db.outcome_closed_since", return_value=None), \
         patch("tasks.review_plan_level_hit.apply_async") as queued, \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("db.log_signal") as log_signal:
        n = tasks.watch_plan_levels()
    return n, log_signal.call_args_list, act.call_args_list, queued


def test_watcher_plan_level_hit_rows_identical_with_the_switch_on_or_off():
    """A price under both the model's invalidation and the structural stop."""
    off = _watch(_plan(), {"NVDA": 470.0}, False)
    stop_lock._redis().store.clear()
    on = _watch(_plan(), {"NVDA": 470.0}, True)
    assert off[0] == on[0] == 1
    assert [c.args for c in off[1]] == [c.args for c in on[1]]
    strip = lambda calls: [{k: v for k, v in c.kwargs["provenance"].items()} for c in calls]
    assert strip(off[1]) == strip(on[1])
    assert off[2] == [] and len(on[2]) == 1 and on[2][0].args[0] == "plan_compiled_hit"
    assert on[2][0].args[4]["newly_crossed"] == ["structural_stop"]
    assert not off[3].called and not on[3].called


def test_a_compiled_crossing_alone_writes_no_plan_level_hit():
    n, signals, acts, queued = _watch(_plan(), {"NVDA": 580.0}, True)     # t1 crossed, no model level
    assert n == 0 and signals == [] and len(acts) == 1 and not queued.called


def test_record_queues_and_trades_nothing():
    src = inspect.getsource(tasks._plan_compiled_hit_record) + inspect.getsource(plans.compiled_level_state)
    for call in ("apply_async(", ".delay(", "send_task(", "place_", "cancel", "log_signal(",
                 "touch_plan(", "insert_plan(", "set_state("):
        assert call not in src, call
