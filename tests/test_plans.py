"""Shadow trade plans (roadmap 1): built from the analyst's fields, later
verdicts classified against them, nothing fed back into trading."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agents import plans


@pytest.fixture(autouse=True)
def _broker_unknown():
    """No fresh snapshot unless a test says otherwise: held falls back to
    the stop row, and no test reads a real portfolio_snapshots row."""
    with patch("tasks._broker_held", return_value=(None, "no_snapshot", {})):
        yield

TODAY = date(2026, 10, 29)
NOW = datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc)

PARSED = {
    "valuation": {"bull_case_target": "$650 (12 months)", "downside_scenario": "$480"},
    "catalyst": {"event": "Q1 earnings", "date": "2026-11-20"},
    "expectation_gap": {"primary_risk": "capex slowdown"},
    "thesis_bucket": "Accelerator", "thesis_summary": "AI capex beneficiary",
}


def _plan(**kw):
    p = plans.plan_from_verdict("NVDA", PARSED, 7, 540.0, TODAY)
    p.update(id=1, created_at=NOW - timedelta(days=2), last_seen_at=None, revision=0)
    p.update(kw)
    return p


def test_plan_from_verdict_reads_analyst_fields():
    p = plans.plan_from_verdict("NVDA", PARSED, 7, 540.0, TODAY)
    assert p["target_price"] == 650.0 and p["invalidation_price"] == 480.0
    assert p["catalyst_date"] == date(2026, 11, 20)
    assert p["horizon_end"] == date(2026, 11, 25)       # catalyst + 5-day grace
    assert p["thesis"] == "AI capex beneficiary" and p["primary_risk"] == "capex slowdown"


def test_downside_pct_fallback_and_nonsense_levels_dropped():
    parsed = {**PARSED, "valuation": {"bull_case_target": "$500", "downside_scenario": "n/a"},
              "expected_downside_pct": -15.0, "catalyst": {"date": "2026-12"}}
    p = plans.plan_from_verdict("NVDA", parsed, 7, 540.0, TODAY)
    assert p["invalidation_price"] == 459.0
    assert p["target_price"] is None                     # a "target" below price isn't one
    assert p["catalyst_date"] == date(2026, 12, 31)      # month reads as month end


def test_no_dated_catalyst_uses_bucket_horizon():
    parsed = {**PARSED, "catalyst": {"date": "ongoing"}}
    p = plans.plan_from_verdict("NVDA", parsed, 7, 540.0, TODAY)
    assert p["horizon_end"] == TODAY + timedelta(days=90)


def test_classify_follow_diverge_invalidate():
    plan = _plan()
    assert plans.classify(plan, "HOLD", None, 560.0, TODAY, held=True)["relation"] == "FOLLOW"
    assert plans.classify(plan, "TRIM", None, 560.0, TODAY, held=True)["relation"] == "DIVERGE"
    assert plans.classify(plan, "HOLD", None, 470.0, TODAY, held=True)["relation"] == "INVALIDATE"
    # target reached: the plan says sell, a HOLD diverges
    r = plans.classify(plan, "HOLD", None, 660.0, TODAY, held=True)
    assert r["plan_says"] == "sell" and r["relation"] == "DIVERGE"
    # not held, near entry: waiting (WATCH) still follows a buy plan
    assert plans.classify(plan, "WATCH", None, 545.0, TODAY, held=False)["relation"] == "FOLLOW"


def test_open_with_moved_target_is_a_revision():
    plan = _plan()
    new = plans.plan_from_verdict("NVDA", {**PARSED, "valuation": {
        "bull_case_target": "$800", "downside_scenario": "$480"}}, 8, 545.0, TODAY)
    assert plans.classify(plan, "ADD", new, 545.0, TODAY, held=True)["relation"] == "REVISE"


def test_observe_creates_then_follows_and_coalesces():
    brief = "...\nCurrent Price: $540.00\n..."
    store = {}

    def insert(p):
        store["plan"] = {**p, "id": 7, "created_at": NOW, "last_seen_at": None}
        return 7

    def touch(pid, rel, close_reason=None):
        store["plan"]["last_seen_at"] = NOW
        store["touched"] = (pid, rel, close_reason)

    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", side_effect=lambda t: store.get("plan")), \
         patch("db.insert_plan", side_effect=insert), patch("db.touch_plan", side_effect=touch):
        r1 = plans.observe("NVDA", PARSED, "OPEN", 7, brief, now=NOW)
        r2 = plans.observe("NVDA", PARSED, "WATCH", 6, brief, now=NOW + timedelta(hours=1))
        r3 = plans.observe("NVDA", PARSED, "WATCH", 6, brief, now=NOW + timedelta(hours=6))
    assert r1["relation"] == "NEW_PLAN" and r1["plan_id"] == 7
    assert r2["relation"] == "FOLLOW" and r2["would_coalesce"] is True
    assert r2["plan_live"] is False and r2["held"] is False      # an unfilled OPEN's plan (A0100-4)
    assert r3["would_coalesce"] is False
    assert store["touched"] == (7, "FOLLOW", None)


def test_observe_seeds_held_position_and_never_raises():
    stop = {"entry_price": 500.0, "stop_price": 450.0, "thesis_bucket": "Compounder"}
    with patch("db.get_position_stop", return_value=stop), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=3) as ins:
        r = plans.observe("MSFT", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "SEEDED"
    seed = ins.call_args.args[0]
    assert seed["source"] == "held_seed" and seed["invalidation_price"] == 450.0
    with patch("db.get_position_stop", side_effect=RuntimeError("db down")):
        assert plans.observe("MSFT", PARSED, "HOLD", 6, "", now=NOW)["relation"] == "ERROR"


def test_no_price_no_plan():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None):
        assert plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: UNKNOWN", now=NOW)["relation"] == "NO_PLAN"


def test_dollar_parse_skips_magnitudes():
    """Professor M94: '$2B' is not a $2 price target."""
    assert plans._dollars("$2B revenue run-rate, target $650") == 650.0
    assert plans._dollars("$40 million contract") is None
    assert plans._dollars("$1,250.50 (12 months)") == 1250.5
    assert plans._dollars("a target of $650.") == 650.0


def test_expired_or_closed_position_plan_is_closed_first():
    old = _plan(horizon_end=TODAY - timedelta(days=1))
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=old), \
         patch("db.touch_plan") as touch, patch("db.insert_plan", return_value=9):
        r = plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    touch.assert_called_once_with(1, "CLOSED", close_reason="expired")
    assert r["relation"] == "NEW_PLAN"
    seeded = _plan(source="held_seed")
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=seeded), patch("db.touch_plan") as touch:
        r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $540.00", now=NOW)
    touch.assert_called_once_with(1, "CLOSED", close_reason="position closed")
    assert r["relation"] == "NO_PLAN"


def test_compile_plan_uses_ladder_levels():
    """Audit A2114-1: shadow plan from analyst view + price ladder."""
    ladder = {"atr": 10.0, "fair_value_blended": 640.0, "rungs": [
        {"price": 500.0, "band_high": 503.0, "types": ["ma50"]},
        {"price": 535.0, "band_high": 538.0, "types": ["ma10"]},      # within 1 ATR: ignored
        {"price": 575.0, "band_high": 578.0, "types": ["swing_high"]},
        {"price": 610.0, "band_high": 613.0, "types": ["52w_high"]}]}
    p = plans.compile_plan(PARSED, 540.0, ladder, 500.0, TODAY)
    assert p["structural_stop"] == 495.0 and p["support_types"] == ["ma50"]
    assert p["t1"] == 575.0 and p["add_trigger"] == 578.0 and p["t2"] == 650.0
    assert p["r_to_t1"] == round(35 / 45, 2) and p["stop_gap_pct"] == round(5 / 540 * 100, 2)
    bare = plans.compile_plan(PARSED, 540.0, None, None, TODAY)
    assert bare["source"] == "analyst_only" and bare["structural_stop"] == 480.0 and bare["t1"] is None


def test_level_hits_for_the_watcher():
    """Audit A2114-4/-7: levels and dates a level-triggered review would use."""
    plan = _plan(catalyst_date=date(2026, 11, 20), horizon_end=date(2026, 11, 25))
    assert plans.level_hits(plan, 540.0, TODAY) == []
    assert plans.level_hits(plan, 470.0, TODAY) == [("invalidation", 480.0)]
    assert plans.level_hits(plan, 660.0, TODAY) == [("target", 650.0)]
    assert plans.level_hits(plan, None, date(2026, 11, 21)) == [("catalyst_passed", "2026-11-20")]
    assert plans.level_hits(plan, None, date(2026, 11, 26)) == [("horizon_end", "2026-11-25")]


def test_watcher_logs_once_per_plan_and_kind():
    import tasks
    plan = _plan(catalyst_date=None)
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value={"NVDA": 470.0}), \
         patch("tasks._get_client") as client, \
         patch("db.log_signal") as log_signal:
        assert tasks.watch_plan_levels() == 1
        assert tasks.watch_plan_levels() == 0          # same plan and kind today: not again
    assert log_signal.call_args.args[1] == "PLAN_LEVEL_HIT"
    assert not client.called          # no IBKR connection from the ingestion worker


def test_watcher_says_when_price_checks_were_skipped(caplog):
    import logging
    import tasks
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[_plan(catalyst_date=None)]), \
         patch("tasks._held_prices", return_value={}), patch("db.log_signal"), \
         caplog.at_level(logging.WARNING, logger="tasks"):
        tasks.watch_plan_levels()
    assert "price checks skipped for 1 plan" in caplog.text and "no fresh held prices" in caplog.text


def _watch(enabled):
    import tasks
    plan = _plan(catalyst_date=None)
    redis = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    redis.set.return_value = True
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value={"NVDA": 470.0}), \
         patch("stop_lock._redis", return_value=redis), \
         patch("config.PLAN_LEVEL_REVIEW_ENABLED", enabled), \
         patch("tasks.review_plan_level_hit.apply_async") as queued, \
         patch("db.log_signal") as log_signal:
        tasks.watch_plan_levels()
    return queued, log_signal


def test_level_hit_only_logs_while_the_review_switch_is_off():
    queued, log_signal = _watch(False)
    assert not queued.called
    assert log_signal.call_args.kwargs["provenance"]["review_queued"] is False


def test_level_hit_on_a_held_name_queues_a_review_when_switched_on():
    queued, log_signal = _watch(True)
    assert queued.call_args.kwargs["args"][:2] == ["NVDA", "invalidation"]
    assert log_signal.call_args.kwargs["provenance"]["review_queued"] is True


def test_plan_level_review_skips_a_name_no_longer_held():
    from agents import crew
    with patch("ibkr.IBKRClient"), patch("agents.crew._get_positions", return_value={"MSFT": {}}), \
         patch("agents.crew._analyze_held_position") as analyze:
        assert crew.review_plan_level("NVDA", "target", "650", 660.0) is None
    assert not analyze.called
    with patch("ibkr.IBKRClient"), patch("agents.crew._get_positions", return_value={"NVDA": {}}), \
         patch("agents.market_regime.get_cached_regime", return_value={}), \
         patch("agents.crew._analyze_held_position", return_value={"action": "HOLD"}) as analyze, \
         patch("agents.crew._capture_latest_thesis"), patch("agents.crew._route_to_approval") as route, \
         patch("db.get_state", return_value='{"PLTR": "x"}'), patch("db.set_state") as set_state:
        assert crew.review_plan_level("NVDA", "target", "650", 660.0) == {"action": "HOLD"}
    import json
    saved = json.loads(set_state.call_args.args[1])
    assert set(saved) == {"PLTR", "NVDA"}
    assert analyze.call_args.kwargs["path"] == "plan_level_review"
    assert "target" in analyze.call_args.kwargs["event"]["title"] and not route.called


def test_two_level_kinds_on_one_ticker_queue_one_review_per_day():
    import tasks
    from unittest.mock import MagicMock
    plan = _plan(catalyst_date=date(2026, 10, 1))      # invalidation AND catalyst_passed
    seen = set()
    redis = MagicMock()
    redis.set.side_effect = lambda k, v, nx=False, ex=None: (k not in seen) and not seen.add(k)
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value={"NVDA": 470.0}), \
         patch("stop_lock._redis", return_value=redis), \
         patch("config.PLAN_LEVEL_REVIEW_ENABLED", True), \
         patch("tasks.review_plan_level_hit.apply_async") as queued, patch("db.log_signal"):
        assert tasks.watch_plan_levels() == 2
    assert queued.call_count == 1


def test_broker_position_without_a_stop_row_is_recorded_not_seeded():
    """Audit A1031-0453-2: logged only; plan behavior still keys on the stop row."""
    with patch("tasks._broker_held", return_value=(True, "snapshot", {"qty": 3})), \
         patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), patch("db.insert_plan") as ins:
        r = plans.observe("MSFT", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    ins.assert_not_called()
    assert r["relation"] == "NO_PLAN" and r["held"] is False
    assert r["held_broker"] is True and r["held_mismatch"] is True


def test_broker_flat_with_a_stop_row_is_recorded_and_the_plan_kept():
    seeded = _plan(source="held_seed")
    with patch("tasks._broker_held", return_value=(False, "snapshot", {})), \
         patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 450.0}), \
         patch("db.get_active_plan", return_value=seeded), patch("db.touch_plan") as touch:
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert all(c.args[1] != "CLOSED" for c in touch.call_args_list)
    assert r["held"] is True and r["held_broker"] is False and r["held_mismatch"] is True
    assert r["held_broker_reason"] == "snapshot"


def test_held_hold_records_would_revise_when_its_levels_move():
    """Audit A1031-0053-1a: a held name's HOLD can't REVISE; record when it would."""
    plan = _plan(target_price=600.0)                     # verdict's bull target is $650: +8%
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 450.0}), \
         patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "FOLLOW" and r["would_revise"] is False
    plan = _plan(target_price=560.0, catalyst_date=date(2026, 12, 15))
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 450.0}), \
         patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "FOLLOW"
    assert r["would_revise"] is True and r["would_revise_why"] == ["target", "catalyst_date"]


def test_held_seed_invalidation_is_the_stop_and_is_not_compared():
    plan = _plan(source="held_seed", invalidation_price=300.0)
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 300.0}), \
         patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert "invalidation" not in r["would_revise_why"]


def test_expiry_marks_the_decision_at_horizon():
    """Audit A1031-0053-1b: expiry used to re-seed silently."""
    old = _plan(source="held_seed", horizon_end=TODAY - timedelta(days=1))
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 450.0}), \
         patch("db.get_active_plan", return_value=old), patch("db.touch_plan") as touch, \
         patch("db.insert_plan", return_value=12):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    touch.assert_called_once_with(1, "CLOSED", close_reason="expired")
    assert r["relation"] == "SEEDED" and r["decision_at_horizon"] is True and r["expired_plan_id"] == 1
