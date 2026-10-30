"""Shadow trade plans (roadmap 1): built from the analyst's fields, later
verdicts classified against them, nothing fed back into trading."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

from agents import plans

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
         patch("tasks._get_client") as client, \
         patch("db.log_signal") as log_signal:
        from types import SimpleNamespace
        client.return_value.get_all_positions.return_value = [SimpleNamespace(symbol="NVDA", current_price=470.0)]
        assert tasks.watch_plan_levels() == 1
        assert tasks.watch_plan_levels() == 0          # same plan and kind today: not again
    assert log_signal.call_args.args[1] == "PLAN_LEVEL_HIT"
