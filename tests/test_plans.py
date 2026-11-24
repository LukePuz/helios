"""Shadow trade plans (roadmap 1): built from the analyst's fields, later
verdicts classified against them, nothing fed back into trading."""
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import psycopg2.errors
import psycopg2.extras

import pytest

from agents import plans
import db

# The real lookup, kept before the autouse fixture below patches it.
_real_outcome_closed_since = db.outcome_closed_since


@pytest.fixture(autouse=True)
def _broker_unknown():
    """No fresh snapshot unless a test says otherwise: held falls back to
    the stop row, and no test reads a real portfolio_snapshots row."""
    with patch("tasks._broker_held", return_value=(None, "no_snapshot", {})):
        yield


@pytest.fixture(autouse=True)
def _no_closed_outcome():
    """No closed trade after any plan unless a test says otherwise, and no
    test reads a real trade_outcomes row."""
    with patch("db.outcome_closed_since", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _no_ladder():
    """No trade_ladder row unless a test says otherwise: a new plan's levels
    come from the analyst only, and no test reads a real trade_ladder row."""
    with patch("db.get_trade_ladder", return_value=None):
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


def test_add_under_a_hold_plan_stays_follow_and_is_recorded():
    """A held ADD while the plan says hold is FOLLOW ("adding within plan");
    the record says so, with the plan's add_trigger and the price against it."""
    plan = _plan(add_trigger=560.0)
    r = plans.classify(plan, "ADD", dict(plan), 545.0, TODAY, held=True)
    assert r["relation"] == "FOLLOW" and r["plan_says"] == "hold"
    assert r["add_under_hold_plan"] is True and r["add_trigger"] == 560.0
    assert r["price_vs_add_trigger_pct"] == -2.68
    # a held_seed plan has no add_trigger
    r = plans.classify(_plan(source="held_seed"), "ADD", None, 545.0, TODAY, held=True)
    assert r["relation"] == "FOLLOW" and r["add_under_hold_plan"] is True
    assert r["add_trigger"] is None and r["price_vs_add_trigger_pct"] is None
    # not held, near the plan entry: the plan says buy, so not under a hold plan
    r = plans.classify(plan, "OPEN", dict(plan), 545.0, TODAY, held=False)
    assert r["relation"] == "FOLLOW" and r["add_under_hold_plan"] is False
    # a HOLD verdict carries none of these keys
    r = plans.classify(plan, "HOLD", None, 545.0, TODAY, held=True)
    assert not any(k in r for k in ("add_under_hold_plan", "add_trigger", "price_vs_add_trigger_pct"))


def test_add_plan_record_failure_keeps_the_relation(caplog):
    plan = _plan(add_trigger="not a number")
    with caplog.at_level("WARNING"):
        r = plans.classify(plan, "ADD", dict(plan), 545.0, TODAY, held=True)
    assert r["relation"] == "FOLLOW" and "add_under_hold_plan" not in r
    assert "add record failed" in caplog.text


def test_add_under_hold_plan_reaches_the_verdict_record():
    plan = _plan(source="analyst", add_trigger=530.0)
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 470.0}), \
         patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"):
        out = plans.observe("NVDA", PARSED, "ADD", 7, "Current Price: $545.00", now=NOW)
    assert out["relation"] == "FOLLOW" and out["add_under_hold_plan"] is True
    assert out["add_trigger"] == 530.0 and out["price_vs_add_trigger_pct"] == 2.83


def test_open_with_moved_target_is_a_revision():
    plan = _plan()
    new = plans.plan_from_verdict("NVDA", {**PARSED, "valuation": {
        "bull_case_target": "$800", "downside_scenario": "$480"}}, 8, 545.0, TODAY)
    assert plans.classify(plan, "ADD", new, 545.0, TODAY, held=True)["relation"] == "REVISE"


def _classify_before(plan, action, new, price, today, held):
    """classify as it was before revise_triggers was added, for the
    unchanged-relation check."""
    says, why = plans.plan_says(plan, price, today, held)
    out = {"plan_id": plan["id"], "plan_says": says, "plan_why": why,
           "verdict_dir": plans._DIRECTION.get(action, plans._HOLD)}
    if why == "price at or below invalidation":
        out["relation"] = "INVALIDATE"
        return out
    if new and action in ("OPEN", "ADD") and (
            plans._moved(plan.get("target_price"), new.get("target_price"))
            or plans._moved(plan.get("invalidation_price"), new.get("invalidation_price"))
            or (plan.get("catalyst_date") and new.get("catalyst_date")
                and plan["catalyst_date"] != new["catalyst_date"])):
        out["relation"] = "REVISE"
        return out
    agree = (out["verdict_dir"] == says
             or (says == plans._BUY and out["verdict_dir"] == plans._HOLD)
             or (says == plans._HOLD and out["verdict_dir"] == plans._BUY and held))
    out["relation"] = "FOLLOW" if agree else "DIVERGE"
    return out


def _rev(**new_kw):
    plan = _plan()       # target 650, invalidation 480, catalyst 2026-11-20
    new = {**plan, **new_kw}
    return plans.classify(plan, "ADD", new, 545.0, TODAY, held=True)


@pytest.mark.parametrize("kw, fired", [
    ({"target_price": 800.0}, ["target"]),
    ({"invalidation_price": 400.0}, ["invalidation"]),
    ({"catalyst_date": date(2026, 12, 5)}, ["catalyst_date"]),
    ({"target_price": 800.0, "invalidation_price": 400.0}, ["target", "invalidation"]),
    ({"target_price": 800.0, "catalyst_date": date(2026, 12, 5)}, ["target", "catalyst_date"]),
    ({"target_price": 800.0, "invalidation_price": 400.0, "catalyst_date": date(2026, 12, 5)},
     ["target", "invalidation", "catalyst_date"]),
])
def test_revise_triggers_name_each_condition_that_fired(kw, fired):
    r = _rev(**kw)
    assert r["relation"] == "REVISE" and r["revise_triggers"] == fired
    assert r["catalyst_month_only_pair"] is False


def test_revise_triggers_ignore_a_move_under_the_threshold():
    r = _rev(target_price=660.0, catalyst_date=date(2026, 12, 5))   # +1.5%: not a trigger
    assert r["revise_triggers"] == ["catalyst_date"]


@pytest.mark.parametrize("old, new, flag", [
    (date(2026, 11, 30), date(2026, 11, 4), True),     # "2026-11" vs "2026-11-04"
    (date(2026, 11, 4), date(2026, 11, 30), True),     # either order
    (date(2026, 12, 31), date(2026, 12, 15), True),
    (date(2026, 11, 20), date(2026, 11, 4), False),    # same month, neither a month end
    (date(2026, 11, 30), date(2026, 12, 4), False),    # month end, different month
    (date(2026, 10, 31), date(2026, 11, 30), False),   # both month ends, different months
])
def test_catalyst_month_only_pair(old, new, flag):
    plan = _plan(catalyst_date=old)
    r = plans.classify(plan, "OPEN", {**plan, "catalyst_date": new}, 545.0, TODAY, held=False)
    assert r["relation"] == "REVISE" and r["revise_triggers"] == ["catalyst_date"]
    assert r["catalyst_month_only_pair"] is flag


def test_catalyst_month_only_pair_false_when_a_level_also_moved():
    plan = _plan(catalyst_date=date(2026, 11, 30))
    new = {**plan, "catalyst_date": date(2026, 11, 4), "target_price": 800.0}
    r = plans.classify(plan, "ADD", new, 545.0, TODAY, held=True)
    assert r["revise_triggers"] == ["target", "catalyst_date"]
    assert r["catalyst_month_only_pair"] is False


def test_month_only_catalyst_string_against_exact_date_is_flagged():
    plan = _plan(catalyst_date=plans._catalyst_date("2026-11"))
    new = {**plan, "catalyst_date": plans._catalyst_date("2026-11-04")}
    r = plans.classify(plan, "ADD", new, 545.0, TODAY, held=True)
    assert r["relation"] == "REVISE" and r["catalyst_month_only_pair"] is True


def test_non_revise_relations_carry_no_trigger_keys():
    plan = _plan()
    for action, new, price in [("HOLD", None, 560.0), ("TRIM", None, 560.0),
                               ("HOLD", None, 470.0), ("ADD", dict(plan), 545.0)]:
        r = plans.classify(plan, action, new, price, TODAY, held=True)
        assert r["relation"] != "REVISE"
        assert "revise_triggers" not in r and "catalyst_month_only_pair" not in r


def test_classify_unchanged_apart_from_the_new_keys():
    import itertools
    base = _plan()
    levels = [None, 650.0, 660.0, 800.0]
    stops = [None, 480.0, 400.0]
    cats = [None, date(2026, 11, 20), date(2026, 11, 30), date(2026, 12, 5)]
    n = 0
    for plan_cat in (None, date(2026, 11, 20)):
        plan = {**base, "catalyst_date": plan_cat}
        for t, i, c in itertools.product(levels, stops, cats):
            new = {**plan, "target_price": t, "invalidation_price": i, "catalyst_date": c}
            for action in ("OPEN", "ADD", "HOLD", "TRIM", "WATCH"):
                for price in (470.0, 545.0, 600.0, 660.0, None):
                    for held in (True, False):
                        for nw in (new, None):
                            got = plans.classify(plan, action, nw, price, TODAY, held)
                            want = _classify_before(plan, action, nw, price, TODAY, held)
                            # INVALIDATE-only log keys (audit 2026-11-08 21:30 #3g)
                            inv = {k: got.pop(k) for k in ("action", "buy_below_invalidation") if k in got}
                            assert bool(inv) == (want["relation"] == "INVALIDATE")
                            if inv:
                                assert inv == {"action": action,
                                               "buy_below_invalidation": action in ("OPEN", "ADD")}
                            extra = {k: got.pop(k) for k in ("revise_triggers",
                                                             "catalyst_month_only_pair")
                                     if k in got}
                            # OPEN/ADD-only log keys (_add_plan_record)
                            add = {k: got.pop(k) for k in ("add_under_hold_plan", "add_trigger",
                                                           "price_vs_add_trigger_pct") if k in got}
                            assert bool(add) == (action in ("OPEN", "ADD"))
                            if add:
                                assert add["add_under_hold_plan"] == (
                                    action == "ADD" and want["plan_says"] == plans._HOLD)
                            assert json.dumps(got, default=str) == json.dumps(want, default=str)
                            assert bool(extra) == (want["relation"] == "REVISE")
                            if extra:
                                assert extra["revise_triggers"]
                            n += 1
    assert n > 1000


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


@patch("config.PLAN_CLOSE_ON_EXIT", True)
def test_open_plan_closes_when_its_position_closed():
    """An OPEN/ADD plan whose trade closed after it is closed before the next
    verdict, so the next OPEN starts a new plan instead of following it."""
    plan = _plan(source="analyst")
    closed = {"id": 41, "exit_date": NOW - timedelta(days=1)}
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=plan), \
         patch("db.outcome_closed_since", return_value=closed) as look, \
         patch("db.touch_plan") as touch, patch("db.insert_plan", return_value=9):
        r = plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    look.assert_called_once_with("NVDA", plan["created_at"], entered_since=plan["created_at"])
    touch.assert_called_once_with(1, "CLOSED", close_reason="position closed")
    assert r["relation"] == "NEW_PLAN" and r["plan_id"] == 9



@patch("config.PLAN_CLOSE_ON_EXIT", True)
def test_revised_plan_closes_by_its_first_versions_entry():
    """M282: a REVISE plan postdates its trade's entry, so the entry bound is
    the first plan in the chain; the exit bound stays the current plan."""
    plan = _plan(source="analyst", revision=2)
    root = plan["created_at"] - timedelta(days=3)
    closed = {"id": 41, "exit_date": NOW - timedelta(hours=1)}
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=plan), \
         patch("db.plan_root_created_at", return_value=root) as rootq, \
         patch("db.outcome_closed_since", return_value=closed) as look, \
         patch("db.touch_plan") as touch, patch("db.insert_plan", return_value=9):
        plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    rootq.assert_called_once_with(plan["id"])
    look.assert_called_once_with("NVDA", plan["created_at"], entered_since=root)
    touch.assert_any_call(1, "CLOSED", close_reason="position closed")


def test_first_version_plan_skips_the_chain_lookup():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst", revision=0)), \
         patch("db.plan_root_created_at") as rootq, patch("db.touch_plan"):
        plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    rootq.assert_not_called()


@patch("config.PLAN_CLOSE_ON_EXIT", True)
def test_chain_lookup_failure_keeps_the_plan(caplog):
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst", revision=1)), \
         patch("db.plan_root_created_at", side_effect=RuntimeError("db down")), \
         patch("db.touch_plan") as touch, caplog.at_level("WARNING"):
        r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    touch.assert_called_once_with(1, "FOLLOW")
    assert "closed-trade lookup failed" in caplog.text

def test_unfilled_open_plan_stays_open():
    """No closed trade after the plan (it never filled): kept and judged."""
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), \
         patch("db.touch_plan") as touch:
        r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    touch.assert_called_once_with(1, "FOLLOW")
    assert r["relation"] == "FOLLOW"


def test_exited_plan_stays_open_by_default_and_records_the_exit():
    """PLAN_CLOSE_ON_EXIT off (G75 undecided): the analyst plan whose trade
    closed is not closed; the verdict is judged against it, exited_since_plan
    records the sale, and would_close_on_exit records what the switch would
    have done (professor M314, M315)."""
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), \
         patch("db.outcome_closed_since", return_value={"id": 41, "exit_date": NOW}) as look, \
         patch("db.sell_trades_since", return_value={"n": 1, "last_trade_id": 5,
                                                     "last_recorded_at": NOW, "last_reason": "stop"}), \
         patch("db.touch_plan") as touch:
        r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    look.assert_called_once()
    assert all(c.args[1] != "CLOSED" for c in touch.call_args_list)
    assert r["relation"] == "FOLLOW" and r["exited_since_plan"] is True
    assert (r["would_close_on_exit"], r["close_outcome_id"], r["close_check_status"]) == (True, 41, "ok")
    assert r["exit_sells"]["last_trade_id"] == 5


def test_held_open_plan_does_not_look_up_closed_trades():
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 450.0}), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), \
         patch("db.outcome_closed_since") as look, patch("db.touch_plan") as touch:
        plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    look.assert_not_called()
    assert all(c.args[1] != "CLOSED" for c in touch.call_args_list)


@patch("config.PLAN_CLOSE_ON_EXIT", True)
def test_closed_trade_lookup_failure_keeps_the_plan(caplog):
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), \
         patch("db.outcome_closed_since", side_effect=RuntimeError("db down")), \
         patch("db.touch_plan") as touch, caplog.at_level("WARNING"):
        r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    touch.assert_called_once_with(1, "FOLLOW")
    assert r["relation"] == "FOLLOW"
    assert "closed-trade lookup failed" in caplog.text


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
    # The row names its own code and config version (SEF run 100).
    prov = log_signal.call_args.kwargs["provenance"]
    assert "code_sha" in prov and "env_sha" in prov


def _watch_rows(plan, held_prices):
    import tasks
    redis = MagicMock()
    redis.set.return_value = True
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value=held_prices), \
         patch("stop_lock._redis", return_value=redis), \
         patch("config.PLAN_LEVEL_REVIEW_ENABLED", False), \
         patch("db.outcome_closed_since", return_value=None), \
         patch("db.log_signal") as log_signal:
        tasks.watch_plan_levels()
    return [(c.args[3], c.kwargs["provenance"]) for c in log_signal.call_args_list]


def test_level_hit_on_an_unheld_name_records_no_price_not_zero():
    """Audit 2026-11-17 #11: an unheld name has no position price; its date
    hits said "at $0.00". The text says "no price" and the price is null."""
    plan = _plan(ticker="PLTR", catalyst_date=None, horizon_end=date(2026, 10, 1))
    rows = _watch_rows(plan, {"NVDA": 470.0})
    assert len(rows) == 1
    text, prov = rows[0]
    assert text == "plan 1: horizon_end (2026-10-01) no price; would re-review"
    assert "$0.00" not in text and prov["price"] is None and prov["kind"] == "horizon_end"


def test_level_hit_on_a_held_name_keeps_its_price_and_text():
    rows = _watch_rows(_plan(catalyst_date=None), {"NVDA": 470.0})
    assert len(rows) == 1 and rows[0][1]["price"] == 470.0
    assert rows[0][0] == "plan 1: invalidation (480.0) at $470.00; would re-review"


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


def test_level_hit_records_whether_the_position_exited_since_the_plan():
    """Record-only (audit 2026-11-08 11:07 #2): held -> False; unheld with an
    exit after the plan -> True; nothing held at all -> None."""
    import tasks
    plan = _plan(catalyst_date=None)
    assert tasks._plan_exited(plan, {"NVDA": 470.0}) is False
    with patch("db.outcome_closed_since", return_value={"id": 7, "exit_date": None}):
        assert tasks._plan_exited(plan, {"MSFT": 400.0}) is True
    with patch("db.outcome_closed_since", return_value=None):
        assert tasks._plan_exited(plan, {"MSFT": 400.0}) is False
    with patch("db.outcome_closed_since", side_effect=RuntimeError("db")):
        assert tasks._plan_exited(plan, {"MSFT": 400.0}) is None
    assert tasks._plan_exited(plan, {}) is None
    _, log_signal = _watch(False)
    assert log_signal.call_args.kwargs["provenance"]["exited_since_plan"] is False


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


def test_malformed_fields_on_a_held_hold_still_touch_the_plan():
    """Review of eed0fcb: a would_revise failure must not turn FOLLOW into ERROR."""
    bad = {**PARSED, "valuation": "n/a"}
    with patch("db.get_position_stop", return_value={"entry_price": 500.0, "stop_price": 450.0}), \
         patch("db.get_active_plan", return_value=_plan()), patch("db.touch_plan") as touch:
        r = plans.observe("NVDA", bad, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "FOLLOW" and r["would_revise"] is None
    touch.assert_called_once_with(1, "FOLLOW")


def test_closed_trade_lookup_needs_entry_and_fill_after_the_plan():
    """An older position closed late by reconcile must not close a plan made
    after its fill (professor m480): the query bounds the entry and uses the
    exit fill time where known."""
    from unittest.mock import MagicMock
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None
    since = datetime(2026, 11, 4, 15, 10, tzinfo=timezone.utc)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert _real_outcome_closed_since("NVDA", since) is None
    sql, params = cur.execute.call_args[0]
    assert "COALESCE(exit_fill_at, exit_date) > %s" in sql
    assert "entry_date >= %s - INTERVAL '5 minutes'" in sql
    assert params[2:] == (since, since)


def test_closed_trade_lookup_bounds_entry_by_entered_since():
    from unittest.mock import MagicMock
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None
    since = datetime(2026, 11, 4, 15, 10, tzinfo=timezone.utc)
    root = since - timedelta(days=3)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        _real_outcome_closed_since("NVDA", since, entered_since=root)
    assert cur.execute.call_args[0][1][2:] == (since, root)


# --- named levels on the plan row (migration 82) -------------------------------

_LADDER = {"atr": 10.0, "fair_value_blended": 640.0, "rungs": [
    {"price": 500.0, "band_high": 503.0, "types": ["ma50"]},
    {"price": 575.0, "band_high": 578.0, "types": ["swing_high"]}]}


def test_migration_82_is_unique_and_adds_the_level_columns():
    versions = [v for v, _ in db._MIGRATIONS]
    assert len(versions) == len(set(versions)) and versions.count(82) == 1
    assert versions == sorted(versions)
    sql = dict(db._MIGRATIONS)[82]
    assert sql.startswith("ALTER TABLE trade_plans")
    for col in db._PLAN_LEVEL_COLS[:5]:
        assert f"ADD COLUMN IF NOT EXISTS {col} " in sql
        assert col in db._PLAN_COLS
        assert sum(f"EXISTS {col} " in s for _, s in db._MIGRATIONS) == 1


def test_new_plan_stores_the_compiled_levels():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.get_trade_ladder", return_value=_LADDER) as lad, \
         patch("db.insert_plan", return_value=7) as ins:
        r = plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "NEW_PLAN" and r["plan_id"] == 7
    lad.assert_called_once_with("NVDA")
    row = ins.call_args.args[0]
    assert (row["structural_stop"], row["t1"], row["t2"], row["add_trigger"],
            row["levels_source"]) == (495.0, 575.0, 650.0, 578.0, "ladder")
    assert row["invalidation_price"] == 480.0 and row["target_price"] == 650.0   # unchanged


def test_revised_plan_stores_the_levels_and_seed_leaves_them_null():
    plan = _plan()
    moved = {**PARSED, "valuation": {"bull_case_target": "$800", "downside_scenario": "$480"}}
    with patch("db.get_position_stop", return_value={"entry_price": 540.0, "stop_price": 480.0}), \
         patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"), \
         patch("db.insert_plan", return_value=8) as ins:
        r = plans.observe("NVDA", moved, "ADD", 8, "Current Price: $545.00", now=NOW)
    assert r["relation"] == "REVISE" and r["new_plan_id"] == 8
    row = ins.call_args.args[0]
    assert row["levels_source"] == "analyst_only" and row["structural_stop"] == 480.0
    assert row["t2"] == 800.0 and row["replaces_id"] == 1
    stop = {"entry_price": 500.0, "stop_price": 450.0, "thesis_bucket": "Compounder"}
    with patch("db.get_position_stop", return_value=stop), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=3) as ins:
        assert plans.observe("MSFT", PARSED, "HOLD", 6, "Current Price: $540.00",
                             now=NOW)["relation"] == "SEEDED"
    assert not any(k in ins.call_args.args[0] for k in db._PLAN_LEVEL_COLS)


def test_new_plan_is_inserted_without_levels_when_the_ladder_lookup_fails(caplog):
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.get_trade_ladder", side_effect=RuntimeError("ladder down")), \
         patch("db.insert_plan", return_value=7) as ins:
        r = plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "NEW_PLAN" and r["plan_id"] == 7
    row = ins.call_args.args[0]
    assert not any(k in row for k in db._PLAN_LEVEL_COLS)
    assert row["entry_price"] == 540.0 and row["target_price"] == 650.0
    assert "plan levels not compiled" in caplog.text and "ladder down" in caplog.text


def _plan_conn(execute):
    conn, cur = MagicMock(), MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    cur.execute.side_effect = execute
    return conn, cur


def _missing(col):
    return psycopg2.errors.UndefinedColumn(f'column "{col}" of relation "trade_plans" does not exist')


def test_insert_plan_writes_the_level_columns():
    conn, cur = _plan_conn(None)
    cur.fetchone.return_value = (11,)
    plan = {**_plan(), "structural_stop": 495.0, "t1": 575.0, "t2": 650.0,
            "add_trigger": 578.0, "levels_source": "ladder"}
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.insert_plan(plan) == 11
    sql, vals = [c.args for c in cur.execute.call_args_list if c.args[0].startswith("INSERT")][0]
    assert "levels_source, level_sources, ladder_computed_at) VALUES" in sql
    assert vals[-7:] == (495.0, 575.0, 650.0, 578.0, "ladder", None, None)
    conn.commit.assert_called_once()


def test_insert_plan_without_the_migration_keeps_the_plan_and_the_supersede(caplog):
    db._missing_column_warned.discard("structural_stop")

    def execute(sql, *a):
        if sql.startswith("INSERT") and "structural_stop" in sql:
            raise _missing("structural_stop")
    conn, cur = _plan_conn(execute)
    cur.fetchone.return_value = (12,)
    plan = {**_plan(), "replaces_id": 1, "structural_stop": 495.0, "levels_source": "ladder"}
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.insert_plan(plan) == 12
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert sqls[0].startswith("UPDATE trade_plans SET status='superseded'")
    assert "ROLLBACK TO SAVEPOINT plan_levels" in sqls
    assert sqls[-1].startswith("INSERT") and "structural_stop" not in sqls[-1]
    conn.rollback.assert_not_called()                     # the supersede was not undone
    conn.commit.assert_called_once()
    assert "column structural_stop is missing (migration pending)" in caplog.text


def test_insert_plan_without_levels_uses_the_old_statement():
    conn, cur = _plan_conn(None)
    cur.fetchone.return_value = (13,)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.insert_plan(_plan()) == 13
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert len(sqls) == 1 and "structural_stop" not in sqls[0]


def test_get_active_plan_reads_without_the_level_columns_before_the_migration():
    base = tuple(c for c in db._PLAN_COLS if c not in db._PLAN_LEVEL_COLS)
    row = tuple({"id": 5, "ticker": "NVDA", "entry_price": 540, "target_price": 650}.get(c)
                for c in base)

    def execute(sql, *a):
        if "structural_stop" in sql:
            raise _missing("structural_stop")
    conn, cur = _plan_conn(execute)
    cur.fetchall.return_value = [row]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        plan = db.get_active_plan("NVDA")
    assert plan["id"] == 5 and plan["entry_price"] == 540.0 and plan["target_price"] == 650.0
    assert all(plan[c] is None for c in db._PLAN_LEVEL_COLS)
    conn.rollback.assert_called_once()


def test_get_active_plan_returns_the_levels():
    vals = {"id": 5, "ticker": "NVDA", "entry_price": 540, "structural_stop": 495,
            "t1": 575, "levels_source": "ladder"}
    conn, cur = _plan_conn(None)
    cur.fetchall.return_value = [tuple(vals.get(c) for c in db._PLAN_COLS)]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        plan = db.get_active_plan("NVDA")
    assert plan["structural_stop"] == 495.0 and isinstance(plan["t1"], float)
    assert plan["levels_source"] == "ladder" and plan["t2"] is None


# --- per-level sources and the ladder's computed_at (migration 84) --------------

_LADDER_AT = datetime(2026, 10, 29, 13, 30, tzinfo=timezone.utc)
_NO_TARGET = {**PARSED, "valuation": {"downside_scenario": "$480"}}


def test_migration_84_is_unique_and_adds_level_sources_and_ladder_computed_at():
    versions = [v for v, _ in db._MIGRATIONS]
    assert versions.count(84) == 1 and versions == sorted(versions)
    sql = dict(db._MIGRATIONS)[84]
    assert sql.startswith("ALTER TABLE trade_plans")
    assert "ADD COLUMN IF NOT EXISTS level_sources JSONB" in sql
    assert "ADD COLUMN IF NOT EXISTS ladder_computed_at TIMESTAMPTZ" in sql
    for col in ("level_sources", "ladder_computed_at"):
        assert col in db._PLAN_LEVEL_COLS and col in db._PLAN_COLS
        assert sum(f"EXISTS {col} " in s for _, s in db._MIGRATIONS) == 1
    assert "levels_source" in db._PLAN_LEVEL_COLS                   # kept as is


def test_compile_plan_level_sources_with_a_ladder():
    p = plans.compile_plan(PARSED, 540.0, _LADDER, None, TODAY)
    assert p["source"] == "ladder"
    assert p["level_sources"] == {"structural_stop": "ladder_support", "t1": "ladder_resistance",
                                  "t2": "analyst_target", "add_trigger": "ladder_resistance"}
    # a ladder row with no support rung: the stop is the analyst's invalidation
    # although "source" still says ladder (professor m488)
    high = {"atr": 10.0, "rungs": [{"price": 575.0, "band_high": 578.0, "types": ["swing_high"]}]}
    p = plans.compile_plan(_NO_TARGET, 540.0, high, None, TODAY)
    assert p["source"] == "ladder" and p["structural_stop"] == 480.0 and p["t2"] is None
    assert p["level_sources"] == {"structural_stop": "analyst_invalidation",
                                  "t1": "ladder_resistance", "t2": "none",
                                  "add_trigger": "ladder_resistance"}


def test_compile_plan_level_sources_without_a_ladder():
    p = plans.compile_plan(PARSED, 540.0, None, None, TODAY)
    assert p["level_sources"] == {"structural_stop": "analyst_invalidation", "t1": "none",
                                  "t2": "analyst_target", "add_trigger": "none"}
    p = plans.compile_plan({"thesis_bucket": "Accelerator"}, 540.0, None, None, TODAY)
    assert p["structural_stop"] is None
    assert p["level_sources"] == dict.fromkeys(("structural_stop", "t1", "t2", "add_trigger"), "none")


def test_compile_plan_level_sources_for_the_t2_fallbacks():
    p = plans.compile_plan(_NO_TARGET, 540.0, _LADDER, None, TODAY)
    assert p["t2"] == 640.0 and p["level_sources"]["t2"] == "fair_value"
    ladder = {**_LADDER, "fair_value_blended": 530.0, "rungs": _LADDER["rungs"] + [
        {"price": 610.0, "band_high": 613.0, "types": ["52w_high"]}]}
    p = plans.compile_plan(_NO_TARGET, 540.0, ladder, None, TODAY)
    assert p["t2"] == 610.0 and p["level_sources"]["t2"] == "ladder_resistance"
    assert p["level_sources"]["t1"] == "ladder_resistance" and p["t1"] == 575.0


def test_compile_plan_old_keys_are_unchanged():
    """level_sources is the only new key; every other key and value, and
    their order, is what compile_plan returned before it."""
    p = plans.compile_plan(PARSED, 540.0, _LADDER, 500.0, TODAY)
    assert list(p)[-1] == "level_sources"
    p.pop("level_sources")
    assert json.dumps(p) == json.dumps({
        "source": "ladder", "invalidation_analyst": 480.0, "time_stop": p["time_stop"],
        "catalyst_date": "2026-11-20", "structural_stop": 495.0, "t1": 575.0, "t2": 650.0,
        "add_trigger": 578.0, "support_types": ["ma50"], "resistance_types": ["swing_high"],
        "actual_stop": 500.0, "structural_stop_pct": round(45 / 540 * 100, 2),
        "r_to_t1": round(35 / 45, 2), "stop_gap_pct": round(5 / 540 * 100, 2)})
    bare = plans.compile_plan(_NO_TARGET, 540.0, None, None, TODAY)
    bare.pop("level_sources")
    assert json.dumps(bare) == json.dumps({
        "source": "analyst_only", "invalidation_analyst": 480.0, "time_stop": bare["time_stop"],
        "catalyst_date": "2026-11-20", "structural_stop": 480.0, "t1": None, "t2": None,
        "add_trigger": None, "support_types": None, "resistance_types": None,
        "actual_stop": None, "structural_stop_pct": round(60 / 540 * 100, 2)})


def test_new_plan_stores_level_sources_and_ladder_computed_at():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.get_trade_ladder", return_value={**_LADDER, "computed_at": _LADDER_AT}), \
         patch("db.insert_plan", return_value=7) as ins:
        plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    row = ins.call_args.args[0]
    assert row["ladder_computed_at"] == _LADDER_AT and row["levels_source"] == "ladder"
    assert row["level_sources"] == {"structural_stop": "ladder_support", "t1": "ladder_resistance",
                                    "t2": "analyst_target", "add_trigger": "ladder_resistance"}
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=8) as ins:
        plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    row = ins.call_args.args[0]
    assert row["ladder_computed_at"] is None and row["levels_source"] == "analyst_only"
    assert row["level_sources"]["structural_stop"] == "analyst_invalidation"


def test_insert_plan_writes_level_sources_as_json_and_ladder_computed_at():
    conn, cur = _plan_conn(None)
    cur.fetchone.return_value = (14,)
    srcs = {"structural_stop": "ladder_support", "t1": "ladder_resistance",
            "t2": "analyst_target", "add_trigger": "ladder_resistance"}
    plan = {**_plan(), "structural_stop": 495.0, "levels_source": "ladder",
            "level_sources": srcs, "ladder_computed_at": _LADDER_AT}
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.insert_plan(plan) == 14
    sql, vals = [c.args for c in cur.execute.call_args_list if c.args[0].startswith("INSERT")][0]
    assert isinstance(vals[-2], psycopg2.extras.Json) and vals[-2].adapted == srcs
    assert vals[-1] == _LADDER_AT


def test_insert_plan_with_migration_84_pending_writes_the_plan_without_the_levels(caplog):
    db._missing_column_warned.discard("level_sources")

    def execute(sql, *a):
        if sql.startswith("INSERT") and "level_sources" in sql:
            raise _missing("level_sources")
    conn, cur = _plan_conn(execute)
    cur.fetchone.return_value = (15,)
    plan = {**_plan(), "replaces_id": 1, "structural_stop": 495.0, "levels_source": "ladder",
            "level_sources": {"t1": "none"}, "ladder_computed_at": _LADDER_AT}
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.insert_plan(plan) == 15
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert "ROLLBACK TO SAVEPOINT plan_levels" in sqls
    assert sqls[-1].startswith("INSERT") and "level_sources" not in sqls[-1] \
        and "ladder_computed_at" not in sqls[-1]
    conn.rollback.assert_not_called()
    conn.commit.assert_called_once()
    assert "column level_sources is missing (migration pending)" in caplog.text


def test_get_active_plan_with_migration_84_pending_reads_the_new_columns_as_none():
    base = tuple(c for c in db._PLAN_COLS if c not in db._PLAN_LEVEL_COLS)
    row = tuple({"id": 5, "ticker": "NVDA", "entry_price": 540}.get(c) for c in base)

    def execute(sql, *a):
        if "ladder_computed_at" in sql:
            raise _missing("ladder_computed_at")
    conn, cur = _plan_conn(execute)
    cur.fetchall.return_value = [row]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        plan = db.get_active_plan("NVDA")
    assert plan["id"] == 5 and plan["entry_price"] == 540.0
    assert plan["level_sources"] is None and plan["ladder_computed_at"] is None


def test_get_active_plan_returns_level_sources_and_ladder_computed_at():
    srcs = {"structural_stop": "analyst_invalidation", "t1": "none",
            "t2": "analyst_target", "add_trigger": "none"}
    vals = {"id": 5, "ticker": "NVDA", "structural_stop": 480, "levels_source": "ladder",
            "level_sources": srcs, "ladder_computed_at": _LADDER_AT}
    conn, cur = _plan_conn(None)
    cur.fetchall.return_value = [tuple(vals.get(c) for c in db._PLAN_COLS)]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        plan = db.get_active_plan("NVDA")
    assert plan["level_sources"] == srcs and plan["ladder_computed_at"] == _LADDER_AT
    assert plan["structural_stop"] == 480.0


def test_revise_trigger_failure_keeps_the_revise(caplog):
    """The trigger keys are log-only: a failure drops them, not the REVISE."""
    plan = _plan(source="analyst")
    new = {**plan, "target_price": (plan.get("target_price") or 100.0) * 2}
    with patch("agents.plans._revise_triggers", side_effect=TypeError("bad date")), \
         caplog.at_level("WARNING"):
        out = plans.classify(plan, "OPEN", new, 540.0, TODAY, held=False)
    assert out["relation"] == "REVISE" and "revise_triggers" not in out
    assert "revise_triggers failed" in caplog.text

def test_exited_since_plan_marks_an_analyst_plan_whose_position_was_sold():
    plan = _plan(source="analyst")
    brief = "Current Price: $540.00"
    sale = {"n": 2, "last_trade_id": 77, "last_recorded_at": NOW, "last_reason": "stop filled"}
    for sold, want, status in ((sale, True, "ok"), (None, False, "ok"),
                               (RuntimeError("db down"), None, "error")):
        kw = {"side_effect": sold} if isinstance(sold, Exception) else {"return_value": sold}
        with patch("db.get_position_stop", return_value=None), \
             patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"), \
             patch("db.sell_trades_since", **kw) as q:
            r = plans.observe("NVDA", PARSED, "WATCH", 6, brief, now=NOW)
        assert r["exited_since_plan"] is want and r["exited_since_plan_status"] == status, sold
        assert r["exit_sells"] == (sold if isinstance(sold, dict) else None)
        assert q.call_args.args == ("NVDA", plan["created_at"], NOW)
        assert r["plan_source"] == "analyst"      # record-only: the plan is not closed
    stop = {"entry_price": 500.0, "stop_price": 450.0, "thesis_bucket": "Compounder"}
    with patch("db.get_position_stop", return_value=stop), \
         patch("db.get_active_plan", return_value=plan), patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, brief, now=NOW)
    assert r["exited_since_plan"] is None and r["exited_since_plan_status"] == "not_checked"


def test_sell_trades_since_names_the_last_sale():
    from unittest.mock import MagicMock
    import db
    cur = MagicMock()
    cur.fetchone.return_value = (3, 91, NOW, "TRIM 1/3 at target")
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.sell_trades_since("NVDA", NOW - timedelta(days=1), NOW) == {
            "n": 3, "last_trade_id": 91, "last_recorded_at": NOW.isoformat(),
            "last_reason": "TRIM 1/3 at target"}
        json.dumps(db.sell_trades_since("NVDA", NOW))      # provenance-safe
        assert db.sell_trade_since("NVDA", NOW - timedelta(days=1)) is True
    assert "ORDER BY timestamp DESC" in cur.execute.call_args.args[0]


def test_sell_trade_since_bounds_the_window():
    from unittest.mock import MagicMock
    import db
    cur = MagicMock()
    cur.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.sell_trade_since("NVDA", NOW - timedelta(days=1), NOW) is False
        sql, args = cur.execute.call_args.args
        assert "timestamp <= %s" in sql and args[-1] == NOW
        db.sell_trade_since("NVDA", NOW - timedelta(days=1))
        assert "timestamp <= %s" not in cur.execute.call_args.args[0]


def test_plan_close_switch_is_off_by_default_and_named_in_the_treatment_config():
    """m585: the switch defaults off in config.py source, and the verdict's
    treatment config names it, so a row shows which plan rule wrote it."""
    from pathlib import Path
    import config
    from agents.analyst import _treatment_config
    assert ('PLAN_CLOSE_ON_EXIT = os.getenv("PLAN_CLOSE_ON_EXIT", "false").lower() == "true"'
            in Path(config.__file__).read_text())
    for value in (False, True):
        with patch("config.PLAN_CLOSE_ON_EXIT", value):
            assert _treatment_config()["PLAN_CLOSE_ON_EXIT"] is value


def test_close_check_is_recorded_when_nothing_closed_or_the_lookup_fails():
    for kw, want in (({"return_value": None}, (False, None, "ok")),
                     ({"side_effect": RuntimeError("db down")}, (None, None, "error"))):
        with patch("db.get_position_stop", return_value=None), \
             patch("db.get_active_plan", return_value=_plan(source="analyst")), \
             patch("db.outcome_closed_since", **kw), patch("db.touch_plan"):
            r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
        assert (r["would_close_on_exit"], r["close_outcome_id"], r["close_check_status"]) == want
    stop = {"entry_price": 500.0, "stop_price": 450.0, "thesis_bucket": "Compounder"}
    with patch("db.get_position_stop", return_value=stop), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["close_check_status"] == "not_checked"


def test_held_plan_records_invalidation_beside_the_broker_stop():
    """Audit 11-06 03:34 #1: the plan says wrong below 480, the stop sells at 460."""
    stop = {"entry_price": 540.0, "stop_price": 460.0, "thesis_bucket": "Accelerator"}
    with patch("db.get_position_stop", return_value=stop), \
         patch("db.get_active_plan", return_value=_plan()), \
         patch("db.get_recent_verdicts", return_value=[]), \
         patch("db.touch_plan"), patch("db.insert_plan", return_value=9):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    g = r["invalidation_vs_stop"]
    assert g["invalidation"] == 480.0 and g["stop"] == 460.0
    assert g["gap_frac"] == round(480 / 460 - 1, 4) and g["stop_below_invalidation"] is True
    assert g["trailing_active"] is None          # the record had no trail flag (professor 59q)


def test_unheld_plan_has_no_invalidation_vs_stop():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan()), \
         patch("db.sell_trades_since", return_value=None), \
         patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "WATCH", 6, "Current Price: $540.00", now=NOW)
    assert r["invalidation_vs_stop"] is None


def test_stop_read_is_strict_so_an_error_never_reads_as_not_held():
    """SEF 139g: a non-strict None on a DB error would close a held_seed plan."""
    with patch("db.get_position_stop", return_value=None) as gps, \
         patch("db.get_active_plan", return_value=None):
        plans.observe("MSFT", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert gps.call_args.kwargs.get("strict") is True


# ── invalidation loosening on a REVISE (record only) ─────────────────────────

_REVISE_STOP = {"entry_price": 540.0, "stop_price": 480.0}


def _observe_revise(downside, price, ladder=None, ladder_exc=None, plan=None, target="$650",
                    action="ADD"):
    """An OPEN/ADD verdict against _plan() (invalidation 480, target 650)."""
    parsed = {**PARSED, "valuation": {"bull_case_target": target, "downside_scenario": downside}}
    lad = (patch("db.get_trade_ladder", side_effect=ladder_exc) if ladder_exc
           else patch("db.get_trade_ladder", return_value=ladder))
    with patch("db.get_position_stop", return_value=_REVISE_STOP), \
         patch("db.get_active_plan", return_value=plan or _plan()), \
         patch("db.touch_plan") as touch, patch("db.insert_plan", return_value=8) as ins, \
         lad as ladder_read, \
         patch("tasks._broker_held", return_value=(None, "no_snapshot", {})) as broker:
        r = plans.observe("NVDA", parsed, action, 8, f"Current Price: ${price:.2f}", now=NOW)
    # The record reads nothing new: the one broker-snapshot read and the one
    # ladder read (for the new revision's levels) are the ones there before.
    assert broker.call_count == 1 and ladder_read.call_count == (1 if r["relation"] == "REVISE" else 0)
    return r, ins, touch, broker


def test_revise_that_lowers_invalidation_far_from_it_is_recorded():
    r, ins, touch, _ = _observe_revise("$400", 545.0, ladder=_LADDER)
    assert r["relation"] == "REVISE" and r["revise_triggers"] == ["invalidation"]
    assert r["invalidation_prev"] == 480.0 and r["invalidation_new"] == 400.0
    assert r["invalidation_loosened"] is True
    assert r["ladder_atr"] == 10.0 and r["price_vs_prev_invalidation_atr"] == 6.5
    assert r["add_near_invalidation"] is False                 # 65 above, 1 ATR is 10
    assert ins.call_args.args[0]["invalidation_price"] == 400.0   # the row written is unchanged
    touch.assert_called_once_with(1, "REVISE")


def test_revise_within_one_atr_of_the_old_invalidation_is_near():
    r, *_ = _observe_revise("$400", 485.0, ladder=_LADDER)
    assert r["relation"] == "REVISE" and r["invalidation_loosened"] is True
    assert r["add_near_invalidation"] is True and r["price_vs_prev_invalidation_atr"] == 0.5
    edge, *_ = _observe_revise("$400", 490.0, ladder=_LADDER)     # exactly 1 ATR
    assert edge["add_near_invalidation"] is True and edge["price_vs_prev_invalidation_atr"] == 1.0
    opened, *_ = _observe_revise("$400", 485.0, ladder=_LADDER, action="OPEN")
    assert opened["relation"] == "REVISE" and opened["add_near_invalidation"] is True


def test_revise_that_raises_invalidation_is_not_loosened():
    r, *_ = _observe_revise("$530", 545.0, ladder=_LADDER)      # 480 -> 530
    assert r["relation"] == "REVISE" and r["invalidation_loosened"] is False
    assert (r["invalidation_prev"], r["invalidation_new"]) == (480.0, 530.0)
    assert r["add_near_invalidation"] is False


def test_unknown_atr_price_or_level_gives_none():
    no_ladder, *_ = _observe_revise("$400", 485.0)
    assert no_ladder["invalidation_loosened"] is True
    assert no_ladder["add_near_invalidation"] is None and no_ladder["ladder_atr"] is None
    assert no_ladder["price_vs_prev_invalidation_atr"] is None
    failed, ins, _, _ = _observe_revise("$400", 485.0, ladder_exc=RuntimeError("ladder down"))
    assert failed["relation"] == "REVISE" and failed["new_plan_id"] == 8
    assert failed["add_near_invalidation"] is None and failed["invalidation_loosened"] is True
    zero_atr, *_ = _observe_revise("$400", 485.0, ladder={**_LADDER, "atr": 0})
    assert zero_atr["add_near_invalidation"] is None
    # A revision whose new invalidation is unknown (a "downside" above the
    # price is dropped), revising on the target.
    no_new, *_ = _observe_revise("$600", 545.0, ladder=_LADDER, target="$800")
    assert no_new["relation"] == "REVISE" and no_new["revise_triggers"] == ["target"]
    assert no_new["invalidation_new"] is None and no_new["invalidation_loosened"] is None
    assert no_new["add_near_invalidation"] is False
    # A plan with no invalidation of its own.
    no_prev, *_ = _observe_revise("$400", 545.0, ladder=_LADDER, target="$800",
                                  plan=_plan(invalidation_price=None))
    assert no_prev["relation"] == "REVISE" and no_prev["invalidation_prev"] is None
    assert no_prev["invalidation_loosened"] is None and no_prev["add_near_invalidation"] is None
    # No price in the brief: no plan is proposed, so no REVISE and no record.
    with patch("db.get_position_stop", return_value=_REVISE_STOP), \
         patch("db.get_active_plan", return_value=_plan()), patch("db.touch_plan"):
        r = plans.observe("NVDA", PARSED, "ADD", 8, "no price here", now=NOW)
    assert r["relation"] != "REVISE" and "invalidation_loosened" not in r


def test_invalidation_record_only_on_revise_and_never_raises():
    follow, *_ = _observe_revise("$480", 545.0, ladder=_LADDER)    # same levels: FOLLOW
    assert follow["relation"] == "FOLLOW"
    assert not any(k in follow for k in plans._INVALIDATION_RECORD_KEYS)
    bad = plans._invalidation_record(None, {}, 500.0, 10.0)
    assert {k: bad[k] for k in plans._INVALIDATION_RECORD_KEYS} == \
        {k: None for k in plans._INVALIDATION_RECORD_KEYS}
    assert bad["invalidation_record_error"] == "AttributeError"
    odd = plans._invalidation_record({"invalidation_price": "x"}, {"invalidation_price": float("nan")},
                                     "y", "z")
    assert set(odd) == set(plans._INVALIDATION_RECORD_KEYS) and not any(odd.values())


def test_revise_relation_and_existing_keys_unchanged_by_the_record():
    for downside, price in (("$400", 545.0), ("$400", 485.0), ("$530", 545.0)):
        with_rec, ins_a, *_ = _observe_revise(downside, price, ladder=_LADDER)
        with patch("agents.plans._invalidation_record", return_value={}):
            without, ins_b, *_ = _observe_revise(downside, price, ladder=_LADDER)
        assert {k: v for k, v in with_rec.items() if k not in plans._INVALIDATION_RECORD_KEYS} \
            == without
        assert ins_a.call_args.args[0] == ins_b.call_args.args[0]



def _watch_redis(plan, held_prices, review=False, log_id=7, cap_taken=False, rows=None):
    import tasks
    rows = [] if rows is None else rows
    redis = MagicMock()
    redis.set.side_effect = lambda key, *a, **k: not (cap_taken and key.startswith("helios:plan_review:"))
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value=held_prices), \
         patch("stop_lock._redis", return_value=redis), \
         patch("config.PLAN_LEVEL_REVIEW_ENABLED", review), \
         patch("tasks.review_plan_level_hit.apply_async"), \
         patch("db.outcome_closed_since", return_value=None), \
         patch("db.log_signal", return_value=log_id) as ls:
        tasks.watch_plan_levels()
    rows.extend(ls.call_args_list)
    ttl = {c.args[0]: c.kwargs.get("ex") for c in redis.set.call_args_list
           if str(c.args[0]).startswith("helios:plan_hit:")}
    ttl.update({c.args[0]: c.args[1] for c in redis.expire.call_args_list})
    return ttl


def test_date_hits_are_deduped_once_per_plan_price_hits_daily():
    """catalyst_passed and horizon_end happen once per plan: their key
    outlives the plan's horizon. invalidation and target keep one day."""
    import tasks
    keys = _watch_redis(_plan(catalyst_date=None, horizon_end=date(2026, 10, 1)), {"NVDA": 470.0})
    assert keys == {"helios:plan_hit:1:invalidation": 86400,
                    "helios:plan_hit:1:horizon_end": tasks._PLAN_DATE_HIT_TTL_S}
    keys = _watch_redis(_plan(catalyst_date=date(2026, 10, 1), horizon_end=date(2099, 1, 1)),
                        {"NVDA": 660.0})
    assert keys == {"helios:plan_hit:1:target": 86400,
                    "helios:plan_hit:1:catalyst_passed": tasks._PLAN_DATE_HIT_TTL_S}
    assert tasks._PLAN_DATE_HIT_TTL_S >= 365 * 86400
    assert tasks._plan_hit_ttl("structural_stop") == 86400


def test_a_date_hit_logs_once_across_days():
    """With a Redis that keeps the key, the same horizon_end logs on the
    first day only, however many days the plan stays active."""
    import tasks
    store = {}

    def _set(key, val, nx=False, ex=None):
        if nx and key in store:
            return None
        store[key] = ex
        return True

    def _expire(key, ex):
        store[key] = ex
    redis = MagicMock()
    redis.set.side_effect = _set
    redis.expire.side_effect = _expire
    redis.delete.side_effect = lambda key: store.pop(key, None)
    plan = _plan(ticker="PLTR", catalyst_date=None, horizon_end=date(2026, 10, 1))
    counts = []
    for day in range(3):
        with patch("market_data.is_market_hours", return_value=True), \
             patch("db.get_active_plans", return_value=[plan]), \
             patch("tasks._held_prices", return_value={}), \
             patch("stop_lock._redis", return_value=redis), \
             patch("config.PLAN_LEVEL_REVIEW_ENABLED", False), \
             patch("db.outcome_closed_since", return_value=None), \
             patch("db.log_signal", return_value=7) as ls:
            tasks.watch_plan_levels()
        counts.append(sum(1 for c in ls.call_args_list if c.args[1] == "PLAN_LEVEL_HIT"))
        # A one-day key would expire here; the date key does not.
        for k, ex in list(store.items()):
            if ex is not None and ex <= 86400:
                store.pop(k)
    assert counts == [1, 0, 0]



def test_date_hit_without_a_queued_review_keeps_the_one_day_key():
    """Reviews on, the day's review cap already taken: the date hit's key keeps
    one day, so its review comes on a later day; the row says daily."""
    import tasks
    rows = []
    plan = _plan(catalyst_date=None, horizon_end=date(2026, 10, 1))
    keys = _watch_redis(plan, {"NVDA": 540.0}, review=True, cap_taken=True, rows=rows)
    assert keys == {"helios:plan_hit:1:horizon_end": 86400}
    assert rows[0].kwargs["provenance"]["hit_dedupe"] == "daily"
    assert rows[0].kwargs["provenance"]["review_queued"] is False
    rows = []
    keys = _watch_redis(plan, {"NVDA": 540.0}, review=True, rows=rows)        # queued
    assert keys["helios:plan_hit:1:horizon_end"] == tasks._PLAN_DATE_HIT_TTL_S
    assert rows[0].kwargs["provenance"]["hit_dedupe"] == "once_per_plan"


def test_hit_dedupe_is_on_every_row():
    rows = []
    _watch_redis(_plan(catalyst_date=None, horizon_end=date(2026, 10, 1)), {"NVDA": 470.0}, rows=rows)
    got = {c.kwargs["provenance"]["kind"]: c.kwargs["provenance"]["hit_dedupe"] for c in rows}
    assert got == {"invalidation": "daily", "horizon_end": "once_per_plan"}


def test_a_failed_row_write_keeps_the_one_day_key(caplog):
    """log_signal returned no id (spilled or degraded): the one-day claim stays,
    so it is tried again the next day, not every tick; a warning says so."""
    import logging
    import tasks
    redis = MagicMock()
    redis.set.return_value = True
    plan = _plan(catalyst_date=None, horizon_end=date(2026, 10, 1))
    with patch("market_data.is_market_hours", return_value=True), \
         patch("db.get_active_plans", return_value=[plan]), \
         patch("tasks._held_prices", return_value={}), \
         patch("stop_lock._redis", return_value=redis), \
         patch("config.PLAN_LEVEL_REVIEW_ENABLED", False), \
         patch("db.outcome_closed_since", return_value=None), \
         patch("db.log_signal", return_value=None), \
         caplog.at_level(logging.WARNING, logger="tasks"):
        tasks.watch_plan_levels()
    assert redis.set.call_args.args[0] == "helios:plan_hit:1:horizon_end"
    assert redis.set.call_args.kwargs == {"nx": True, "ex": 86400}
    redis.delete.assert_not_called()
    redis.expire.assert_not_called()
    assert "row not written (spilled or degraded); one-day key kept" in caplog.text
