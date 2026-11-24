"""Record-only plan ledger fields (audit 2026-11-22 04:56 #1, #3c):
reseeded_after_invalidate / reseed on a NEW_PLAN or SEEDED plan,
stopped_out_since_plan on an analyst plan kept active after its exit, and
horizon_basis on every plan a verdict writes. Nothing here changes a plan."""
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agents import plans
import config
import db

_real_last_closed_plan = db.last_closed_plan
_real_outcome_closed_since = db.outcome_closed_since

TODAY = date(2026, 10, 29)
NOW = datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc)
PARSED = {
    "valuation": {"bull_case_target": "$650 (12 months)", "downside_scenario": "$480"},
    "catalyst": {"event": "Q1 earnings", "date": "2026-11-20"},
    "expectation_gap": {"primary_risk": "capex slowdown"},
    "thesis_bucket": "Accelerator", "thesis_summary": "AI capex beneficiary",
}
BREACH = NOW - timedelta(hours=1, minutes=30)
STOP = {"entry_price": 500.0, "stop_price": 431.0, "thesis_bucket": "Compounder",
        "opened_at": NOW - timedelta(days=1)}
PREV = {"id": 5, "source": "analyst", "status": "closed", "close_reason": "price at or below invalidation",
        "closed_at": BREACH, "entry_price": 500.0, "invalidation_price": 472.0}


@pytest.fixture(autouse=True)
def _isolated():
    with patch("tasks._broker_held", return_value=(None, "no_snapshot", {})), \
         patch("db.outcome_closed_since", return_value=None), \
         patch("db.get_trade_ladder", return_value=None), \
         patch("db.touch_plan"), \
         patch.object(config, "PLAN_RESEED_RECORD_ENABLED", True), \
         patch.object(config, "PLAN_RESEED_WINDOW_HOURS", None):
        yield


def _plan(**kw):
    p = plans.plan_from_verdict("NVDA", PARSED, 7, 540.0, TODAY)
    p.update(id=1, created_at=NOW - timedelta(days=2), last_seen_at=None, revision=0)
    p.update(kw)
    return p


def _seed(prev=PREV, stop=STOP, prev_exc=None, now=NOW):
    look = (patch("db.last_closed_plan", side_effect=prev_exc) if prev_exc
            else patch("db.last_closed_plan", return_value=prev))
    with patch("db.get_position_stop", return_value=stop), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=10) as ins, look as read:
        r = plans.observe("ANET", PARSED, "HOLD", 6, "Current Price: $540.00", now=now)
    return r, ins, read


# --- reseeded_after_invalidate ----------------------------------------------

def test_seed_after_a_breach_on_the_same_position_is_a_reseed():
    r, ins, read = _seed()
    assert r["relation"] == "SEEDED" and r["reseeded_after_invalidate"] is True
    rec = r["reseed"]
    assert rec["status"] == "ok" and rec["prev_plan_id"] == 5 and rec["prev_plan_source"] == "analyst"
    assert rec["prev_invalidation_rel"] == 0.944            # 472 / 500
    assert rec["new_invalidation_rel"] == 0.862             # the seed: stop 431 / entry 500
    assert rec["new_vs_prev_invalidation"] == round(431 / 472 - 1, 4)
    assert rec["breach_at"] == BREACH.isoformat() and rec["breach_basis"] == "invalidate_verdict"
    assert rec["hours_since_breach"] == 1.5 and rec["held"] is True
    assert rec["position_open_at_breach"] is True and rec["window_hours"] is None
    read.assert_called_once_with("ANET", timeout_ms=500)
    seed = ins.call_args.args[0]
    assert seed["source"] == "held_seed" and seed["invalidation_price"] == 431.0   # the plan written is unchanged


def test_reseed_record_holds_no_prices_and_serialises():
    r, *_ = _seed()
    json.dumps(r)                                            # provenance is written with plain json.dumps
    rec = r["reseed"]
    for k in ("prev_invalidation_rel", "new_invalidation_rel"):
        assert 0 < rec[k] < 2
    assert not {"invalidation_price", "entry_price", "stop_price"} & set(rec)


def test_window_bounds_the_flag_and_is_stored():
    with patch.object(config, "PLAN_RESEED_WINDOW_HOURS", 1):
        r, *_ = _seed()
    assert r["reseeded_after_invalidate"] is False
    assert r["reseed"]["window_hours"] == 1 and r["reseed"]["hours_since_breach"] == 1.5
    with patch.object(config, "PLAN_RESEED_WINDOW_HOURS", 2):
        assert _seed()[0]["reseeded_after_invalidate"] is True


@pytest.mark.parametrize("reason", ["revised", "position closed", "expired", None])
def test_a_previous_plan_closed_otherwise_is_not_a_reseed(reason):
    r, *_ = _seed(prev={**PREV, "close_reason": reason, "status": "superseded"})
    assert r["reseeded_after_invalidate"] is False
    assert r["reseed"]["breach_at"] is None and r["reseed"]["hours_since_breach"] is None
    assert r["reseed"]["prev_plan_id"] == 5 and r["reseed"]["prev_close_reason"] == reason


def test_no_previous_plan():
    r, *_ = _seed(prev=None)
    assert r["reseeded_after_invalidate"] is False and r["reseed"]["status"] == "no_previous_plan"


def test_a_failed_read_is_unknown_not_false():
    r, ins, _ = _seed(prev_exc=RuntimeError("canceling statement due to statement timeout"))
    assert r["relation"] == "SEEDED" and r["plan_id"] == 10      # the seed is still written
    assert r["reseeded_after_invalidate"] is None
    assert r["reseed"]["status"] == "error:RuntimeError" and r["reseed"]["read_ms"] is not None
    ins.assert_called_once()


def test_a_position_opened_after_the_breach_is_not_a_reseed():
    r, *_ = _seed(stop={**STOP, "opened_at": BREACH + timedelta(minutes=10)})
    assert r["reseeded_after_invalidate"] is False and r["reseed"]["position_open_at_breach"] is False


def test_a_stop_row_without_opened_at_leaves_the_flag_unknown():
    r, *_ = _seed(stop={k: v for k, v in STOP.items() if k != "opened_at"})
    assert r["reseeded_after_invalidate"] is None and r["reseed"]["position_open_at_breach"] is None


def test_an_unheld_new_plan_after_a_breach_is_not_a_reseed():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=11), \
         patch("db.last_closed_plan", return_value=PREV):
        r = plans.observe("ANET", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "NEW_PLAN" and r["reseeded_after_invalidate"] is False
    assert r["reseed"]["held"] is False and r["reseed"]["hours_since_breach"] == 1.5
    assert r["reseed"]["new_invalidation_rel"] == round(480 / 540, 4)


def test_a_held_new_plan_after_a_breach_is_a_reseed():
    with patch("db.get_position_stop", return_value=STOP), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=11), \
         patch("db.last_closed_plan", return_value=PREV):
        r = plans.observe("ANET", PARSED, "ADD", 7, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "NEW_PLAN" and r["reseeded_after_invalidate"] is True


def test_switch_off_reads_nothing_and_records_nothing():
    with patch.object(config, "PLAN_RESEED_RECORD_ENABLED", False):
        r, _, read = _seed()
    read.assert_not_called()
    assert "reseed" not in r and "reseeded_after_invalidate" not in r
    assert r["relation"] == "SEEDED" and r["horizon_basis"] == "catalyst"


def test_reseed_is_read_only_on_new_plans():
    with patch("db.get_position_stop", return_value=STOP), \
         patch("db.get_active_plan", return_value=_plan(source="held_seed")), \
         patch("db.last_closed_plan") as read:
        r = plans.observe("ANET", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "FOLLOW" and "reseed" not in r
    read.assert_not_called()


def test_reseed_record_never_raises():
    with patch("db.last_closed_plan", return_value={"id": 5, "closed_at": "garbage",
                                                    "close_reason": "price at or below invalidation"}):
        out = plans._reseed_record("ANET", {"invalidation_price": "x"}, STOP, True, NOW)
    assert out["reseeded_after_invalidate"] is None and out["reseed"]["status"].startswith("error:")


def test_the_same_verdict_that_closes_a_plan_sees_it_as_the_previous_plan():
    """A held_seed plan closed by this verdict (expired) is written first,
    so the read made right after sees it; it isn't a breach."""
    expired = _plan(source="held_seed", horizon_end=TODAY - timedelta(days=1))
    with patch("db.get_position_stop", return_value=STOP), \
         patch("db.get_active_plan", return_value=expired), \
         patch("db.insert_plan", return_value=12), \
         patch("db.last_closed_plan", return_value={**PREV, "id": 1, "close_reason": "expired"}):
        r = plans.observe("ANET", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "SEEDED" and r["expired_plan_id"] == 1
    assert r["reseeded_after_invalidate"] is False and r["reseed"]["prev_plan_id"] == 1


# --- db.last_closed_plan -----------------------------------------------------

def _mock_conn(row):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = row
    return conn, cur


def test_last_closed_plan_is_capped_and_reads_inactive_plans():
    conn, cur = _mock_conn((5, "analyst", "closed", "price at or below invalidation", BREACH, 500, 472))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        out = _real_last_closed_plan("ANET", timeout_ms=500)
    first, second = cur.execute.call_args_list
    assert first.args[0] == "SET LOCAL statement_timeout = 500"
    assert "status <> 'active'" in second.args[0] and "ORDER BY id DESC LIMIT 1" in second.args[0]
    assert second.args[1] == ("ANET", db.TRADING_ENV)
    assert out == {"id": 5, "source": "analyst", "status": "closed",
                   "close_reason": "price at or below invalidation", "closed_at": BREACH,
                   "entry_price": 500.0, "invalidation_price": 472.0}
    conn.rollback.assert_called_once()
    put.assert_called_once_with(conn)


def test_last_closed_plan_none_and_raises():
    conn, _ = _mock_conn(None)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert _real_last_closed_plan("ANET") is None
    conn, cur = _mock_conn(None)
    cur.execute.side_effect = RuntimeError("timeout")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        with pytest.raises(RuntimeError):
            _real_last_closed_plan("ANET")
    put.assert_called_once_with(conn)


def test_outcome_closed_since_returns_the_exit_time_and_source():
    fill = NOW - timedelta(hours=3)
    conn, cur = _mock_conn((41, NOW, fill, "broker_exit"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = _real_outcome_closed_since("MSFT", NOW - timedelta(days=5))
    assert out == {"id": 41, "exit_date": NOW, "exit_at": fill, "exit_source": "broker_exit"}
    assert "COALESCE(exit_fill_at, exit_date), exit_source" in cur.execute.call_args.args[0]


# --- stopped_out_since_plan --------------------------------------------------

def _exited(closed):
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), \
         patch("db.outcome_closed_since", return_value=closed), \
         patch("db.sell_trades_since", return_value={"n": 1, "last_trade_id": 5,
                                                     "last_recorded_at": NOW.isoformat(),
                                                     "last_reason": "Reconciled exit"}), \
         patch("db.touch_plan") as touch:
        r = plans.observe("MSFT", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    return r, touch


def test_an_exit_after_the_plan_is_recorded_while_the_plan_stays_active():
    fill = NOW - timedelta(hours=3)
    r, touch = _exited({"id": 41, "exit_date": NOW, "exit_at": fill, "exit_source": "broker_exit"})
    assert all(c.args[1] != "CLOSED" for c in touch.call_args_list)
    assert r["relation"] == "FOLLOW" and r["exited_since_plan"] is True
    assert r["stopped_out_since_plan"] == {"exit_at": fill.isoformat(), "kind": "broker_exit",
                                           "outcome_id": 41}
    json.dumps(r)


def test_an_exit_with_no_source_is_unknown_and_falls_back_to_exit_date():
    r, _ = _exited({"id": 41, "exit_date": NOW})
    assert r["stopped_out_since_plan"] == {"exit_at": NOW.isoformat(), "kind": "unknown", "outcome_id": 41}


def test_no_exit_and_held_plans_record_none():
    r, _ = _exited(None)
    assert r["stopped_out_since_plan"] is None and r["close_check_status"] == "ok"
    with patch("db.get_position_stop", return_value=STOP), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")):
        r = plans.observe("MSFT", PARSED, "HOLD", 6, "Current Price: $545.00", now=NOW)
    assert r["stopped_out_since_plan"] is None and r["close_check_status"] == "not_checked"


def test_with_close_on_exit_on_the_plan_closes_and_no_record_is_kept():
    with patch.object(config, "PLAN_CLOSE_ON_EXIT", True), \
         patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=_plan(source="analyst")), \
         patch("db.outcome_closed_since", return_value={"id": 41, "exit_date": NOW,
                                                         "exit_at": NOW, "exit_source": "broker_exit"}), \
         patch("db.last_closed_plan", return_value=None), \
         patch("db.touch_plan") as touch:
        r = plans.observe("MSFT", PARSED, "WATCH", 6, "Current Price: $545.00", now=NOW)
    touch.assert_called_once_with(1, "CLOSED", close_reason="position closed")
    assert r["relation"] == "NO_PLAN" and "stopped_out_since_plan" not in r


# --- horizon_basis -------------------------------------------------------------

@pytest.mark.parametrize("cdate, basis, on_or_before", [
    ("2026-10-29", "catalyst", True),           # today: catalyst + grace
    ("2026-10-28", "bucket_default", True),     # yesterday: the bucket's days
    ("2026-11-20", "catalyst", False),
    (None, "bucket_default", None),
    ("2026-10", "catalyst", False),             # month-only: read as its last day, 10-31
    ("2026-09", "bucket_default", True),
])
def test_horizon_basis_matches_the_branch_plan_from_verdict_took(cdate, basis, on_or_before):
    parsed = {**PARSED, "catalyst": {"event": "x", "date": cdate}}
    plan = plans.plan_from_verdict("NVDA", parsed, 7, 540.0, TODAY)
    rec = plans._horizon_record(plan, TODAY)
    cd = plan["catalyst_date"]
    assert rec["horizon_basis"] == basis and rec["catalyst_on_or_before_today"] is on_or_before
    grace_end = cd + timedelta(days=plans._CATALYST_GRACE_DAYS) if cd else None
    assert (plan["horizon_end"] == grace_end) is (basis == "catalyst")
    assert rec["horizon_days"] == (plan["horizon_end"] - TODAY).days
    assert rec["catalyst_days_from_today"] == ((cd - TODAY).days if cd else None)


def test_yesterdays_catalyst_gets_the_bucket_horizon():
    """The #3c case: a catalyst dated yesterday gets the bucket default (90 d
    for Accelerator), one dated today gets catalyst + 5 d. Recorded, not changed."""
    y = plans.plan_from_verdict("NVDA", {**PARSED, "catalyst": {"date": "2026-10-28"}}, 7, 540.0, TODAY)
    t = plans.plan_from_verdict("NVDA", {**PARSED, "catalyst": {"date": "2026-10-29"}}, 7, 540.0, TODAY)
    assert plans._horizon_record(y, TODAY)["horizon_days"] == plans._HORIZON_DAYS["Accelerator"]
    assert plans._horizon_record(t, TODAY)["horizon_days"] == plans._CATALYST_GRACE_DAYS


def test_new_plan_and_revise_carry_the_horizon_basis():
    with patch("db.get_position_stop", return_value=None), \
         patch("db.get_active_plan", return_value=None), \
         patch("db.insert_plan", return_value=7), \
         patch("db.last_closed_plan", return_value=None):
        r = plans.observe("NVDA", PARSED, "OPEN", 7, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "NEW_PLAN" and r["horizon_basis"] == "catalyst"
    assert r["catalyst_days_from_today"] == 22 and r["horizon_days"] == 27
    parsed = {**PARSED, "catalyst": {"event": "x", "date": "2026-10-28"}}
    with patch("db.get_position_stop", return_value=STOP), \
         patch("db.get_active_plan", return_value=_plan()), \
         patch("db.insert_plan", return_value=8) as ins:
        r = plans.observe("NVDA", parsed, "ADD", 7, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "REVISE" and r["horizon_basis"] == "bucket_default"
    assert r["catalyst_on_or_before_today"] is True and r["horizon_days"] == 90
    assert ins.call_args.args[0]["horizon_end"] == TODAY + timedelta(days=90)   # unchanged rule
    assert "reseed" not in r


def test_follow_rows_carry_no_horizon_keys():
    with patch("db.get_position_stop", return_value=STOP), \
         patch("db.get_active_plan", return_value=_plan()):
        r = plans.observe("NVDA", PARSED, "HOLD", 6, "Current Price: $540.00", now=NOW)
    assert r["relation"] == "FOLLOW" and "horizon_basis" not in r


def test_horizon_record_never_raises():
    assert plans._horizon_record({"catalyst_date": "2026-10-28"}, TODAY)["horizon_basis"] is None


# --- batch_verify plan_ledger_record and the registry stage ------------------

def _bv():
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "batch_verify.py")
    spec = importlib.util.spec_from_file_location("batch_verify_pl", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_RES = {"status": "ok", "prev_plan_id": 5}


def _row(i, **plan):
    return {"id": i, "plan": plan}


def test_check_ok_tallies_the_flags_and_basis():
    bv = _bv()
    rows = [_row(1, relation="SEEDED", horizon_basis="catalyst", reseeded_after_invalidate=True, reseed=_RES),
            _row(2, relation="NEW_PLAN", horizon_basis="bucket_default", reseeded_after_invalidate=False,
                 reseed={"status": "no_previous_plan"}),
            _row(3, relation="NEW_PLAN", horizon_basis="catalyst", reseeded_after_invalidate=None,
                 reseed={"status": "error:OperationalError"}),
            _row(4, relation="REVISE", horizon_basis="catalyst"),
            _row(5, relation="FOLLOW", stopped_out_since_plan={"kind": "broker_exit"}),
            _row(6, relation="FOLLOW"), {"id": 7, "plan": None}]
    c = bv.check_plan_ledger_record(rows, True)
    assert c["status"] == "FAIL"                     # 1 of 3 reseed reads is an error: over 20%
    assert c["triggered"] == 4 and c["rows"] == 6 and not c["missing"]
    assert dict(c["reads"]["reseeded_after_invalidate"]) == {"true": 1, "false": 1, "none": 1}
    assert dict(c["reads"]["horizon_basis"]) == {"catalyst": 3, "bucket_default": 1}
    assert dict(c["reads"]["reseed.status"]) == {"ok": 1, "no_previous_plan": 1, "error:OperationalError": 1}
    assert any("on 1 plan observations" in n for n in c["notes"])
    ok = bv.check_plan_ledger_record(rows[:2] + rows[3:], True)
    assert ok["status"] == "OK"


def test_check_fails_on_a_missing_key():
    bv = _bv()
    c = bv.check_plan_ledger_record([_row(1, relation="SEEDED", horizon_basis="catalyst"),
                                     _row(2, relation="REVISE")], True)
    assert c["status"] == "FAIL" and c["missing"] == [1, 2]


def test_check_with_the_switch_off_wants_only_the_horizon():
    bv = _bv()
    rows = [_row(1, relation="SEEDED", horizon_basis="catalyst"),
            _row(2, relation="NEW_PLAN", horizon_basis="catalyst", reseeded_after_invalidate=False, reseed=_RES)]
    c = bv.check_plan_ledger_record(rows, False)
    assert c["status"] == "OK" and "reseed.status" not in c["reads"]
    assert any("1 NEW_PLAN/SEEDED rows carry reseed" in n for n in c["notes"])


def test_check_warns_with_no_plan_written_and_counts_a_horizon_error():
    bv = _bv()
    assert bv.check_plan_ledger_record([_row(1, relation="FOLLOW")], True)["status"] == "WARN"
    c = bv.check_plan_ledger_record([_row(1, relation="REVISE", horizon_basis=None,
                                          horizon_record_error="TypeError")], True)
    assert c["status"] == "FAIL" and "error: TypeError" in c["reads"]["horizon_basis"]


def test_check_reads_a_real_observation():
    bv = _bv()
    r, *_ = _seed()
    c = bv.check_plan_ledger_record([{"id": 1, "plan": json.loads(json.dumps(r))}], True)
    assert c["status"] == "OK" and dict(c["reads"]["reseeded_after_invalidate"]) == {"true": 1}


def test_fetch_verdicts_reads_the_plan_and_run_checks_passes_the_switch():
    import inspect
    bv = _bv()
    assert "provenance->'plan'" in inspect.getsource(bv._fetch_verdicts)
    rows = [_row(1, relation="SEEDED", horizon_basis="catalyst")]
    checks, _ = bv.run_checks([], [], [], [], [], [], {}, NOW, NOW, "live", verdict_rows=rows,
                              plan_reseed_on=False)
    assert next(c for c in checks if c["name"] == "plan_ledger_record")["status"] == "OK"
    checks, _ = bv.run_checks([], [], [], [], [], [], {}, NOW, NOW, "live", verdict_rows=rows)
    assert next(c for c in checks if c["name"] == "plan_ledger_record")["status"] == "FAIL"


def test_registry_stage_is_conditional_on_plan_creations():
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_registry_check.py")
    spec = importlib.util.spec_from_file_location("shadow_registry_pl", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    st = next(s for s in m.STAGES if s["stage"] == "plan_ledger_record")
    assert st["switch"] == "PLAN_RESEED_RECORD_ENABLED" and st["expect_rows"] == "conditional"
    assert "IN ('NEW_PLAN', 'SEEDED')" in st["trigger"][1]
    assert st["sources"] == ((st["trigger"][0], "provenance->'plan' ? 'reseed'"),)
    assert hasattr(config, st["switch"])
