"""Per-lot R at an ADD's stop (execution.add_lot_r / _record_add_lot_r):
record only, ratios only; never changes the stop, never raises."""
import inspect
from unittest.mock import MagicMock, patch

import pytest

import config
import execution


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "ADD_LOT_R_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "ADD_LOT_R_THRESHOLD", 1.2)


def _lot(lid, entry, own, src="fill_risk_per_share"):
    return {"lot_id": lid, "approval_id": lid + 1000, "entry_date": "2026-11-05", "entry_price": entry,
            "own_stop": own, "own_stop_source": src}


# ── pure ─────────────────────────────────────────────────────────────────────

def test_nvda_197_case():
    """Own stop 10.7% under entry; the placed stop 20.1% under: about 1.88R."""
    rec = execution.add_lot_r([_lot(197, 100.0, 89.3), _lot(198, 90.0, 79.0)], 79.9, 1.2)
    a, b = rec["lots"]
    assert a["r_at_placed_stop"] == pytest.approx(1.879, abs=1e-3)
    assert a["own_stop_pct"] == pytest.approx(-10.7) and a["placed_stop_pct"] == pytest.approx(-20.1)
    assert b["r_at_placed_stop"] == pytest.approx((90 - 79.9) / 11, abs=1e-3)
    assert rec["max_lot_r"] == a["r_at_placed_stop"] and rec["lots_over_threshold"] == 1
    assert rec["n_lots"] == 2 and rec["n_with_r"] == 2
    assert "entry_price" not in a and "own_stop" not in a      # ratios only


def test_at_own_stop_is_one_and_tighter_is_below_one():
    rec = execution.add_lot_r([_lot(1, 100.0, 90.0)], 90.0)
    assert rec["lots"][0]["r_at_placed_stop"] == 1.0 and rec["lots_over_threshold"] == 0
    assert execution.add_lot_r([_lot(1, 100.0, 90.0)], 95.0)["lots"][0]["r_at_placed_stop"] == 0.5


def test_threshold_is_strictly_above():
    rec = execution.add_lot_r([_lot(1, 100.0, 90.0)], 88.0, 1.2)
    assert rec["lots"][0]["r_at_placed_stop"] == 1.2 and rec["lots_over_threshold"] == 0


@pytest.mark.parametrize("lot", [_lot(1, 100.0, None), _lot(1, 100.0, 100.0), _lot(1, 100.0, 105.0),
                                 _lot(1, None, 90.0), _lot(1, "x", 90.0), _lot(1, 0.0, 90.0),
                                 _lot(1, float("nan"), 90.0)])
def test_unusable_lots_get_no_r(lot):
    rec = execution.add_lot_r([lot], 80.0)
    assert rec["lots"][0]["r_at_placed_stop"] is None
    assert rec["max_lot_r"] is None and rec["n_with_r"] == 0 and rec["lots_over_threshold"] == 0


def test_threshold_compares_the_unrounded_r():
    """R 1.2004 shows as 1.2 but is over a 1.2 threshold."""
    rec = execution.add_lot_r([_lot(1, 100.0, 90.0)], 87.996, 1.2)
    assert rec["lots"][0]["r_at_placed_stop"] == 1.2 and rec["lots_over_threshold"] == 1


@pytest.mark.parametrize("field", ["entry_price", "own_stop"])
def test_bool_is_not_a_price(field):
    lot = _lot(1, 100.0, 90.0)
    lot[field] = True
    assert execution.add_lot_r([lot], 80.0)["lots"][0]["r_at_placed_stop"] is None
    assert execution.add_lot_r([_lot(1, 100.0, 90.0)], True)["n_with_r"] == 0


def test_bad_placed_stop_and_empty():
    assert execution.add_lot_r([_lot(1, 100.0, 90.0)], None)["n_with_r"] == 0
    assert execution.add_lot_r([], 80.0) == {"lots": [], "n_lots": 0, "n_with_r": 0, "max_lot_r": None,
                                             "threshold": 1.2, "lots_over_threshold": 0}


# ── recorder ─────────────────────────────────────────────────────────────────

APPROVAL = {"id": 171, "ticker": "NVDA", "action": "ADD"}


def _record(rows=None, rows_exc=None, log_exc=None, path="whole_position"):
    kw = {"side_effect": rows_exc} if rows_exc else {"return_value": rows or []}
    lk = {"side_effect": log_exc} if log_exc else {"return_value": True}
    with patch("db.get_open_lots_initial_stops", **kw) as read, patch("db.log_activity_queued", **lk) as act:
        execution._record_add_lot_r("NVDA", APPROVAL, 79.9, 90.0, 79.0, path, "placed")
    return read, act


@pytest.mark.switches(ADD_LOT_R_SHADOW_ENABLED=False)
def test_off_reads_and_writes_nothing():
    read, act = _record(rows=[])
    read.assert_not_called()
    act.assert_not_called()


def test_row_has_existing_lots_and_this_fill(on):
    rows = [{"lot_id": 197, "approval_id": 169, "action": "OPEN", "entry_price": 100.0,
             "entry_date": "2026-11-05T16:08:40+00:00", "fill_risk_per_share": 10.7, "stop_loss_price": 89.0},
            {"lot_id": 150, "approval_id": 140, "action": "OPEN", "entry_price": 100.0,
             "entry_date": "2026-10-01", "fill_risk_per_share": None, "stop_loss_price": 92.0},
            # this fill's own row already written: skipped (it is added from the fill)
            {"lot_id": 198, "approval_id": 171, "action": "ADD", "entry_price": 90.0,
             "entry_date": "2026-11-10", "fill_risk_per_share": 11.0, "stop_loss_price": 79.0}]
    read, act = _record(rows=rows)
    read.assert_called_once_with("NVDA", strict=True)
    stage, ticker, message, sentiment, rec = act.call_args.args
    assert (stage, ticker, sentiment) == ("add_lot_r", "NVDA", "neutral")
    assert rec["path"] == "whole_position" and rec["read"] == "ok" and rec["approval_id"] == 171
    assert rec["stop_outcome"] == "placed" and rec["action"] == "ADD"
    by = {l["approval_id"]: l for l in rec["lots"]}
    assert set(by) == {169, 140, 171}
    assert by[169]["own_stop_source"] == "fill_risk_per_share"
    assert by[169]["r_at_placed_stop"] == pytest.approx((100 - 79.9) / 10.7, abs=1e-3)
    assert by[140]["own_stop_source"] == "approval_stop"
    assert by[140]["r_at_placed_stop"] == pytest.approx((100 - 79.9) / 8.0, abs=1e-3)
    assert by[171]["own_stop_source"] == "this_fill" and by[171]["lot_id"] is None
    assert rec["lots_over_threshold"] == 2 and "record only" in message


def test_read_error_still_records_this_fill(on):
    _, act = _record(rows_exc=RuntimeError("pool"))
    rec = act.call_args.args[4]
    assert rec["read"] == "error: RuntimeError"
    assert [l["own_stop_source"] for l in rec["lots"]] == ["this_fill"]


def test_never_raises(on):
    _record(rows=[], log_exc=RuntimeError("db down"))           # no exception
    with patch("execution.add_lot_r", side_effect=ValueError("boom")), \
         patch("db.get_open_lots_initial_stops", return_value=[]), patch("db.log_activity_queued") as act:
        execution._record_add_lot_r("NVDA", APPROVAL, 79.9, 90.0, 79.0, "stale_record", "placed")
    act.assert_not_called()


# ── db read ──────────────────────────────────────────────────────────────────

def test_db_read_shape_and_strict():
    import db
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [(197, 169, "OPEN", 100.0, None, 10.7, 89.3)]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.get_open_lots_initial_stops("NVDA") == [
            {"lot_id": 197, "approval_id": 169, "action": "OPEN", "entry_price": 100.0,
             "entry_date": None, "fill_risk_per_share": 10.7, "stop_loss_price": 89.3}]
    sql = cur.execute.call_args.args[0]
    assert "exit_date IS NULL" in sql and "orphaned_at IS NULL" in sql and "trading_env" in sql
    cur.execute.side_effect = RuntimeError("q")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.get_open_lots_initial_stops("NVDA") == []
        with pytest.raises(RuntimeError):
            db.get_open_lots_initial_stops("NVDA", strict=True)


# ── stale-record flag in _whole_position_stop ────────────────────────────────

def _stale_branch(action):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=1, avg_entry_price=100.0)
    status = {}
    with patch("db.get_position_stop", return_value={"stop_price": 98.0}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(1.0, 100.0)), \
         patch("db.log_activity_direct"), patch("db.log_activity_queued"), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("tasks._resting_sell_stops", return_value=[{"ticker": "NVDA"}]), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        out = execution._whole_position_stop("NVDA", 95.0, 1, status=status, action=action)
    cancel.assert_not_called()
    return out, status


def test_stale_branch_sets_the_flag():
    # An OPEN over an existing record: the record is rewritten at the new stop.
    out, status = _stale_branch("OPEN")
    assert out == (95.0, None) and status.get("stale_record") is True


def test_an_add_on_the_stale_branch_keeps_the_record_and_sets_no_flag():
    # Nothing held before the order (None), the old stop still working: the
    # kept-record route, no lot-R row.
    out, status = _stale_branch("ADD")
    assert out == (95.0, None) and "stale_record" not in status
    assert status["prior_unreadable"] is True


# ── execute_approved: the recorder runs on a re-stop, and the stop is unchanged ─

def _execute_add(whole, enabled, monkeypatch, record_patch=None, place_kw=None, send=None):
    """execute_approved on a filled 1-share ADD with _whole_position_stop
    patched (whole: its return value, or a side_effect callable). Returns
    (manage_stop_place mock, _record_add_lot_r mock or None, db mocks)."""
    import db
    monkeypatch.setattr(config, "ADD_LOT_R_SHADOW_ENABLED", enabled)
    approval = {"id": 171, "ticker": "NVDA", "action": "ADD", "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
                "risk_json": {"shares_initial": 1, "price_at_sizing": 100.0, "stop_loss_price": 90.0}}
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["get_open_lots_initial_stops"] = MagicMock(return_value=[
        {"lot_id": 197, "approval_id": 169, "action": "OPEN", "entry_price": 110.0,
         "entry_date": "2026-11-05", "fill_risk_per_share": 8.0, "stop_loss_price": 102.0}])
    wkw = {"side_effect": whole} if callable(whole) else {"return_value": whole}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 1, "fill_price": 100.0,
                                                        "order_id": 1, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value={"price": 100.0, "is_stale": False,
                                                             "age_minutes": 1.0, "market_open": True}), \
         patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("execution._whole_position_stop", **wkw), \
         patch("tasks.manage_stop_place", **(place_kw or {"return_value": {"order_id": 9}})) as place, \
         patch("alerts._send", side_effect=send), patch("alerts.send_trade_alert"):
        client.return_value.get_open_position.side_effect = Exception("No open position")
        client.return_value.get_all_positions.return_value = []
        if record_patch is not None:
            with patch("execution._record_add_lot_r", **record_patch) as rec:
                execution.execute_approved(approval)
            return place, rec, db_fns
        execution.execute_approved(approval)
    return place, None, db_fns


def _rows(db_fns):
    return [c for c in db_fns["log_activity_queued"].call_args_list if c.args and c.args[0] == "add_lot_r"]


def test_whole_position_restop_is_recorded_and_stop_unchanged(monkeypatch):
    on_place, _, on_db = _execute_add((98.0, (105.0, 2.0)), True, monkeypatch)
    off_place, _, off_db = _execute_add((98.0, (105.0, 2.0)), False, monkeypatch)
    assert on_place.called and on_place.call_args_list == off_place.call_args_list
    rows = _rows(on_db)
    assert len(rows) == 1
    rec = rows[0].args[4]
    assert rec["path"] == "whole_position" and rec["stop_outcome"] == "placed"
    lot = next(l for l in rec["lots"] if l["approval_id"] == 169)
    assert lot["r_at_placed_stop"] == pytest.approx((110 - 98) / 8.0, abs=1e-3)
    mine = next(l for l in rec["lots"] if l["own_stop_source"] == "this_fill")
    assert mine["approval_id"] == 171 and mine["r_at_placed_stop"] is not None
    assert _rows(off_db) == []
    off_db["get_open_lots_initial_stops"].assert_not_called()


def test_stale_record_path_is_recorded(monkeypatch):
    def stale(ticker, stop, qty, status=None, **kw):
        status["stale_record"] = True
        return stop, None
    place, rec, _ = _execute_add(stale, True, monkeypatch, record_patch={"return_value": None})
    rec.assert_called_once()
    assert rec.call_args.args[5] == "stale_record"
    assert rec.call_args.args[2] == rec.call_args.args[4]      # placed at the new shares' own stop
    assert place.called


def test_fresh_entry_records_nothing(monkeypatch):
    place, rec, _ = _execute_add((95.0, None), True, monkeypatch, record_patch={"return_value": None})
    rec.assert_not_called()
    assert place.called


def test_recorder_failure_inside_does_not_change_the_stop(monkeypatch):
    ok_place, _, _ = _execute_add((98.0, (105.0, 2.0)), False, monkeypatch)
    with patch("execution.add_lot_r", side_effect=ValueError("boom")):
        bad_place, _, bad_db = _execute_add((98.0, (105.0, 2.0)), True, monkeypatch)
    assert bad_place.call_args_list == ok_place.call_args_list
    assert _rows(bad_db) == []


def test_failed_stop_reprotects_and_alerts_before_the_row(monkeypatch):
    """manage_stop_place fails 3 times with the shadow on: the reprotect and
    the UNPROTECTED alert still fire, and the add_lot_r row is written after
    both."""
    order = []
    real_record = execution._record_add_lot_r

    def record(*a, **k):
        order.append("add_lot_r")
        return real_record(*a, **k)

    def send(msg, *a, **k):
        if "UNPROTECTED" in msg:
            order.append("alert")
    with patch("time.sleep"), \
         patch("tasks._queue_reprotect", side_effect=lambda *a, **k: order.append("reprotect")) as rp, \
         patch("tasks.open_manual_action"), \
         patch("execution._record_add_lot_r", side_effect=record):
        place, _, db_fns = _execute_add((98.0, (105.0, 2.0)), True, monkeypatch,
                                        place_kw={"side_effect": RuntimeError("broker")}, send=send)
    assert place.call_count == 3
    rp.assert_called_once()
    entry = rp.call_args.args[1]       # the new shares' own stop rides on the queued retry
    assert entry["intent"] == "BUY" and entry["own_stop"] == 90.0   # not the merged 98
    assert order == ["reprotect", "alert", "add_lot_r"]
    rows = _rows(db_fns)
    assert len(rows) == 1 and rows[0].args[4]["stop_outcome"] == "failed"


def test_guard_kept_record_level_is_what_the_record_gets(monkeypatch):
    """_whole_position_stop sets status["record_stop"] when the guard lowered
    the stop on a non-fill price: the broker stop goes at the lowered level,
    the record keeps the higher one."""
    seen = {}

    def whole(ticker, stop, qty, status=None, **kw):
        seen.update(kw)
        status["record_stop"] = 99.0
        return 98.0, (105.0, 2.0)
    place, _, db_fns = _execute_add(whole, False, monkeypatch)
    assert seen["price_is_fill"] is True                    # the harness fills at a price
    assert place.call_args.args[2] == 98.0
    assert db_fns["upsert_position_stop"].call_args.kwargs["stop_price"] == 99.0
