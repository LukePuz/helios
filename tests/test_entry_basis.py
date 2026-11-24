"""Record-only entry_basis row (execution.entry_basis / _record_entry_basis):
on an ADD that writes the broker's average cost as the stop record's entry,
the fill-weighted average of the bot's own fills over the open lots and
where the trail arm, profit target and breakeven sit measured from it.
Nothing written to position_stops changes; it never raises."""
import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
import execution


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "ENTRY_BASIS_RECORD_ENABLED", True)
    monkeypatch.setattr(config, "SETUP_GATES_ENABLED", False)


def test_switch_ships_on():
    assert config.ENTRY_BASIS_RECORD_ENABLED is True


# ── pure ─────────────────────────────────────────────────────────────────────

def test_fill_weighted_average_and_gap():
    lots = [{"price": 100.0, "qty": 10.0}, {"price": 120.0, "qty": 10.0}]
    r = execution.entry_basis(lots, [], 147.4, 20.0, "Compounder")
    assert r["fill_avg"] == pytest.approx(110.0) and r["basis_gap_frac"] == pytest.approx(0.34)
    assert r["qty_coverage"] == 1.0 and r["lots_without_fill"] == 0 and r["n_lots"] == 2
    # Compounder: the +30% arm sits at 1.30 x 1.34 - 1 over the fills; no profit target.
    assert r["trail_arm"] == 0.30 and r["trail_arm_vs_fill"] == pytest.approx(0.742)
    assert r["pt_by_style"] == {} and r["pt_vs_fill_by_style"] == {}
    assert r["breakeven_vs_fill"] == pytest.approx(0.34) and r["legacy_tables_apply"] is True


def test_profit_target_shifts_by_style():
    r = execution.entry_basis([{"price": 100.0, "qty": 4.0}], [], 110.0, 4.0, "Accelerator")
    assert r["pt_by_style"] == {"Trending/Expanding": 0.25, "Choppy/Distribution": 0.125}
    assert r["pt_vs_fill_by_style"]["Trending/Expanding"] == pytest.approx(1.25 * 1.10 - 1)
    assert r["pt_vs_fill_by_style"]["Choppy/Distribution"] == pytest.approx(1.125 * 1.10 - 1)
    assert r["trail_arm_vs_fill"] == pytest.approx(1.20 * 1.10 - 1)
    m = execution.entry_basis([{"price": 10.0, "qty": 4.0}], [], 10.0, 4.0, "Moonshot")
    assert m["trail_arm"] is None and m["trail_arm_vs_fill"] is None
    assert m["pt_vs_fill_by_style"]["Choppy/Distribution"] == pytest.approx(0.25)
    assert m["basis_gap_frac"] == 0.0


def test_partial_exit_is_taken_from_the_oldest_lot():
    lots = [{"price": 100.0, "qty": 10.0}, {"price": 120.0, "qty": 10.0}]
    r = execution.entry_basis(lots, [{"qty": 5.0}], 115.0, 15.0, "Compounder")
    assert r["fill_avg"] == pytest.approx((5 * 100 + 10 * 120) / 15)
    assert r["sold_since_oldest_lot"] == 5.0 and r["unmatched_sold"] == 0
    assert r["qty_coverage"] == 1.0
    assert lots[0]["qty"] == 10.0                                  # inputs not changed


def test_unknown_lots_and_oversold():
    r = execution.entry_basis([{"price": 100.0, "qty": None}, {"price": 90.0, "qty": 2.0}],
                              [], 95.0, 5.0, "Compounder")
    assert r["lots_without_fill"] == 1 and r["fill_avg"] == pytest.approx(90.0)
    assert r["qty_coverage"] == pytest.approx(0.4)
    r = execution.entry_basis([{"price": 90.0, "qty": 2.0}], [{"qty": 3.0}], 95.0, 1.0, "Compounder")
    assert r["fill_avg"] is None and r["basis_gap_frac"] is None and r["unmatched_sold"] == 1.0
    assert r["qty_coverage"] == 0.0                     # broker qty known, no fill qty left
    assert r["trail_arm_vs_fill"] is None and r["breakeven_vs_fill"] is None
    assert execution.entry_basis([], [], None, None, None)["fill_avg"] is None


# ── the recorder ─────────────────────────────────────────────────────────────

APPROVAL = {"id": 171, "action": "ADD", "thesis_bucket": "Compounder"}
T0 = datetime(2026, 11, 5, 16, 8, tzinfo=timezone.utc)


def _got():
    return {"lots": [{"approval_id": 169, "entry_date": T0, "outcome_entry": 101.0,
                      "fills": [(100.0, 6.0), (102.0, 4.0)]},
                     {"approval_id": 171, "entry_date": T0, "outcome_entry": 120.0,
                      "fills": [(120.0, 10.0)]}],          # this fill's own row, already written
            "sells": [{"ts": T0, "qty": 2.0}]}


def test_row(on):
    with patch("db.get_open_lot_fills", return_value=_got()) as read, \
         patch("db.log_activity_queued") as write:
        execution._record_entry_basis("NVDA", APPROVAL, 140.0, 18.0, 120.0, 10.0, True, "placed",
                                      fill_source="fill")
    read.assert_called_once_with("NVDA", strict=True)
    stage, ticker, msg, sentiment, rec = write.call_args.args
    assert stage == "entry_basis" and ticker == "NVDA" and "$" not in msg
    # lot 169 at its two fills' average (100.8), FIFO-reduced by 2; this fill once.
    assert rec["n_lots"] == 2 and rec["fill_avg"] == pytest.approx((8 * 100.8 + 10 * 120) / 18)
    assert rec["basis_gap_frac"] == pytest.approx(round(140 / ((8 * 100.8 + 1200) / 18) - 1, 4))
    assert rec["read"] == "ok" and rec["record_written"] is True and rec["approval_id"] == 171
    assert rec["this_fill_source"] == "fill" and rec["stop_outcome"] == "placed"
    assert rec["truncated"] is False and msg.startswith("ADD record entry is the broker average")
    assert rec["env"] == config.TRADING_ENV


def test_read_error_still_records_this_fill(on):
    with patch("db.get_open_lot_fills", side_effect=RuntimeError("pool")), \
         patch("db.log_activity_queued") as write:
        execution._record_entry_basis("NVDA", APPROVAL, 110.0, 1.0, 100.0, 1.0, False, "queued")
    rec = write.call_args.args[4]
    assert rec["read"] == "error: RuntimeError" and rec["n_lots"] == 1
    assert rec["basis_gap_frac"] == pytest.approx(0.10) and rec["record_written"] is False
    assert write.call_args.args[2].startswith("ADD record write failed")


def test_off_reads_and_writes_nothing(monkeypatch):
    monkeypatch.setattr(config, "ENTRY_BASIS_RECORD_ENABLED", False)
    with patch("db.get_open_lot_fills") as read, patch("db.log_activity_queued") as write:
        execution._record_entry_basis("NVDA", APPROVAL, 110.0, 1.0, 100.0, 1.0, True, "placed")
    read.assert_not_called()
    write.assert_not_called()


def test_never_raises(on):
    with patch("db.get_open_lot_fills", return_value=_got()), \
         patch("db.log_activity_queued", side_effect=OSError("queue")):
        execution._record_entry_basis("NVDA", APPROVAL, 140.0, 18.0, 120.0, 10.0, True, "placed")
    with patch("execution.entry_basis", side_effect=ValueError("x")), \
         patch("db.get_open_lot_fills", return_value=_got()), patch("db.log_activity_queued") as w:
        execution._record_entry_basis("NVDA", APPROVAL, 140.0, 18.0, 120.0, 10.0, True, "placed")
    w.assert_not_called()


def test_ladder_bucket_is_flagged(on, monkeypatch):
    monkeypatch.setattr(config, "SETUP_GATES_ENABLED", True)
    monkeypatch.setattr(config, "CATEGORY_RISK_PCT_PER_TRADE", {"Breakout": 0.01})
    with patch("db.get_open_lot_fills", return_value={"lots": [], "sells": []}), \
         patch("db.log_activity_queued") as write:
        execution._record_entry_basis("NVDA", dict(APPROVAL, thesis_bucket="Breakout"),
                                      110.0, 1.0, 100.0, 1.0, True, "placed")
    assert write.call_args.args[4]["legacy_tables_apply"] is False


def test_db_read_is_one_bounded_query():
    import db
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("lot", T0, 169, 101.0, 100.0, 6.0), ("lot", T0, 169, 101.0, 102.0, 4.0),
        ("lot", T0, 170, 99.0, None, None), ("sell", T0, None, None, None, -2.0)]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        got = db.get_open_lot_fills("NVDA")
    assert got["lots"][0]["fills"] == [(100.0, 6.0), (102.0, 4.0)]
    assert got["lots"][1] == {"approval_id": 170, "entry_date": T0, "outcome_entry": 99.0,
                              "fills": []}
    assert got["sells"] == [{"ts": T0, "qty": 2.0}] and got["truncated"] is False
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert len([s for s in sqls if s.lstrip().startswith(("WITH", "SELECT"))]) == 1
    assert "statement_timeout" in sqls[0] and "LIMIT 500" in sqls[1]
    assert "exit_date IS NULL" in sqls[1] and "orphaned_at IS NULL" in sqls[1]
    assert "side='SELL'" in sqls[1] and "trading_env" in sqls[1] and "o.ticker = %s" in sqls[1]
    assert cur.execute.call_args_list[1].args[1] == ("NVDA", config.TRADING_ENV) * 3
    for word in ("INSERT", "UPDATE", "DELETE"):
        assert word not in sqls[1]
    conn.rollback.assert_called()
    put.assert_called_once()
    cur.execute.side_effect = RuntimeError("q")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.get_open_lot_fills("NVDA") == {"lots": [], "sells": [], "truncated": False}
        with pytest.raises(RuntimeError):
            db.get_open_lot_fills("NVDA", strict=True)


# ── execute_approved: after the stop, only on an ADD, the stop unchanged ─────

def _execute(whole, enabled, monkeypatch, place_kw=None):
    import db
    monkeypatch.setattr(config, "ENTRY_BASIS_RECORD_ENABLED", enabled)
    monkeypatch.setattr(config, "SETUP_GATES_ENABLED", False)
    approval = {"id": 171, "ticker": "NVDA", "action": "ADD", "conviction": 7,
                "thesis_bucket": "Compounder", "stop_loss_price": 90.0,
                "risk_json": {"shares_initial": 1, "price_at_sizing": 100.0, "stop_loss_price": 90.0}}
    order = []
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["get_open_lot_fills"] = MagicMock(side_effect=lambda *a, **k: order.append("read") or {
        "lots": [{"approval_id": 169, "entry_date": T0, "outcome_entry": 110.0,
                  "fills": [(110.0, 1.0)]}], "sells": []})
    db_fns["get_open_lots_initial_stops"] = MagicMock(return_value=[])
    db_fns["upsert_position_stop"] = MagicMock(side_effect=lambda **k: order.append(("record", k))
                                               or True)
    wkw = {"side_effect": whole} if callable(whole) else {"return_value": whole}
    pkw = place_kw or {"side_effect": lambda *a, **k: order.append("place") or {"order_id": 9}}
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
         patch("tasks.manage_stop_place", **pkw) as place, \
         patch("alerts._send"), patch("alerts.send_trade_alert"), patch("time.sleep"):
        client.return_value.get_open_position.side_effect = Exception("No open position")
        client.return_value.get_all_positions.return_value = []
        execution.execute_approved(approval)
    rows = [c.args[4] for c in db_fns["log_activity_queued"].call_args_list
            if c.args and c.args[0] == "entry_basis"]
    return place, rows, order


def test_add_writes_the_row_after_the_stop_and_the_record_is_unchanged(monkeypatch):
    on_place, rows, order = _execute((98.0, (105.0, 2.0)), True, monkeypatch)
    off_place, off_rows, off_order = _execute((98.0, (105.0, 2.0)), False, monkeypatch)
    assert on_place.call_args_list == off_place.call_args_list
    records = [o for o in order if isinstance(o, tuple)]
    assert records == [o for o in off_order if isinstance(o, tuple)]     # same record write
    assert records[0][1]["entry_price"] == 105.0 and records[0][1]["entry_price_source"] == "broker_avg"
    assert order.index("read") > order.index("place")                    # read after the stop
    assert len(rows) == 1 and off_rows == [] and "read" not in off_order
    rec = rows[0]
    assert rec["broker_avg"] == 105.0 and rec["fill_avg"] == pytest.approx(105.0)
    assert rec["basis_gap_frac"] == 0.0 and rec["this_fill_source"] == "fill"


def test_fresh_entry_writes_no_row(monkeypatch):
    _, rows, order = _execute((95.0, None), True, monkeypatch)
    assert rows == [] and "read" not in order


def test_a_failed_stop_still_records_after_the_retries(monkeypatch):
    place, rows, order = _execute((98.0, (105.0, 2.0)), True, monkeypatch,
                                  place_kw={"side_effect": RuntimeError("down")})
    assert place.call_count == 3 and len(rows) == 1 and rows[0]["stop_outcome"] == "failed"



def test_a_full_read_is_flagged_truncated(on):
    import db
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("sell", T0, None, None, None, 1.0)] * db._OPEN_LOT_FILLS_LIMIT
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.get_open_lot_fills("NVDA")["truncated"] is True
    with patch("db.get_open_lot_fills", return_value={"lots": [], "sells": [], "truncated": True}), \
         patch("db.log_activity_queued") as write:
        execution._record_entry_basis("NVDA", APPROVAL, 110.0, 1.0, 100.0, 1.0, True, "placed")
    assert write.call_args.args[4]["truncated"] is True
