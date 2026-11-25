"""An ADD's whole-position stop is sized from this process's ib.positions()
count (execution._add_stop_qty, ibkr.cached_stock_qty). 11-13 NVDA
14:28: cache 1, fresh 4 (a duplicated row), broker 2 within 5 s; the stop for
4 was refused (SellExceedsPosition) and NVDA had no stop for about 2 min.
Since 7ba0857 the fresh read is deduplicated, so the count is a safety net
against a duplicated fresh read; the 11-13 tests below simulate that
pre-dedupe doubled read with fresh=(4, ...)."""
import inspect
from unittest.mock import MagicMock, call, patch

import pytest

import config
import execution


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", True)


def _wps(broker, cache=1, fresh=(4.0, 190.0), held_before=1, filled=1, action="ADD"):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=cache, avg_entry_price=190.0)
    bkw = {"side_effect": broker} if isinstance(broker, Exception) else {"return_value": broker}
    with patch("db.get_position_stop", return_value={"stop_price": 146.24}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=fresh), \
         patch("ibkr.cached_stock_qty", **bkw) as bread, \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("db.log_activity_queued") as queued, patch("db.log_activity_direct"), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("alerts._send") as send:
        out = execution._whole_position_stop("NVDA", 137.15, filled, status={}, market_price=162.63,
                                             action=action, held_before=held_before)
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "add_stop_qty"]
    alerts = [c.args[0] for c in send.call_args_list]
    return out, rows, alerts, bread, cancel


def test_1113_sequence_sizes_the_stop_from_the_broker(on):
    """The pre-dedupe doubled fresh read (simulated): the count, below the
    reads, is used, and that is alerted, never a silent undersize."""
    (stop, (avg, qty)), rows, alerts, _, cancel = _wps(2.0)
    assert stop == 146.24 and qty == 2.0
    cancel.assert_called_once_with("NVDA", stuck_out=[])
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "broker" and r["broker_read"] == "ok"
    assert (r["cache_qty"], r["fresh_qty"], r["broker_qty"]) == (1.0, 4.0, 2.0)
    assert r["read_qty"] == 4.0 and r["used_qty"] == 2.0 and r["expected_qty"] == 2.0
    assert r["undersize_flags"] == ["below the position reads (4)"]
    assert len(alerts) == 1 and "MAY NOT COVER" in alerts[0]


@pytest.mark.switches(ADD_STOP_QTY_FROM_BROKER=False)
def test_switch_off_sizes_from_the_reads_and_reads_no_broker_count():
    (_, (_, qty)), rows, alerts, bread, _ = _wps(2.0)
    assert qty == 4.0                              # today's sizing
    bread.assert_not_called()
    assert rows == [] and alerts == []


def test_unreadable_broker_count_keeps_the_reads_and_alerts(on):
    (_, (_, qty)), rows, alerts, _, _ = _wps(TimeoutError("ib"))
    assert qty == 4.0
    assert rows[0]["source"] == "position_reads" and rows[0]["broker_read"] == "error: TimeoutError"
    assert len(alerts) == 1 and "ADD STOP SIZE UNCONFIRMED" in alerts[0]


def test_zero_broker_count_keeps_the_reads_and_alerts(on):
    (_, (_, qty)), rows, alerts, _, _ = _wps(0.0)
    assert qty == 4.0 and rows[0]["broker_read"] == "zero"
    assert len(alerts) == 1 and "UNCONFIRMED" in alerts[0]


def test_broker_below_the_expected_count_is_used_and_alerted(on):
    # held 3 before, 1 filled, reads agree on 4, the broker still shows 3
    (_, (_, qty)), rows, alerts, _, _ = _wps(3.0, cache=4, fresh=(4.0, 190.0), held_before=3)
    assert qty == 3.0 and rows[0]["expected_qty"] == 4.0
    assert len(alerts) == 1 and "MAY NOT COVER" in alerts[0]


def _helper(broker, read_qty=4.0, filled=1.0, held_before=None):
    bkw = {"side_effect": broker} if isinstance(broker, Exception) else {"return_value": broker}
    with patch("ibkr.cached_stock_qty", **bkw), \
         patch("db.log_activity_queued") as queued, patch("alerts._send") as send:
        used = execution._add_stop_qty("NVDA", read_qty, filled, 1.0, read_qty, held_before)
    return used, queued.call_args.args, [c.args[0] for c in send.call_args_list]


def test_undersize_is_alerted_without_a_held_before_count(on):
    used, row, alerts = _helper(2.0, read_qty=4.0, held_before=None)
    assert used == 2.0 and row[3] == "negative"
    assert len(alerts) == 1 and "below the position reads (4)" in alerts[0]


def test_count_not_above_the_fill_is_alerted(on):
    used, row, alerts = _helper(1.0, read_qty=1.0, filled=1.0, held_before=None)
    assert used == 1.0 and row[3] == "negative"
    assert row[4]["undersize_flags"] == ["not above this fill (1)"]
    assert len(alerts) == 1 and "MAY NOT COVER" in alerts[0]


def test_count_above_the_reads_is_used_quietly(on):
    used, row, alerts = _helper(5.0, read_qty=4.0, held_before=None)
    assert used == 5.0 and row[3] == "neutral" and row[4]["undersize_flags"] == []
    assert alerts == []


@pytest.mark.parametrize("broker,read", [(-2.0, "short"), (float("nan"), "not_finite"),
                                         (float("inf"), "not_finite")])
def test_short_or_non_finite_count_keeps_the_reads_and_alerts(on, broker, read):
    used, row, alerts = _helper(broker, read_qty=4.0)
    assert used == 4.0 and row[4]["broker_read"] == read and row[3] == "negative"
    assert len(alerts) == 1 and "UNCONFIRMED" in alerts[0]


@pytest.mark.switches(ADD_STOP_QTY_FROM_BROKER=False)
def test_switch_off_reads_and_writes_nothing():
    with patch("ibkr.cached_stock_qty") as bread, patch("db.log_activity_queued") as queued:
        assert execution._add_stop_qty("NVDA", 4.0, 1.0, 1.0, 4.0, None) == 4.0
    bread.assert_not_called()
    queued.assert_not_called()


def test_reclassified_open_on_a_held_ticker_sizes_from_the_count(on):
    """An OPEN on a ticker held before the order is handled as an ADD: its
    whole-position stop is sized from the count too."""
    (_, (_, qty)), rows, alerts, bread, _ = _wps(2.0, cache=1, fresh=(2.0, 190.0), held_before=1,
                                                 action="OPEN")
    assert qty == 2.0 and rows[0]["source"] == "broker" and alerts == []
    bread.assert_called_once_with("NVDA")


def test_agreeing_counts_change_nothing(on):
    (stop, whole), rows, alerts, _, _ = _wps(2.0, cache=2, fresh=(2.0, 190.0))
    assert whole == (190.0, 2.0) and stop == 146.24
    assert rows[0]["source"] == "broker" and alerts == []


def test_ambiguous_and_stale_paths_read_no_broker_count(on):
    # fresh below cache: ambiguous; reads at the fill: stale
    out, rows, _, bread, _ = _wps(2.0, cache=3, fresh=(2.0, 190.0), held_before=None)
    assert out == (137.15, None)
    out2, rows2, _, bread2, _ = _wps(1.0, cache=1, fresh=(1.0, 190.0), held_before=None)
    assert out2 == (137.15, None)
    bread.assert_not_called()
    bread2.assert_not_called()
    assert rows == [] and rows2 == []


def test_helper_never_raises(on):
    with patch("ibkr.cached_stock_qty", return_value=2.0), \
         patch("db.log_activity_queued", side_effect=RuntimeError("q")), \
         patch("alerts._send", side_effect=RuntimeError("tg")):
        assert execution._add_stop_qty("NVDA", 4.0, 1, 1.0, 4.0, 1.0) == 2.0
    with patch("ibkr.cached_stock_qty", side_effect=RuntimeError("ib")), \
         patch("db.log_activity_queued"), patch("alerts._send", side_effect=RuntimeError("tg")):
        assert execution._add_stop_qty("NVDA", 4.0, 1, 1.0, 4.0, 1.0) == 4.0


# ── ibkr read ────────────────────────────────────────────────────────────────

def _pos(symbol, qty, sec="STK"):
    return MagicMock(contract=MagicMock(symbol=symbol, secType=sec), position=qty)


def test_broker_count_reads_stock_positions_only():
    import ibkr
    ib = MagicMock()
    ib.positions.return_value = [_pos("NVDA", 2.0), _pos("NVDA", -1.0, "OPT"), _pos("AMD", 5.0)]
    with patch("ibkr.get_ib", return_value=ib):
        assert ibkr.get_broker_stock_qty("NVDA") == 2.0
        assert ibkr.get_broker_stock_qty("KLAC") == 0.0
    with patch("ibkr.get_ib", side_effect=RuntimeError("down")):
        with pytest.raises(RuntimeError):
            ibkr.get_broker_stock_qty("NVDA")


def test_add_count_is_the_cached_read_and_never_reconnects(monkeypatch):
    """Review of xb-windows: the ADD's count is cached_stock_qty (one
    ib.sleep(0), no request, no get_ib(): the new shares have no stop in
    that window). Not connected keeps the position reads with today's alert."""
    import config
    import execution
    import ibkr
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", True)
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.positions.return_value = [_pos("NVDA", 2.0)]
    with patch.object(ibkr, "_ib", ib), patch("ibkr.get_ib", side_effect=AssertionError("get_ib")), \
         patch("db.log_activity_queued"), patch("alerts._send") as send:
        assert execution._add_stop_qty("NVDA", 2.0, 1.0, 2.0, 2.0, 1.0) == 2.0
    ib.sleep.assert_called_once_with(0)
    send.assert_not_called()
    with patch.object(ibkr, "_ib", None), patch("ibkr.get_ib", side_effect=AssertionError("get_ib")), \
         patch("db.log_activity_queued") as queued, patch("alerts._send") as send:
        assert execution._add_stop_qty("NVDA", 2.0, 1.0, 2.0, 2.0, 1.0) == 2.0   # read_qty kept
    assert queued.call_args.args[4]["broker_read"] == "error: ConnectionError"
    assert queued.call_args.args[4]["source"] == "position_reads"
    assert "ADD STOP SIZE UNCONFIRMED" in send.call_args.args[0]


# ── execute_approved: the 11-13 ADD end to end ───────────────────────────────

def _execute_1113(monkeypatch, enabled):
    import db
    monkeypatch.setattr(config, "ADD_STOP_QTY_FROM_BROKER", enabled)
    approval = {"id": 173, "ticker": "NVDA", "action": "ADD", "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": 137.15,
                "risk_json": {"shares_initial": 1, "price_at_sizing": 162.63,
                              "stop_loss_price": 137.15}}
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["get_position_stop"] = MagicMock(return_value={"stop_price": 146.24, "entry_price": 190.0})
    db_fns["upsert_position_stop"] = MagicMock(return_value=True)
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 1, "fill_price": 162.63,
                                                        "order_id": 141, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=162.63), \
         patch("ibkr.get_fresh_stock_position", return_value=(4.0, 193.7)), \
         patch("ibkr.cached_stock_qty", return_value=2.0), \
         patch("market_data.get_priced_quote", return_value={"price": 162.63, "is_stale": False,
                                                             "age_minutes": 1.0, "market_open": True}), \
         patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks._resting_sell_stops", return_value=[{"ticker": "NVDA", "qty": 1.0,
                                                           "stop_price": 146.24, "status": "Submitted"}]), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 50}) as place, \
         patch("alerts._send"), patch("alerts.send_trade_alert"):
        client.return_value.get_open_position.return_value = MagicMock(qty=1, avg_entry_price=193.2)
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=1)]
        execution.execute_approved(approval)
    return place, cancel, db_fns


def test_1113_add_places_one_stop_for_the_broker_count(monkeypatch):
    place, cancel, db_fns = _execute_1113(monkeypatch, True)
    cancel.assert_called_once_with("NVDA", stuck_out=[])
    assert place.call_args_list == [call("NVDA", 2.0, 146.24, market_price=162.63)]
    rows = [c.args[4] for c in db_fns["log_activity_queued"].call_args_list
            if c.args and c.args[0] == "add_stop_qty"]
    assert len(rows) == 1 and rows[0]["read_qty"] == 4.0 and rows[0]["used_qty"] == 2.0


def test_1113_add_with_the_switch_off_sizes_for_the_reads(monkeypatch):
    place, _, _ = _execute_1113(monkeypatch, False)
    assert place.call_args_list == [call("NVDA", 4.0, 146.24, market_price=162.63)]   # today
