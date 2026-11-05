"""Resting stops for more shares than held get rebuilt (professor M54)."""
from unittest.mock import MagicMock, call, patch

import pytest

import tasks


def _orders(*qtys):
    return [{"order_id": i, "ticker": "AMD", "action": "SELL", "order_type": "STP",
             "qty": q, "stop_price": 150.0, "status": "Submitted"} for i, q in enumerate(qtys)]


def test_fix_overstop_rebuilds_one_stop_for_the_held_qty():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), []]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    cancel.assert_called_once_with("AMD")
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)
    assert "placed a stop for 10" in msg


def test_fix_overstop_does_nothing_when_cleared_on_recheck():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10)), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    assert "fine on re-check" in msg


def test_fix_overstop_without_recorded_stop_only_alerts():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10, 5)), \
         patch("db.get_position_stop", return_value=None), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    assert "fix the stops in TWS" in msg


def test_stuck_stop_is_reported_not_papered_over():
    """A2253-6: a stop that won't cancel must not get a full-size stop added."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), _orders(10, 10)]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place") as place:
        msg = tasks._fix_overstop("AMD")
    place.assert_not_called()
    assert "won't cancel" in msg


def test_partly_covered_position_gets_the_missing_shares_stopped():
    """A2253-4: a stop for 4 of 10 shares used to pass forever."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(4)), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    cancel.assert_not_called()
    place.assert_called_once_with("AMD", 6.0, 150.0, market_price=200.0)
    assert "only 4 of 10" in msg


def test_naked_position_gets_a_stop_at_the_recorded_level():
    """A0053-2: no stop at all used to be alert-only."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)
    assert msg.startswith("🔧")


def test_fresh_read_quantity_stops_a_position_the_cache_reads_flat_and_never_cancels():
    """Roadmap 00 (d): the auto-stop for a disagreeing ticker sizes from the
    fresh read while the cache shows no position, and only ever adds."""
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD", fresh_qty=10.0)
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)
    assert msg.startswith("🔧")
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10, 10)), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place") as place:
        msg = tasks._fix_overstop("AMD", fresh_qty=10.0)
    cancel.assert_not_called(); place.assert_not_called()
    assert "nothing cancelled" in msg


def test_no_stop_placed_at_or_above_the_last_price():
    """A stop above the market would sell at once; alert instead."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    for last in (140.0, None):
        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
             patch("ibkr.get_latest_price", return_value=last), \
             patch("tasks.manage_stop_place") as place:
            msg = tasks._fix_overstop("AMD")
        place.assert_not_called()
        assert "nothing was placed" in msg


class TestBackstopSellReads:
    """Professor M61: read errors, shorts and stuck stops in the backstop."""

    _alert = {"current_price": 140.0, "stop_price": 150.0}

    def _client(self, qty):
        client = MagicMock()
        client.get_all_positions.return_value = [] if qty is None else [MagicMock(symbol="AMD", qty=qty)]
        return client

    def test_read_error_keeps_the_record_and_retries(self):
        with patch("ibkr.get_client", side_effect=RuntimeError("down")), \
             patch("db.delete_position_stop") as delete, \
             patch("tasks._schedule_backstop_retry") as retry, \
             patch("ibkr.place_market_order") as order:
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        delete.assert_not_called(); order.assert_not_called(); retry.assert_called_once()

    def test_confirmed_flat_drops_the_record(self):
        with patch("ibkr.get_client", return_value=self._client(None)), \
             patch("db.delete_position_stop") as delete, \
             patch("ibkr.place_market_order") as order:
            assert tasks._execute_backstop_sell("AMD", self._alert) is None
        delete.assert_called_once_with("AMD"); order.assert_not_called()

    def test_a_retry_with_nothing_held_does_not_claim_the_position_is_unprotected(self, caplog):
        """11-03 09:30: six stray retries for an unheld AMD each logged
        "still unprotected", which read like a naked position."""
        with patch("ibkr.get_client", return_value=self._client(None)), \
             patch("db.delete_position_stop"), \
             patch("ibkr.place_market_order") as order, \
             caplog.at_level("INFO", logger="tasks"):
            tasks.retry_backstop_sell.run("AMD", self._alert)
        order.assert_not_called()
        text = caplog.text
        assert "no shares held" in text
        assert "unprotected" not in text
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    def test_short_is_not_sold(self):
        with patch("ibkr.get_client", return_value=self._client(-10)), \
             patch("ibkr.place_market_order") as order, \
             patch("alerts._send") as send:
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        order.assert_not_called()
        assert "SHORT" in send.call_args[0][0]

    def test_stuck_stop_shares_are_not_sold_again(self):
        with patch("ibkr.get_client", return_value=self._client(10)), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=_orders(4)), \
             patch("ibkr.place_market_order", return_value={"filled_qty": 6, "fill_price": 140.0}) as order, \
             patch("db.log_trade"), patch("db.delete_position_stop") as delete, \
             patch("db.close_trade_outcome") as close, patch("alerts._send"):
            assert tasks._execute_backstop_sell("AMD", self._alert) is True
        order.assert_called_once_with("AMD", "SELL", 6.0)
        delete.assert_not_called(); close.assert_not_called()


def test_backstop_sale_gets_an_orders_row_and_a_part_fill_waits_for_reconcile():
    """Professor (review of c13548c): a part-filled backstop had no orders row,
    so its remainder was never reconciled."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    for filled, status, closes in ((6, "partial", False), (10, "filled", True)):
        with patch("ibkr.get_client", return_value=client), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 77, "filled_qty": filled, "fill_price": 140.0}), \
             patch("db.log_trade"), patch("db.delete_position_stop") as delete, \
             patch("db.log_order", return_value=5) as log_order, \
             patch("db.update_order_fill") as upd, \
             patch("db.close_trade_outcome") as close, patch("alerts._send"):
            assert tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert) is True
        assert log_order.call_args.kwargs["intent"] == "BACKSTOP"
        assert log_order.call_args.kwargs["broker_order_id"] == 77
        upd.assert_called_once_with(5, 140.0, filled, status, filled_at=None)
        assert close.called is closes
        # A part fill keeps the stop record: a broker-cancelled remainder
        # leaves held shares the next cycle must see (professor, M128 class).
        assert delete.called is closes


def test_backstop_does_not_sell_shares_a_working_sell_already_covers():
    """Review of 16009ad: a second backstop while the first one's remainder
    was still working would short once both filled."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    working = [{"order_id": 77, "ticker": "AMD", "action": "SELL", "order_type": "MKT",
                "qty": 10.0, "stop_price": None, "status": "Submitted", "remaining": 4.0}]
    for open_orders, sold in ((working, 6.0), ([{**working[0], "remaining": 10.0}], None)):
        with patch("ibkr.get_client", return_value=client), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.manage_cancel_stop_orders_for") as cancel, \
             patch("ibkr.get_open_orders", return_value=open_orders), \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 78, "filled_qty": 6, "fill_price": 140.0}) as order, \
             patch("db.log_trade"), patch("db.delete_position_stop") as delete, \
             patch("db.log_order", return_value=5), \
             patch("db.update_order_fill"), patch("db.close_trade_outcome") as close, patch("alerts._send"):
            tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert)
        if sold is None:
            order.assert_not_called()
            cancel.assert_not_called()   # nothing to sell: stops stay (audit 2026-11-02 #11)
        else:
            order.assert_called_once_with("AMD", "SELL", sold)
            # The working order may still be cancelled: keep the record, and
            # leave the outcome for the sale that finishes it (professor M128).
            delete.assert_not_called(); close.assert_not_called()
    # A manual limit sell (e.g. above the market in TWS) is not cover.
    lmt = [{**working[0], "order_type": "LMT", "remaining": 10.0}]
    with patch("ibkr.get_client", return_value=client), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.get_open_orders", return_value=lmt), \
         patch("ibkr.place_market_order",
               return_value={"order_id": 79, "filled_qty": 10, "fill_price": 140.0}) as order, \
         patch("db.log_trade"), patch("db.delete_position_stop"), patch("db.log_order", return_value=5), \
         patch("db.update_order_fill"), patch("db.close_trade_outcome"), patch("alerts._send"):
        tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert)
    order.assert_called_once_with("AMD", "SELL", 10.0)


def test_fix_overstop_leaves_out_shares_a_working_market_sell_is_selling():
    """Professor M129: a stop sized for those shares too would short."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    mkt = {"order_id": 9, "ticker": "AMD", "action": "SELL", "order_type": "MKT", "qty": 10.0,
           "stop_price": None, "status": "Submitted", "remaining": 4.0}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[mkt]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        tasks._fix_overstop("AMD")
    place.assert_called_once_with("AMD", 6.0, 150.0, market_price=200.0)


def test_late_sell_reprotect_sizes_the_stop_net_of_the_working_remainder():
    """Professor M129: 20 left, 15 of them still being sold by our MKT order."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=20)]
    mkt = {"order_id": 9, "ticker": "CRWD", "action": "SELL", "order_type": "MKT", "qty": 50.0,
           "stop_price": None, "status": "Submitted", "remaining": 15.0}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[mkt]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        assert tasks._reprotect_after_late_sell("CRWD", {"intent": "TRIM"}, 175.0) is True
    place.assert_called_once_with("CRWD", 5.0, 150.0)


def _late_sell_reprotect(open_orders):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=20)]
    kw = {"side_effect": open_orders} if isinstance(open_orders, Exception) else {"return_value": open_orders}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", **kw), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        ok = tasks._reprotect_after_late_sell("CRWD", {"intent": "TRIM"}, 175.0)
    return ok, place


def test_late_sell_reprotect_stops_everything_when_open_orders_cannot_be_read():
    ok, place = _late_sell_reprotect(RuntimeError("gateway down"))
    assert ok is True
    place.assert_called_once_with("CRWD", 20.0, 150.0)


def test_late_sell_reprotect_places_nothing_when_the_working_sell_covers_all():
    mkt = {"order_id": 9, "ticker": "CRWD", "action": "SELL", "order_type": "MKT", "qty": 20.0,
           "stop_price": None, "status": "Submitted", "remaining": 20.0}
    ok, place = _late_sell_reprotect([mkt])
    assert ok is True
    place.assert_not_called()


def test_cancel_failure_in_overcover_returns_a_line_not_a_crash():
    """A0253-7: label was unbound when the cancel raised."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10, 10)), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=tasks.OrdersCallUnknown("x")):
        msg = tasks._fix_overstop("AMD")
    assert "fix failed" in msg


def test_inactive_stop_is_not_protection_in_reconcile():
    """A0253-6: section 3 used raw status, so an Inactive stop hid a naked
    position."""
    from datetime import datetime, timedelta, timezone
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    inactive = [{**_orders(10)[0], "status": "Inactive"}]
    t = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    state = {"no_stop_since_AMD": t, "no_stop_seen_AMD": t}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=inactive), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._fix_overstop_safe", return_value="🚨 AMD: x") as fix, \
         patch("alerts._send"):
        tasks.reconcile_stop_orders.run()
    fix.assert_called_once_with("AMD")


def _run_2b(position_qty):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=position_qty)]
    mkt = {"order_id": 7, "ticker": "AMD", "action": "SELL", "order_type": "MKT", "qty": 10.0,
           "stop_price": None, "status": "Submitted", "remaining": 10.0}
    state = {}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(10) + [mkt]), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._fix_overstop_safe", return_value=None) as fix, \
         patch("alerts._send"):
        tasks.reconcile_stop_orders.run()
    return state.get("overstop_pending_AMD"), fix


def test_2b_counts_a_working_market_sell_as_cover_for_a_long():
    """Professor M129: 20 held, 10 under a stop, 10 being sold now."""
    pending, fix = _run_2b(20)
    assert not pending
    fix.assert_not_called()


def test_2b_does_not_net_a_market_sell_against_a_short():
    """Review of 6e6d169: a SELL adds to a short; the mismatch must surface."""
    pending, fix = _run_2b(-20)
    assert pending
    fix.assert_not_called()   # confirmed next cycle


def test_slow_cancel_gets_a_second_look():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    pending = [{**o, "status": "PendingCancel"} for o in _orders(10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), pending, []]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        tasks._fix_overstop("AMD")
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)


def test_short_position_gets_no_sell_stop():
    """Professor M63: a SELL stop on a short adds to it."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=-10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks.manage_stop_place") as place:
        msg = tasks._fix_overstop("AMD")
    place.assert_not_called()
    assert "SHORT" in msg


def test_queued_reprotect_places_no_stop_on_a_short():
    """Professor M63 residual: the queue path used abs(qty)."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=-5)]
    with patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place") as place, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("alerts._send") as send:
        assert tasks._reprotect_after_late_sell("AMD", {"intent": "RESTORE"}, None,
                                                from_queue=True) is None
    place.assert_not_called(); cancel.assert_not_called()
    assert "SHORT" in send.call_args[0][0]



from datetime import datetime as _dt, timedelta as _td, timezone as _tz
_T0 = _dt(2026, 10, 30, 15, 0, tzinfo=_tz.utc)


def _queued_flat(intent, row, open_orders=(), stop=None, order_id=5, outcomes=()):
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client), \
         patch("db.get_position_stop", return_value=stop), \
         patch("db.get_order_fill_state", **({"side_effect": row} if isinstance(row, Exception)
                                            else {"return_value": row})), \
         patch("ibkr.get_open_orders", return_value=list(open_orders)), \
         patch("ibkr.get_recent_sell_fill") as fill, patch("db.delete_position_stop"), \
         patch("db.get_open_outcomes", return_value=list(outcomes)), \
         patch("db.close_trade_outcome") as close, patch("alerts._send") as alert:
        r = tasks._reprotect_after_late_sell("CRWD", {"intent": intent, "order_id": order_id}, 175.0,
                                             from_queue=True)
    fill.assert_not_called()   # never the execution-blend lookup
    return r, close, alert


def test_queued_flat_exit_closes_at_its_orders_average_fill():
    """Professor M124: the part-filled EXIT deleted the stop record, so the
    stale-stop branch never closed the outcome."""
    r, close, alert = _queued_flat("EXIT", {"status": "filled", "fill_price": 171.5, "broker_order_id": 77, "created_at": _T0})
    assert r is None
    close.assert_called_once_with("CRWD", exit_price=171.5, exit_source="late_exit", exit_price_source="fill", exit_fill_at=None)
    alert.assert_not_called()


def test_queued_flat_exit_waits_while_its_remainder_works_or_a_lookup_fails():
    row = {"status": "partial", "fill_price": 171.5, "broker_order_id": 77}
    working = {"order_id": 77, "status": "Submitted"}
    for kw, due, alerted in (({"row": row, "open_orders": [working]}, True, True),
                             ({"row": row, "open_orders": [working]}, False, False),   # throttled
                             ({"row": RuntimeError("db")}, True, False)):
        with patch("tasks._reconcile_alert_due", return_value=due):
            r, close, alert = _queued_flat("BACKSTOP", **kw)
        assert r == "wait"
        close.assert_not_called()
        assert alert.called is alerted   # flat + working sell = a short if it fills


def test_queued_flat_exit_leaves_mixed_and_owned_cases_alone():
    row = {"status": "partial", "fill_price": 171.5, "broker_order_id": 77}
    for kw in ({"row": row, "open_orders": [{"order_id": 77, "status": "Cancelled"}]},   # mixed close
               {"row": row, "order_id": None}):                                          # old entry
        r, close, alert = _queued_flat("EXIT", **kw)
        assert r is None
        close.assert_not_called()
        alert.assert_not_called()
    r, close, alert = _queued_flat("TRIM", {"status": "filled", "fill_price": 171.5, "broker_order_id": 77, "created_at": _T0})
    assert r is None
    close.assert_not_called()


def test_retry_keeps_a_waiting_exit_without_counting_a_try():
    import json
    state = {tasks._REPROTECT_STATE_KEY: json.dumps({"CRWD": {"intent": "EXIT", "order_id": 5, "nonce": "n"}})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
         patch("tasks._reprotect_after_late_sell", return_value="wait"), \
         patch("alerts._send") as alert:
        tasks._retry_pending_reprotects()
    kept = json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]
    assert kept["waits"] == 1 and "tries" not in kept
    alert.assert_not_called()


def test_a_queued_profit_target_keeps_a_later_exits_order_for_the_flat_close():
    """Professor m111: the PROFIT_TARGET entry keeps priority, and the EXIT's
    order row id rides along so its flat close still finds the order."""
    import json
    state = {tasks._REPROTECT_STATE_KEY: json.dumps(
        {"CRWD": {"intent": "PROFIT_TARGET", "stop_price": 180.0, "nonce": "a"}})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
        tasks._queue_reprotect("CRWD", {"intent": "EXIT", "id": 5}, 171.0)
    rec = json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]
    assert rec["intent"] == "PROFIT_TARGET" and rec["exit_order_id"] == 5
    r, close, alert = _queued_flat("PROFIT_TARGET", {"status": "filled", "fill_price": 171.5,
                                                     "broker_order_id": 77, "created_at": _T0}, order_id=None)
    close.assert_not_called()   # no exit fields: unchanged
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client), \
         patch("db.get_position_stop", return_value=None), \
         patch("db.get_open_outcomes", return_value=[]), patch("db.delete_position_stop"), \
         patch("db.get_order_fill_state", return_value={"status": "filled", "fill_price": 171.5,
                                                         "broker_order_id": 77, "created_at": _T0}) as row, \
         patch("db.close_trade_outcome") as close:
        tasks._reprotect_after_late_sell("CRWD", rec, 171.0, from_queue=True)
    row.assert_called_once_with(5)
    close.assert_called_once_with("CRWD", exit_price=171.5, exit_source="late_exit", exit_price_source="fill", exit_fill_at=None)


def test_a_restore_replacing_a_queued_exit_keeps_the_exits_order():
    """Review of 03efab7: the same gap as m111, for a RESTORE/BUY re-queue."""
    import json
    state = {tasks._REPROTECT_STATE_KEY: json.dumps(
        {"CRWD": {"intent": "EXIT", "order_id": 5, "nonce": "a"}})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
        tasks._queue_reprotect("CRWD", {"intent": "RESTORE"}, None)
    rec = json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]
    assert rec["intent"] == "RESTORE" and rec["exit_intent"] == "EXIT" and rec["exit_order_id"] == 5


def test_a_buy_never_inherits_a_queued_exits_order():
    """Review of 245c9af: a flat read after a re-entry would close the new
    position's outcome at the old exit's fill."""
    import json
    for old in ({"intent": "EXIT", "order_id": 5, "nonce": "a"},
                {"intent": "PROFIT_TARGET", "stop_price": 180.0, "exit_intent": "EXIT",
                 "exit_order_id": 5, "nonce": "a"}):
        state = {tasks._REPROTECT_STATE_KEY: json.dumps({"CRWD": old})}
        with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
            tasks._queue_reprotect("CRWD", {"intent": "BUY"}, None)
        rec = json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]
        assert "exit_intent" not in rec and "exit_order_id" not in rec
    assert rec["had_buy"] is True and rec["intent"] == "PROFIT_TARGET"   # M80 ruling: flat-BUY protection


def test_a_buy_merged_into_a_profit_target_gets_the_flat_buy_retry():
    """Professor ruling on M80: one flat read after the BUY must not drop it."""
    import json
    rec = {"intent": "PROFIT_TARGET", "stop_price": 180.0, "had_buy": True, "nonce": "a"}
    state = {tasks._REPROTECT_STATE_KEY: json.dumps({"CRWD": rec})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
         patch("tasks._reprotect_after_late_sell", return_value=None), \
         patch("alerts._send") as alert, patch("tasks.open_manual_action"):
        tasks._retry_pending_reprotects()
        assert json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]["tries"] == 1
        alert.assert_not_called()
        tasks._retry_pending_reprotects()
    alert.assert_called_once()
    assert "STOP NOT PLACED" in alert.call_args.args[0]


def test_an_exit_after_a_merged_buy_clears_the_buy_flag():
    """Review of ec2fffb: PROFIT_TARGET, then BUY, then EXIT; the exit's own
    sale makes the flat read expected, so no STOP NOT PLACED alert."""
    import json
    state = {tasks._REPROTECT_STATE_KEY: json.dumps(
        {"CRWD": {"intent": "PROFIT_TARGET", "stop_price": 180.0, "nonce": "a"}})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
        tasks._queue_reprotect("CRWD", {"intent": "BUY"}, None)
        assert json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]["had_buy"] is True
        tasks._queue_reprotect("CRWD", {"intent": "EXIT", "id": 5}, 171.0)
    rec = json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]
    assert "had_buy" not in rec and rec["exit_order_id"] == 5


def test_a_profit_target_replacing_a_queued_buy_keeps_its_flat_read_protection():
    """Review of b6b6a77: a target sale leaves shares, so a flat read after a
    BUY is still unexplained."""
    import json
    for old in ({"intent": "BUY", "nonce": "a"},
                {"intent": "PROFIT_TARGET", "stop_price": 180.0, "had_buy": True, "nonce": "a"}):
        state = {tasks._REPROTECT_STATE_KEY: json.dumps({"CRWD": old})}
        with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
            tasks._queue_reprotect("CRWD", {"intent": "PROFIT_TARGET", "stop_price": 185.0}, 190.0)
        rec = json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"]
        assert rec["intent"] == "PROFIT_TARGET" and rec["stop_price"] == 185.0 and rec["had_buy"] is True
    for old, new, expect in (({"intent": "BUY", "nonce": "a"}, {"intent": "RESTORE"}, True),
                             ({"intent": "RESTORE", "nonce": "a"}, {"intent": "PROFIT_TARGET"}, False),
                             ({"intent": "BUY", "nonce": "a"}, {"intent": "EXIT", "id": 5}, False),
                             ({"intent": "BUY", "nonce": "a"}, {"intent": "TRIM", "id": 6}, False),
                             ({"intent": "BUY", "nonce": "a"}, {"intent": "MANUAL_SELL", "id": 7}, False)):
        state = {tasks._REPROTECT_STATE_KEY: json.dumps({"CRWD": old})}
        with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
            tasks._queue_reprotect("CRWD", new, None)
        assert bool(json.loads(state[tasks._REPROTECT_STATE_KEY])["CRWD"].get("had_buy")) is expect


def test_a_queued_flat_backstop_with_its_kept_record_closes_from_its_fill():
    """Review of f464bd7: a part-filled backstop keeps the record, and the
    stale-stop branch leaves the backstop's own fills out (false no_fill)."""
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0, "opened_at": _T0 - _td(days=2)}), \
         patch("db.get_open_outcomes", return_value=[{"ticker": "CRWD", "entry_date": _T0 - _td(days=2)}]), \
         patch("db.get_order_fill_state", return_value={"status": "filled", "fill_price": 171.5,
                                                         "broker_order_id": 77, "created_at": _T0}), \
         patch("db.close_trade_outcome") as close, patch("db.delete_position_stop") as delete:
        r = tasks._reprotect_after_late_sell("CRWD", {"intent": "BACKSTOP", "order_id": 5}, 171.0,
                                             from_queue=True)
    assert r is None
    close.assert_called_once_with("CRWD", exit_price=171.5, exit_source="late_backstop", exit_price_source="fill", exit_fill_at=None)
    delete.assert_called_once_with("CRWD")


def test_a_queued_flat_exit_never_closes_a_position_opened_after_it():
    """Review of 6f48241: a newer record or outcome is another position's."""
    filled = {"status": "filled", "fill_price": 171.5, "broker_order_id": 77, "created_at": _T0}
    for kw, closes in (({"stop": {"stop_price": 150.0, "opened_at": _T0 + _td(minutes=20)}}, False),
                       ({"outcomes": [{"ticker": "CRWD", "entry_date": _T0 + _td(minutes=20)}]}, False),
                       ({"outcomes": [{"ticker": "CRWD", "entry_date": _T0 - _td(days=1)}]}, True)):
        r, close, alert = _queued_flat("EXIT", filled, **kw)
        assert r is None
        assert close.called is closes


def test_the_newer_position_check_waits_when_a_read_fails():
    """Professor m114: a swallowed DB error used to pass as "nothing newer"."""
    filled = {"status": "filled", "fill_price": 171.5, "broker_order_id": 77, "created_at": _T0}
    client = MagicMock()
    client.get_all_positions.return_value = []
    for stop_kw, outcomes_kw in (({"side_effect": RuntimeError("db")}, {"return_value": []}),
                                 ({"return_value": None}, {"side_effect": RuntimeError("db")})):
        with patch("ibkr.get_client", return_value=client), \
             patch("db.get_order_fill_state", return_value=filled), \
             patch("db.get_position_stop", **stop_kw) as stop, \
             patch("db.get_open_outcomes", **outcomes_kw) as outcomes, \
             patch("db.close_trade_outcome") as close, patch("db.delete_position_stop") as delete:
            r = tasks._reprotect_after_late_sell("CRWD", {"intent": "EXIT", "order_id": 5}, 171.0, from_queue=True)
        assert r == "wait"
        close.assert_not_called()
        delete.assert_not_called()
        assert stop.call_args.kwargs == {"strict": True}


def _queued_restore(open_orders, before, after, intent="RESTORE", held=20):
    """from_queue path with coverage read before the cancel (before) and the
    stops left after it (after)."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=held)]
    kw = {"side_effect": open_orders} if isinstance(open_orders, Exception) else {"return_value": open_orders}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", **kw), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("db.mark_profit_target_taken"), \
         patch("execution._stops_still_resting", return_value=before), \
         patch("tasks._resting_sell_stops",
               **({"side_effect": after} if isinstance(after, Exception) else {"return_value": after})), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        ok = tasks._reprotect_after_late_sell("CRWD", {"intent": intent, "stop_price": 160.0}, None,
                                              from_queue=True)
    return ok, place, cancel


def test_queued_restore_waits_when_open_orders_cannot_be_read():
    """m116: a queued retry does not stop blind; it counts a try."""
    ok, place, cancel = _queued_restore(TimeoutError("no reply"), 0.0, [])
    assert ok is False
    place.assert_not_called(); cancel.assert_not_called()


def test_queued_restore_waits_when_resting_stops_cannot_be_read():
    ok, place, cancel = _queued_restore([], None, [])
    assert ok is False
    place.assert_not_called(); cancel.assert_not_called()


def test_queued_restore_covers_only_what_a_stop_it_cannot_cancel_leaves():
    """m116: another client's stop on 8 shares stays; ours goes on 12."""
    other = {"order_id": 5, "ticker": "CRWD", "qty": 8.0, "stop_price": 140.0}
    ok, place, cancel = _queued_restore([], 8.0, [other])
    assert ok is True
    cancel.assert_called_once_with("CRWD")
    place.assert_called_once_with("CRWD", 12.0, 150.0)


def test_queued_restore_nets_a_working_market_sell():
    """m116 / M129: remaining - selling - covered."""
    mkt = {"order_id": 9, "ticker": "CRWD", "action": "SELL", "order_type": "MKT", "qty": 5.0,
           "stop_price": None, "status": "Submitted", "remaining": 5.0}
    other = {"order_id": 5, "ticker": "CRWD", "qty": 8.0, "stop_price": 140.0}
    ok, place, _ = _queued_restore([mkt], 8.0, [other])
    assert ok is True
    place.assert_called_once_with("CRWD", 7.0, 150.0)


def test_queued_restore_with_no_cover_places_without_a_cancel_round_trip():
    ok, place, cancel = _queued_restore([], 0.0, [])
    assert ok is True
    cancel.assert_not_called()
    place.assert_called_once_with("CRWD", 20.0, 150.0)


def test_queued_profit_target_replaces_ours_and_sizes_to_what_is_left():
    """A profit target still re-places (it raises the stop), sized net of a
    stop the cancel could not remove."""
    other = {"order_id": 5, "ticker": "CRWD", "qty": 8.0, "stop_price": 140.0}
    ok, place, cancel = _queued_restore([], 20.0, [other], intent="PROFIT_TARGET")
    assert ok is True
    cancel.assert_called_once_with("CRWD")
    place.assert_called_once_with("CRWD", 12.0, 160.0)


def test_queued_restore_gives_up_with_an_alert_and_a_manual_action():
    """m116 condition 1: unknown coverage ends in STOP NOT RESTORED plus a
    manual action, not a silent drop."""
    import json
    state = {"reprotect_pending": json.dumps(
        {"CRWD": {"intent": "RESTORE", "tries": tasks._REPROTECT_MAX_TRIES - 1, "nonce": "n"}})}
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=20)]
    with patch("db.get_state", side_effect=lambda k: state.get(k)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=TimeoutError("no reply")), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("tasks.manage_stop_place") as place, \
         patch("tasks.open_manual_action") as manual, \
         patch("alerts._send") as send:
        tasks._retry_pending_reprotects()
    place.assert_not_called()
    assert json.loads(state["reprotect_pending"]) == {}
    assert "STOP NOT RESTORED" in send.call_args[0][0]
    manual.assert_called_once()


def test_queued_restore_giving_up_on_a_sign_flip_says_check_long_or_short():
    # last try refused: cache long 20, fresh read short. The give-up alert
    # must not tell anyone to place a SELL stop
    import json
    state = {"reprotect_pending": json.dumps(
        {"CRWD": {"intent": "RESTORE", "tries": tasks._REPROTECT_MAX_TRIES - 1, "nonce": "n"}})}
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=20)]
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_positions", return_value={"CRWD": -20.0}), \
         patch("tasks._reconcile_alert_due", return_value=False), \
         patch("tasks.manage_stop_place") as place, \
         patch("tasks.open_manual_action") as manual, \
         patch("alerts._send") as send:
        tasks._retry_pending_reprotects()
    place.assert_not_called()
    assert json.loads(state["reprotect_pending"]) == {}
    msg = send.call_args[0][0]
    assert "STOP NOT RESTORED" in msg and "long or short before placing anything" in msg
    assert "place a stop in TWS" not in msg
    assert "long or short" in manual.call_args[0][1]
    assert "CRWD" in tasks._REPROTECT_SIGN_FLIP
    tasks._REPROTECT_SIGN_FLIP.discard("CRWD")


def test_queued_restore_stops_everything_if_the_re_read_after_its_cancel_fails():
    """Review of 48581aa: ours is already cancelled, so a failed re-read must
    not leave the shares naked."""
    ok, place, cancel = _queued_restore([], 8.0, TimeoutError("no reply"))
    assert ok is True
    cancel.assert_called_once_with("CRWD")
    place.assert_called_once_with("CRWD", 20.0, 150.0)


def test_queued_restore_takes_a_second_look_at_a_pending_cancel():
    ours = {"order_id": 4, "ticker": "CRWD", "qty": 20.0, "stop_price": 150.0, "status": "PendingCancel"}
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="CRWD", qty=20)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("execution._stops_still_resting", return_value=5.0), \
         patch("tasks._resting_sell_stops", side_effect=[[ours], []]), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        assert tasks._reprotect_after_late_sell("CRWD", {"intent": "RESTORE"}, None, from_queue=True) is True
    place.assert_called_once_with("CRWD", 20.0, 150.0)


def test_queued_flat_exit_passes_its_orders_fill_time():
    """Professor M110: the queued close can run ~4 h after the fill."""
    r, close, _ = _queued_flat("EXIT", {"status": "filled", "fill_price": 171.5, "broker_order_id": 77,
                                        "created_at": _T0, "filled_at": "FILLED-AT"})
    assert r is None
    assert close.call_args.kwargs["exit_fill_at"] == "FILLED-AT"


def test_backstop_outside_regular_hours_cancels_nothing_and_retries_at_the_open():
    """Audit 2026-11-02 #11: the backstop cancelled the stops, then the
    after-hours market SELL was cancelled after its fill wait, leaving the
    position with no stop overnight."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("market_data.is_market_hours", return_value=False), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.place_market_order") as order, \
         patch("tasks._schedule_backstop_retry") as retry:
        assert tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert) is False
    cancel.assert_not_called()
    order.assert_not_called()
    retry.assert_called_once()


def _retry_run(open_orders, qty=10, open_orders_error=None):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=qty)]
    oo = {"side_effect": open_orders_error} if open_orders_error else {"return_value": open_orders}
    with patch("ibkr.get_client", return_value=client), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("ibkr.get_open_orders", **oo), \
         patch("ibkr.place_market_order",
               return_value={"order_id": 80, "filled_qty": qty, "fill_price": 140.0}) as order, \
         patch("db.log_trade"), patch("db.delete_position_stop"), patch("db.log_order", return_value=5), \
         patch("db.update_order_fill"), patch("db.close_trade_outcome"), \
         patch("tasks._schedule_backstop_retry"), patch("alerts._send") as alert:
        tasks.retry_backstop_sell.run("AMD", TestBackstopSellReads._alert)
    return cancel, order, alert


def test_backstop_retries_at_the_open_sell_only_once_per_session():
    """Review of 5f13c44: several retry_backstop_sell tasks queued for the same
    open each cancelled the stops and sold."""
    _, order, _ = _retry_run([])
    order.assert_called_once()
    cancel, order, alert = _retry_run([])
    cancel.assert_not_called()
    order.assert_not_called()
    assert "BACKSTOP RETRY SKIPPED" in alert.call_args.args[0]


def test_backstop_retry_rechecks_cover_and_working_sells_first():
    stop = _orders(10)
    cancel, order, _ = _retry_run(stop)                 # a stop covers it again
    cancel.assert_not_called(); order.assert_not_called()
    working = [{"order_id": 77, "ticker": "AMD", "action": "SELL", "order_type": "MKT",
                "qty": 4.0, "stop_price": None, "status": "Submitted", "remaining": 4.0}]
    cancel, order, alert = _retry_run(working)          # part of it is being sold
    cancel.assert_not_called(); order.assert_not_called()
    assert "still working" in alert.call_args.args[0]
    cancel, order, alert = _retry_run(None, open_orders_error=RuntimeError("gateway"))
    cancel.assert_not_called(); order.assert_not_called()


def test_backstop_retry_failing_after_the_cancel_sends_the_critical_alert():
    """Roadmap 00 (a): a retry has no further retry, so an exception after its
    stops were cancelled must page, not only log."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    for raise_in, expect in (("place", "BACKSTOP RETRY FAILED"), ("claim", "BACKSTOP RETRY SKIPPED")):
        with patch("ibkr.get_client", return_value=client), \
             patch("market_data.is_market_hours", return_value=True), \
             patch("tasks.manage_cancel_stop_orders_for") as cancel, \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("tasks._claim_backstop_retry_send",
                   side_effect=RuntimeError("boom") if raise_in == "claim" else None,
                   return_value=True), \
             patch("ibkr.place_market_order", side_effect=RuntimeError("gateway")), \
             patch("tasks._schedule_backstop_retry") as retry, \
             patch("alerts._send") as alert:
            assert tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert,
                                                _is_retry=True) is False
        retry.assert_not_called()
        assert cancel.called is (raise_in == "place")
        assert expect in alert.call_args.args[0]


def test_2b_marker_needs_the_minimum_cycle_spacing():
    """A Celery retry (+2 min) must not count as the confirming cycle: the
    overstop marker uses the same 600 s spacing as the orphan markers."""
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=5)]
    for age_s, fixed in ((120, False), (600, True)):
        armed = (now - timedelta(seconds=age_s)).isoformat()
        state = {"overstop_pending_AMD": armed}
        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=_orders(10)), \
             patch("db.get_all_position_stops", return_value=[]), \
             patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("tasks._reconcile_now", return_value=now), \
             patch("tasks._fix_overstop_safe", return_value="ok") as fix, \
             patch("alerts._send"):
            tasks.reconcile_stop_orders.run()
        assert fix.called is fixed, age_s
        assert state["overstop_pending_AMD"] == ("" if fixed else armed)


# ── Own cancel still pending: never counted as cover ───────────────────

def _cancelling(qty, status="PendingCancel", order_id=0):
    return {**_orders(qty)[0], "order_id": order_id, "status": status}


def _overstop(reads, queue_effect=None):
    """_fix_overstop on 10 held with stops for 20; reads = the open-orders
    reads after the first (the post-cancel looks)."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    queue = MagicMock(side_effect=queue_effect)
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[_orders(10, 10), *reads]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("time.sleep") as sleep, \
         patch.object(tasks.recheck_stop_coverage, "apply_async", new=queue), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    return msg, place, sleep, queue


def test_own_cancel_still_pending_is_not_cover_and_a_stop_is_placed():
    # both of our stops still PendingCancel after 3 x 5 s: the old code
    # counted them as stuck cover (need = 0) and the position went bare
    # once the cancels landed
    stuck = [_cancelling(10, order_id=0), _cancelling(10, order_id=1)]
    msg, place, sleep, queue = _overstop([stuck] * 4)
    assert sleep.call_count == tasks._CANCEL_SETTLE_LOOKS
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)
    queue.assert_called_once_with(args=["AMD"], kwargs={"attempt": 1}, countdown=60)
    assert msg.startswith("🔧") and "still being cancelled" in msg


def test_api_pending_cancel_is_waited_for_too():
    msg, place, sleep, queue = _overstop([[_cancelling(10, "ApiPendingCancel")], []])
    assert sleep.call_count == 1
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)
    queue.assert_not_called()
    assert "still being cancelled" not in msg


def test_cancel_that_lands_on_a_later_look_stops_the_waits():
    pending = [_cancelling(10)]
    msg, place, sleep, queue = _overstop([pending, pending, []])
    assert sleep.call_count == 2
    place.assert_called_once_with("AMD", 10.0, 150.0, market_price=200.0)
    queue.assert_not_called()


def test_working_stop_that_did_not_cancel_covers_with_no_success_line():
    # a stop for 10 that refused the cancel covers the 10 held; another is
    # still cancelling: nothing placed, and not reported as a fix
    msg, place, _, queue = _overstop([[_orders(10)[0], _cancelling(10, order_id=1)]] * 4)
    place.assert_not_called()
    assert not msg.startswith("🔧") and "nothing placed" in msg
    queue.assert_called_once()


def test_fix_overstop_survives_a_recheck_queue_failure():
    stuck = [_cancelling(10)]
    msg, place, _, _ = _overstop([stuck] * 4, queue_effect=RuntimeError("broker down"))
    place.assert_called_once()
    assert msg.startswith("🔧") and "reconcile re-checks" in msg


def test_under_cover_does_not_count_a_cancelling_stop():
    # 4 working + 3 cancelling of 10 held: stop the 6 not working-covered
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    queue = MagicMock()
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=_orders(4) + [_cancelling(3, order_id=1)]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch.object(tasks.recheck_stop_coverage, "apply_async", new=queue), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD")
    place.assert_called_once_with("AMD", 6.0, 150.0, market_price=200.0)
    queue.assert_called_once()


# ── recheck_stop_coverage: one lock around the check and the fix ──────

def test_recheck_precheck_and_fix_run_under_the_ticker_lock():
    import stop_lock
    seen = []

    def read(ticker, *a, **k):
        seen.append(("read", stop_lock._local.held.get(ticker)))
        return []

    def fix(ticker, **k):
        seen.append(("fix", stop_lock._local.held.get(ticker)))
        return "AMD: stop coverage fine on re-check (10 vs 10)"
    with patch("tasks._resting_sell_stops", side_effect=read), \
         patch("tasks._fix_overstop_safe", side_effect=fix):
        tasks.recheck_stop_coverage("AMD")
    assert seen == [("read", 1), ("fix", 1)]
    assert not stop_lock._local.held.get("AMD")


def test_recheck_lock_nests_with_fix_overstops_own():
    # the real _fix_overstop (@locked_by_ticker) inside the task's lock
    import stop_lock
    depth = []
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]

    def orders():
        depth.append(stop_lock._local.held.get("AMD"))
        return _orders(10)
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=orders), \
         patch("alerts._send") as send:
        tasks.recheck_stop_coverage("AMD")
    assert depth == [1, 2]   # precheck under the task's lock, then nested
    send.assert_not_called()


def test_recheck_busy_ticker_requeues():
    import stop_lock
    from conftest import _FakeLock
    queue = MagicMock()
    with patch.object(stop_lock, "DEFAULT_WAIT_S", 0), \
         patch.object(tasks.recheck_stop_coverage, "apply_async", new=queue), \
         patch("tasks._fix_overstop_safe") as fix:
        _FakeLock.held.add("helios:stop_lock:AMD")
        try:
            tasks.recheck_stop_coverage("AMD")
        finally:
            _FakeLock.held.discard("helios:stop_lock:AMD")
    fix.assert_not_called()
    queue.assert_called_once_with(args=["AMD"], kwargs={"attempt": 2}, countdown=60)


def test_recheck_requeue_failure_is_caught():
    queue = MagicMock(side_effect=RuntimeError("broker down"))
    with patch("tasks._resting_sell_stops", return_value=[_cancelling(10)]), \
         patch.object(tasks.recheck_stop_coverage, "apply_async", new=queue), \
         patch("tasks._fix_overstop_safe") as fix:
        tasks.recheck_stop_coverage("AMD")   # must not raise
    queue.assert_called_once()
    fix.assert_not_called()


# ── Stops that refuse a cancel are remembered, not cancelled again ─────

class _Broker:
    """Open orders on a clock. A cross-client stop answers a cancel with
    PendingCancel for 180 s, then reverts to working (MRVL 08-04,
    ibkr.cancel_order); ours cancel at once, or after slow_s."""

    def __init__(self, held=100):
        self.clock, self.held, self.orders, self.next_id = 0.0, held, {}, 100
        self.cancels = {}      # order id -> cancel requests
        self.queued = []       # (due time, attempt)
        self.state = {}        # system_state
        self.sent = []
        self.by_id = []        # (order id, symbol, perm_id) per single cancel
        self.fresh = None      # a fresh reqPositions() qty other than held
        self.fresh_read = None   # the whole fresh read (or an exception), if set
        self.revert_s = 180    # how long a cross-client stop shows the cancel

    def add(self, oid, qty, cross=False, slow_s=0, perm=None):
        self.orders[oid] = {"qty": qty, "status": "Submitted", "cross": cross,
                            "slow_s": slow_s, "at": None, "perm": perm}

    def _tick(self):
        for oid, o in list(self.orders.items()):
            if o["status"] == "PendingCancel" and self.clock >= o["at"]:
                if o["cross"]:
                    o["status"], o["at"] = "Submitted", None
                else:
                    del self.orders[oid]

    def open_orders(self):
        self._tick()
        return [{"order_id": k, "perm_id": v["perm"], "ticker": "AMD", "action": "SELL",
                 "order_type": "STP", "qty": v["qty"], "stop_price": 150.0,
                 "status": v["status"]} for k, v in self.orders.items()]

    def cancel(self, oid, symbol=None, perm_id=None):
        if symbol is not None:   # the per-order path (manage_cancel_order)
            self.by_id.append((oid, symbol, perm_id))
        self.cancels[oid] = self.cancels.get(oid, 0) + 1
        o = self.orders.get(oid)
        if o is None:
            return False
        if o["cross"]:
            o["status"], o["at"] = "PendingCancel", self.clock + self.revert_s
        elif o["slow_s"]:
            o["status"], o["at"] = "PendingCancel", self.clock + o["slow_s"]
        else:
            del self.orders[oid]
        self.clock += 1
        return not o["cross"] and not o["slow_s"]

    def cancel_all(self, ticker):
        return sum(self.cancel(oid) for oid in list(self.orders))

    def place(self, ticker, qty, price, market_price=None):
        self.next_id += 1
        self.add(self.next_id, qty)
        return {"order_id": self.next_id}

    def sleep(self, s):
        self.clock += s

    def queue(self, args=None, kwargs=None, countdown=0):
        self.queued.append((self.clock + countdown, kwargs["attempt"]))

    def cover(self):
        self._tick()
        return sum(v["qty"] for v in self.orders.values())

    def patches(self):
        client = MagicMock()
        client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=self.held)]
        fresh = self.held if self.fresh is None else self.fresh
        read = {"AMD": float(fresh)} if self.fresh_read is None else self.fresh_read
        read_kw = ({"side_effect": read} if isinstance(read, Exception)
                   else {"return_value": read})
        return [patch("ibkr.get_client", return_value=client),
                patch("ibkr.get_fresh_stock_positions", **read_kw),
                patch("ibkr.get_open_orders", side_effect=self.open_orders),
                patch("db.get_position_stop", return_value={"stop_price": 150.0}),
                patch("db.get_state", side_effect=lambda k, d=None: self.state.get(k, d)),
                patch("db.set_state", side_effect=self.state.__setitem__),
                patch("ibkr.get_latest_price", return_value=200.0),
                patch("tasks.manage_cancel_stop_orders_for", side_effect=self.cancel_all),
                patch("tasks.manage_cancel_order", side_effect=self.cancel),
                patch("time.sleep", side_effect=self.sleep),
                patch.object(tasks.recheck_stop_coverage, "apply_async", new=self.queue),
                patch("tasks.manage_stop_place", side_effect=self.place),
                patch("db.set_stop_broker_order_id"),
                patch("alerts._send", side_effect=self.sent.append)]

    def run(self, fn):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.patches():
                stack.enter_context(p)
            return fn()

    def run_rechecks(self):
        while self.queued:
            due, attempt = self.queued.pop(0)
            self.clock = max(self.clock, due)
            self.run(lambda: tasks.recheck_stop_coverage("AMD", attempt=attempt))


def test_cross_client_stop_that_refuses_the_cancel_converges_to_exact_cover():
    # held 100, our stop 100, a cross-client stop 40 that reverts each cancel
    b = _Broker()
    b.add(1, 100)
    b.add(2, 40, cross=True, perm=7002)
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    assert msg.startswith("🔧") and "still being cancelled" in msg
    assert "stop_cancel_refused_AMD" in b.state
    b.run_rechecks()
    assert b.cover() == 100
    assert b.orders[2]["status"] == "Submitted" and b.orders[2]["qty"] == 40
    assert b.cancels[2] == 1           # the refused stop was cancelled once
    # later reconcile fixes leave it alone: exact cover, nothing cancelled
    for _ in range(3):
        b.clock += 1800
        msg = b.run(lambda: tasks._fix_overstop_safe("AMD"))
        assert "fine on re-check" in msg
    assert b.cover() == 100 and b.cancels[2] == 1


def test_refused_stop_seen_by_reconcile_only_cancels_our_excess():
    # the re-check chain gave up while the refused stop still showed the
    # cancel; the next reconcile sees it working again
    b = _Broker()
    b.add(1, 100)
    b.add(2, 40, cross=True, perm=7002)
    b.run(lambda: tasks._fix_overstop("AMD"))
    b.queued.clear()
    b.clock += 900
    msg = b.run(lambda: tasks._fix_overstop_safe("AMD"))
    assert msg.startswith("🔧") and "placed a stop for 60" in msg
    assert b.cover() == 100 and b.cancels[2] == 1


def test_our_slow_cancel_that_lands_gives_exact_cover():
    b = _Broker()
    b.add(1, 100, slow_s=20)
    b.add(2, 100)
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    b.run_rechecks()
    b.clock += 60
    assert b.cover() == 100
    assert sum(1 for o in b.orders.values() if o["status"] == "Submitted") == 1
    assert "placed a stop for 100" in msg


def test_refused_stops_covering_the_position_alone_place_nothing():
    b = _Broker()
    b.add(1, 100)
    b.add(2, 100, cross=True, perm=7002)
    b.run(lambda: tasks._fix_overstop("AMD"))
    b.queued.clear()
    b.clock += 900
    placed_before = b.next_id
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    assert msg.startswith("🚨") and "won't cancel" in msg
    assert b.next_id == placed_before     # nothing placed
    assert b.cover() == 100 and b.cancels[2] == 1


_PLACED_PENDING = ("🔧 AMD: stops covered 140 shares but 100 held; placed a stop for 100 @ "
                   "$150.00; 40 shares of stops still being cancelled, re-check queued")
_WONT_CANCEL = ("🚨 AMD: stop order(s) 2 for 140 shares won't cancel and exceed the 100 held "
                "— cancel them in TWS")
_REFUSED_COVER = ("🚨 AMD: stop order(s) 2 for 100 shares refused a cancel (won't cancel) and "
                  "cover the 100 held; nothing placed — cancel them in TWS if they aren't wanted")
_SETTLED = ("AMD: stop coverage settled around stop(s) that refused a cancel (40 working, "
            "0 still showing the cancel, 60 ours vs 100 held); nothing cancelled")
_NOTHING_PLACED = ("⚠️ AMD: stops covered 140 shares but 100 held; stop(s) that did not cancel "
                   "cover the 100 held, nothing placed; 40 shares of stops still being "
                   "cancelled, re-check queued — check the stops in TWS")


def test_a_placed_stop_line_is_never_throttled():
    with patch("tasks._reconcile_alert_due", return_value=False) as due:
        assert tasks._stop_fix_alert_due("AMD", _PLACED_PENDING) is True
        assert tasks._stop_fix_alert_due("AMD", "🔧 AMD: stops covered only 0 of 10 shares; "
                                                "placed a stop for 10 @ $150.00") is True
    due.assert_not_called()


def test_refused_cancel_alert_has_its_own_throttle_key():
    with patch("tasks._reconcile_alert_due", return_value=False) as due:
        assert tasks._stop_fix_alert_due("AMD", _REFUSED_COVER) is False
    due.assert_called_once_with("stop_cancel_refused_AMD")


def test_wont_cancel_and_exceed_is_never_throttled():
    # stops over the position could open a short: always sent
    with patch("tasks._reconcile_alert_due", return_value=False) as due:
        assert tasks._stop_fix_alert_due("AMD", _WONT_CANCEL) is True
    due.assert_not_called()


def test_informational_cancel_lines_share_the_cancelling_key():
    with patch("tasks._reconcile_alert_due", return_value=False) as due:
        assert tasks._stop_fix_alert_due("AMD", _SETTLED) is False
        assert tasks._stop_fix_alert_due("AMD", _NOTHING_PLACED) is False
    assert due.call_args_list == [call("stop_cancelling_AMD")] * 2


def test_other_alerts_are_not_throttled():
    with patch("tasks._reconcile_alert_due", return_value=False) as due:
        assert tasks._stop_fix_alert_due("AMD", "🚨 AMD: broker shows a SHORT of -5") is True
    due.assert_not_called()


def test_a_won_t_cancel_alert_is_not_held_back_by_earlier_informational_lines():
    # real throttle on a state dict: the cancelling lines spend their own
    # key, never the 🚨's
    state = {}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=state.__setitem__):
        assert tasks._stop_fix_alert_due("AMD", _NOTHING_PLACED) is True
        assert tasks._stop_fix_alert_due("AMD", _SETTLED) is False      # throttled
        assert tasks._stop_fix_alert_due("AMD", _PLACED_PENDING) is True
        assert tasks._stop_fix_alert_due("AMD", _REFUSED_COVER) is True   # own key, first
        assert tasks._stop_fix_alert_due("AMD", _REFUSED_COVER) is False  # then throttled
        # the refused-cover 🚨 spent its key; the short-risk 🚨 still goes out
        assert tasks._stop_fix_alert_due("AMD", _WONT_CANCEL) is True
        assert tasks._stop_fix_alert_due("AMD", _WONT_CANCEL) is True
    assert set(state) == {"reconcile_alert_stop_cancelling_AMD",
                          "reconcile_alert_stop_cancel_refused_AMD"}


def test_fix_overstop_cancels_one_order_with_its_symbol_and_perm_id():
    b = _Broker()
    b.add(1, 100)
    b.add(2, 40, cross=True, perm=7002)
    b.run(lambda: tasks._fix_overstop("AMD"))
    b.queued.clear()
    b.clock += 900
    b.by_id = []
    b.run(lambda: tasks._fix_overstop("AMD"))
    assert b.by_id == [(101, "AMD", None)]   # ours only, with its ticker


def test_refused_record_expires():
    from datetime import datetime, timedelta, timezone
    import json
    old = (datetime.now(timezone.utc) - timedelta(seconds=tasks._CANCEL_REFUSED_TTL_S + 60)).isoformat()
    rec = json.dumps({"at": old, "stops": [{"kind": "perm", "id": 2}]})
    with patch("db.get_state", return_value=rec):
        assert tasks._cancel_refused_ids("AMD") == set()
    with patch("db.get_state", return_value=json.dumps({**json.loads(rec),
                                                          "at": datetime.now(timezone.utc).isoformat()})):
        assert tasks._cancel_refused_ids("AMD") == {("perm", 2)}


@pytest.mark.parametrize("revert_s", [200, 250, 290])
def test_a_late_revert_is_caught_by_the_recheck_chain(revert_s):
    b = _Broker()
    b.revert_s = revert_s
    b.add(1, 100)
    b.add(2, 40, cross=True, perm=7002)
    b.run(lambda: tasks._fix_overstop("AMD"))
    b.run_rechecks()
    assert b.cover() == 100           # trimmed in the chain, not left to reconcile
    assert b.orders[2]["status"] == "Submitted" and b.cancels[2] == 1


def _refused_over_position(cross=True):
    """50 held after a partial sale; our stop 50 and a recorded refused
    stop for 100 (it would short 50 if it filled)."""
    import json
    from datetime import datetime, timezone
    b = _Broker(held=50)
    b.add(1, 50)
    b.add(2, 100, cross=cross, perm=7002)
    b.state["stop_cancel_refused_AMD"] = json.dumps(
        {"at": datetime.now(timezone.utc).isoformat(), "stops": [{"kind": "perm", "id": 7002}]})
    return b


def test_refused_stop_over_the_position_alerts_unthrottled_and_keeps_our_stop():
    b = _refused_over_position()
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    assert msg.startswith("🚨") and "won't cancel and exceed the 50 held" in msg
    assert "refused a cancel" not in msg
    assert b.cancels == {2: 1}                    # one retry of the refused stop
    assert 1 in b.orders and b.orders[1]["status"] == "Submitted"   # ours stays
    with patch("tasks._reconcile_alert_due", return_value=False):
        assert tasks._stop_fix_alert_due("AMD", msg) is True


def test_refused_stop_over_the_position_that_cancels_on_retry_leaves_ours():
    b = _refused_over_position(cross=False)
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    assert msg.startswith("🔧") and "our stops cover 50" in msg
    assert b.cover() == 50 and list(b.orders) == [1]


def test_ours_over_the_position_after_the_retry_lands_says_so_and_queues():
    b = _refused_over_position(cross=False)
    b.add(3, 30)                     # ours: 50 + 30 against 50 held
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    assert msg.startswith("⚠️") and "over-stopped" in msg and "cover" not in msg
    assert "re-check queued" in msg and len(b.queued) == 1
    assert 2 not in b.orders and b.cover() == 80   # ours left for the re-check


def test_refused_stop_equal_to_the_position_keeps_the_cover_path():
    b = _refused_over_position()
    b.orders[2]["qty"] = 50
    msg = b.run(lambda: tasks._fix_overstop("AMD"))
    assert "refused a cancel" in msg and "cover the 50 held" in msg
    assert 2 not in b.cancels


def test_recheck_with_a_stale_cache_after_an_add_cancels_nothing():
    # cache 100, fresh 150 (an ADD the cache hasn't caught); stops 100 + a
    # reverted 40: sized from the cache alone this would cancel and re-place
    # for 100, leaving the new shares bare
    b = _Broker(held=100)
    b.fresh = 150
    b.add(1, 100)
    b.add(2, 40)
    with patch("tasks._fix_overstop_safe", wraps=tasks._fix_overstop_safe) as fix:
        b.run(lambda: tasks.recheck_stop_coverage("AMD"))
    assert fix.call_args.kwargs["allow_cancel"] is False
    assert fix.call_args.kwargs["max_qty"] == 100
    assert b.cancels == {}
    assert b.cover() == 140


def test_recheck_reads_that_agree_allow_cancels():
    with patch("ibkr.get_fresh_stock_positions", return_value={"AMD": 100.0}), \
         patch("ibkr.get_client") as client:
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=100)]
        assert tasks._recheck_position_reads("AMD") == {}


@pytest.mark.parametrize("fresh", [TimeoutError("read"), {}])
def test_a_fresh_read_failing_or_empty_is_no_cross_check(fresh):
    # a failed or empty fresh read may be a reconnect gap: add-only, no
    # max_qty (fresh 0 vs cache 50 is not a disagreement)
    kw = {"side_effect": fresh} if isinstance(fresh, Exception) else {"return_value": fresh}
    with patch("ibkr.get_fresh_stock_positions", **kw):
        out = tasks._recheck_position_reads("AMD")
    assert out["allow_cancel"] is False and "max_qty" not in out and "fresh_qty" not in out
    assert out["no_cancel_reason"].startswith("the fresh position read")


def test_a_fresh_read_with_other_tickers_but_not_this_one_does_nothing():
    # zero positions are left out of the fresh read, so this may be a real
    # flat: no add, no cancel, no alert
    b = _Broker(held=50)
    b.fresh_read = {"NVDA": 5.0}
    b.add(1, 20)
    with patch("tasks._fix_overstop_safe") as fix:
        b.run(lambda: tasks.recheck_stop_coverage("AMD"))
    fix.assert_not_called()
    assert b.sent == [] and b.cancels == {} and b.cover() == 20


def test_a_fresh_read_without_the_ticker_queues_another_look_within_the_cap():
    b = _Broker(held=50)
    b.fresh_read = {"NVDA": 5.0}
    b.add(1, 20)
    with patch("tasks._fix_overstop_safe") as fix:
        b.run(lambda: tasks.recheck_stop_coverage("AMD", attempt=2))
        assert [a for _, a in b.queued] == [3]
        b.queued.clear()
        b.run(lambda: tasks.recheck_stop_coverage("AMD", attempt=tasks._COVERAGE_RECHECK_TRIES))
    assert b.queued == []            # last try: no further look
    fix.assert_not_called()
    assert b.sent == []


def test_reads_that_disagree_on_sign_add_no_stop():
    # cache long 50, fresh short -10: max_qty 0, as reconcile's _smaller_read
    b = _Broker(held=50)
    b.fresh = -10
    b.add(1, 20)
    with patch("tasks._fix_overstop_safe", wraps=tasks._fix_overstop_safe) as fix:
        b.run(lambda: tasks.recheck_stop_coverage("AMD"))
    assert fix.call_args.kwargs["max_qty"] == 0.0
    assert fix.call_args.kwargs["allow_cancel"] is False
    assert b.next_id == 100 and b.cancels == {}   # nothing placed or cancelled


@pytest.mark.parametrize("fresh", [TimeoutError("read"), {}])
def test_recheck_over_cover_with_no_cross_check_sends_no_alert(fresh):
    # cache 50, our 50 + a reverted 40: nothing cancelled, no 🚨, and the
    # line doesn't claim the reads disagree
    b = _Broker(held=50)
    b.add(1, 50)
    b.add(2, 40)
    b.fresh_read = fresh
    msgs = []
    with patch("tasks._fix_overstop_safe",
               side_effect=lambda *a, **k: msgs.append(tasks._fix_overstop(*a, **k)) or msgs[-1]):
        b.run(lambda: tasks.recheck_stop_coverage("AMD"))
    assert b.cancels == {} and b.cover() == 90
    assert b.sent == []
    assert "disagree" not in msgs[0] and "reconcile checks it" in msgs[0]
    assert not msgs[0].startswith(("🚨", "⚠️", "🔧"))


def test_under_cover_with_no_cross_check_still_adds():
    b = _Broker(held=50)
    b.add(1, 20)
    with patch("db.set_stop_broker_order_id"):
        msg = b.run(lambda: tasks._fix_overstop(
            "AMD", allow_cancel=False, no_cancel_reason="the fresh position read failed"))
    assert msg.startswith("🔧") and "placed a stop for 30" in msg


def test_a_stop_with_no_perm_id_is_not_recorded_as_refused():
    state = {}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=state.__setitem__):
        tasks._remember_cancel_refused("AMD", [{"order_id": 2, "perm_id": 0},
                                               {"order_id": 3, "perm_id": 7003}], [])
        assert tasks._cancel_refused_ids("AMD") == {("perm", 7003)}
