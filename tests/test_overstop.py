"""Resting stops for more shares than held get rebuilt (professor M54)."""
from unittest.mock import MagicMock, patch

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
    place.assert_called_once_with("AMD", 10.0, 150.0)
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
    place.assert_called_once_with("AMD", 6.0, 150.0)
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
    place.assert_called_once_with("AMD", 10.0, 150.0)
    assert msg.startswith("🔧")


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
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        delete.assert_called_once_with("AMD"); order.assert_not_called()

    def test_short_is_not_sold(self):
        with patch("ibkr.get_client", return_value=self._client(-10)), \
             patch("ibkr.place_market_order") as order, \
             patch("alerts._send") as send:
            assert tasks._execute_backstop_sell("AMD", self._alert) is False
        order.assert_not_called()
        assert "SHORT" in send.call_args[0][0]

    def test_stuck_stop_shares_are_not_sold_again(self):
        with patch("ibkr.get_client", return_value=self._client(10)), \
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
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=[]), \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 77, "filled_qty": filled, "fill_price": 140.0}), \
             patch("db.log_trade"), patch("db.delete_position_stop"), \
             patch("db.log_order", return_value=5) as log_order, \
             patch("db.update_order_fill") as upd, \
             patch("db.close_trade_outcome") as close, patch("alerts._send"):
            assert tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert) is True
        assert log_order.call_args.kwargs["intent"] == "BACKSTOP"
        assert log_order.call_args.kwargs["broker_order_id"] == 77
        upd.assert_called_once_with(5, 140.0, filled, status)
        assert close.called is closes


def test_backstop_does_not_sell_shares_a_working_sell_already_covers():
    """Review of 16009ad: a second backstop while the first one's remainder
    was still working would short once both filled."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    working = [{"order_id": 77, "ticker": "AMD", "action": "SELL", "order_type": "MKT",
                "qty": 10.0, "stop_price": None, "status": "Submitted", "remaining": 4.0}]
    for open_orders, sold in ((working, 6.0), ([{**working[0], "remaining": 10.0}], None)):
        with patch("ibkr.get_client", return_value=client), \
             patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.get_open_orders", return_value=open_orders), \
             patch("ibkr.place_market_order",
                   return_value={"order_id": 78, "filled_qty": 6, "fill_price": 140.0}) as order, \
             patch("db.log_trade"), patch("db.delete_position_stop") as delete, \
             patch("db.log_order", return_value=5), \
             patch("db.update_order_fill"), patch("db.close_trade_outcome") as close, patch("alerts._send"):
            tasks._execute_backstop_sell("AMD", TestBackstopSellReads._alert)
        if sold is None:
            order.assert_not_called()
        else:
            order.assert_called_once_with("AMD", "SELL", sold)
            # The working order may still be cancelled: keep the record, and
            # leave the outcome for the sale that finishes it (professor M128).
            delete.assert_not_called(); close.assert_not_called()
    # A manual limit sell (e.g. above the market in TWS) is not cover.
    lmt = [{**working[0], "order_type": "LMT", "remaining": 10.0}]
    with patch("ibkr.get_client", return_value=client), \
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
    place.assert_called_once_with("AMD", 6.0, 150.0)


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
    place.assert_called_once_with("AMD", 10.0, 150.0)


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



def _queued_flat(intent, row, open_orders=(), stop=None, order_id=5):
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("ibkr.get_client", return_value=client), \
         patch("db.get_position_stop", return_value=stop), \
         patch("db.get_order_fill_state", **({"side_effect": row} if isinstance(row, Exception)
                                            else {"return_value": row})), \
         patch("ibkr.get_open_orders", return_value=list(open_orders)), \
         patch("ibkr.get_recent_sell_fill") as fill, \
         patch("db.close_trade_outcome") as close, patch("alerts._send") as alert:
        r = tasks._reprotect_after_late_sell("CRWD", {"intent": intent, "order_id": order_id}, 175.0,
                                             from_queue=True)
    fill.assert_not_called()   # never the execution-blend lookup
    alert.assert_not_called()
    return r, close


def test_queued_flat_exit_closes_at_its_orders_average_fill():
    """Professor M124: the part-filled EXIT deleted the stop record, so the
    stale-stop branch never closed the outcome."""
    r, close = _queued_flat("EXIT", {"status": "filled", "fill_price": 171.5, "broker_order_id": 77})
    assert r is None
    close.assert_called_once_with("CRWD", exit_price=171.5, exit_source="late_exit", exit_price_source="fill")


def test_queued_flat_exit_waits_while_its_remainder_works_or_a_lookup_fails():
    row = {"status": "partial", "fill_price": 171.5, "broker_order_id": 77}
    working = {"order_id": 77, "status": "Submitted"}
    for kw in ({"row": row, "open_orders": [working]}, {"row": RuntimeError("db")}):
        r, close = _queued_flat("BACKSTOP", **kw)
        assert r == "wait"
        close.assert_not_called()


def test_queued_flat_exit_leaves_mixed_and_owned_cases_alone():
    row = {"status": "partial", "fill_price": 171.5, "broker_order_id": 77}
    for kw in ({"row": row, "open_orders": [{"order_id": 77, "status": "Cancelled"}]},   # mixed close
               {"row": row, "stop": {"stop_price": 150.0}},                              # stale branch's
               {"row": row, "order_id": None}):                                          # old entry
        r, close = _queued_flat("EXIT", **kw)
        assert r is None
        close.assert_not_called()
    r, close = _queued_flat("TRIM", {"status": "filled", "fill_price": 171.5, "broker_order_id": 77})
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
