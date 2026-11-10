"""Stops for the shares a TRIM or manual sell leaves (audit 11-06 21:36 #4,
professor 62i): a part fill places the stop now on what the working rest of
the sell won't sell; a missing record alerts and opens a manual action; an
unreadable record queues the restore, carrying the pre-cancel floor."""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import execution


def _approval(action="TRIM", shares_initial=100, price_at_sizing=100.0, stop_loss_price=90.0):
    return {
        "id": 1,
        "ticker": "NVDA",
        "action": action,
        "conviction": 7,
        "thesis_bucket": "Accelerator",
        "stop_loss_price": stop_loss_price,
        "risk_json": {
            "shares_initial": shares_initial,
            "price_at_sizing": price_at_sizing,
            "stop_loss_price": stop_loss_price,
        },
    }



def _common(st, working):
    st.enter_context(patch("ibkr.get_open_orders", return_value=[]))
    st.enter_context(patch("tasks._working_mkt_sell_qty",
                           side_effect=working if isinstance(working, Exception) else None,
                           return_value=working))
    st.enter_context(patch("db.set_stop_broker_order_id"))
    queue = st.enter_context(patch("tasks._queue_reprotect"))
    place = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 9}))
    alert = st.enter_context(patch("alerts._send"))
    manual = st.enter_context(patch("tasks.open_manual_action"))
    return queue, place, alert, manual


def _trim(filled, held_after, stop_rec=None, stop_err=None, working=0.0):
    approval = _approval(action="TRIM", shares_initial=30)
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=90.0)
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=held_after)]
    stop_mock = MagicMock(side_effect=stop_err) if stop_err else MagicMock(return_value=stop_rec)
    with ExitStack() as st:
        for p in (patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client),
                  patch("ibkr.cancel_stop_orders_for", return_value=0),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=None),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("db.log_signal"),
                  patch("config.USE_BRACKET_ORDERS", False),
                  patch("db.log_trade"), patch("db.log_order", return_value=1),
                  patch("db.update_order_fill"), patch("db.delete_position_stop"),
                  patch("db.log_trade_outcome"), patch("db.close_trade_outcome"),
                  patch("alerts.send_trade_alert"), patch("db.get_position_stop", stop_mock)):
            st.enter_context(p)
        order = st.enter_context(patch("ibkr.place_market_order"))
        order.return_value = {"order_id": 1, "fill_price": 100.0, "filled_qty": filled}
        queue, place, alert, manual = _common(st, working)
        execution.execute_approved(approval)
    return queue, place, alert, manual


def test_part_filled_trim_stops_now_what_the_working_sell_wont_sell():
    with patch("risk_manager._floored_level", return_value=90.0):
        queue, place, alert, _ = _trim(filled=20.0, held_after=70.0, stop_rec={"stop_price": 90.0},
                                       working=10.0)
    assert place.call_args.args[1:3] == (60.0, 90.0)
    queue.assert_not_called()


def test_part_filled_trim_covered_by_the_working_sell_places_nothing():
    queue, place, _, manual = _trim(filled=20.0, held_after=70.0, stop_rec={"stop_price": 90.0},
                                    working=70.0)
    place.assert_not_called()
    queue.assert_not_called()
    manual.assert_not_called()


def test_part_fill_with_an_unreadable_open_order_read_subtracts_the_known_working_rest():
    """Professor M321: stopping all 70 would leave the account short once the
    10 still working fill; subtract them and queue a RESTORE."""
    with patch("risk_manager._floored_level", return_value=90.0):
        queue, place, _, _ = _trim(filled=20.0, held_after=70.0, stop_rec={"stop_price": 90.0},
                                   working=RuntimeError("timeout"))
    assert place.call_args.args[1] == 60.0     # 30 ordered - 20 filled = 10 working
    assert queue.call_args.args[1]["intent"] == "RESTORE"


def test_missing_record_alerts_and_opens_a_manual_action_now():
    queue, place, alert, manual = _trim(filled=30.0, held_after=60.0, stop_rec=None)
    place.assert_not_called()
    queue.assert_not_called()
    assert any("no stop on record. Place one in TWS" in c.args[0] for c in alert.call_args_list)
    assert manual.call_args.args[0] == "NVDA"


def test_unreadable_record_queues_the_restore_with_the_floor():
    queue, place, alert, _ = _trim(filled=30.0, held_after=60.0, stop_err=RuntimeError("pool"))
    place.assert_not_called()
    assert queue.call_args.args[1]["intent"] == "RESTORE" and "floor" in queue.call_args.args[1]
    assert any("TRIM STOP RECORD UNREADABLE" in c.args[0] for c in alert.call_args_list)


def test_full_fill_with_a_record_still_replaces_the_stop_at_once():
    with patch("risk_manager._floored_level", return_value=90.0):
        queue, place, alert, _ = _trim(filled=30.0, held_after=60.0, stop_rec={"stop_price": 90.0})
    assert place.call_args.args[1] == 60.0
    queue.assert_not_called()


def _manual(filled, stop_mock, working=0.0):
    with ExitStack() as st:
        client = st.enter_context(patch("ibkr.get_client"))
        client.return_value.get_open_position.return_value = MagicMock(qty=10)
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=10 - filled)]
        for p in (patch("ibkr.get_ib"), patch("tasks.manage_cancel_stop_orders_for", return_value=1),
                  patch("execution._stops_still_resting", return_value=0),
                  patch("db.get_state", return_value=None), patch("db.log_trade"),
                  patch("db.log_order", return_value=1), patch("db.update_order_fill", return_value=True),
                  patch("db.log_signal"), patch("db.close_trade_outcome"),
                  patch("db.get_position_stop", stop_mock), patch("feed.emit"),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=None),
                  patch("market_data.is_market_hours", return_value=True)):
            st.enter_context(p)
        order = st.enter_context(patch("ibkr.place_market_order"))
        order.return_value = {"order_id": 1, "fill_price": 100.0, "filled_qty": filled}
        queue, place, alert, manual = _common(st, working)
        execution.execute_manual("NVDA", "SELL", 5, "test")
    return queue, place, alert, manual


def test_manual_sell_with_an_unreadable_record_queues_the_stop():
    queue, place, alert, _ = _manual(5.0, MagicMock(side_effect=RuntimeError("pool")))
    place.assert_not_called()
    assert queue.call_args.args[1]["intent"] == "RESTORE"
    assert any("MANUAL SELL STOP RECORD UNREADABLE" in c.args[0] for c in alert.call_args_list)


def test_manual_sell_part_fill_stops_the_unworked_remainder_now():
    with patch("risk_manager._floored_level", return_value=90.0):
        queue, place, _, _ = _manual(3.0, MagicMock(return_value={"stop_price": 90.0}), working=2.0)
    assert place.call_args.args[1] == 5.0
    queue.assert_not_called()


def test_manual_sell_with_no_record_opens_a_manual_action():
    queue, place, alert, manual = _manual(5.0, MagicMock(return_value=None))
    place.assert_not_called()
    queue.assert_not_called()
    assert manual.call_args.args[0] == "NVDA"
    assert any("no stop on record" in c.args[0] for c in alert.call_args_list)


def test_queued_entry_carries_the_floor():
    import tasks
    seen = {}
    with patch("tasks._reprotect_queue_edit", side_effect=lambda fn: seen.update(fn({}))):
        tasks._queue_reprotect("NVDA", {"intent": "RESTORE", "floor": 95.0}, None)
        assert seen["NVDA"]["floor"] == 95.0
        tasks._queue_reprotect("AMD", {"intent": "RESTORE", "floor": None}, None)
        assert "floor" not in seen["AMD"]


def _queued_restore(floor, latest):
    import tasks
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=20)]
    kw = {"side_effect": latest} if isinstance(latest, Exception) else {"return_value": latest}
    with patch("ibkr.get_client", return_value=client), patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price", **kw), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        assert tasks._reprotect_after_late_sell("NVDA", {"intent": "RESTORE", "floor": floor}, None) is True
    return place


def test_queued_restore_raises_the_record_to_the_floor_with_a_fresh_price():
    place = _queued_restore(95.0, 100.0)
    place.assert_called_once_with("NVDA", 20.0, 95.0, market_price=100.0)


def test_queued_restore_skips_a_floor_at_or_above_the_market():
    # After a drop the pre-cancel stop sits above the price: placing it there
    # fills at once (professor 62j).
    place = _queued_restore(95.0, 93.0)
    place.assert_called_once_with("NVDA", 20.0, 90.0, market_price=93.0)


def test_queued_restore_skips_the_floor_without_a_fresh_price():
    for latest in (None, RuntimeError("no bars")):
        place = _queued_restore(95.0, latest)
        place.assert_called_once_with("NVDA", 20.0, 90.0)


def test_a_restore_merged_into_a_queued_target_keeps_the_higher_floor():
    import tasks
    pending = {"NVDA": {"intent": "PROFIT_TARGET", "stop_price": 92.0, "floor": 94.0}}
    with patch("tasks._reprotect_queue_edit", side_effect=lambda fn: pending.update(fn(pending))):
        tasks._queue_reprotect("NVDA", {"intent": "RESTORE", "floor": 96.0}, None)
        assert pending["NVDA"]["intent"] == "PROFIT_TARGET" and pending["NVDA"]["floor"] == 96.0
        tasks._queue_reprotect("NVDA", {"intent": "RESTORE", "floor": 93.0}, None)
        assert pending["NVDA"]["floor"] == 96.0


def test_queued_target_stop_is_floored_after_the_target_price():
    import tasks
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=20)]
    with patch("ibkr.get_client", return_value=client), patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.mark_profit_target_taken") as mark, \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        tasks._reprotect_after_late_sell("NVDA", {"intent": "PROFIT_TARGET", "stop_price": 92.0,
                                                  "floor": 96.0}, None)
    mark.assert_called_once_with("NVDA", 92.0)
    assert place.call_args.args[2] == 96.0


def test_a_floored_queued_stop_records_the_placed_price():
    import tasks
    with patch("db.update_trailing_stop") as upd:
        place = _queued_restore_rec(95.0, 100.0)
    assert place.call_args.args[2] == 95.0
    upd.assert_called_once_with("NVDA", 95.0, 99.0, native_trail=False, strict=True)
    with patch("db.update_trailing_stop") as upd:
        _queued_restore_rec(95.0, 93.0)   # floor skipped: record untouched
    upd.assert_not_called()


def _queued_restore_rec(floor, latest):
    import tasks
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=20)]
    with patch("ibkr.get_client", return_value=client), patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price", return_value=latest), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0, "trailing_high": 99.0}), \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"):
        assert tasks._reprotect_after_late_sell("NVDA", {"intent": "RESTORE", "floor": floor}, None) is True
    return place


def _seeded_restore(seed, latest):
    import tasks
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=20)]
    kw = {"side_effect": latest} if isinstance(latest, Exception) else {"return_value": latest}
    with patch("ibkr.get_client", return_value=client), patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price", **kw), patch("db.get_position_stop", return_value=None), \
         patch("execution._stops_still_resting", return_value=0.0), \
         patch("tasks.manage_cancel_stop_orders_for"), patch("time.sleep"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"), patch("db.log_activity_direct") as act:
        ok = tasks._reprotect_after_late_sell("NVDA", {"intent": "RESTORE", "seed_stop": seed}, None)
    place.activity = act
    return ok, place


def test_seeded_restore_places_the_pre_cancel_stop_under_a_fresh_price():
    """m606: no record, the stop that rested before the sell is put back."""
    ok, place = _seeded_restore(95.0, 100.0)
    assert ok is True and place.call_args.args[1:3] == (20.0, 95.0)
    stage, ticker, _, _, extra = place.activity.call_args.args
    assert (stage, ticker, extra["order_id"], extra["stop_price"]) == ("seeded_stop", "NVDA", 3, 95.0)


def test_seeded_restore_waits_at_or_above_the_market_or_without_a_price():
    for latest in (94.0, None, RuntimeError("no bars")):
        ok, place = _seeded_restore(95.0, latest)
        assert ok is False, latest
        place.assert_not_called()


def test_missing_record_with_a_known_floor_queues_the_seeded_restore():
    import execution
    with patch("tasks._queue_reprotect") as q, patch("alerts._send") as a, \
         patch("tasks.open_manual_action") as m:
        execution._alert_no_stop_record("NVDA", "A TRIM", 60.0, 95.0)
    assert q.call_args.args[1] == {"intent": "RESTORE", "seed_stop": 95.0}
    assert "is queued" in a.call_args.args[0] and m.called
    with patch("tasks._queue_reprotect") as q, patch("alerts._send"), patch("tasks.open_manual_action"):
        execution._alert_no_stop_record("NVDA", "A TRIM", 60.0, None)
    q.assert_not_called()


def test_queued_entry_carries_the_seed_stop():
    import tasks
    seen = {}
    with patch("tasks._reprotect_queue_edit", side_effect=lambda fn: seen.update(fn({}))):
        tasks._queue_reprotect("NVDA", {"intent": "RESTORE", "seed_stop": 95.0}, None)
    assert seen["NVDA"]["seed_stop"] == 95.0


def test_seed_stop_survives_a_merge_into_a_queued_target():
    import tasks
    pending = {"NVDA": {"intent": "PROFIT_TARGET", "stop_price": 92.0}}
    with patch("tasks._reprotect_queue_edit", side_effect=lambda fn: pending.update(fn(pending))):
        tasks._queue_reprotect("NVDA", {"intent": "RESTORE", "seed_stop": 95.0}, None)
    assert pending["NVDA"]["intent"] == "PROFIT_TARGET" and pending["NVDA"]["seed_stop"] == 95.0


def test_scan_summary_counts_pass_from_the_disqualified_list():
    import tasks
    with patch("agents.scanner._regime_threshold", return_value=65):
        dq = [{"priority": "PASS"}] * 18 + [{"reason": "No price data"}] * 2
        line = tasks._scan_summary({"regime": "Neutral", "score": 50}, [{"priority": "MEDIUM"}], dq)
    assert "PASS 18 skipped 2" in line
