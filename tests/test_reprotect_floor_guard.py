"""tasks._reprotect_after_late_sell re-places one stop at the record's level
after cancelling ours: the queued retry (an ADD's own stop that failed, a
late ADD fill) and a late SELL fill. That stop is floored (live) at the
highest resting stop read before the cancel, a resting stop at or above the
price not counting (risk_manager._floor_at_resting_stop). The above-market
guard (risk_manager._stop_below_market) runs in config.STOP_GUARD_MODE:
shadow by default (a row, the placement unchanged), enforce in the tests
marked so.

Prices are scaled as in test_merge_after_lagging_add: L197 = 89.3 (lot 197's
own stop), the 11-10 ADD filled 1.8% above it with its own stop L198 10.7%
below L197; the 11-13 ADD filled 0.6% below L197."""
from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

import pytest

import config
import tasks


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(config, "STOP_GUARD_MODE", "enforce")

L197 = 89.3
ADD_FILL = round(L197 * 1.018, 2)
L198 = round(L197 * (1 - 0.107), 2)
FILL_1113 = round(L197 * (1 - 0.006), 2)


def _stp(price, qty=1.0, status="Submitted"):
    return {"ticker": "NVDA", "action": "SELL", "order_type": "STP", "qty": qty,
            "stop_price": price, "status": status, "order_id": 11}


def _run(order, record_stop, held, covered, before_cancel, price, from_queue=True,
         after_cancel=(), fill_price=None, fill_at=None, rth=True, guard_off=False):
    client = MagicMock()
    guard = (patch("risk_manager._stop_below_market", side_effect=lambda t, level, *a, **k: level)
             if guard_off else nullcontext())
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=held)]
    cancelled = []          # reads before the cancel see before_cancel, after it after_cancel
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price", return_value=price), \
         patch("ibkr.get_fresh_minute_price", return_value=price) as price_read, \
         patch("db.get_position_stop", return_value={"stop_price": record_stop, "trailing_high": None}), \
         patch("execution._stops_still_resting", return_value=covered), \
         patch("tasks._resting_sell_stops",
               side_effect=lambda *a, **k: list(after_cancel if cancelled else before_cancel)), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for",
               side_effect=lambda *a, **k: cancelled.append(1)) as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"), \
         patch("db.update_trailing_stop") as record, \
         patch("db.log_activity_queued") as queued, \
         patch("market_data.is_market_hours", return_value=rth), guard, \
         patch("alerts._send") as send, patch("time.sleep"):
        out = tasks._reprotect_after_late_sell("NVDA", order, fill_price, from_queue=from_queue,
                                               exit_fill_at=fill_at)
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "stop_above_market_guard"]
    alerts = [c.args[0] for c in send.call_args_list if "🚨" in c.args[0]]
    _run.price_read = price_read
    _run.info = [c.args[0] for c in send.call_args_list if "stop guard (shadow)" in c.args[0]]
    return out, cancel, place, record, rows, alerts


def test_171_failed_add_stop_queued_restop_keeps_lot_197_at_its_stop():
    """11-10: the ADD's own stop failed 3 times (queued as intent BUY); the
    record was rewritten at L198 while lot 197's L197 stop rests. The queue
    re-stops both shares: at L197, not loosened to L198."""
    out, cancel, place, record, rows, alerts = _run(
        {"intent": "BUY"}, L198, held=2, covered=1.0, before_cancel=[_stp(L197)], price=ADD_FILL)
    assert out is True
    cancel.assert_called_once_with("NVDA")
    assert place.call_args_list == [call("NVDA", 2.0, L197, market_price=ADD_FILL)]
    assert L197 < ADD_FILL                                 # nothing at or above the price
    record.assert_called_once_with("NVDA", L197, None, native_trail=False, strict=True)
    assert rows == [] and alerts == []


def test_171_queued_restop_after_a_drop_below_lot_197s_stop_stays_below_the_price():
    """11-12: the price fell under L197 before the retry. The resting L197 is
    at/above the price, so it is not a floor; L198 goes on."""
    price = round(L197 * 0.99, 2)
    out, _, place, record, rows, _ = _run(
        {"intent": "BUY"}, L198, held=2, covered=1.0, before_cancel=[_stp(L197)], price=price)
    assert place.call_args_list == [call("NVDA", 2.0, L198)] and L198 < price
    record.assert_not_called()
    assert rows == []


def test_outside_regular_hours_the_floor_applies_without_a_price():
    """No minute-bar read outside RTH: the floor applies with no market check
    (its no-price behaviour), so a resting L197 is kept."""
    price = round(L197 * 0.99, 2)
    _, _, place, _, rows, _ = _run(
        {"intent": "BUY"}, L198, held=2, covered=1.0, before_cancel=[_stp(L197)], price=price,
        rth=False)
    _run.price_read.assert_not_called()
    assert place.call_args_list == [call("NVDA", 2.0, L197)]
    assert rows[0]["outcome"] == "would_price_unread"


@pytest.mark.usefixtures("enforce")
def test_late_add_fill_with_a_kept_record_above_the_price_goes_below_it():
    """A late ADD fill (reconcile_unconfirmed_orders calls this with {}) keeps
    the record at L197 after lot 197's stop filled; the remaining share's L198
    rests; the fill is 0.6% under L197. Placed at L198, not L197."""
    out, cancel, place, record, rows, alerts = _run(
        {}, L197, held=2, covered=1.0, before_cancel=[_stp(L198)], price=FILL_1113)
    assert out is True
    cancel.assert_called_once_with("NVDA")
    assert place.call_args_list == [call("NVDA", 2.0, L198, market_price=FILL_1113)]
    assert L198 < FILL_1113
    record.assert_not_called()                 # a minute-bar price: the record stays higher
    assert rows[0]["record"] == "kept_higher" and "still shows" in alerts[0]
    assert len(rows) == 1 and rows[0]["outcome"] == "placed_below_market"
    assert rows[0]["path"] == "queued_reprotect"
    assert len(alerts) == 1 and "REFUSED" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_queued_restop_with_nothing_below_the_price_places_as_before_and_alerts():
    out, _, place, _, rows, alerts = _run(
        {}, L197, held=1, covered=0.0, before_cancel=[], price=FILL_1113)
    assert place.call_args_list == [call("NVDA", 1.0, L197)]   # today's behaviour
    assert rows[0]["outcome"] == "no_level_below_market"
    assert len(alerts) == 1 and "may sell at once" in alerts[0]


def test_late_partial_sell_after_a_lagging_add_keeps_lot_197_at_its_stop():
    """A late TRIM fill (not queued): lot 197's L197 stop still rests over
    a record at L198. Its re-place is floored at L197."""
    out, cancel, place, record, rows, _ = _run(
        {"intent": "TRIM", "side": "SELL"}, L198, held=2, covered=1.0,
        before_cancel=[_stp(L197)], price=ADD_FILL, from_queue=False)
    assert out is True
    cancel.assert_called_once_with("NVDA")
    assert place.call_args_list == [call("NVDA", 2.0, L197, market_price=ADD_FILL)]
    record.assert_called_once()
    assert rows == []


def test_no_floor_or_guard_leaves_the_placement_unchanged():
    for from_queue, covered in ((True, 0.0), (True, 1.0), (False, 0.0)):
        _, _, place, record, rows, alerts = _run(
            {"intent": "BUY"} if from_queue else {"intent": "TRIM"}, 150.0, held=2,
            covered=covered, before_cancel=[_stp(150.0)] if covered else [], price=200.0,
            from_queue=from_queue)
        assert place.call_args_list == [call("NVDA", 2.0, 150.0)]
        record.assert_not_called()
        assert rows == [] and alerts == []


def test_unreadable_price_places_as_before_and_records_it():
    _, _, place, _, rows, alerts = _run(
        {}, L197, held=2, covered=1.0, before_cancel=[_stp(L198)], price=None)
    assert place.call_args_list == [call("NVDA", 2.0, L197)]
    assert rows[0]["outcome"] == "would_price_unread" and alerts == []


def test_already_covered_reads_nothing_more():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=2)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price") as price, \
         patch("db.get_position_stop", return_value={"stop_price": L198}), \
         patch("execution._stops_still_resting", return_value=2.0), \
         patch("tasks._resting_sell_stops") as reads, \
         patch("tasks.manage_stop_place") as place:
        assert tasks._reprotect_after_late_sell("NVDA", {"intent": "BUY"}, None, from_queue=True) is True
    price.assert_not_called()
    reads.assert_not_called()
    place.assert_not_called()


# ── the queued BUY carries its new shares' own stop as the guard's fallback ──

L199 = round(L198 * (1 - 0.062), 2)


@pytest.mark.usefixtures("enforce")
def test_queued_buy_with_a_kept_record_above_the_price_places_the_carried_stop():
    out, _, place, record, rows, alerts = _run(
        {"intent": "BUY", "own_stop": L199}, L197, held=2, covered=0.0, before_cancel=[],
        price=FILL_1113)
    assert out is True
    assert place.call_args_list == [call("NVDA", 2.0, L199, market_price=FILL_1113)]
    assert L199 < FILL_1113
    record.assert_not_called()
    assert len(rows) == 1 and rows[0]["outcome"] == "placed_below_market"
    assert rows[0]["fallback_vs_price_pct"] == round((L199 / FILL_1113 - 1) * 100, 3)
    assert len(alerts) == 1 and "REFUSED" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_queued_buy_takes_a_higher_resting_stop_over_the_carried_one():
    _, _, place, _, rows, _ = _run(
        {"intent": "BUY", "own_stop": L199}, L197, held=2, covered=1.0,
        before_cancel=[_stp(L198)], price=FILL_1113)
    assert place.call_args_list == [call("NVDA", 2.0, L198, market_price=FILL_1113)]
    assert rows[0]["outcome"] == "placed_below_market"


@pytest.mark.usefixtures("enforce")
def test_legacy_queued_buy_without_the_field_is_unchanged():
    _, _, place, _, rows, alerts = _run(
        {"intent": "BUY"}, L197, held=2, covered=0.0, before_cancel=[], price=FILL_1113)
    assert place.call_args_list == [call("NVDA", 2.0, L197)]
    assert rows[0]["outcome"] == "no_level_below_market" and rows[0]["fallback_vs_price_pct"] is None
    assert len(alerts) == 1 and "may sell at once" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_carried_stop_at_or_above_the_price_falls_to_the_resting_search():
    above = round(FILL_1113 * 1.001, 2)
    _, _, place, _, rows, _ = _run(
        {"intent": "BUY", "own_stop": above}, L197, held=2, covered=1.0,
        before_cancel=[_stp(L198)], price=FILL_1113)
    assert place.call_args_list == [call("NVDA", 2.0, L198, market_price=FILL_1113)]
    assert rows[0]["outcome"] == "placed_below_market" and rows[0]["n_resting_below"] == 1
    _, _, place, _, rows, alerts = _run(
        {"intent": "BUY", "own_stop": above}, L197, held=2, covered=0.0, before_cancel=[],
        price=FILL_1113)
    assert place.call_args_list == [call("NVDA", 2.0, L197)]
    assert rows[0]["outcome"] == "no_level_below_market" and len(alerts) == 1


def _queue(order, pending=None):
    store = {"p": dict(pending or {})}

    def edit(fn):
        store["p"] = fn(dict(store["p"]))
    with patch("tasks._reprotect_queue_edit", side_effect=edit):
        tasks._queue_reprotect("NVDA", order, None)
    return store["p"]["NVDA"]


def test_queue_entry_carries_the_own_stop_and_legacy_has_none():
    assert _queue({"intent": "BUY", "own_stop": L199})["own_stop"] == L199
    assert "own_stop" not in _queue({"intent": "BUY"})


def test_queued_buy_over_a_profit_target_entry_keeps_the_own_stop():
    pt = {"NVDA": {"intent": "PROFIT_TARGET", "stop_price": 120.0, "nonce": "x", "tries": 2}}
    merged = _queue({"intent": "BUY", "own_stop": L199}, pending=pt)
    assert merged["intent"] == "PROFIT_TARGET" and merged["stop_price"] == 120.0
    assert merged["own_stop"] == L199 and merged["had_buy"] is True


def test_queue_stop_for_retry_passes_the_own_stop_only_when_given():
    import execution
    with patch("tasks._queue_reprotect") as q, patch("alerts._send"):
        execution._queue_stop_for_retry("NVDA", 1, 90.0, alert=False, own_stop=L199)
        execution._queue_stop_for_retry("NVDA", 1, 90.0, alert=False)
    assert q.call_args_list == [call("NVDA", {"intent": "BUY", "own_stop": L199}, None),
                                call("NVDA", {"intent": "BUY"}, None)]


# ── the guard's price: this order's recent fill, else a bounded minute bar ──

def _ago(seconds):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


@pytest.mark.usefixtures("enforce")
def test_recent_fill_price_is_used_without_a_read_and_the_record_follows():
    """A late TRIM fill 0.6% under the kept L197, filled a minute ago: the
    guard uses the fill (no price read), places L198 and moves the record."""
    out, _, place, record, rows, alerts = _run(
        {"intent": "TRIM", "side": "SELL"}, L197, held=2, covered=1.0,
        before_cancel=[_stp(L198)], price=None, from_queue=False,
        fill_price=FILL_1113, fill_at=_ago(60))
    _run.price_read.assert_not_called()
    assert place.call_args_list == [call("NVDA", 2.0, L198, market_price=FILL_1113)]
    record.assert_called_once_with("NVDA", L198, None, native_trail=False, strict=True)
    assert rows[0]["record"] == "moved_to_placed" and "is moved to" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_old_fill_price_falls_to_the_minute_bar_and_the_record_stays():
    _, _, place, record, rows, alerts = _run(
        {"intent": "TRIM", "side": "SELL"}, L197, held=2, covered=1.0,
        before_cancel=[_stp(L198)], price=FILL_1113, from_queue=False,
        fill_price=FILL_1113, fill_at=_ago(3600))
    _run.price_read.assert_called_once()
    kw = _run.price_read.call_args.kwargs
    assert kw["timeout_s"] == tasks._GUARD_PRICE_TIMEOUT_S and kw["max_age_s"] == tasks._GUARD_PRICE_MAX_AGE_S
    assert place.call_args_list == [call("NVDA", 2.0, L198, market_price=FILL_1113)]
    record.assert_not_called()
    assert rows[0]["record"] == "kept_higher"


@pytest.mark.usefixtures("enforce")
def test_no_fresh_price_places_as_before():
    _, _, place, record, rows, alerts = _run(
        {"intent": "TRIM", "side": "SELL"}, L197, held=2, covered=1.0,
        before_cancel=[_stp(L198)], price=None, from_queue=False,
        fill_price=FILL_1113, fill_at=_ago(3600))
    assert place.call_args_list == [call("NVDA", 2.0, L197)]
    record.assert_not_called()
    assert rows[0]["outcome"] == "price_unread" and alerts == []


def test_guard_price_sources():
    with patch("market_data.is_market_hours", return_value=False), \
         patch("ibkr.get_fresh_minute_price", return_value=88.0) as read:
        assert tasks._guard_price("NVDA", 90.0, _ago(10)) == (90.0, "fill")   # a fill, any hour
        assert tasks._guard_price("NVDA", None, None) == (None, "outside_rth")
        read.assert_not_called()
    with patch("market_data.is_market_hours", return_value=True), \
         patch("ibkr.get_fresh_minute_price", return_value=None) as read:
        assert tasks._guard_price("NVDA", 90.0, _ago(10)) == (90.0, "fill")
        read.assert_not_called()
        assert tasks._guard_price("NVDA", 90.0, None) == (None, "unread")
        assert tasks._guard_price("NVDA", 90.0, "not a time") == (None, "unread")
    with patch("market_data.is_market_hours", return_value=True):
        with patch("ibkr.get_fresh_minute_price", return_value=88.0):
            assert tasks._guard_price("NVDA", None, None) == (88.0, "minute_bar")
        with patch("ibkr.get_fresh_minute_price", side_effect=RuntimeError("x")):
            assert tasks._guard_price("NVDA", None, None) == (None, "unread")


# ── queue merges keep a BUY's own stop ───────────────────────────────────────

def test_restore_or_bare_buy_replacing_a_buy_keeps_its_own_stop():
    buy = {"NVDA": {"intent": "BUY", "own_stop": L199, "nonce": "x"}}
    assert _queue({"intent": "RESTORE", "floor": 80.0}, pending=buy)["own_stop"] == L199
    assert _queue({"intent": "TRIM"}, pending=buy)["own_stop"] == L199
    assert _queue({"intent": "BUY"}, pending=buy)["own_stop"] == L199
    assert _queue({"intent": "BUY", "own_stop": 70.0}, pending=buy)["own_stop"] == 70.0
    assert "own_stop" not in _queue({"intent": "RESTORE"}, pending={})


def test_floored_record_write_skipped_without_a_record():
    with patch("db.update_trailing_stop") as write:
        tasks._record_floored_stop("NVDA", 95.0, 90.0, {})
        write.assert_not_called()
        tasks._record_floored_stop("NVDA", 95.0, 90.0, {"stop_price": 90.0, "trailing_high": None})
        write.assert_called_once()


# ── shadow (the default): the placement and the record are as with no guard ──

SHADOW_CASES = {
    "late_add_kept_record_above": (dict(order={}, record_stop=L197, held=2, covered=1.0,
                                        before_cancel=[_stp(L198)], price=FILL_1113),
                                   "would_placed_below_market"),
    "nothing_below": (dict(order={}, record_stop=L197, held=1, covered=0.0, before_cancel=[],
                           price=FILL_1113), "would_no_level_below_market"),
    "carried_own_stop": (dict(order={"intent": "BUY", "own_stop": L199}, record_stop=L197, held=2,
                              covered=0.0, before_cancel=[], price=FILL_1113),
                         "would_placed_below_market"),
    "recent_fill": (dict(order={"intent": "TRIM", "side": "SELL"}, record_stop=L197, held=2,
                         covered=1.0, before_cancel=[_stp(L198)], price=None, from_queue=False,
                         fill_price=FILL_1113, fill_at=_ago(60)), "would_placed_below_market"),
    "queued_floor_171": (dict(order={"intent": "BUY"}, record_stop=L198, held=2, covered=1.0,
                              before_cancel=[_stp(L197)], price=ADD_FILL), None),
    "price_unread": (dict(order={}, record_stop=L197, held=2, covered=1.0,
                          before_cancel=[_stp(L198)], price=None), "would_price_unread"),
}


@pytest.mark.parametrize("name", list(SHADOW_CASES))
def test_shadow_places_exactly_as_with_no_guard(name):
    kw, outcome = SHADOW_CASES[name]
    _, _, off_place, off_record, _, _ = _run(**kw, guard_off=True)
    out, _, place, record, rows, alerts = _run(**kw)
    assert out is True
    assert place.call_args_list == off_place.call_args_list        # identical manage_stop_place args
    assert record.call_args_list == off_record.call_args_list      # no record change from the guard
    assert alerts == []
    if outcome is None:
        assert rows == [] and _run.info == []
    else:
        assert len(rows) == 1 and rows[0]["outcome"] == outcome and rows[0]["mode"] == "shadow"
        assert len(_run.info) == (0 if outcome == "would_price_unread" else 1)
