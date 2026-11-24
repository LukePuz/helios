"""The above-market guard (risk_manager._stop_below_market) on the stop
re-places that use the record's level: the TRIM re-place and the manual-sell
re-place (execution), and the reconcile resize (tasks._fix_overstop and
_fix_refused_over_position). In config.STOP_GUARD_MODE "shadow" (the
default) the guard writes its row and an info alert and the manage_stop_place
args are identical to a run with no guard. _fix_overstop's own refusal of a
recorded stop not below the last price runs first and unchanged."""
from contextlib import ExitStack, nullcontext
from unittest.mock import MagicMock, patch

import pytest

import config
import execution
import risk_manager
import tasks

FILL = 100.0


def _stp(price, qty=1.0, status="Submitted", oid=11):
    return {"ticker": "NVDA", "action": "SELL", "order_type": "STP", "qty": qty,
            "stop_price": price, "status": status, "order_id": oid}


def _approval():
    return {"id": 1, "ticker": "NVDA", "action": "TRIM", "conviction": 7,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 30, "price_at_sizing": FILL, "stop_loss_price": 90.0}}


def _guard_patches(st, resting, guard_off, rth, sold, events, place_err=None):
    """sold: a list the sell's order mock appends to; the resting read shows
    `resting` only after the sell (before it, the sell's own pre-sell check
    must see no stop left). events: the order of the broker reads after the
    sell, the place and the alerts."""
    if guard_off:
        st.enter_context(patch("risk_manager._stop_below_market",
                               side_effect=lambda t, level, *a, **k: level))
    else:
        st.enter_context(nullcontext())

    def resting_read(*a, **k):
        if sold:
            events.append("resting_read")
            return list(resting)
        return []

    def open_orders(*a, **k):
        if sold:
            events.append("open_orders")
        return []
    st.enter_context(patch("tasks._resting_sell_stops", side_effect=resting_read))
    st.enter_context(patch("ibkr.get_open_orders", side_effect=open_orders))
    st.enter_context(patch("market_data.is_market_hours", return_value=rth))
    queued = st.enter_context(patch("db.log_activity_queued"))
    send = st.enter_context(patch("alerts._send", side_effect=lambda text, *a, **k:
                                  events.append(("alert", text))))
    def do_place(*a, **k):
        events.append("place")
        if place_err:
            raise place_err
        return {"order_id": 9}
    place = st.enter_context(patch("tasks.manage_stop_place", side_effect=do_place))
    return queued, send, place


def _out(place, queued, send, record, events):
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "stop_above_market_guard"]
    info = [c.args[0] for c in send.call_args_list if "stop guard (shadow)" in c.args[0]]
    other = [c.args[0] for c in send.call_args_list if "stop guard (shadow)" not in c.args[0]]
    return {"place": place.call_args_list, "rows": rows, "info": info, "alerts": other,
            "record": record.call_args_list if record is not None else None,
            "events": [e if isinstance(e, str) else ("info" if "stop guard (shadow)" in e[1]
                                                     else "alert") for e in events]}


def _sell_order(sold, fill_price, filled_qty):
    return lambda *a, **k: (sold.append(1), {"order_id": 1, "fill_price": fill_price,
                                             "filled_qty": filled_qty})[1]


def _trim(record_stop, resting=(), fill_price=FILL, guard_off=False, rth=True, floor=None,
          place_err=None):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=90.0)
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=60.0)]
    sold, events = [], []
    with ExitStack() as st:
        for p in (patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client),
                  patch("ibkr.cancel_stop_orders_for", return_value=0),
                  patch("ibkr.get_latest_price", return_value=FILL),
                  patch("execution._floor_before_cancel", return_value=floor),
                  patch("market_data.get_priced_quote", return_value=None),
                  patch("db.log_signal"), patch("config.USE_BRACKET_ORDERS", False),
                  patch("db.log_trade"), patch("db.log_order", return_value=1),
                  patch("db.update_order_fill"), patch("db.delete_position_stop"),
                  patch("db.log_trade_outcome"), patch("db.close_trade_outcome"),
                  patch("alerts.send_trade_alert"), patch("db.set_stop_broker_order_id"),
                  patch("tasks._working_mkt_sell_qty", return_value=0.0),
                  patch("tasks._queue_reprotect"), patch("tasks.open_manual_action"),
                  patch("db.get_position_stop", return_value={"stop_price": record_stop})):
            st.enter_context(p)
        queued, send, place = _guard_patches(st, resting, guard_off, rth, sold, events,
                                              place_err)
        record = st.enter_context(patch("db.update_trailing_stop"))
        order = st.enter_context(patch("ibkr.place_market_order"))
        order.side_effect = _sell_order(sold, fill_price, 30.0)
        execution.execute_approved(_approval())
    return _out(place, queued, send, record, events)


def _manual(record_stop, resting=(), fill_price=FILL, guard_off=False, rth=True, floor=None,
          place_err=None):
    sold, events = [], []
    with ExitStack() as st:
        client = st.enter_context(patch("ibkr.get_client"))
        client.return_value.get_open_position.return_value = MagicMock(qty=10)
        client.return_value.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=5)]
        for p in (patch("ibkr.get_ib"), patch("tasks.manage_cancel_stop_orders_for", return_value=1),
                  patch("execution._stops_still_resting", return_value=0),
                  patch("execution._floor_before_cancel", return_value=floor),
                  patch("db.get_state", return_value=None), patch("db.log_trade"),
                  patch("db.log_order", return_value=1), patch("db.update_order_fill", return_value=True),
                  patch("db.log_signal"), patch("db.close_trade_outcome"), patch("feed.emit"),
                  patch("db.get_position_stop", return_value={"stop_price": record_stop}),
                  patch("ibkr.get_latest_price", return_value=FILL),
                  patch("market_data.get_priced_quote", return_value=None),
                  patch("db.set_stop_broker_order_id"),
                  patch("tasks._working_mkt_sell_qty", return_value=0.0),
                  patch("tasks._queue_reprotect"), patch("tasks.open_manual_action")):
            st.enter_context(p)
        queued, send, place = _guard_patches(st, resting, guard_off, rth, sold, events,
                                              place_err)
        record = st.enter_context(patch("db.update_trailing_stop"))
        order = st.enter_context(patch("ibkr.place_market_order"))
        order.side_effect = _sell_order(sold, fill_price, 5.0)
        execution.execute_manual("NVDA", "SELL", 5, "test")
    return _out(place, queued, send, record, events)


SELL_CASES = {
    # record kept above the fill; the stop the sell cancelled is below it
    "above_with_floor_below": (dict(record_stop=105.0, floor=95.0),
                               "would_placed_below_market", True),
    # a resting stop after the cancel is not read on these paths
    "above_resting_not_read": (dict(record_stop=105.0, resting=[_stp(95.0)]),
                               "would_no_level_below_market", True),
    "above_nothing_below": (dict(record_stop=105.0), "would_no_level_below_market", True),
    "at_the_price": (dict(record_stop=FILL), "would_no_level_below_market", True),
    # no fill price: the quote stands in, so the stale check runs in RTH
    "above_quote_price": (dict(record_stop=105.0, floor=95.0, fill_price=None),
                          "would_placed_below_market", False),
    "below_the_price": (dict(record_stop=90.0), None, None),
}


@pytest.mark.parametrize("run", [_trim, _manual], ids=["trim", "manual_sell"])
@pytest.mark.parametrize("name", list(SELL_CASES))
def test_shadow_sell_replace_places_exactly_as_with_no_guard(run, name):
    kw, outcome, is_fill = SELL_CASES[name]
    off = run(**kw, guard_off=True)
    on = run(**kw)
    assert config.STOP_GUARD_MODE == "shadow"
    assert len(on["place"]) == 1
    assert on["place"] == off["place"]          # identical manage_stop_place args
    assert on["place"][0].args[2] == kw["record_stop"]
    assert on["record"] == off["record"]        # no record change from the guard
    assert on["alerts"] == off["alerts"]        # no 🚨 from the guard
    assert "resting_read" not in on["events"] and "open_orders" not in on["events"]
    if outcome is None:
        assert on["rows"] == [] and on["info"] == []
        return
    assert len(on["rows"]) == 1 and len(on["info"]) == 1
    row = on["rows"][0]
    path = "trim_replace" if run is _trim else "manual_sell_replace"
    assert (row["mode"], row["path"], row["outcome"]) == ("shadow", path, outcome)
    assert row["price_is_fill"] is is_fill
    assert row["resting_read"] == "ok" and row["n_resting_below"] == 0
    assert row["stale_check"] == ("skipped_fill_price" if is_fill else "made")
    assert off["rows"] == [] and off["info"] == []


@pytest.mark.parametrize("run", [_trim, _manual], ids=["trim", "manual_sell"])
@pytest.mark.parametrize("fill_price", [FILL, None], ids=["fill", "quote"])
def test_nothing_slow_between_the_cancel_and_the_place(run, fill_price):
    """Level at or above the price: no broker read between the sell and the
    place, and the shadow alert is sent after the place."""
    on = run(record_stop=105.0, resting=[_stp(95.0)], fill_price=fill_price)
    ev = on["events"]
    assert ev.index("place") < ev.index("info")
    assert "resting_read" not in ev and "open_orders" not in ev[:ev.index("place")]
    assert on["info"][0].endswith("Not applied (shadow): the stop goes on at its level.")


@pytest.mark.parametrize("run", [_trim, _manual], ids=["trim", "manual_sell"])
def test_shadow_alert_is_sent_even_when_the_place_fails(run):
    on = run(record_stop=105.0, floor=95.0, place_err=RuntimeError("down"))
    assert len(on["info"]) == 1 and on["events"].index("place") < on["events"].index("info")


@pytest.mark.parametrize("run", [_trim, _manual], ids=["trim", "manual_sell"])
def test_sell_replace_calls_the_guard_with_its_own_price_and_floor(run):
    with patch("risk_manager._stop_below_market", side_effect=lambda t, level, *a, **k: level) as g:
        run(record_stop=90.0, floor=85.0)
    g.assert_called_once()
    args, kw = g.call_args.args, g.call_args.kwargs
    assert args[:4] == ("NVDA", 90.0, 85.0, FILL)
    assert kw["path"] == ("trim_replace" if run is _trim else "manual_sell_replace")
    assert kw["price_is_fill"] is True and "record_follows" not in kw
    assert kw["resting"] == [] and kw["alerts_out"] == []


def test_trim_guard_failure_still_places_at_the_record():
    with patch("risk_manager._plausible_price", side_effect=RuntimeError("boom")):
        on = _trim(record_stop=105.0, floor=95.0)
    assert len(on["place"]) == 1 and on["place"][0].args[2] == 105.0


def test_guard_with_a_bad_level_returns_it_and_never_raises():
    import risk_manager
    assert risk_manager._stop_below_market("NVDA", "x", None, FILL, "w", path="p") == "x"


def test_trim_enforce_places_below_the_price_and_keeps_the_record(monkeypatch):
    monkeypatch.setattr(config, "STOP_GUARD_MODE", "enforce")
    on = _trim(record_stop=105.0, floor=95.0)
    assert on["place"][0].args[2] == 95.0
    assert on["record"] == []                    # the record keeps its level
    assert on["rows"][0]["outcome"] == "placed_below_market"
    assert on["rows"][0]["record"] == "kept_higher"
    ev = on["events"]
    assert ev.index("place") < ev.index("alert")     # the 🚨 goes after the place too


# ── reconcile resize: tasks._fix_overstop ──

def _orders(*qtys, price=150.0):
    return [{"order_id": i, "ticker": "AMD", "action": "SELL", "order_type": "STP",
             "qty": q, "stop_price": price, "status": "Submitted"} for i, q in enumerate(qtys)]


def _fix(open_orders, record_stop, last, guard_off=False, wrap=False, **kw):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with ExitStack() as st:
        for p in (patch("ibkr.get_client", return_value=client),
                  patch("ibkr.get_open_orders", side_effect=open_orders),
                  patch("db.get_position_stop", return_value={"stop_price": record_stop}),
                  patch("ibkr.get_latest_price", return_value=last),
                  patch("tasks.manage_cancel_stop_orders_for"), patch("db.set_stop_broker_order_id"),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("time.sleep")):
            st.enter_context(p)
        if guard_off:
            st.enter_context(patch("risk_manager._stop_below_market",
                                   side_effect=lambda t, level, *a, **k: level))
        guard = (st.enter_context(patch("risk_manager._stop_below_market",
                                        wraps=risk_manager._stop_below_market))
                 if wrap else None)
        queued = st.enter_context(patch("db.log_activity_queued"))
        send = st.enter_context(patch("alerts._send"))
        place = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 9}))
        msg = tasks._fix_overstop("AMD", **kw)
    out = _out(place, queued, send, None, [])
    out.update(msg=msg, guard=guard)
    return out


FIX_CASES = {
    "over_covered": lambda: [_orders(10, 10), []],
    "under_covered": lambda: [_orders(4)] * 2,
    "naked": lambda: [[]] * 2,
}


@pytest.mark.parametrize("name", list(FIX_CASES))
def test_shadow_overstop_resize_places_exactly_as_with_no_guard(name):
    off = _fix(FIX_CASES[name](), 150.0, 200.0, guard_off=True)
    on = _fix(FIX_CASES[name](), 150.0, 200.0, wrap=True)
    assert len(on["place"]) == 1 and on["place"] == off["place"]
    assert on["place"][0].args[2] == 150.0 and on["place"][0].kwargs == {"market_price": 200.0}
    assert on["msg"] == off["msg"]
    on["guard"].assert_called_once()
    g = on["guard"].call_args
    assert g.args[:4] == ("AMD", 150.0, None, 200.0)
    assert g.kwargs["path"] == "overstop_resize" and g.kwargs["price_is_fill"] is False
    assert on["rows"] == [] and on["info"] == [] and on["alerts"] == []


@pytest.mark.parametrize("record_stop,last", [(200.0, 200.0), (210.0, 200.0), (150.0, 0.0),
                                              (150.0, None)])
def test_overstop_refusal_runs_first_and_unchanged(record_stop, last):
    on = _fix([_orders(4)] * 2, record_stop, last, wrap=True)
    shown = "unknown" if not last else f"${last:.2f}"
    assert on["msg"] == (f"🚨 AMD: stops cover 4 of 10 shares; recorded stop "
                         f"${record_stop:.2f} is not below the last price ({shown}), so "
                         f"nothing was placed — set the stop in TWS")
    on["guard"].assert_not_called()
    assert on["place"] == [] and on["rows"] == []


def test_overstop_resize_passes_the_resting_read_it_already_has():
    """The under-cover branch hands the guard its own open-orders read, so
    the guard makes no second resting read."""
    with patch("tasks._resting_sell_stops") as reread:
        on = _fix([_orders(4)] * 2, 150.0, 200.0, wrap=True)
    reread.assert_not_called()
    assert [o["qty"] for o in on["guard"].call_args.kwargs["resting"]] == [4]


def test_refused_over_position_resize_is_guarded_and_unchanged_in_shadow():
    mine = [_stp(150.0, qty=12, oid=1)]
    stuck_after = [_stp(150.0, qty=4, oid=2)]

    def run(guard_off):
        with ExitStack() as st:
            st.enter_context(patch("tasks.manage_cancel_order"))
            st.enter_context(patch("tasks._resting_sell_stops", return_value=stuck_after))
            st.enter_context(patch("db.set_stop_broker_order_id"))
            st.enter_context(patch("time.sleep"))
            if guard_off:
                st.enter_context(patch("risk_manager._stop_below_market",
                                       side_effect=lambda t, level, *a, **k: level))
            g = st.enter_context(patch("risk_manager._stop_below_market",
                                       wraps=risk_manager._stop_below_market))
            place = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 9}))
            msg = tasks._fix_refused_over_position("NVDA", mine, lambda o: o["order_id"] == 1,
                                                   10.0, 150.0, 200.0)
        return place.call_args_list, msg, g

    off_place, off_msg, _ = run(True)
    on_place, on_msg, g = run(False)
    assert on_place == off_place and on_msg == off_msg
    assert on_place[0].args == ("NVDA", 6.0, 150.0)
    assert g.call_args.kwargs["path"] == "overstop_resize"
    assert g.call_args.kwargs["resting"] == stuck_after


def test_overstop_guarded_level_never_raises():
    assert tasks._overstop_guarded_level("AMD", "bad", 200.0, []) == "bad"
    with patch("risk_manager._stop_below_market", side_effect=RuntimeError("x")):
        assert tasks._overstop_guarded_level("AMD", 150.0, 200.0, []) == 150.0
