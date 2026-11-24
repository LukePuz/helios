"""The above-market guard's one bounded re-read of the resting stops
(risk_manager._stop_below_market, _reread_resting_stops) when the no-loosen
floor's read failed (resting_unread).

resting_read: "reread_ok" (the re-read rows used as a normal read would),
"reread_failed" (fallback alone; reread_error the exception class),
"reread_skipped_window" (the caller is between a stop cancel and its
re-place: no re-read, fallback alone). The re-read runs in enforce only:
shadow records "reread_skipped_shadow" and makes no broker call, so the
caller's orders and their timing match a run with no guard.

Prices are scaled as in test_stop_above_market_guard: L197 = 89.3 kept on the
record, L198 10.7% below it resting, L199 6.2% below L198 the new shares'
own stop, the fill 0.6% below L197."""
from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

import pytest

import config
import execution
import ibkr
import risk_manager
import tasks

L197 = 89.3
L198 = round(L197 * (1 - 0.107), 2)
L199 = round(L198 * (1 - 0.062), 2)
FILL = round(L197 * (1 - 0.006), 2)


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(config, "STOP_GUARD_MODE", "enforce")


def _stp(price, qty=1.0, status="Submitted", order_type="STP", ticker="NVDA", action="SELL"):
    return {"ticker": ticker, "action": action, "order_type": order_type, "qty": qty,
            "stop_price": price, "status": status, "order_id": 11, "remaining": None}


def _guard(level=L197, fallback=L199, price=FILL, reread=None, reread_effect=None,
           resting_unread=True, cancel_window=False, price_is_fill=True, rth=True):
    """_stop_below_market with the re-read mocked; returns (level, rows, info, alerts, reread)."""
    rr = (patch("risk_manager._reread_resting_stops", side_effect=reread_effect) if reread_effect
          else patch("risk_manager._reread_resting_stops", return_value=list(reread or [])))
    with rr as reread_mock, \
         patch("tasks._resting_sell_stops", side_effect=AssertionError("no second plain read")), \
         patch("market_data.is_market_hours", return_value=rth), \
         patch("db.log_activity_queued") as queued, \
         patch("alerts._send") as send:
        out = risk_manager._stop_below_market(
            "NVDA", level, fallback, price, "ADD whole-position stop", path="add_whole_position",
            resting=None, resting_unread=resting_unread, cancel_window=cancel_window,
            price_is_fill=price_is_fill)
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "stop_above_market_guard"]
    info = [c.args[0] for c in send.call_args_list if "stop guard (shadow)" in c.args[0]]
    alerts = [c.args[0] for c in send.call_args_list if "🚨" in c.args[0]]
    return out, rows, info, alerts, reread_mock


def _pct(x):
    return pytest.approx((x / FILL - 1) * 100, abs=1e-3)


# ── the guard ────────────────────────────────────────────────────────────────

def test_shadow_makes_no_reread_and_works_from_the_fallback():
    out, rows, info, alerts, reread = _guard(reread=[_stp(L198)])
    assert out == L197                                   # placed as before
    reread.assert_not_called()
    (r,) = rows
    assert r["resting_read"] == "reread_skipped_shadow" and r["n_resting_below"] is None
    assert r["outcome"] == "would_placed_below_market" and r["mode"] == "shadow"
    assert r["placed_vs_price_pct"] == _pct(L199)        # the fallback alone, as at 8f1cb22
    assert r["reread_error"] is None
    assert r["fallback_only_outcome"] is None and r["fallback_only_placed_vs_price_pct"] is None
    assert alerts == [] and len(info) == 1 and "$" not in info[0]


@pytest.mark.parametrize("cancel_window", [False, True])
def test_shadow_never_calls_the_broker_from_the_guard(cancel_window):
    """The real _reread_resting_stops: in shadow neither it nor
    ibkr.get_open_orders nor tasks._resting_sell_stops is called."""
    with patch("ibkr.get_open_orders") as read, \
         patch("tasks._resting_sell_stops") as plain, \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued") as queued, patch("alerts._send"):
        out = risk_manager._stop_below_market(
            "NVDA", L197, L199, FILL, "x", path="add_whole_position", resting=None,
            resting_unread=True, cancel_window=cancel_window, price_is_fill=True)
    assert out == L197
    read.assert_not_called()
    plain.assert_not_called()
    assert queued.call_args.args[4]["resting_read"] == "reread_skipped_shadow"


@pytest.mark.usefixtures("enforce")
def test_enforce_reread_ok_records_the_fallback_alone_beside_it():
    out, rows, _, _, reread = _guard(reread=[_stp(L198)])
    reread.assert_called_once_with("NVDA")
    r = rows[0]
    assert out == L198
    assert r["resting_read"] == "reread_ok" and r["n_resting_below"] == 1
    assert r["placed_vs_price_pct"] == _pct(L198)
    assert r["fallback_only_outcome"] == "placed_below_market"
    assert r["fallback_only_placed_vs_price_pct"] == _pct(L199)
    assert r["reread_error"] is None


@pytest.mark.usefixtures("enforce")
def test_enforce_reread_ok_places_the_highest_resting_stop_below_the_price():
    out, rows, _, alerts, _ = _guard(reread=[_stp(L198)])
    assert out == L198 and L199 < L198 < FILL
    assert rows[0]["resting_read"] == "reread_ok" and rows[0]["outcome"] == "placed_below_market"
    assert rows[0]["fallback_only_outcome"] == "placed_below_market"
    assert len(alerts) == 1


@pytest.mark.usefixtures("enforce")
def test_enforce_reread_ok_with_nothing_below_but_the_fallback_matches_fallback_alone():
    out, rows, _, _, _ = _guard(reread=[])
    assert out == L199
    assert rows[0]["resting_read"] == "reread_ok" and rows[0]["n_resting_below"] == 0


@pytest.mark.usefixtures("enforce")
def test_enforce_reread_rows_feed_the_stale_price_check():
    """A working STP at/above a non-fill price in regular hours: the price
    looks old, so the level is kept, as on a normal read."""
    out, rows, _, alerts, _ = _guard(reread=[_stp(L197), _stp(L198)], price_is_fill=False)
    assert out == L197
    assert rows[0]["outcome"] == "resting_at_or_above_price" and rows[0]["stale_check"] == "made"
    assert rows[0]["n_resting_at_or_above"] == 1


@pytest.mark.usefixtures("enforce")
def test_reread_failure_works_from_the_fallback_and_never_raises():
    out, rows, _, _, reread = _guard(reread_effect=TimeoutError("IBKR reqAllOpenOrders timed out"))
    assert out == L199
    reread.assert_called_once()
    r = rows[0]
    assert r["resting_read"] == "reread_failed" and r["reread_error"] == "TimeoutError"
    assert r["n_resting_below"] is None and r["fallback_only_outcome"] is None


@pytest.mark.usefixtures("enforce")
def test_cancel_window_skips_the_reread():
    out, rows, _, _, reread = _guard(reread=[_stp(L198)], cancel_window=True)
    reread.assert_not_called()
    assert out == L199
    assert rows[0]["resting_read"] == "reread_skipped_window"
    assert rows[0]["fallback_only_outcome"] is None


@pytest.mark.usefixtures("enforce")
def test_no_reread_when_the_level_is_below_the_price():
    out, rows, _, _, reread = _guard(level=L198, reread=[_stp(L198)])
    assert out == L198 and rows == []
    reread.assert_not_called()


@pytest.mark.usefixtures("enforce")
def test_no_reread_when_the_price_is_unread():
    out, rows, _, _, reread = _guard(price=None, reread=[_stp(L198)])
    assert out == L197 and rows[0]["outcome"] == "price_unread"
    reread.assert_not_called()


@pytest.mark.usefixtures("enforce")
def test_no_reread_when_the_callers_read_worked():
    with patch("risk_manager._reread_resting_stops") as reread, \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued") as queued, patch("alerts._send"):
        risk_manager._stop_below_market("NVDA", L197, L199, FILL, "x", path="add_whole_position",
                                        resting=[_stp(L198)], price_is_fill=True)
    reread.assert_not_called()
    assert queued.call_args.args[4]["resting_read"] == "ok"


@pytest.mark.usefixtures("enforce")
def test_guard_never_raises_when_the_reread_helper_is_broken():
    out, rows, _, _, _ = _guard(reread_effect=TypeError("bad"))
    assert out == L199 and rows[0]["resting_read"] == "reread_failed"


# ── the bounded read ─────────────────────────────────────────────────────────

def test_reread_is_one_bounded_open_orders_read_without_reconnect():
    orders = [_stp(L198), _stp(L197, ticker="AMD"), _stp(L199, order_type="LMT"),
              _stp(L199, status="Cancelled"), {**_stp(L199), "action": "BUY"}]
    with patch("ibkr.get_open_orders", return_value=orders) as read:
        rows = risk_manager._reread_resting_stops("NVDA")
    read.assert_called_once_with(timeout=risk_manager._GUARD_REREAD_TIMEOUT_S, reconnect=False)
    assert [r["stop_price"] for r in rows] == [L198]
    assert risk_manager._GUARD_REREAD_TIMEOUT_S <= 5.0


def test_open_orders_without_reconnect_never_connects():
    with patch.object(ibkr, "_ib", None), \
         patch.object(ibkr, "get_ib", side_effect=AssertionError("connect")):
        with pytest.raises(ConnectionError):
            ibkr.get_open_orders(timeout=3.0, reconnect=False)
    dead = MagicMock()
    dead.isConnected.return_value = False
    with patch.object(ibkr, "_ib", dead), \
         patch.object(ibkr, "get_ib", side_effect=AssertionError("connect")):
        with pytest.raises(ConnectionError):
            ibkr.get_open_orders(timeout=3.0, reconnect=False)
    dead.reqAllOpenOrders.assert_not_called()


def test_open_orders_timeout_bounds_the_request():
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.RequestTimeout = 0
    seen = []
    ib.reqAllOpenOrders.side_effect = lambda: seen.append(ib.RequestTimeout) or []
    with patch.object(ibkr, "_ib", ib), \
         patch.object(ibkr, "get_ib", side_effect=AssertionError("connect")):
        assert ibkr.get_open_orders(timeout=3.0, reconnect=False) == []
    assert seen == [3.0] and ib.RequestTimeout == 0


def test_open_orders_default_read_is_unchanged():
    ib = MagicMock()
    ib.RequestTimeout = 0
    seen = []
    ib.reqAllOpenOrders.side_effect = lambda: seen.append(ib.RequestTimeout) or []
    with patch.object(ibkr, "get_ib", return_value=ib) as connect:
        ibkr.get_open_orders()
    connect.assert_called_once_with()
    assert seen == [ibkr._REQ_TIMEOUT_S]


# ── the placed stop: same placeOrder with and without the re-read ────────────

def _add_and_place(reread=None, reread_effect=None, guard_off=False, real_reread=False):
    """The ADD merge with the floor's read failing, then the placement the
    caller makes from its result (ibkr.place_stop_order on a fake IB).
    Returns (placeOrder args, call order)."""
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=2, avg_entry_price=95.0)
    seq = MagicMock()
    rr = (nullcontext() if real_reread or guard_off
          else patch("risk_manager._reread_resting_stops", side_effect=reread_effect) if reread_effect
          else patch("risk_manager._reread_resting_stops",
                     side_effect=lambda t: (seq.reread(t), list(reread or []))[1]))
    guard = (patch("risk_manager._stop_below_market", side_effect=lambda t, level, *a, **k: level)
             if guard_off else nullcontext())
    ib = MagicMock()
    trade = MagicMock()
    trade.order.orderId = 77
    trade.orderStatus.status = "PreSubmitted"
    ib.placeOrder.side_effect = lambda c, o: (seq.place(o.action, o.orderType, o.totalQuantity,
                                                        o.auxPrice, o.tif), trade)[1]
    with patch("db.get_position_stop", return_value={"stop_price": L197}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=(2.0, 95.0)), \
         patch("tasks._resting_sell_stops",
               side_effect=lambda *a, **k: (seq.plain_read(), (_ for _ in ()).throw(
                   TimeoutError("read")))[1]), \
         patch("ibkr.get_open_orders",
               side_effect=lambda *a, **k: (seq.get_open_orders(*a, **k), [])[1]), \
         rr, guard, \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("db.log_activity_queued"), patch("alerts._send"), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=lambda t: seq.cancel(t)):
        stop, whole = execution._whole_position_stop("NVDA", L199, 1, status={},
                                                     market_price=FILL, action="ADD",
                                                     held_before=1, price_is_fill=True)
        with patch.object(ibkr, "get_ib", return_value=ib), \
             patch.object(ibkr, "_stock", return_value=MagicMock()), \
             patch.object(ibkr, "_refuse_sell_on_short"):
            ibkr.place_stop_order("NVDA", whole[1], stop, market_price=FILL)
    names = [c[0] for c in seq.mock_calls
             if c[0] in ("plain_read", "get_open_orders", "reread", "cancel", "place")]
    return [c for c in seq.mock_calls if c[0] == "place"], names


def test_shadow_orders_and_their_order_match_a_run_with_no_guard():
    """Shadow, the floor's read failing, the real re-read helper: the guard
    calls no broker read, and the calls are the floor's read, then cancel,
    then place, with the same placeOrder args, exactly as with no guard (and
    as at 8f1cb22, where the guard made no read on this case)."""
    placed, order = _add_and_place(reread=[_stp(L198)], real_reread=True)
    no_guard, order_none = _add_and_place(guard_off=True)
    assert placed == no_guard == [call.place("SELL", "STP", 2.0, L197, "GTC")]
    assert order == order_none == ["plain_read", "cancel", "place"]


@pytest.mark.usefixtures("enforce")
def test_enforce_placeorder_uses_the_reread_result():
    placed, order = _add_and_place(reread=[_stp(L198)])
    assert placed == [call.place("SELL", "STP", 2.0, L198, "GTC")]
    assert order == ["plain_read", "reread", "cancel", "place"]   # before the cancel
    fallback, _ = _add_and_place(reread_effect=TimeoutError("read"))
    assert fallback == [call.place("SELL", "STP", 2.0, L199, "GTC")]


# ── callers: which ones sit between a cancel and the place ──────────────────

def _late(order, from_queue, mode="enforce"):
    """tasks._reprotect_after_late_sell with the floor's read failing (covered
    with no rows back); returns the guard row and whether the re-read ran."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=2)]
    with patch.object(config, "STOP_GUARD_MODE", mode), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_latest_price", return_value=FILL), \
         patch("ibkr.get_fresh_minute_price", return_value=FILL), \
         patch("db.get_position_stop", return_value={"stop_price": L197, "trailing_high": None}), \
         patch("execution._stops_still_resting", return_value=1.0), \
         patch("tasks._resting_sell_stops", side_effect=TimeoutError("read")), \
         patch("risk_manager._reread_resting_stops", return_value=[_stp(L198)]) as reread, \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", return_value={"order_id": 3}) as place, \
         patch("db.set_stop_broker_order_id"), patch("db.update_trailing_stop"), \
         patch("db.mark_profit_target_taken"), \
         patch("db.log_activity_queued") as queued, \
         patch("market_data.is_market_hours", return_value=True), \
         patch("alerts._send"), patch("time.sleep"):
        tasks._reprotect_after_late_sell("NVDA", order, None, from_queue=from_queue)
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "stop_above_market_guard"]
    return rows, reread, place


@pytest.mark.parametrize("order,from_queue", [({}, True), ({"intent": "BUY"}, True)])
def test_late_or_queued_buy_follows_no_cancel_so_enforce_rereads(order, from_queue):
    rows, reread, place = _late(order, from_queue)
    reread.assert_called_once_with("NVDA")
    assert rows[0]["resting_read"] == "reread_ok"
    assert place.call_args.args[2] == L198          # the re-read's resting stop below the price


@pytest.mark.parametrize("order,from_queue", [
    ({"intent": "RESTORE"}, True), ({"intent": "EXIT"}, True),
    ({"intent": "TRIM", "side": "SELL"}, False), ({"side": "SELL"}, False)])
def test_reprotect_after_a_sales_cancel_skips_the_reread(order, from_queue):
    rows, reread, place = _late(order, from_queue)
    reread.assert_not_called()
    assert rows[0]["resting_read"] == "reread_skipped_window"
    assert place.call_args.args[2] == L197          # no fallback, nothing below: kept


@pytest.mark.parametrize("order,from_queue", [
    ({}, True), ({"intent": "BUY"}, True), ({"intent": "RESTORE"}, True),
    ({"side": "SELL"}, False)])
def test_reprotect_in_shadow_never_rereads(order, from_queue):
    rows, reread, place = _late(order, from_queue, mode="shadow")
    reread.assert_not_called()
    assert rows[0]["resting_read"] == "reread_skipped_shadow"
    assert place.call_args.args[2] == L197


def _guard_calls():
    """{path keyword: cancel_window keyword} for every
    risk_manager._stop_below_market call in execution.py and tasks.py,
    read from the parsed source (keyword values, not comments)."""
    import ast
    import inspect
    out = {}
    for mod in (execution, tasks):
        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_stop_below_market"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "risk_manager"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            path = kw.get("path")
            key = path.value if isinstance(path, ast.Constant) else ast.unparse(path)
            cw = kw.get("cancel_window")
            out[key] = (None if cw is None else
                        cw.value if isinstance(cw, ast.Constant) else ast.unparse(cw))
    return out


def test_callers_state_their_window():
    """Each _stop_below_market caller passes cancel_window: whether it sits
    between a stop cancel and its re-place."""
    assert _guard_calls() == {"add_whole_position": False, "trim_replace": True,
                              "manual_sell_replace": True, "path": "after_sell_cancel",
                              "overstop_resize": True}


@pytest.mark.usefixtures("enforce")
def test_enforce_reread_through_the_real_signature_succeeds():
    """The real _reread_resting_stops against an autospec'd
    ibkr.get_open_orders: a call the real signature rejects raises inside
    the guard and would read as reread_failed, so this must be reread_ok."""
    with patch("ibkr.get_open_orders", autospec=True, return_value=[_stp(L198)]) as read, \
         patch("tasks._resting_sell_stops", side_effect=AssertionError("no plain read")), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("db.log_activity_queued") as queued, patch("alerts._send"):
        out = risk_manager._stop_below_market(
            "NVDA", L197, L199, FILL, "x", path="add_whole_position", resting=None,
            resting_unread=True, price_is_fill=True)
    read.assert_called_once_with(timeout=risk_manager._GUARD_REREAD_TIMEOUT_S, reconnect=False)
    r = queued.call_args.args[4]
    assert r["resting_read"] == "reread_ok" and r["reread_error"] is None
    assert out == L198
