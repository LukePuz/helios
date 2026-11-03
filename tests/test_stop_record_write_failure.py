"""A failed position_stops write (pool exhausted, DB down) must not keep the
protective stop off the broker after a fill: the stop is placed with the same
arguments, one alert says its record is missing, and the stop check leaves an
unrecorded stop on a held position alone."""
import inspect
from unittest.mock import MagicMock, patch

import psycopg2.pool
import pytest

import db
import execution
import main
import tasks

FAILURES = [psycopg2.pool.PoolError("connection pool exhausted"), Exception("db down")]


@pytest.fixture(autouse=True)
def _gap_check_reads_no_gap(monkeypatch):
    import market_data
    monkeypatch.setattr(market_data, "check_gap_confirmation",
                        lambda ticker, today=None: {"status": "no_gap"})


@pytest.fixture(autouse=True)
def _regular_hours():
    with patch("market_data.is_market_hours", return_value=True):
        yield


def _approval():
    return {"id": 1, "ticker": "NVDA", "action": "OPEN", "conviction": 7,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 10, "price_at_sizing": 100.0,
                          "stop_loss_price": 90.0}}


_LAST: dict = {}


def _entry(upsert=None, set_id=None, prior=None, place_effect=None, positions=()):
    """execute_approved for an OPEN filled at $100 (no bracket). prior: the
    get_position_stop side effect. positions: (qty, avg) the broker shows."""
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    if upsert is not None:
        db_fns["upsert_position_stop"] = MagicMock(side_effect=upsert)
    if set_id is not None:
        db_fns["set_stop_broker_order_id"] = MagicMock(side_effect=set_id)
    if prior is not None:
        db_fns["get_position_stop"] = MagicMock(side_effect=prior)
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", return_value={"filled_qty": 10, "fill_price": 100.0,
                                                        "order_id": 1, "status": "Filled"}), \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=None), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9},
               side_effect=place_effect) as place, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks._queue_reprotect") as queued, \
         patch("tasks.open_manual_action") as action, \
         patch("alerts._send") as send:
        if positions:
            qty, avg = positions
            client.return_value.get_open_position.return_value = MagicMock(qty=qty, avg_entry_price=avg)
            client.return_value.get_all_positions.return_value = [
                MagicMock(symbol="NVDA", qty=qty, avg_entry_price=avg)]
        else:
            client.return_value.get_open_position.side_effect = Exception("No open position")
            client.return_value.get_all_positions.return_value = []
        execution.execute_approved(_approval())
    _LAST.update(cancel=cancel, action=action)
    return place, send, db_fns, queued


def _texts(send):
    return [c[0][0] for c in send.call_args_list]


def _record_alerts(send):
    return [t for t in _texts(send) if "STOP RECORD MISSING" in t]


@pytest.mark.parametrize("err", FAILURES, ids=["PoolError", "Exception"])
def test_entry_stop_is_placed_when_the_record_write_raises(err):
    h_place, h_send, h_db, _ = _entry()
    place, send, fns, queued = _entry(upsert=err)
    assert h_place.call_count == 1
    assert place.call_args_list == h_place.call_args_list   # same stop, same arguments
    fns["upsert_position_stop"].assert_called_once_with(**h_db["upsert_position_stop"].call_args.kwargs)
    alerts = _record_alerts(send)
    assert len(alerts) == 1 and "NVDA" in alerts[0] and "$90.00" in alerts[0]
    assert "placed at the broker" in alerts[0]
    queued.assert_not_called()
    # Every other alert is the one the healthy run sends.
    assert [t for t in _texts(send) if t not in alerts] == _texts(h_send)


def test_entry_healthy_path_sends_no_record_alert():
    place, send, fns, queued = _entry()
    place.assert_called_once()
    fns["set_stop_broker_order_id"].assert_called_once_with("NVDA", 9)
    assert not _record_alerts(send)
    queued.assert_not_called()


def test_entry_swallowed_record_write_failure_alerts_too():
    """upsert_position_stop logs and swallows a failed query: it returns False."""
    place, send, _, _ = _entry(upsert=lambda **k: False)
    place.assert_called_once()
    assert len(_record_alerts(send)) == 1


def test_entry_order_id_write_failure_does_not_place_a_second_stop():
    place, send, _, _ = _entry(set_id=psycopg2.pool.PoolError("exhausted"))
    assert place.call_count == 1
    assert not any("UNPROTECTED" in t for t in _texts(send))


def _manual(upsert=None, set_id=None):
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"
              and n not in ("upsert_position_stop", "set_stop_broker_order_id")}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "fill_price": 100.0,
                                                        "filled_qty": 2}), \
         patch("ibkr.get_latest_price", return_value=None), \
         patch("db.upsert_position_stop", side_effect=upsert) as up, \
         patch("db.set_stop_broker_order_id", side_effect=set_id), \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}) as place, \
         patch("tasks._queue_reprotect") as queued, \
         patch("tasks.open_manual_action"), patch("alerts._send") as send:
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.return_value = (95.0, "ATR")
        r = execution.execute_manual("ZZZ", "BUY", 2, "test")
    return r, place, send, up, queued


@pytest.mark.parametrize("err", FAILURES, ids=["PoolError", "Exception"])
def test_manual_buy_stop_is_placed_when_the_record_write_raises(err):
    h_r, h_place, h_send, h_up, _ = _manual()
    r, place, send, up, queued = _manual(upsert=err)
    assert h_place.call_count == 1
    assert place.call_args_list == h_place.call_args_list
    assert up.call_args == h_up.call_args
    assert r["success"] and r["stop_price"] == h_r["stop_price"] == 95.0
    alerts = _record_alerts(send)
    assert len(alerts) == 1 and "ZZZ" in alerts[0] and "$95.00" in alerts[0]
    queued.assert_not_called()
    assert [t for t in _texts(send) if t not in alerts] == _texts(h_send)


def test_manual_buy_healthy_path_sends_no_record_alert():
    r, place, send, up, queued = _manual()
    place.assert_called_once_with("ZZZ", 2, 95.0, market_price=100.0)
    up.assert_called_once()
    assert not _record_alerts(send)


def test_manual_buy_order_id_write_failure_does_not_place_a_second_stop():
    r, place, send, _, _ = _manual(set_id=Exception("db down"))
    assert place.call_count == 1 and r["success"]


def _adopt(upsert=None, resting=()):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(avg_entry_price=100.0)
    rm = MagicMock()
    rm.calculate_stop.return_value = (90.0, "ATR")
    place = MagicMock(return_value={"order_id": 7})
    with patch("db.get_position_stop", return_value=None), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("db.upsert_position_stop", side_effect=upsert), patch("db.set_stop_broker_order_id"):
        msg = main._adopt_position(client, rm, "AMD", 10.0, place, lambda t: list(resting))
    return msg, place


@pytest.mark.parametrize("err", FAILURES, ids=["PoolError", "Exception"])
def test_adoption_places_the_stop_when_the_record_write_raises(err):
    h_msg, h_place = _adopt()
    msg, place = _adopt(upsert=err)
    assert place.call_args_list == h_place.call_args_list
    place.assert_called_once_with("AMD", 10.0, 90.0, market_price=200.0)
    assert "STOP RECORD MISSING" in msg and "$90.00" in msg and "AMD" in msg
    assert h_msg == "AMD (10 shares, stop placed @ $90.00)"


def test_adoption_record_only_failure_is_reported():
    msg, place = _adopt(upsert=Exception("db down"), resting=[{"qty": 10.0}])
    place.assert_not_called()
    assert "STOP RECORD MISSING" in msg


def test_stop_check_leaves_an_unrecorded_stop_on_a_held_position_alone():
    """No position_stops row, a held position, and a resting stop covering it:
    sections 1-3 of the stop check cancel nothing and place nothing."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=10)]
    stop = [{"order_id": 5, "ticker": "NVDA", "action": "SELL", "order_type": "STP",
             "qty": 10.0, "stop_price": 90.0, "status": "Submitted"}]
    state = {}
    for _ in range(3):   # several cycles: no pending marker arms a cancel
        with patch("ibkr.get_client", return_value=client), \
             patch("ibkr.get_open_orders", return_value=stop), \
             patch("db.get_all_position_stops", return_value=[]), \
             patch("db.get_position_stop", return_value=None), \
             patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("tasks.manage_cancel_order") as cancel, \
             patch("tasks.manage_cancel_stop_orders_for") as cancel_all, \
             patch("tasks.manage_stop_place") as place, \
             patch("alerts._send"):
            tasks.reconcile_stop_orders.run()
        cancel.assert_not_called(); cancel_all.assert_not_called(); place.assert_not_called()


def test_stop_check_never_cancels_an_unrecorded_overcover():
    """Over-covered with no record: _fix_overstop reports, cancels nothing."""
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=10)]
    stops = [{"order_id": i, "ticker": "NVDA", "action": "SELL", "order_type": "STP",
              "qty": 10.0, "stop_price": 90.0, "status": "Submitted"} for i in (5, 6)]
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=stops), \
         patch("db.get_position_stop", return_value=None), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel_all, \
         patch("tasks.manage_stop_place") as place:
        msg = tasks._fix_overstop("NVDA")
    cancel_all.assert_not_called(); place.assert_not_called()
    assert "no recorded stop" in msg


# -- follow-ups: the prior-record read, the ADD case, manual action, queued text --

def _prior_read_fails(ticker, strict=False):
    raise psycopg2.pool.PoolError("connection pool exhausted")


def _reads(*results):
    """get_position_stop side effect: one result per call (an exception raises)."""
    it = iter(results)

    def read(ticker, strict=False):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r
    return read


def _check_stops(send):
    return [t for t in _texts(send) if "CHECK STOPS" in t]


def _entry_ordered(prior, positions=(), upsert=None):
    """_entry, also recording whether the record write came before placement."""
    order: list[str] = []

    def place_eff(*a, **k):
        order.append("place")
        return {"order_id": 9}

    def up_eff(**k):
        order.append("upsert")
        if isinstance(upsert, Exception):
            raise upsert
        return None
    out = _entry(prior=prior, positions=positions, upsert=up_eff, place_effect=place_eff)
    return out, order


@pytest.mark.parametrize("positions", [(), (10, 100.0)], ids=["reads_flat", "reads_exactly_the_fill"])
def test_unreadable_prior_never_writes_before_placement_whatever_the_positions_read(positions):
    """The portfolio cache lags a fill: a real add can read flat or as exactly
    the fill, so the positions read must not decide the write."""
    (place, send, fns, _), order = _entry_ordered(
        _reads(psycopg2.pool.PoolError("exhausted"), {"stop_price": 85.0}), positions)
    assert order == ["place"]                           # no write: a record exists now
    assert len(_check_stops(send)) == 1
    (place, send, fns, _), order = _entry_ordered(
        _reads(psycopg2.pool.PoolError("exhausted"), None), positions)
    assert order == ["place", "upsert"]                 # none exists: written after placement


def test_no_record_after_placement_is_written_with_the_healthy_kwargs():
    h_place, h_send, h_db, _ = _entry()
    place, send, fns, _ = _entry(prior=_reads(psycopg2.pool.PoolError("x"), None))
    assert place.call_args_list == h_place.call_args_list
    place.assert_called_once_with("NVDA", 10, 90.0, market_price=100.0)
    fns["upsert_position_stop"].assert_called_once_with(**h_db["upsert_position_stop"].call_args.kwargs)
    assert _texts(send) == _texts(h_send)               # quiet: nothing to report
    _LAST["cancel"].assert_not_called()
    _LAST["action"].assert_not_called()


def test_no_record_after_placement_and_a_failed_write_alerts_record_missing():
    place, send, fns, _ = _entry(prior=_reads(psycopg2.pool.PoolError("x"), None),
                                 upsert=Exception("db down"))
    place.assert_called_once_with("NVDA", 10, 90.0, market_price=100.0)
    (text,) = _record_alerts(send)
    assert "$90.00" in text and "adopts the record" in text
    assert not _check_stops(send)
    _LAST["action"].assert_called_once()


def test_prior_record_is_read_strictly():
    _, _, fns, _ = _entry()
    assert fns["get_position_stop"].call_args_list[0] == (("NVDA",), {"strict": True})


def test_a_query_error_takes_the_unreadable_branch_not_fresh():
    """Non-strict, a query error read as None (a fresh position) and the new
    shares' level would overwrite the record."""
    def read(ticker, strict=False):
        if strict:
            raise RuntimeError("query failed")
        return None
    place, send, fns, _ = _entry(prior=read, positions=(25, 98.0))
    fns["upsert_position_stop"].assert_not_called()
    place.assert_called_once_with("NVDA", 10, 90.0, market_price=100.0)
    assert len(_check_stops(send)) == 1


def test_unreadable_prior_on_an_add_keeps_the_old_record_and_stops_the_new_shares():
    old = {"stop_price": 85.0}
    place, send, fns, _ = _entry(prior=_reads(psycopg2.pool.PoolError("exhausted"), old),
                                 positions=(25, 98.0))
    fns["upsert_position_stop"].assert_not_called()                      # old record untouched
    place.assert_called_once_with("NVDA", 10, 90.0, market_price=100.0)  # the new shares only
    _LAST["cancel"].assert_not_called()                                  # old stop left in place
    assert not _record_alerts(send)
    (text,) = _check_stops(send)
    assert "KEPT, not overwritten" in text and "old stop $85.00" in text
    assert "10 new shares got their own stop @ $90.00" in text
    assert "could LOWER the new shares' stop below $90.00" in text
    assert "clears once resting stops cover the position" in text
    _LAST["action"].assert_called_once()
    assert _LAST["action"].call_args.kwargs == {"clear_on_flat": True}


def test_kept_record_above_the_new_stop_has_no_lowering_warning():
    _, send, _, _ = _entry(prior=_reads(psycopg2.pool.PoolError("x"), {"stop_price": 93.0}),
                           positions=(25, 98.0))
    (text,) = _check_stops(send)
    assert "KEPT" in text and "LOWER" not in text


def test_unreadable_prior_twice_covers_both_cases_in_one_alert():
    place, send, fns, _ = _entry(prior=_prior_read_fails, positions=(25, 98.0))
    fns["upsert_position_stop"].assert_not_called()
    place.assert_called_once_with("NVDA", 10, 90.0, market_price=100.0)
    (text,) = _check_stops(send)
    assert "unknown" in text and "KEPT" in text and "STOP RECORD MISSING" in text
    assert len([t for t in _texts(send) if "STOP" in t.split("\n")[0]]) == 1


def test_unreadable_prior_with_a_queued_stop_says_place_it_in_tws_now():
    _, send, fns, _ = _entry(prior=_prior_read_fails, positions=(25, 98.0),
                             place_effect=tasks.OrdersCallExpired("x"))
    fns["upsert_position_stop"].assert_not_called()
    (text,) = _check_stops(send)
    assert "was NOT placed" in text and "place it in TWS now" in text


def test_add_with_a_failed_write_says_the_stale_record_stays():
    place, send, _, _ = _entry(prior=lambda t, strict=False: {"stop_price": 92.0},
                               upsert=psycopg2.pool.PoolError("exhausted"), positions=(25, 98.0))
    place.assert_called_once_with("NVDA", 25.0, 92.0, market_price=100.0)   # whole position
    (text,) = _record_alerts(send)
    assert "stale record remains at the old stop $92.00" in text
    assert "will NOT fix it" in text
    assert "adopts the record" not in text
    assert "clears once resting stops cover the position" in text


def test_fresh_position_failed_write_points_at_startup_adoption():
    _, send, _, _ = _entry(upsert=Exception("db down"))
    (text,) = _record_alerts(send)
    assert "adopts the record" in text and "stale record" not in text


def test_a_missing_record_opens_a_manual_action():
    _entry(upsert=Exception("db down"))
    action = _LAST["action"]
    action.assert_called_once()
    assert action.call_args.args[0] == "NVDA" and "$90.00" in action.call_args.args[1]
    assert action.call_args.kwargs == {"clear_on_flat": True}


def test_healthy_path_opens_no_manual_action():
    _entry()
    _LAST["action"].assert_not_called()


@pytest.mark.parametrize("make", [lambda: tasks.OrdersCallExpired("x"),
                                  lambda: type("LockLost", (Exception,), {})("x"),
                                  lambda: type("SellExceedsPosition", (Exception,), {})("x")],
                         ids=["OrdersCallExpired", "LockLost", "SellExceedsPosition"])
def test_queued_stop_with_no_record_says_place_it_in_tws_now(make):
    place, send, _, queued = _entry(upsert=Exception("db down"), place_effect=make())
    assert place.call_count == 1
    queued.assert_called_once()
    (text,) = _record_alerts(send)
    assert "was NOT placed" in text and "place it in TWS now" in text
    assert "can't place a stop without the record" in text


def test_unknown_stop_with_no_record_says_check_tws():
    _, send, _, _ = _entry(upsert=Exception("db down"), place_effect=tasks.OrdersCallUnknown("x"))
    (text,) = _record_alerts(send)
    assert "may or may not be resting" in text and "place it in TWS now" in text


def test_manual_buy_queued_stop_with_no_record_says_place_it_in_tws_now():
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["upsert_position_stop"] = MagicMock(side_effect=Exception("db down"))
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order", return_value={"order_id": 1, "fill_price": 100.0,
                                                        "filled_qty": 2}), \
         patch("ibkr.get_latest_price", return_value=None), \
         patch("tasks.manage_stop_place", side_effect=tasks.OrdersCallExpired("x")) as place, \
         patch("tasks._queue_reprotect"), patch("tasks.open_manual_action") as action, \
         patch("alerts._send") as send:
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.calculate_stop.return_value = (95.0, "ATR")
        execution.execute_manual("ZZZ", "BUY", 2, "test")
    assert place.call_count == 1
    (text,) = _record_alerts(send)
    assert "was NOT placed" in text and "place it in TWS now" in text
    assert any(c.kwargs.get("clear_on_flat") is True for c in action.call_args_list)
