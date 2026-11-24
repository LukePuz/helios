"""Profit-target sale, cancel-to-place window
(risk_manager.RiskManager._partial_sell_and_resize_stop):
- the sale's record-only rows (log_trade, log_order, update_order_fill,
  set_order_reconcile_note) are written after the remainder's stop is
  placed, and still written when the place raises;
- a failed re-place queues a PROFIT_TARGET reprotect at the computed stop
  and opens a manual action before the UNPROTECTED alert; the queued retry
  then places that stop for the count it reads."""
import json
from unittest.mock import MagicMock, patch

import pytest

import risk_manager
import tasks


@pytest.fixture(autouse=True)
def _cancel_confirmed():
    with patch("execution._stops_still_resting", return_value=0.0):
        yield


@pytest.fixture(autouse=True)
def _in_market_hours(monkeypatch):
    import market_data
    monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)


def _sale(place_effect=None, remaining=70, queue_effect=None, alert_effect=None, log_effect=None):
    """Sell 30 of 100 at a fill of 120 with the breakeven stop at 100. Returns
    (result, events); events lists each call in order."""
    events = _sale.events = []

    def rec(name, value=None, effect=None):
        def f(*a, **k):
            events.append((name, a, k))
            if effect is not None:
                raise effect
            return value
        return f

    def send(text, *a, **k):
        events.append(("alert", (text,), k))
        if alert_effect is not None:
            raise alert_effect
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=remaining)
    place = place_effect if place_effect is not None else rec("place", {"order_id": 8})
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks.manage_cancel_stop_orders_for", side_effect=rec("cancel")), \
         patch("stop_lock.assert_held"), \
         patch("ibkr.place_market_order", side_effect=rec("sell", {"filled_qty": 30, "fill_price": 120.0,
                                                                  "order_id": 5})), \
         patch("ibkr.get_client", return_value=client), \
         patch("db.log_trade", side_effect=rec("log_trade", effect=log_effect)), \
         patch("db.log_order", side_effect=rec("log_order", 11)), \
         patch("db.update_order_fill", side_effect=rec("update_order_fill", True)), \
         patch("db.set_order_reconcile_note", side_effect=rec("reconcile_note")), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_stop_place", side_effect=place), \
         patch("tasks._queue_reprotect", side_effect=rec("queue", effect=queue_effect)), \
         patch("tasks.open_manual_action", side_effect=rec("manual_action")), \
         patch("alerts._send", side_effect=send):
        result = object.__new__(risk_manager.RiskManager)._partial_sell_and_resize_stop(
            "NVDA", 100, 30, 100.0, decision_price=120.0)
    return result, events


def _names(events):
    return [e[0] for e in events]


def _raise(*a, **k):
    raise RuntimeError("ib down")


def test_sale_rows_are_written_after_the_remainders_stop():
    result, events = _sale()
    names = _names(events)
    assert names[:3] == ["cancel", "sell", "place"]
    assert names.index("place") < names.index("log_trade") < names.index("log_order") \
        < names.index("update_order_fill")
    assert result["new_stop"] == 100.0
    place = next(e for e in events if e[0] == "place")
    assert place[1] == ("NVDA", 70.0, 100.0)


def test_failed_place_queues_logs_then_opens_a_manual_action_and_alerts():
    """The place has failed (it can take ~150 s to say so): the rows go in
    right after the queue, before the manual action and Telegram, so
    reconcile's untracked-sell booking sees the sale as the bot's."""
    result, events = _sale(place_effect=_raise)
    names = _names(events)
    assert names == ["cancel", "sell", "queue", "log_trade", "log_order", "update_order_fill",
                     "manual_action", "alert"]
    queue = next(e for e in events if e[0] == "queue")
    assert queue[1] == ("NVDA", {"intent": "PROFIT_TARGET", "stop_price": 100.0}, None)
    manual = next(e for e in events if e[0] == "manual_action")
    assert manual[1][0] == "NVDA" and "$100.00" in manual[1][1] and "profit-target" in manual[1][1]
    text = next(e for e in events if e[0] == "alert")[1][0]
    assert text.startswith("🚨 UNPROTECTED POSITION") and "queued and retried" in text
    assert "for 70.0 shares" in text
    assert result["new_stop"] == 100.0                 # the caller still marks the target taken


def test_failed_queue_keeps_the_place_it_manually_wording_and_still_logs():
    _, events = _sale(place_effect=_raise, queue_effect=RuntimeError("redis"))
    names = _names(events)
    assert names.index("queue") < names.index("log_trade") < names.index("manual_action") \
        < names.index("alert")
    assert names.count("log_trade") == 1
    text = next(e for e in events if e[0] == "alert")[1][0]
    assert "queued" not in text and "Place stop @ $100.00 for 70.0 shares manually." in text


def test_failed_alert_still_logs_and_returns():
    result, events = _sale(place_effect=_raise, alert_effect=RuntimeError("tg"))
    assert "log_trade" in _names(events) and result["new_stop"] == 100.0


def test_no_remainder_places_nothing_and_still_logs():
    _, events = _sale(remaining=0)
    names = _names(events)
    assert "place" not in names and "log_trade" in names


@pytest.mark.parametrize("place_fails", [False, True])
def test_failed_logging_never_raises_or_touches_the_stop(place_fails):
    result, events = _sale(place_effect=_raise if place_fails else None,
                           log_effect=RuntimeError("db"))
    names = _names(events)
    assert ("place" in names) is (not place_fails) and "log_trade" in names
    assert "log_order" not in names                      # the block stops at its first failure
    assert result["new_stop"] == 100.0


# ── the queued retry places the remainder stop ──────────────────────────────

def test_queued_profit_target_retry_places_new_stop_for_the_then_current_count():
    """A failed re-place queues {PROFIT_TARGET, stop_price 100}; the next
    reconcile retry reads 70 held and places 100 for 70, marks the target
    taken at 100 and drops the entry."""
    _, events = _sale(place_effect=_raise)
    queued = next(e for e in events if e[0] == "queue")
    state = {}
    get = lambda k, d=None: state.get(k, d)               # noqa: E731
    put = lambda k, v: state.__setitem__(k, v)            # noqa: E731
    with patch("db.get_state", side_effect=get), patch("db.set_state", side_effect=put):
        tasks._queue_reprotect(*queued[1], **queued[2])   # what the sale queued, for real
    entry = json.loads(state["reprotect_pending"])["NVDA"]
    assert entry["intent"] == "PROFIT_TARGET" and entry["stop_price"] == 100.0

    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=70.0)]
    placed = []
    with patch("db.get_state", side_effect=get), patch("db.set_state", side_effect=put), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._fresh_read_short", return_value=False), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0, "entry_price": 100.0}), \
         patch("db.mark_profit_target_taken") as marked, \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("ibkr.get_latest_price", return_value=None), \
         patch("ibkr.get_fresh_minute_price", return_value=None), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", side_effect=lambda t, q, px, **k: (
             placed.append((t, q, px)), {"order_id": 9})[1]), \
         patch("db.set_stop_broker_order_id"), patch("db.log_activity_queued"), \
         patch("db.update_trailing_stop"), patch("alerts._send"):
        tasks._retry_pending_reprotects()
    assert placed == [("NVDA", 70.0, 100.0)]
    marked.assert_called_once_with("NVDA", 100.0)
    assert json.loads(state["reprotect_pending"]) == {}


def test_profit_target_entry_replaces_a_queued_buy_and_keeps_its_buy_markers():
    """The merge rule: a PROFIT_TARGET entry over a queued BUY replaces it
    (its stop_price is what the retry places) and carries had_buy and the
    BUY's own_stop."""
    state = {"reprotect_pending": json.dumps({"NVDA": {"intent": "BUY", "own_stop": 95.0}})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        tasks._queue_reprotect("NVDA", {"intent": "PROFIT_TARGET", "stop_price": 100.0}, None)
    entry = json.loads(state["reprotect_pending"])["NVDA"]
    assert entry["intent"] == "PROFIT_TARGET" and entry["stop_price"] == 100.0
    assert entry["had_buy"] is True and entry["own_stop"] == 95.0


# ── review fixes ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("place_fails", [False, True])
def test_rows_are_written_exactly_once(place_fails):
    _, events = _sale(place_effect=_raise if place_fails else None)
    names = _names(events)
    assert names.count("log_trade") == names.count("log_order") == 1


def test_rows_are_written_once_even_if_the_sizing_check_raises():
    with patch("risk_manager._pt_stop_qty", side_effect=RuntimeError("bug")), \
            pytest.raises(RuntimeError):
        _sale()
    names = _names(_sale.events)
    assert names.count("log_trade") == 1 and "place" not in names


def test_queue_entry_carries_the_floor():
    with patch("risk_manager._floor_at_resting_stop", side_effect=lambda t, s, *a, **k: (s, 99.5)):
        _, events = _sale(place_effect=_raise)
    queue = next(e for e in events if e[0] == "queue")
    assert queue[1] == ("NVDA", {"intent": "PROFIT_TARGET", "stop_price": 100.0, "floor": 99.5}, None)


def test_retry_places_the_stop_when_marking_the_target_fails():
    """A pool error on the flag must not abort the retry pass: the stop
    still goes on and the entry is done."""
    state = {"reprotect_pending": json.dumps({"NVDA": {"intent": "PROFIT_TARGET", "stop_price": 100.0,
                                                      "fill_price": None}})}
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", qty=70.0)]
    placed = []
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks._fresh_read_short", return_value=False), \
         patch("db.get_position_stop", return_value={"stop_price": 90.0, "entry_price": 100.0}), \
         patch("db.mark_profit_target_taken", side_effect=RuntimeError("pool exhausted")), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks._resting_sell_stops", return_value=[]), \
         patch("ibkr.get_latest_price", return_value=None), \
         patch("ibkr.get_fresh_minute_price", return_value=None), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("tasks.manage_stop_place", side_effect=lambda t, q, px, **k: (
             placed.append((t, q, px)), {"order_id": 9})[1]), \
         patch("db.set_stop_broker_order_id"), patch("db.log_activity_queued"), \
         patch("db.update_trailing_stop"), patch("alerts._send"):
        tasks._retry_pending_reprotects()
    assert placed == [("NVDA", 70.0, 100.0)]
    assert json.loads(state["reprotect_pending"]) == {}


# ── a held stop lock means a bot stop change may be under way ───────────────

def test_held_elsewhere_reads_the_key_without_taking_it():
    import stop_lock
    from conftest import _FakeLock
    assert stop_lock.held_elsewhere("NVDA") is False
    _FakeLock.held.add("helios:stop_lock:NVDA")            # another process holds it
    assert stop_lock.held_elsewhere("NVDA") is True
    assert "helios:stop_lock:NVDA" in _FakeLock.held        # not taken or released
    _FakeLock.held.discard("helios:stop_lock:NVDA")


def test_held_elsewhere_is_false_inside_this_threads_own_lock():
    import stop_lock
    with stop_lock.ticker_lock("NVDA"):
        assert stop_lock.held_elsewhere("NVDA") is False
        assert stop_lock.held_elsewhere("AMD") is False


def test_held_elsewhere_unreadable_is_none(monkeypatch):
    import stop_lock

    class Down:
        def exists(self, key):
            raise ConnectionError("redis")
    monkeypatch.setattr(stop_lock, "_redis", lambda: Down())
    assert stop_lock.held_elsewhere("NVDA") is None
