"""_book_partial_stop_fills / _net_of_partial_bookings: a SELL the bot didn't
place that closes part of a still-held long is booked once, and the stale-stop
exit later nets it out (roadmap 04z, NVDA 11-12; review B1, S1-S3)."""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import tasks
from config import TRADING_ENV

KEY = tasks._PARTIAL_STOP_BOOKED_KEY.format(env=TRADING_ENV)
NOW = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(hours=5)
DAY = "2026-11-12"
ENTRY = NOW - timedelta(days=2)
BUY = lambda sym, q, ts=ENTRY: {"symbol": sym, "side": "BUY", "quantity": q, "timestamp": ts}
SELL = lambda sym, q, ts=ENTRY: {"symbol": sym, "side": "SELL", "quantity": q, "timestamp": ts}


def _st(perms, day=DAY):
    return json.dumps({"day": day, "perms": perms})


def _order(oid, perm, qty, price, ft=OLD):
    return {"order_id": oid, "perm_id": perm, "qty": qty, "price": price, "fill_time": ft}


def _fill(*orders):
    total = sum(o["qty"] for o in orders)
    return {"filled_qty": total, "fill_price": sum(o["qty"] * o["price"] for o in orders) / total,
            "order_ids": [o["order_id"] for o in orders], "by_order": list(orders)}


NVDA_FILL = _fill(_order(42, 9001, 1.0, 152.4))


def _run(held, ledger, fill, state=None, recorded=(set(), set()), log_ok=True,
         pending=frozenset(), skip=frozenset(), verify_fails=False, read_err=None):
    state = {} if state is None else state
    calls = {"trades": [], "alerts": [], "slip": [], "lookups": 0}

    def _get_strict(key, default=""):
        if read_err:
            raise read_err
        return state.get(key, default)

    def _verified(writes):
        if verify_fails:
            return [k for k, _ in writes]
        for k, v in writes:
            state[k] = v
        return []

    def _lookup(*a, **k):
        calls["lookups"] += 1
        return fill

    with patch("tasks._et_today", return_value=DAY), \
         patch("db.get_ledger_trades", return_value=ledger), \
         patch("db.get_state_strict", side_effect=_get_strict), \
         patch("db.set_state_verified", side_effect=_verified), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.recorded_sell_order_ids", return_value=recorded), \
         patch("db.log_trade", side_effect=lambda **kw: calls["trades"].append(kw) or log_ok), \
         patch("ibkr.get_recent_sell_fill", side_effect=_lookup), \
         patch("tasks._log_stop_slippage", side_effect=lambda *a, **k: calls["slip"].append(a)), \
         patch("alerts._send", side_effect=lambda m: calls["alerts"].append(m)):
        msgs = tasks._book_partial_stop_fills(
            held, [{"ticker": "NVDA", "stop_price": 146.24}], pending, skip=skip, now=NOW)
    return msgs, calls, state


def _booked(state):
    raw = state.get(KEY)
    return json.loads(raw)["perms"] if raw else {}


def test_nvda_11_12_first_lot_is_booked():
    msgs, calls, state = _run({"NVDA": 1.0}, [BUY("NVDA", 1), BUY("NVDA", 1)], NVDA_FILL)
    assert len(calls["trades"]) == 1
    t = calls["trades"][0]
    assert (t["symbol"], t["side"], t["quantity"], t["price"], t["price_source"]) == \
        ("NVDA", "SELL", 1.0, 152.4, "fill")
    assert len(calls["alerts"]) == 1 and "NVDA" in calls["alerts"][0]
    assert calls["slip"] == []   # stop_slippage keeps its whole-exit meaning (review r3)
    assert _booked(state) == {"9001": 1.0} and len(msgs) == 1


def test_booked_shares_are_not_booked_again():
    state = {KEY: _st({"9001": 1.0})}
    # Even if the ledger write had been lost, the booked state alone prevents a repeat.
    msgs, calls, _ = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, state=state)
    assert calls["trades"] == [] and msgs == []


def test_after_booking_the_ledger_gap_closes():
    msgs, calls, _ = _run({"NVDA": 1.0}, [BUY("NVDA", 2), SELL("NVDA", 1)], NVDA_FILL)
    assert msgs == [] and calls["lookups"] == 0


def test_remainder_of_a_partly_booked_order_is_booked_later():   # review S1
    state = {KEY: _st({"9001": 1.0})}
    fill = _fill(_order(42, 9001, 2.0, 150.0))
    msgs, calls, state = _run({"NVDA": 1.0}, [BUY("NVDA", 3), SELL("NVDA", 1)], fill, state=state)
    assert [t["quantity"] for t in calls["trades"]] == [1.0]
    assert _booked(state) == {"9001": 2.0}


def test_order_larger_than_the_gap_is_not_booked():   # review S1: no partial cap
    fill = _fill(_order(42, 1, 3.0, 10.0))
    msgs, calls, state = _run({"NVDA": 1.0}, [BUY("NVDA", 3)], fill)
    assert calls["trades"] == [] and _booked(state) == {}


def test_pending_bot_sell_skips_the_ticker():   # review S2
    msgs, calls, _ = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, pending={"NVDA"})
    assert calls["lookups"] == 0 and calls["trades"] == []


def test_unknown_pending_state_skips_everything():   # review S2
    msgs, calls, _ = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, pending=None)
    assert calls["lookups"] == 0 and calls["trades"] == []


def test_young_execution_waits():   # review S2
    fill = _fill(_order(42, 9001, 1.0, 152.4, ft=NOW - timedelta(seconds=60)))
    msgs, calls, _ = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], fill)
    assert calls["trades"] == []


def test_disagreeing_reads_skip_the_ticker():
    msgs, calls, _ = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, skip={"NVDA"})
    assert calls["lookups"] == 0


def test_state_write_failure_books_nothing():   # review S3
    msgs, calls, state = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, verify_fails=True)
    assert calls["trades"] == [] and calls["alerts"] == []


def test_spilled_trade_write_keeps_the_booking():   # review SF2
    msgs, calls, state = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, log_ok=False)
    assert calls["alerts"] == [] and msgs == [] and _booked(state) == {"9001": 1.0}
    again = _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, state=state)[1]
    assert again["trades"] == []


def test_sell_before_the_latest_buy_is_not_booked():   # review SF1: exit then re-entry
    ledger = [BUY("NVDA", 2), SELL("NVDA", 2, NOW - timedelta(hours=4)),
              BUY("NVDA", 2, NOW - timedelta(hours=2)), BUY("NVDA", 1, NOW - timedelta(hours=1))]
    # The 10:00 stop execution is older than the re-entry BUYs.
    msgs, calls, _ = _run({"NVDA": 2.0}, ledger, NVDA_FILL)
    assert calls["trades"] == []


def test_execution_without_a_timezone_aware_time_is_not_booked():
    fill = _fill(_order(42, 9001, 1.0, 152.4, ft=None))
    assert _run({"NVDA": 1.0}, [BUY("NVDA", 2)], fill)[1]["trades"] == []


def test_yesterdays_bookings_read_as_empty():
    with patch("tasks._et_today", return_value=DAY), \
         patch("db.get_state_strict", return_value=_st({"9001": 1.0}, day="2026-11-11")):
        assert tasks._partial_booked_read() == {}


def test_corrupt_booked_state_alerts_and_raises():
    sent = []
    with patch("tasks._et_today", return_value=DAY), \
         patch("db.get_state_strict", return_value="not json"), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("alerts._send", side_effect=sent.append), pytest.raises(ValueError):
        tasks._partial_booked_read()
    assert sent and "unreadable" in sent[0]


def test_full_exit_is_left_to_the_stale_stop_path():
    msgs, calls, _ = _run({}, [BUY("NVDA", 2)], NVDA_FILL)
    assert msgs == [] and calls["trades"] == []


def test_no_unrecorded_sell_and_missing_perm_book_nothing():
    assert _run({"NVDA": 1.0}, [BUY("NVDA", 2)], None)[1]["trades"] == []
    fill = _fill(_order(42, None, 1.0, 10.0))
    assert _run({"NVDA": 1.0}, [BUY("NVDA", 2)], fill)[1]["trades"] == []


def test_unreadable_state_ledger_or_recorded_lookup_wait():
    assert _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, read_err=RuntimeError("db"))[1]["lookups"] == 0
    assert _run({"NVDA": 1.0}, None, NVDA_FILL)[1]["lookups"] == 0
    assert _run({"NVDA": 1.0}, [BUY("NVDA", 2)], NVDA_FILL, recorded=None)[1]["lookups"] == 0


def test_never_raises():
    with patch("db.get_ledger_trades", side_effect=RuntimeError("boom")):
        assert tasks._book_partial_stop_fills({"NVDA": 1.0}, [], set()) == []


# --- _net_of_partial_bookings: what the stale-stop exit sees (review B1) ---

def test_stale_exit_after_a_partial_booking_counts_only_the_last_lot():
    fill = _fill(_order(42, 9001, 1.0, 152.4), _order(46, 9002, 1.0, 140.0))
    out = tasks._net_of_partial_bookings(fill, {9001: 1.0})
    assert out["filled_qty"] == 1.0 and out["fill_price"] == pytest.approx(140.0)
    assert out["order_ids"] == [46] and [o["perm_id"] for o in out["by_order"]] == [9002]


def test_netting_leaves_unrelated_fills_alone():
    assert tasks._net_of_partial_bookings(NVDA_FILL, {}) is NVDA_FILL
    assert tasks._net_of_partial_bookings(NVDA_FILL, {5555: 1.0}) is NVDA_FILL
    assert tasks._net_of_partial_bookings(None, {9001: 1.0}) is None


def test_netting_everything_returns_none():
    assert tasks._net_of_partial_bookings(NVDA_FILL, {9001: 1.0}) is None


def test_netting_without_per_order_split_raises():
    with pytest.raises(ValueError):
        tasks._net_of_partial_bookings({"filled_qty": 1.0, "fill_price": 1.0}, {9001: 1.0})


def test_booked_state_read_rejects_non_mapping():
    with patch("db.get_state_strict", return_value="[9001]"), \
         patch("tasks._reconcile_alert_due", return_value=False), pytest.raises(ValueError):
        tasks._partial_booked_read()
    with patch("db.get_state_strict", return_value=""):
        assert tasks._partial_booked_read() == {}


def test_netted_fill_time_comes_from_the_kept_order():
    a = _order(42, 9001, 1.0, 152.4, ft=NOW - timedelta(hours=1))
    b = _order(46, 9002, 1.0, 140.0, ft=NOW - timedelta(hours=3))
    fill = {**_fill(a, b), "fill_time": a["fill_time"]}
    assert tasks._net_of_partial_bookings(fill, {9001: 1.0})["fill_time"] == b["fill_time"]
