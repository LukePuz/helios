"""ibkr.place_market_order's market-closed cancel is confirmed, not assumed.

It used to send cancelOrder, sleep one second and report "Cancelled". A SELL
reported that way is treated as dead by execution (the full-size stop is
restored, the order row marked cancelled), so a fill landing after the
second left a full stop resting on sold shares with no booking. Now the
status is polled like cancel_order's (_poll_after_cancel): Cancelled /
ApiCancelled only when the broker says so, a fill is returned as a fill,
Inactive is returned as dead, and anything else is returned as it is.

ib_async's error handler sets the status to Cancelled locally on any order
error, a rejected cancel (10148, 161, 10147) among them. A Cancelled counts
only when the newest trade.log entry with that status has error code 0 (an
orderStatus) or 202 (Order Canceled); otherwise it comes back as
ibkr.CANCEL_REJECTED_STATUS, which the callers don't treat as dead.
"""
import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ib_async.objects import TradeLogEntry

import ibkr
import execution


class _Order:
    def __init__(self, order_id=4242):
        self.orderId = order_id
        self.permId = 0
        self.tif = None


class _Status:
    def __init__(self, status="PreSubmitted"):
        self.status = status
        self.filled = 0.0
        self.avgFillPrice = 0.0


class _Trade:
    def __init__(self, status="PreSubmitted"):
        self.order = _Order()
        self.orderStatus = _Status(status)
        self.fills = []
        self.log = []

    def isDone(self):
        return self.orderStatus.status in ("Filled", "Cancelled", "ApiCancelled", "Inactive")

    def set(self, status, error_code=0, filled=None, avg=None):
        """The status change and its trade.log entry, as ib_async's
        orderStatus (error code 0) or error handler (the code) makes them."""
        self.orderStatus.status = status
        if filled is not None:
            self.orderStatus.filled = filled
        if avg is not None:
            self.orderStatus.avgFillPrice = avg
        self.log.append(TradeLogEntry(datetime.now(timezone.utc), status, "", error_code))


def _run(trade, on_sleep=None, market_open=False, side="SELL"):
    """place_market_order with a fake session; on_sleep(n) runs on the n-th
    ib.sleep (1-based; the first is the quote subscription's)."""
    ib = MagicMock()
    ib.placeOrder.return_value = trade
    calls = {"n": 0}

    def _sleep(_):
        calls["n"] += 1
        if on_sleep is not None:
            on_sleep(calls["n"])
    ib.sleep.side_effect = _sleep
    with patch("ibkr.get_ib", return_value=ib), \
         patch("ibkr._stock", return_value=MagicMock()), \
         patch("ibkr._refuse_sell_on_short"), \
         patch("ibkr._record_order_quote"), \
         patch("ibkr.ibi.MarketOrder", return_value=trade.order), \
         patch("market_data.is_market_hours", return_value=market_open):
        result = ibkr.place_market_order("NVDA", side, 10, fill_timeout=0.05)
    return result, ib, calls["n"]


def test_a_confirmed_cancel_is_reported_cancelled():
    trade = _Trade()

    def on_sleep(n):
        if n == 3:      # the second poll read
            trade.set("Cancelled")
    result, ib, sleeps = _run(trade, on_sleep)
    ib.cancelOrder.assert_called_once_with(trade.order)
    assert result["status"] == "Cancelled"
    assert result["filled_qty"] == 0.0
    assert result["fill_price"] is None
    assert sleeps == 3   # stops polling at the confirmation


def test_an_api_cancel_is_reported_as_the_broker_said():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.set("ApiCancelled")
    result, _, _ = _run(trade, on_sleep)
    assert result["status"] == "ApiCancelled"


def test_a_fill_during_the_cancel_is_returned_as_the_fill():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.set("Filled", filled=10.0, avg=101.25)
    result, _, sleeps = _run(trade, on_sleep)
    assert result["status"] == "Filled"
    assert result["filled_qty"] == 10.0
    assert result["fill_price"] == 101.25
    assert sleeps == 2


def test_a_part_fill_then_cancel_carries_the_part_fill():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.set("Cancelled", 202, filled=4.0, avg=99.5)
    result, _, _ = _run(trade, on_sleep)
    assert result["status"] == "Cancelled"
    assert result["filled_qty"] == 4.0
    assert result["fill_price"] == 99.5


def test_no_terminal_status_within_the_bound_returns_the_real_status():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.set("PendingCancel")
    result, ib, sleeps = _run(trade, on_sleep)
    ib.cancelOrder.assert_called_once_with(trade.order)
    assert result["status"] == "PendingCancel"
    assert result["filled_qty"] == 0.0
    assert sleeps == 1 + ibkr.MARKET_CLOSED_CANCEL_POLL_S


def test_a_status_that_never_moves_is_returned_unchanged():
    result, _, _ = _run(_Trade("Submitted"))
    assert result["status"] == "Submitted"


def test_the_returned_pending_statuses_are_not_dead_to_the_callers():
    """execute_approved and the manual path restore the stop only on a
    status in _DEAD_ORDER_STATUSES; a pending one takes the unconfirmed
    path (an orders row for reconcile, the stop left down)."""
    for status in ("PendingCancel", "PreSubmitted", "PendingSubmit", "Submitted"):
        assert status not in execution._DEAD_ORDER_STATUSES


def test_the_poll_is_no_longer_than_cancel_orders():
    default = inspect.signature(ibkr.cancel_order).parameters["_poll_seconds"].default
    assert ibkr.MARKET_CLOSED_CANCEL_POLL_S <= default


def test_the_open_market_path_sends_no_cancel():
    trade = _Trade()
    result, ib, sleeps = _run(trade, market_open=True)
    ib.cancelOrder.assert_not_called()
    assert result["status"] == "PreSubmitted"
    assert sleeps == 1   # the quote subscription's only


def test_a_raise_during_the_poll_leaves_the_pending_status():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            raise ConnectionError("dropped")
    result, ib, _ = _run(trade, on_sleep)
    ib.cancelOrder.assert_called_once_with(trade.order)
    assert result["status"] == "PreSubmitted"


def test_a_cancel_confirmed_by_code_202_is_cancelled():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.set("Cancelled", 202)
    result, _, sleeps = _run(trade, on_sleep)
    assert result["status"] == "Cancelled"
    assert sleeps == 2


def test_a_rejected_cancel_is_not_reported_cancelled():
    """A Cancelled set by error 10148 (the cancel rejected, the order
    filling): polled to the bound, returned as CancelRejected."""
    for code in (10148, 161, 10147):
        trade = _Trade("Submitted")

        def on_sleep(n, trade=trade, code=code):
            if n == 2:
                trade.set("Cancelled", code)
        result, _, sleeps = _run(trade, on_sleep)
        assert result["status"] == ibkr.CANCEL_REJECTED_STATUS
        assert result["filled_qty"] == 0.0
        assert sleeps == 1 + ibkr.MARKET_CLOSED_CANCEL_POLL_S


def test_a_fill_after_a_rejected_cancel_is_returned_as_the_fill():
    trade = _Trade("Submitted")

    def on_sleep(n):
        if n == 2:
            trade.set("Cancelled", 10148)
        if n == 4:
            trade.set("Filled", filled=10.0, avg=100.75)
    result, _, sleeps = _run(trade, on_sleep)
    assert result["status"] == "Filled"
    assert result["filled_qty"] == 10.0
    assert result["fill_price"] == 100.75
    assert sleeps == 4


def test_a_part_fill_with_a_rejected_cancel_carries_the_part_fill():
    trade = _Trade("Submitted")

    def on_sleep(n):
        if n == 2:
            trade.orderStatus.filled = 3.0
            trade.orderStatus.avgFillPrice = 100.5
            trade.set("Cancelled", 10148)
    result, _, _ = _run(trade, on_sleep)
    assert result["status"] == ibkr.CANCEL_REJECTED_STATUS
    assert result["filled_qty"] == 3.0
    assert result["fill_price"] == 100.5


def test_the_newest_entry_with_the_status_decides():
    trade = _Trade()
    trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "Cancelled", "", 0))
    trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "Cancelled", "", 10148))
    trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "Submitted", "", 0))
    assert ibkr._cancel_confirmed_by_log(trade, "Cancelled") is False
    trade.log.append(TradeLogEntry(datetime.now(timezone.utc), "Cancelled", "", 202))
    assert ibkr._cancel_confirmed_by_log(trade, "Cancelled") is True


def test_a_cancelled_with_no_log_entry_is_not_confirmed():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.orderStatus.status = "Cancelled"     # no trade.log entry
    result, _, _ = _run(trade, on_sleep)
    assert result["status"] == ibkr.CANCEL_REJECTED_STATUS


def test_a_trade_without_a_usable_log_is_not_confirmed():
    trade = MagicMock()
    trade.log = None
    assert ibkr._cancel_confirmed_by_log(trade, "Cancelled") is False


def test_inactive_stops_the_poll_and_is_dead():
    trade = _Trade()

    def on_sleep(n):
        if n == 2:
            trade.set("Inactive", 201)
    result, _, sleeps = _run(trade, on_sleep)
    assert result["status"] == "Inactive"
    assert sleeps == 2
    assert "Inactive" in execution._DEAD_ORDER_STATUSES


def test_cancel_rejected_is_not_dead_to_the_callers():
    assert ibkr.CANCEL_REJECTED_STATUS not in execution._DEAD_ORDER_STATUSES
