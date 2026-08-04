"""
Tests for ibkr.place_market_order's fill-wait timing — regression coverage
for the 2026-07-28 CRWD incident: the wait loop counted iterations of
ib.waitOnUpdate(timeout=1) as a proxy for elapsed seconds. At 9:30:02 ET
market open, a burst of already-queued order/market-data events (many other
tickers' stop orders re-broadcasting status) made waitOnUpdate return
near-instantly on every call, so the loop burned through all 30 "seconds"
in about one real second and gave up -- logging "no fill" for a BUY that
filled for real 12 seconds later, live and untracked at the broker.
"""
import time
from unittest.mock import MagicMock, patch

import ibkr


class _FakeOrder:
    def __init__(self, order_id=999):
        self.orderId = order_id
        self.tif = None


class _FakeOrderStatus:
    def __init__(self):
        self.status = "PendingSubmit"
        self.filled = 0.0
        self.avgFillPrice = 0.0


class _FakeTrade:
    """isDone() never resolves True within the test -- forces the wait
    loop to run until its own time-based exit condition, not an event."""
    def __init__(self):
        self.order = _FakeOrder()
        self.orderStatus = _FakeOrderStatus()

    def isDone(self):
        return False


class TestPlaceMarketOrderWaitTiming:
    def test_fast_returning_events_do_not_shorten_the_real_wait(self):
        """Old bug: waited += 1 per loop iteration, so a waitOnUpdate that
        returns instantly (event burst) exhausted the "30 seconds" budget
        in a fraction of a real second. The fix must keep looping until
        actual wall-clock time reaches fill_timeout, regardless of how
        fast waitOnUpdate itself returns."""
        fake_ib = MagicMock()
        fake_ib.waitOnUpdate.return_value = True  # returns instantly, every call
        fake_trade = _FakeTrade()
        fake_ib.placeOrder.return_value = fake_trade

        with patch("ibkr.get_ib", return_value=fake_ib), \
             patch("ibkr._stock", return_value=MagicMock()), \
             patch("ibkr.ibi.MarketOrder", return_value=_FakeOrder()):
            start = time.monotonic()
            result = ibkr.place_market_order("NVDA", "BUY", 10, fill_timeout=0.3)
            elapsed = time.monotonic() - start

        assert elapsed >= 0.25, (
            f"wait loop exited after only {elapsed:.3f}s despite fill_timeout=0.3s "
            f"and isDone() never returning True -- the wall-clock fix regressed"
        )
        assert result["filled_qty"] == 0.0
        assert result["fill_price"] is None

    def test_immediate_fill_does_not_wait_the_full_timeout(self):
        """The other direction: isDone() True right away must return fast,
        not busy-wait for the full window."""
        fake_ib = MagicMock()
        fake_ib.waitOnUpdate.return_value = True
        fake_trade = _FakeTrade()
        fake_trade.isDone = MagicMock(return_value=True)
        fake_trade.orderStatus.status = "Filled"
        fake_trade.orderStatus.filled = 10.0
        fake_trade.orderStatus.avgFillPrice = 181.30
        fake_ib.placeOrder.return_value = fake_trade

        with patch("ibkr.get_ib", return_value=fake_ib), \
             patch("ibkr._stock", return_value=MagicMock()), \
             patch("ibkr.ibi.MarketOrder", return_value=_FakeOrder()):
            start = time.monotonic()
            result = ibkr.place_market_order("NVDA", "BUY", 10, fill_timeout=5.0)
            elapsed = time.monotonic() - start

        assert elapsed < 2.0
        assert result["filled_qty"] == 10.0
        assert result["fill_price"] == 181.30


class _FakeStopOrder:
    def __init__(self, order_id, action="SELL", order_type="STP", qty=10.0):
        self.orderId = order_id
        self.action = action
        self.orderType = order_type
        self.totalQuantity = qty
        self.auxPrice = 100.0


class _FakeStopTrade:
    def __init__(self, symbol, order_id, status="PreSubmitted"):
        self.contract = MagicMock(symbol=symbol)
        self.order = _FakeStopOrder(order_id)
        self.orderStatus = MagicMock(status=status)


class TestCancelStopOrdersForVerification:
    """Regression coverage for the 2026-08-03 incident: this used to
    fire-and-forget ib.cancelOrder() and count every matching order as
    cancelled regardless of whether IBKR actually confirmed it -- the same
    silent-failure shape cancel_order() was hardened against on 2026-07-27
    (SNDK), where a client that doesn't own an order can call cancelOrder()
    without IBKR raising anything and the order just stays open. Every
    caller here treats a nonzero return as "safe to place a replacement
    stop", so a silent failure meant a fresh stop landed on top of a still-
    live old one -- exactly what happened to ORCL/SNDK/VRT that day."""

    def test_only_confirmed_cancels_are_counted(self):
        """One order's status actually flips to Cancelled after the
        cancelOrder() call (this connection owns it); the other's status
        never changes (owned by a different clientId) -- only the first
        should count, and the second must be logged as stuck, not silently
        treated as gone."""
        owned_trade   = _FakeStopTrade("ORCL", order_id=8,   status="PreSubmitted")
        foreign_trade = _FakeStopTrade("ORCL", order_id=322, status="PreSubmitted")

        def fake_cancel_order(order):
            if order.orderId == 8:
                owned_trade.orderStatus.status = "Cancelled"
            # foreign_trade's status deliberately does not change --
            # simulates a cross-clientId cancel that IBKR silently ignores

        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [owned_trade, foreign_trade]
        fake_ib.cancelOrder.side_effect = fake_cancel_order

        with patch("ibkr.get_ib", return_value=fake_ib):
            confirmed = ibkr.cancel_stop_orders_for("ORCL")

        assert confirmed == 1
        fake_ib.cancelOrder.assert_any_call(owned_trade.order)
        fake_ib.cancelOrder.assert_any_call(foreign_trade.order)

    def test_all_confirmed_returns_full_count(self):
        trade_a = _FakeStopTrade("KLAC", order_id=1, status="PreSubmitted")
        trade_b = _FakeStopTrade("KLAC", order_id=2, status="PreSubmitted")

        def fake_cancel_order(order):
            {1: trade_a, 2: trade_b}[order.orderId].orderStatus.status = "Cancelled"

        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [trade_a, trade_b]
        fake_ib.cancelOrder.side_effect = fake_cancel_order

        with patch("ibkr.get_ib", return_value=fake_ib):
            confirmed = ibkr.cancel_stop_orders_for("KLAC")

        assert confirmed == 2

    def test_ignores_orders_for_other_tickers_and_non_stop_types(self):
        stop_for_ticker    = _FakeStopTrade("AMD", order_id=1)
        stop_other_ticker  = _FakeStopTrade("NVDA", order_id=2)
        buy_order          = _FakeStopTrade("AMD", order_id=3)
        buy_order.order.action = "BUY"

        def fake_cancel_order(order):
            {1: stop_for_ticker}[order.orderId].orderStatus.status = "Cancelled"

        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [stop_for_ticker, stop_other_ticker, buy_order]
        fake_ib.cancelOrder.side_effect = fake_cancel_order

        with patch("ibkr.get_ib", return_value=fake_ib):
            confirmed = ibkr.cancel_stop_orders_for("AMD")

        assert confirmed == 1
        fake_ib.cancelOrder.assert_called_once_with(stop_for_ticker.order)
