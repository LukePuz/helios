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
