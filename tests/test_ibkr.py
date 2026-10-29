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


class TestCancelOrderPollsForTerminalStatus:
    """Regression coverage for the 2026-08-04 MRVL incident: cancel_order
    used to accept "PendingCancel" itself as confirmed -- a genuinely
    transient in-flight state, not a terminal one. Live in production,
    a cross-clientId cancel flipped to PendingCancel within a second
    (exactly what the old check trusted) then silently reverted to
    PreSubmitted three minutes later once IBKR finished rejecting it --
    by then a replacement stop had already been placed on top, believing
    the old one gone. Now polls for a few seconds for a genuinely
    terminal status instead of trusting the first read."""

    def _trade(self, order_id=1, symbol="NVDA"):
        return _FakeStopTrade(symbol, order_id=order_id, status="PreSubmitted")

    def test_immediate_cancelled_status_returns_true_fast(self):
        """A same-clientId cancel that resolves on the very first check
        must not need to burn the whole poll window."""
        trade = self._trade()
        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [trade]
        fake_ib.sleep.side_effect = lambda _: setattr(trade.orderStatus, "status", "Cancelled")

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.cancel_order(1)

        assert result is True
        assert fake_ib.sleep.call_count == 1

    def test_pending_cancel_that_resolves_to_cancelled_returns_true(self):
        trade = self._trade()
        statuses = iter(["PendingCancel", "PendingCancel", "Cancelled"])
        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [trade]
        fake_ib.sleep.side_effect = lambda _: setattr(
            trade.orderStatus, "status", next(statuses, "Cancelled"))

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.cancel_order(1)

        assert result is True

    def test_pending_cancel_that_never_resolves_returns_false(self):
        trade = self._trade(order_id=2167, symbol="MRVL")
        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [trade]
        fake_ib.sleep.side_effect = lambda _: setattr(trade.orderStatus, "status", "PendingCancel")

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.cancel_order(2167)

        assert result is False

    def test_pending_cancel_that_reverts_returns_false(self):
        """The exact MRVL sequence: PendingCancel, then reverts back to
        PreSubmitted -- must not be trusted just because it briefly looked
        like a real cancel was in flight."""
        trade = self._trade(order_id=2167, symbol="MRVL")
        statuses = iter(["PendingCancel", "PendingCancel", "PreSubmitted",
                          "PreSubmitted", "PreSubmitted"])
        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [trade]
        fake_ib.sleep.side_effect = lambda _: setattr(
            trade.orderStatus, "status", next(statuses, "PreSubmitted"))

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.cancel_order(2167)

        assert result is False


class _FakeExecution:
    def __init__(self, order_id, shares, avg_price, side, time=None):
        self.time = time
        self.orderId = order_id
        self.shares = shares
        self.avgPrice = avg_price
        self.side = side  # "BOT" or "SLD"


class _FakeFillRecord:
    def __init__(self, symbol, order_id, shares, avg_price, side, time=None):
        self.contract = MagicMock(symbol=symbol)
        self.execution = _FakeExecution(order_id, shares, avg_price, side, time)


class TestGetRecentSellFill:
    """Regression coverage for the 2026-08-06 DDOG incident: a position
    can disappear from the broker (its own tracked stop order never
    fills, or someone sells it directly in TWS) with no bot-initiated
    sell on record. This searches by symbol rather than a specific
    orderId so reconcile_stop_orders can find that exit through ANY sell
    order for the ticker, not just the one this system happened to be
    watching."""

    def test_finds_a_sell_by_symbol_ignoring_other_tickers_and_buys(self):
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [
            _FakeFillRecord("DDOG", order_id=11, shares=84.0, avg_price=227.80, side="SLD"),
            _FakeFillRecord("DDOG", order_id=99, shares=10.0, avg_price=500.0, side="BOT"),  # different order, a BUY -- must be ignored
            _FakeFillRecord("NVDA", order_id=12, shares=50.0, avg_price=200.0, side="SLD"),  # different ticker -- must be ignored
        ]

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.get_recent_sell_fill("DDOG")

        assert result["filled_qty"] == 84.0
        assert result["fill_price"] == 227.80

    def test_aggregates_partial_sells_across_multiple_orders(self):
        """The exit doesn't have to be the tracked stop order -- e.g. a
        manual sale placed directly in TWS uses a different orderId
        entirely, and must still be found and aggregated correctly."""
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [
            _FakeFillRecord("DDOG", order_id=11, shares=50.0, avg_price=228.00, side="SLD"),
            _FakeFillRecord("DDOG", order_id=847, shares=34.0, avg_price=226.50, side="SLD"),
        ]

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.get_recent_sell_fill("DDOG")

        assert result["filled_qty"] == 84.0
        expected_avg = (50.0 * 228.00 + 34.0 * 226.50) / 84.0
        assert result["fill_price"] == expected_avg

    def test_no_sell_execution_today_returns_none(self):
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [
            _FakeFillRecord("DDOG", order_id=99, shares=10.0, avg_price=500.0, side="BOT"),
        ]

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.get_recent_sell_fill("DDOG")

        assert result is None

    def test_no_executions_at_all_returns_none(self):
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = []

        with patch("ibkr.get_ib", return_value=fake_ib):
            result = ibkr.get_recent_sell_fill("DDOG")

        assert result is None


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


class TestGetFillForOrderMatching:
    """Professor M-4: orderIds are per clientId and reqExecutions returns all
    clients' executions, so the id alone can match an unrelated order."""

    def _lookup(self, records, **kw):
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = records
        with patch("ibkr.get_ib", return_value=fake_ib):
            return ibkr.get_fill_for_order(764, **kw)

    def test_same_id_other_ticker_or_side_is_ignored(self):
        recs = [_FakeFillRecord("NVDA", 764, 10.0, 200.0, "BOT"),
                _FakeFillRecord("CRWD", 764, 5.0, 180.0, "SLD"),
                _FakeFillRecord("CRWD", 764, 144.0, 181.3, "BOT")]
        r = self._lookup(recs, ticker="CRWD", side="BUY")
        assert r == {"filled_qty": 144.0, "fill_price": 181.3}

    def test_execution_before_the_order_was_placed_is_ignored(self):
        from datetime import datetime, timezone, timedelta
        placed = datetime(2026, 10, 28, 14, 0, tzinfo=timezone.utc)
        recs = [_FakeFillRecord("CRWD", 764, 144.0, 181.3, "BOT", time=placed - timedelta(hours=1))]
        assert self._lookup(recs, ticker="CRWD", side="BUY", placed_at=placed) is None
        recs = [_FakeFillRecord("CRWD", 764, 144.0, 181.3, "BOT", time=placed + timedelta(seconds=40))]
        assert self._lookup(recs, ticker="CRWD", side="BUY", placed_at=placed)["filled_qty"] == 144.0


class TestGetOpenOrdersSnapshot:
    """A0053-1: a stop the orders worker cancelled stays in another
    process's openTrades() cache; only the fresh snapshot counts."""

    def test_cancelled_stop_in_stale_cache_is_not_reported(self):
        live = _FakeStopTrade("PLTR", 1)
        ghost = _FakeStopTrade("PLTR", 2)  # cancelled by another client
        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [live, ghost]
        fake_ib.reqAllOpenOrders.return_value = [live, live]
        with patch.object(ibkr, "get_ib", return_value=fake_ib):
            orders = ibkr.get_open_orders()
        assert [o["order_id"] for o in orders] == [1]
        fake_ib.openTrades.assert_not_called()


def test_trail_stop_price_is_the_trigger_not_the_distance():
    """A0053-4/A0653-4: auxPrice on a trail_amount TRAIL is dollars of trail."""
    trail = _FakeStopTrade("PLTR", 7)
    trail.order.orderType = "TRAIL"
    trail.order.auxPrice = 12.0
    trail.order.trailStopPrice = 185.5
    fake_ib = MagicMock()
    fake_ib.reqAllOpenOrders.return_value = [trail]
    with patch.object(ibkr, "get_ib", return_value=fake_ib):
        [o] = ibkr.get_open_orders()
    assert o["stop_price"] == 185.5


def test_sell_orders_refused_on_a_short():
    """Professor M63 residual: one check where every SELL is placed."""
    import pytest
    fake_ib = MagicMock()
    fake_ib.positions.return_value = [MagicMock(contract=MagicMock(symbol="AMD"), position=-3)]
    fake_ib.reqPositions.return_value = fake_ib.positions.return_value
    with patch.object(ibkr, "get_ib", return_value=fake_ib):
        for call in (lambda: ibkr.place_stop_order("AMD", 3, 90.0),
                     lambda: ibkr.place_trailing_stop_order("AMD", 3, trail_percent=8.0),
                     lambda: ibkr.place_market_order("AMD", "SELL", 3)):
            with pytest.raises(ibkr.SellExceedsPosition):
                call()
    fake_ib.placeOrder.assert_not_called()


def test_sell_above_the_long_position_is_refused_after_a_fresh_read():
    """Run 11 follow-up: flat or smaller long also opens a short; a lagging
    cache is re-read before refusing."""
    import pytest
    pos = lambda q: [MagicMock(contract=MagicMock(symbol="AMD"), position=q)]
    fake_ib = MagicMock()
    fake_ib.positions.return_value = pos(2)
    fake_ib.reqPositions.return_value = pos(2)
    with patch.object(ibkr, "get_ib", return_value=fake_ib):
        with pytest.raises(ibkr.SellExceedsPosition):
            ibkr.place_stop_order("AMD", 3, 90.0)
        fake_ib.positions.return_value, fake_ib.reqPositions.return_value = [], []
        with pytest.raises(ibkr.SellExceedsPosition):
            ibkr._refuse_sell_on_short(fake_ib, "AMD", 1)
        # cache lags after a fill, fresh read shows the shares: allowed
        fake_ib.positions.return_value, fake_ib.reqPositions.return_value = pos(0), pos(3)
        ibkr._refuse_sell_on_short(fake_ib, "AMD", 3)


def test_unset_price_fields_read_as_no_price():
    """A0853-4: ib_async UNSET_DOUBLE must not become a stop price."""
    assert ibkr._price_or_none(1.7976931348623157e308) is None
    assert ibkr._price_or_none(0) is None and ibkr._price_or_none(None) is None
    assert ibkr._price_or_none(185.5) == 185.5
