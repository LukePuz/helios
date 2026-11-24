"""ibkr.cancel_stop_orders_for with config.CANCEL_PARALLEL_ENABLED: 2+ stops
get every cancelOrder first and one poll loop bounded by the one-stop 5 s;
the confirmation rule, the return value, the order matching and the stuck
warning are the same as the one-at-a-time path. One stop runs today's path."""
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import config
import ibkr


class _Trade:
    def __init__(self, oid, symbol="NVDA", perm=None, order_type="STP", action="SELL"):
        self.order = SimpleNamespace(orderId=oid, permId=perm if perm is not None else 1000 + oid,
                                     action=action, orderType=order_type, totalQuantity=1.0)
        self.contract = SimpleNamespace(symbol=symbol)
        self.orderStatus = SimpleNamespace(status="PreSubmitted")


DONE = ("Filled", "Cancelled", "ApiCancelled", "Inactive")


class _FakeIB:
    """A clock that ib.sleep(1) advances. outcome[orderId] = (seconds after
    its cancel, final status); no entry: the cancel never resolves. raise_on:
    orderIds whose cancelOrder raises. later_trades: what openTrades returns
    after the first call (a stop gone between the match read and its cancel).
    As in ib_async 2.1.0: openTrades leaves out done trades, and cancelOrder
    sets a trade that isn't done to PendingCancel locally."""

    def __init__(self, trades, outcome=None, raise_on=(), later_trades=None):
        self.trades, self.outcome, self.raise_on = trades, outcome or {}, set(raise_on)
        self.later = later_trades
        self.t, self.sent, self.calls, self._reads = 0, {}, [], 0

    def openTrades(self):
        self._reads += 1
        self.calls.append("openTrades")
        src = self.trades if self.later is None or self._reads == 1 else self.later
        return [t for t in src if t.orderStatus.status not in DONE]

    def cancelOrder(self, order):
        self.calls.append(("cancelOrder", order.orderId, order.permId))
        if order.orderId in self.raise_on:
            raise ConnectionError("not connected")
        self.sent[id(order)] = self.t
        for tr in self.trades:
            if tr.order is order and tr.orderStatus.status not in DONE:
                tr.orderStatus.status = "PendingCancel"

    def sleep(self, s):
        self.calls.append("sleep")
        self.t += s
        for tr in self.trades:
            at = self.sent.get(id(tr.order))
            res = self.outcome.get(tr.order.orderId)
            if at is not None and res and self.t - at >= res[0]:
                tr.orderStatus.status = res[1]


def _run(fake, parallel, ticker="NVDA"):
    with patch("ibkr.get_ib", return_value=fake), \
         patch.object(config, "CANCEL_PARALLEL_ENABLED", parallel):
        try:
            return ibkr.cancel_stop_orders_for(ticker), None
        except Exception as e:
            return None, e


def _both(make, caplog, **kw):
    """Run a scenario both ways on fresh fakes: (sequential, parallel), each
    (result, exception, fake, stuck-warning lines)."""
    out = []
    for parallel in (False, True):
        caplog.clear()
        fake = make()
        res, exc = _run(fake, parallel, **kw)
        warn = [r.getMessage() for r in caplog.records if "did NOT actually cancel" in r.getMessage()]
        out.append((res, exc, fake, warn))
    return out


@pytest.fixture(autouse=True)
def _log(caplog):
    caplog.set_level(logging.INFO, logger="ibkr")


def test_switch_ships_off():
    assert config.CANCEL_PARALLEL_ENABLED is False


def test_one_stop_runs_todays_path_exactly(caplog):
    for outcome in ({1: (1, "Cancelled")}, {}, {1: (2, "Filled")}):
        seq, par = _both(lambda: _FakeIB([_Trade(1)], outcome), caplog)
        assert par[0] == seq[0] and par[3] == seq[3]
        assert par[2].calls == seq[2].calls                   # same calls, same order
    with patch.object(ibkr, "_cancel_stops_parallel") as p, \
         patch("ibkr.get_ib", return_value=_FakeIB([_Trade(1)], {1: (1, "Cancelled")})), \
         patch.object(config, "CANCEL_PARALLEL_ENABLED", True):
        assert ibkr.cancel_stop_orders_for("NVDA") == 1
    p.assert_not_called()


@pytest.mark.parametrize("n", [2, 3])
def test_all_cancels_sent_before_one_poll_loop(n, caplog):
    make = lambda: _FakeIB([_Trade(i) for i in range(1, n + 1)],
                           {i: (1, "Cancelled") for i in range(1, n + 1)})
    seq, par = _both(make, caplog)
    assert seq[0] == par[0] == n and par[3] == seq[3] == []
    calls = par[2].calls
    cancels = [i for i, c in enumerate(calls) if isinstance(c, tuple)]
    assert len(cancels) == n and max(cancels) < calls.index("sleep")    # every cancel, then sleeps
    assert calls.count("sleep") == 1 and seq[2].calls.count("sleep") == n
    assert par[2].t <= 5


def test_slow_cancels_share_one_window(caplog):
    make = lambda: _FakeIB([_Trade(1), _Trade(2), _Trade(3)],
                           {1: (4, "Cancelled"), 2: (5, "ApiCancelled"), 3: (3, "Cancelled")})
    seq, par = _both(make, caplog)
    assert seq[0] == par[0] == 3
    assert par[2].t == 5 and seq[2].t == 12


def test_one_stuck_stop_is_reported_the_same(caplog):
    make = lambda: _FakeIB([_Trade(1), _Trade(2), _Trade(3)],
                           {1: (1, "Cancelled"), 3: (2, "Cancelled")})      # 2 never resolves
    seq, par = _both(make, caplog)
    assert seq[0] == par[0] == 2
    assert par[3] == seq[3] and "orderIds=[2]" in par[3][0]
    assert par[2].t == 5 and seq[2].t == 1 + 5 + 2
    assert any("Cancel sent for orderId=2 but status is still 'PendingCancel' after 5s" in
               r.getMessage() for r in caplog.records)


def test_pending_cancel_then_revert_is_not_confirmed(caplog):
    make = lambda: _FakeIB([_Trade(1), _Trade(2)], {1: (1, "PendingCancel"), 2: (1, "Cancelled")})
    seq, par = _both(make, caplog)
    assert seq[0] == par[0] == 1 and par[3] == seq[3] and "orderIds=[1]" in par[3][0]


def test_a_stop_that_fills_during_the_cancel_is_reported_as_today(caplog):
    make = lambda: _FakeIB([_Trade(1), _Trade(2)], {1: (1, "Filled"), 2: (1, "Cancelled")})
    seq, par = _both(make, caplog)
    assert seq[0] == par[0] == 1
    assert par[3] == seq[3] and "orderIds=[1]" in par[3][0]
    assert any("orderId=1 but status is still 'Filled'" in r.getMessage() for r in caplog.records)


def test_cancel_error_on_the_second_of_three_stops_the_sending_like_today(caplog):
    make = lambda: _FakeIB([_Trade(1), _Trade(2), _Trade(3)],
                           {1: (1, "Cancelled"), 3: (1, "Cancelled")}, raise_on={2})
    seq, par = _both(make, caplog)
    sent = lambda f: [c[1] for c in f.calls if isinstance(c, tuple)]
    # both: 1 and 2 sent, the raise reaches the caller, 3 is never sent
    assert isinstance(seq[1], ConnectionError) and isinstance(par[1], ConnectionError)
    assert sent(seq[2]) == sent(par[2]) == [1, 2]
    # parallel polls the one already sent before re-raising
    assert [t.orderStatus.status for t in par[2].trades] == ["Cancelled", "PreSubmitted", "PreSubmitted"]
    assert par[2].t == 1
    assert par[3] and "orderIds=[2, 3]" in par[3][0]


def test_cancel_error_on_the_first_sends_nothing_else(caplog):
    make = lambda: _FakeIB([_Trade(1), _Trade(2)], {2: (1, "Cancelled")}, raise_on={1})
    seq, par = _both(make, caplog)
    sent = lambda f: [c[1] for c in f.calls if isinstance(c, tuple)]
    assert sent(seq[2]) == sent(par[2]) == [1]
    assert isinstance(par[1], ConnectionError) and par[2].calls.count("sleep") == 0


def test_order_matching_is_unchanged(caplog):
    """orderIds repeat across clientIds: an id on another ticker, or with
    another permId, is skipped as cancel_order skips it."""
    def make():
        return _FakeIB([_Trade(7, symbol="AMD"), _Trade(7, perm=555), _Trade(7, perm=1007),
                        _Trade(8)], {7: (1, "Cancelled"), 8: (1, "Cancelled")})
    seq, par = _both(make, caplog)
    assert seq[0] == par[0]
    sent = lambda f: [c for c in f.calls if isinstance(c, tuple)]
    assert sent(par[2]) == sent(seq[2])
    assert par[3] == seq[3]


def test_a_stop_gone_before_its_cancel_is_not_confirmed(caplog):
    def make():                             # fresh trades per run: the fake mutates status
        t1, t2 = _Trade(1), _Trade(2)
        return _FakeIB([t1, t2], {1: (1, "Cancelled")}, later_trades=[t1])
    seq, par = _both(make, caplog)
    assert seq[0] == par[0] == 1 and par[3] == seq[3] and "orderIds=[2]" in par[3][0]


def test_only_sell_stops_of_the_ticker_are_cancelled(caplog):
    trades = [_Trade(1), _Trade(2, order_type="LMT"), _Trade(3, action="BUY"),
              _Trade(4, symbol="AMD"), _Trade(5, order_type="TRAIL"), _Trade(6, order_type="STP LMT")]
    fake = _FakeIB(trades, {i: (1, "Cancelled") for i in range(1, 7)})
    res, exc = _run(fake, True)
    assert exc is None and res == 3
    assert sorted(c[1] for c in fake.calls if isinstance(c, tuple)) == [1, 5, 6]


def test_switch_read_failure_falls_back_to_one_at_a_time(caplog):
    fake = _FakeIB([_Trade(1), _Trade(2)], {1: (1, "Cancelled"), 2: (1, "Cancelled")})
    with patch("ibkr.get_ib", return_value=fake), \
         patch.object(ibkr, "_cancel_parallel_enabled", return_value=False):
        assert ibkr.cancel_stop_orders_for("NVDA") == 2
    assert fake.calls.count("sleep") == 2


@pytest.mark.parametrize("perms", [(0, 9002), (9001, 0), (0, 0), (9001, 9002)])
def test_two_stops_sharing_an_orderid_are_each_cancelled_once(perms, caplog):
    """orderIds repeat across clientIds. A sent trade stays PendingCancel in
    openTrades(), so with a permId of 0 the second stop's lookup must not
    find the first again (cancelled twice, counted twice, the second left
    resting under a new stop)."""
    make = lambda: _FakeIB([_Trade(57, perm=perms[0]), _Trade(57, perm=perms[1])],
                           {57: (1, "Cancelled")})
    seq, par = _both(make, caplog)
    for res, exc, fake, warn in (seq, par):
        assert exc is None and res == 2 and warn == []
        assert [t.orderStatus.status for t in fake.trades] == ["Cancelled", "Cancelled"]
        assert sum(isinstance(c, tuple) for c in fake.calls) == 2           # two sends
        assert list(fake.sent) == [id(t.order) for t in fake.trades]        # one to each
    assert par[2].calls.count("sleep") == 1
    assert not any("not cancelled" in r.getMessage() for r in caplog.records)


def test_a_raise_while_finding_the_next_stop_still_polls_the_sent_ones(caplog):
    fake = _FakeIB([_Trade(1), _Trade(2), _Trade(3)], {1: (1, "Cancelled"), 2: (1, "Cancelled")})
    real, n = fake.openTrades, {"reads": 0}

    def flaky():
        n["reads"] += 1
        if n["reads"] == 3:                     # the match read, stop 1, then stop 2's lookup
            raise ConnectionError("not connected")
        return real()
    fake.openTrades = flaky
    res, exc = _run(fake, True)
    assert isinstance(exc, ConnectionError) and res is None
    assert [c[1] for c in fake.calls if isinstance(c, tuple)] == [1]
    assert [t.orderStatus.status for t in fake.trades] == ["Cancelled", "PreSubmitted", "PreSubmitted"]
    assert any("orderIds=[2, 3]" in r.getMessage() for r in caplog.records)
