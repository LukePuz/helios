"""Execution re-checks cash and the position cap when a buy is sent: buys
sized before the open run later with stored share counts, and buys deferred
to the same open run in parallel against the same cash (pre-pull reviews 10-29)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import execution

pytestmark = pytest.mark.real_buy_room


def _rm(cash, held=(), pv=10_000.0):
    rm = MagicMock()
    rm.cash.return_value = cash
    rm.portfolio_value.return_value = pv
    rm._client.get_all_positions.return_value = [SimpleNamespace(symbol=s) for s in held]
    return rm


def _check(rm, action="OPEN", shares=1, price=1_400.0, ticker="AVGO"):
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager", return_value=rm):
        return execution._buy_room_at_execution(ticker, action, shares, price)


def test_one_share_within_the_one_share_cash_limit():
    assert _check(_rm(3_000.0)) == (1, None)            # 47% of cash <= 50%
    shares, reason = _check(_rm(2_000.0), ticker="AVGO2")  # 70% of cash
    assert shares == 0 and "cash" in reason


def test_parallel_buys_see_each_others_reservations():
    # Same cash read for both, as at the open: the second must not pass.
    assert _check(_rm(3_000.0), ticker="A") == (1, None)
    shares, reason = _check(_rm(3_000.0), ticker="B")   # 3000 - 1400 = 1600; 1400 > 50%
    assert shares == 0


def test_multi_share_buy_cut_to_the_per_trade_cash_limit():
    assert _check(_rm(1_000.0), shares=10, price=10.0) == (10, None)     # $100 <= 20% of $1000
    # The first buy reserved $100: 20% of the remaining $900 buys 3 at $50.
    assert _check(_rm(1_000.0), shares=10, price=50.0, ticker="X") == (3, None)


def test_position_cap_counts_buys_in_flight():
    seven = [f"T{i}" for i in range(7)]
    assert _check(_rm(50_000.0, held=seven), ticker="NEW1", price=100.0)[0] >= 1
    shares, reason = _check(_rm(50_000.0, held=seven), ticker="NEW2", price=100.0)
    assert shares == 0 and "position cap" in reason
    assert _check(_rm(50_000.0, held=seven + ["NEW1"]), action="ADD", ticker="T1", price=100.0)[0] >= 1


def test_fails_closed_without_price_or_on_error():
    assert _check(_rm(3_000.0), price=None)[0] == 0
    rm = _rm(3_000.0)
    rm.cash.side_effect = RuntimeError("gateway down")
    shares, reason = _check(rm, ticker="Z")
    assert shares == 0 and "check failed" in reason
