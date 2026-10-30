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


def _check(rm, action="OPEN", shares=1, price=1_400.0, ticker="AVGO", sized_at=None):
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager", return_value=rm):
        return execution._buy_room_at_execution(ticker, action, shares, price, sized_at)


def test_one_share_within_the_one_share_cash_limit():
    assert _check(_rm(3_000.0)) == (1, None)            # 47% of cash <= 50%
    shares, reason = _check(_rm(2_000.0), ticker="AVGO2")  # 70% of cash
    assert shares == 0 and "cash" in reason


def test_parallel_buys_see_each_others_reservations():
    # Same cash read for both, as at the open: the second must not pass.
    assert _check(_rm(3_000.0), ticker="A") == (1, None)
    shares, reason = _check(_rm(3_000.0), ticker="B")   # 3000 - 1400 = 1600; 1400 > 50% x1.08
    assert shares == 0


def test_multi_share_buy_cut_to_the_per_trade_cash_limit():
    assert _check(_rm(1_000.0), shares=10, price=10.0) == (10, None)     # $100 <= 20% of $1000
    # The first buy reserved $100: 20% (x1.08 drift tolerance) of $900 buys 3 at $50.
    assert _check(_rm(1_000.0), shares=10, price=50.0, ticker="X") == (3, None)


def test_small_uptick_does_not_trim_a_correctly_sized_buy():
    # 2 shares sized at $100 against $1000 (20%); price now $100.50.
    assert _check(_rm(1_000.0), shares=2, price=100.5, sized_at=100.0) == (2, None)


def test_filled_buy_is_not_counted_twice_once_cash_shows_it():
    assert _check(_rm(1_000.0), ticker="A", price=450.0) == (1, None)      # reserves $450
    # Cash now shows the fill (1000 -> 550): B's $250 share fits 50% of $550.
    assert _check(_rm(550.0), ticker="B", price=250.0) == (1, None)
    # Before cash updates (still 1000), A's $450 is pending: 1000-450=550, still fits.
    import stop_lock
    execution._release_buy("B")
    assert _check(_rm(1_000.0), ticker="C", price=300.0)[0] == 0          # 300 > 50% of 550*1.08


def test_failed_order_releases_its_reservation():
    assert _check(_rm(3_000.0), ticker="A") == (1, None)
    execution._release_buy("A")
    assert _check(_rm(3_000.0), ticker="B") == (1, None)                   # A's $1400 no longer held


def test_fails_closed_without_redis():
    import stop_lock
    with patch.object(type(stop_lock._redis()), "ping", side_effect=ConnectionError("redis down")):
        shares, reason = _check(_rm(3_000.0), ticker="R")
    assert shares == 0 and "check failed" in reason


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


def test_second_buy_of_the_same_name_counts_its_own_reservation():
    """Review 4 #5: AAA reserved, BBB reserved and unfilled, AAA filled; a
    second AAA buy must still see BBB's pending $1000."""
    assert _check(_rm(10_000.0), ticker="AAA", shares=10, price=100.0) == (10, None)
    assert _check(_rm(10_000.0), ticker="BBB", shares=10, price=100.0) == (10, None)
    # AAA filled (cash 10000 -> 9000), BBB still pending: 8000 x 20% buys 16
    # (excluding AAA's own reservation read BBB as filled and allowed 18).
    assert _check(_rm(9_000.0), ticker="AAA", shares=20, price=100.0) == (16, None)


def test_release_frees_only_that_orders_reservation():
    assert _check(_rm(10_000.0), ticker="A", shares=10, price=100.0) == (10, None)
    assert _check(_rm(10_000.0), ticker="A", shares=10, price=100.0) == (10, None)
    execution._release_buy("A")                       # the second order died
    assert [r["ticker"] for r in execution._buy_reservations()] == ["A"]


def test_caps_are_exact_and_only_price_drift_is_tolerated():
    """Professor M98: the 8% tolerance covers price drift since sizing, not the caps."""
    # No price change: exactly 20% of cash ($200) buys 2 at $100, not 2.16.
    assert _check(_rm(1_000.0), shares=5, price=100.0, sized_at=100.0) == (2, None)
    # Cash fell since sizing, price unchanged: the exact cap cuts (1 share), no 8% slack.
    assert _check(_rm(900.0), ticker="B", shares=2, price=100.0, sized_at=100.0)[0] == 1
    # Price rose 9% (beyond the tolerance): priced at today's $109, 200/109 = 1.
    assert _check(_rm(1_000.0), ticker="C", shares=2, price=109.0, sized_at=100.0)[0] == 1
    # One share: exactly 50% of cash, no 54%.
    assert _check(_rm(1_000.0), ticker="D", shares=1, price=520.0, sized_at=520.0)[0] == 0


def test_manual_buy_uses_cash_net_of_buys_in_flight_and_reserves():
    """Professor M99: dashboard / !urgent buys skipped the check and were invisible to it."""
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager", return_value=_rm(1_000.0)):
        assert execution._manual_buy_room("M1", 6, 100.0) is None          # $600 of $1000, no per-trade cap
        reason = execution._manual_buy_room("M2", 5, 100.0)               # $500 > $400 left
    assert reason and "not enough cash" in reason
    assert _check(_rm(1_000.0), ticker="A", shares=1, price=350.0)[0] == 0  # automated buy sees M1's $600
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager", return_value=_rm(1_000.0)):
        assert "no price" in execution._manual_buy_room("M3", 1, None)


def test_execute_manual_buy_blocked_without_cash():
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager") as rm_cls, \
         patch("market_data.get_priced_quote", return_value={"price": 100.0}), \
         patch("ibkr.place_market_order") as order:
        rm_cls.return_value.is_halted.return_value = False
        rm_cls.return_value.cash.return_value = 200.0
        r = execution.execute_manual("ZZZ", "BUY", 5, "test")
    assert r["success"] is False and "not enough cash" in r["message"] and not order.called
