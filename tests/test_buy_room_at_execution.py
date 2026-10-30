"""Execution re-checks cash and the position cap when a buy is sent: buys
sized before the open each saw the same pre-market cash (pre-pull review 10-29)."""
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


def test_second_presized_buy_skipped_when_cash_is_gone():
    assert _check(_rm(3_000.0)) == (1, None)
    shares, reason = _check(_rm(1_000.0))
    assert shares == 0 and "not enough cash" in reason


def test_multi_share_buy_cut_to_what_cash_covers():
    assert _check(_rm(1_000.0), shares=10, price=100.0) == (9, None)   # 98% buffer


def test_position_cap_rechecked_for_opens_only():
    eight = [f"T{i}" for i in range(8)]
    shares, reason = _check(_rm(50_000.0, held=eight))
    assert shares == 0 and "position cap" in reason
    assert _check(_rm(50_000.0, held=eight + ["AVGO"]), action="ADD") == (1, None)


def test_fails_closed_when_it_cannot_check():
    rm = _rm(3_000.0)
    rm.cash.side_effect = RuntimeError("gateway down")
    shares, reason = _check(rm)
    assert shares == 0 and "check failed" in reason
