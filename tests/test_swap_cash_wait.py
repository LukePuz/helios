"""A swap re-sizes the buy only once the sale shows in cash (audit A1031-1253-3)."""
from unittest.mock import MagicMock, patch

from agents import crew


def test_returns_once_the_proceeds_show():
    rm = MagicMock()
    rm.cash.side_effect = [100.0, 100.0, 1000.0]
    ib = MagicMock()
    with patch("ibkr.get_ib", return_value=ib), patch("time.sleep") as sleep:
        assert crew._wait_for_swap_cash(rm, 100.0, {"filled_qty": 2, "fill_price": 500.0}, "NVDA") is True
    assert ib.sleep.call_count == 2          # the IB loop runs, so cash can update
    sleep.assert_not_called()


def test_times_out_and_lets_the_resize_run():
    rm = MagicMock()
    rm.cash.return_value = 100.0
    with patch("ibkr.get_ib", side_effect=RuntimeError("no gateway")), patch("time.sleep") as sleep:
        assert crew._wait_for_swap_cash(rm, 100.0, {"filled_qty": 2, "fill_price": 500.0}, "NVDA") is False
    assert sleep.call_count == crew.SWAP_CASH_WAIT_S // 2   # falls back to time.sleep


def test_unknown_inputs_skip_the_wait():
    rm = MagicMock()
    with patch("time.sleep") as sleep:
        assert crew._wait_for_swap_cash(rm, None, {"filled_qty": 2, "fill_price": 500.0}, "NVDA") is False
        assert crew._wait_for_swap_cash(rm, 100.0, {"filled_qty": 0, "fill_price": None}, "NVDA") is False
    sleep.assert_not_called()
    rm.cash.assert_not_called()
