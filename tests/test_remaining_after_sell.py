"""A failed position read after a sell must not delete the stop record
(professor M22, 2026-10-28)."""
from unittest.mock import MagicMock, patch

import execution


def _pos(sym, qty):
    p = MagicMock(symbol=sym, qty=qty)
    return p


def _client(positions=None, error=None):
    c = MagicMock()
    if error:
        c.get_all_positions.side_effect = error
    else:
        c.get_all_positions.return_value = positions
    return c


def test_read_failure_falls_back_to_expected_remainder():
    with patch("ibkr.get_client", return_value=_client(error=RuntimeError("timeout"))):
        assert execution._remaining_after_sell("NVDA", 10, 3) == 7


def test_lagging_read_cannot_raise_the_stop_size():
    with patch("ibkr.get_client", return_value=_client([_pos("NVDA", 10)])):
        assert execution._remaining_after_sell("NVDA", 10, 3) == 7


def test_broker_lower_than_expected_wins():
    with patch("ibkr.get_client", return_value=_client([_pos("NVDA", 5)])):
        assert execution._remaining_after_sell("NVDA", 10, 3) == 5


def test_absent_position_with_shares_expected_keeps_them():
    with patch("ibkr.get_client", return_value=_client([_pos("AMD", 4)])):
        assert execution._remaining_after_sell("NVDA", 10, 3) == 7


def test_sold_everything_is_zero():
    with patch("ibkr.get_client", return_value=_client([])):
        assert execution._remaining_after_sell("NVDA", 1, 1) == 0
