"""A reqPositions() result can hold two rows for one position when it updates
mid-request (NVDA 2026-11-13 14:28: 2 shares read as 4 and the whole-position
stop for 4 was refused). The fresh reads keep the last row per (account, conId)."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

import ibkr


def _pos(sym, qty, cost, con_id, account="A1", sec="STK"):
    return NS(account=account, position=qty, avgCost=cost,
              contract=NS(symbol=sym, conId=con_id, secType=sec))


NVDA_ROWS = [_pos("NVDA", 2.0, 193.1956, 4815747), _pos("MSFT", 1.0, 523.55, 272093),
             _pos("NVDA", 2.0, 193.6956, 4815747)]


def _with_rows(rows):
    ib = NS(reqPositions=lambda: rows, positions=lambda: rows)
    return patch("ibkr.get_ib", return_value=ib), patch("ibkr._bounded_req", side_effect=lambda ib, n, f: f())


def test_fresh_stock_position_counts_a_repeated_row_once():
    a, b = _with_rows(NVDA_ROWS)
    with a, b:
        qty, cost = ibkr.get_fresh_stock_position("NVDA")
    assert qty == 2.0 and cost == pytest.approx(193.6956)


def test_fresh_stock_positions_counts_a_repeated_row_once():
    a, b = _with_rows(NVDA_ROWS)
    with a, b:
        assert ibkr.get_fresh_stock_positions() == {"NVDA": 2.0, "MSFT": 1.0}


def test_latest_row_wins_including_a_close_to_zero():
    rows = [_pos("NVDA", 2.0, 190.0, 1), _pos("NVDA", 0.0, 0.0, 1)]
    assert [p.position for p in ibkr._latest_positions(rows)] == [0.0]


def test_rows_in_other_accounts_and_without_con_id_are_kept():
    rows = [_pos("NVDA", 1.0, 1.0, 1, "A1"), _pos("NVDA", 1.0, 1.0, 1, "A2"),
            _pos("X", 1.0, 1.0, None)]
    assert len(ibkr._latest_positions(rows)) == 3
    assert ibkr._latest_positions(None) == []


def test_sell_guard_does_not_double_count_a_repeated_row():
    # The cache shows 1 (lagging); the fresh read repeats the 2-share row.
    cache = [_pos("NVDA", 1.0, 190.0, 4815747)]
    fresh = [_pos("NVDA", 2.0, 190.0, 4815747), _pos("NVDA", 2.0, 191.0, 4815747)]
    ib = NS(positions=lambda: cache, reqPositions=lambda: fresh)
    with patch("ibkr._bounded_req", side_effect=lambda ib, n, f: f()):
        ibkr._refuse_sell_on_short(ib, "NVDA", 2.0)          # 2 held: allowed
        with pytest.raises(ibkr.SellExceedsPosition):
            ibkr._refuse_sell_on_short(ib, "NVDA", 4.0)      # was allowed when 2+2=4
