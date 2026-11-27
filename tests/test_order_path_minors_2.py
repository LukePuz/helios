"""Order-path minors, second set (order-path-minors-triage.md):
- m117: a queued flat exit's remainder-working check matches the open order on
  permId when both sides have one, else orderId, and always the ticker.
- m120: a SELL refused on the cached position read after the confirm read
  failed says so, instead of stating the cache's count as the broker's.
- get_fill_for_order matches the execution's permId when the order's is known.
"""
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import db
import execution
import ibkr
import tasks

_T0 = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)


# ── m117: the remainder-working match ────────────────────────────────────────

def _open(order_id, perm_id, ticker="CRWD", status="Submitted"):
    return {"order_id": order_id, "perm_id": perm_id, "ticker": ticker, "action": "SELL",
            "order_type": "MKT", "qty": 4.0, "status": status}


def _flat_close(row, open_orders):
    """_close_queued_flat_exit on a part-filled EXIT row, with the open orders
    given. True means the remainder reads as still working (wait)."""
    with patch("db.get_order_fill_state", return_value=row), \
         patch("ibkr.get_open_orders", return_value=open_orders), \
         patch("tasks._reconcile_alert_due", return_value=False), patch("alerts._send"):
        return tasks._close_queued_flat_exit("CRWD", {"intent": "EXIT", "order_id": 5}, [])


def _part(perm_id):
    return {"status": "partial", "fill_price": 171.5, "broker_order_id": 77, "created_at": _T0,
            "perm_id": perm_id}


def test_another_clients_order_with_the_same_id_is_not_our_remainder():
    # Same orderId and ticker, another permId: another client's order.
    assert _flat_close(_part(9001), [_open(77, 5555)]) is False
    # Same orderId, another ticker, no permId on either side: not ours either.
    assert _flat_close(_part(None), [_open(77, None, ticker="NVDA")]) is False
    assert _flat_close(_part(9001), [_open(77, None, ticker="NVDA")]) is False


def test_our_remainder_is_found_by_perm_id_or_else_by_order_id():
    assert _flat_close(_part(9001), [_open(77, 5555), _open(77, 9001)]) is True
    # Our permId on an order whose orderId reads differently is still ours.
    assert _flat_close(_part(9001), [_open(12, 9001)]) is True
    # permId missing on one side: orderId plus ticker.
    assert _flat_close(_part(None), [_open(77, 5555)]) is True
    assert _flat_close(_part(9001), [_open(77, None)]) is True
    # A finished order is not working, whoever's it is.
    assert _flat_close(_part(9001), [_open(77, 9001, status="Cancelled")]) is False


def test_order_fill_state_carries_the_perm_id():
    cur = MagicMock()
    cur.fetchone.return_value = ("partial", 171.5, 3.0, 77, _T0, None, "receipt", 9001)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        row = db.get_order_fill_state(5)
    assert "perm_id" in cur.execute.call_args.args[0]
    assert row["perm_id"] == 9001 and row["filled_at_source"] == "receipt"
    assert row["broker_order_id"] == 77 and row["created_at"] == _T0


def test_order_fill_state_without_the_source_column_keeps_the_perm_id():
    import psycopg2
    cur = MagicMock()
    cur.execute.side_effect = [psycopg2.errors.UndefinedColumn('column "filled_at_source" does not exist'),
                               None]
    cur.fetchone.return_value = ("partial", 171.5, 3.0, 77, _T0, None, None, 9001)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        row = db.get_order_fill_state(5)
    retried = cur.execute.call_args_list[1].args[0]
    assert "filled_at_source" not in retried and "perm_id" in retried
    assert row["perm_id"] == 9001 and row["filled_at_source"] is None


# ── m120: a failed confirm read ──────────────────────────────────────────────

def _pos(q, symbol="AMD"):
    return [MagicMock(contract=MagicMock(symbol=symbol, secType="STK"), position=q)]


def test_a_failed_confirm_read_refuses_on_the_cache_and_says_so(caplog):
    ib = MagicMock()
    ib.positions.return_value = _pos(-3)
    with patch.object(ibkr, "_bounded_req", side_effect=TimeoutError("reqPositions timed out")), \
         caplog.at_level(logging.WARNING, logger="ibkr"):
        with pytest.raises(ibkr.SellExceedsPosition) as e:
            ibkr._refuse_sell_on_short(ib, "AMD", 3)
    msg = str(e.value)
    assert "positions read failed; cached read shows -3 shares" in msg
    assert "broker shows" not in msg
    assert any("positions read failed" in r.getMessage() and "cached read" in r.getMessage()
               for r in caplog.records)
    # A confirmed short still names the broker; a fresh read that clears it allows.
    with patch.object(ibkr, "_bounded_req", return_value=_pos(-2)):
        with pytest.raises(ibkr.SellExceedsPosition) as e:
            ibkr._refuse_sell_on_short(ib, "AMD", 3)
    assert str(e.value).endswith("broker shows -2 shares")
    with patch.object(ibkr, "_bounded_req", return_value=_pos(5)):
        ibkr._refuse_sell_on_short(ib, "AMD", 3)          # the fresh read clears it


def test_the_refused_stop_alert_says_the_read_failed_instead_of_blaming_lag():
    with patch("alerts._send") as send:
        execution._alert_stop_refused(
            "AMD", 3.0, RuntimeError("AMD: SELL 3 refused; positions read failed; cached read shows -3 shares"))
        failed = send.call_args.args[0]
        execution._alert_stop_refused("AMD", 3.0, RuntimeError("AMD: SELL 3 refused; broker shows 0 shares"))
        lag = send.call_args.args[0]
    assert "re-read failed" in failed and "cached one" in failed
    assert "hasn't caught up" not in failed and "queued and retried" in failed
    assert "Usually the broker's position hasn't caught up with the fill yet; the stop is queued" in lag
    # The alert's wording follows the exception the check really raises.
    ib = MagicMock()
    ib.positions.return_value = _pos(-3)
    with patch.object(ibkr, "_bounded_req", side_effect=TimeoutError("timed out")):
        with pytest.raises(ibkr.SellExceedsPosition) as e:
            ibkr._refuse_sell_on_short(ib, "AMD", 3)
    with patch("alerts._send") as send:
        execution._alert_stop_refused("AMD", 3.0, e.value)
    assert "re-read failed" in send.call_args.args[0]


# ── get_fill_for_order: permId ───────────────────────────────────────────────

class _Exec:
    def __init__(self, symbol, order_id, shares, price, side="BOT", perm_id=0, time=None):
        self.contract = MagicMock(symbol=symbol)
        self.execution = MagicMock(orderId=order_id, shares=shares, price=price, avgPrice=price,
                                   side=side, permId=perm_id, time=time or _T0)
        self.time = time or _T0


def _lookup(records, **kw):
    ib = MagicMock()
    ib.reqExecutions.return_value = records
    with patch("ibkr.get_ib", return_value=ib):
        return ibkr.get_fill_for_order(764, ticker="CRWD", side="BUY", **kw)


def test_a_known_perm_id_leaves_another_clients_execution_out():
    recs = [_Exec("CRWD", 764, 144.0, 181.3, perm_id=9001),
            _Exec("CRWD", 764, 10.0, 200.0, perm_id=5555)]     # another client, same id
    r = _lookup(recs, perm_id=9001)
    assert r["filled_qty"] == 144.0 and r["fill_price"] == pytest.approx(181.3)
    assert _lookup(recs, perm_id=7777) is None
    # permIds are account-unique: our permId under another orderId is ours.
    assert _lookup([_Exec("CRWD", 999, 144.0, 181.3, perm_id=9001)], perm_id=9001)["filled_qty"] == 144.0
    # Without a perm_id the lookup is as before (orderId, ticker, side).
    assert _lookup(recs)["filled_qty"] == 154.0


def test_an_execution_without_a_perm_id_falls_back_to_the_order_id():
    recs = [_Exec("CRWD", 764, 144.0, 181.3, perm_id=0), _Exec("CRWD", 765, 9.0, 181.0, perm_id=0),
            _Exec("CRWD", 764, 6.0, 181.3, perm_id=5555)]
    assert _lookup(recs, perm_id=9001)["filled_qty"] == 144.0
