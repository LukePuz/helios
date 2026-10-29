"""
Tests for tasks.py's stop-order clientId routing (manage_stop_place,
manage_cancel_stop_orders_for, manage_cancel_order, manage_place_trailing_stop).

Regression coverage for the 2026-08-11 finding: MRVL/SNDK/VRT/ORCL/DDOG's
stuck-order incidents all traced back to one root cause -- a stop order
placed under one process's clientId can never be cancelled or modified by a
different process's clientId (IBKR silently ignores the request). Confirmed
live this isn't fixable via clientId=0's documented "master client" status
either: the cancel looked to succeed on that connection's own event log, but
a neutral third connection showed the order still resting at the broker
afterward -- the same looks-cancelled-isn't shape ibkr.cancel_order was
hardened against for the MRVL incident.

The fix funnels every stop-order touch through exactly one dedicated
process (celery_orders_worker, IB_CLIENT_ID=30, concurrency 1). These
functions decide whether the CURRENT process already IS that one process
(direct call -- cheap, no Celery round-trip) or needs to ask it to do the
work (dispatch a task to the "orders" queue and wait for the result).
"""
import pytest
from unittest.mock import MagicMock, patch

import tasks



def _assert_dispatched(mock_dispatch, args, kwargs=None):
    """Dispatch to the orders queue with an expiry and the caller's deadline."""
    call = mock_dispatch.call_args
    assert mock_dispatch.call_count == 1
    assert call.kwargs["args"] == args and call.kwargs["queue"] == "orders"
    assert call.kwargs["expires"] == 30
    kw = dict(call.kwargs["kwargs"]); assert kw.pop("deadline") > 0
    assert kw == (kwargs or {})

class TestIsOrdersWorker:
    def test_true_when_client_id_matches(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "30")
        assert tasks._is_orders_worker() is True

    def test_false_for_llm_worker_client_id(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        assert tasks._is_orders_worker() is False

    def test_false_for_ops_worker_client_id(self, monkeypatch):
        """The ops worker's base is 10 (forks add 0/1/2 on top) -- none of
        those ever equal the orders worker's fixed 30."""
        monkeypatch.setenv("IB_CLIENT_ID", "11")
        assert tasks._is_orders_worker() is False

    def test_false_when_unset(self, monkeypatch):
        monkeypatch.delenv("IB_CLIENT_ID", raising=False)
        assert tasks._is_orders_worker() is False


class TestManageStopPlace:
    def test_direct_call_when_already_orders_worker(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "30")
        with patch("ibkr.place_stop_order", return_value={"order_id": 42}) as mock_place:
            result = tasks.manage_stop_place("NVDA", 10, 180.0)

        mock_place.assert_called_once_with("NVDA", 10, 180.0)
        assert result == {"order_id": 42}

    def test_dispatches_and_waits_when_not_orders_worker(self, monkeypatch):
        """The LLM worker (clientId=20) is where every BUY entry's initial
        stop gets triggered from -- it must never place that stop itself
        anymore, only ask the orders worker to."""
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake_result = MagicMock()
        fake_result.get.return_value = {"order_id": 99}

        with patch.object(tasks._broker_place_stop_order, "apply_async",
                           return_value=fake_result) as mock_dispatch, \
             patch("ibkr.place_stop_order") as mock_direct:
            result = tasks.manage_stop_place("NVDA", 10, 180.0)

        _assert_dispatched(mock_dispatch, ["NVDA", 10, 180.0])
        fake_result.get.assert_called_once_with(timeout=30)
        mock_direct.assert_not_called()
        assert result == {"order_id": 99}


class TestManageCancelStopOrdersFor:
    def test_direct_call_when_already_orders_worker(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "30")
        with patch("ibkr.cancel_stop_orders_for", return_value=1) as mock_cancel:
            result = tasks.manage_cancel_stop_orders_for("DDOG")

        mock_cancel.assert_called_once_with("DDOG")
        assert result == 1

    def test_dispatches_and_waits_when_not_orders_worker(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "1")  # e.g. main.py's Telegram path
        fake_result = MagicMock()
        fake_result.get.return_value = 2

        with patch.object(tasks._broker_cancel_stop_orders_for, "apply_async",
                           return_value=fake_result) as mock_dispatch:
            result = tasks.manage_cancel_stop_orders_for("DDOG")

        _assert_dispatched(mock_dispatch, ["DDOG"])
        assert result == 2


class TestManageCancelOrder:
    def test_direct_call_when_already_orders_worker(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "30")
        with patch("ibkr.cancel_order", return_value=True) as mock_cancel:
            result = tasks.manage_cancel_order(11)

        mock_cancel.assert_called_once_with(11)
        assert result is True

    def test_dispatches_and_waits_when_not_orders_worker(self, monkeypatch):
        """This is also what main.py's startup stale-BUY-order cleanup
        (clientId=1) goes through now -- not just stops."""
        monkeypatch.setenv("IB_CLIENT_ID", "1")
        fake_result = MagicMock()
        fake_result.get.return_value = False

        with patch.object(tasks._broker_cancel_order, "apply_async",
                           return_value=fake_result) as mock_dispatch:
            result = tasks.manage_cancel_order(11)

        _assert_dispatched(mock_dispatch, [11])
        assert result is False


class TestManagePlaceTrailingStop:
    def test_direct_call_when_already_orders_worker(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "30")
        with patch("ibkr.place_trailing_stop_order",
                    return_value={"order_id": 7}) as mock_place:
            result = tasks.manage_place_trailing_stop("SNOW", 91, trail_amount=12.5)

        mock_place.assert_called_once_with("SNOW", 91, trail_amount=12.5, trail_percent=None)
        assert result == {"order_id": 7}

    def test_dispatches_and_waits_when_not_orders_worker(self, monkeypatch):
        """risk_manager.check_trailing_stops runs on the ops worker
        (clientId 10/11/12) -- it's NOT the orders worker either, even
        though this is the one call site that used to be "correctly"
        clientId-consistent by accident (both ops-only) before this fix.
        Ops's own 3-way fork concurrency meant even that was fragmented
        across 10/11/12 -- it needs the same dispatch now."""
        monkeypatch.setenv("IB_CLIENT_ID", "11")
        fake_result = MagicMock()
        fake_result.get.return_value = {"order_id": 55}

        with patch.object(tasks._broker_place_trailing_stop_order, "apply_async",
                           return_value=fake_result) as mock_dispatch:
            result = tasks.manage_place_trailing_stop("SNOW", 91, trail_percent=8.0)

        _assert_dispatched(mock_dispatch, ["SNOW", 91], {"trail_amount": None, "trail_percent": 8.0})
        assert result == {"order_id": 55}


class TestOrdersCallResult:
    """Professor C21/C22 (2026-10-29): judge a timed-out broker call by its
    real result, never by guessing from open orders."""

    def test_slow_reply_is_waited_for(self, monkeypatch):
        from celery.exceptions import TimeoutError as CeleryTimeout
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = [CeleryTimeout(), 2]
        with patch.object(tasks._broker_cancel_stop_orders_for, "apply_async", return_value=fake):
            assert tasks.manage_cancel_stop_orders_for("NVDA") == 2
        assert fake.get.call_args_list[1].kwargs["timeout"] == tasks._ORDERS_EXTRA_WAIT_S

    def test_no_result_at_all_is_unknown(self, monkeypatch):
        from celery.exceptions import TimeoutError as CeleryTimeout
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = CeleryTimeout()
        with patch.object(tasks._broker_place_stop_order, "apply_async", return_value=fake):
            with pytest.raises(tasks.OrdersCallUnknown):
                tasks.manage_stop_place("NVDA", 10, 180.0)

    def test_revoked_before_running_is_expired(self, monkeypatch):
        from celery.exceptions import TaskRevokedError
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = TaskRevokedError("expired")
        with patch.object(tasks._broker_cancel_order, "apply_async", return_value=fake):
            with pytest.raises(tasks.OrdersCallExpired):
                tasks.manage_cancel_order(5)

    def test_worker_refusal_after_deadline_is_expired(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.return_value = tasks._EXPIRED
        with patch.object(tasks._broker_place_trailing_stop_order, "apply_async", return_value=fake):
            with pytest.raises(tasks.OrdersCallExpired):
                tasks.manage_place_trailing_stop("SNOW", 91, trail_percent=8.0)

    def test_broker_task_does_nothing_past_its_deadline(self):
        with patch("ibkr.cancel_stop_orders_for") as cancel:
            assert tasks._broker_cancel_stop_orders_for.run("NVDA", deadline=1.0) == tasks._EXPIRED
        cancel.assert_not_called()

    def test_not_working_and_part_filled_stops_counted_by_remaining(self):
        orders = [
            {"order_id": 1, "ticker": "AMD", "action": "SELL", "order_type": "STP", "qty": 10.0,
             "stop_price": 90.0, "status": "Submitted", "remaining": 4.0},
            {"order_id": 2, "ticker": "AMD", "action": "SELL", "order_type": "STP", "qty": 10.0,
             "stop_price": 90.0, "status": "PendingCancel", "remaining": 10.0},
        ]
        with patch("ibkr.get_open_orders", return_value=orders):
            rs = tasks._resting_sell_stops("AMD")
        assert [(o["order_id"], o["qty"]) for o in rs] == [(1, 4.0)]
