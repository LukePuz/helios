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

        mock_dispatch.assert_called_once_with(args=["NVDA", 10, 180.0], kwargs={}, queue="orders",
                                              expires=30)
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

        mock_dispatch.assert_called_once_with(args=["DDOG"], kwargs={}, queue="orders", expires=30)
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

        mock_dispatch.assert_called_once_with(args=[11], kwargs={}, queue="orders", expires=30)
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

        mock_dispatch.assert_called_once_with(
            args=["SNOW", 91], kwargs={"trail_amount": None, "trail_percent": 8.0},
            queue="orders", expires=30,
        )
        assert result == {"order_id": 55}


class TestTimedOutDispatch:
    """Professor C17 (2026-10-28): a timed-out request stayed queued and ran
    later outside the lock, and a retry could stack a second stop."""

    def test_timeout_revokes_the_request(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = TimeoutError("no reply")
        still = [{"order_id": 5, "ticker": "NVDA", "action": "SELL", "order_type": "STP",
                  "qty": 10.0, "stop_price": 150.0, "status": "Submitted"}]
        with patch.object(tasks._broker_cancel_stop_orders_for, "apply_async", return_value=fake), \
             patch("ibkr.get_open_orders", return_value=still), patch("time.sleep"):
            with pytest.raises(TimeoutError):
                tasks.manage_cancel_stop_orders_for("NVDA")
        fake.revoke.assert_called_once()

    def test_lost_stop_reply_adopts_the_resting_stop(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = TimeoutError("no reply")
        resting = [{"order_id": 77, "ticker": "NVDA", "action": "SELL", "order_type": "STP",
                    "qty": 10.0, "stop_price": 180.0, "status": "Submitted"}]
        with patch.object(tasks._broker_place_stop_order, "apply_async", return_value=fake), \
             patch("ibkr.get_open_orders", return_value=resting), \
             patch("time.sleep"), patch("db.clear_trailing_active"):
            assert tasks.manage_stop_place("NVDA", 10, 180.0) == {"order_id": 77, "stop_price": 180.0}

    def test_lost_stop_reply_with_nothing_resting_raises(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = TimeoutError("no reply")
        with patch.object(tasks._broker_place_stop_order, "apply_async", return_value=fake), \
             patch("ibkr.get_open_orders", return_value=[]), patch("time.sleep"):
            with pytest.raises(TimeoutError):
                tasks.manage_stop_place("NVDA", 10, 180.0)

    def test_lost_cancel_reply_with_no_stops_resting_counts_as_done(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = TimeoutError("no reply")
        with patch.object(tasks._broker_cancel_stop_orders_for, "apply_async", return_value=fake), \
             patch("ibkr.get_open_orders", return_value=[]), patch("time.sleep"):
            assert tasks.manage_cancel_stop_orders_for("NVDA") == 0

    def test_lost_trail_reply_adopts_the_resting_trail(self, monkeypatch):
        monkeypatch.setenv("IB_CLIENT_ID", "20")
        fake = MagicMock()
        fake.get.side_effect = TimeoutError("no reply")
        resting = [{"order_id": 88, "ticker": "SNOW", "action": "SELL", "order_type": "TRAIL",
                    "qty": 91.0, "stop_price": None, "status": "Submitted"}]
        with patch.object(tasks._broker_place_trailing_stop_order, "apply_async", return_value=fake), \
             patch("ibkr.get_open_orders", return_value=resting), patch("time.sleep"):
            assert tasks.manage_place_trailing_stop("SNOW", 91, trail_percent=8.0) == {"order_id": 88}
