"""Startup reconciliation waits for the orders worker (audit 00:53 A0053-3)."""
from unittest.mock import patch

import main


def test_reconcile_runs_once_the_orders_worker_answers():
    with patch("main._orders_worker_ready", return_value=True), \
         patch("main._reconcile_positions") as pos, \
         patch("main._reconcile_open_orders") as orders:
        main._startup_reconcile()
    pos.assert_called_once()
    orders.assert_called_once()


def test_reconcile_skipped_with_alert_when_worker_never_answers():
    with patch("main._orders_worker_ready", return_value=False), \
         patch("main._reconcile_positions") as pos, \
         patch("main._reconcile_open_orders") as orders, \
         patch("alerts._send") as send:
        main._startup_reconcile()
    pos.assert_not_called()
    orders.assert_not_called()
    assert "skipped" in send.call_args[0][0]


def test_ready_polls_until_a_ping_answers():
    replies = iter([[], [{"orders@h": {"ok": "pong"}}]])
    with patch("tasks.app.control.ping", side_effect=lambda **k: next(replies)), \
         patch("main.time.sleep"):
        assert main._orders_worker_ready(wait_s=60) is True
