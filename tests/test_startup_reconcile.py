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


def test_orders_worker_starts_before_reconcile_and_the_rest_after():
    """A0253-5: beat and the other workers must not race the startup pass."""
    import threading
    calls = []
    services = {"celery_beat": {"label": "beat"}, "celery_orders_worker": {"label": "orders"},
                "celery_worker": {"label": "ops"}}
    started = threading.Event()

    def fake_thread(target, **k):
        class T:
            def start(self):
                target(); started.set()
        return T()

    with patch("process_manager.SERVICES", services), \
         patch("process_manager.start", side_effect=lambda n: calls.append(n) or "ok"), \
         patch("main._startup_reconcile", side_effect=lambda: calls.append("reconcile")), \
         patch("main.time.sleep"), patch("main.threading.Thread", fake_thread):
        main._auto_start_services()
    assert calls[:2] == ["celery_orders_worker", "reconcile"]
    assert set(calls[2:]) == {"celery_beat", "celery_worker"}


def _adopt(record=None, resting=()):
    from unittest.mock import MagicMock
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(avg_entry_price=100.0)
    rm = MagicMock()
    rm.calculate_stop.return_value = (90.0, "ATR")
    place = MagicMock(return_value={"order_id": 7})
    with patch("db.get_position_stop", return_value=record), \
         patch("db.upsert_position_stop") as upsert, patch("db.set_stop_broker_order_id"):
        msg = main._adopt_position(client, rm, "AMD", 10.0, place, lambda t: list(resting))
    return msg, place, upsert


def test_adoption_skips_a_position_recorded_meanwhile():
    msg, place, upsert = _adopt(record={"stop_price": 95.0})
    assert msg is None
    place.assert_not_called(); upsert.assert_not_called()


def test_adoption_stops_only_the_uncovered_shares():
    msg, place, _ = _adopt(resting=[{"qty": 4.0}])
    place.assert_called_once_with("AMD", 6.0, 90.0)
    msg, place, upsert = _adopt(resting=[{"qty": 10.0}])
    place.assert_not_called(); upsert.assert_called_once()
