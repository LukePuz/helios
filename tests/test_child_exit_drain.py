"""Pool children leave through os._exit (no atexit), so the activity queue
is drained from worker_process_shutdown (audit 2026-11-08 15:08 #6)."""
import os
import queue
from unittest.mock import patch

import db
import tasks


def test_drain_runs_for_this_process_only(monkeypatch):
    q = queue.Queue()
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", q)
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", (os.getpid(), None))
    with patch.object(db, "_drain_activity_queue") as drain:
        db.drain_activity_queue_now()
    drain.assert_called_once_with(q, os.getpid())
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", (os.getpid() + 1, None))
    with patch.object(db, "_drain_activity_queue") as drain:
        db.drain_activity_queue_now()
    assert not drain.called
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    db.drain_activity_queue_now()        # nothing queued: no error


def test_the_signal_handler_is_connected_and_calls_it():
    from celery.signals import worker_process_shutdown
    assert any(getattr(r[1](), "__name__", "") == "_drain_on_child_exit"
               for r in worker_process_shutdown.receivers if callable(r[1]))
    with patch("db.drain_activity_queue_now") as drain:
        tasks._drain_on_child_exit()
    assert drain.called


def test_commission_drain_waits_for_the_writer(monkeypatch):
    import threading
    import ibkr
    q = queue.Queue()
    writes = []
    monkeypatch.setattr(ibkr, "_write_commission", lambda s, r: writes.append(s))
    t = threading.Thread(target=ibkr._commission_writer, args=(q,), daemon=True)
    t.start()
    monkeypatch.setattr(ibkr, "_COMMISSION_QUEUE", q)
    monkeypatch.setattr(ibkr, "_COMMISSION_WRITER", (os.getpid(), t))
    for s in ("A", "B", "C"):
        q.put((s, {}))
    ibkr.drain_commission_queue(timeout_s=5)
    assert writes == ["A", "B", "C"] and q.unfinished_tasks == 0


def test_commission_drain_skips_other_processes(monkeypatch):
    import ibkr
    q = queue.Queue()
    q.put(("A", {}))
    monkeypatch.setattr(ibkr, "_COMMISSION_QUEUE", q)
    monkeypatch.setattr(ibkr, "_COMMISSION_WRITER", (os.getpid() + 1, None))
    ibkr.drain_commission_queue(timeout_s=5)      # returns at once
    assert q.unfinished_tasks == 1
