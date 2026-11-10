"""db.log_activity_queued: one writer thread per process drains queued
log-only rows, so a burst opens one DB session at a time (professor 66m)."""
import os
import threading
import time
from unittest.mock import patch

import db


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.01)
    return pred()


def test_a_burst_is_written_in_order_by_one_thread(monkeypatch):
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", None)
    seen, threads = [], set()

    def write(stage, ticker, msg, sentiment, extra):
        seen.append(extra["i"])
        threads.add(threading.current_thread().name)
    with patch("db.log_activity_direct", side_effect=write):
        for i in range(50):
            db.log_activity_queued("task_lost", "", "m", "neutral", {"i": i})
        assert _wait(lambda: len(seen) == 50)
    assert seen == list(range(50)) and threads == {"activity-writer"}
    assert db._ACTIVITY_WRITER[0] == os.getpid()


def test_a_full_queue_spills_instead_of_growing(monkeypatch):
    """Professor m620: bounded queue; a row that doesn't fit is spilled."""
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", None)
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE_MAX", 2)
    gate = threading.Event()
    with patch("db.log_activity_direct", side_effect=lambda *a: gate.wait(2)), \
         patch("db._spill") as spill, patch("atexit.register"):
        for i in range(6):
            db.log_activity_queued("x", "", "m", "neutral", {"i": i})
        # one row in the writer, two queued, the rest spilled
        assert 3 <= spill.call_count <= 4
        assert spill.call_args.args[0] == "log_activity"
        assert spill.call_args.args[2] == "activity queue full"
        gate.set()
        assert _wait(lambda: db._ACTIVITY_QUEUE.unfinished_tasks == 0)


def test_the_exit_drain_spills_what_the_writer_did_not_write(monkeypatch):
    import queue
    q = queue.Queue()
    q.put(("x", "", "m", "neutral", {"i": 1}))
    monkeypatch.setattr(db, "_ACTIVITY_DRAIN_S", 0.05)
    with patch("db._spill") as spill:
        db._drain_activity_queue(q, os.getpid())
        assert spill.call_count == 1 and spill.call_args.args[1]["extra"] == {"i": 1}
        assert q.unfinished_tasks == 0
        q.put(("x", "", "m", "neutral", None))
        db._drain_activity_queue(q, os.getpid() + 1)      # another process's queue
        assert spill.call_count == 1


def test_a_failing_row_does_not_stop_the_writer(monkeypatch):
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", None)
    seen = []

    def write(stage, ticker, msg, sentiment, extra):
        if extra["i"] == 0:
            raise RuntimeError("db")
        seen.append(extra["i"])
    with patch("db.log_activity_direct", side_effect=write):
        db.log_activity_queued("x", "", "m", "neutral", {"i": 0})
        db.log_activity_queued("x", "", "m", "neutral", {"i": 1})
        assert _wait(lambda: seen == [1])


def test_a_dead_or_foreign_writer_is_replaced(monkeypatch):
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", (os.getpid(), dead))
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    with patch("db.log_activity_direct"):
        db.log_activity_queued("x", "", "m")
        assert db._ACTIVITY_WRITER[1] is not dead and db._ACTIVITY_WRITER[1].is_alive()
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", (os.getpid() + 1, db._ACTIVITY_WRITER[1]))
    with patch("db.log_activity_direct"):
        db.log_activity_queued("x", "", "m")
        assert db._ACTIVITY_WRITER[0] == os.getpid()


def test_rows_carry_their_enqueue_time(monkeypatch):
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", None)
    seen = []
    with patch("db.log_activity_direct", side_effect=lambda *a: seen.append(a[4])):
        db.log_activity_queued("x", "", "m", "neutral", {"k": 1})
        assert _wait(lambda: seen)
    assert seen[0]["k"] == 1 and seen[0]["queued_at"].endswith("+00:00")


def test_concurrent_first_calls_start_one_writer(monkeypatch):
    monkeypatch.setattr(db, "_ACTIVITY_QUEUE", None)
    monkeypatch.setattr(db, "_ACTIVITY_WRITER", None)
    with patch("db.log_activity_direct"):
        ts = [threading.Thread(target=db.log_activity_queued, args=("x", "", "m")) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        names = [t for t in threading.enumerate() if t.name == "activity-writer" and t.is_alive()]
        assert db._ACTIVITY_WRITER[1] in names
