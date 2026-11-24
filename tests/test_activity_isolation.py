"""conftest's per-test settle of db's activity queue: rows a finished test
queued are dropped, a row in flight gets a bounded wait, and a queue still
busy after it is detached so the next test starts fresh."""
import queue
import time

import conftest
import db


def test_queued_rows_of_a_finished_test_are_dropped():
    q = queue.Queue()
    for i in range(3):
        q.put(("add_stop_qty", "NVDA", "m", "neutral", {"i": i}))
    assert conftest._settle_activity_queue(q) is True
    assert q.empty() and q.unfinished_tasks == 0
    assert conftest._settle_activity_queue(None) is True


def test_a_row_in_flight_is_waited_for_with_a_bound(monkeypatch):
    monkeypatch.setattr(conftest, "_ACTIVITY_SETTLE_S", 0.1)
    q = queue.Queue()
    q.put(("x", "", "m", "neutral", None))
    q.get()                        # taken by a writer that never finishes it
    start = time.monotonic()
    assert conftest._settle_activity_queue(q) is False
    assert time.monotonic() - start < 1.0


def test_the_fixture_leaves_db_runtime_code_alone():
    import inspect
    src = inspect.getsource(conftest._isolated_activity_writer)
    assert "db._ACTIVITY_QUEUE = None" in src and "db._activity_writer" not in src
    assert callable(db.log_activity_queued) and callable(db._drain_activity_queue)
