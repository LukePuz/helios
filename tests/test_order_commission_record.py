"""Each IBKR commission report is written once, by exec_id, to an
order_commission activity row (audit 2026-11-07 07:07 #4; professor 64o:
TWS re-sends every execution's report on each connect and reconcile).
Log-only."""
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import os
import pytest

import ibkr

_REAL_ENQUEUE = ibkr._enqueue_commission


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(ibkr, "_COMMISSION_SEEN", set())
    monkeypatch.setattr(ibkr, "_enqueue_commission", MagicMock())
    redis = MagicMock()
    redis.set.return_value = True
    monkeypatch.setattr(ibkr, "_commission_redis", lambda: redis)
    return redis


def _args(commission=1.000003, exec_id="0001.01"):
    trade = NS(contract=NS(symbol="MSFT"), order=NS(orderId=79, permId=1903836920))
    import datetime as dt
    fill = NS(execution=NS(execId=exec_id, side="BOT", shares=1.0, price=517.0, permId=1903836920,
                           time=dt.datetime(2026, 11, 4, 1, 7, 21, tzinfo=dt.timezone.utc)))
    report = NS(execId=exec_id, commission=commission, currency="USD")
    return trade, fill, report


def test_record_fields():
    rec = ibkr._commission_record(*_args())
    assert rec == {"exec_id": "0001.01", "order_id": 79, "perm_id": 1903836920, "side": "BOT",
                   "shares": 1.0, "price": 517.0, "commission": 1.000003, "currency": "USD",
                   "exec_time": "2026-11-04T01:07:21+00:00"}


def test_missing_parts_read_none():
    rec = ibkr._commission_record(None, None, NS(commission="bad"))
    assert all(v is None for v in rec.values())


def test_a_report_is_queued_once_per_exec_id_in_a_process():
    for _ in range(3):
        ibkr._on_commission_report(*_args())
    ibkr._on_commission_report(*_args(exec_id="0001.02"))
    calls = ibkr._enqueue_commission.call_args_list
    assert [c.args[1]["exec_id"] for c in calls] == ["0001.01", "0001.02"]
    assert calls[0].args[0] == "MSFT"


def test_the_handler_does_not_touch_redis(_fresh_state):
    """Professor 64p: the claim runs in the writer, not on the event loop."""
    ibkr._on_commission_report(*_args())
    _fresh_state.set.assert_not_called()


def test_writer_claims_then_writes(_fresh_state):
    rec = ibkr._commission_record(*_args())
    with patch("db.log_activity_direct", return_value=True) as act:
        ibkr._write_commission("MSFT", rec)
    assert _fresh_state.set.call_args.args[0] == "helios:commission:0001.01"
    assert _fresh_state.set.call_args.kwargs["nx"] is True
    act.assert_called_once()


def test_another_process_claimed_it_first(_fresh_state):
    _fresh_state.set.return_value = None   # SET NX lost
    with patch("db.log_activity_direct") as act:
        ibkr._write_commission("MSFT", ibkr._commission_record(*_args()))
    act.assert_not_called()


def test_a_raising_write_releases_the_claim_and_a_spilled_one_keeps_it(_fresh_state):
    with patch("db.log_activity_direct", side_effect=RuntimeError("db")):
        ibkr._write_commission("MSFT", ibkr._commission_record(*_args()))
    _fresh_state.delete.assert_called_once_with("helios:commission:0001.01")
    _fresh_state.delete.reset_mock()
    with patch("db.log_activity_direct", return_value=False):   # spilled: replay retries it
        ibkr._write_commission("MSFT", ibkr._commission_record(*_args()))
    _fresh_state.delete.assert_not_called()


def test_redis_down_still_writes(_fresh_state):
    _fresh_state.set.side_effect = ConnectionError("down")
    with patch("db.log_activity_direct", return_value=True) as act:
        ibkr._write_commission("MSFT", ibkr._commission_record(*_args()))
    act.assert_called_once()


def test_placeholder_and_missing_commissions_are_skipped_and_not_claimed():
    ibkr._on_commission_report(*_args(commission=1.7976931348623157e308))
    ibkr._on_commission_report(*_args(commission=None))
    ibkr._enqueue_commission.assert_not_called()
    assert ibkr._COMMISSION_SEEN == set()
    ibkr._on_commission_report(*_args(commission=1.0))   # the real report follows
    assert ibkr._enqueue_commission.call_count == 1


def test_handler_never_raises():
    with patch("ibkr._commission_record", side_effect=RuntimeError("x")):
        ibkr._on_commission_report(None, None, None)


def test_writer_drains_the_queue_on_one_thread(monkeypatch):
    import queue
    import threading
    q = queue.Queue()
    done = threading.Event()
    written = []

    def act(*a, **k):
        written.append(a)
        if len(written) == 2:
            done.set()
    with patch("db.log_activity_direct", side_effect=act):
        threading.Thread(target=ibkr._commission_writer, args=(q,), daemon=True).start()
        for i in range(2):
            q.put(("MSFT", {**ibkr._commission_record(*_args(exec_id=f"e{i}"))}))
        assert done.wait(5)
    assert [w[0] for w in written] == ["order_commission", "order_commission"]
    assert written[0][4]["exec_id"] == "e0" and "env" in written[0][4]


def test_get_ib_attaches_the_handler():
    """conftest replaces ibkr.get_ib so no test can connect; read its source."""
    from pathlib import Path
    src = Path(ibkr.__file__).read_text()
    body = src[src.index("def get_ib("):src.index("if not _ib.isConnected():")]
    assert "_ib.commissionReportEvent += _on_commission_report" in body



def test_enqueue_restarts_the_writer_after_a_fork_or_death(monkeypatch):
    """Professor 64p: a forked child, or a dead writer, gets a new thread."""
    import threading
    made = []

    class T:
        def __init__(self, target=None, args=(), name=None, daemon=None):
            self.alive = True
            made.append(self)

        def start(self):
            pass

        def is_alive(self):
            return self.alive
    monkeypatch.setattr(threading, "Thread", T)
    monkeypatch.setattr(ibkr, "_COMMISSION_QUEUE", None)
    monkeypatch.setattr(ibkr, "_COMMISSION_WRITER", None)
    _REAL_ENQUEUE("MSFT", {"exec_id": "a"})
    _REAL_ENQUEUE("MSFT", {"exec_id": "b"})
    assert len(made) == 1                              # reused while alive
    made[0].alive = False
    _REAL_ENQUEUE("MSFT", {"exec_id": "c"})
    assert len(made) == 2                              # dead writer replaced
    monkeypatch.setattr(ibkr, "_COMMISSION_WRITER", (-1, made[1]))
    _REAL_ENQUEUE("MSFT", {"exec_id": "d"})
    assert len(made) == 3                              # another pid: a fork
    assert ibkr._COMMISSION_QUEUE.qsize() == 1


def test_full_queue_drops_without_blocking(monkeypatch):
    """Bounded queue: a put on the event loop never blocks (audit 2026-11-08 15:08 #6)."""
    import queue as _q
    import threading as _t
    import ibkr as _ibkr
    full = _q.Queue(maxsize=1)
    full.put(("X", {}))
    alive = _t.Thread(target=lambda: None)
    monkeypatch.setattr(_ibkr, "_COMMISSION_QUEUE", full)
    monkeypatch.setattr(_ibkr, "_COMMISSION_WRITER", (os.getpid(), alive))
    monkeypatch.setattr(alive, "is_alive", lambda: True)
    from unittest.mock import MagicMock
    fake = MagicMock()
    monkeypatch.setattr(_ibkr, "log", fake)
    _REAL_ENQUEUE("NVDA", {"exec_id": "e1"})
    assert full.qsize() == 1
    assert "commission queue full" in fake.warning.call_args[0][0]
