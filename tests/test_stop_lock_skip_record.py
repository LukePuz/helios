"""A stop change that runs unlocked (Redis unreachable) writes a
stop_lock_skipped row on a daemon thread, at most once a minute per ticker
(audit 2026-11-07 13:07 #9). Log-only."""
import threading
from unittest.mock import patch

import stop_lock


def _down():
    raise ConnectionError("redis unreachable")


def _join():
    for t in threading.enumerate():
        if t.name == "stop-lock-skip":
            t.join(timeout=2)


def test_a_skip_writes_one_row_and_the_change_still_runs(monkeypatch):
    monkeypatch.setattr(stop_lock, "_redis", _down)
    monkeypatch.setattr(stop_lock, "_SKIP_LOGGED", {})
    monkeypatch.setattr(stop_lock, "_SKIP_SUPPRESSED", {})
    with patch("db.log_activity_queued") as act:
        with stop_lock.ticker_lock("AMD"):
            ran = True
        _join()
    assert ran
    (call,) = act.call_args_list
    assert call.args[0] == "stop_lock_skipped" and call.args[1] == "AMD"
    assert call.args[4]["error_type"] == "ConnectionError"


def test_repeat_skips_within_a_minute_write_once(monkeypatch):
    monkeypatch.setattr(stop_lock, "_redis", _down)
    monkeypatch.setattr(stop_lock, "_SKIP_LOGGED", {})
    monkeypatch.setattr(stop_lock, "_SKIP_SUPPRESSED", {})
    with patch("db.log_activity_queued") as act:
        for _ in range(3):
            with stop_lock.ticker_lock("AMD"):
                pass
        with stop_lock.ticker_lock("NVDA"):
            pass
        _join()
    assert [c.args[1] for c in act.call_args_list] == ["AMD", "NVDA"]


def test_a_failed_row_never_blocks_the_change(monkeypatch):
    monkeypatch.setattr(stop_lock, "_redis", _down)
    monkeypatch.setattr(stop_lock, "_SKIP_LOGGED", {})
    monkeypatch.setattr(stop_lock, "_SKIP_SUPPRESSED", {})
    with patch("db.log_activity_queued", side_effect=RuntimeError("db")):
        with stop_lock.ticker_lock("AMD"):
            ran = True
        _join()
    assert ran


def test_the_next_row_counts_the_suppressed_skips(monkeypatch):
    monkeypatch.setattr(stop_lock, "_redis", _down)
    monkeypatch.setattr(stop_lock, "_SKIP_LOGGED", {})
    monkeypatch.setattr(stop_lock, "_SKIP_SUPPRESSED", {})
    with patch("db.log_activity_queued") as act:
        for _ in range(3):
            with stop_lock.ticker_lock("AMD"):
                pass
        stop_lock._SKIP_LOGGED["AMD"] -= stop_lock._SKIP_ROW_EVERY_S + 1   # a minute later
        with stop_lock.ticker_lock("AMD"):
            pass
        _join()
    assert [c.args[4]["skipped_since_last"] for c in act.call_args_list] == [0, 2]


def test_the_first_locked_change_after_an_outage_flushes_the_carry(monkeypatch):
    from conftest import _FakeLock
    monkeypatch.setattr(stop_lock, "_SKIP_LOGGED", {})
    monkeypatch.setattr(stop_lock, "_SKIP_SUPPRESSED", {})
    monkeypatch.setattr(stop_lock, "_redis", _down)
    with patch("db.log_activity_queued") as act:
        for _ in range(3):
            with stop_lock.ticker_lock("AMD"):
                pass
        _join()

        class _Up:
            def lock(self, *a, **k):
                return _FakeLock(*a, **k)
        monkeypatch.setattr(stop_lock, "_redis", lambda: _Up())
        with stop_lock.ticker_lock("AMD"):
            pass
        with stop_lock.ticker_lock("AMD"):          # nothing left to flush
            pass
        _join()
    rows = [c.args[4] for c in act.call_args_list]
    assert rows[0]["skipped_since_last"] == 0
    assert rows[1]["recovered"] is True and rows[1]["skipped_since_last"] == 2
    assert len(rows) == 2 and "AMD" not in stop_lock._SKIP_LOGGED


def test_a_single_skip_still_gets_a_recovered_row(monkeypatch):
    """Professor m623: the closing row was written only when skips had been
    suppressed, so a one-skip outage had no end."""
    from conftest import _FakeLock
    monkeypatch.setattr(stop_lock, "_SKIP_LOGGED", {})
    monkeypatch.setattr(stop_lock, "_SKIP_SUPPRESSED", {})
    monkeypatch.setattr(stop_lock, "_redis", _down)
    with patch("db.log_activity_queued") as act:
        with stop_lock.ticker_lock("AMD"):
            pass

        class _Up:
            def lock(self, *a, **k):
                return _FakeLock(*a, **k)
        monkeypatch.setattr(stop_lock, "_redis", lambda: _Up())
        with stop_lock.ticker_lock("AMD"):
            pass
        with stop_lock.ticker_lock("NVDA"):          # never skipped: no row
            pass
    rows = [(c.args[1], c.args[4]) for c in act.call_args_list]
    assert len(rows) == 2 and rows[1][0] == "AMD"
    assert rows[1][1]["recovered"] is True and rows[1][1]["skipped_since_last"] == 0
