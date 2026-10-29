"""Per-ticker stop lock (professor M-5, 2026-10-28)."""
import stop_lock
from conftest import _FakeLock


def test_reentrant_for_same_ticker():
    calls = []

    @stop_lock.locked_by_ticker(lambda t: t)
    def inner(t):
        calls.append("inner")
        return "ok"

    @stop_lock.locked_by_ticker(lambda t: t)
    def outer(t):
        return inner(t)

    assert outer("AMD") == "ok" and calls == ["inner"]
    assert _FakeLock.held == set()  # released after the outer section


def test_busy_ticker_returns_on_busy_without_running(monkeypatch):
    monkeypatch.setattr(stop_lock, "DEFAULT_WAIT_S", 0)
    _FakeLock.held.add("helios:stop_lock:AMD")  # another process holds it
    ran = []

    @stop_lock.locked_by_ticker(lambda t: t, on_busy=False)
    def change(t):
        ran.append(t)
        return True

    assert change("AMD") is False and ran == []
    assert change("NVDA") is True and ran == ["NVDA"]  # other tickers unaffected


def test_redis_down_runs_unlocked(monkeypatch):
    def _down():
        raise ConnectionError("redis unreachable")
    monkeypatch.setattr(stop_lock, "_redis", _down)

    @stop_lock.locked_by_ticker(lambda t: t)
    def change(t):
        return "placed"

    assert change("AMD") == "placed"


def test_lock_released_when_section_raises():
    try:
        with stop_lock.ticker_lock("AMD"):
            raise RuntimeError("broker error")
    except RuntimeError:
        pass
    assert _FakeLock.held == set()


def test_execute_approved_skips_when_ticker_busy(monkeypatch):
    import execution
    monkeypatch.setattr(stop_lock, "DEFAULT_WAIT_S", 0)
    _FakeLock.held.add("helios:stop_lock:AMD")
    assert execution.execute_approved({"ticker": "AMD", "action": "EXIT"}) is False


def test_lock_is_renewed_while_a_long_section_runs(monkeypatch):
    """Professor M49: a section longer than the TTL must keep the lock."""
    import time
    monkeypatch.setattr(stop_lock, "LOCK_RENEW_S", 0.02)
    with stop_lock.ticker_lock("AMD"):
        time.sleep(0.15)
    n = len(_FakeLock.renewed)
    assert n >= 3 and set(_FakeLock.renewed) == {"helios:stop_lock:AMD"}
    time.sleep(0.1)
    assert len(_FakeLock.renewed) == n  # renewing stops once released


def test_renewal_stops_after_the_hold_cap(monkeypatch):
    """A0253-2: a hung holder must not keep the ticker forever."""
    import threading
    import time
    from unittest.mock import MagicMock, patch
    monkeypatch.setattr(stop_lock, "LOCK_RENEW_S", 0.01)
    lock, done = MagicMock(), threading.Event()
    with patch("alerts._send") as send:
        t = threading.Thread(target=stop_lock._keep_renewed, args=(lock, "AMD", done, 0.05))
        t.start(); t.join(1.0)
    assert not t.is_alive()  # gave up on its own, done never set
    assert lock.reacquire.call_count >= 1
    send.assert_called_once()


def test_lost_lock_stops_further_broker_calls(monkeypatch):
    """Professor M69: after the renewal gives up, the holder must not act."""
    import time
    import pytest
    import tasks
    from unittest.mock import patch
    monkeypatch.setattr(stop_lock, "LOCK_RENEW_S", 0.01)
    monkeypatch.setattr(stop_lock, "MAX_HOLD_S", 0.03)
    with patch("alerts._send"), patch.object(tasks, "_orders_call") as call:
        with stop_lock.ticker_lock("AMD"):
            time.sleep(0.15)
            with pytest.raises(stop_lock.LockLost):
                tasks.manage_stop_place("AMD", 1, 90.0)
    call.assert_not_called()


def test_assert_held_is_a_no_op_outside_the_lock():
    stop_lock.assert_held("AMD")


def test_expired_lock_is_detected_and_taken_by_another(monkeypatch):
    """C23: when the TTL runs out mid-section, the renew fails, assert_held
    raises LockLost, and another process can take the ticker."""
    import time
    import pytest
    monkeypatch.setattr(stop_lock, "LOCK_RENEW_S", 0.02)
    with stop_lock.ticker_lock("AMD"):
        stop_lock.assert_held("AMD")
        _FakeLock.expire("helios:stop_lock:AMD")
        time.sleep(0.1)
        with pytest.raises(stop_lock.LockLost):
            stop_lock.assert_held("AMD")
        other = _FakeLock("helios:stop_lock:AMD")
        assert other.acquire()
    # Our release fails (logged) and leaves the other holder's lock alone.
    assert "helios:stop_lock:AMD" in _FakeLock.held


def test_every_stop_changing_function_is_locked():
    """Professor run 13: a helper inserted above a decorated function steals its
    @locked_by_ticker (5ee5806 did it to _execute_backstop_sell; an earlier edit on 10-29
    did it to _reprotect_after_late_sell). Direct-call tests can't see that, so check
    the wrapper is on each function that changes stops or sells."""
    import execution
    import risk_manager
    import tasks
    locked = [execution.execute_approved, execution.execute_manual,
              tasks._reprotect_after_late_sell, tasks._execute_backstop_sell, tasks._fix_overstop,
              risk_manager.RiskManager._ratchet_ladder_stop,
              risk_manager.RiskManager._breakeven_without_sale,
              risk_manager.RiskManager._partial_sell_and_resize_stop]
    for fn in locked:
        assert hasattr(fn, "__wrapped__"), f"{fn.__qualname__} lost its ticker lock"
    assert not hasattr(tasks._split_ratio, "__wrapped__")
