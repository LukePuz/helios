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
