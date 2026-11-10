"""
Per-ticker lock around stop changes (professor M-5, 2026-10-28).

Every stop change is cancel-then-place, and it runs from several processes:
the ops worker (3 forks: trailing stops, profit targets, reconcile every
300 s), the LLM worker (crew executions) and the Telegram bot (approvals,
!urgent). Two of them interleaving on one ticker could leave two stops
resting (an oversell, or a short if both trigger) or none. This serialises
them per ticker with a Redis lock, Redis being the Celery broker every one
of those processes already depends on.

Re-entrant per thread, so a locked function can call another locked one
for the same ticker. The Redis key expires on its own (LOCK_TTL_S), so a
killed process can't hold a ticker forever. If Redis itself can't be
reached the change runs unlocked: protecting a position beats serialising.
"""
import functools
import logging
import os
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)

# A locked section can make several orders-queue calls of up to 150 s each,
# so the key is renewed every LOCK_RENEW_S while its holder runs instead of
# relying on one fixed TTL (professor M49). LOCK_TTL_S is how long a killed
# holder keeps the ticker.
LOCK_TTL_S = 90
LOCK_RENEW_S = 30
MAX_HOLD_S = 900  # cap on renewal, see _keep_renewed
DEFAULT_WAIT_S = 120

_local = threading.local()
_client = None


class TickerBusy(Exception):
    """Another process held the ticker's stop lock for the whole wait."""


class LockLost(Exception):
    """This thread's lock on the ticker lapsed (a renewal failed or the hold
    cap passed), so another process may now be changing its stops: stop
    before sending anything else (professor M69)."""


def assert_held(ticker: str) -> None:
    """Raise LockLost if this thread holds ticker's lock but it has lapsed.
    No-op when the thread isn't inside ticker_lock for it.

    This is check-then-act: it can't catch a lapse during a broker call that
    is already in flight (up to ~150 s). A call sent just before a lapse can
    overlap another process's stop change; reconcile_stop_orders' over-stop
    and naked checks are the backstop for that window (professor m40)."""
    ev = getattr(_local, "lost", {}).get(ticker)
    if ev is not None and ev.is_set():
        raise LockLost(ticker)


# Per process: ticker -> monotonic time of its last stop_lock_skipped row,
# so a long Redis outage writes about one row per ticker per minute.
_SKIP_LOGGED: dict = {}
_SKIP_SUPPRESSED: dict = {}          # ticker -> skips not written since its last row
_SKIP_ROW_EVERY_S = 60


def _record_lock_skip(ticker: str, error: Exception) -> None:
    """One activity row (stage stop_lock_skipped) when a stop change runs
    unlocked, so Redis outage windows can be measured (audit 2026-11-07
    13:07 #9, professor). Queued to the process's one activity writer: the
    stop change must never wait on the DB. Log-only; never raises."""
    import time
    try:
        now = time.monotonic()
        if now - _SKIP_LOGGED.get(ticker, -_SKIP_ROW_EVERY_S) < _SKIP_ROW_EVERY_S:
            _SKIP_SUPPRESSED[ticker] = _SKIP_SUPPRESSED.get(ticker, 0) + 1
            return
        _SKIP_LOGGED[ticker] = now
        extra = {"error_type": type(error).__name__, "error": str(error)[:200], "pid": os.getpid(),
                 "skipped_since_last": _SKIP_SUPPRESSED.pop(ticker, 0)}

        import db
        db.log_activity_queued("stop_lock_skipped", ticker,
                               f"stop lock unavailable, change ran unlocked ({extra['error_type']})",
                               "neutral", {**extra, "env": db.TRADING_ENV})
    except Exception:
        pass


def _flush_skip_carry(ticker: str) -> None:
    """On the first locked change after an outage, one closing row (stage
    stop_lock_skipped, recovered=True) with the skips suppressed since the
    ticker's last row, so they count toward that outage (SEF 153j). Written
    after any skip, so every outage has an end (professor m623). No-op when
    the ticker had no skip. Queued to the activity writer; never raises."""
    try:
        n = _SKIP_SUPPRESSED.pop(ticker, 0)
        had_skip = _SKIP_LOGGED.pop(ticker, None) is not None
        if not n and not had_skip:
            return
        extra = {"recovered": True, "skipped_since_last": n, "pid": os.getpid()}

        import db
        db.log_activity_queued("stop_lock_skipped", ticker,
                               f"stop lock back; {n} unlocked change(s) since the last row",
                               "neutral", {**extra, "env": db.TRADING_ENV})
    except Exception:
        pass


_REDIS_SOCKET_TIMEOUT_S = 5.0


def _redis():
    global _client
    if _client is None:
        import redis
        # A hung Redis must not hang a stop change forever (audit 2026-11-08
        # 15:08 #8). Each call is a short SET/Lua script (the blocking wait is
        # a client-side sleep loop), so 5 s only trips on a stuck server; the
        # timeout then takes the fail-open path in ticker_lock.
        _client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                       socket_timeout=_REDIS_SOCKET_TIMEOUT_S,
                                       socket_connect_timeout=_REDIS_SOCKET_TIMEOUT_S)
    return _client


@contextmanager
def ticker_lock(ticker: str, wait: float | None = None):
    held = getattr(_local, "held", None)
    if held is None:
        held = _local.held = {}
    if held.get(ticker):
        held[ticker] += 1
        try:
            yield
        finally:
            held[ticker] -= 1
        return

    lock = None
    try:
        # thread_local=False so the renew thread can use the token.
        lock = _redis().lock(f"helios:stop_lock:{ticker}", timeout=LOCK_TTL_S,
                             blocking_timeout=DEFAULT_WAIT_S if wait is None else wait,
                             thread_local=False)
        acquired = lock.acquire()
    except Exception as e:
        # Fails open on purpose: Redis is also the Celery broker, so while it
        # is down few task runners can race on a ticker, and blocking a stop
        # move would leave a position less protected than an unlocked change
        # (professor, audit 2026-11-07 13:07 #9). Each skip is recorded.
        log.warning(f"[{ticker}] stop lock unavailable, continuing unlocked: {e}")
        _record_lock_skip(ticker, e)
        lock, acquired = None, True
    if not acquired:
        raise TickerBusy(ticker)
    if lock is not None and ticker in _SKIP_LOGGED:
        _flush_skip_carry(ticker)

    held[ticker] = 1
    done = threading.Event()
    lost = threading.Event()
    if not hasattr(_local, "lost"):
        _local.lost = {}
    _local.lost[ticker] = lost
    if lock is not None:
        threading.Thread(target=_keep_renewed, args=(lock, ticker, done, None, lost),
                         daemon=True, name=f"stop-lock-{ticker}").start()
    try:
        yield
    finally:
        done.set()
        held.pop(ticker, None)
        _local.lost.pop(ticker, None)
        if lock is not None:
            try:
                lock.release()
            except Exception as e:
                # Lost (a renew failed) or Redis went away.
                log.warning(f"[{ticker}] stop lock release failed: {e}")


def _keep_renewed(lock, ticker: str, done: threading.Event,
                  max_hold_s: float | None = None,
                  lost: threading.Event | None = None) -> None:
    """Renew until done, but never past max_hold_s: an IBKR call can hang
    with no timeout, and a lock renewed forever would freeze every stop
    change on the ticker until a restart (audit 02:53, A0253-2)."""
    import time
    limit = time.monotonic() + (MAX_HOLD_S if max_hold_s is None else max_hold_s)
    while not done.wait(LOCK_RENEW_S):
        if time.monotonic() > limit:
            if lost is not None:
                lost.set()
            log.error(f"[{ticker}] lock held past {max_hold_s or MAX_HOLD_S:.0f}s; no longer "
                      f"renewing (it expires in {LOCK_TTL_S}s)")
            try:
                import alerts
                alerts._send(f"⚠️ <b>{ticker}</b>: a stop change has been running for over "
                             f"{(max_hold_s or MAX_HOLD_S) / 60:.0f} min (likely a hung broker "
                             f"call). Its lock is being released; check the orders worker.")
            except Exception:
                pass
            return
        try:
            lock.reacquire()
        except Exception as e:
            if lost is not None:
                lost.set()
            log.error(f"[{ticker}] stop lock renew failed (another stop change may get in): {e}")
            return


def locked_by_ticker(ticker_of, on_busy=None):
    """Decorator: run the function under ticker_lock(ticker_of(*args, **kwargs)).
    If the lock stays busy, log and return on_busy instead of running (or
    on_busy() if it is callable)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ticker = ticker_of(*args, **kwargs)
            try:
                with ticker_lock(ticker):
                    return fn(*args, **kwargs)
            except TickerBusy:
                log.error(f"[{ticker}] {fn.__name__} skipped: another stop change held the "
                          f"ticker for {DEFAULT_WAIT_S}s")
                return on_busy() if callable(on_busy) else on_busy
        return wrapper
    return deco
