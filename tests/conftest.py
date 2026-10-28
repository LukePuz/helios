import pytest


@pytest.fixture(autouse=True)
def _default_to_orders_worker_context(monkeypatch):
    """Most tests exercise stop-order logic (placement, cancellation,
    reconciliation) without caring about WHICH process is allowed to touch
    the broker. tasks.manage_stop_place and its siblings (added 2026-08-11,
    see their module-level docstring in tasks.py) route through Celery
    dispatch-and-wait unless IB_CLIENT_ID matches the dedicated orders
    worker -- a bare test environment has no IB_CLIENT_ID set at all, which
    would otherwise send every test down the dispatch branch and try to
    actually talk to Redis (hangs on .get(timeout=...) since nothing is
    consuming the queue). Default every test to "already the orders worker"
    -- the direct-call branch, which is what every existing ibkr.* mock
    already targets. Tests that specifically exercise the routing logic
    itself (tests/test_order_routing.py) override this locally per-test."""
    monkeypatch.setenv("IB_CLIENT_ID", "30")


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    """db.py loads .env at import, so in the live checkout an unpatched db.*
    call reaches the production database. test_sell_not_blocked_when_market_closed
    did exactly that from 08-04 to 10-19: 42 fake NVDA EXIT rows at $100 in
    trade_outcomes, and real NVDA rows closed at $100. Any test that needs
    the DB must mock the db function it calls."""
    import db

    def _blocked(*args, **kwargs):
        raise RuntimeError("tests must not touch the real database; mock the db.* call")

    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_get_pool", _blocked)
    monkeypatch.setattr(db.psycopg2.pool, "ThreadedConnectionPool", _blocked)
    monkeypatch.setattr(db.psycopg2, "connect", _blocked)


@pytest.fixture(autouse=True)
def _static_provenance(monkeypatch):
    """analyst._provenance shells out to git and asks Ollama for the model
    digest once per process; tests use a fixed stub instead."""
    from agents import analyst
    from agents import screener
    monkeypatch.setattr(screener, "_DIGEST_CACHE", {"d": "testdigest"})
    monkeypatch.setattr(analyst, "_CODE_SHA", {"code_sha": "test", "head_sha": "testhead"})
    monkeypatch.setattr(analyst, "_PROVENANCE_STATIC",
                        {"code_sha": "test", "model": "test", "model_digest": None,
                         "prompt_template_sha": "test"})


class _FakeLock:
    """In-process stand-in for a redis-py Lock: tests never touch the live
    Redis, where a test ticker's lock could block the running bot."""
    held: set = set()

    def __init__(self, name, **_):
        self.name = name

    def acquire(self):
        if self.name in _FakeLock.held:
            return False
        _FakeLock.held.add(self.name)
        return True

    def release(self):
        _FakeLock.held.discard(self.name)


class _FakeRedis:
    def lock(self, name, **kw):
        return _FakeLock(name, **kw)


@pytest.fixture(autouse=True)
def _no_live_redis(monkeypatch):
    import stop_lock
    _FakeLock.held = set()
    monkeypatch.setattr(stop_lock, "_redis", lambda: _FakeRedis())
