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
    Redis, where a test ticker's lock could block the running bot.
    _FakeLock.expire(name) simulates the TTL running out: the holder's
    renew and release then fail as redis-py's do (C23)."""
    held: set = set()
    renewed: list = []
    owner: dict = {}

    def __init__(self, name, **_):
        self.name = name

    def acquire(self):
        if self.name in _FakeLock.held:
            return False
        _FakeLock.held.add(self.name)
        _FakeLock.owner[self.name] = self
        return True

    def _check_owned(self):
        if self.name in _FakeLock.owner and _FakeLock.owner[self.name] is not self:
            raise RuntimeError(f"LockNotOwnedError: {self.name}")
        if self.name not in _FakeLock.held:
            raise RuntimeError(f"LockNotOwnedError: {self.name} expired")

    def release(self):
        self._check_owned()
        _FakeLock.held.discard(self.name)
        _FakeLock.owner.pop(self.name, None)

    def reacquire(self):
        self._check_owned()
        _FakeLock.renewed.append(self.name)

    @classmethod
    def expire(cls, name):
        cls.held.discard(name)
        cls.owner.pop(name, None)


class _FakeRedis:
    store: dict = {}

    def lock(self, name, **kw):
        return _FakeLock(name, **kw)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in _FakeRedis.store:
            return None
        _FakeRedis.store[key] = value
        return True

    def exists(self, key):
        return int(key in _FakeRedis.store)

    def get(self, key):
        v = _FakeRedis.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def scan_iter(self, pattern, count=None):
        import fnmatch
        return [k for k in list(_FakeRedis.store) if fnmatch.fnmatch(k, pattern)]

    def delete(self, key):
        return int(_FakeRedis.store.pop(key, None) is not None)

    def ping(self):
        return True


@pytest.fixture(autouse=True)
def _no_live_redis(monkeypatch):
    import stop_lock
    _FakeLock.held = set()
    _FakeLock.renewed = []
    _FakeLock.owner = {}
    _FakeRedis.store = {}
    monkeypatch.setattr(stop_lock, "_redis", lambda: _FakeRedis())


@pytest.fixture(autouse=True)
def _spill_to_tmp(tmp_path, monkeypatch):
    """A test whose db write hits the blocked connection gets spilled; keep
    that out of logs/db_write_spill.jsonl, and reset the alert throttle."""
    import db
    monkeypatch.setattr(db, "_SPILL_PATH", tmp_path / "db_write_spill.jsonl")
    monkeypatch.setattr(db, "_LAST_WRITE_ALERT", {})


@pytest.fixture(autouse=True)
def _no_live_ibkr(monkeypatch):
    """Like the DB guard: an unmocked broker call fails fast instead of
    trying to reach a real gateway (one test spent 9 s retrying, 10-28)."""
    import ibkr

    def _blocked(*a, **k):
        raise RuntimeError("tests must not connect to IBKR; mock the ibkr.* call")
    monkeypatch.setattr(ibkr, "get_ib", _blocked)


@pytest.fixture(autouse=True)
def _buy_room_passes(request, monkeypatch):
    """Execution tests mock the broker; the execution-time cash/cap check
    (execution._buy_room_at_execution) passes through unless a test marks
    itself `real_buy_room` to test the check itself."""
    if request.node.get_closest_marker("real_buy_room"):
        return
    import execution
    monkeypatch.setattr(execution, "_buy_room_at_execution",
                        lambda ticker, action, shares, price: (shares, None))


def pytest_configure(config):
    config.addinivalue_line("markers", "real_buy_room: run the real execution-time cash/cap check")
