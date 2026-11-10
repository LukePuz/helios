import os

# Point Celery and every other Redis user at a dead port before any project
# module is imported, so no test can reach the live broker even if a publish
# path gets past the send_task trap below. load_dotenv never overrides a set
# variable, so .env can't undo this.
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
# No test may reach the real Telegram bot: with these blank, alerts.py,
# telegram_bot.py, deploy/watchdog.py and deploy/notify_failure.py all skip
# sending. Tests of the send path patch the module token and requests.post.
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""
# The Ollama servers on this box are the live model servers (one GPU slot).
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:1"
os.environ["OLLAMA_CPU_HOST"] = "http://127.0.0.1:1"
# ingestion.edgar builds its ticker -> CIK map from SEC at import time.
os.environ["HELIOS_EDGAR_OFFLINE"] = "1"

import tempfile
from pathlib import Path

# Same idea for the db write spill: db._SPILL_PATH defaults to the repo's
# logs/db_write_spill.jsonl, and a per-test monkeypatch only covers the test
# body. A feed-emit thread still running while that patch is undone (the undo
# is last-in-first-out, so _SPILL_PATH comes back before the db guard does)
# appended real lines to the live spill. With this set before db is imported,
# every undo lands on a session temp file instead.
_SESSION_SPILL_DIR = tempfile.mkdtemp(prefix="helios-test-spill-")
os.environ["HELIOS_DB_SPILL_PATH"] = os.path.join(_SESSION_SPILL_DIR, "db_write_spill.jsonl")
_REPO_SPILL = Path(__file__).resolve().parent.parent / "logs" / "db_write_spill.jsonl"

import pytest


def _spill_size():
    try:
        return _REPO_SPILL.stat().st_size
    except OSError:
        return None


def pytest_sessionstart(session):
    session.config._repo_spill_size_at_start = _spill_size()


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if the repo's logs/db_write_spill.jsonl grew: no test may
    write there. Only the file size is checked, never its contents."""
    import threading
    for t in threading.enumerate():
        if t.name == "feed-emit":
            t.join(timeout=5)
    before = getattr(session.config, "_repo_spill_size_at_start", None)
    after = _spill_size()
    if after is not None and after != (before or 0):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        msg = (f"tests wrote to {_REPO_SPILL} (size {before} -> {after} bytes); "
               f"a db write escaped the spill redirect")
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.ensure_newline()
            reporter.write_line(msg, red=True)
        else:
            print(msg)


class _SyncThread:
    """feed.emit's daemon thread, run inline: the db write then happens
    inside the test, under its patches, and can't outlive them."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
        self.name, self.daemon = name, daemon

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


@pytest.fixture(autouse=True)
def _feed_emit_inline(monkeypatch):
    """feed.emit looks up threading.Thread at call time, so this also covers
    modules holding a reference to emit itself."""
    import threading
    import types
    import feed
    monkeypatch.setattr(feed, "threading", types.SimpleNamespace(Thread=_SyncThread))
    yield
    for t in threading.enumerate():
        if t.name == "feed-emit":
            t.join(timeout=5)


_BLOCKED_HTTP = []
_LIVE_HTTP = ("api.telegram.org", "localhost:11434", "127.0.0.1:11434", "localhost:11435", "127.0.0.1:11435")


@pytest.fixture(autouse=True)
def _no_real_telegram(monkeypatch):
    """Backstop for the blank token and Ollama hosts: an unmocked HTTP request
    to Telegram or a live Ollama server is refused and fails the test, even
    when the code under test swallows the exception."""
    import requests.sessions
    real = requests.sessions.Session.request
    blocked = _BLOCKED_HTTP
    blocked.clear()

    def guarded(self, method, url, *a, **k):
        if any(h in str(url) for h in _LIVE_HTTP):
            blocked.append(str(url).split("/bot")[0])
            raise RuntimeError("test tried to call the real Telegram API or a live Ollama server")
        return real(self, method, url, *a, **k)
    monkeypatch.setattr(requests.sessions.Session, "request", guarded)
    yield
    assert not blocked, f"unmocked request to a live service: {blocked}"


@pytest.fixture(autouse=True)
def _no_extra_option_chains(monkeypatch):
    """The options shadow prices extra expiries over the network (yfinance):
    stubbed unless a test patches it itself."""
    from ingestion import options_free
    monkeypatch.setattr(options_free, "extra_expiry_moves",
                        lambda ticker, earnings_date, today: {"fetched": True, "stub": True})


@pytest.fixture(autouse=True)
def _no_earnings_dates_fetch(monkeypatch):
    """Earnings reviews read yfinance's earnings_dates for BMO/AMC: no
    network in tests."""
    from agents import crew
    monkeypatch.setattr(crew, "_fetch_earnings_dates", lambda ticker: None)


@pytest.fixture(autouse=True)
def _no_boot_notice(monkeypatch, request):
    """watchdog.main() sends a once-per-boot Telegram notice from the real
    journal and writes logs/.watchdog_boot_id: stubbed except in its own
    tests. The watchdog loads as "watchdog" or "deploy.watchdog"."""
    import sys
    if request.module.__name__.endswith("test_watchdog_boot_notice"):
        return
    for name in ("watchdog", "deploy.watchdog"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "_send_boot_notice"):
            monkeypatch.setattr(mod, "_send_boot_notice", lambda: None)


@pytest.fixture(autouse=True)
def _no_live_vix(monkeypatch):
    """Verdict provenance reads live ^VIX from Redis (cache_live_vix fetches
    it from yfinance): no network or Redis in tests."""
    from agents import analyst, market_regime
    monkeypatch.setattr(market_regime, "get_intraday_vix", lambda: None)
    monkeypatch.setattr(analyst, "_vix_cache_get", lambda: None)


@pytest.fixture(autouse=True)
def _no_kombu_publish(monkeypatch):
    """Backstop for the send_task trap: any message published through kombu
    (control broadcasts, a second app, a raw Producer) fails the test."""
    import kombu.messaging

    def _blocked(*a, **k):
        raise RuntimeError("test tried to publish a message through kombu")
    monkeypatch.setattr(kombu.messaging.Producer, "publish", _blocked)


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
def _strict_state_reads_follow_get_state(monkeypatch):
    """db.get_state_strict reads through whatever db.get_state is patched to
    in a test, so existing state mocks cover both; a mock that raises makes
    the strict read raise (professor M142). test_get_state_strict tests the
    real one."""
    import db
    real = getattr(db, "_real_get_state_strict", None) or db.get_state_strict
    monkeypatch.setattr(db, "_real_get_state_strict", real, raising=False)
    monkeypatch.setattr(db, "get_state_strict", lambda key, default=None: db.get_state(key, default))


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
    # The Ollama identity lookup (digest, version, model parameters): a fresh
    # fixed value, so no test reaches a live Ollama server.
    monkeypatch.setattr(analyst, "_OLLAMA_ID",
                        {"at": float("inf"), "value": {"model_digest": "testdigest",
                                                       "ollama_version": "test"}})


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
    # Transient connect retries still run, without the sleeps.
    monkeypatch.setattr(db, "_CONNECT_RETRY_DELAYS", (0.0, 0.0))


@pytest.fixture(autouse=True)
def _alert_policy_to_tmp(tmp_path, monkeypatch):
    """Keep the Telegram outbox and the dedup state out of logs/, fresh per
    test."""
    import alert_policy
    monkeypatch.setattr(alert_policy, "OUTBOX_PATH", tmp_path / "telegram_outbox.jsonl")
    monkeypatch.setattr(alert_policy, "STATE_PATH", tmp_path / ".alert_policy_state.json")


@pytest.fixture(autouse=True)
def _no_live_ibkr(monkeypatch):
    """Like the DB guard: an unmocked broker call fails fast instead of
    trying to reach a real gateway (one test spent 9 s retrying, 10-28)."""
    import ibkr

    def _blocked(*a, **k):
        raise RuntimeError("tests must not connect to IBKR; mock the ibkr.* call")
    monkeypatch.setattr(ibkr, "get_ib", _blocked)


@pytest.fixture(autouse=True)
def _no_live_celery_publish(monkeypatch):
    """An unpatched apply_async publishes to the live Redis broker, and the
    running ops worker then executes it against the live database: two
    held-sell tests queued tasks.retry_held_sell(1) on every run, and the
    worker ran retry tasks for live approval #1 (IONQ, 3185 h old) and
    re-analysed it. The publish raises, and the test fails at teardown even
    when the code under test swallows the error."""
    import tasks
    hits = []

    def _blocked(name, *a, **k):
        hits.append(name)
        raise RuntimeError(f"tests must not publish {name} to the live broker; "
                           f"patch its apply_async")
    monkeypatch.setattr(tasks.app, "send_task", _blocked)
    yield hits
    if hits:
        pytest.fail(f"test published Celery task(s) to the live broker: {hits}")


@pytest.fixture(autouse=True)
def _buy_room_passes(request, monkeypatch):
    """Execution tests mock the broker; the execution-time cash/cap check
    (execution._buy_room_at_execution) passes through unless a test marks
    itself `real_buy_room` to test the check itself."""
    if request.node.get_closest_marker("real_buy_room"):
        return
    import execution
    monkeypatch.setattr(execution, "_buy_room_at_execution",
                        lambda ticker, action, shares, price, sized_at=None: (shares, None))
    monkeypatch.setattr(execution, "_manual_buy_room", lambda ticker, qty, price: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_buy_room: run the real execution-time cash/cap check")


@pytest.fixture(autouse=True)
def _new_fact_gate_off(monkeypatch):
    """The buy gate on the screener's new_fact (crew.new_fact_gate) is on by
    default; tests not about it route buys with it off, as before it existed.
    Its own tests patch config.BUY_REQUIRES_NEW_FACT back on."""
    import config
    monkeypatch.setattr(config, "BUY_REQUIRES_NEW_FACT", False)


@pytest.fixture(autouse=True)
def _clear_qualified_contracts():
    """ibkr._stock caches qualified contracts per process; keep tests apart."""
    try:
        import ibkr
        ibkr._QUALIFIED.clear()
    except Exception:
        pass
    yield
