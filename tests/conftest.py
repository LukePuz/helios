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
# pyarrow fixes its allocator when it is first imported (pandas imports it).
# config.py sets the system pool first in the bot; some test modules import
# pandas before config, and whichever ran first decided the pool for the
# whole session. Set it here, before any test module, as config does.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

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

# HELIOS_TEST_SHIPPED_DEFAULTS=1 runs the suite on config.py's shipped switch
# values: the switch-off fixtures below (the ones calling _switch_off) leave config
# alone, so every test sees the combination that deploys. The fixtures that
# stub DB, network and broker reads stay on in both modes. A test that needs a
# switch at a given value says so with @pytest.mark.switches(NAME=value),
# applied in both modes after the switch-off fixtures (_explicit_switches).
# Run: HELIOS_TEST_SHIPPED_DEFAULTS=1 pytest -q tests  (see tests/README.md).
SHIPPED_DEFAULTS = os.getenv("HELIOS_TEST_SHIPPED_DEFAULTS", "").strip() == "1"


def _switch_off(monkeypatch, config, name, value):
    """A switch-off fixture's setattr: applied in the default mode only."""
    if not SHIPPED_DEFAULTS:
        monkeypatch.setattr(config, name, value)


def pytest_report_header(config):
    return ("config switches: SHIPPED DEFAULTS (HELIOS_TEST_SHIPPED_DEFAULTS=1)"
            if SHIPPED_DEFAULTS else "config switches: conftest switch-off defaults")


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
def _no_regime_shadow_fetch(monkeypatch):
    """Each regime assessment makes a second, wider bar request for its
    log-only shadow_200 record: no network in tests. Stubbed as "no
    credentials" unless a test patches it itself."""
    from agents import market_regime
    monkeypatch.setattr(market_regime, "_fetch_shadow_bars", lambda ticker: None)



@pytest.fixture(autouse=True)
def _no_stopout_reentry_read(monkeypatch):
    """Each OPEN/ADD past the rebuy cooldown reads the last exit the bot did
    not place, for its record-only stop-out re-entry shadow. Stubbed as "no
    such exit" (no row) unless a test patches it itself."""
    import db
    monkeypatch.setattr(db, "get_last_broker_exit", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_entry_quality_row(monkeypatch):
    """Each OPEN/ADD past the rebuy cooldown writes a record-only
    ENTRY_QUALITY_SHADOW row from a price_history read. Off here so existing
    tests see no new row or read; tests/test_entry_quality_shadow.py turns it
    on."""
    import config
    _switch_off(monkeypatch, config, "ENTRY_QUALITY_SHADOW_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_reclaim_book_io(monkeypatch):
    """An entry gate block stores a record-only reclaim shadow record
    (agents/crew.py:_entry_gate_reclaim_record) in system_state under a Redis
    lock. Stubbed as "not stored" (no Redis or DB) unless a test restores it;
    tests/test_entry_gate_reclaim_shadow.py does."""
    from agents import crew
    monkeypatch.setattr(crew, "_reclaim_book_edit", lambda fn, wait_s=None: False)


@pytest.fixture(autouse=True)
def _no_deal_state_io(monkeypatch):
    """A screened deal item is folded into a record-only per-ticker deal
    record (agents/material_event.deal_state_note) in system_state under a
    Redis lock. Stubbed as "not stored" (no Redis or DB) unless a test
    restores it; tests/test_material_event.py does."""
    from agents import material_event
    monkeypatch.setattr(material_event, "deal_state_note", lambda *a, **k: False)


@pytest.fixture(autouse=True)
def _no_vol_shadow_read(monkeypatch):
    """size_position's record-only volatility shadows read cached bars from
    price_history. Stubbed as "no cached bars" (no DB read) unless a test
    patches it itself; tests/test_vol_target_shadow.py does."""
    import risk_manager
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda ticker, window: [])


@pytest.fixture(autouse=True)
def _no_book_heat_read(monkeypatch):
    """The route's record-only book heat reads the newest portfolio snapshot
    and position_stops. Stubbed as "no snapshot" (no DB read) unless a test
    patches it itself; tests/test_book_heat.py does."""
    from agents import crew
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: None)


@pytest.fixture(autouse=True)
def _no_recheck_contradiction_read(monkeypatch):
    """Every OPEN/ADD past the rebuy cooldown reads the recent verdicts for a
    record-only field (agents/crew.py:_recheck_contradiction_record). Off here
    so existing routing tests see no new read or field;
    tests/test_recheck_contradiction.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "RECHECK_CONTRADICTION_SHADOW_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_same_day_consensus_read(monkeypatch):
    """Every OPEN/ADD past the rebuy cooldown reads today's earlier verdicts
    for a record-only field (agents/crew.py:_same_day_consensus_record). Off
    here so existing routing tests see no new read or field;
    tests/test_same_day_consensus.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "SAME_DAY_CONSENSUS_RECORD_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_news_timeline(monkeypatch):
    """Each event/scan verdict queues a record-only news-timeline job for a
    side thread (agents/news_timeline.py). Off here so existing analyst
    tests queue nothing; tests/test_news_timeline.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "NEWS_TIMELINE_RECORD_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_scan_tier_funnel(monkeypatch):
    """Each daily scan writes a record-only tier funnel row after its
    enqueue (agents/scan_funnel.py). Off here so existing scan tests write
    nothing; tests/test_scan_tier_funnel.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "SCAN_TIER_FUNNEL_RECORD_ENABLED", False)


@pytest.fixture(autouse=True)
def _inert_news_timeline_lane(request, monkeypatch):
    """In both modes, a news-timeline job queued by any test outside the
    lane's own (@pytest.mark.news_timeline_lane) goes to a list on
    agents.news_timeline (_TEST_JOBS), not to a thread: a thread would run
    the job after the verdict, possibly into this test's own db mocks (a
    mocked log_activity_queued) or the next test's. The hook and its switch
    still run as shipped; only the thread is not started."""
    from agents import news_timeline
    if request.node.get_closest_marker("news_timeline_lane"):
        return
    jobs = []
    monkeypatch.setattr(news_timeline, "_TEST_JOBS", jobs, raising=False)
    monkeypatch.setattr(news_timeline, "enqueue", lambda job: jobs.append(job) or True)


@pytest.fixture(autouse=True)
def _entry_gates_off(monkeypatch):
    """The live entry gates (agents/crew.py:_entry_gate) skip an OPEN/ADD after
    a recent broker exit or below both moving averages. Off here so existing
    routing tests route as before; tests/test_entry_gate.py and
    tests/test_add_loser_gate.py turn them on."""
    import config
    _switch_off(monkeypatch, config, "STOPOUT_REENTRY_GATE_DAYS", 0)
    _switch_off(monkeypatch, config, "ENTRY_TREND_GATE_ENABLED", False)
    _switch_off(monkeypatch, config, "ADD_LOSER_GATE_MODE", "off")


@pytest.fixture(autouse=True)
def _no_add_lot_r_read(monkeypatch):
    """A BUY stopped over an existing stop record writes a record-only
    add_lot_r activity row from a trade_outcomes read (execution._record_add_lot_r).
    Off here so existing tests see no new read or row; tests/test_add_lot_r.py
    turns it on."""
    import config
    _switch_off(monkeypatch, config, "ADD_LOT_R_SHADOW_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_add_stop_qty_from_broker(monkeypatch):
    """An ADD's whole-position stop is sized from ibkr.cached_stock_qty
    (execution._add_stop_qty), a read the existing ADD tests don't mock. Off
    here so they keep sizing from the position reads;
    tests/test_add_stop_qty.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "ADD_STOP_QTY_FROM_BROKER", False)


@pytest.fixture(autouse=True)
def _no_more_stop_qty_from_broker(monkeypatch):
    """The profit-target remainder stop (risk_manager._pt_stop_qty,
    ibkr.cached_stock_qty) and a late BUY fill's expect_at_least
    (tasks._late_fill_expect, ibkr.get_broker_stock_qty) read counts, which the existing tests don't mock. Off here;
    tests/test_stop_qty_more.py turns them on."""
    import config
    _switch_off(monkeypatch, config, "PT_STOP_QTY_FROM_BROKER", False)
    _switch_off(monkeypatch, config, "LATE_FILL_EXPECT_MODE", "off")


@pytest.fixture(autouse=True)
def _fresh_positions_from_the_cache(monkeypatch):
    """ibkr.get_fresh_stock_positions (a fresh reqPositions() read) reads
    back the test's own portfolio cache (ibkr.get_client().get_all_positions),
    so the re-protect's real helpers always run without each test mocking
    the read. The real read runs instead when the test gave it the session
    that read uses: ibkr._ib set (any read), or a patched ibkr.get_ib for a
    reconnecting read (the default). A reconnect=False read (the
    re-protect's) uses ibkr._ib only, so a patched get_ib alone would make
    it raise ConnectionError where live has a session: that reads the cache.
    A cache that raises raises here too, as a failed read; a test patches
    the function itself to fake a disagreement or a missing session."""
    import ibkr
    real = ibkr.get_fresh_stock_positions

    def from_cache(*args, **kw):
        reconnect = kw.get("reconnect", args[1] if len(args) > 1 else True)
        if getattr(ibkr, "_ib", None) is not None or (reconnect and ibkr.get_ib is not _blocked_get_ib):
            return real(*args, **kw)
        out: dict = {}
        for p in ibkr.get_client().get_all_positions():
            qty = float(p.qty)
            if qty:
                out[p.symbol] = out.get(p.symbol, 0.0) + qty
        return out
    monkeypatch.setattr(ibkr, "get_fresh_stock_positions", from_cache)


@pytest.fixture(autouse=True)
def _no_fill_record_wait(monkeypatch):
    """ibkr.place_market_order's fill-record wait (config.FILL_TIME_WAIT_S,
    m141) is off unless a test sets the cap itself: many fake trades are
    Filled with no fills, and their ib.sleep returns at once, so the wait
    would spin to the cap. tests/test_fill_time_complete.py sets it."""
    import config
    monkeypatch.setattr(config, "FILL_TIME_WAIT_S", 0)


@pytest.fixture(autouse=True)
def _no_would_be_stop_read(monkeypatch):
    """A gate-blocked buy's record-only would-be sizing reads cached bars for
    its stop (agents/crew.py:_would_be_stop). Stubbed as "no stop" (no DB
    read) unless a test patches it itself; tests/test_blocked_sizing_shadow.py
    does. Its vol bars and book heat inputs are stubbed by the two fixtures
    above."""
    from agents import crew

    def _no_stop(ticker, price, bucket):
        raise RuntimeError("stubbed in tests")
    monkeypatch.setattr(crew, "_would_be_stop", _no_stop)


@pytest.fixture(autouse=True)
def _no_story_dedup_read(monkeypatch):
    """Each event analysis reads the ticker's earlier event verdicts for its
    record-only story de-dup shadow (agents/story_dedup.py). Off here so
    existing tests see no new read or trigger key; tests/test_story_dedup.py
    turns it on."""
    import config
    _switch_off(monkeypatch, config, "STORY_DEDUP_SHADOW_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_plan_compiled_hit_row(monkeypatch):
    """watch_plan_levels writes a record-only plan_compiled_hit activity row
    the first time a compiled plan level is crossed (tasks._plan_compiled_hit_record).
    Off here so existing watcher tests see no new row or Redis key;
    tests/test_plan_compiled_hit.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "PLAN_COMPILED_HIT_SHADOW_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_late_fill_held_before(monkeypatch):
    """execute_approved stores a BUY's held count before the order on its row,
    and a late OPEN fill reads it (tasks._late_fill_held_decide). Off here so
    existing tests see no new read, write or row; tests/test_late_fill_held_before.py
    sets "shadow" or "enforce"."""
    import config
    _switch_off(monkeypatch, config, "LATE_FILL_HELD_BEFORE_MODE", "off")


@pytest.fixture(autouse=True)
def _no_held_exit_reads(monkeypatch):
    """Each trim/exit rule row reads cached bars and the active trade plan for
    its record-only held-exit candidates (agents/held_exit.py). Off here so
    existing tests see no new read; tests/test_held_exit.py turns it on."""
    import config
    _switch_off(monkeypatch, config, "HELD_EXIT_SHADOW_ENABLED", False)


@pytest.fixture(autouse=True)
def _explicit_switches(request, monkeypatch, _no_entry_quality_row, _no_recheck_contradiction_read,
                       _no_same_day_consensus_read, _no_news_timeline, _no_scan_tier_funnel,
                       _entry_gates_off, _no_add_lot_r_read, _no_add_stop_qty_from_broker,
                       _no_more_stop_qty_from_broker, _no_story_dedup_read,
                       _no_plan_compiled_hit_row, _no_late_fill_held_before, _no_held_exit_reads):
    """@pytest.mark.switches(NAME=value, ...): set by name on config, in both
    modes. It depends on every switch-off fixture, so it runs after them and
    the marker wins. A name config.py lacks is an error (a typo would test
    nothing)."""
    import config
    for mark in reversed(list(request.node.iter_markers("switches"))):   # closest mark last
        for name, value in mark.kwargs.items():
            if not hasattr(config, name):
                raise AttributeError(f"switches marker: config has no {name}")
            monkeypatch.setattr(config, name, value)


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


_ACTIVITY_SETTLE_S = 2.0   # how long a test's teardown waits for the writer's row in flight


def _settle_activity_queue(q) -> bool:
    """Drop the rows still queued on q (their test is over; nothing may write
    them through the next test's patches) and wait, bounded, for the row the
    writer is on. True when q is idle. Never raises or blocks unbounded."""
    import queue
    import time
    if q is None:
        return True
    try:
        while True:
            q.get_nowait()
            q.task_done()
    except queue.Empty:
        pass
    except Exception:
        return False
    end = time.monotonic() + _ACTIVITY_SETTLE_S
    while getattr(q, "unfinished_tasks", 0) and time.monotonic() < end:
        time.sleep(0.01)
    return not getattr(q, "unfinished_tasks", 0)


@pytest.fixture(autouse=True)
def _isolated_activity_writer(_no_real_database, monkeypatch):
    """db.log_activity_queued's writer thread outlives the test that queued a
    row and calls whatever db.log_activity_direct is at write time, so a row
    queued by one test could reach the next test's patched writer (an
    add_stop_qty row failed a later assert_not_called). After each test, the
    queue the test started with and the one it ended with are settled
    (_settle_activity_queue). A queue whose writer is still busy after the
    bound is dropped from db's module references, so the next test starts a
    fresh queue and writer. db.py's runtime behaviour is unchanged. Depends
    on _no_real_database and monkeypatch so it tears down first: the DB is
    still blocked, and a test's own monkeypatched queue is still in place."""
    import db
    before = getattr(db, "_ACTIVITY_QUEUE", None)
    yield
    after = getattr(db, "_ACTIVITY_QUEUE", None)
    idle = all([_settle_activity_queue(q) for q in {id(x): x for x in (before, after)
                                                  if x is not None}.values()])
    if not idle:
        db._ACTIVITY_QUEUE = None
        db._ACTIVITY_WRITER = None
    # The news-timeline thread (agents/news_timeline.py) the same way: its
    # jobs read and write through whatever db is when they run.
    from agents import news_timeline
    if not _settle_activity_queue(news_timeline._QUEUE):
        news_timeline._QUEUE = None
        news_timeline._WORKER = None


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
    # ex is honoured against this test clock (seconds), which only a test
    # moves (_FakeRedis.clock += 51): a key set with ex is gone once the clock
    # reaches its expiry. Reset per test.
    clock: float = 0.0
    expires: dict = {}

    @classmethod
    def _expire(cls):
        for k, at in list(cls.expires.items()):
            if at <= cls.clock:
                cls.store.pop(k, None)
                cls.expires.pop(k, None)

    def lock(self, name, **kw):
        return _FakeLock(name, **kw)

    def set(self, key, value, ex=None, nx=False):
        _FakeRedis._expire()
        if nx and key in _FakeRedis.store:
            return None
        _FakeRedis.store[key] = value
        if ex is not None:
            _FakeRedis.expires[key] = _FakeRedis.clock + float(ex)
        else:
            _FakeRedis.expires.pop(key, None)
        return True

    def exists(self, key):
        # A held lock is an ordinary key in Redis.
        _FakeRedis._expire()
        return int(key in _FakeRedis.store or key in _FakeLock.held)

    def get(self, key):
        _FakeRedis._expire()
        v = _FakeRedis.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def scan_iter(self, pattern, count=None):
        import fnmatch
        _FakeRedis._expire()
        return [k for k in list(_FakeRedis.store) if fnmatch.fnmatch(k, pattern)]

    def delete(self, key):
        _FakeRedis._expire()
        _FakeRedis.expires.pop(key, None)
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
    _FakeRedis.clock = 0.0
    _FakeRedis.expires = {}
    monkeypatch.setattr(stop_lock, "_redis", lambda: _FakeRedis())
    # The stop coverage re-check's per-ticker dedupe key (tasks._RECHECK_KEY).
    monkeypatch.setattr("tasks._recheck_redis", lambda: _FakeRedis())


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


def _blocked_get_ib(*a, **k):
    raise RuntimeError("tests must not connect to IBKR; mock the ibkr.* call")


@pytest.fixture(autouse=True)
def _no_live_ibkr(monkeypatch):
    """Like the DB guard: an unmocked broker call fails fast instead of
    trying to reach a real gateway (one test spent 9 s retrying, 10-28)."""
    import ibkr

    monkeypatch.setattr(ibkr, "get_ib", _blocked_get_ib)


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
    config.addinivalue_line("markers", "news_timeline_lane: start the real news-timeline thread "
                            "(tests/conftest.py:_inert_news_timeline_lane)")
    config.addinivalue_line(
        "markers", "switches(**names): set these config switches for the test, in both the "
                   "default and the shipped-defaults mode (_explicit_switches)")


@pytest.fixture(autouse=True)
def _new_fact_gate_off(monkeypatch):
    """The buy gate on the screener's new_fact (crew.new_fact_gate) is on by
    default; tests not about it route buys with it off, as before it existed.
    Its own tests patch config.BUY_REQUIRES_NEW_FACT back on."""
    import config
    monkeypatch.setattr(config, "BUY_REQUIRES_NEW_FACT", False)


@pytest.fixture(autouse=True)
def _trim_exit_rule_off(monkeypatch):
    """The trim/exit rule (agents/trim_exit.py) is off in tests whatever the
    checkout's .env says: switched on, every held review writes one more
    signals row, and tests that read log_signal's calls would see it. Its own
    tests turn it on (tests/test_trim_exit.py:_flags)."""
    import config
    monkeypatch.setattr(config, "TRIM_EXIT_ENABLED", False)
    monkeypatch.setattr(config, "TRIM_EXIT_LIVE", False)


@pytest.fixture(autouse=True)
def _clear_qualified_contracts():
    """ibkr._stock caches qualified contracts per process; keep tests apart."""
    try:
        import ibkr
        ibkr._QUALIFIED.clear()
    except Exception:
        pass
    yield
