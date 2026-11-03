"""calculate_stop's flat-percent ATR fallback writes one log-only
activity_feed row (stage stop_atr_fallback) per ticker per ET day, from a
daemon thread, and the stop it returns is the same whether or not that
logging works."""
import sys
import threading
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import alerts
import risk_manager


class _FakeRedis:
    def __init__(self, calls):
        self.kv, self.sets, self.calls = {}, {}, calls

    def _rec(self, op):
        self.calls.append((op, threading.current_thread().name))

    def set(self, key, value, nx=False, ex=None):
        self._rec("set")
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def delete(self, key):
        self._rec("delete")
        self.kv.pop(key, None)

    def smembers(self, key):
        self._rec("smembers")
        return {t.encode() for t in self.sets.get(key, set())}

    def pipeline(self, transaction=True):
        return _FakePipe(self)


class _FakePipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def sadd(self, key, member):
        self.ops.append(("sadd", key, member))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def execute(self):
        self.r._rec("pipeline")
        for op in self.ops:
            if op[0] == "sadd":
                self.r.sets.setdefault(op[1], set()).add(op[2])
        return [True] * len(self.ops)


def _bars(n, rng=2.0):
    return [{"high": 100 + rng / 2, "low": 100 - rng / 2, "close": 100.0} for _ in range(n)]


def _rm():
    return risk_manager.RiskManager.__new__(risk_manager.RiskManager)


def _joined_spawn(spawned):
    def spawn(target, name, args=()):
        spawned.append(name)
        t = threading.Thread(target=target, args=args, name=name, daemon=True)
        t.start()
        t.join(timeout=5)
    return spawn


class _Env:
    def __init__(self):
        self.calls, self.spawned, self.rows = [], [], []
        self.redis = _FakeRedis(self.calls)
        self.write_ok = True

    def log_activity(self, *args):
        self.calls.append(("log_activity", threading.current_thread().name))
        if isinstance(self.write_ok, Exception):
            raise self.write_ok
        self.rows.append(args)
        return self.write_ok

    def fallback_rows(self):
        return [r for r in self.rows if r[0] == "stop_atr_fallback"]


@pytest.fixture
def env():
    e = _Env()

    def pool(*a, **k):
        e.calls.append(("pool", threading.current_thread().name))
        raise AssertionError("the fallback logger must not use the shared pool")
    with patch.object(risk_manager, "SETUP_GATES_ENABLED", False), \
         patch.object(risk_manager, "TRADING_ENV", "paper"), \
         patch.object(risk_manager, "_ATR_FALLBACK_SEM", threading.BoundedSemaphore(1)), \
         patch.object(risk_manager, "_ATR_FALLBACK_PENDING", []), \
         patch.object(risk_manager, "_ATR_FALLBACK_DROPPED", {"n": 0}), \
         patch.object(risk_manager, "_atr_fallback_redis", return_value=e.redis), \
         patch.object(risk_manager, "_spawn", _joined_spawn(e.spawned)), \
         patch("db.log_activity_direct", e.log_activity), \
         patch("db.log_activity", pool), \
         patch("db._get_conn", pool), \
         patch("db._get_pool", pool), \
         patch("alerts._send"):
        yield e
    assert all(op != "pool" for op, _ in e.calls)


def _stop(bars=None, raises=None, ticker="ZS", price=100.0, bucket="Accelerator", **kw):
    side = raises if raises else (lambda *a, **k: list(bars))
    with patch("ibkr.get_historical_bars", side_effect=side):
        return _rm().calculate_stop(ticker, price, bucket, **kw)


def test_no_redis_or_db_call_on_the_callers_thread(env):
    me = threading.current_thread().name
    _stop([], ticker="ZS")
    _stop([], ticker="ZS")          # deduped: a Redis call only
    assert env.calls and all(name != me for _op, name in env.calls)
    assert {name for _op, name in env.calls} == {"atr-fallback-log-ZS"}
    assert env.spawned == ["atr-fallback-log-ZS", "atr-fallback-log-ZS"]


def test_default_spawn_is_a_named_daemon_thread():
    seen = {}
    done = threading.Event()

    def target(x):
        t = threading.current_thread()
        seen.update(name=t.name, daemon=t.daemon, x=x)
        done.set()
    risk_manager._spawn(target, "atr-fallback-log-ZS", (1,))
    assert done.wait(5)
    assert seen == {"name": "atr-fallback-log-ZS", "daemon": True, "x": 1}


@pytest.mark.parametrize("bucket,pct", [("Accelerator", 0.05), ("Compounder", 0.08)])
def test_stop_is_identical_when_the_logging_raises(env, bucket, pct):
    base = _stop([], price=123.45, bucket=bucket, log_fallback=False)
    assert base == (round(123.45 * (1 - pct), 2),
                    f"Fallback stop ({pct:.0%} below entry — ATR unavailable)")
    with patch.object(risk_manager, "_spawn", side_effect=RuntimeError("no threads")):
        assert _stop([], price=123.45, bucket=bucket) == base
    with patch.object(risk_manager, "_queue_atr_fallback_log", side_effect=RuntimeError("boom")):
        assert _stop([], price=123.45, bucket=bucket) == base
    with patch.object(risk_manager, "_atr_fallback_reason", side_effect=RuntimeError("x")):
        assert _stop([], ticker="ORCL", price=123.45, bucket=bucket) == base
    env.write_ok = RuntimeError("db down")
    assert _stop([], ticker="SNPS", price=123.45, bucket=bucket) == base
    with patch.object(risk_manager, "_atr_fallback_redis", side_effect=RuntimeError("redis down")):
        assert _stop([], ticker="PLTR", price=123.45, bucket=bucket) == base
    assert _stop([], ticker="KLAC", price=123.45, bucket=bucket, log_fallback=False) == base


def test_a_fallback_writes_one_row_via_the_hook(env):
    analyst = type(sys)("agents.analyst")
    analyst.shadow_versions = lambda: {"code_sha": "c1", "env_sha": "e1"}
    with patch.dict(sys.modules, {"agents.analyst": analyst}):
        stop, _ = _stop([], ticker="ZS", price=200.0)
    rows = env.fallback_rows()
    assert len(rows) == 1 and env.spawned == ["atr-fallback-log-ZS"]
    stage, ticker, _msg, sentiment, extra = rows[0]
    assert (stage, ticker, sentiment) == ("stop_atr_fallback", "ZS", "neutral")
    assert extra == {
        "ticker": "ZS", "caller": "other", "reason": "no_bars", "n_bars": 0, "error": None, "atr": None,
        "stop_basis": "flat_pct_fallback", "fallback_pct": 0.05, "stop_price": stop,
        "price": 200.0, "bucket": "Accelerator", "env": "paper",
        "et_date": extra["et_date"], "dedup": "ok", "code_sha": "c1", "env_sha": "e1",
        "queued_behind_logger": False, "dropped_before": 0}
    assert stop == 190.0
    assert risk_manager.atr_fallback_tickers_today() == ["ZS"]


def test_versions_are_none_without_analyst_loaded(env):
    with patch.dict(sys.modules, {"agents.analyst": None}):
        _stop([], ticker="ZS")
    extra = env.fallback_rows()[0][4]
    assert extra["code_sha"] is None and extra["env_sha"] is None


def test_reasons(env):
    _stop(_bars(1), ticker="SNPS")
    _stop(raises=RuntimeError("HMDS query returned no data"), ticker="MSFT")
    _stop(_bars(5, rng=0.0), ticker="MRVL")   # ATR of 0
    reasons = {r[1]: (r[4]["reason"], r[4]["n_bars"], r[4]["error"]) for r in env.fallback_rows()}
    assert reasons == {"SNPS": ("too_few_bars", 1, None),
                       "MSFT": ("exception", 0, "RuntimeError"),
                       "MRVL": ("other", 5, None)}
    assert env.fallback_rows()[2][4]["atr"] == 0


def test_get_atr_replaced_reads_other(env):
    rm = _rm()
    _stop(_bars(1), ticker="ZS")           # leaves ZS diagnostics on this thread
    with patch.object(risk_manager.RiskManager, "get_atr", return_value=None):
        rm.calculate_stop("ZS", 100.0, "Accelerator")
        rm.calculate_stop("KLAC", 100.0, "Accelerator")
    assert [(r[1], r[4]["reason"]) for r in env.fallback_rows()] == [
        ("ZS", "too_few_bars"), ("KLAC", "other")]


def test_no_row_on_the_atr_path(env):
    stop, method = _stop(_bars(15), ticker="NVDA", price=100.0)
    assert method.endswith("below entry = $" + f"{stop:.2f}") and "ATR" in method
    assert env.fallback_rows() == [] and env.calls == [] and env.spawned == []


def test_moonshot_stop_does_not_log(env):
    with patch("ibkr.get_historical_bars", return_value=[]):
        _rm().calculate_stop("RGTI", 10.0, "Moonshot")
    assert env.fallback_rows() == [] and env.spawned == []


def test_log_fallback_false_writes_nothing(env):
    _stop([], ticker="ZS", log_fallback=False)
    assert env.fallback_rows() == [] and env.calls == [] and env.spawned == []


def test_size_position_passes_log_fallback_through(env):
    seen = []

    def fake_stop(self, t, p, b, log_fallback=True, log_caller="other"):
        seen.append((log_fallback, log_caller))
        raise RuntimeError("stop here")
    rm = _rm()
    with patch.object(risk_manager.RiskManager, "calculate_stop", fake_stop), \
         patch.object(risk_manager.RiskManager, "is_halted", return_value=False, create=True), \
         patch.object(risk_manager.RiskManager, "portfolio_value", return_value=100_000.0, create=True), \
         patch.object(risk_manager.RiskManager, "cash", return_value=100_000.0, create=True):
        for flag in (True, False):
            try:
                rm.size_position("ZS", 7, "Accelerator", "Neutral", 100.0, log_fallback=flag)
            except Exception:
                pass
    assert seen == [(True, "size_position"), (False, "size_position")]


def test_swap_simulation_does_not_log():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=100_000.0, cash=0.0,
                                                buying_power=0.0)
    client.get_all_positions.return_value = []
    seen = []

    def fake_size(self_, *a, **k):
        seen.append(k.get("log_fallback"))
        return {"shares_initial": 0}
    with patch("db.get_state", return_value=None), patch("db.set_state"):
        rm = risk_manager.RiskManager(client)
        with patch.object(risk_manager.RiskManager, "size_position", fake_size):
            rm.would_size_after_swap("NEW", 7, "Accelerator", "Neutral", 100.0,
                                     {"ticker": "OLD", "market_value": 0.0})
    assert seen == [False]


def test_marker_deleted_when_the_write_fails(env):
    env.write_ok = False
    _stop([], ticker="ZS")
    assert env.redis.kv == {} and ("delete", "atr-fallback-log-ZS") in env.calls
    assert risk_manager.atr_fallback_tickers_today() == []
    env.write_ok = True
    _stop([], ticker="ZS")                 # retried the same day
    assert [r[1] for r in env.fallback_rows()] == ["ZS", "ZS"]   # failed attempt, then the retry
    assert len(env.redis.kv) == 1 and risk_manager.atr_fallback_tickers_today() == ["ZS"]


def test_dedup_within_a_day_and_next_day_logs_again(env):
    day1 = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)

    class _DT(datetime):
        at = day1

        @classmethod
        def now(cls, tz=None):
            return cls.at.astimezone(tz) if tz else cls.at

    with patch.object(risk_manager, "datetime", _DT):
        first = _stop([], ticker="ZS")
        assert _stop([], ticker="ZS") == first
        _stop([], ticker="ZS")
        _stop([], ticker="ORCL")
        assert [r[1] for r in env.fallback_rows()] == ["ZS", "ORCL"]
        assert risk_manager.atr_fallback_tickers_today(now=day1) == ["ORCL", "ZS"]
        _DT.at = day2
        _stop([], ticker="ZS")
        assert risk_manager.atr_fallback_tickers_today(now=day2) == ["ZS"]
    rows = env.fallback_rows()
    assert [r[1] for r in rows] == ["ZS", "ORCL", "ZS"]
    assert [r[4]["et_date"] for r in rows] == ["2026-11-03", "2026-11-03", "2026-11-04"]


def test_dedup_failure_still_logs_and_skips_other_redis_calls(env):
    with patch.object(risk_manager, "_atr_fallback_redis", side_effect=ConnectionError("no redis")):
        _stop([], ticker="ZS")
        _stop([], ticker="ZS")
    rows = env.fallback_rows()
    assert len(rows) == 2 and all(r[4]["dedup"] == "unavailable" for r in rows)
    assert [op for op, _ in env.calls] == ["log_activity", "log_activity"]


def test_tickers_unavailable_without_redis():
    with patch.object(risk_manager, "_atr_fallback_redis", side_effect=ConnectionError("x")):
        assert risk_manager.atr_fallback_tickers_today() is None


def test_own_redis_client_has_short_timeouts():
    with patch.dict(risk_manager._ATR_FALLBACK_REDIS, clear=True):
        client = risk_manager._atr_fallback_redis()
        kw = client.connection_pool.connection_kwargs
        assert kw["socket_timeout"] == 0.5 and kw["socket_connect_timeout"] == 0.5
        assert risk_manager._atr_fallback_redis() is client


def test_digest_reads_the_ticker_set(env):
    env.redis.sets["helios:stop_atr_fallback_tickers:paper:2026-11-03"] = {"ZS", "ORCL"}
    tickers = risk_manager.atr_fallback_tickers_today(
        now=datetime(2026, 11, 3, 21, 15, tzinfo=timezone.utc))
    sent = []
    with patch.object(alerts, "_send", side_effect=sent.append):
        kw = dict(portfolio_value=1000.0, day_pl=0.0, day_pl_pct=0.0, spy_pl_pct=None,
                  positions_text=[], catalysts_text=[], pending_count=0, halted=False,
                  ingestion_issues=[])
        alerts.send_daily_digest(**kw)
        alerts.send_daily_digest(**kw, atr_fallback_tickers=tickers)
        alerts.send_daily_digest(**kw, atr_fallback_tickers=["ZS"])
    assert "ATR fallback" not in sent[0]
    assert "ATR fallback (sizing or stop) today: 2 tickers (ORCL, ZS)" in sent[1]
    assert "ATR fallback (sizing or stop) today: 1 ticker (ZS)" in sent[2]


def test_a_second_concurrent_log_is_queued_for_the_running_logger(env):
    """Professor m308: a fallback while the logger ran was dropped, so rows
    and the digest undercounted. It now waits and the same thread logs it."""
    release = threading.Event()
    started = []

    def slow_spawn(target, name, args=()):
        started.append(name)

        def body():
            release.wait(5)
            target(*args)
        t = threading.Thread(target=body, name=name, daemon=True)
        t.start()
        slow_spawn.threads.append(t)
    slow_spawn.threads = []
    with patch.object(risk_manager, "_spawn", slow_spawn):
        first = _stop([], ticker="ZS")
        second = _stop([], ticker="ORCL")      # the first logger still holds the slot
        assert first == second and first[0] == 95.0
        assert started == ["atr-fallback-log-ZS"]
        release.set()
        slow_spawn.threads[0].join(5)
        _stop([], ticker="MSFT")               # slot released in the thread's finally
        slow_spawn.threads[1].join(5)
    assert started == ["atr-fallback-log-ZS", "atr-fallback-log-MSFT"]
    rows = env.fallback_rows()
    assert [r[1] for r in rows] == ["ZS", "ORCL", "MSFT"]
    assert [r[4]["queued_behind_logger"] for r in rows] == [False, True, False]
    assert all(r[4]["dropped_before"] == 0 for r in rows)
    assert {name for op, name in env.calls if op == "log_activity"} == {
        "atr-fallback-log-ZS", "atr-fallback-log-MSFT"}


def test_past_the_queue_cap_fallbacks_are_counted_on_the_next_row(env):
    release = threading.Event()
    threads = []

    def slow_spawn(target, name, args=()):
        def body():
            release.wait(5)
            target(*args)
        t = threading.Thread(target=body, name=name, daemon=True)
        t.start()
        threads.append(t)
    with patch.object(risk_manager, "_spawn", slow_spawn), \
         patch.object(risk_manager, "_ATR_FALLBACK_PENDING_MAX", 2):
        _stop([], ticker="ZS")                  # holds the slot
        for t in ("ORCL", "MSFT", "AMD", "NVDA"):
            _stop([], ticker=t)                 # 2 queued, 2 dropped
        assert risk_manager._ATR_FALLBACK_DROPPED["n"] == 2
        release.set()
        threads[0].join(5)
    rows = env.fallback_rows()
    assert [r[1] for r in rows] == ["ZS", "ORCL", "MSFT"]
    assert rows[0][4]["dropped_before"] == 2 and rows[1][4]["dropped_before"] == 0
    assert risk_manager._ATR_FALLBACK_DROPPED["n"] == 0


class _SlotFreedAfterFirstTry:
    """The logger slot reads busy on the first try and free after: the
    running logger finished its last queue check between this caller's
    failed acquire and its append."""

    def __init__(self):
        self.sem = threading.BoundedSemaphore(1)
        self.tries = 0

    def acquire(self, blocking=True):
        self.tries += 1
        if self.tries == 1:
            return False
        return self.sem.acquire(blocking=blocking)

    def release(self):
        self.sem.release()


def test_an_item_queued_as_the_logger_finishes_is_not_stranded(env):
    slot = _SlotFreedAfterFirstTry()
    with patch.object(risk_manager, "_ATR_FALLBACK_SEM", slot):
        _stop([], ticker="ORCL")
        assert slot.tries >= 2
        assert risk_manager._ATR_FALLBACK_PENDING == []
        rows = env.fallback_rows()
        assert [r[1] for r in rows] == ["ORCL"] and rows[0][4]["queued_behind_logger"] is True
        assert env.spawned == ["atr-fallback-log-ORCL"]
        assert slot.sem.acquire(blocking=False)      # the slot was released
        slot.sem.release()


def test_drain_spawn_failure_releases_the_slot_and_keeps_the_item(env):
    slot = _SlotFreedAfterFirstTry()
    with patch.object(risk_manager, "_ATR_FALLBACK_SEM", slot), \
         patch.object(risk_manager, "_spawn", side_effect=RuntimeError("no threads")):
        _stop([], ticker="ORCL")
    assert len(risk_manager._ATR_FALLBACK_PENDING) == 1
    assert slot.sem.acquire(blocking=False)
    slot.sem.release()


def test_a_failed_write_keeps_the_dropped_count(env):
    risk_manager._ATR_FALLBACK_DROPPED["n"] = 3
    env.write_ok = False
    _stop([], ticker="ZS")
    assert risk_manager._ATR_FALLBACK_DROPPED["n"] == 3
    env.write_ok = True
    _stop([], ticker="ORCL")
    assert env.fallback_rows()[-1][4]["dropped_before"] == 3
    assert risk_manager._ATR_FALLBACK_DROPPED["n"] == 0


def test_slot_released_when_spawn_fails(env):
    with patch.object(risk_manager, "_spawn", side_effect=RuntimeError("no threads")):
        _stop([], ticker="ZS")
    _stop([], ticker="ZS")
    assert [r[1] for r in env.fallback_rows()] == ["ZS"]


def test_logger_never_touches_the_shared_pool(env):
    _stop([], ticker="ZS", log_caller="post_fill")
    _stop([], ticker="ZS", log_caller="post_fill")
    env.write_ok = False
    _stop([], ticker="ORCL")
    assert all(op != "pool" for op, _ in env.calls)
    assert [op for op, _ in env.calls].count("log_activity") == 2


def test_caller_is_recorded_and_deduped_per_caller(env):
    rm = _rm()
    with patch("ibkr.get_historical_bars", return_value=[]):
        rm.calculate_stop("ZS", 100.0, "Accelerator", log_caller="size_position")
        rm.calculate_stop("ZS", 100.0, "Accelerator", log_caller="size_position")
        rm.calculate_stop("ZS", 100.0, "Accelerator", log_caller="post_fill")
        rm.calculate_stop("ZS", 100.0, "Accelerator", log_caller="post_fill")
        rm.calculate_stop("ZS", 100.0, "Accelerator", log_caller="manual")
        rm.calculate_stop("ZS", 100.0, "Accelerator")
        rm.calculate_stop("ZS", 100.0, "Accelerator", log_caller="bogus")
    assert [r[4]["caller"] for r in env.fallback_rows()] == [
        "size_position", "post_fill", "manual", "other"]
    assert risk_manager.atr_fallback_tickers_today() == ["ZS"]


def test_size_position_post_fill_and_main_pass_their_caller():
    import inspect
    import execution
    import main
    assert 'log_caller="size_position"' in inspect.getsource(risk_manager.RiskManager.size_position)
    assert 'log_caller="post_fill"' in inspect.getsource(execution)
    assert 'log_caller="manual"' in inspect.getsource(main)


def test_log_activity_direct_spills_without_alert(tmp_path):
    import json
    import db
    spill = tmp_path / "spill.jsonl"
    with patch.object(db, "_SPILL_PATH", spill), \
         patch("psycopg2.connect", side_effect=RuntimeError("db down")), \
         patch.object(db, "_get_conn", side_effect=AssertionError("pool")), \
         patch("alerts._send") as send:
        assert db.log_activity_direct("stop_atr_fallback", "ZS", "m", "neutral", {"a": 1}) is False
    line = json.loads(spill.read_text().strip())
    assert line["writer"] == "log_activity" and line["kind"] == "activity"
    assert line["record"]["stage"] == "stop_atr_fallback"
    send.assert_not_called()
