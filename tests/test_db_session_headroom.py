"""DB session headroom (roadmap 04e7): a process builds one pool even when
two threads ask at once (the lock is fork-safe), the pool's log line gives
its port and maxconn, and the screen and ingestion workers use the
transaction pooler (6543) while the others stay on the session pooler. No
DB: the pool class is a fake."""
import ast
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import db
import process_manager

_REAL_GET_POOL = db._get_pool     # conftest blocks it per test; the real one is called here
ROOT = Path(__file__).resolve().parent.parent


class _FakePool:
    made = 0

    def __init__(self, **kw):
        type(self).made += 1
        self.kw = kw
        time.sleep(0.05)          # a connect takes a while: other threads arrive meanwhile


@pytest.fixture
def fake_pool(monkeypatch):
    _FakePool.made = 0
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db.psycopg2.pool, "ThreadedConnectionPool", _FakePool)
    return _FakePool


# ── (a) one pool per process ────────────────────────────────────────────────

def test_threads_that_ask_at_once_get_one_pool(fake_pool):
    got, start = [], threading.Barrier(8)

    def ask():
        start.wait()
        got.append(_REAL_GET_POOL())
    ts = [threading.Thread(target=ask) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert fake_pool.made == 1
    assert len(got) == 8 and all(p is got[0] for p in got) and db._pool is got[0]
    assert _REAL_GET_POOL() is got[0] and fake_pool.made == 1      # later calls: no lock, no build


def test_a_failed_build_leaves_no_pool_and_the_next_call_retries(monkeypatch, fake_pool):
    calls = []

    def flaky(**kw):
        calls.append(1)
        if len(calls) <= 3:
            raise OSError("pooler down")
        return _FakePool(**kw)
    monkeypatch.setattr(db.psycopg2.pool, "ThreadedConnectionPool", flaky)
    monkeypatch.setattr(db._time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        _REAL_GET_POOL()
    assert db._pool is None and not db._POOL_CREATE_LOCK.locked()
    assert isinstance(_REAL_GET_POOL(), _FakePool) and len(calls) == 4


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")
def test_a_forked_child_does_not_inherit_a_held_lock():
    """Celery forks its pool children; one must not start with the creation
    lock another thread of the parent held at the fork."""
    db._POOL_CREATE_LOCK.acquire()
    try:
        pid = os.fork()
        if pid == 0:                                   # the child
            ok = db._POOL_CREATE_LOCK.acquire(timeout=2)
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
    finally:
        db._POOL_CREATE_LOCK.release()
    assert os.waitstatus_to_exitcode(status) == 0


def test_the_pool_log_line_gives_port_and_maxconn_but_no_host_or_credentials(
        monkeypatch, fake_pool, caplog):
    monkeypatch.setenv("SUPABASE_HOST", "pooler.example.invalid")
    monkeypatch.setenv("SUPABASE_USER", "someone.abcdef")
    monkeypatch.setenv("SUPABASE_PASSWORD", "not-a-real-secret")
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("DB_MAXCONN", "2")
    with caplog.at_level(logging.INFO):
        pool = _REAL_GET_POOL()
    assert pool.kw["port"] == 6543 and pool.kw["maxconn"] == 2 and pool.kw["minconn"] == 1
    lines = [r.getMessage() for r in caplog.records if "DB connection pool created" in r.getMessage()]
    assert lines == [f"DB connection pool created (port=6543, maxconn=2, pid={os.getpid()})."]
    assert not any(s in lines[0] for s in ("example.invalid", "someone", "not-a-real-secret"))


# ── (b) the screen and ingestion workers on the transaction pooler ──────────

def _started_env(monkeypatch, tmp_path, name):
    seen = {}

    def popen(cmd, **kw):
        seen.update(kw["env"])
        return MagicMock(pid=12345)
    monkeypatch.setitem(process_manager.SERVICES[name], "log", str(tmp_path / f"{name}.log"))
    monkeypatch.setattr(process_manager, "_load", lambda: {})
    monkeypatch.setattr(process_manager, "_save", lambda pids: None)
    monkeypatch.setattr(process_manager.subprocess, "Popen", popen)
    assert process_manager.start(name) == "ok"
    return seen


@pytest.mark.parametrize("name", sorted(process_manager.SERVICES))
def test_only_the_screen_and_ingestion_workers_get_the_transaction_pooler(monkeypatch, tmp_path, name):
    monkeypatch.delenv("SUPABASE_PORT", raising=False)
    env = _started_env(monkeypatch, tmp_path, name)
    if name in ("celery_screen_worker", "celery_ingestion_worker"):
        assert env["SUPABASE_PORT"] == "6543" and env["DB_MAXCONN"] == "2"
    else:
        assert "SUPABASE_PORT" not in env           # .env's value (the session pooler) applies


def test_the_worker_port_wins_over_dotenv(monkeypatch, tmp_path):
    """db.py's load_dotenv doesn't override, so the worker's 6543 stands even
    when the parent's environment (or .env) says 5432."""
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert _started_env(monkeypatch, tmp_path, "celery_screen_worker")["SUPABASE_PORT"] == "6543"
    assert _started_env(monkeypatch, tmp_path, "celery_worker")["SUPABASE_PORT"] == "5432"
    calls = [(rel, c) for rel, tree in _modules()
             for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", getattr(c.func, "attr", None)) == "load_dotenv"]
    assert any(rel == "db.py" for rel, _ in calls)
    assert not [rel for rel, c in calls if any(k.arg == "override" for k in c.keywords)]


def test_the_port_constant_matches_db():
    assert process_manager._TRANSACTION_POOLER_PORT == str(db._TRANSACTION_POOLER_PORT)


def test_the_concurrent_index_build_still_refuses_on_the_workers_port(monkeypatch):
    monkeypatch.setenv("SUPABASE_PORT", process_manager._TRANSACTION_POOLER_PORT)
    conn = MagicMock()
    version, sql = next((v, s) for v, s in db._MIGRATIONS if db._CONCURRENT_INDEX_RE.match(s))
    assert db._apply_concurrent_index(conn, version, sql) is False
    conn.commit.assert_not_called()
    conn.cursor.assert_not_called()
    assert conn.autocommit is not True


def _modules():
    """(path relative to the repo, AST) for every non-test module."""
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "venv/", ".claude/")) or "/site-packages/" in rel:
            continue
        try:
            yield rel, ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue


_SESSION_ONLY = re.compile(
    r"pg_advisory_lock\(|pg_try_advisory_lock\(|pg_advisory_unlock\(|\bLISTEN\b|\bNOTIFY\b|"
    r"pg_notify|CREATE TEMP|TEMPORARY TABLE|WITH HOLD|\bPREPARE\b|\bDEALLOCATE\b|set_config\(|"
    r"pg_backend_pid|^\s*(SET|RESET)\s+(?!LOCAL\b|TRANSACTION\b)\w", re.I)


def test_session_only_sql_and_autocommit_live_only_in_the_migration_runners():
    """Everything the transaction pooler breaks (a session lock or SET,
    LISTEN/NOTIFY, temp tables, held cursors, prepared statements, backend
    identity, autocommit on) appears in code only inside
    db._apply_concurrent_index and db.apply_one_migration, outside scripts/.
    Docstrings and comments don't count."""
    allowed = {("db.py", "_apply_concurrent_index"), ("db.py", "apply_one_migration")}
    found = set()
    for rel, tree in _modules():
        if rel.startswith("scripts/"):
            continue
        docs, owner = set(), {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docs.add(id(body[0].value))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(node):
                    owner[id(n)] = node.name       # innermost wins: walk visits outer first
        for n in ast.walk(tree):
            hit = False
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                hit = bool(_SESSION_ONLY.search(n.value))
            elif isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "cursor":
                hit = any(k.arg in ("name", "withhold") for k in n.keywords) or bool(n.args)
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is True:
                hit = any(isinstance(t, ast.Attribute) and t.attr == "autocommit" for t in n.targets)
            if hit:
                found.add((rel, owner.get(id(n), "<module>")))
    assert found <= allowed, sorted(found - allowed)
    assert ("db.py", "_apply_concurrent_index") in found       # the scan does see the real ones


def test_no_worker_module_reaches_the_session_only_code():
    """The session advisory lock, session SET and autocommit live only in
    db._apply_concurrent_index, which init_db and the concurrent-index script
    call. Outside db.py, only main.py and scripts/ call either."""
    names = {"init_db", "apply_concurrent_index_migrations", "_apply_concurrent_index"}
    callers = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "venv/", ".claude/")) or "/site-packages/" in rel:
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                owner = getattr(f.value, "id", None) if isinstance(f, ast.Attribute) else None
                if nm in names and not (rel.startswith("catalyst_service/") and owner != "db"):
                    callers.add(rel)
    assert callers <= {"db.py", "main.py", "scripts/setup_db.py", "scripts/apply_concurrent_indexes.py"}, callers
    assert "tasks.py" not in callers and not any(c.startswith(("ingestion/", "agents/")) for c in callers)
