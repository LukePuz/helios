"""Migrations 90-91: signals indexes built CONCURRENTLY, outside a transaction,
by init_db's runner (db._apply_concurrent_index) or scripts/apply_concurrent_indexes.py."""
from unittest.mock import patch

import pytest

import db


def test_the_migration_sql_is_exactly_the_two_indexes():
    m = dict(db._MIGRATIONS)
    assert m[90] == ("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_ts "
                     "ON signals (symbol, timestamp DESC);")
    assert m[91] == "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_ts ON signals (timestamp DESC);"
    versions = [v for v, _ in db._MIGRATIONS]
    assert len(set(versions)) == len(versions) and versions == sorted(versions)
    concurrent = [v for v, s in db._MIGRATIONS if db._CONCURRENT_INDEX_RE.match(s)]
    assert concurrent == [90, 91]
    assert db._CONCURRENT_INDEX_RE.match(m[90]).group(1) == "idx_signals_symbol_ts"
    assert db._CONCURRENT_INDEX_RE.match(m[91]).group(1) == "idx_signals_ts"
    for v in (90, 91):                                        # additive: one statement, no drop
        assert m[v].count(";") == 1 and "DROP" not in m[v].upper()


def test_every_concurrently_migration_takes_the_concurrent_runner():
    """A CONCURRENTLY migration the regex missed would run inside a transaction and fail."""
    for v, sql in db._MIGRATIONS:
        if "CONCURRENTLY" in sql.upper():
            assert db._CONCURRENT_INDEX_RE.match(sql), v


class _Conn:
    """A connection that records each statement with the autocommit state it
    ran in, and models one index's pg_index state."""

    def __init__(self, applied=(), state=None, after_create=True, lock_free=True, fail_on=None):
        self._autocommit = False
        self.closed = False
        self.log = []
        self.applied = list(applied)
        self.state = {}                       # index name -> True/False (indisvalid)
        self.initial = state                  # applied to every index on first lookup
        self.after_create = after_create      # validity a CREATE leaves behind
        self.lock_free = lock_free
        self.locks = 0
        self.fail_on = fail_on
        self.autocommit_error = None
        conn = self

        class Cur:
            def __init__(self):
                self._rows = []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, args=None):
                conn.log.append((sql, conn._autocommit))
                if conn.fail_on and conn.fail_on in sql:
                    raise RuntimeError("canceling statement due to lock timeout")
                self._rows = []
                if sql.startswith("SELECT version FROM schema_version"):
                    self._rows = [(v,) for v in conn.applied]
                elif sql.startswith("SELECT to_regclass('public.schema_version')"):
                    self._rows = [(True,)]
                elif "indisvalid" in sql:
                    name = args[0]
                    if name not in conn.state and conn.initial is not None:
                        conn.state[name] = conn.initial
                    self._rows = [] if name not in conn.state else [(conn.state[name],)]
                elif sql.startswith("SELECT pg_try_advisory_lock"):
                    self._rows = [(conn.lock_free,)]
                    conn.locks += conn.lock_free
                elif sql.startswith("SELECT pg_advisory_unlock"):
                    conn.locks -= 1
                elif sql.startswith("DROP INDEX CONCURRENTLY"):
                    conn.state.pop(sql.split()[-1], None)
                elif sql.startswith("CREATE INDEX CONCURRENTLY"):
                    conn.state[db._CONCURRENT_INDEX_RE.match(sql).group(1)] = conn.after_create
                elif sql.startswith("INSERT INTO schema_version"):
                    conn.applied.append(args[0])

            def fetchall(self):
                return self._rows

            def fetchone(self):
                return self._rows[0] if self._rows else None

        self._cur = Cur

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, v):
        if self.autocommit_error and not v:
            raise self.autocommit_error
        self._autocommit = v

    def cursor(self):
        if self.closed:
            raise db.psycopg2.InterfaceError("connection already closed")
        return self._cur()

    def commit(self):
        if self.closed:
            raise db.psycopg2.InterfaceError("connection already closed")

    def rollback(self):
        pass


def _plain_applied():
    return [v for v, s in db._MIGRATIONS if not db._CONCURRENT_INDEX_RE.match(s)]


@pytest.fixture(autouse=True)
def _session_port(monkeypatch):
    monkeypatch.setenv("SUPABASE_PORT", "5432")


def test_init_db_builds_them_outside_a_transaction_under_the_lock():
    conn = _Conn(applied=_plain_applied())
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.init_db()
    m = dict(db._MIGRATIONS)
    for v in (90, 91):
        assert [(s, ac) for s, ac in conn.log if s == m[v]] == [(m[v], True)]   # autocommit
    stmts = [s for s, _ in conn.log]
    assert not any(s.startswith("SET LOCAL") for s in stmts)
    assert sum(s.startswith("SET lock_timeout") and "30s" in s for s in stmts) == 2
    assert stmts.count("RESET lock_timeout") == 2
    assert sum(s.startswith("SELECT pg_try_advisory_lock") for s in stmts) == 2 and conn.locks == 0
    # validity read again after each CREATE, before the version row
    i = stmts.index(m[90])
    assert "indisvalid" in stmts[i + 1] and stmts[i + 2].startswith("INSERT INTO schema_version")
    assert "to_regclass('public.' ||" in stmts[i + 1]                         # tied to the table
    assert 90 in conn.applied and 91 in conn.applied and conn.autocommit is False


def test_an_invalid_leftover_is_dropped_then_rebuilt():
    conn = _Conn(state=False)
    assert db._apply_concurrent_index(conn, 90, dict(db._MIGRATIONS)[90]) is True
    stmts = [s for s, _ in conn.log]
    assert stmts.index("DROP INDEX CONCURRENTLY IF EXISTS idx_signals_symbol_ts") < \
        stmts.index(dict(db._MIGRATIONS)[90])
    assert conn.applied == [90]


def test_an_index_still_invalid_after_the_build_is_not_recorded():
    conn = _Conn(after_create=False)
    assert db._apply_concurrent_index(conn, 90, dict(db._MIGRATIONS)[90]) is False
    assert conn.applied == [] and "idx_signals_symbol_ts" not in conn.state   # dropped
    assert conn.locks == 0 and conn.autocommit is False


def test_a_held_lock_skips_the_build():
    conn = _Conn(lock_free=False)
    assert db._apply_concurrent_index(conn, 90, dict(db._MIGRATIONS)[90]) is False
    stmts = [s for s, _ in conn.log]
    assert not any(s.startswith(("CREATE", "DROP", "SET lock_timeout", "SELECT pg_advisory_unlock"))
                   for s in stmts)
    assert conn.autocommit is False


def test_a_failed_build_does_not_fail_the_boot_or_record_the_version():
    conn = _Conn(applied=_plain_applied(), state=False, fail_on="idx_signals_symbol_ts ON signals")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.init_db()                                          # no raise
    assert 90 not in conn.applied and 91 in conn.applied      # the next one still ran
    assert "DROP INDEX CONCURRENTLY IF EXISTS idx_signals_symbol_ts" in [s for s, _ in conn.log]
    assert conn.locks == 0 and conn.autocommit is False


def test_a_lost_connection_does_not_raise():
    conn = _Conn()
    conn.closed = True
    assert db._apply_concurrent_index(conn, 90, dict(db._MIGRATIONS)[90]) is False
    conn = _Conn(fail_on="pg_try_advisory_lock")
    conn.autocommit_error = db.psycopg2.InterfaceError("connection already closed")
    assert db._apply_concurrent_index(conn, 90, dict(db._MIGRATIONS)[90]) is False


def test_refused_on_the_transaction_pooler(monkeypatch):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    conn = _Conn()
    assert db._apply_concurrent_index(conn, 91, dict(db._MIGRATIONS)[91]) is False
    assert conn.log == [] and conn.autocommit is False


def test_script_entry_builds_only_the_concurrent_ones_and_the_dry_run_only_reads(monkeypatch):
    conn = _Conn(applied=[v for v in _plain_applied() if v < 80])   # older ALTERs pending too
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.apply_concurrent_index_migrations(dry_run=True) == [(90, "pending"), (91, "pending")]
        assert all(s.startswith("SELECT") for s, _ in conn.log)        # no CREATE TABLE either
        monkeypatch.setenv("SUPABASE_PORT", "6543")
        assert db.apply_concurrent_index_migrations(dry_run=True) == [
            (90, "pending (a real run is refused on port 6543)"),
            (91, "pending (a real run is refused on port 6543)")]
        monkeypatch.setenv("SUPABASE_PORT", "5432")
        assert db.apply_concurrent_index_migrations() == [(90, "applied"), (91, "applied")]
        assert db.apply_concurrent_index_migrations() == [(90, "already applied"),
                                                          (91, "already applied")]
    assert not any(s.startswith("ALTER") for s, _ in conn.log)         # no other migration ran


# ---- the jsonb readers: rows are picked by id first, the JSON read after ----

_REWRITTEN = ("get_latest_signal_for", "get_recent_verdicts", "get_recent_verdict_levels",
              "get_latest_verdicts", "recent_model_udrs")


@pytest.mark.parametrize("name", _REWRITTEN)
def test_readers_pick_ids_before_reading_jsonb(name):
    import inspect
    src = inspect.getsource(getattr(db, name))
    assert ") k JOIN signals s ON s.id = k.id" in src
    inner = src[src.index("SELECT id, ") if "SELECT id, " in src else src.index("SELECT DISTINCT ON (symbol) id,"):]
    inner = inner[:min(inner.index(e) for e in ("LIMIT", ") k JOIN") if e in inner)]   # the id pick
    assert "decision->" not in inner and "provenance->" not in inner          # no jsonb read before LIMIT
