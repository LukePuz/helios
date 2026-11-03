"""h3d migration runner: own schema, qualified names, idempotent, dry-run
reads only. No real database: a fake connection records SQL."""
import contextlib
import re

import pytest

from helios3d.schema import migrate


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        s = sql.strip()
        if s.startswith("SELECT to_regclass"):
            self._result = [("h3d.schema_version" if self.conn.has_schema else None,)]
        elif s.startswith("SELECT version FROM h3d.schema_version"):
            self._result = [(v,) for v in sorted(self.conn.versions)]
        elif s.startswith("CREATE SCHEMA IF NOT EXISTS h3d"):
            self.conn.has_schema = True
        elif s.startswith("INSERT INTO h3d.schema_version"):
            self.conn.versions.add(params[0])

    def fetchone(self):
        return self._result[0]

    def fetchall(self):
        return list(self._result)


class FakeConn:
    def __init__(self, has_schema=False, versions=()):
        self.has_schema = has_schema
        self.versions = set(versions)
        self.executed = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


_OBJ = r"([\w\.\"]+)"
_DDL_RE = re.compile(r"\b(?:CREATE|ALTER|DROP)\s+(?:UNIQUE\s+)?(?:TABLE|INDEX|SCHEMA|SEQUENCE|VIEW)"
                     r"(?:\s+IF\s+(?:NOT\s+)?EXISTS)?\s+" + _OBJ, re.I)
_DML_RE = re.compile(r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|TRUNCATE(?:\s+TABLE)?)\s+" + _OBJ, re.I)
_ON_RE = re.compile(r"\bINDEX\b[^;]*?\bON\s+" + _OBJ, re.I)


def _tables_touched(sql):
    """Objects a statement creates or writes, except index names (an index
    lives in its table's schema; its table is checked through ON)."""
    out = [m.group(1) for m in _DML_RE.finditer(sql)]
    for m in _DDL_RE.finditer(sql):
        if not re.match(r"\w+\s+(?:UNIQUE\s+)?INDEX", m.group(0), re.I):
            out.append(m.group(1))
    out += [m.group(1) for m in _ON_RE.finditer(sql)]
    return out


def test_versions_unique_ascending_and_first_is_verdict_labels():
    versions = [v for v, _, _ in migrate.MIGRATIONS]
    assert versions == sorted(set(versions))
    assert migrate.MIGRATIONS[0][:2] == (1, "verdict_labels")


@pytest.mark.parametrize("version,name,sql", migrate.MIGRATIONS)
def test_every_migration_only_touches_h3d(version, name, sql):
    touched = _tables_touched(sql)
    assert touched
    bad = [t for t in touched if not t.lower().startswith("h3d.")]
    assert not bad, f"migration {version} touches {bad}"
    assert "public." not in sql.lower()
    for stmt in filter(str.strip, sql.split(";")):
        assert "IF NOT EXISTS" in stmt.upper(), stmt


def test_bootstrap_only_touches_h3d():
    for stmt in migrate.BOOTSTRAP_SQL:
        assert stmt.startswith("CREATE SCHEMA IF NOT EXISTS h3d") or \
            _tables_touched(stmt) == ["h3d.schema_version"], stmt


def test_verdict_labels_columns_and_unique_key():
    sql = migrate.MIGRATIONS[0][2]
    for col in ("signal_id", "ticker", "verdict_at", "action", "conviction", "horizon_days",
                "entry_ref_price", "fwd_price", "fwd_return", "spy_fwd_return", "excess_return",
                "label_source", "computed_at"):
        assert re.search(rf"^\s*{col}\s", sql, re.M), col
    assert re.search(r"UNIQUE\s*\(\s*signal_id\s*,\s*horizon_days\s*\)", sql)


def test_label_provenance_migration_keeps_dates_shas_skips_and_history():
    v, name, sql = migrate.MIGRATIONS[1]
    assert (v, name) == (2, "label_provenance")
    for col in ("entry_date", "fwd_date", "entry_close_at", "code_sha", "price_snapshot_sha"):
        assert f"ADD COLUMN IF NOT EXISTS {col} " in sql, col
    assert "CREATE TABLE IF NOT EXISTS h3d.label_skips" in sql
    assert "CREATE TABLE IF NOT EXISTS h3d.verdict_label_history" in sql
    assert re.search(r"label_skips_signal_horizon_uq UNIQUE\s*\(\s*signal_id\s*,\s*horizon_days\s*\)", sql)


def test_label_columns_exist_in_the_tables():
    """Every column labels.py writes exists in the table it writes to."""
    from helios3d.record import labels as L
    labels_sql = migrate.MIGRATIONS[0][2] + migrate.MIGRATIONS[1][2]
    for col in L._COLS:
        assert re.search(rf"(?:^\s*|ADD COLUMN IF NOT EXISTS ){col}\s", labels_sql, re.M), col
    hist = migrate.MIGRATIONS[1][2].split("h3d.verdict_label_history (", 1)[1]
    skips = migrate.MIGRATIONS[1][2].split("h3d.label_skips (", 1)[1]
    for col in L._COLS + ("computed_at",):
        assert re.search(rf"^\s*{col}\s", hist, re.M), col
    for col in L._SKIP_COLS:
        assert re.search(rf"^\s*{col}\s", skips, re.M), col


def test_apply_creates_schema_then_migrates_with_search_path():
    conn = FakeConn()
    assert migrate.apply(conn) == [v for v, _, _ in migrate.MIGRATIONS]
    sqls = [s for s, _ in conn.executed]
    assert sqls[0].startswith("CREATE SCHEMA IF NOT EXISTS h3d")
    i = next(k for k, s in enumerate(sqls) if "CREATE TABLE IF NOT EXISTS h3d.verdict_labels" in s)
    assert sqls[i - 1] == "SET LOCAL search_path TO h3d"
    assert conn.versions == {v for v, _, _ in migrate.MIGRATIONS}


def test_apply_is_idempotent():
    conn = FakeConn()
    migrate.apply(conn)
    n = len(conn.executed)
    assert migrate.apply(conn) == []
    rerun = [s for s, _ in conn.executed[n:]]
    assert not any("h3d.verdict_labels" in s for s in rerun)


def test_dry_run_pending_reads_only():
    conn = FakeConn()
    todo = migrate.pending(conn)
    assert [v for v, _, _ in todo] == [v for v, _, _ in migrate.MIGRATIONS]
    assert conn.commits == 0
    assert all(s.lstrip().upper().startswith("SELECT") for s, _ in conn.executed)


def test_cli_defaults_to_dry_run(monkeypatch, capsys):
    conn = FakeConn(has_schema=True, versions={v for v, _, _ in migrate.MIGRATIONS})

    @contextlib.contextmanager
    def fake_connection():
        yield conn

    monkeypatch.setattr("helios3d.schema.conn.connection", fake_connection)
    assert migrate.main([]) == 0
    assert conn.commits == 0
    assert "pending: none" in capsys.readouterr().out


def test_real_connection_is_blocked_in_tests():
    """The conftest DB guard covers helios3d too (it borrows db.py's pool)."""
    from helios3d.schema.conn import connection
    with pytest.raises(RuntimeError, match="must not touch the real database"):
        with connection():
            pass



def test_a_dedicated_role_dsn_is_used_when_set(monkeypatch):
    """HELIOS3D_DATABASE_URL gives helios3d its own role; the pool is
    not touched, and the connection is closed afterwards."""
    from unittest.mock import MagicMock, patch
    from helios3d.schema import conn as h3conn
    fake = MagicMock()
    monkeypatch.setenv("HELIOS3D_DATABASE_URL", "postgresql://h3d@localhost/x")
    with patch("db.psycopg2.connect", return_value=fake) as connect, \
         patch("db._get_conn") as pool:
        with h3conn.connection() as c:
            assert c is fake
    assert connect.call_args.args[0] == "postgresql://h3d@localhost/x"
    pool.assert_not_called()
    fake.rollback.assert_called_once(); fake.close.assert_called_once()
