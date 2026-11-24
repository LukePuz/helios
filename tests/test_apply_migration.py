"""db.apply_one_migration / scripts/apply_migration.py: one plain migration,
applied and recorded in schema_version the way init_db does, under a short
lock and statement timeout; the dry run only reads."""
import importlib.util
import os
from unittest.mock import patch

import pytest

import db

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "apply_migration.py")
_spec = importlib.util.spec_from_file_location("apply_migration", _PATH)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)


class _Conn:
    """Records (sql, args) per statement, commits and rollbacks; models
    schema_version rows, one column's existence and other sessions' locks."""

    def __init__(self, applied=(), column=False, locks=(), fail_on=None):
        self.applied = set(applied)
        self.column = column
        self.locks = list(locks)
        self.fail_on = fail_on
        self.log = []
        self.events = []
        self.pending_rows = set()
        conn = self

        class Cur:
            def __init__(self):
                self._rows = []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, args=None):
                conn.log.append((sql, args))
                conn.events.append("exec")
                if conn.fail_on and conn.fail_on in sql:
                    raise RuntimeError("canceling statement due to lock timeout")
                self._rows = []
                if sql.startswith("SELECT to_regclass('public.schema_version')"):
                    self._rows = [(True,)]
                elif sql.startswith("SELECT version FROM schema_version"):
                    self._rows = [(v,) for v in conn.applied]
                elif sql.startswith("SELECT 1 FROM schema_version"):
                    self._rows = [(1,)] if args[0] in conn.applied else []
                elif "information_schema.columns" in sql:
                    self._rows = [(1,)] if conn.column else []
                elif "pg_locks" in sql:
                    self._rows = list(conn.locks)
                elif sql.startswith("INSERT INTO schema_version"):
                    conn.pending_rows.add(args[0])

            def fetchone(self):
                return self._rows[0] if self._rows else None

            def fetchall(self):
                return list(self._rows)

        self._cur = Cur

    def cursor(self):
        return self._cur()

    def commit(self):
        self.events.append("commit")
        self.applied |= self.pending_rows
        self.pending_rows = set()

    def rollback(self):
        self.events.append("rollback")
        self.pending_rows = set()

    def sqls(self):
        return [s for s, _ in self.log]


def _run(conn, version=92, **kw):
    with patch.object(db, "_get_conn", return_value=conn), patch.object(db, "_put_conn") as put:
        out = db.apply_one_migration(version, **kw)
    put.assert_called_once_with(conn)
    return out


M92 = dict(db._MIGRATIONS)[92]


def test_migration_92_is_the_held_before_column():
    assert M92 == "ALTER TABLE orders ADD COLUMN IF NOT EXISTS held_before_qty DECIMAL;"
    assert db._ADD_COLUMN_RE.match(M92).groups() == ("orders", "held_before_qty")


def test_applies_the_sql_and_records_the_version_in_one_transaction():
    conn = _Conn()
    out = _run(conn)
    assert out["status"] == "applied" and 92 in conn.applied
    s = conn.sqls()
    i_lock = s.index("SET LOCAL lock_timeout = '5s'")
    i_stmt = s.index("SET LOCAL statement_timeout = '60s'")
    i_sql = s.index(M92)
    i_ins = s.index("INSERT INTO schema_version (version) VALUES (%s)")
    assert i_lock < i_sql and i_stmt < i_sql < i_ins
    # no commit between the SQL and its version row
    ev = conn.events
    sql_at = [k for k, e in enumerate(ev) if e == "exec"][i_sql]
    ins_at = [k for k, e in enumerate(ev) if e == "exec"][i_ins]
    assert "commit" not in ev[sql_at:ins_at] and ev[-1] == "commit"
    assert not any("ALTER" in q and q != M92 for q in s)        # only this migration


def test_already_recorded_runs_nothing():
    conn = _Conn(applied={92})
    out = _run(conn)
    assert out["status"] == "already applied"
    assert M92 not in conn.sqls() and not any(q.startswith("INSERT") for q in conn.sqls())


def test_a_lock_timeout_rolls_back_the_change_and_the_row():
    conn = _Conn(fail_on="ALTER TABLE orders")
    out = _run(conn)
    assert out["status"].startswith("failed: RuntimeError: canceling statement due to lock timeout")
    assert 92 not in conn.applied and conn.events[-1] == "rollback"
    assert not any(q.startswith("INSERT") for q in conn.sqls())


def test_custom_timeouts_and_bad_timeouts():
    conn = _Conn()
    _run(conn, lock_timeout="3s", statement_timeout="500ms")
    assert "SET LOCAL lock_timeout = '3s'" in conn.sqls()
    assert "SET LOCAL statement_timeout = '500ms'" in conn.sqls()
    for bad in ("5", "5s; DROP TABLE orders", "0s", "", "1h"):
        with pytest.raises(ValueError):
            db.apply_one_migration(92, lock_timeout=bad)


def test_dry_run_only_reads():
    conn = _Conn(column=False, locks=[("AccessShareLock", True), ("AccessExclusiveLock", False)])
    out = _run(conn, dry_run=True)
    assert out["status"] == "pending" and out["recorded"] is False and out["column_exists"] is False
    assert out["other_locks"] == ["AccessShareLock", "AccessExclusiveLock (waiting)"]
    s = conn.sqls()
    assert s[0] == "SET TRANSACTION READ ONLY"
    assert M92 not in s and not any(q.split()[0].upper() in ("INSERT", "CREATE", "ALTER", "UPDATE", "DELETE")
                                    for q in s)
    assert "commit" not in conn.events and conn.events[-1] == "rollback"


def test_dry_run_reports_a_recorded_version_and_an_existing_column():
    out = _run(_Conn(applied={92}, column=True), dry_run=True)
    assert out["status"] == "already applied" and out["recorded"] and out["column_exists"]


def test_refuses_concurrent_index_and_unknown_versions():
    with patch.object(db, "_get_conn") as get:
        out = db.apply_one_migration(90)
        assert out["status"].startswith("refused") and not get.called
        with pytest.raises(ValueError):
            db.apply_one_migration(99999)


def test_init_db_skips_a_version_the_script_recorded():
    """The row the script writes is the one init_db reads, so the next boot skips 92."""
    conn = _Conn(applied={v for v, _ in db._MIGRATIONS if v != 92})
    _run(conn)
    before = len(conn.log)
    with patch.object(db, "_get_conn", return_value=conn), patch.object(db, "_put_conn"):
        db.init_db()
    boot = [q for q, _ in conn.log[before:]]
    assert M92 not in boot and not any(q.startswith("INSERT INTO schema_version") for q in boot)


def test_script_exit_codes(capsys):
    with patch.object(db, "apply_one_migration",
                      return_value={"version": 92, "sql": M92, "status": "pending", "recorded": False,
                                    "column_exists": False, "other_locks": []}) as f:
        assert am.main(["--id", "92", "--dry-run"]) == 0
    f.assert_called_once_with(92, dry_run=True, lock_timeout="5s", statement_timeout="60s")
    out = capsys.readouterr().out
    assert "migration 92: pending" in out and "column_exists: False" in out
    with patch.object(db, "apply_one_migration",
                      return_value={"version": 92, "sql": M92, "status": "failed: X: y"}):
        assert am.main(["--id", "92"]) == 1
    with patch.object(db, "apply_one_migration", side_effect=ValueError("no migration 7")):
        assert am.main(["--id", "7"]) == 2
