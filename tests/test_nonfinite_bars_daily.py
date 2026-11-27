"""scripts/nonfinite_bars_daily.py: batch_verify's nonfinite_bars check as one
daily line, by import. Always exits 0. A fake db and connection; no DB."""
import importlib.util
import inspect
import os
import sys
import types
from datetime import datetime

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nd = _load("nonfinite_bars_daily_t", "nonfinite_bars_daily.py")
bv = _load("batch_verify_nonfinite_daily_t", "batch_verify.py")

NY = nd.NY
NOW = datetime(2026, 11, 27, 18, 5, tzinfo=NY)
NAN4 = ("NaN", "NaN", "NaN", "NaN")
ROWS = [("SNDK", "2026-11-19", *NAN4), ("SNDK", "2026-11-20", *NAN4),
        ("X", "2026-11-02", None, None, None, "Infinity")]


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append(sql)
        if self.conn.raise_on and self.conn.raise_on in sql:
            raise RuntimeError("canceling statement due to statement timeout")

    def fetchall(self):
        return list(self.conn.rows)


class _Conn:
    def __init__(self, rows=(), raise_on=None):
        self.rows, self.raise_on, self.log = rows, raise_on, []

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.log.append("ROLLBACK")


@pytest.fixture
def fake_db(monkeypatch):
    """A fake db in sys.modules (batch_verify's lazy `import db` reads it) and
    _import_db stubbed to it; returns a setter for the connection."""
    state = {"put": [], "imports": 0}

    def use(conn):
        mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=state["put"].append)
        monkeypatch.setitem(sys.modules, "db", mod)

        def imp():
            state["imports"] += 1
            return mod
        monkeypatch.setattr(nd, "_import_db", imp)
        return state
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    return use


def _run(capsys, argv=(), now=NOW):
    rc = nd.main(list(argv), now=now)
    out = capsys.readouterr().out
    return rc, out


# --- the line ----------------------------------------------------------------

def test_no_bad_bar_is_one_ok_line(fake_db, capsys):
    fake_db(_Conn(rows=[]))
    rc, out = _run(capsys)
    assert rc == 0
    assert out == ("nonfinite_bars 2026-11-27 18:05 ET OK: no non-finite OHLC bar in any "
                   "ticker's last 260 rows\n")


def test_bad_bars_name_count_tickers_and_dates(fake_db, capsys):
    fake_db(_Conn(rows=ROWS))
    rc, out = _run(capsys)
    assert rc == 0 and out.count("\n") == 1
    assert out == ("nonfinite_bars 2026-11-27 18:05 ET WARN: 3 non-finite OHLC bar(s) in 2 "
                   "ticker(s) within each ticker's last 260 rows: SNDK 2 (2026-11-19, "
                   "2026-11-20), X 1 (2026-11-02)\n")


def test_the_text_is_batch_verifys_own(fake_db, capsys):
    """The same rows through batch_verify's fetch and check give the same text."""
    fake_db(_Conn(rows=ROWS))
    _, out = _run(capsys)
    fake_db(_Conn(rows=ROWS))
    chk = bv.check_nonfinite_bars(bv.fetch_nonfinite_bars())
    assert out.rstrip("\n").endswith(f"ET {chk['status']}: {chk['why']}")


def test_no_value_is_printed(fake_db, capsys):
    fake_db(_Conn(rows=ROWS))
    _, out = _run(capsys)
    assert "NaN" not in out and "Infinity" not in out


def test_the_read_is_batch_verifys_read_only_rolled_back_read(fake_db, capsys):
    conn = _Conn(rows=ROWS)
    st = fake_db(conn)
    _run(capsys)
    assert conn.log == ["SET TRANSACTION READ ONLY", "SET LOCAL statement_timeout = 30000",
                        bv.nonfinite_bars_sql(260), "ROLLBACK"]
    assert st["put"] == [conn] and st["imports"] == 1


def test_a_failed_read_is_batch_verifys_warn_and_exit_0(fake_db, capsys):
    conn = _Conn(raise_on="WITH r AS")
    st = fake_db(conn)
    rc, out = _run(capsys)
    assert rc == 0
    assert out == "nonfinite_bars 2026-11-27 18:05 ET WARN: price_history read failed (RuntimeError)\n"
    assert conn.log[-1] == "ROLLBACK" and st["put"] == [conn]


# --- guards: each a SKIP line, exit 0 ----------------------------------------

@pytest.mark.parametrize("port", [None, "5432", ""])
def test_the_port_guard_skips_without_importing_db(fake_db, monkeypatch, capsys, port):
    conn = _Conn(rows=ROWS)
    st = fake_db(conn)
    if port is None:
        monkeypatch.delenv("SUPABASE_PORT", raising=False)
    else:
        monkeypatch.setenv("SUPABASE_PORT", port)
    rc, out = _run(capsys)
    assert rc == 0 and st["imports"] == 0 and conn.log == []
    assert out == ("nonfinite_bars 2026-11-27 18:05 ET SKIP: SUPABASE_PORT must be 6543 "
                   "(the transaction pooler); not reading\n")


@pytest.mark.parametrize("hm, reads", [((4, 49), True), ((4, 50), False), ((5, 0), False),
                                        ((5, 11), False), ((5, 12), True)])
def test_the_quiet_window(fake_db, capsys, hm, reads):
    conn = _Conn(rows=[])
    st = fake_db(conn)
    rc, out = _run(capsys, now=datetime(2026, 11, 27, *hm, tzinfo=NY))
    assert rc == 0 and (st["imports"] == 1) is reads and bool(conn.log) is reads
    assert ("SKIP: 04:50-05:12 ET is the restart window; not reading" in out) is not reads


def test_the_quiet_window_reads_et_from_a_utc_now(fake_db, capsys):
    st = fake_db(_Conn(rows=[]))
    _, out = _run(capsys, now=datetime.fromisoformat("2026-11-27T09:55:00+00:00"))
    assert st["imports"] == 0 and out.startswith("nonfinite_bars 2026-11-27 04:55 ET SKIP:")


def test_a_db_import_failure_is_a_skip(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "6543")

    def boom():
        raise ImportError("no module named db")
    monkeypatch.setattr(nd, "_import_db", boom)
    rc, out = _run(capsys)
    assert rc == 0 and out == "nonfinite_bars 2026-11-27 18:05 ET SKIP: db import failed (ImportError)\n"


def test_a_batch_verify_failure_is_a_skip(fake_db, monkeypatch, capsys):
    fake_db(_Conn(rows=[]))

    def boom():
        raise SyntaxError("bad")
    monkeypatch.setattr(nd, "_load_batch_verify", boom)
    rc, out = _run(capsys)
    assert rc == 0 and out.endswith("SKIP: batch_verify check failed (SyntaxError)\n")


# --- --out -------------------------------------------------------------------

def test_out_appends_one_line_per_run(fake_db, capsys, tmp_path):
    p = tmp_path / "nonfinite.log"
    p.write_text("earlier\n")
    fake_db(_Conn(rows=[]))
    _run(capsys, ["--out", str(p)])
    fake_db(_Conn(rows=ROWS))
    rc, out = _run(capsys, ["--out", str(p)])
    lines = p.read_text().splitlines()
    assert rc == 0 and len(lines) == 3 and lines[0] == "earlier"
    assert " OK: " in lines[1] and lines[2] + "\n" == out


def test_an_unwritable_out_still_prints_and_exits_0(fake_db, capsys, tmp_path):
    fake_db(_Conn(rows=[]))
    rc = nd.main(["--out", str(tmp_path / "no" / "such" / "dir.log")], now=NOW)
    cap = capsys.readouterr()
    assert rc == 0 and " OK: " in cap.out and "could not append" in cap.err


# --- one rule, no copy -------------------------------------------------------

def test_no_check_is_defined_here():
    src = inspect.getsource(nd)
    for n in ("SELECT", "price_history WHERE", "isfinite", "is_finite", "'NaN'",
              "Infinity'", "260", "row_number"):
        assert n not in src.split('"""', 2)[2], n   # outside the module docstring
    assert "bv.check_nonfinite_bars(bv.fetch_nonfinite_bars(bv.NONFINITE_BAR_ROWS)" in src


def test_batch_verify_is_loaded_from_this_scripts_directory():
    assert nd.BATCH_VERIFY == os.path.join(_SCRIPTS, "batch_verify.py")


def test_loading_batch_verify_restores_sys_path():
    before = list(sys.path)
    mod = nd._load_batch_verify()
    assert sys.path == before
    assert mod.NONFINITE_BAR_ROWS == 260 and mod.NONFINITE_BAR_COLUMNS == ("open", "high", "low", "close")


def test_import_db_takes_the_working_directorys_db(monkeypatch, tmp_path):
    (tmp_path / "db.py").write_text("MARK = 'cwd'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "db", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    db = nd._import_db()
    assert db.MARK == "cwd" and sys.path[0] == str(tmp_path)
    monkeypatch.delitem(sys.modules, "db", raising=False)


def test_import_db_refuses_a_db_from_elsewhere(monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    fake = types.ModuleType("db")
    fake.__file__ = str(other / "db.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "db", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(RuntimeError, match="not the working directory"):
        nd._import_db()
