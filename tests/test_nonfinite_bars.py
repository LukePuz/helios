"""scripts/batch_verify.py nonfinite_bars: price_history bars with a NaN or
infinite open/high/low/close within each ticker's last 260 rows. WARN-only,
read-only, and no finite price leaves the DB."""
import importlib.util
import inspect
import os
import sys
import types
from datetime import datetime

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "batch_verify.py")
_spec = importlib.util.spec_from_file_location("batch_verify_nonfinite", _PATH)
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)

NY = bv.NY
SINCE = datetime(2026, 11, 24, 5, 0, tzinfo=NY)
NOW = datetime(2026, 11, 24, 12, 10, tzinfo=NY)
ALL_NAN = {"open": "NaN", "high": "NaN", "low": "NaN", "close": "NaN"}


def _bar(ticker, day, values=None):
    return {"ticker": ticker, "date": day, "values": dict(ALL_NAN if values is None else values)}


# --- the check --------------------------------------------------------------

def test_no_bad_bar_is_ok():
    c = bv.check_nonfinite_bars([])
    assert c["status"] == "OK" and c["rows"] == 0
    assert "last 260 rows" in c["why"]


def test_bad_bars_warn_naming_each_ticker_with_count_and_dates():
    # The audit's case: SNDK's split-day bar and its 11-25 bar, beside an old one.
    c = bv.check_nonfinite_bars([_bar("AMAT", "2025-11-20"), _bar("SNDK", "2026-11-20"),
                                 _bar("SNDK", "2026-11-25")])
    assert c["status"] == "WARN" and c["rows"] == c["triggered"] == 3
    assert c["why"].startswith("3 non-finite OHLC bar(s) in 2 ticker(s) within each ticker's last 260 rows: ")
    assert "SNDK 2 (2026-11-20, 2026-11-25)" in c["why"]
    assert "AMAT 1 (2025-11-20)" in c["why"]
    # The newest bad bar leads, so a fresh defect is first on the line.
    assert c["why"].index("SNDK") < c["why"].index("AMAT")


def test_it_never_fails():
    many = [_bar(f"T{i}", "2026-11-25") for i in range(500)]
    assert bv.check_nonfinite_bars(many)["status"] == "WARN"
    assert bv.check_nonfinite_bars(RuntimeError("boom"))["status"] == "WARN"


def test_the_columns_and_values_are_tallied():
    c = bv.check_nonfinite_bars([_bar("SNDK", "2026-11-25"),
                                 _bar("X", "2026-11-02", {"close": "Infinity"}),
                                 _bar("Y", "2026-11-03", {"low": "-Infinity", "high": "NaN"})])
    assert dict(c["reads"]["columns"]) == {"open,high,low,close": 1, "close": 1, "high,low": 1}
    assert dict(c["reads"]["value"]) == {"NaN": 1, "Infinity": 1, "-Infinity/NaN": 1}


def test_a_failed_read_is_a_warn_naming_the_error_class():
    c = bv.check_nonfinite_bars(TimeoutError("statement timeout"))
    assert c["status"] == "WARN" and c["why"] == "price_history read failed (TimeoutError)"


def test_the_readout_renders_the_line():
    c = bv.check_nonfinite_bars([_bar("SNDK", "2026-11-25")])
    text = bv.readout([c], SINCE, "live", "abc", NOW)
    assert "| WARN | nonfinite_bars | 1 | 0 | 1 non-finite OHLC bar(s) in 1 ticker(s)" in text
    assert "## nonfinite_bars: WARN" in text


# --- the SQL ----------------------------------------------------------------

def test_the_sql_reads_each_tickers_last_260_rows_on_all_four_columns():
    sql = bv.nonfinite_bars_sql()
    assert "row_number() OVER (PARTITION BY ticker ORDER BY date DESC)" in sql
    assert "WHERE rn <= 260 AND (" in sql
    for col in ("open", "high", "low", "close"):
        assert bv._nonfinite_sql(col) in sql
    assert "rn <= 30 AND" in bv.nonfinite_bars_sql(30)


def test_the_predicate_catches_nan_and_both_infinities_but_not_null():
    p = bv._nonfinite_sql("close")
    # NULL is not a defect; NaN fails both comparisons; +/-Infinity fail one.
    # Pinned to the text checked on Postgres 17 (2026-11-26, read-only): it
    # matched the 84 NaN rows in 65 tickers the audit counted.
    assert p == ("(close IS NOT NULL AND NOT (close::float8 > '-Infinity'::float8 "
                 "AND close::float8 < 'Infinity'::float8))")


def test_no_finite_price_is_selected():
    sql = bv.nonfinite_bars_sql()
    select_list = sql.split(" FROM r ")[0].split(") SELECT ", 1)[1]
    assert select_list.startswith("ticker, date::text, CASE WHEN")
    for col in ("open", "high", "low", "close"):
        assert f"THEN {col}::text END" in select_list
    # Every price column in the outer select sits inside a CASE that is NULL
    # unless the value is non-finite.
    assert select_list.count("CASE WHEN") == 4 and select_list.count(" END") == 4


# --- the read ---------------------------------------------------------------

class _Cur:
    def __init__(self, log, rows=(), raise_on=None):
        self.log, self.rows, self.raise_on = log, list(rows), raise_on

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append(sql)
        if self.raise_on and self.raise_on in sql:
            raise RuntimeError("canceling statement due to statement timeout")

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, log, rows=(), raise_on=None):
        self.log, self.rows, self.raise_on = log, rows, raise_on

    def cursor(self):
        return _Cur(self.log, self.rows, self.raise_on)

    def rollback(self):
        self.log.append("ROLLBACK")


def _fake_db(monkeypatch, conn):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append)
    monkeypatch.setitem(sys.modules, "db", mod)
    return put


def test_the_read_is_read_only_bounded_and_rolled_back(monkeypatch):
    log = []
    conn = _Conn(log, rows=[("SNDK", "2026-11-20", "NaN", "NaN", "NaN", "NaN"),
                            ("X", "2026-11-02", None, None, None, "Infinity")])
    put = _fake_db(monkeypatch, conn)
    out = bv.fetch_nonfinite_bars()
    assert log[0] == "SET TRANSACTION READ ONLY"
    assert log[1] == "SET LOCAL statement_timeout = 30000"
    assert log[2] == bv.nonfinite_bars_sql()
    assert log[-1] == "ROLLBACK" and put == [conn]
    assert out == [{"ticker": "SNDK", "date": "2026-11-20", "values": ALL_NAN},
                   {"ticker": "X", "date": "2026-11-02", "values": {"close": "Infinity"}}]


def test_a_read_that_raises_returns_the_error_and_still_returns_the_conn(monkeypatch):
    log = []
    conn = _Conn(log, raise_on="WITH r AS")
    put = _fake_db(monkeypatch, conn)
    out = bv.fetch_nonfinite_bars()
    assert isinstance(out, RuntimeError)
    assert log[-1] == "ROLLBACK" and put == [conn]
    assert bv.check_nonfinite_bars(out)["status"] == "WARN"


def test_a_failed_connection_is_returned_as_the_error(monkeypatch):
    def boom():
        raise ConnectionError("pool exhausted")
    monkeypatch.setitem(sys.modules, "db", types.SimpleNamespace(_get_conn=boom, _put_conn=None))
    assert isinstance(bv.fetch_nonfinite_bars(), ConnectionError)


# --- wiring -----------------------------------------------------------------

def test_run_checks_adds_the_line_only_when_given():
    base, _ = bv.run_checks([], [], [], [], [], [], {}, SINCE, NOW, "live")
    assert "nonfinite_bars" not in [c["name"] for c in base]
    checks, _ = bv.run_checks([], [], [], [], [], [], {}, SINCE, NOW, "live",
                              nonfinite_bars=[_bar("SNDK", "2026-11-25")])
    line = [c for c in checks if c["name"] == "nonfinite_bars"]
    assert len(line) == 1 and line[0]["status"] == "WARN"
    assert len(checks) == len(base) + 1


def test_main_runs_the_read():
    assert "nonfinite_bars=fetch_nonfinite_bars()" in inspect.getsource(bv.main)


def test_a_nonfinite_warn_does_not_change_the_exit_status():
    # main exits 1 only on FAIL; this line is never FAIL.
    assert 'sys.exit(1 if any(c["status"] == "FAIL" for c in checks) else 0)' in inspect.getsource(bv.main)


def test_the_docstring_lists_the_record():
    assert "  nonfinite_bars      price_history bars with a non-finite" in bv.__doc__


@pytest.mark.parametrize("rows", [260, 30])
def test_the_window_argument_reaches_the_sql_and_the_line(rows):
    assert f"rn <= {rows} AND" in bv.nonfinite_bars_sql(rows)
    assert f"last {rows} rows" in bv.check_nonfinite_bars([], rows)["why"]
