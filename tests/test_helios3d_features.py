"""Point-in-time feature store: every reader filters on its as-of column in
SQL and refuses later rows in Python, and flags asof_safe. Plus the
append-only price capture and its migration. No real database: a fake
connection records SQL and returns canned rows (including rows dated after
as_of, as a stand-in for a filter that failed)."""
import contextlib
import re
from datetime import date, datetime, timedelta, timezone

import pytest

from helios3d.features import capture as C
from helios3d.features import store as S
from helios3d.record.labels import ET, session_close_at
from helios3d.schema import migrate

AS_OF = datetime(2026, 10, 30, 12, 0, tzinfo=ET)  # a Friday, mid-session
BEFORE = AS_OF - timedelta(hours=1)
AFTER = AS_OF + timedelta(minutes=1)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self._rows = []
        self.rowcount = -1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        for key, (cols, rows) in self.conn.responses.items():
            if key in sql:
                self.description = [(c,) for c in cols]
                self._rows = [tuple(r.get(c) for c in cols) for r in rows] if rows and isinstance(rows[0], dict) else list(rows)
                break
        else:
            self.description, self._rows = None, []
        if sql.lstrip().startswith("WITH") and "INSERT INTO h3d.price_history_asof" in sql:
            self.rowcount = self.conn.insert_rowcount

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, responses=None, insert_rowcount=0):
        self.responses = responses or {}
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.insert_rowcount = insert_rowcount

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _conn_for(table, rows):
    cols = list(rows[0]) if rows else ["id"]
    return FakeConn({table: (cols, rows)})


def _reads(conn):
    return [(s, p) for s, p in conn.executed if "SET TRANSACTION" not in s]


def _assert_read_only(conn):
    sqls = [s for s, _ in conn.executed]
    assert "SET TRANSACTION READ ONLY" in sqls
    assert conn.commits == 0


# --- as_of handling --------------------------------------------------------

def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError):
        S.fundamentals("NVDA", datetime(2026, 10, 30, 12, 0), conn=FakeConn())


def test_get_dispatches_and_rejects_unknown_source():
    conn = _conn_for("public.fundamentals_history",
                     [{"id": 1, "ticker": "NVDA", "recorded_at": BEFORE}])
    res = S.get("fundamentals", "nvda", AS_OF, conn=conn)
    assert res.source == "fundamentals" and res.ticker == "NVDA"
    with pytest.raises(ValueError, match="unknown feature source"):
        S.get("tea_leaves", "NVDA", AS_OF)


def test_every_source_is_registered_with_its_flag():
    expected = {"price_bars": False, "price_bars_asof": True, "fundamentals": True,
                "regime": True, "earnings_calendar": True, "earnings_consensus": True, "news": True,
                "portfolio_snapshot": True, "position_stops": False}
    assert set(S.SOURCES) == set(expected)
    for source, safe in expected.items():
        res = S.get(source, "NVDA", AS_OF, conn=FakeConn({"FROM": (["id"], [])}))
        assert res.asof_safe is safe, source
        assert res.reason and res.asof_column and res.as_of == AS_OF


# --- price_history (unsafe) ------------------------------------------------

def test_price_bars_refuse_unclosed_session_and_are_flagged_unsafe():
    rows = [{"ticker": "NVDA", "date": d, "open": 1, "high": 1, "low": 1, "close": c, "volume": 1}
            for d, c in ((date(2026, 10, 30), 3.0), (date(2026, 10, 29), 2.0), (date(2026, 10, 28), 1.0))]
    conn = _conn_for("public.price_history", rows)
    res = S.price_bars("NVDA", AS_OF, days=5, conn=conn)
    assert [r["date"] for r in res.rows] == [date(2026, 10, 28), date(2026, 10, 29)]  # oldest first
    assert res.refused == 1  # 10-30's session closes at 16:00, after as_of
    assert res.asof_safe is False and "upserted" in res.reason
    sql, params = _reads(conn)[0]
    assert "date <= %s" in sql and params[1] == date(2026, 10, 30)
    _assert_read_only(conn)


def test_price_bars_after_close_include_that_day_and_trim_to_days():
    rows = [{"ticker": "NVDA", "date": date(2026, 10, 30 - i), "close": float(i)} for i in range(3)]
    conn = _conn_for("public.price_history", rows)
    res = S.price_bars("NVDA", session_close_at(date(2026, 10, 30)), days=2, conn=conn)
    assert [r["date"] for r in res.rows] == [date(2026, 10, 29), date(2026, 10, 30)]
    assert res.refused == 0


# --- h3d.price_history_asof (safe) ------------------------------------------

def test_price_bars_asof_refuses_later_captures_and_unclosed_bars():
    d1, d0 = date(2026, 10, 29), date(2026, 10, 30)
    rows = [
        {"ticker": "NVDA", "date": d0, "close": 9.0, "captured_at": BEFORE},       # session not closed
        {"ticker": "NVDA", "date": d1, "close": 2.0, "captured_at": AFTER},        # captured later
        {"ticker": "NVDA", "date": date(2026, 10, 28), "close": 1.0,
         "captured_at": session_close_at(date(2026, 10, 28)) - timedelta(hours=2)},  # partial
        {"ticker": "NVDA", "date": date(2026, 10, 27), "close": 0.5,
         "captured_at": session_close_at(date(2026, 10, 27)) + timedelta(hours=3)},
    ]
    conn = _conn_for("h3d.price_history_asof", rows)
    res = S.price_bars_asof("NVDA", AS_OF, conn=conn)
    assert [r["date"] for r in res.rows] == [date(2026, 10, 27), date(2026, 10, 28)]
    assert res.refused == 2
    assert [r["captured_before_close"] for r in res.rows] == [False, True]
    assert res.asof_safe is True
    sql, params = _reads(conn)[0]
    assert "captured_at <= %s" in sql and params[1] == AS_OF
    assert "DISTINCT ON (date)" in sql


# --- fundamentals ------------------------------------------------------------

def test_fundamentals_refuse_rows_recorded_after_as_of():
    rows = [{"id": 2, "ticker": "NVDA", "recorded_at": AFTER, "forward_pe": 99},
            {"id": 1, "ticker": "NVDA", "recorded_at": BEFORE, "forward_pe": 30}]
    conn = _conn_for("public.fundamentals_history", rows)
    res = S.fundamentals("NVDA", AS_OF, limit=2, conn=conn)
    assert [r["id"] for r in res.rows] == [1] and res.refused == 1
    assert res.asof_safe is True
    sql, params = _reads(conn)[0]
    assert "recorded_at <= %s" in sql and params == ("NVDA", AS_OF, 2)
    _assert_read_only(conn)


def test_naive_db_timestamps_are_read_as_utc():
    naive_after = AFTER.astimezone(timezone.utc).replace(tzinfo=None)
    conn = _conn_for("public.fundamentals_history",
                     [{"id": 1, "ticker": "NVDA", "recorded_at": naive_after}])
    assert S.fundamentals("NVDA", AS_OF, conn=conn).refused == 1


def test_rows_with_no_knowledge_time_are_refused():
    conn = _conn_for("public.fundamentals_history", [{"id": 1, "ticker": "NVDA", "recorded_at": None}])
    res = S.fundamentals("NVDA", AS_OF, conn=conn)
    assert res.rows == [] and res.refused == 1


# --- regime ---------------------------------------------------------------

def test_regime_refuses_later_rows_and_rows_replayed_after_as_of():
    rows = [
        {"id": 3, "timestamp": AFTER, "regime": "Bull", "replayed_at": None},
        {"id": 2, "timestamp": BEFORE, "regime": "Bear", "replayed_at": AFTER},  # backdated replay
        {"id": 1, "timestamp": BEFORE - timedelta(hours=1), "regime": "Neutral",
         "replayed_at": BEFORE},
    ]
    conn = _conn_for("public.market_regime_log", rows)
    res = S.regime(None, AS_OF, limit=3, conn=conn)
    assert [r["id"] for r in res.rows] == [1] and res.refused == 2
    assert res.asof_safe is True
    sql, params = _reads(conn)[0]
    assert "r.timestamp <= %s" in sql and "x.row_id IS NULL" in sql
    assert "a.ts > %s" in sql  # the replay scan is bounded to after as_of
    assert "stage = 'spill_replay'" in sql and "extra_json->>'kind' = 'regime'" in sql
    assert params == (AS_OF, AS_OF, 3)


def test_regime_type_overall_matches_legacy_nulls():
    conn = _conn_for("public.market_regime_log", [])
    S.regime("NVDA", AS_OF, regime_type="overall", conn=conn)
    assert "r.regime_type IS NULL" in _reads(conn)[0][0]
    conn = _conn_for("public.market_regime_log", [])
    S.regime(None, AS_OF, regime_type="sector", sector_etf="soxx", conn=conn)
    assert _reads(conn)[0][1] == (AS_OF, AS_OF, "sector", "SOXX", 1)


# --- earnings calendar -----------------------------------------------------

def test_earnings_calendar_refuses_dates_recorded_after_as_of():
    rows = [{"id": 2, "ticker": "NVDA", "earnings_date": date(2026, 11, 20), "recorded_at": AFTER},
            {"id": 1, "ticker": "NVDA", "earnings_date": date(2026, 11, 19), "recorded_at": BEFORE}]
    conn = _conn_for("public.earnings_calendar_history", rows)
    res = S.earnings_calendar("NVDA", AS_OF, conn=conn)
    assert [r["earnings_date"] for r in res.rows] == [date(2026, 11, 19)] and res.refused == 1
    assert res.asof_safe is True and "recorded_at <= %s" in _reads(conn)[0][0]


# --- news -----------------------------------------------------------------

def test_news_refuses_later_headlines_and_names_the_insert_time():
    rows = [{"id": 2, "ticker": "NVDA", "title": "later", "published_at": AFTER},
            {"id": 1, "ticker": "NVDA", "title": "earlier", "published_at": BEFORE}]
    conn = _conn_for("public.news_log", rows)
    res = S.news("NVDA", AS_OF, since=AS_OF - timedelta(days=3), conn=conn)
    assert [r["title"] for r in res.rows] == ["earlier"] and res.refused == 1
    assert res.rows[0]["logged_at"] == BEFORE
    assert res.asof_safe is True and "insert time" in res.reason
    sql, params = _reads(conn)[0]
    assert "published_at <= %s" in sql and "published_at >= %s" in sql
    assert params[1] == AS_OF


def test_news_log_writer_never_sets_published_at():
    """The asof_safe=True flag rests on this: published_at is DEFAULT NOW()."""
    import inspect
    import db
    src = inspect.getsource(db.log_news_item)
    assert "published_at" not in src


# --- portfolio ------------------------------------------------------------

def test_portfolio_snapshot_refuses_later_snapshots_and_scopes_env():
    rows = [{"id": 2, "timestamp": AFTER, "trading_env": "live"},
            {"id": 1, "timestamp": BEFORE, "trading_env": "live"}]
    conn = _conn_for("public.portfolio_snapshots", rows)
    res = S.portfolio_snapshot(None, AS_OF, trading_env="live", limit=2, conn=conn)
    assert [r["id"] for r in res.rows] == [1] and res.refused == 1 and res.asof_safe is True
    assert _reads(conn)[0][1] == ("live", AS_OF, 2)


def test_portfolio_snapshot_defaults_to_config_env(monkeypatch):
    import config
    monkeypatch.setattr(config, "TRADING_ENV", "paper")
    conn = _conn_for("public.portfolio_snapshots", [])
    S.portfolio_snapshot(None, AS_OF, conn=conn)
    assert _reads(conn)[0][1][0] == "paper"


def test_position_stops_refuse_rows_changed_after_as_of_and_are_unsafe():
    rows = [
        {"ticker": "AMD", "opened_at": BEFORE, "updated_at": BEFORE, "thesis_updated_at": None},
        {"ticker": "KLAC", "opened_at": BEFORE, "updated_at": AFTER, "thesis_updated_at": None},
        {"ticker": "ORCL", "opened_at": BEFORE, "updated_at": BEFORE, "thesis_updated_at": AFTER},
        {"ticker": "MU", "opened_at": AFTER, "updated_at": AFTER, "thesis_updated_at": None},
    ]
    conn = _conn_for("public.position_stops", rows)
    res = S.position_stops(None, AS_OF, trading_env="live", conn=conn)
    assert [r["ticker"] for r in res.rows] == ["AMD"] and res.refused == 3
    assert res.asof_safe is False and "deleted" in res.reason
    sql, params = _reads(conn)[0]
    assert "opened_at <= %s" in sql and params == ("live", AS_OF)
    conn = _conn_for("public.position_stops", [])
    S.position_stops("amd", AS_OF, trading_env="live", conn=conn)
    assert _reads(conn)[0][1] == ("live", AS_OF, "AMD")


@pytest.mark.parametrize("source", sorted(S.SOURCES))
def test_every_reader_qualifies_its_tables_and_reads_only(source):
    conn = FakeConn({"FROM": (["id"], [])})
    S.get(source, "NVDA", AS_OF, conn=conn)
    sql = _reads(conn)[0][0]
    tables = re.findall(r"\b(?:FROM|JOIN)\s+([\w\.]+)", sql)
    assert tables and all(t.startswith(("public.", "h3d.")) or t in ("replayed",) for t in tables), tables
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b", sql)
    _assert_read_only(conn)


def test_reader_without_conn_uses_the_helios3d_connection(monkeypatch):
    conn = _conn_for("public.news_log", [])

    @contextlib.contextmanager
    def fake_connection():
        yield conn

    monkeypatch.setattr("helios3d.schema.conn.connection", fake_connection)
    assert S.news("NVDA", AS_OF).rows == []
    assert _reads(conn)


# --- migration 3 ------------------------------------------------------------

def test_migration_3_is_price_history_asof():
    v, name, sql = migrate.MIGRATIONS[2]
    assert (v, name) == (3, "price_history_asof")
    assert "CREATE TABLE IF NOT EXISTS h3d.price_history_asof" in sql
    for col in ("ticker", "date", "open", "high", "low", "close", "volume", "captured_at", "code_sha"):
        assert re.search(rf"^\s*{col}\s", sql, re.M), col
    assert "ON h3d.price_history_asof (ticker, date, captured_at DESC)" in sql


def test_capture_and_reader_columns_exist_in_migration_3():
    sql = migrate.MIGRATIONS[2][2]
    cols = re.search(r"INSERT INTO h3d.price_history_asof\s*\(([^)]*)\)", C.capture_sql()[0]).group(1)
    for col in (c.strip() for c in cols.split(",")):
        assert re.search(rf"^\s*{col}\s", sql, re.M), col


# --- capture ------------------------------------------------------------

def test_capture_sql_appends_only_new_or_changed_bars():
    sql, params = C.capture_sql(["NVDA"], sha="abc")
    assert sql.startswith("WITH latest AS")
    assert "INSERT INTO h3d.price_history_asof" in sql
    assert "FROM public.price_history p" in sql
    assert "l.ticker IS NULL OR (p.open, p.high, p.low, p.close, p.volume) IS DISTINCT FROM" in sql
    assert not re.search(r"\b(UPDATE|DELETE)\b", sql)
    assert params == (["NVDA"], "abc", ["NVDA"])
    assert sql.count("%s") == len(params)
    sql, params = C.capture_sql(sha="abc")
    assert params == ("abc",) and sql.count("%s") == 1


def test_count_sql_placeholders_match():
    for t in (None, ["NVDA", "SPY"]):
        sql, params = C.count_sql(t)
        assert sql.count("%s") == len(params)


def test_dry_run_counts_without_writing():
    conn = FakeConn({"to_regclass": (["r"], [("h3d.price_history_asof",)]),
                     "SELECT COUNT(*), COUNT(*) FILTER": (["a", "b", "c"], [(7, 5, 2)])})
    out = C.run(apply=False, conn=conn)
    assert out == {"mode": "dry-run", "tickers": "all", "table_exists": True,
                   "would_append": 7, "new": 5, "changed": 2}
    _assert_read_only(conn)
    assert not any("INSERT" in s for s, _ in conn.executed)


def test_dry_run_before_migration_counts_all_rows():
    conn = FakeConn({"to_regclass": (["r"], [(None,)]),
                     "SELECT COUNT(*) FROM public.price_history": (["n"], [(42,)])})
    out = C.run(apply=False, tickers=["nvda"], conn=conn)
    assert out["table_exists"] is False and out["would_append"] == 42
    assert out["tickers"] == ["NVDA"]
    assert conn.executed[-1][1] == (["NVDA"],)
    assert conn.commits == 0


def test_apply_migrates_then_inserts_and_commits(monkeypatch):
    applied = []
    monkeypatch.setattr(migrate, "apply", lambda c: applied.append(c) or [])
    monkeypatch.setattr("helios3d.record.labels.code_sha", lambda: "deadbeef")
    conn = FakeConn(insert_rowcount=11)
    out = C.run(apply=True, conn=conn)
    assert applied == [conn]
    assert out["appended"] == 11 and conn.commits == 1
    sqls = [s for s, _ in conn.executed]
    sql, params = conn.executed[-1]
    assert "INSERT INTO h3d.price_history_asof" in sql and params == ("deadbeef",)
    assert "statement_timestamp()" in sql
    i = sqls.index(C.LOCK_SQL)
    assert conn.executed[i][1] == (C.CAPTURE_LOCK_KEY,)
    assert sqls[i - 1] == C.ISOLATION_SQL and i == len(sqls) - 2  # lock, then the INSERT
    assert -2**63 <= C.CAPTURE_LOCK_KEY < 2**63


def test_cli_defaults_to_dry_run(monkeypatch, capsys):
    conn = FakeConn({"to_regclass": (["r"], [(None,)]),
                     "SELECT COUNT(*) FROM public.price_history": (["n"], [(0,)])})

    @contextlib.contextmanager
    def fake_connection():
        yield conn

    monkeypatch.setattr("helios3d.schema.conn.connection", fake_connection)
    assert C.main([]) == 0
    assert conn.commits == 0
    assert "mode: dry-run" in capsys.readouterr().out


def test_capture_not_scheduled():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    assert "helios3d" not in (root / "tasks.py").read_text(encoding="utf-8")


# --- latest capture per date (pure) ---------------------------------------

def test_latest_capture_per_date_takes_newest_capture_at_or_before_as_of():
    d1, d2 = date(2026, 10, 28), date(2026, 10, 29)
    rows = [
        {"id": 1, "date": d1, "close": 1.0, "captured_at": BEFORE - timedelta(days=1)},
        {"id": 2, "date": d1, "close": 1.1, "captured_at": BEFORE},          # newest by as_of
        {"id": 3, "date": d1, "close": 9.9, "captured_at": AFTER},           # after as_of
        {"id": 4, "date": d2, "close": 2.0, "captured_at": BEFORE},
        {"id": 5, "date": d2, "close": 2.1, "captured_at": BEFORE},          # same time, higher id
        {"id": 6, "date": d2, "close": 0.0, "captured_at": None},
        {"id": 7, "date": date(2026, 10, 30), "close": 3.0, "captured_at": AFTER},
    ]
    out = S.latest_capture_per_date(rows, AS_OF)
    assert [(r["date"], r["id"]) for r in out] == [(d1, 2), (d2, 5)]
    assert S.latest_capture_per_date(list(reversed(rows)), AS_OF) == out
    assert S.latest_capture_per_date(rows, BEFORE - timedelta(days=2)) == []


def test_price_bars_asof_applies_the_rule_to_the_sql_result():
    """Even if the SQL returned several captures for one date, one survives."""
    d = date(2026, 10, 27)
    close = session_close_at(d)
    rows = [{"id": 2, "ticker": "NVDA", "date": d, "close": 2.0, "captured_at": close + timedelta(hours=2)},
            {"id": 1, "ticker": "NVDA", "date": d, "close": 1.0, "captured_at": close + timedelta(hours=1)}]
    res = S.price_bars_asof("NVDA", AS_OF, conn=_conn_for("h3d.price_history_asof", rows))
    assert [(r["id"], r["close"]) for r in res.rows] == [(2, 2.0)]


@pytest.mark.parametrize("source,kw", [("price_bars", {"days": -5}), ("price_bars_asof", {"days": -5}),
                                       ("fundamentals", {"limit": -1}), ("news", {"limit": -1})])
def test_negative_days_or_limit_are_clamped_to_zero(source, kw):
    conn = FakeConn({"FROM": (["id"], [])})
    res = S.get(source, "NVDA", AS_OF, conn=conn, **kw)
    assert res.rows == []
    assert _reads(conn)[0][1][-1] in (0, 1)  # LIMIT 0 (or days+1 = 1), never negative


def test_every_reader_documents_the_rollback_of_a_supplied_conn():
    for name, fn in S.SOURCES.items():
        assert "rolled back (read_only())" in (fn.__doc__ or ""), name


# --- guards behind asof_safe=True ------------------------------------------

import ast as _ast
import pathlib as _pathlib

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
# table -> (as-of column, the only functions that may insert into it).
# No code may UPDATE, DELETE FROM or TRUNCATE them (append-only).
_SAFE_WRITERS = {
    "news_log": ("published_at", {("db.py", "log_news_item")}),
    "fundamentals_history": ("recorded_at", {("db.py", "log_fundamentals_history")}),
    "market_regime_log": ("timestamp", {("db.py", "log_regime"),
                                        ("scripts/replay_spill.py", "insert_row")}),
    "portfolio_snapshots": ("timestamp", {("db.py", "log_snapshot")}),
    "earnings_calendar_history": ("recorded_at", {("db.py", "upsert_earnings_date")}),
    "earnings_consensus_history": ("fetched_at", {("db.py", "log_earnings_consensus")}),
}
_SKIP_DIRS = ("helios3d", "tests", "venv", ".venv", ".git", ".claude")


_WRITE_VERBS = {"insert": r"INSERT\s+INTO", "update": r"(?<!DO\s)UPDATE",
                "delete": r"DELETE\s+FROM", "truncate": r"TRUNCATE(?:\s+TABLE)?(?:\s+ONLY)?"}


def _writes_by_function():
    """{(table, verb): {(file, enclosing function)}} for every string literal
    in the repo's Python (outside helios3d/, tests/ and tooling) that
    inserts, updates, deletes from or truncates a table guarded above."""
    found = {(t, v): set() for t in _SAFE_WRITERS for v in _WRITE_VERBS}
    pats = {(t, v): re.compile(rf"\b{vp}\s+(?:public\.)?{t}\b", re.I)
            for t in _SAFE_WRITERS for v, vp in _WRITE_VERBS.items()}
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT)
        if rel.parts[0] in _SKIP_DIRS:
            continue
        tree = _ast.parse(path.read_text(encoding="utf-8"))

        def visit(node, fn):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                fn = node.name
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                for t, pat in pats.items():
                    if pat.search(node.value):
                        found[t].add((rel.as_posix(), fn))
            for child in _ast.iter_child_nodes(node):
                visit(child, fn)
        visit(tree, None)
    return found


def test_only_the_known_writers_insert_into_asof_safe_tables_and_nothing_mutates_them():
    found = _writes_by_function()
    for table, (_, writers) in _SAFE_WRITERS.items():
        assert found[(table, "insert")] == writers, (table, found[(table, "insert")])
        for verb in ("update", "delete", "truncate"):
            assert found[(table, verb)] == set(), (table, verb, found[(table, verb)])


@pytest.mark.parametrize("sql,verb", [
    ("UPDATE news_log SET title='x'", "update"), ("DELETE FROM public.news_log WHERE id=1", "delete"),
    ("TRUNCATE TABLE news_log", "truncate"), ("TRUNCATE news_log", "truncate"),
    ("INSERT INTO news_log (ticker) VALUES (1)", "insert")])
def test_writer_patterns_match_each_verb(sql, verb):
    pat = re.compile(rf"\b{_WRITE_VERBS[verb]}\s+(?:public\.)?news_log\b", re.I)
    assert pat.search(sql)
    assert not re.search(rf"\b{_WRITE_VERBS['update']}\s+news_log\b",
                         "INSERT INTO news_log ... ON CONFLICT (id) DO UPDATE SET title='x'")


@pytest.mark.parametrize("table", sorted(t for t in _SAFE_WRITERS))
def test_db_writer_never_names_the_asof_column(table):
    """asof_safe=True rests on the as-of column being DEFAULT NOW() at insert:
    the db.py writer must never set it."""
    import inspect
    import db
    col, writers = _SAFE_WRITERS[table]
    for path, fn in writers:
        if path != "db.py":
            continue
        src = inspect.getsource(getattr(db, fn))
        assert not re.search(rf"\b{col}\b", src), f"db.{fn} names {col}"


def test_replay_spill_still_dates_replayed_regime_rows():
    """The regime reader's replay rule needs replay_spill.py to keep writing an
    activity_feed row with stage 'spill_replay' and extra_json kind + row_id."""
    tree = _ast.parse((_ROOT / "scripts" / "replay_spill.py").read_text(encoding="utf-8"))
    ok = False
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call) and getattr(node.func, "attr", "") == "execute"
                and node.args and isinstance(node.args[0], _ast.Constant)
                and "INSERT INTO activity_feed" in str(node.args[0].value)):
            continue
        params = node.args[1] if len(node.args) > 1 else None
        if not isinstance(params, _ast.Tuple):
            continue
        consts = [e.value for e in params.elts if isinstance(e, _ast.Constant)]
        dicts = [d for e in params.elts for d in _ast.walk(e) if isinstance(d, _ast.Dict)]
        keys = {k.value for d in dicts for k in d.keys if isinstance(k, _ast.Constant)}
        if "spill_replay" in consts and {"kind", "row_id"} <= keys:
            ok = True
    assert ok
    assert "regime" in ast_literal_set(tree, "REPLAYABLE")


def ast_literal_set(tree, name):
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return _ast.literal_eval(node.value)
    return set()


def test_capture_refuses_an_autocommit_connection(monkeypatch):
    applied = []
    monkeypatch.setattr(migrate, "apply", lambda c: applied.append(c) or [])
    conn = FakeConn(insert_rowcount=3)
    conn.autocommit = True
    with pytest.raises(ValueError, match="autocommit"):
        C.capture(conn, sha="x")
    assert applied == [] and conn.executed == [] and conn.commits == 0


def test_price_bars_asof_selects_id_for_the_tie_break():
    conn = _conn_for("h3d.price_history_asof", [])
    S.price_bars_asof("NVDA", AS_OF, conn=conn)
    sql = _reads(conn)[0][0]
    assert re.search(r"SELECT DISTINCT ON \(date\) id,", sql)
