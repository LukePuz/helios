"""Forward-return labels: entry/forward session rules, skips, read-only verdict
read, writes only to h3d, dry-run default. DB and yfinance are mocked."""
import contextlib
import sys
import types
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import db
from helios3d.record import labels as L

ET = L.ET


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


SESSIONS = _weekdays(date(2026, 9, 1), 120)          # Sep 1 2026 onward
SPY = {d: 100.0 + i for i, d in enumerate(SESSIONS)}
AAA = {d: 50.0 * (1.01 ** i) for i, d in enumerate(SESSIONS)}
NOW = L.session_close_at(SESSIONS[-1]) + timedelta(hours=1)


def _v(sid, ticker, at, action="WATCH", conviction=6):
    return {"signal_id": sid, "ticker": ticker, "verdict_at": at, "action": action, "conviction": conviction}


# --- pure computation ------------------------------------------------------

def test_intraday_verdict_uses_same_day_close_and_returns_are_close_to_close():
    d0 = SESSIONS[10]
    at = datetime(d0.year, d0.month, d0.day, 10, 0, tzinfo=ET)
    rows, skips = L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA, "SPY": SPY}, NOW)
    by_h = {r["horizon_days"]: r for r in rows}
    assert set(by_h) == {5, 10, 20, 60}
    r5 = by_h[5]
    assert r5["entry_date"] == d0 and r5["fwd_date"] == SESSIONS[15]
    assert r5["entry_ref_price"] == AAA[d0]
    assert r5["fwd_return"] == pytest.approx(1.01 ** 5 - 1)
    assert r5["spy_fwd_return"] == pytest.approx(SPY[SESSIONS[15]] / SPY[d0] - 1)
    assert r5["excess_return"] == pytest.approx(r5["fwd_return"] - r5["spy_fwd_return"])
    assert r5["label_source"] == L.LABEL_SOURCE
    assert sum(skips.values()) == 0


def test_after_close_verdict_uses_next_session():
    d0 = SESSIONS[10]
    at = datetime(d0.year, d0.month, d0.day, 17, 0, tzinfo=ET)
    rows, _ = L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA, "SPY": SPY}, NOW)
    assert {r["entry_date"] for r in rows} == {SESSIONS[11]}


def test_exactly_at_close_uses_that_close():
    d0 = SESSIONS[10]
    rows, _ = L.compute_labels([_v(1, "AAA", L.session_close_at(d0))], {"AAA": AAA, "SPY": SPY}, NOW)
    assert {r["entry_date"] for r in rows} == {d0}


def test_utc_timestamp_is_converted_to_et():
    d0 = SESSIONS[10]  # 21:30 UTC = 17:30 EDT (after the close) in September
    at = datetime(d0.year, d0.month, d0.day, 21, 30, tzinfo=timezone.utc)
    rows, _ = L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA, "SPY": SPY}, NOW)
    assert {r["entry_date"] for r in rows} == {SESSIONS[11]}


def test_weekend_verdict_uses_monday():
    sat = date(2026, 9, 12)
    assert sat.weekday() == 5
    rows, _ = L.compute_labels([_v(1, "AAA", datetime(2026, 9, 12, 12, tzinfo=ET))],
                               {"AAA": AAA, "SPY": SPY}, NOW)
    assert {r["entry_date"] for r in rows} == {date(2026, 9, 14)}


def test_unreached_horizons_are_skipped():
    d0 = SESSIONS[-12]  # 11 completed sessions after entry
    at = datetime(d0.year, d0.month, d0.day, 9, 45, tzinfo=ET)
    rows, skips = L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA, "SPY": SPY}, NOW)
    assert sorted(r["horizon_days"] for r in rows) == [5, 10]
    assert skips["horizon_not_reached"] == 2


def test_in_progress_bar_is_not_a_completed_session():
    today = SESSIONS[-1]
    now = datetime(today.year, today.month, today.day, 11, 0, tzinfo=ET)
    assert L.completed_sessions(SPY, now)[-1] == SESSIONS[-2]
    at = datetime(today.year, today.month, today.day, 10, 0, tzinfo=ET)
    rows, skips = L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA, "SPY": SPY}, now)
    assert rows == [] and skips["entry_not_reached"] == 4 and skips["no_entry_session"] == 0


def test_missing_prices_skip_not_zero():
    d0 = SESSIONS[10]
    at = datetime(d0.year, d0.month, d0.day, 10, tzinfo=ET)
    gappy = dict(AAA)
    del gappy[SESSIONS[15]]
    rows, skips = L.compute_labels([_v(1, "AAA", at), _v(2, "ZZZ", at)], {"AAA": gappy, "SPY": SPY}, NOW)
    assert sorted((r["signal_id"], r["horizon_days"]) for r in rows) == [(1, 10), (1, 20), (1, 60)]
    assert skips["missing_ticker_price"] == 1 + 4


def test_no_benchmark_means_no_labels():
    at = datetime(2026, 9, 15, 10, tzinfo=ET)
    rows, skips = L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA}, NOW)
    assert rows == [] and skips["no_entry_session"] == 4


def test_yf_symbol_maps_class_shares():
    assert L.yf_symbol("brk.b") == "BRK-B"


# --- verdict read ----------------------------------------------------------

class RecCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))

    def fetchall(self):
        return list(self.conn.rows)


class RecConn:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return RecCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_verdict_query_uses_existing_verdict_filter():
    sql, params = L.verdict_query(None)
    assert "FROM signals" in sql and "signal NOT IN %s" in sql
    assert params[0] is db._NON_ANALYST_SIGNALS
    assert params[2] == db.DEGRADED_SIGNAL_PREFIX + "%"
    sql2, params2 = L.verdict_query(datetime(2026, 9, 17, tzinfo=ET))
    assert "timestamp >= %s" in sql2 and len(params2) == 4


def test_fetch_verdicts_reads_in_read_only_transaction_and_never_commits():
    ts = datetime(2026, 9, 17, 14, tzinfo=timezone.utc)
    conn = RecConn(rows=[(7, "aaa ", ts, "OPEN", 0.7), (8, "BBB", ts.replace(tzinfo=None), "WATCH", None)])
    out = L.fetch_verdicts(conn)
    sqls = [s for s, _ in conn.executed]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert sqls[1].startswith("SELECT")
    assert conn.commits == 0 and conn.rollbacks >= 1
    assert out[0] == {"signal_id": 7, "ticker": "AAA", "verdict_at": ts, "action": "OPEN", "conviction": 7}
    assert out[1]["verdict_at"].tzinfo is not None and out[1]["conviction"] is None


# --- yfinance (mocked) -----------------------------------------------------

def _fake_yf(df, calls):
    mod = types.ModuleType("yfinance")

    def download(tickers, **kw):
        calls.append((tickers, kw))
        return df
    mod.download = download
    return mod


def test_fetch_closes_batches_and_maps_symbols(monkeypatch):
    idx = pd.DatetimeIndex(["2026-09-14", "2026-09-15"])
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["BRK-B", "SPY", "DEAD"]])
    df = pd.DataFrame([[400.0, 600.0, float("nan"), 1, 1, 1],
                       [404.0, 606.0, float("nan"), 1, 1, 1]], index=idx, columns=cols)
    calls = []
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(df, calls))
    out = L.fetch_closes(["BRK.B", "SPY", "DEAD"], date(2026, 9, 10), date(2026, 9, 15))
    assert len(calls) == 1
    assert sorted(calls[0][0]) == ["BRK-B", "DEAD", "SPY"]
    assert calls[0][1]["auto_adjust"] is False and calls[0][1]["end"] == "2026-09-16"
    assert out == {"BRK.B": {date(2026, 9, 14): 400.0, date(2026, 9, 15): 404.0},
                   "SPY": {date(2026, 9, 14): 600.0, date(2026, 9, 15): 606.0}}


def test_fetch_closes_empty_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(pd.DataFrame(), []))
    assert L.fetch_closes(["AAA"], date(2026, 9, 1), date(2026, 9, 2)) == {}


# --- job -------------------------------------------------------------------

def _patch_job(monkeypatch, verdicts):
    monkeypatch.setattr(L, "fetch_verdicts", lambda conn, since=None: verdicts)
    monkeypatch.setattr(L, "fetch_closes", lambda t, s, e: {"AAA": AAA, "SPY": SPY})


def test_dry_run_writes_nothing(monkeypatch):
    at = datetime.combine(SESSIONS[10], datetime.min.time(), tzinfo=ET).replace(hour=10)
    _patch_job(monkeypatch, [_v(1, "AAA", at)])
    monkeypatch.setattr(L, "upsert_labels", lambda *a, **k: pytest.fail("dry-run wrote"))
    conn = RecConn()
    s = L.run(apply=False, now=NOW, conn=conn)
    assert s["mode"] == "dry-run" and s["labels"] == 4 and s["written"] == 0
    assert conn.commits == 0 and conn.executed == []


def test_apply_migrates_then_upserts_into_h3d_only(monkeypatch):
    at = datetime.combine(SESSIONS[10], datetime.min.time(), tzinfo=ET).replace(hour=10)
    _patch_job(monkeypatch, [_v(1, "AAA", at)])
    from helios3d.schema import migrate
    order = []
    monkeypatch.setattr(migrate, "apply", lambda conn: order.append("migrate") or [1])
    captured = {}

    def fake_execute_values(cur, sql, values, template=None, page_size=100):
        name = ("archive" if "verdict_label_history" in sql else
                "clear" if sql.startswith("DELETE FROM h3d.label_skips") else
                "skips" if "label_skips" in sql else "upsert")
        order.append(name)
        captured[name] = dict(sql=sql, values=values, template=template)
    monkeypatch.setattr("psycopg2.extras.execute_values", fake_execute_values)
    conn = RecConn()
    s = L.run(apply=True, now=NOW, conn=conn)
    assert order == ["migrate", "archive", "upsert", "clear"]   # no skipped pairs here
    assert s["written"] == 4 and s["skips_written"] == 0 and conn.commits == 2
    assert captured["clear"]["values"] == [(1, h) for h in (5, 10, 20, 60)]
    assert "public." not in captured["clear"]["sql"]
    up = captured["upsert"]
    assert up["sql"].startswith("INSERT INTO h3d.verdict_labels")
    assert "ON CONFLICT (signal_id, horizon_days) DO UPDATE" in up["sql"]
    assert "public." not in up["sql"]
    assert len(up["values"][0]) == len(L._COLS)
    row = dict(zip(L._COLS, up["values"][0]))
    assert row["entry_date"] == SESSIONS[10] and row["code_sha"] == "test"
    assert len(row["price_snapshot_sha"]) == 16
    arch = captured["archive"]
    assert arch["sql"].startswith("INSERT INTO h3d.verdict_label_history")
    assert "IS DISTINCT FROM n.price_snapshot_sha" in arch["sql"]
    assert arch["values"][0] == (row["signal_id"], row["horizon_days"], row["price_snapshot_sha"])


def test_apply_writes_failed_pairs_but_not_pending_ones(monkeypatch):
    """A name with no prices is kept in h3d.label_skips with its
    reason; a horizon not yet reached is not a failure and is not written."""
    at = datetime.combine(SESSIONS[10], datetime.min.time(), tzinfo=ET).replace(hour=10)
    late = datetime.combine(SESSIONS[-3], datetime.min.time(), tzinfo=ET).replace(hour=10)
    _patch_job(monkeypatch, [_v(1, "AAA", at), _v(2, "DEAD", at), _v(3, "AAA", late)])
    from helios3d.schema import migrate
    monkeypatch.setattr(migrate, "apply", lambda conn: [])
    captured = {}

    def fake_execute_values(cur, sql, values, template=None, page_size=100):
        if sql.startswith("DELETE FROM h3d.label_skips"):
            captured.setdefault("clear", []).extend(values)
            return
        captured.setdefault("skips" if "label_skips" in sql else "other", []).extend(values)
        if "label_skips" in sql:
            captured["skips_sql"], captured["template"] = sql, template
    monkeypatch.setattr("psycopg2.extras.execute_values", fake_execute_values)
    s = L.run(apply=True, now=NOW, conn=RecConn())
    rows = [dict(zip(L._SKIP_COLS, v)) for v in captured["skips"]]
    assert {(r["signal_id"], r["reason"]) for r in rows} == {(2, "missing_ticker_price")}
    assert sorted(r["horizon_days"] for r in rows) == [5, 10, 20, 60]
    assert rows[0]["entry_date"] == SESSIONS[10] and rows[0]["code_sha"] == "test"
    assert s["skips_written"] == 4 and s["skips"]["horizon_not_reached"] == 4
    assert captured["skips_sql"].startswith("INSERT INTO h3d.label_skips")
    assert "ON CONFLICT (signal_id, horizon_days) DO UPDATE" in captured["skips_sql"]
    assert "first_seen_at=" not in captured["skips_sql"]          # first sighting kept
    assert "public." not in captured["skips_sql"]


def test_price_snapshot_sha_changes_with_any_input_close():
    a = L.price_snapshot_sha("AAA", SESSIONS[0], SESSIONS[5], 50.0, 51.0, 100.0, 105.0)
    assert a == L.price_snapshot_sha("AAA", SESSIONS[0], SESSIONS[5], 50.0, 51.0, 100.0, 105.0)
    assert a != L.price_snapshot_sha("AAA", SESSIONS[0], SESSIONS[5], 50.0, 51.0, 100.0, 105.01)
    assert a != L.price_snapshot_sha("AAA", SESSIONS[0], SESSIONS[6], 50.0, 51.0, 100.0, 105.0)
    # last-bit float noise between downloads is not a revision
    assert a == L.price_snapshot_sha("AAA", SESSIONS[0], SESSIONS[5], 50.0 + 1e-12, 51.0, 100.0, 105.0)


def test_a_pending_entry_is_not_written_as_a_skip(monkeypatch):
    """A verdict after the last completed close is not due yet, not a dropout."""
    today = SESSIONS[-1]
    now = datetime(today.year, today.month, today.day, 11, 0, tzinfo=ET)
    at = datetime(today.year, today.month, today.day, 10, 0, tzinfo=ET)
    skipped = []
    L.compute_labels([_v(1, "AAA", at)], {"AAA": AAA, "SPY": SPY}, now, skipped=skipped)
    assert {s["reason"] for s in skipped} == {"entry_not_reached"}
    assert all(s["reason"] in L._PENDING_SKIPS for s in skipped)


def test_half_day_closes_at_1300_et():
    """2026-11-27 is an NYSE early close (13:00 ET). A 14:00 verdict
    that day comes after the close and uses the next session."""
    days = _weekdays(date(2026, 11, 2), 80)
    days = [d for d in days if d != date(2026, 11, 26)]           # Thanksgiving
    spy = {d: 100.0 + i for i, d in enumerate(days)}
    aaa = {d: 50.0 + i for i, d in enumerate(days)}
    now = L.session_close_at(days[-1]) + timedelta(hours=1)
    half = date(2026, 11, 27)
    assert L.session_close_at(half) == datetime(2026, 11, 27, 13, 0, tzinfo=ET)
    assert L.session_close_at(date(2026, 11, 30)).hour == 16
    rows, _ = L.compute_labels([_v(1, "AAA", datetime(2026, 11, 27, 14, 0, tzinfo=ET)),
                                _v(2, "AAA", datetime(2026, 11, 27, 12, 0, tzinfo=ET))],
                               {"AAA": aaa, "SPY": spy}, now)
    entry = {r["signal_id"]: r["entry_date"] for r in rows}
    assert entry == {1: date(2026, 11, 30), 2: half}
    r2 = next(r for r in rows if r["signal_id"] == 2)
    assert r2["entry_close_at"] == datetime(2026, 11, 27, 13, 0, tzinfo=ET)
    # a 13:30 bar on the half day is completed once 13:00 has passed
    assert L.completed_sessions(spy, datetime(2026, 11, 27, 13, 30, tzinfo=ET))[-1] == half


def test_labels_import_no_broker_layer():
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(L.__file__).read_text())
    mods = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "market_data" not in mods and "market_calendar" in mods


def test_upsert_sql_targets_only_h3d():
    import re
    tables = re.findall(r"\b(?:INSERT INTO|UPDATE|DELETE FROM)\s+([\w\.]+)", L.UPSERT_SQL)
    # "DO UPDATE SET" is the conflict clause on the same row, not a table
    assert [t for t in tables if t != "SET"] == ["h3d.verdict_labels"]


def test_cli_default_is_dry_run(monkeypatch, capsys):
    seen = {}

    def fake_run(apply=False, since=None, **kw):
        seen.update(apply=apply, since=since)
        return {"mode": "apply" if apply else "dry-run"}
    monkeypatch.setattr(L, "run", fake_run)
    assert L.main([]) == 0
    assert seen["apply"] is False
    L.main(["--since", "2026-09-17"])
    assert seen["since"] == datetime(2026, 9, 17, tzinfo=ET)
    L.main(["--apply"])
    assert seen["apply"] is True
    with pytest.raises(SystemExit):
        L.main(["--apply", "--dry-run"])


def test_real_run_hits_conftest_db_guard():
    with pytest.raises(RuntimeError, match="must not touch the real database"):
        L.run(apply=False)
