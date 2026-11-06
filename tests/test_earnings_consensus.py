"""Earnings consensus snapshots (ingestion/earnings_consensus.py): parsing
through yfinance's own parser from a quoteSummary-shaped fixture (no
network), NaN handling, ETF/no-data status, append-only writes, the beat
entry, and that nothing the analyst sees reads the table."""
import json
import pathlib
from datetime import date

import numpy as np
import pytest
import yfinance
from yfinance.scrapers.analysis import Analysis

import db
from ingestion import earnings_consensus as EC

ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = date(2026, 11, 3)


def _v(raw):
    return {"raw": raw, "fmt": str(raw)}


# quoteSummary responses in Yahoo's shape (formatted=false), values made up.
EARNINGS_TREND = {"quoteSummary": {"result": [{"earningsTrend": {"trend": [
    {"period": "0q", "endDate": "2026-12-31",
     "earningsEstimate": {"avg": _v(1.25), "low": _v(1.10), "high": _v(1.40),
                          "yearAgoEps": _v(0.98), "numberOfAnalysts": _v(31), "growth": _v(0.276),
                          "earningsCurrency": "USD"},
     "revenueEstimate": {"avg": _v(41.2e9), "low": _v(39.9e9), "high": _v(43.0e9),
                         "numberOfAnalysts": _v(29), "yearAgoRevenue": _v(30.1e9), "growth": _v(0.369),
                         "revenueCurrency": "USD"}},
    # +1q has no growth or yearAgo fields: pandas fills them with NaN.
    {"period": "+1q", "endDate": "2027-03-31",
     "earningsEstimate": {"avg": _v(1.37), "low": _v(1.15), "high": _v(1.60),
                          "numberOfAnalysts": _v(27), "earningsCurrency": "USD"},
     "revenueEstimate": {"avg": _v(44.0e9), "low": _v(41.0e9), "high": _v(47.5e9),
                         "numberOfAnalysts": _v(25), "revenueCurrency": "USD"}},
    {"period": "0y", "endDate": "2026-12-31",
     "earningsEstimate": {"avg": _v(4.6), "numberOfAnalysts": _v(40)},
     "revenueEstimate": {"avg": _v(150e9), "numberOfAnalysts": _v(38)}},
]}}]}}


def _hist(q, actual, est):
    return {"quarter": {"raw": 0, "fmt": q}, "epsActual": _v(actual), "epsEstimate": _v(est),
            "epsDifference": _v(round(actual - est, 4)),
            "surprisePercent": _v(round((actual - est) / est, 4)), "currency": "USD", "period": "-1q"}


EARNINGS_HISTORY = {"quoteSummary": {"result": [{"earningsHistory": {"history": [
    _hist("2025-12-31", 0.89, 0.85), _hist("2026-03-31", 0.96, 0.93),
    _hist("2026-06-30", 1.05, 1.01), _hist("2026-09-30", 1.12, 1.09),
]}}]}}


class FakeData:
    """Stands in for yfinance's YfData: answers by quoteSummary module; None
    is what yfinance's _fetch returns after a 404 (e.g. an ETF)."""
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def get_raw_json(self, url, params=None):
        self.calls.append(params["modules"])
        return self.responses.get(params["modules"])


class FakeTicker:
    def __init__(self, responses, calendar=None):
        # The attribute name yfinance's Ticker uses (base.TickerBase).
        self._analysis = Analysis(FakeData(responses), "NVDA")
        self.calendar = calendar

    earnings_estimate = property(lambda self: self._analysis.earnings_estimate)
    revenue_estimate = property(lambda self: self._analysis.revenue_estimate)
    earnings_history = property(lambda self: self._analysis.earnings_history)


def _full_ticker():
    return FakeTicker({"earningsTrend": EARNINGS_TREND, "earningsHistory": EARNINGS_HISTORY},
                      calendar={"Earnings Date": [date(2026, 11, 19)], "Earnings Average": 1.25})


# --- parsing ---------------------------------------------------------------

def test_parses_estimates_history_and_next_date():
    rec = EC.fetch_consensus("NVDA", yf_ticker=_full_ticker(), today=TODAY)
    assert rec["status"] == "ok" and rec["error_class"] is None and rec["source"] == "yfinance"
    assert rec["parts"] == {"calendar": "ok", "eps_estimate": "ok",
                            "revenue_estimate": "ok", "eps_history": "ok",
                            "yfinance_version": yfinance.__version__}
    assert rec["next_earnings_date"] == date(2026, 11, 19)
    assert rec["next_earnings_is_estimate"] is False
    eps = rec["eps_estimate"]
    assert set(eps) == {"0q", "+1q"}  # annual periods not kept
    assert eps["0q"]["avg"] == 1.25 and eps["0q"]["low"] == 1.10 and eps["0q"]["high"] == 1.40
    assert eps["0q"]["numberOfAnalysts"] == 31 and eps["0q"]["currency"] == "USD"
    rev = rec["revenue_estimate"]
    assert rev["+1q"]["avg"] == 44.0e9 and rev["+1q"]["numberOfAnalysts"] == 25
    hist = rec["eps_history"]
    assert [h["quarter"] for h in hist] == ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"]
    assert hist[0]["epsActual"] == 1.12 and hist[0]["epsEstimate"] == 1.09
    assert hist[0]["surprisePercent"] == pytest.approx(0.0275)
    json.dumps(rec, default=str, allow_nan=False)  # JSON-safe as stored


def test_estimated_window_takes_its_start_and_is_flagged():
    t = _full_ticker()
    t.calendar = {"Earnings Date": [date(2026, 11, 18), date(2026, 11, 24)]}
    rec = EC.fetch_consensus("NVDA", yf_ticker=t, today=TODAY)
    assert rec["next_earnings_date"] == date(2026, 11, 18)
    assert rec["next_earnings_is_estimate"] is True


def test_each_quarter_row_records_its_end_date():
    rec = EC.fetch_consensus("NVDA", yf_ticker=_full_ticker(), today=TODAY)
    for part in ("eps_estimate", "revenue_estimate"):
        assert rec[part]["0q"]["end_date"] == "2026-12-31"
        assert rec[part]["+1q"]["end_date"] == "2027-03-31"


def test_end_date_missing_is_null_and_does_not_fail():
    trend = json.loads(json.dumps(EARNINGS_TREND))
    del trend["quoteSummary"]["result"][0]["earningsTrend"]["trend"][1]["endDate"]
    t = FakeTicker({"earningsTrend": trend, "earningsHistory": EARNINGS_HISTORY})
    rec = EC.fetch_consensus("NVDA", yf_ticker=t, today=TODAY)
    assert rec["eps_estimate"]["0q"]["end_date"] == "2026-12-31"
    assert rec["eps_estimate"]["+1q"]["end_date"] is None
    assert rec["status"] == "ok"
    # No trend data reachable at all (a ticker object without _analysis).
    class Bare:
        calendar = None
        earnings_estimate = property(lambda self: t._analysis.earnings_estimate)
        revenue_estimate = property(lambda self: t._analysis.revenue_estimate)
        earnings_history = property(lambda self: t._analysis.earnings_history)
    rec = EC.fetch_consensus("NVDA", yf_ticker=Bare(), today=TODAY)
    assert rec["eps_estimate"]["0q"]["end_date"] is None and rec["parts"]["eps_estimate"] == "ok"


def test_formatted_end_date_reads_its_fmt():
    class T:
        class _analysis:
            _earnings_trend = [{"period": "0q", "endDate": {"raw": 1798675200, "fmt": "2026-12-31"}},
                               {"period": "+1q", "endDate": 5}]
    assert EC._trend_end_dates(T()) == {"0q": "2026-12-31", "+1q": None}


def test_is_estimate_counts_only_future_dates():
    """m454: a past date next to one future date is a set date, not a window."""
    t = _full_ticker()
    t.calendar = {"Earnings Date": [date(2026, 10, 28), date(2026, 11, 19)]}
    rec = EC.fetch_consensus("NVDA", yf_ticker=t, today=TODAY)
    assert rec["next_earnings_date"] == date(2026, 11, 19)
    assert rec["next_earnings_is_estimate"] is False


def test_history_keeps_at_most_eight_quarters():
    hist = {"quoteSummary": {"result": [{"earningsHistory": {"history": [
        _hist(f"20{20 + i // 4}-{3 * (i % 4) + 3:02d}-28", 1.0 + i / 10, 1.0) for i in range(10)]}}]}}
    t = FakeTicker({"earningsTrend": EARNINGS_TREND, "earningsHistory": hist})
    rows = EC.fetch_consensus("NVDA", yf_ticker=t, today=TODAY)["eps_history"]
    assert len(rows) == 8 and rows[0]["quarter"] == "2022-06-28"


# --- NaN -------------------------------------------------------------------

def test_missing_fields_and_nonfinite_values_become_null():
    rec = EC.fetch_consensus("NVDA", yf_ticker=_full_ticker(), today=TODAY)
    # +1q had no growth/yearAgoEps: pandas NaN in the frame, None here.
    assert rec["eps_estimate"]["+1q"]["growth"] is None
    assert rec["eps_estimate"]["+1q"]["yearAgoEps"] is None
    for v in (float("nan"), float("inf"), np.float64("nan"), np.float64("-inf")):
        assert EC._clean(v) is None
    import pandas as pd
    assert EC._clean(pd.NaT) is None
    assert EC._clean(np.int64(7)) == 7 and type(EC._clean(np.int64(7))) is int
    assert EC._clean(pd.Timestamp("2026-09-30")) == "2026-09-30"


def test_writer_coerces_nan_left_in_a_record(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(db, "_get_conn", lambda: conn)
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    assert db.log_earnings_consensus("NVDA", {"source": "yfinance", "status": "ok",
                                              "eps_estimate": {"0q": {"avg": float("nan")}}})
    params = conn.executed[0][1]
    stored = [p.adapted for p in params if hasattr(p, "adapted")]
    assert {"0q": {"avg": None}} in stored


# --- ETF / no data / errors ------------------------------------------------

def test_a_404_ticker_is_no_data_not_error():
    # yfinance answers a 404 by logging and returning None from _fetch.
    t = FakeTicker({}, calendar={})
    rec = EC.fetch_consensus("QTUM", yf_ticker=t, today=TODAY)
    assert rec["status"] == "no_data" and rec["error_class"] is None
    assert set(v for k, v in rec["parts"].items() if k != "yfinance_version") == {"no_data"}
    assert rec["eps_estimate"] is None and rec["eps_history"] is None


def test_a_part_that_raises_is_recorded_by_class():
    class Boom(_FullTickerBase):
        @property
        def earnings_history(self):
            raise ConnectionError("reset")
    rec = EC.fetch_consensus("NVDA", yf_ticker=Boom(), today=TODAY)
    assert rec["status"] == "partial" and rec["error_class"] == "ConnectionError"
    assert rec["parts"]["eps_history"] == "error:ConnectionError"
    assert rec["eps_estimate"]["0q"]["avg"] == 1.25


class _FullTickerBase(FakeTicker):
    def __init__(self):
        super().__init__({"earningsTrend": EARNINGS_TREND, "earningsHistory": EARNINGS_HISTORY},
                         calendar={"Earnings Date": [date(2026, 11, 19)]})


def test_run_skips_etfs_records_status_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: {"DRAM", "QTUM"})
    fetched, written = [], []

    def fetch(t):
        fetched.append(t)
        if t == "BAD":
            raise TimeoutError("x")
        if t == "THIN":
            return EC.fetch_consensus(t, yf_ticker=FakeTicker({}, calendar={}), today=TODAY)
        return EC.fetch_consensus(t, yf_ticker=_full_ticker(), today=TODAY)

    caplog.set_level("INFO", logger=EC.log.name)
    counts = EC.refresh_all_earnings_consensus(
        ["NVDA", "DRAM", "QTUM", "THIN", "BAD"], fetch=fetch,
        write=lambda t, r: written.append((t, r)) or True, pause_s=0)
    assert fetched == ["NVDA", "THIN", "BAD"]  # ETFs not fetched
    status = {t: r["status"] for t, r in written}
    assert status == {"NVDA": "ok", "DRAM": "no_data", "QTUM": "no_data", "THIN": "no_data",
                      "BAD": "error"}
    assert dict(written)["DRAM"]["parts"] == {"skipped": "etf",
                                              "yfinance_version": yfinance.__version__}
    assert dict(written)["BAD"]["error_class"] == "TimeoutError"
    assert counts == {"ok": 1, "partial": 0, "no_data": 3, "error": 1, "written": 5,
                      "stopped": None}
    recs = [r for r in caplog.records if r.name == EC.log.name]
    assert [r.levelname for r in recs if r.levelno >= 30] == ["WARNING"]
    assert not any(r.levelno >= 40 for r in recs)


def test_run_with_only_expected_no_data_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: {"DRAM"})
    caplog.set_level("INFO", logger=EC.log.name)
    EC.refresh_all_earnings_consensus(
        ["DRAM", "THIN"], fetch=lambda t: EC.fetch_consensus(
            t, yf_ticker=FakeTicker({}, calendar={}), today=TODAY),
        write=lambda t, r: True, pause_s=0)
    assert not [r for r in caplog.records if r.name == EC.log.name and r.levelno >= 30]


# --- append-only -----------------------------------------------------------

class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))


class _FakeConn:
    def __init__(self):
        self.executed, self.commits = [], 0

    def cursor(self, *a, **k):
        return _Cur(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def test_two_fetches_append_two_rows_and_never_update(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(db, "_get_conn", lambda: conn)
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    for _ in range(2):
        rec = EC.fetch_consensus("NVDA", yf_ticker=_full_ticker(), today=TODAY)
        assert db.log_earnings_consensus("NVDA", rec) is True
    assert len(conn.executed) == 2 and conn.commits == 2
    for sql, params in conn.executed:
        assert sql.startswith("INSERT INTO earnings_consensus_history")
        assert "UPDATE" not in sql.upper() and "ON CONFLICT" not in sql.upper()
        assert "fetched_at" not in sql  # the insert time is the knowledge time
        assert params[0] == "NVDA"


def test_migration_79_creates_the_append_only_table():
    sql = dict(db._MIGRATIONS)[79]
    assert "CREATE TABLE IF NOT EXISTS earnings_consensus_history" in sql
    assert "fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in sql
    for col in ("ticker", "source", "status", "error_class", "parts", "next_earnings_date",
                "next_earnings_is_estimate", "eps_estimate", "revenue_estimate", "eps_history"):
        assert col in sql, col
    versions = [v for v, _ in db._MIGRATIONS]
    assert versions == sorted(versions) and len(set(versions)) == len(versions)


# --- schedule --------------------------------------------------------------

def test_beat_entry_runs_after_the_close_on_the_ingestion_queue():
    import tasks
    entry = tasks.app.conf.beat_schedule["refresh-earnings-consensus"]
    assert entry["task"] == "tasks.refresh_earnings_consensus"
    assert entry["options"]["queue"] == tasks.INGESTION_QUEUE
    sched = entry["schedule"]
    assert sched.hour == {17} and sched.minute == {30} and sched.day_of_week == {1, 2, 3, 4, 5}
    assert entry["options"]["expires"] <= 90 * 60  # before the 19:00 stack-down
    task = tasks.app.tasks["tasks.refresh_earnings_consensus"]
    assert task.soft_time_limit and task.time_limit and task.soft_time_limit < task.time_limit


# --- nothing the analyst sees reads it -------------------------------------

_ALLOWED = {"db.py", "tasks.py", "ingestion/earnings_consensus.py", "helios3d/features/store.py"}
_SKIP = ("tests", "venv", ".venv", ".git", ".claude")


def test_only_the_writer_side_and_research_reader_name_the_data():
    named = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in _SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        if "earnings_consensus" in text or "consensus_history" in text:
            named.add(rel.as_posix())
    assert named == _ALLOWED, named
    assert not any(p.startswith("agents/") or p == "context_engine.py" for p in named)


def test_shadow_panel_roles_cannot_read_it():
    from helios3d.panel import roles
    assert all("earnings_consensus" not in s for s in roles.ROLE_SOURCES.values())


# --- write failures --------------------------------------------------------

def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= 30]


def test_writer_returns_false_when_the_connection_fails(monkeypatch):
    def boom():
        raise RuntimeError("connection pool exhausted")
    monkeypatch.setattr(db, "_get_conn", boom)
    put = []
    monkeypatch.setattr(db, "_put_conn", put.append)
    errors = []
    assert db.log_earnings_consensus("NVDA", {"source": "yfinance", "status": "ok"}, errors) is False
    assert errors == ["RuntimeError: connection pool exhausted"] and put == []


class _NoTableConn(_FakeConn):
    def cursor(self, *a, **k):
        class C(_Cur):
            def execute(self, sql, params=None):
                raise RuntimeError('relation "earnings_consensus_history" does not exist')
        return C(self)


def test_missing_table_warns_once_per_run_not_per_ticker(monkeypatch, caplog):
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: set())
    monkeypatch.setattr(db, "_get_conn", lambda: _NoTableConn())
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    caplog.set_level("DEBUG")
    counts = EC.refresh_all_earnings_consensus(
        ["NVDA", "AMD", "MU"], fetch=lambda t: EC.fetch_consensus(t, yf_ticker=_full_ticker(), today=TODAY),
        pause_s=0)
    assert counts["written"] == 0 and counts["ok"] == 3
    warns = _warnings(caplog)
    assert len(warns) == 1 and "3 row(s) not written" in warns[0].getMessage()
    assert "does not exist" in warns[0].getMessage()


def test_a_write_that_raises_does_not_abort_the_run(monkeypatch, caplog):
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: set())
    seen = []

    def write(t, r):
        seen.append(t)
        if t == "AMD":
            raise ConnectionError("db blip")
        return True
    caplog.set_level("INFO")
    counts = EC.refresh_all_earnings_consensus(
        ["NVDA", "AMD", "MU"], fetch=lambda t: EC.fetch_consensus(t, yf_ticker=_full_ticker(), today=TODAY),
        write=write, pause_s=0)
    assert seen == ["NVDA", "AMD", "MU"] and counts["written"] == 2
    assert len(_warnings(caplog)) == 1


# --- rate limit ------------------------------------------------------------

def test_fetch_raises_a_rate_limit_instead_of_recording_it():
    from yfinance.exceptions import YFRateLimitError

    class Limited(_FullTickerBase):
        @property
        def earnings_estimate(self):
            raise YFRateLimitError()
    with pytest.raises(YFRateLimitError):
        EC.fetch_consensus("NVDA", yf_ticker=Limited(), today=TODAY)


def test_run_stops_at_the_first_rate_limit_and_marks_the_rest(monkeypatch, caplog):
    from yfinance.exceptions import YFRateLimitError
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: {"DRAM"})
    fetched, written = [], []

    def fetch(t):
        fetched.append(t)
        if t == "AMD":
            raise YFRateLimitError()
        return EC.fetch_consensus(t, yf_ticker=_full_ticker(), today=TODAY)
    caplog.set_level("INFO")
    counts = EC.refresh_all_earnings_consensus(
        ["NVDA", "AMD", "MU", "DRAM", "AVGO"], fetch=fetch,
        write=lambda t, r: written.append((t, r)) or True, pause_s=0)
    assert fetched == ["NVDA", "AMD"]  # Yahoo not called after the 429
    rows = dict(written)
    assert list(rows) == ["NVDA", "AMD", "MU", "DRAM", "AVGO"]
    for t in ("AMD", "MU", "AVGO"):
        assert rows[t]["status"] == "error" and rows[t]["error_class"] == "rate_limited", t
        assert rows[t]["parts"] == {"skipped": "rate_limited",
                                    "yfinance_version": yfinance.__version__}
    assert rows["DRAM"]["parts"] == {"skipped": "etf", "yfinance_version": yfinance.__version__}
    assert counts["stopped"] == "rate_limited" and counts["error"] == 3
    warns = _warnings(caplog)
    assert len(warns) == 1 and "rate_limited" in warns[0].getMessage()


# --- soft time limit and deadline ------------------------------------------

def test_soft_time_limit_in_a_part_is_raised_not_recorded():
    from celery.exceptions import SoftTimeLimitExceeded

    class Slow(_FullTickerBase):
        @property
        def earnings_history(self):
            raise SoftTimeLimitExceeded()
    with pytest.raises(SoftTimeLimitExceeded):
        EC.fetch_consensus("NVDA", yf_ticker=Slow(), today=TODAY)


@pytest.mark.parametrize("where", ["fetch", "write"])
def test_soft_time_limit_in_the_run_is_raised(monkeypatch, where):
    from celery.exceptions import SoftTimeLimitExceeded
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: set())
    calls = []

    def fetch(t):
        calls.append(t)
        if where == "fetch":
            raise SoftTimeLimitExceeded()
        return EC.fetch_consensus(t, yf_ticker=_full_ticker(), today=TODAY)

    def write(t, r):
        if where == "write":
            raise SoftTimeLimitExceeded()
        return True
    with pytest.raises(SoftTimeLimitExceeded):
        EC.refresh_all_earnings_consensus(["NVDA", "AMD"], fetch=fetch, write=write, pause_s=0)
    assert calls == ["NVDA"]


def test_run_stops_fetching_at_the_deadline(monkeypatch, caplog):
    monkeypatch.setattr("knowledge_loader.get_etf_tickers", lambda: set())
    now = [0.0]
    fetched, written = [], []

    def fetch(t):
        fetched.append(t)
        now[0] += 400  # each fetch "takes" 400 s
        return EC.fetch_consensus(t, yf_ticker=_full_ticker(), today=TODAY)
    caplog.set_level("INFO")
    counts = EC.refresh_all_earnings_consensus(
        ["A", "B", "C", "D", "E"], fetch=fetch, write=lambda t, r: written.append((t, r)) or True,
        pause_s=0, deadline_s=1000, clock=lambda: now[0])
    assert fetched == ["A", "B", "C"]
    rows = dict(written)
    assert [rows[t]["error_class"] for t in ("D", "E")] == ["deadline", "deadline"]
    assert counts["stopped"] == "deadline" and counts["written"] == 5
    assert len(_warnings(caplog)) == 1


def test_task_passes_a_deadline_before_its_soft_limit_and_fails_the_heartbeat(monkeypatch):
    import tasks
    from celery.exceptions import SoftTimeLimitExceeded
    got, fails = {}, []
    monkeypatch.setattr(EC, "refresh_all_earnings_consensus",
                        lambda **kw: got.update(kw) or {"written": 1})
    monkeypatch.setattr(tasks, "_heartbeat_ok", lambda s, c: None)
    tasks.refresh_earnings_consensus.run()
    task = tasks.app.tasks["tasks.refresh_earnings_consensus"]
    assert 0 < got["deadline_s"] < task.soft_time_limit

    def limited(**kw):
        raise SoftTimeLimitExceeded()
    monkeypatch.setattr(EC, "refresh_all_earnings_consensus", limited)
    monkeypatch.setattr(tasks, "_heartbeat_fail", lambda s, e: fails.append((s, e)))
    tasks.refresh_earnings_consensus.run()
    assert fails and fails[0][0] == "earnings_consensus"
