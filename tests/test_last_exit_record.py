"""Record-only last_exit on buy-side verdicts (audit 2026-11-10 10:53 #7)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from agents import analyst


NOW = datetime(2026, 11, 5, 16, 8, tzinfo=timezone.utc)


def _patch(monkeypatch, fn):
    import db
    monkeypatch.setattr(db, "last_closed_outcome", fn)


def test_recent_exit_reports_hours_and_outcome(monkeypatch):
    exit_at = NOW - timedelta(minutes=98)
    _patch(monkeypatch, lambda t: {"id": 7, "entry_date": NOW - timedelta(days=9), "exit_at": exit_at,
                                   "outcome_pct": -0.093, "exit_source": "broker_exit"})
    rec = analyst._last_exit_record("NVDA", now=NOW)
    assert rec == {"exited": True, "outcome_id": 7, "exit_at": exit_at.isoformat(),
                   "hours_since_exit": pytest.approx(1.63, abs=0.01), "outcome_pct": -0.093,
                   "exit_source": "broker_exit"}
    json.dumps(rec)


def test_naive_exit_time_is_read_as_utc(monkeypatch):
    naive = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    _patch(monkeypatch, lambda t: {"id": 1, "entry_date": None, "exit_at": naive,
                                   "outcome_pct": None, "exit_source": None})
    rec = analyst._last_exit_record("ANET", now=NOW)
    assert rec["hours_since_exit"] == 2.0 and rec["outcome_pct"] is None


def test_never_exited(monkeypatch):
    _patch(monkeypatch, lambda t: None)
    assert analyst._last_exit_record("AMD", now=NOW) == {"exited": False}


def test_read_error_is_recorded_not_raised(monkeypatch):
    def boom(t):
        raise RuntimeError("db down")
    _patch(monkeypatch, boom)
    assert analyst._last_exit_record("AMD", now=NOW) == {"error": "RuntimeError"}


def test_last_closed_outcome_query_shape(monkeypatch):
    import db
    seen = {}

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params): seen["sql"], seen["params"] = sql, params
        def fetchone(self): return (3, "e", "x", 0.05, "broker_exit")

    class Conn:
        def cursor(self): return Cur()

    monkeypatch.setattr(db, "_get_conn", lambda: Conn())
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    out = db.last_closed_outcome("NVDA")
    assert out == {"id": 3, "entry_date": "e", "exit_at": "x", "outcome_pct": 0.05,
                   "exit_source": "broker_exit"}
    assert "exit_date IS NOT NULL" in seen["sql"] and "trading_env=%s" in seen["sql"]
    assert "ORDER BY COALESCE(exit_fill_at, exit_date) DESC" in seen["sql"]
    assert seen["params"][0] == "NVDA"
