"""Failed verdict/trade writes are spilled and alerted, and the missing-column
fallback keeps env or marks the row degraded (professor M40/C11, 2026-10-28)."""
import json
from unittest.mock import MagicMock, patch

import psycopg2.errors
import pytest

import db


@pytest.fixture
def spill(tmp_path, monkeypatch):
    path = tmp_path / "spill.jsonl"
    monkeypatch.setattr(db, "_SPILL_PATH", path)
    monkeypatch.setattr(db, "_LAST_WRITE_ALERT", {})
    return path


def _conn(side_effects):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.execute.side_effect = side_effects
    return conn, cur


def test_log_trade_failure_spills_and_alerts_once(spill):
    conn, _ = _conn(RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("alerts._send") as send:
        db.log_trade("NVDA", "SELL", 3, 120.0, "x")
        db.log_trade("NVDA", "SELL", 3, 120.0, "x")
    rows = [json.loads(l) for l in spill.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["writer"] == "log_trade"
    assert rows[0]["record"]["quantity"] == 3
    send.assert_called_once()  # throttled


def test_signal_fallback_keeps_env(spill):
    conn, cur = _conn([psycopg2.errors.UndefinedColumn("no prompt"), None])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_signal("NVDA", "HOLD", 0.6, "r", model="m", provenance={"code_sha": "x"},
                      decision={"a": 1}, prompt="secret brief")
    sql = cur.execute.call_args.args[0]
    assert "trading_env" in sql and "prompt" not in sql
    rec = json.loads(spill.read_text())["record"]
    assert rec["decision"] == {"a": 1} and "prompt" not in rec  # prompt never spilled


def test_signal_bare_fallback_is_marked_degraded(spill):
    conn, cur = _conn([psycopg2.errors.UndefinedColumn("x"), psycopg2.errors.UndefinedColumn("y"), None])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_signal("NVDA", "HOLD", 0.6, "reason")
    params = cur.execute.call_args.args[1]
    assert params[3] == db.DEGRADED_SIGNAL_PREFIX + "reason"


def test_env_filter_excludes_degraded_rows():
    conn, cur = _conn(None)
    cur.fetchone.return_value = None
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.get_latest_signal_for("NVDA")
    sql, params = cur.execute.call_args.args
    assert "NOT LIKE %s" in sql and params[-1] == db.DEGRADED_SIGNAL_PREFIX + "%"


@pytest.mark.parametrize("call, writer", [
    (lambda: db.log_trade("GOOG", "BUY", 1, 300.0, "x"), "log_trade"),
    (lambda: db.log_signal("GOOG", "SCREENED_OUT", 0.0, "roundup"), "log_signal"),
    (lambda: db.log_order("GOOG", "BUY", "MARKET", 1, status="unconfirmed"), "log_order"),
])
def test_connection_failure_spills_instead_of_raising(spill, call, writer):
    """Limitations audit 18:53 #8: a DNS failure getting the connection
    (GOOG/CRWV SCREENED_OUT, 17:20) raised past the spill and lost the row."""
    err = psycopg2.OperationalError('could not translate host name "db" to address')
    with patch("db._get_conn", side_effect=err), patch("db._put_conn") as put, \
         patch("alerts._send"):
        result = call()
    rows = [json.loads(l) for l in spill.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["writer"] == writer
    assert rows[0]["record"].get("symbol", rows[0]["record"].get("ticker")) == "GOOG"
    put.assert_not_called()
    if writer == "log_order":
        assert result == -1


_DNS = 'could not translate host name "db" to address'


def test_pending_approval_connect_failure_spills_and_returns_minus_one(spill):
    with patch("db._get_conn", side_effect=psycopg2.OperationalError(_DNS)), \
         patch("db._put_conn") as put, patch("alerts._send"):
        got = db.log_pending_approval("NVDA", "OPEN", 7, "growth", 0.03, 95.0, "s", "r",
                                      {"action": "OPEN"}, {"shares_initial": 3})
    assert got == -1
    put.assert_not_called()
    row = json.loads(spill.read_text())
    assert row["writer"] == "log_pending_approval" and row["record"]["ticker"] == "NVDA"


def test_pending_approval_insert_failure_spills(spill):
    conn, _ = _conn(RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), patch("alerts._send"):
        got = db.log_pending_approval("NVDA", "OPEN", 7, "growth", 0.03, 95.0, "s", "r", {}, {})
    assert got == -1 and json.loads(spill.read_text())["writer"] == "log_pending_approval"


def test_log_activity_connect_failure_spills_and_returns_false(spill):
    with patch("db._get_conn", side_effect=psycopg2.OperationalError(_DNS)), \
         patch("db._put_conn"), patch("alerts._send"):
        assert db.log_activity("halt", "", "m", extra={"event": "halt"}) is False
    row = json.loads(spill.read_text())
    assert row["writer"] == "log_activity" and row["record"]["extra"] == {"event": "halt"}


def test_log_activity_returns_true_on_success(spill):
    conn, cur = _conn(None)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_activity("halt", "NVDA", "m") is True
    assert "activity_feed" in cur.execute.call_args.args[0] and not spill.exists()


def test_log_activity_insert_failure_returns_false(spill):
    conn, _ = _conn(RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), patch("alerts._send"):
        assert db.log_activity("halt", "NVDA", "m") is False
    assert json.loads(spill.read_text())["writer"] == "log_activity"


def test_log_regime_connect_failure_spills(spill):
    err = psycopg2.OperationalError("pool exhausted")
    with patch("db._get_conn", side_effect=err), patch("db._put_conn"), patch("alerts._send"):
        db.log_regime("Neutral", 0.1, 0.5, {"vix": 17}, {}, "r", regime_type="gics_sector", sector_etf="XLK")
    row = json.loads(spill.read_text())
    assert row["writer"] == "log_regime" and row["record"]["sector_etf"] == "XLK"


def test_log_regime_insert_failure_spills_and_keeps_the_log_line(spill, caplog):
    import logging
    conn, _ = _conn(RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), patch("alerts._send"):
        db.log_regime("Bull", 0.6, 0.7, {}, {}, "r")
    row = json.loads(spill.read_text())
    assert row["writer"] == "log_regime" and row["record"]["regime"] == "Bull"
    assert any("log_regime failed" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
