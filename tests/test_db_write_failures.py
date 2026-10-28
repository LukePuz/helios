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
