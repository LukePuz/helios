"""scripts/replay_spill.py: dry run by default, writes only with --apply,
idempotent through its ledger and a natural-key check."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

import db

_spec = importlib.util.spec_from_file_location(
    "replay_spill", Path(__file__).resolve().parent.parent / "scripts" / "replay_spill.py")
replay_spill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_spill)


class FakeCur:
    """Records statements; SELECT 1 finds a row when its first key value is
    in `existing`; INSERT ... RETURNING id hands out ids."""
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.inserts: list[str] = []
        self._one = None
        self._next = 100

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if sql.startswith("SELECT 1"):
            self._one = (1,) if params and params[0] in self.existing else None
        elif sql.startswith("INSERT"):
            table = sql.split()[2]
            self.inserts.append(table)
            if "RETURNING id" in sql:
                self._next += 1
                self._one = (self._next,)
                self.existing.add(params[0])

    def fetchone(self):
        return self._one


def _conn(cur):
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


def _write_spill(tmp_path, monkeypatch, lines):
    monkeypatch.setattr(db, "TRADING_ENV", "live")
    for writer, rec in lines:
        db._spill(writer, rec, "could not translate host name")
    return db._SPILL_PATH


def test_spill_lines_carry_uuid_kind_and_env(tmp_path, monkeypatch):
    path = _write_spill(tmp_path, monkeypatch, [("log_trade", {"symbol": "NVDA"})])
    line = json.loads(path.read_text())
    assert len(line["uuid"]) == 32 and line["kind"] == "trade" and line["env"] == "live"


def _lines():
    return [
        ("log_trade", {"symbol": "NVDA", "side": "BUY", "quantity": 3, "price": 100.0,
                       "reason": "r", "price_source": "fill"}),
        ("log_regime", {"regime": "Neutral", "score": 0.1, "confidence": 0.5, "signals": {},
                        "portfolio_impact": {}, "reasoning": "x", "regime_type": "overall",
                        "sector_etf": None}),
        ("log_order", {"ticker": "NVDA", "side": "BUY", "qty": 3, "broker_order_id": 7}),
        ("log_activity", {"stage": "halt", "ticker": "", "message": "AMD halted", "sentiment": "alert",
                          "extra": None}),
    ]


def _run(cur, args, capsys):
    with patch("db._get_conn", return_value=_conn(cur)), patch("db._put_conn"):
        assert replay_spill.main(args) == 0
    return capsys.readouterr().out


def test_dry_run_reports_counts_and_writes_nothing(tmp_path, monkeypatch, capsys):
    path = _write_spill(tmp_path, monkeypatch, _lines())
    cur = FakeCur(existing={"halt"})   # the activity row is already in the DB
    out = _run(cur, ["--spill", str(path)], capsys)
    assert cur.inserts == []
    assert "log_trade: 1 total, missing 1" in out
    assert "log_order: 1 total, report_only 1" in out
    assert "log_activity: 1 total, exists 1" in out
    assert "would be written" in out and "NVDA" not in out      # no record values printed
    assert not path.with_name(path.name + ".replayed").exists()


def test_apply_writes_missing_rows_once(tmp_path, monkeypatch, capsys):
    path = _write_spill(tmp_path, monkeypatch, _lines())
    cur = FakeCur()
    _run(cur, ["--spill", str(path), "--apply"], capsys)
    # trade, regime and activity replayed (each with an audit row); the order is not
    assert cur.inserts.count("trades") == 1 and cur.inserts.count("market_regime_log") == 1
    assert "orders" not in cur.inserts
    assert cur.inserts.count("activity_feed") == 1 + 3      # the replayed row + 3 audit rows
    ledger = path.with_name(path.name + ".replayed").read_text().split()
    assert len(ledger) == 3
    cur2 = FakeCur()
    out = _run(cur2, ["--spill", str(path), "--apply"], capsys)
    assert cur2.inserts == [] and "applied: 0 of 0" in out


def test_old_lines_without_env_are_not_replayed(tmp_path, capsys):
    path = tmp_path / "spill.jsonl"
    path.write_text(json.dumps({"at": "2026-11-01T14:00:00+00:00", "writer": "log_trade",
                                "error": "x", "record": {"symbol": "NVDA", "side": "SELL"}}) + "\n")
    cur = FakeCur()
    out = _run(cur, ["--spill", str(path), "--apply"], capsys)
    assert "missing_env" in out and cur.inserts == []
    assert "sha256:" in out                     # stable id for a pre-uuid line


def test_unparsable_line_is_counted_not_fatal(tmp_path, capsys):
    path = tmp_path / "spill.jsonl"
    path.write_text("{not json\n")
    out = _run(FakeCur(), ["--spill", str(path)], capsys)
    assert "unparsable" in out


def test_transient_connect_errors_are_retried_before_spilling():
    err = psycopg2.OperationalError('could not translate host name "db" to address')
    conn = MagicMock()
    with patch("db._get_conn", side_effect=[err, err, conn]) as get, patch("db._put_conn"):
        db.log_trade("NVDA", "BUY", 1, 100.0, "r")
    assert get.call_count == 3 and not db._SPILL_PATH.exists()


def test_transient_retries_are_bounded(monkeypatch):
    err = psycopg2.OperationalError("connection refused")
    with patch("db._get_conn", side_effect=err) as get, patch("db._put_conn"), patch("alerts._send"):
        db.log_trade("NVDA", "BUY", 1, 100.0, "r")
    assert get.call_count == 1 + len(db._CONNECT_RETRY_DELAYS)
    assert json.loads(db._SPILL_PATH.read_text())["writer"] == "log_trade"


def test_non_transient_errors_spill_without_retry():
    with patch("db._get_conn", side_effect=RuntimeError("pool exhausted")) as get, \
         patch("db._put_conn"), patch("alerts._send"):
        db.log_trade("NVDA", "BUY", 1, 100.0, "r")
    assert get.call_count == 1


@pytest.mark.parametrize("msg, transient", [
    ('could not translate host name "x"', True), ("Connection refused", True),
    ("connection reset by peer", True), ("syntax error at or near", False),
])
def test_transient_classifier(msg, transient):
    assert db._is_transient_connect_error(Exception(msg)) is transient
