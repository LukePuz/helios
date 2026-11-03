"""An approval carries its verdict's signals.id, a test approval is tagged,
and a failed approval write is spilled (M174)."""
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


def _conn(side_effects=None, fetch=(41,)):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.execute.side_effect = side_effects
    cur.fetchone.return_value = fetch
    return conn, cur


_KW = dict(ticker="NVDA", action="OPEN", conviction=7, bucket="Accelerator", size_pct=0.05,
           stop_price=150.0, entry_strategy="55% initial", reasoning="r",
           risk_json={"shares_initial": 1, "initial_dollar": 180.0, "cash_available": 999.0})


def test_migration_77_adds_both_columns_idempotently():
    versions = [v for v, _ in db._MIGRATIONS]
    assert len(versions) == len(set(versions)) and versions == sorted(versions)
    sql = dict(db._MIGRATIONS)[77]
    assert "pending_approvals" in sql
    assert "ADD COLUMN IF NOT EXISTS signal_id INTEGER" in sql
    assert "ADD COLUMN IF NOT EXISTS is_test BOOLEAN" in sql
    assert "DEFAULT" not in sql      # not backfilled: older test rows aren't tagged


def test_every_migration_version_is_unique():
    """Two entries with one number: init_db records the version after the
    first and silently never runs the second."""
    from collections import Counter
    dupes = [v for v, n in Counter(v for v, _ in db._MIGRATIONS).items() if n > 1]
    assert dupes == []


def test_log_signal_returns_the_new_id():
    conn, cur = _conn(fetch=(123,))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_signal("NVDA", "OPEN", 0.7, "r", provenance={"a": 1}) == 123
    assert "RETURNING id" in cur.execute.call_args.args[0]


def test_log_signal_fallback_returns_none(spill):
    conn, _ = _conn([psycopg2.errors.UndefinedColumn("no prompt"), None])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_signal("NVDA", "OPEN", 0.7, "r") is None


def test_approval_gets_the_signal_id_from_the_verdict_provenance():
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_pending_approval(**_KW, analyst_json={"provenance": {"signal_id": 77}}) == 41
    sql, params = cur.execute.call_args.args
    assert "signal_id, is_test" in sql and params[-2:] == (77, False)


def test_test_approval_is_tagged_and_explicit_id_wins():
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_pending_approval(**_KW, analyst_json={"provenance": {"signal_id": 77}},
                                signal_id=5, is_test=True)
    assert cur.execute.call_args.args[1][-2:] == (5, True)


def test_approval_without_migration_77_still_writes():
    conn, cur = _conn([psycopg2.errors.UndefinedColumn("signal_id"), None])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_pending_approval(**_KW, analyst_json={}) == 41
    assert "signal_id" not in cur.execute.call_args.args[0]


def test_failed_approval_write_is_spilled_and_alerted(spill):
    conn, _ = _conn(RuntimeError("db down"))
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("alerts._send") as send:
        assert db.log_pending_approval(**_KW, analyst_json={"provenance": {"signal_id": 9}}) == -1
    row = json.loads(spill.read_text())
    assert row["writer"] == "log_pending_approval"
    assert row["record"]["signal_id"] == 9 and row["record"]["ticker"] == "NVDA"
    # The writer's argument names (scripts/replay_spill.py), plus the new fields.
    assert {"bucket", "size_pct", "stop_price", "expires_at", "is_test"} <= set(row["record"])
    assert row["record"]["is_test"] is False
    assert "cash_available" not in row["record"]["risk_json"]       # no account figures
    assert row["record"]["risk_json"]["initial_dollar"] == 180.0
    send.assert_called_once()


def test_no_connection_is_spilled_too(spill):
    with patch("db._get_conn", side_effect=RuntimeError("pool down")), patch("alerts._send"):
        assert db.log_pending_approval(**_KW, analyst_json={}) == -1
    assert json.loads(spill.read_text())["writer"] == "log_pending_approval"


def test_verdict_result_carries_its_signal_id():
    from contextlib import ExitStack
    from agents import analyst
    body = json.dumps({"action": "WATCH", "conviction": 5, "reasoning": "r", "sources": []})
    with ExitStack() as stack:
        for p in (patch("agents.analyst._call_ollama", return_value=body),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1}),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.log_signal", return_value=314), patch("feed.emit")):
            stack.enter_context(p)
        out = analyst.analyze({"ticker": "MSFT", "role": "Compounder"}, situation_brief="b",
                              regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})
    assert out["provenance"]["signal_id"] == 314


def test_fire_test_tags_its_approval():
    from pathlib import Path
    src = (Path(db.__file__).parent / "scripts" / "fire_test.py").read_text()
    assert "is_test       = True" in src
