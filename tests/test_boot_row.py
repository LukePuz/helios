"""A boot row in activity_feed at process start (M181, m248)."""
import json
from unittest.mock import MagicMock, patch

import db
from agents import analyst


def test_boot_record_fields(tmp_path, monkeypatch):
    rec = analyst.boot_record("celery:llm")
    assert rec["service"] == "celery:llm" and isinstance(rec["pid"], int)
    assert rec["code_sha"] == "test" and rec["head_sha"] == "testhead" and rec["dirty"] is False
    assert rec["process_started_at"] and rec["code_read_at"]
    assert len(rec["venv_sha"]) == 12 and rec["venv_packages"] > 10
    assert "sp500_cache_sha" in rec
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    assert analyst.boot_record("celery:llm")["db_port"] == "6543"
    monkeypatch.delenv("SUPABASE_PORT")
    assert analyst.boot_record("celery:llm")["db_port"] == "5432"


def test_boot_record_says_whether_the_spill_path_is_the_repo_default(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.setattr(db, "_SPILL_PATH", tmp_path / "spill.jsonl")   # an override
    assert analyst.boot_record("x")["spill_path_default"] is False
    monkeypatch.setattr(db, "_SPILL_PATH",
                        Path(db.__file__).parent / "logs" / "db_write_spill.jsonl")
    assert analyst.boot_record("x")["spill_path_default"] is True


def test_dirty_and_cache_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(analyst, "_CODE_SHA", {"code_sha": "abc-dirty", "head_sha": "abc-dirty"})
    assert analyst.boot_record("x")["dirty"] is True
    f = tmp_path / "c.json"
    f.write_text('{"tickers": ["A"]}')
    assert len(analyst._file_sha(f)) == 12
    assert analyst._file_sha(tmp_path / "missing.json") is None


def test_venv_hash_is_stable_and_order_free():
    assert analyst._venv_record() == analyst._venv_record()


def test_record_boot_writes_a_boot_stage_row():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    with patch("db.psycopg2.connect", return_value=conn) as connect:
        rec = analyst.record_boot("celery:ops")
    sql, params = cur.execute.call_args.args
    assert "INSERT INTO activity_feed" in sql
    assert params[0] == "boot" and len(params[0]) <= 20
    extra = json.loads(params[4])
    assert extra["service"] == "celery:ops" and extra["code_sha"] == rec["code_sha"]
    assert "env" in extra
    conn.commit.assert_called_once()
    conn.close.assert_called_once()
    assert connect.call_count == 1
    kw = connect.call_args.kwargs
    assert "options" not in kw   # a pooler can refuse startup options
    assert cur.execute.call_args_list[0].args[0] == "SET LOCAL statement_timeout = 5000"
    assert kw["keepalives"] == 1 and kw["tcp_user_timeout"] == 10000 and kw["connect_timeout"] == 5


def test_boot_row_failure_never_raises():
    # conftest blocks psycopg2.connect: the write fails quietly.
    assert db.log_boot_activity("m", {"a": 1}) is False
    assert analyst.record_boot("celery:ops")["service"] == "celery:ops"


def test_worker_and_beat_boot_hooks_record_a_row():
    import tasks
    sender = MagicMock(hostname="llm@somehost")
    with patch("agents.analyst.record_boot") as rb:
        tasks._pin_code_stamp(sender=sender)
        tasks._record_beat_boot()
    assert [c.args[0] for c in rb.call_args_list] == ["celery:llm", "celery:beat"]


def test_verdict_provenance_names_the_process():
    with patch.object(analyst, "_code_mixed_check", return_value={}):
        p = analyst._provenance("p")["process"]
    assert p["code_read_pid"] == analyst._BOOT["pid"] and p["code_read_at"] == analyst._BOOT["code_read_at"]


def test_boot_row_shows_a_rejected_track_record_bound():
    from unittest.mock import patch
    from agents import analyst
    with patch("config.TRACK_RECORD_SINCE", "2027-13-01", create=True):
        assert analyst.boot_record("t")["track_record_since"] == {
            "set": "2027-13-01", "applied": None, "rejected": True}
    with patch("config.TRACK_RECORD_SINCE", "2027-01-01", create=True):
        assert analyst.boot_record("t")["track_record_since"]["rejected"] is False
    with patch("config.TRACK_RECORD_SINCE", "", create=True):
        assert analyst.boot_record("t")["track_record_since"] == {
            "set": None, "applied": None, "rejected": False}


def test_a_failed_boot_row_is_spilled_for_replay():
    """Professor m614: it returned False with nothing to replay."""
    from unittest.mock import patch
    with patch("db._spill") as spill:
        assert db.log_boot_activity("m", {"a": 1}) is False
    writer, rec, err = spill.call_args.args
    assert writer == "log_activity" and rec["stage"] == "boot" and rec["ticker"] == ""
    assert rec["message"] == "m" and rec["extra"]["a"] == 1 and rec["extra"]["env"] == db.TRADING_ENV
    assert err
