"""Shadow prior-verdict record on buy verdicts (audit 2026-10-28 14:53, #1b)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import db
from agents import analyst


def test_prior_verdicts_shape():
    rows = [{"timestamp": datetime(2026, 10, 28, 14, 0, tzinfo=timezone.utc), "signal": "WATCH",
             "conviction": 6, "udr": 1.4}]
    with patch("db.get_recent_verdicts", return_value=rows):
        assert analyst._prior_verdicts("MSFT") == [
            {"t": "2026-10-28 14:00", "signal": "WATCH", "c": 6, "udr": 1.4}]


def test_prior_verdicts_never_raises():
    with patch("db.get_recent_verdicts", side_effect=RuntimeError("db")):
        assert analyst._prior_verdicts("MSFT") is None


def test_reader_sql_filters_env_window_and_degraded():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [(datetime(2026, 10, 28, tzinfo=timezone.utc), "HOLD", 0.6, "bad")]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.get_recent_verdicts("NVDA", hours=24)
    sql, params = cur.execute.call_args.args
    assert "make_interval(hours => %s)" in sql and "NOT LIKE %s" in sql and 24 in params
    assert out[0]["conviction"] == 6 and out[0]["udr"] is None
