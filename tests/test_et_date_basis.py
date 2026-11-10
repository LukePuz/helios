"""The digest's day-P&L baseline and tomorrow's catalyst date use ET calendar
dates on both sides, not a mix of ET and UTC."""
import datetime as _dtmod
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import tasks
import yfinance  # noqa: F401  (imported before datetime is patched)
import catalyst_service.client  # noqa: F401


def _frozen(at):
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz) if tz else at.replace(tzinfo=None)
    return _DT


def test_digest_baseline_is_the_first_snapshot_of_the_et_day():
    # 2026-11-03 02:00 UTC = 11-02 21:00 EST. The 23:30 UTC 11-02 snapshot
    # (18:30 EST) is today's first; a 00:30 UTC 11-03 snapshot is also 11-02 ET.
    now = datetime(2026, 11, 3, 2, 0, tzinfo=timezone.utc)
    history = [
        {"timestamp": datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc), "total_value": 900.0},   # 11-01 ET
        {"timestamp": datetime(2026, 11, 2, 23, 30, tzinfo=timezone.utc), "total_value": 1000.0},  # 11-02 ET
        {"timestamp": datetime(2026, 11, 3, 0, 30, tzinfo=timezone.utc), "total_value": 1010.0},   # 11-02 ET
    ]
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1100.0)
    client.get_all_positions.return_value = []
    digest = MagicMock()
    vsum = MagicMock(return_value={})
    query = MagicMock(return_value=[])
    with patch.object(_dtmod, "datetime", _frozen(now)), \
         patch.object(tasks, "_get_client", return_value=client), \
         patch.object(tasks, "_get_stale_ingestion_sources", return_value=[]), \
         patch("db.get_portfolio_history", return_value=history), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_verdict_day_summary", vsum), \
         patch("db.get_pending_approvals", return_value=[]), \
         patch("db.get_state", return_value="false"), \
         patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
         patch("catalyst_service.client.query_events", query), \
         patch("alerts.send_daily_digest", digest):
        tasks.send_daily_digest.run()
    assert digest.call_args.kwargs["day_pl"] == 100.0
    assert query.call_args.kwargs == {"start": "2026-11-03", "end": "2026-11-03"}


def test_friday_digest_covers_through_monday_with_held_names_first():
    """Friday 16:15 ET: the window runs Saturday through Monday's session,
    held names first (audit 11-06 17:34 #8)."""
    now = datetime(2026, 11, 6, 21, 15, tzinfo=timezone.utc)   # Fri 16:15 EST
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1100.0)
    pos = MagicMock(symbol="NVDA", qty=1, unrealized_pl=0, avg_entry_price=100.0, current_price=100.0)
    client.get_all_positions.return_value = [pos]
    digest = MagicMock()
    vsum = MagicMock(return_value={"n": 3, "by_signal": {"HOLD": 2, "WATCH": 1},
                                   "last": {"signal": "HOLD", "conviction": 7}})
    query = MagicMock(return_value=[{"ticker": "AAPL", "event_type": "earnings", "event_date": "2026-11-09"},
                                    {"ticker": "NVDA", "event_type": "earnings", "event_date": "2026-11-09"}])
    with patch.object(_dtmod, "datetime", _frozen(now)), \
         patch.object(tasks, "_get_client", return_value=client), \
         patch.object(tasks, "_get_stale_ingestion_sources", return_value=[]), \
         patch("db.get_portfolio_history", return_value=[]), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_verdict_day_summary", vsum), \
         patch("db.get_pending_approvals", return_value=[]), \
         patch("db.get_state", return_value="false"), \
         patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
         patch("catalyst_service.client.query_events", query), \
         patch("alerts.send_daily_digest", digest):
        tasks.send_daily_digest.run()
    assert query.call_args.kwargs == {"start": "2026-11-07", "end": "2026-11-09"}
    kw = digest.call_args.kwargs
    assert kw["catalysts_text"][0] == "2026-11-09 NVDA: earnings (held)"
    assert kw["catalysts_label"].startswith("Catalysts through the next session (Mon Nov 09")
    assert "📋 Today: 3 verdicts (HOLD 2, WATCH 1); last HOLD c7" in kw["positions_text"][0]
    assert vsum.call_args.args[1] == datetime(2026, 11, 6, 5, 0, tzinfo=timezone.utc)   # ET midnight


def test_digest_says_how_many_catalysts_it_left_out():
    import alerts
    with patch("alerts._send") as send:
        alerts.send_daily_digest(portfolio_value=1.0, day_pl=0.0, day_pl_pct=0.0, spy_pl_pct=None,
                                 positions_text=[], catalysts_text=[f"T{i}: earnings" for i in range(13)],
                                 pending_count=0, halted=False, ingestion_issues=[])
    text = send.call_args.args[0]
    assert "T9: earnings" in text and "T10: earnings" not in text and "+3 more" in text


def test_verdict_day_summary_counts_and_last():
    from contextlib import contextmanager
    import db
    cur = MagicMock()
    cur.fetchall.return_value = [(datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc), "HOLD", 0.7),
                                 (datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc), "HOLD", 0.6),
                                 (datetime(2026, 11, 6, 14, 0, tzinfo=timezone.utc), "WATCH", 0.5)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.get_verdict_day_summary("NVDA", datetime(2026, 11, 6, 5, 0, tzinfo=timezone.utc))
    assert out == {"n": 3, "by_signal": {"HOLD": 2, "WATCH": 1},
                   "last": {"timestamp": datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc),
                            "signal": "HOLD", "conviction": 7}}
    assert "timestamp >= %s" in cur.execute.call_args.args[0]
