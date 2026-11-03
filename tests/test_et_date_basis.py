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
    query = MagicMock(return_value=[])
    with patch.object(_dtmod, "datetime", _frozen(now)), \
         patch.object(tasks, "_get_client", return_value=client), \
         patch.object(tasks, "_get_stale_ingestion_sources", return_value=[]), \
         patch("db.get_portfolio_history", return_value=history), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_pending_approvals", return_value=[]), \
         patch("db.get_state", return_value="false"), \
         patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
         patch("catalyst_service.client.query_events", query), \
         patch("alerts.send_daily_digest", digest):
        tasks.send_daily_digest.run()
    assert digest.call_args.kwargs["day_pl"] == 100.0
    assert query.call_args.kwargs == {"start": "2026-11-03", "end": "2026-11-03"}
