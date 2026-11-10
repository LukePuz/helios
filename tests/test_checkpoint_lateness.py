"""outcome_checkpoint rows say how late the price is against its label
(audit 2026-11-07 13:07 #7). Log-only; the stored price is unchanged."""
from datetime import datetime, timezone
from unittest.mock import patch

import tasks


def test_days_late_and_target_date():
    o = {"id": 3, "approval_id": 9, "ticker": "IONQ", "trading_env": "paper",
         "entry_date": datetime(2026, 6, 22, 15, 0, tzinfo=timezone.utc)}
    with patch("db.log_activity") as act:
        tasks._log_outcome_checkpoint(o, 60, 134, 41.5)
    extra = act.call_args.args[4]
    assert extra["days_late"] == 74 and extra["target_date"] == "2026-08-21"
    assert extra["price"] == 41.5 and extra["checkpoint_days"] == 60


def test_unreadable_entry_date_gives_no_target():
    assert tasks._checkpoint_target_date(None, 30) is None
    assert tasks._checkpoint_target_date("2026-06-22", 30) is None
