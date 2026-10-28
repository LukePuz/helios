"""Shadow catalyst-date check in the decision record (audit 2026-10-28 14:53, #5)."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

from agents import analyst


def _check(raw, nxt=None):
    with patch("db.get_next_earnings", return_value=nxt):
        return analyst._catalyst_check("NVDA", {"catalyst": {"date": raw}})


def test_past_date_flagged():
    past = (datetime.now(timezone.utc).date() - timedelta(days=3)).isoformat()
    c = _check(past)
    assert c["past_or_today"] is True and c["month_only"] is False


def test_month_only_flagged_and_compared():
    c = _check("2026-11", nxt=date(2026, 11, 19))
    assert c["month_only"] is True and c["days_model_vs_calendar"] == -4


def test_missing_date_and_db_error_never_raise():
    with patch("db.get_next_earnings", side_effect=RuntimeError("db")):
        c = analyst._catalyst_check("NVDA", {})
    assert c["model_date"] is None and "error" in c
