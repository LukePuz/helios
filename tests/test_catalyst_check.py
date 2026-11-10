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


def test_raw_days_and_the_catalyst_upside_are_recorded():
    """Audit 11-06 09:31 #4 / G31: the inputs for a catalyst-horizon rate."""
    past = (datetime.now(timezone.utc).date() - timedelta(days=10)).isoformat()
    with patch("db.get_next_earnings", return_value=None):
        c = analyst._catalyst_check("NVDA", {"catalyst": {"date": past,
                                                          "expected_upside_if_positive": "+8 to +15%"}})
    assert c["days_to_catalyst_raw"] in (-10, -9)        # ET vs UTC date at the boundary
    assert (c["catalyst_upside_low_pct"], c["catalyst_upside_high_pct"]) == (8.0, 15.0)


def test_upside_range_forms():
    assert analyst._upside_range("+10%") == (10.0, 10.0)
    assert analyst._upside_range("8-12%") == (8.0, 12.0)
    assert analyst._upside_range("-5 to +10%") == (-5.0, 10.0)
    assert analyst._upside_range("Unknown") == (None, None)
    assert analyst._upside_range("+5%, falls 3-4% if miss") == (5.0, 5.0)   # professor 59i
    assert analyst._upside_range("10%-15%") == (10.0, 15.0)
    assert analyst._upside_range("Q3 2026 to +12%") == (12.0, 12.0)            # SEF 138
    assert analyst._upside_range("+100 to +150%") == (100.0, 150.0)           # professor 59j
    assert analyst._upside_range("Q4 to +12%") == (12.0, 12.0)                # SEF 138b
    assert analyst._upside_range("+5% near term, 10-20% on a beat") == (5.0, 5.0)
    assert analyst._upside_range(None) == (None, None)
