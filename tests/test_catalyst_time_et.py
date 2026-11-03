"""weeks_to_catalyst_et: the catalyst date read as ET midnight, log-only,
recorded beside the gate's UTC-midnight figure."""
from datetime import datetime, timezone
from unittest.mock import patch

from agents import analyst
from agents.catalyst_time import today_et, weeks_to_catalyst_et

_EST_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)    # 07:00 EST
_EDT_NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)   # 08:00 EDT


def test_et_midnight_is_five_hours_later_under_est():
    utc = analyst.compute_weeks_to_catalyst("2026-11-09", as_of=_EST_NOW)
    et = weeks_to_catalyst_et("2026-11-09", as_of=_EST_NOW)
    assert abs((et - utc) * 7 * 24 - 5) < 1e-9


def test_et_midnight_is_four_hours_later_under_edt():
    utc = analyst.compute_weeks_to_catalyst("2026-10-30", as_of=_EDT_NOW)
    et = weeks_to_catalyst_et("2026-10-30", as_of=_EDT_NOW)
    assert abs((et - utc) * 7 * 24 - 4) < 1e-9


def test_same_parsing_rules_as_the_gate():
    for raw in (None, "", "ongoing", "Ongoing", "sometime"):
        assert weeks_to_catalyst_et(raw, as_of=_EST_NOW) is None
    assert weeks_to_catalyst_et("2026-10-01", as_of=_EST_NOW) == 1 / 7      # 1-day floor
    mid = weeks_to_catalyst_et("2026-12", as_of=_EST_NOW, month_midpoint=True)
    first = weeks_to_catalyst_et("2026-12", as_of=_EST_NOW, month_midpoint=False)
    assert abs((mid - first) * 7 - 14) < 1e-9


def test_today_et_rolls_at_et_midnight():
    assert str(today_et(datetime(2026, 11, 3, 3, 0, tzinfo=timezone.utc))) == "2026-11-02"
    assert str(today_et(datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc))) == "2026-11-03"


def test_catalyst_check_records_both_and_gate_helper_is_unchanged():
    with patch("db.get_next_earnings", return_value=None):
        c = analyst._catalyst_check("NVDA", {"catalyst": {"date": "2030-01-15"}})
    assert c["weeks_to_catalyst_utc"] is not None and c["weeks_to_catalyst_et"] is not None
    assert c["weeks_to_catalyst_et"] > c["weeks_to_catalyst_utc"]
    with patch("db.get_next_earnings", return_value=None):
        c = analyst._catalyst_check("NVDA", {"catalyst": {"date": "ongoing"}})
    assert c["weeks_to_catalyst_utc"] is None and c["weeks_to_catalyst_et"] is None


def test_past_or_today_uses_the_et_date():
    """At 20:00 EST on 11-02 the UTC date is already 11-03; a catalyst dated
    11-03 is tomorrow in ET, not today."""
    fixed = datetime(2026, 11, 3, 1, 0, tzinfo=timezone.utc)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz else fixed.replace(tzinfo=None)

    with patch.object(analyst, "datetime", _DT), patch("db.get_next_earnings", return_value=None):
        tomorrow = analyst._catalyst_check("NVDA", {"catalyst": {"date": "2026-11-03"}})
        today = analyst._catalyst_check("NVDA", {"catalyst": {"date": "2026-11-02"}})
    assert tomorrow["past_or_today"] is False
    assert today["past_or_today"] is True
