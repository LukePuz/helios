"""Market hours follow America/New_York and the NYSE calendar (professor C14)."""
from datetime import datetime, timezone

import market_data as md


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_summer_hours():
    assert md.is_market_hours(_utc(2026, 10, 28, 13, 30))       # 09:30 EDT
    assert not md.is_market_hours(_utc(2026, 10, 28, 13, 29))
    assert md.is_market_hours(_utc(2026, 10, 28, 20, 0))        # 16:00 EDT


def test_winter_hours_after_dst():
    assert not md.is_market_hours(_utc(2026, 11, 2, 14, 0))     # 09:00 EST: closed
    assert md.is_market_hours(_utc(2026, 11, 2, 14, 30))        # 09:30 EST
    assert md.is_market_hours(_utc(2026, 11, 2, 20, 30))        # 15:30 EST: open
    assert not md.is_market_hours(_utc(2026, 11, 2, 21, 1))     # 16:01 EST


def test_holidays_and_early_close():
    assert not md.is_market_hours(_utc(2026, 11, 26, 16, 0))    # Thanksgiving
    assert md.is_market_hours(_utc(2026, 11, 27, 17, 0))        # 12:00 EST early-close day
    assert not md.is_market_hours(_utc(2026, 11, 27, 18, 30))   # 13:30 EST, closed


def test_weekend_in_et():
    # Fri 21:00 EDT is Sat 01:00 UTC: a UTC weekday check got this wrong way round only at edges
    assert not md.is_market_hours(_utc(2026, 10, 31, 15, 0))    # Saturday


def test_next_open_across_dst_and_holiday():
    assert md.next_market_open(_utc(2026, 10, 30, 21, 0)) == _utc(2026, 11, 2, 14, 30)
    assert md.next_market_open(_utc(2026, 11, 25, 22, 0)) == _utc(2026, 11, 27, 14, 30)
    assert md.next_market_open(_utc(2026, 10, 28, 12, 0)) == _utc(2026, 10, 28, 13, 30)
