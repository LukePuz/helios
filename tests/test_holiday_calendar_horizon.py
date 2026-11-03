"""The NYSE holiday calendar in market_data must reach at least a year past
today. Past its end, holidays count as trading days (only a log warning).
This test starts failing on 1 January of the calendar's last year: extend
_NYSE_HOLIDAYS, _NYSE_EARLY_CLOSES and _CALENDAR_LAST_YEAR from NYSE's
published schedule (nyse.com/markets/hours-calendars)."""
from datetime import date

import market_data


def test_calendar_has_at_least_a_year_left():
    end = date(market_data._CALENDAR_LAST_YEAR, 12, 31)
    left = (end - date.today()).days
    assert left >= 365, (
        f"NYSE holiday calendar ends {end} ({left} days from today, under a year). "
        f"Add the next year's holidays and early closes to market_calendar.NYSE_HOLIDAYS / "
        f"NYSE_EARLY_CLOSES and raise CALENDAR_LAST_YEAR.")


def test_last_year_constant_matches_the_data():
    years = {d.year for d in market_data._NYSE_HOLIDAYS}
    assert max(years) == market_data._CALENDAR_LAST_YEAR
    assert years == set(range(min(years), market_data._CALENDAR_LAST_YEAR + 1)), \
        "a year inside the calendar has no holidays listed"
    assert all(d.year <= market_data._CALENDAR_LAST_YEAR for d in market_data._NYSE_EARLY_CLOSES)
