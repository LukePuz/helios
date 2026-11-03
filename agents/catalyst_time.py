"""Eastern-time readings of the model's catalyst date, for log-only shadows.

analyst.compute_weeks_to_catalyst (which feeds the expected-return-rate
gate) reads a naive catalyst date as UTC midnight, which is 19:00 or 20:00
ET the day before, so the distance shifts by an hour at each DST change.
weeks_to_catalyst_et reads the same date as midnight in America/New_York.
It is recorded beside the UTC figure and is not used by any gate.
"""
import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def today_et(now: datetime | None = None) -> date:
    """The current calendar date in America/New_York."""
    return (now or datetime.now(timezone.utc)).astimezone(ET).date()


def weeks_to_catalyst_et(catalyst_date: str | None, as_of: datetime | None = None,
                         month_midpoint: bool | None = None) -> float | None:
    """compute_weeks_to_catalyst with the date read as ET midnight: same
    formats, same "ongoing"/unparseable -> None, same 1-day floor."""
    if not catalyst_date:
        return None
    text = catalyst_date.strip()
    if text.lower() == "ongoing":
        return None
    parsed = None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    if month_midpoint is None:
        try:
            from config import CATALYST_MONTH_MIDPOINT_ENABLED as month_midpoint
        except ImportError:
            month_midpoint = False
    if month_midpoint and re.fullmatch(r"\d{4}-\d{1,2}", text):
        parsed = parsed.replace(day=15)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    now = as_of or datetime.now(timezone.utc)
    days = (parsed - now).total_seconds() / 86400
    return max(days, 1.0) / 7.0
