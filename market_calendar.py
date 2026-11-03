"""NYSE session calendar: full-day holidays and 13:00 ET early closes.

No broker or network imports, so research tooling can use the same
calendar as market_data.is_market_hours without loading the broker layer.
market_data re-exports these names.
"""
from datetime import date, time as dtime

RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)
EARLY_CLOSE = dtime(13, 0)
# NYSE full-day holidays and 13:00 ET early closes, 2026-2027 (published NYSE
# calendar). Extend before 2028; is_market_hours logs a warning past the end.
NYSE_HOLIDAYS = {date(*d) for d in (
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19),
    (2026, 7, 3), (2026, 9, 7), (2026, 11, 26), (2026, 12, 25),
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26), (2027, 5, 31), (2027, 6, 18),
    (2027, 7, 5), (2027, 9, 6), (2027, 11, 25), (2027, 12, 24),
)}
NYSE_EARLY_CLOSES = {date(*d) for d in ((2026, 11, 27), (2026, 12, 24), (2027, 11, 26))}
CALENDAR_LAST_YEAR = 2027
