"""Non-finite bar flag for the offline readouts (eng-candidates 2026-11-26 #2).

price_history can hold NaN or infinite OHLC values (batch_verify's
nonfinite_bars check counts them). In Python a NaN close makes every
comparison False and every return NaN, so a readout reading such a bar is
silently wrong. The readouts call flag_line on the bars they loaded and print
one line:

    non-finite bars: N (TICKER date, ...)

Flag only: nothing is dropped, skipped or changed here, and no readout value
depends on this module. Whether a readout drops flagged bars is an analysis
choice (Luke's).

A bar is non-finite when any of its OHLC values (open, high, low, close: the
ones the readout loaded) is NaN or infinite. None (a missing value) is not
counted. Bars are dicts with a "date" key, or tuples (date, value, ...).
This is the same rule as batch_verify's SQL form (nonfinite_bars_sql,
_nonfinite_sql: NONFINITE_BAR_COLUMNS, NULL not counted), applied to bars a
readout already fetched. Standard library only; scripts load it with
importlib (scripts/ is not a package).
"""
from __future__ import annotations

import math
from decimal import Decimal

OHLC = ("open", "high", "low", "close")
LABEL = "non-finite bars"


def is_finite(v) -> bool:
    """False only for a NaN or infinite number (float, int, Decimal, or text
    such as 'NaN'); None and anything that isn't a number count as finite."""
    if v is None or isinstance(v, bool):
        return True
    if isinstance(v, Decimal):
        return v.is_finite()
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):
        return True


def _values(bar) -> tuple:
    if isinstance(bar, dict):
        return tuple(bar[k] for k in OHLC if k in bar)
    return tuple(bar[1:])


def _date(bar) -> str:
    d = bar.get("date") if isinstance(bar, dict) else bar[0]
    return str(d)[:10]


def nonfinite_bars(bars_by_ticker: dict) -> list[tuple[str, str]]:
    """(ticker, date) of every bar with a non-finite OHLC value, sorted. Pure."""
    out = []
    for ticker, bars in (bars_by_ticker or {}).items():
        for b in bars or ():
            if not all(is_finite(v) for v in _values(b)):
                out.append((str(ticker), _date(b)))
    return sorted(out)


def flag_line(bars_by_ticker: dict) -> str:
    """'non-finite bars: N (TICKER date, ...)'; just 'non-finite bars: 0' when
    there are none. Pure."""
    found = nonfinite_bars(bars_by_ticker)
    names = f" ({', '.join(f'{t} {d}' for t, d in found)})" if found else ""
    return f"{LABEL}: {len(found)}{names}"


def flag_record(bars_by_ticker: dict) -> dict:
    """The same as a JSON-ready dict: {"n": N, "bars": [[ticker, date], ...]}."""
    found = nonfinite_bars(bars_by_ticker)
    return {"n": len(found), "bars": [list(x) for x in found]}
