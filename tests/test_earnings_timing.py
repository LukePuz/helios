"""Earnings reviews record BMO/AMC from yfinance's earnings_dates in the
trigger provenance (audits 2026-11-08 13:18 #3/#6). Log-only."""
from datetime import date
from unittest.mock import patch

import pandas as pd

from agents import analyst, crew


def _df(*rows):
    idx = pd.DatetimeIndex(pd.to_datetime([r[0] for r in rows], utc=True)
                           .tz_convert("America/New_York"), name="Earnings Date")
    return pd.DataFrame({"EPS Estimate": [1.0] * len(rows),
                         "Reported EPS": [r[1] for r in rows]}, index=idx)


def test_classes():
    df = _df(("2026-11-17 16:00:00-05:00", float("nan")), ("2026-08-26 16:00:00-04:00", 2.2),
             ("2026-11-19 07:00:00-05:00", float("nan")), ("2026-11-20 00:00:00-05:00", float("nan")))
    with patch.object(crew, "_fetch_earnings_dates", return_value=df):
        amc = crew._earnings_timing("NVDA", date(2026, 11, 17))
        bmo = crew._earnings_timing("NVDA", date(2026, 11, 19))
        unk = crew._earnings_timing("NVDA", date(2026, 11, 20))
        past = crew._earnings_timing("NVDA", date(2026, 8, 26))
        miss = crew._earnings_timing("NVDA", date(2026, 12, 1))
    assert (amc["status"], amc["time_of_day"], amc["reported"]) == ("ok", "AMC", False)
    assert bmo["time_of_day"] == "BMO" and unk["time_of_day"] == "unknown"
    assert past["reported"] is True
    assert miss["status"] == "no_match" and miss["time_of_day"] is None
    assert amc["is_estimate"] is None


def test_errors_never_raise():
    with patch.object(crew, "_fetch_earnings_dates", side_effect=RuntimeError("yf")):
        assert crew._earnings_timing("NVDA", date(2026, 11, 17))["status"] == "RuntimeError"


def test_trigger_record_carries_it_and_the_brief_does_not():
    timing = {"status": "ok", "time_of_day": "AMC"}
    ev = {**crew._earnings_review_event("NVDA", date(2026, 11, 17), today=date(2026, 11, 13)),
          "earnings_timing": timing}
    assert analyst.trigger_record("earnings_review", ev)["earnings_timing"] == timing
    assert "earnings_timing" not in analyst.trigger_record("event", {"title": "x"})
