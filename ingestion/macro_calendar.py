"""
Macro (market-wide) catalyst calendar (2026-08-30, user request).

Unlike per-ticker earnings (ingestion/earnings_calendar.py, sourced live
from yfinance), there is no free live API for a forward-looking schedule
of FOMC meetings / BLS release dates / election dates -- but all three ARE
published by their respective authorities on a fixed, known-in-advance
schedule (the Fed publishes the full-year FOMC calendar; the BLS publishes
its "Schedule of Release Dates for Principal Federal Economic Indicators"
each fall for the following year; federal election dates are set by
statute). So this is a maintained table, not a scraper -- refreshed here
from the official sources below and re-verified/extended each
November/December when next year's schedule is published.

Sourced 2026-08-30:
  FOMC 2026 dates  -- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  CPI release dates -- https://www.bls.gov/schedule/news_release/cpi.htm
  Jobs report dates -- https://www.bls.gov/ces/publications/news-release-schedule.htm
  2026 midterms      -- statutory: Tuesday after the first Monday in November

CPI/jobs entries not yet confirmed against an official BLS release (the
site only publishes a few months at a time) are marked confidence=
"estimated" with a same-week placeholder date rather than fabricated
precision -- refresh_macro_calendar() re-upserts on every run, so
correcting an estimated date later just means editing the table below and
re-running; nothing needs an explicit "un-estimate" step.
"""
import logging
from datetime import date

log = logging.getLogger(__name__)

_FED_SOURCE = {
    "name": "federalreserve.gov",
    "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
}
_BLS_CPI_SOURCE = {
    "name": "bls.gov (CPI)",
    "url": "https://www.bls.gov/schedule/news_release/cpi.htm",
}
_BLS_JOBS_SOURCE = {
    "name": "bls.gov (Employment Situation)",
    "url": "https://www.bls.gov/ces/publications/news-release-schedule.htm",
}
_STATUTORY_SOURCE = {
    "name": "federal election law (2 U.S.C. § 7)",
    "url": "https://www.britannica.com/question/When-are-the-2026-midterm-elections",
}

# (event_date, thesis_note) -- decision/statement day only (2nd day of each
# 2-day meeting), 14:00 ET. Confirmed against federalreserve.gov 2026-08-30.
FOMC_2026 = [
    ("2026-01-28", "FOMC meeting (Jan 27-28) -- rate decision + statement"),
    ("2026-03-18", "FOMC meeting (Mar 17-18) -- rate decision, statement, dot plot"),
    ("2026-04-29", "FOMC meeting (Apr 28-29) -- rate decision + statement"),
    ("2026-06-17", "FOMC meeting (Jun 16-17) -- rate decision, statement, dot plot"),
    ("2026-07-29", "FOMC meeting (Jul 28-29) -- rate decision + statement"),
    ("2026-09-16", "FOMC meeting (Sep 15-16) -- rate decision, statement, dot plot"),
    ("2026-10-28", "FOMC meeting (Oct 27-28) -- rate decision + statement"),
    ("2026-12-09", "FOMC meeting (Dec 8-9) -- rate decision, statement, dot plot"),
]

# (event_date, month_covered, confidence) -- 8:30 ET. Entries sourced from
# actual bls.gov release pages are "confirmed"; months BLS hadn't published
# yet as of 2026-08-30 are "estimated" at the historical ~2nd-Tuesday cadence.
CPI_2026 = [
    ("2026-02-11", "Dec 2025", "confirmed"),
    ("2026-03-11", "Jan 2026", "confirmed"),
    ("2026-04-10", "Feb 2026", "confirmed"),
    ("2026-05-12", "Mar 2026", "confirmed"),
    ("2026-06-10", "Apr 2026", "estimated"),
    ("2026-07-14", "May 2026", "estimated"),
    ("2026-08-12", "Jun 2026", "estimated"),
    ("2026-09-11", "Jul 2026", "confirmed"),
    ("2026-10-13", "Aug 2026", "estimated"),
    ("2026-11-12", "Sep 2026", "estimated"),
    ("2026-12-10", "Oct 2026", "estimated"),
]

# (event_date, month_covered) -- 8:30 ET, first Friday of the month
# covering the PRIOR month's data. Confirmed against bls.gov 2026-08-30.
JOBS_REPORT_2026 = [
    ("2026-09-04", "Aug 2026"),
    ("2026-10-02", "Sep 2026"),
    ("2026-11-06", "Oct 2026"),
    ("2026-12-04", "Nov 2026"),
]

ELECTION_2026 = [
    ("2026-11-03", "2026 U.S. midterm elections -- every House seat, 35 Senate seats, plus state/local"),
]


def refresh_macro_calendar() -> int:
    """
    Upserts the tables above into catalyst_service as scope="macro" events.
    Idempotent (upsert_event's dedupe key is ticker/event_type/event_date,
    so re-running just refreshes updated_at/last_verified_at) -- safe to
    run on every scheduled refresh, not just once.

    Returns count of events upserted.
    """
    from catalyst_service.storage import upsert_macro_event

    count = 0
    today = date.today().isoformat()

    for event_date, note in FOMC_2026:
        if event_date < today:
            continue
        try:
            upsert_macro_event(
                event_type="fomc_meeting", event_date=event_date,
                confidence="confirmed", source_tier=1, sources=[_FED_SOURCE],
                thesis_note=note,
                action_flags={"tighten_sizing": True},
            )
            count += 1
        except Exception as e:
            log.warning(f"FOMC event upsert failed for {event_date}: {e}")

    for event_date, month_covered, confidence in CPI_2026:
        if event_date < today:
            continue
        try:
            upsert_macro_event(
                event_type="cpi_release", event_date=event_date,
                confidence=confidence, source_tier=1 if confidence == "confirmed" else 3,
                sources=[_BLS_CPI_SOURCE],
                thesis_note=f"CPI release covering {month_covered}",
                action_flags={"tighten_sizing": True} if confidence == "confirmed" else {},
            )
            count += 1
        except Exception as e:
            log.warning(f"CPI event upsert failed for {event_date}: {e}")

    for event_date, month_covered in JOBS_REPORT_2026:
        if event_date < today:
            continue
        try:
            upsert_macro_event(
                event_type="jobs_report", event_date=event_date,
                confidence="confirmed", source_tier=1, sources=[_BLS_JOBS_SOURCE],
                thesis_note=f"Employment Situation report covering {month_covered}",
                action_flags={"tighten_sizing": True},
            )
            count += 1
        except Exception as e:
            log.warning(f"Jobs report event upsert failed for {event_date}: {e}")

    for event_date, note in ELECTION_2026:
        if event_date < today:
            continue
        try:
            upsert_macro_event(
                event_type="election", event_date=event_date,
                confidence="confirmed", source_tier=1, sources=[_STATUTORY_SOURCE],
                thesis_note=note,
                action_flags={"tighten_sizing": True},
            )
            count += 1
        except Exception as e:
            log.warning(f"Election event upsert failed for {event_date}: {e}")

    log.info(f"Macro calendar refreshed: {count} upcoming events upserted")
    return count
