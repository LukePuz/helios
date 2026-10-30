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
    # Corrected 2026-10-29 from bls.gov/schedule/news_release/cpi.htm: Oct
    # data comes out 11-10 (was 11-12, "Sep 2026"), Nov data 12-10.
    ("2026-11-10", "Oct 2026", "confirmed"),
    ("2026-12-10", "Nov 2026", "confirmed"),
]

# 2027 dates (Luke D18, 10-29), checked against the official pages on
# 2026-10-29. FOMC dates are tentative until the preceding meeting confirms them.
FOMC_2027 = [
    ("2027-01-27", "FOMC meeting (Jan 26-27) -- rate decision + statement"),
    ("2027-03-17", "FOMC meeting (Mar 16-17) -- rate decision, statement, dot plot"),
    ("2027-04-28", "FOMC meeting (Apr 27-28) -- rate decision + statement"),
    ("2027-06-09", "FOMC meeting (Jun 8-9) -- rate decision, statement, dot plot"),
    ("2027-07-28", "FOMC meeting (Jul 27-28) -- rate decision + statement"),
    ("2027-09-15", "FOMC meeting (Sep 14-15) -- rate decision, statement, dot plot"),
    ("2027-10-27", "FOMC meeting (Oct 26-27) -- rate decision + statement"),
    ("2027-12-08", "FOMC meeting (Dec 7-8) -- rate decision, statement, dot plot"),
]

CPI_2027 = [
    ("2027-01-13", "Dec 2026", "confirmed"),
    ("2027-02-11", "Jan 2027", "confirmed"),
    ("2027-03-10", "Feb 2027", "confirmed"),
    ("2027-04-13", "Mar 2027", "confirmed"),
    ("2027-05-12", "Apr 2027", "confirmed"),
    ("2027-06-10", "May 2027", "confirmed"),
    ("2027-07-14", "Jun 2027", "confirmed"),
    ("2027-08-11", "Jul 2027", "confirmed"),
    ("2027-09-14", "Aug 2027", "confirmed"),
    ("2027-10-14", "Sep 2027", "confirmed"),
    ("2027-11-10", "Oct 2027", "confirmed"),
    ("2027-12-10", "Nov 2027", "confirmed"),
]

# (event_date, month_covered) -- 8:30 ET, first Friday of the month
# covering the PRIOR month's data. Confirmed against bls.gov 2026-08-30.
JOBS_REPORT_2026 = [
    ("2026-09-04", "Aug 2026"),
    ("2026-10-02", "Sep 2026"),
    ("2026-11-06", "Oct 2026"),
    ("2026-12-04", "Nov 2026"),
]

JOBS_REPORT_2027 = [
    ("2027-01-08", "Dec 2026"),
    ("2027-02-05", "Jan 2027"),
    ("2027-03-05", "Feb 2027"),
    ("2027-04-02", "Mar 2027"),
    ("2027-05-07", "Apr 2027"),
    ("2027-06-04", "May 2027"),
    ("2027-07-02", "Jun 2027"),
    ("2027-08-06", "Jul 2027"),
    ("2027-09-03", "Aug 2027"),
    ("2027-10-08", "Sep 2027"),
    ("2027-11-05", "Oct 2027"),
    ("2027-12-03", "Nov 2027"),
]

# Warn when the hard-coded calendar has less future than this left (A1653-3).
_MIN_FUTURE_DAYS = 60

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

    for event_date, note in FOMC_2026 + FOMC_2027:
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

    for event_date, month_covered, confidence in CPI_2026 + CPI_2027:
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

    for event_date, month_covered in JOBS_REPORT_2026 + JOBS_REPORT_2027:
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

    _drop_unlisted_future_events(today)
    last = max(d for d, *_ in FOMC_2026 + FOMC_2027 + CPI_2026 + CPI_2027
               + JOBS_REPORT_2026 + JOBS_REPORT_2027)
    import datetime as _dt
    left = (_dt.date.fromisoformat(last) - _dt.date.fromisoformat(today)).days
    if left < _MIN_FUTURE_DAYS:
        log.warning(f"Macro calendar has only {left} days of future events left "
                    f"(last {last}); add the next year's FOMC/CPI/jobs dates")
    log.info(f"Macro calendar refreshed: {count} upcoming events upserted")
    return count


def _drop_unlisted_future_events(today: str) -> None:
    """Delete future FOMC/CPI/jobs rows no longer in the lists above, so a
    corrected date doesn't leave the old one in the analyst's calendar (the
    upsert key includes the date)."""
    keep = {
        "fomc_meeting": {d for d, _ in FOMC_2026 + FOMC_2027},
        "cpi_release": {d for d, *_ in CPI_2026 + CPI_2027},
        "jobs_report": {d for d, _ in JOBS_REPORT_2026 + JOBS_REPORT_2027},
    }
    try:
        from catalyst_service.storage import get_conn
        conn = get_conn()
        try:
            for event_type, dates in keep.items():
                rows = conn.execute(
                    "SELECT event_date FROM catalyst_events WHERE scope='macro' "
                    "AND event_type=? AND event_date>=?", (event_type, today)).fetchall()
                stale = sorted({r[0] for r in rows} - dates)
                for d in stale:
                    conn.execute("DELETE FROM catalyst_events WHERE scope='macro' "
                                 "AND event_type=? AND event_date=?", (event_type, d))
                if stale:
                    log.info(f"Macro calendar: removed {event_type} dates no longer listed: {stale}")
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"Macro calendar cleanup failed: {e}")
