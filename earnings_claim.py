"""Record-only earnings-claim check on a verdict's catalyst (audit 2026-11-22
00:56 #1). After a print, the brief drops the ticker's own next earnings date
when it is more than 45 days out (context_engine._own_earnings_line), and the
model can then name its sector's earnings-cluster week from the catalyst
calendar as the company's own report. This record says whether a catalyst
that names earnings has a date any earnings source knows for the ticker, and
whether that date falls in a sector earnings-cluster window.

It runs after the verdict, from analyst._analyze_inner, and its fields are
merged into decision.catalyst_check. Nothing in the prompt, the brief, the
gates or routing reads it. It never writes: earnings_calendar (which feeds
blackout and the earnings review) is only read. It never raises.

Fields:
  earnings_claim_unmatched    True when catalyst.event names earnings and
                              catalyst.date (a day) is not within
                              EARNINGS_CLAIM_TOL_DAYS of any date in the three
                              sources; False when it is; None when the text
                              names no earnings, the date is not a day
                              (month-only, "ongoing", unparseable) or the read
                              failed
  matches_sector_cluster_date True when catalyst.date falls in a
                              sector_earnings_cluster window (window_start ..
                              window_end, else its event_date) of the
                              ticker's sector ETF; False when it doesn't; None
                              when there is no day date, no ETF or no store
  calendar_source_used        the first matching source in the order
                              earnings_calendar, earnings_calendar_history,
                              earnings_consensus_history; None when none
  earnings_claim              the detail: what was read and why a field is
                              None

catalyst_kind (audit 2026-11-22 02:56 #1) sorts every verdict's catalyst by
how its date relates to the verdict's ET day and the calendars, from the same
two reads (no further Postgres round trip):
  catalyst_kind               already_released  the date is before the
                                                verdict's ET day, or on it
                                                with no scheduled row
                              scheduled_today   the date is the verdict's ET
                                                day and a ticker earnings
                                                date (earnings_calendar or
                                                earnings_calendar_history) or
                                                a ticker catalyst_events row
                                                of the kind the text names is
                                                on it (the calendars carry no
                                                before/after-market time, so
                                                the verdict may precede the
                                                report)
                              calendar_matched  a later day within
                                                CATALYST_KIND_TOL_DAYS of a
                                                calendar row of the same kind
                                                (see _row_counts)
                              unmatched_dated   a later day with no such row
                              ongoing           "ongoing" or a month-only date
                              unparsed          no date, or not a date
                              None              a later day, no match, and a
                                                read it needed failed
  catalyst_days_ahead         catalyst day minus the verdict's ET day (None
                              without a day date)
  catalyst_matched_source     the nearest matching row's source (an earnings
                              source name, or catalyst_events:<scope>); set
                              for any kind with a match, already_released too
  catalyst_matched_event_type that row's event type
  catalyst_kind_detail        the verdict day, every match, the rows within
                              the tolerance whose kind the wording doesn't
                              name (wording_blind), the verdict's ET clock
                              time (verdict_time_et), and the reason for None
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime, timezone

EARNINGS_CLAIM_TOL_DAYS = 3
EARNINGS_CLAIM_READ_TIMEOUT_MS = 500
SOURCES = ("earnings_calendar", "earnings_calendar_history", "earnings_consensus_history")
CATALYST_KIND_TOL_DAYS = EARNINGS_CLAIM_TOL_DAYS
CATALYST_KINDS = ("already_released", "scheduled_today", "calendar_matched", "unmatched_dated",
                  "ongoing", "unparsed")
# The rows that make a same-day catalyst scheduled_today rather than already_released.
SCHEDULED_TODAY_SOURCES = ("earnings_calendar", "earnings_calendar_history", "catalyst_events:ticker")

# "earnings" or "EPS" anywhere, or a quarter/fiscal word with report/results:
# "CPI report", "jobs report" and "trial results" are not earnings.
_EARNINGS_RE = re.compile(
    r"\bearnings\b|\bEPS\b|"
    r"\b(?:Q[1-4]|quarter(?:ly)?|fiscal|FY\s?\d{2,4})\b[^.;]{0,40}\b(?:report|results)\b|"
    r"\b(?:report|results)\b[^.;]{0,20}\b(?:Q[1-4]|quarter(?:ly)?|fiscal|FY\s?\d{2,4})\b", re.I)
_SECTOR_WORDS_RE = re.compile(r"\bsector\b|\bcluster\b", re.I)
# A macro row counts only when the catalyst names its kind of release.
_MACRO_WORDS = {
    "fomc_meeting": re.compile(r"\bFOMC\b|\bFed\b|Federal Reserve|rate decision|\bPowell\b", re.I),
    "cpi_release": re.compile(r"\bCPI\b|\binflation\b|consumer price", re.I),
    "jobs_report": re.compile(r"jobs report|\bpayrolls?\b|nonfarm|employment report|unemployment", re.I),
    "gdp_release": re.compile(r"\bGDP\b", re.I),
    "election": re.compile(r"\belections?\b", re.I),
    "geopolitical": re.compile(r"tariff|geopolitic|sanction|trade war", re.I),
}
_REBALANCE_RE = re.compile(r"rebalanc|index (?:inclusion|addition)", re.I)


def _day(raw) -> tuple[date | None, str | None]:
    """(the catalyst date when it is a day, else None; why not)."""
    s = str(raw or "").strip()
    if not s:
        return None, "no catalyst date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date(), None
        except ValueError:
            return None, "unparseable catalyst date"
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return None, "month-only catalyst date"
    if s.lower() == "ongoing":
        return None, "ongoing catalyst"
    return None, "unparseable catalyst date"


def _as_date(v) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def sector_etf(ticker: str) -> tuple[str | None, str | None]:
    """(the ticker's sector ETF as the brief's catalyst calendar resolves it,
    why None). Read only from an already-imported ingestion.sp500_universe,
    so this never triggers its constituent fetch."""
    mod = sys.modules.get("ingestion.sp500_universe")
    if mod is None:
        return None, "sector map not loaded in this process"
    gics = (mod.get_sp500_constituents().get(str(ticker).upper()) or {}).get("gics_sector")
    if not gics:
        return None, "not an S&P 500 constituent"
    etf = mod.SECTOR_ETF_MAP.get(gics)
    return (etf, None) if etf else (None, f"no ETF for sector {gics}")


def catalyst_events_as_of(ticker, etf, as_of: datetime, path=None) -> list[dict]:
    """The catalyst calendar rows created at or before as_of that can bear on
    ticker: its own (scope ticker), its sector ETF's (scope sector, when etf)
    and every macro row. One query on the SQLite store opened read-only
    (never created). Raises when the store can't be read."""
    import sqlite3
    if path is None:
        from catalyst_service import storage
        path = storage.DB_PATH
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    try:
        rows = conn.execute(
            "SELECT scope, ticker, event_type, event_date, window_start, window_end, thesis_note "
            "FROM catalyst_events WHERE created_at <= ? AND ((scope='ticker' AND ticker=?) "
            "OR (scope='sector' AND ticker=?) OR scope='macro') ORDER BY event_date",
            (as_of.astimezone(timezone.utc).isoformat(), ticker, etf)).fetchall()
    finally:
        conn.close()
    return [{"scope": r[0], "ticker": r[1], "event_type": r[2], "event_date": r[3],
             "window_start": r[4], "window_end": r[5], "note": (r[6] or "")[:120]} for r in rows]


def _cluster_windows(events: list, etf: str) -> list[dict]:
    return [{k: e.get(k) for k in ("event_date", "window_start", "window_end", "note")}
            for e in events if e.get("scope") == "sector" and e.get("ticker") == etf
            and e.get("event_type") == "sector_earnings_cluster"]


def sector_cluster_windows(etf: str, as_of: datetime, path=None) -> list[dict]:
    """The sector_earnings_cluster events of etf created at or before as_of
    (catalyst_events_as_of, read-only). Raises when the store can't be read."""
    return _cluster_windows(catalyst_events_as_of(None, etf, as_of, path=path), etf)


def evaluate(event_text, raw_date, dates: dict | None, windows: list | None,
             tol_days: int = EARNINGS_CLAIM_TOL_DAYS) -> dict:
    """Pure: the fields from the catalyst text and date, the earnings dates by
    source ({source: [date, ...]}, None when unread) and the sector windows
    (None when unread)."""
    text = event_text if isinstance(event_text, str) else ""
    d, why_no_day = _day(raw_date)
    detail = {"names_earnings": bool(_EARNINGS_RE.search(text)),
              "names_sector_or_cluster": bool(_SECTOR_WORDS_RE.search(text)),
              "catalyst_day": d.isoformat() if d else None, "why_no_day": why_no_day,
              "tol_days": tol_days, "n_dates": None, "nearest_days": None,
              "sector_window": None}
    out = {"earnings_claim_unmatched": None, "matches_sector_cluster_date": None,
           "calendar_source_used": None, "earnings_claim": detail}
    if dates is not None:
        detail["n_dates"] = {s: len(dates.get(s) or []) for s in SOURCES}
        if d is not None:
            near = {}
            for s in SOURCES:
                ds = [x for x in (_as_date(v) for v in dates.get(s) or []) if x]
                near[s] = min(((x - d).days for x in ds), key=abs) if ds else None
            detail["nearest_days"] = near
            if detail["names_earnings"]:
                hit = [s for s in SOURCES if near[s] is not None and abs(near[s]) <= tol_days]
                out["earnings_claim_unmatched"] = not hit
                out["calendar_source_used"] = hit[0] if hit else None
    if windows is not None and d is not None:
        for w in windows:
            lo = _as_date(w.get("window_start")) or _as_date(w.get("event_date"))
            hi = _as_date(w.get("window_end")) or _as_date(w.get("event_date"))
            if lo and hi and lo <= d <= hi:
                detail["sector_window"] = {**w, "on_event_date": _as_date(w.get("event_date")) == d}
                break
        out["matches_sector_cluster_date"] = detail["sector_window"] is not None
    return out


def _verdict_day_et(now: datetime) -> date:
    from zoneinfo import ZoneInfo
    return now.astimezone(ZoneInfo("America/New_York")).date()


def _verdict_time_et(now: datetime) -> str:
    """HH:MM ET of the verdict: beside scheduled_today, it shows whether the
    verdict came before or after the close on the report day."""
    from zoneinfo import ZoneInfo
    return now.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")


def _days_off(d: date, lo: date, hi: date) -> int:
    """0 inside [lo, hi], else signed days from d to the nearer edge."""
    return 0 if lo <= d <= hi else (lo - d).days if d < lo else (hi - d).days


def _row_counts(text: str, names_earnings: bool, e: dict) -> bool:
    """Whether a catalyst_events row is of the kind the catalyst text names:
    a ticker earnings row needs earnings wording (other ticker rows count
    as they are); a sector row needs sector or cluster wording (or, for
    index_rebalance, rebalance wording); a macro row needs its release's
    wording (_MACRO_WORDS)."""
    scope, kind = e.get("scope"), e.get("event_type")
    if scope == "ticker":
        return names_earnings if kind == "earnings" else True
    if scope == "sector":
        return bool(_SECTOR_WORDS_RE.search(text)
                    or (kind == "index_rebalance" and _REBALANCE_RE.search(text)))
    if scope == "macro":
        pat = _MACRO_WORDS.get(kind)
        return bool(pat and pat.search(text))
    return False


def catalyst_kind_of(event_text, raw_date, verdict_day: date, dates: dict | None,
                     events: list | None, reads: dict | None = None,
                     tol_days: int = CATALYST_KIND_TOL_DAYS) -> dict:
    """Pure: catalyst_kind and its fields (module docstring) from the catalyst
    text and date, the verdict's ET day, the earnings dates by source (None
    when unread; read only when the text names earnings), the catalyst_events
    rows (None when unread) and the read labels."""
    text = event_text if isinstance(event_text, str) else ""
    reads = reads or {}
    d, why_no_day = _day(raw_date)
    names_earnings = bool(_EARNINGS_RE.search(text))
    detail = {"verdict_day_et": verdict_day.isoformat(), "catalyst_day": d.isoformat() if d else None,
              "why_no_day": why_no_day, "tol_days": tol_days, "names_earnings": names_earnings,
              "matches": [], "wording_blind": [], "reason": None}
    out = {"catalyst_kind": None, "catalyst_days_ahead": None, "catalyst_matched_source": None,
           "catalyst_matched_event_type": None, "catalyst_kind_detail": detail}
    if d is None:
        out["catalyst_kind"] = ("ongoing" if why_no_day in ("ongoing catalyst", "month-only catalyst date")
                                else "unparsed")
        return out
    out["catalyst_days_ahead"] = (d - verdict_day).days
    matches = []
    if names_earnings and dates is not None:
        for s in SOURCES:
            for x in (_as_date(v) for v in dates.get(s) or []):
                if x and abs((x - d).days) <= tol_days:
                    matches.append({"source": s, "event_type": "earnings",
                                    "date": x.isoformat(), "days_off": (x - d).days})
    for e in events or []:
        ev = _as_date(e.get("event_date"))
        lo = _as_date(e.get("window_start")) or ev
        hi = _as_date(e.get("window_end")) or ev
        if not (lo and hi):
            continue
        off = _days_off(d, lo, hi)
        if abs(off) > tol_days:
            continue
        m = {"source": f"catalyst_events:{e.get('scope')}", "event_type": e.get("event_type"),
             "date": ev.isoformat() if ev else None, "days_off": off}
        if e.get("scope") == "sector":
            m["ticker"] = e.get("ticker")
        (matches if _row_counts(text, names_earnings, e) else detail["wording_blind"]).append(m)
    detail["matches"] = matches
    if matches:
        best = min(enumerate(matches), key=lambda im: (abs(im[1]["days_off"]), im[0]))[1]
        out["catalyst_matched_source"] = best["source"]
        out["catalyst_matched_event_type"] = best["event_type"]
    today = [m for m in matches if d == verdict_day and m["days_off"] == 0
             and m["source"] in SCHEDULED_TODAY_SOURCES]
    if today:
        out["catalyst_kind"] = "scheduled_today"
        out["catalyst_matched_source"] = today[0]["source"]
        out["catalyst_matched_event_type"] = today[0]["event_type"]
    elif d <= verdict_day:
        out["catalyst_kind"] = "already_released"
    elif matches:
        out["catalyst_kind"] = "calendar_matched"
    else:
        failed = [k for k in ("events", "dates") if str(reads.get(k, "ok")).startswith("error")]
        sector = str(reads.get("sector", "ok"))
        if _SECTOR_WORDS_RE.search(text) and (sector.startswith("error")
                                               or sector == "skipped: sector map not loaded in this process"):
            failed.append("sector")   # sector wording, and the ETF's rows were not read
        if failed:
            detail["reason"] = "no match and a needed read failed: " + ", ".join(
                f"{k} {reads[k]}" for k in failed)
        else:
            out["catalyst_kind"] = "unmatched_dated"
    return out


def earnings_claim_check(ticker: str, parsed: dict, now: datetime | None = None,
                         read_dates=None, read_events=None) -> dict:
    """The record for a verdict: one bounded Postgres read (db.get_earnings_claim_dates,
    only when the catalyst names earnings) and one read-only SQLite read
    (catalyst_events_as_of), each failing open to a reason. The earnings
    fields (evaluate) and catalyst_kind (catalyst_kind_of) share them. The
    two readers can be passed in (tests, replays). Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        cat = (parsed or {}).get("catalyst") if isinstance(parsed, dict) else None
        cat = cat if isinstance(cat, dict) else {}
        text, raw = cat.get("event"), cat.get("date")
        reads = {"dates": "ok", "sector": "ok", "events": "ok", "sector_etf": None}
        dates = windows = events = None
        d, _ = _day(raw)
        names = bool(_EARNINGS_RE.search(text if isinstance(text, str) else ""))
        if d is None:
            reads["dates"] = reads["sector"] = reads["events"] = "skipped: no day date"
        else:
            try:
                if not names:
                    reads["dates"] = "skipped: catalyst names no earnings"
                elif read_dates is None:
                    import db
                    dates = db.get_earnings_claim_dates(ticker, now, EARNINGS_CLAIM_READ_TIMEOUT_MS)
                else:
                    dates = read_dates(ticker, now)
            except Exception as e:
                reads["dates"] = f"error: {type(e).__name__}"
            etf = why = None
            try:
                etf, why = sector_etf(ticker)
                reads["sector_etf"] = etf
            except Exception as e:
                reads["sector"] = f"error: {type(e).__name__}"
            try:
                events = (catalyst_events_as_of(ticker, etf, now) if read_events is None
                          else read_events(ticker, etf, now))
            except Exception as e:
                reads["events"] = f"error: {type(e).__name__}"
            if reads["sector"] == "ok":
                if etf is None:
                    reads["sector"] = f"skipped: {why}"
                elif events is None:
                    reads["sector"] = reads["events"]
                else:
                    windows = _cluster_windows(events, etf)
        out = evaluate(text, raw, dates, windows)
        out["earnings_claim"].update(read_dates=reads["dates"], read_sector=reads["sector"],
                                     sector_etf=reads["sector_etf"])
        try:
            kind = catalyst_kind_of(text, raw, _verdict_day_et(now), dates, events, reads)
            kind["catalyst_kind_detail"].update(read_events=reads["events"], read_sector=reads["sector"],
                                                verdict_time_et=_verdict_time_et(now))
        except Exception as e:
            kind = {"catalyst_kind": None, "catalyst_days_ahead": None,
                    "catalyst_matched_source": None, "catalyst_matched_event_type": None,
                    "catalyst_kind_detail": {"error": f"{type(e).__name__}: {str(e)[:120]}"}}
        out.update(kind)
        return out
    except Exception as e:
        return {"earnings_claim_unmatched": None, "matches_sector_cluster_date": None,
                "calendar_source_used": None,
                "earnings_claim": {"error": f"{type(e).__name__}: {str(e)[:120]}"},
                "catalyst_kind": None, "catalyst_days_ahead": None,
                "catalyst_matched_source": None, "catalyst_matched_event_type": None,
                "catalyst_kind_detail": {"error": f"{type(e).__name__}: {str(e)[:120]}"}}
