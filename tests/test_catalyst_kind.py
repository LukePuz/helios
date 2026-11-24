"""Record-only catalyst_kind (earnings_claim.catalyst_kind_of): how a verdict's
catalyst date relates to the verdict's ET day and the calendars, from the
earnings-claim check's own reads. Merged into decision.catalyst_check;
nothing in the prompt, gates or routing reads it."""
import inspect
import sqlite3
import sys
import types
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import earnings_claim as ec

DAY = date(2026, 11, 19)                      # the verdict's ET day
NVDA_DATES = {"earnings_calendar": [date(2026, 11, 19)], "earnings_calendar_history": [],
              "earnings_consensus_history": []}
XLK = {"scope": "sector", "ticker": "XLK", "event_type": "sector_earnings_cluster",
       "event_date": "2026-11-30", "window_start": "2026-11-30", "window_end": "2026-12-06"}
CPI = {"scope": "macro", "ticker": "MARKET", "event_type": "cpi_release", "event_date": "2026-12-10",
       "window_start": None, "window_end": None}
FOMC = {"scope": "macro", "ticker": "MARKET", "event_type": "fomc_meeting", "event_date": "2026-12-09",
        "window_start": None, "window_end": None}


def kind(text, raw, dates=None, events=(), reads=None, day=DAY):
    return ec.catalyst_kind_of(text, raw, day, dates, None if events is None else list(events), reads)


def test_a_date_before_the_verdict_day_is_already_released():
    # The print being reacted to the day after still records its match.
    after = kind("Q3 earnings report and guidance", "2026-11-19", NVDA_DATES, day=date(2026, 11, 20))
    assert after["catalyst_kind"] == "already_released"
    assert after["catalyst_days_ahead"] == -1
    assert after["catalyst_matched_source"] == "earnings_calendar"
    assert after["catalyst_matched_event_type"] == "earnings"
    past = kind("Oppenheimer upgrade", "2026-11-09")
    assert past["catalyst_kind"] == "already_released"
    assert past["catalyst_days_ahead"] == -10
    assert past["catalyst_matched_source"] is None


def test_the_verdict_day_with_a_scheduled_ticker_row_is_scheduled_today():
    # The calendars carry no BMO/AMC time: a same-day report may still be ahead.
    for src in ("earnings_calendar", "earnings_calendar_history"):
        dates = {s: [] for s in ec.SOURCES}
        dates[src] = [DAY]
        out = kind("NVDA Q3 earnings report", "2026-11-19", dates)
        assert out["catalyst_kind"] == "scheduled_today", src
        assert out["catalyst_days_ahead"] == 0, src
        assert out["catalyst_matched_source"] == src, src
        assert out["catalyst_matched_event_type"] == "earnings", src
    row = {"scope": "ticker", "ticker": "NVDA", "event_type": "earnings", "event_date": "2026-11-19",
           "window_start": None, "window_end": None}
    ev = kind("NVDA earnings", "2026-11-19", {}, [XLK, row])
    assert ev["catalyst_kind"] == "scheduled_today"
    assert ev["catalyst_matched_source"] == "catalyst_events:ticker"
    conf = {**row, "event_type": "conference"}
    assert kind("GTC keynote", "2026-11-19", None, [conf])["catalyst_kind"] == "scheduled_today"


def test_the_verdict_day_without_a_scheduled_ticker_row_is_already_released():
    # No row on the day: the upgrade or headline being reacted to.
    out = kind("Morgan Stanley raises ANET to Overweight", "2026-11-19", None, [])
    assert out["catalyst_kind"] == "already_released"
    assert out["catalyst_days_ahead"] == 0
    # A row two days off is a match but not on the day.
    near = kind("NVDA earnings", "2026-11-19", {"earnings_calendar": [date(2026, 11, 21)]}, [])
    assert near["catalyst_kind"] == "already_released"
    assert near["catalyst_matched_source"] == "earnings_calendar"
    # The consensus date alone, a sector cluster or a macro row on the day: not scheduled for the ticker.
    yf = {"earnings_calendar": [], "earnings_calendar_history": [], "earnings_consensus_history": [DAY]}
    assert kind("NVDA earnings", "2026-11-19", yf, [])["catalyst_kind"] == "already_released"
    cluster = {**XLK, "event_date": "2026-11-19", "window_start": "2026-11-16", "window_end": "2026-11-22"}
    assert kind("sector earnings cluster", "2026-11-19", None, [cluster])["catalyst_kind"] == "already_released"
    cpi = {**CPI, "event_date": "2026-11-19"}
    assert kind("CPI report", "2026-11-19", None, [cpi])["catalyst_kind"] == "already_released"
    # A ticker earnings row the wording doesn't name doesn't count.
    row = {"scope": "ticker", "ticker": "NVDA", "event_type": "earnings", "event_date": "2026-11-19",
           "window_start": None, "window_end": None}
    assert kind("Oppenheimer upgrade", "2026-11-19", None, [row])["catalyst_kind"] == "already_released"


def test_a_later_day_near_an_earnings_date_is_calendar_matched():
    out = kind("Next earnings report", "2026-11-17", NVDA_DATES, day=date(2026, 11, 5))
    assert out["catalyst_kind"] == "calendar_matched"
    assert out["catalyst_days_ahead"] == 12
    assert out["catalyst_matched_source"] == "earnings_calendar"
    assert out["catalyst_kind_detail"]["matches"][0]["days_off"] == 2
    # Four days off is outside the tolerance.
    far = kind("Next earnings report", "2026-11-15", NVDA_DATES, day=date(2026, 11, 5))
    assert far["catalyst_kind"] == "unmatched_dated"
    assert ec.CATALYST_KIND_TOL_DAYS == 3


def test_the_audit_shape_is_unmatched_with_the_cluster_row_kept_aside():
    out = kind("NVIDIA's Q4 earnings report", "2026-11-30", NVDA_DATES, [XLK])
    assert out["catalyst_kind"] == "unmatched_dated"
    assert out["catalyst_days_ahead"] == 11
    assert out["catalyst_matched_source"] is None
    blind = out["catalyst_kind_detail"]["wording_blind"]
    assert [(m["source"], m["event_type"], m["days_off"]) for m in blind] == [
        ("catalyst_events:sector", "sector_earnings_cluster", 0)]
    # A catalyst that names the cluster itself is matched to it.
    own = kind("XLK sector earnings cluster with CRM, CRWD, MRVL reporting", "2026-11-30",
               NVDA_DATES, [XLK])
    assert own["catalyst_kind"] == "calendar_matched"
    assert own["catalyst_matched_source"] == "catalyst_events:sector"
    assert own["catalyst_matched_event_type"] == "sector_earnings_cluster"
    # Inside the window but off its first day: still a match.
    assert kind("sector cluster week", "2026-12-03", None, [XLK])["catalyst_kind"] == "calendar_matched"
    # Three days past the window end: a match; four: not.
    assert kind("sector cluster", "2026-12-09", None, [XLK])["catalyst_kind"] == "calendar_matched"
    assert kind("sector cluster", "2026-12-10", None, [XLK])["catalyst_kind"] == "unmatched_dated"


def test_macro_rows_count_only_for_their_own_release():
    cpi = kind("November CPI report", "2026-12-10", None, [CPI, FOMC])
    assert cpi["catalyst_kind"] == "calendar_matched"
    assert cpi["catalyst_matched_event_type"] == "cpi_release"
    fed = kind("Fed rate decision", "2026-12-09", None, [CPI, FOMC])
    assert fed["catalyst_matched_event_type"] == "fomc_meeting"
    other = kind("AMD partnership announcement", "2026-12-10", None, [CPI, FOMC])
    assert other["catalyst_kind"] == "unmatched_dated"
    assert {m["event_type"] for m in other["catalyst_kind_detail"]["wording_blind"]} == {
        "cpi_release", "fomc_meeting"}


def test_ticker_rows_and_earnings_dates_follow_the_wording():
    earn = {"scope": "ticker", "ticker": "MU", "event_type": "earnings", "event_date": "2026-12-17",
            "window_start": None, "window_end": None}
    conf = {**earn, "event_type": "conference", "event_date": "2026-12-03"}
    assert kind("MU earnings", "2026-12-16", None, [earn])["catalyst_matched_source"] == \
        "catalyst_events:ticker"
    assert kind("product launch", "2026-12-16", None, [earn])["catalyst_kind"] == "unmatched_dated"
    # A non-earnings ticker row counts whatever the wording.
    assert kind("investor day", "2026-12-03", None, [conf])["catalyst_kind"] == "calendar_matched"
    # Earnings dates count only when the text names earnings.
    assert kind("GTC keynote", "2026-11-20", NVDA_DATES, day=date(2026, 11, 10))["catalyst_kind"] == \
        "unmatched_dated"


def test_the_nearest_match_names_the_source_first_in_order_on_a_tie():
    dates = {"earnings_calendar": [date(2026, 12, 1)], "earnings_calendar_history": [date(2026, 11, 30)],
             "earnings_consensus_history": [date(2026, 11, 30)]}
    out = kind("earnings", "2026-11-30", dates, [{**XLK, "ticker": "XLK"}])
    assert out["catalyst_matched_source"] == "earnings_calendar_history"
    assert len(out["catalyst_kind_detail"]["matches"]) == 3


def test_ongoing_and_unparsed():
    for raw in ("ongoing", "2026-12"):
        out = kind("Resolution of regulatory investigations", raw)
        assert out["catalyst_kind"] == "ongoing", raw
        assert out["catalyst_days_ahead"] is None, raw
    for raw in (None, "", "soon", "2026-13-40", "Q1 2027"):
        out = kind("earnings", raw)
        assert out["catalyst_kind"] == "unparsed", raw
        assert out["catalyst_days_ahead"] is None, raw
    assert set(ec.CATALYST_KINDS) == {"already_released", "scheduled_today", "calendar_matched",
                                      "unmatched_dated", "ongoing", "unparsed"}


def test_a_failed_read_leaves_the_kind_none_only_when_it_could_have_matched():
    out = kind("earnings", "2026-11-30", None, [], {"dates": "error: TimeoutError", "events": "ok"})
    assert out["catalyst_kind"] is None
    assert "dates error: TimeoutError" in out["catalyst_kind_detail"]["reason"]
    # A match from the other read stands.
    got = kind("earnings", "2026-11-30", {"earnings_calendar": [date(2026, 11, 30)]}, None,
               {"dates": "ok", "events": "error: OperationalError"})
    assert got["catalyst_kind"] == "calendar_matched"
    # Past dates need no read.
    assert kind("earnings", "2026-11-18", None, None, {"dates": "error: X", "events": "error: X"})[
        "catalyst_kind"] == "already_released"
    # A failed earnings read doesn't matter when the text names no earnings (it is skipped).
    assert kind("GTC keynote", "2026-11-30", None, [], {"dates": "skipped: catalyst names no earnings"})[
        "catalyst_kind"] == "unmatched_dated"
    # Sector wording with the sector map not loaded: the ETF's rows were never read.
    unread = {"sector": "skipped: sector map not loaded in this process"}
    assert kind("sector cluster", "2026-11-30", None, [], unread)["catalyst_kind"] is None
    assert kind("earnings", "2026-11-30", {}, [], unread)["catalyst_kind"] == "unmatched_dated"
    assert kind("sector cluster", "2026-11-30", None, [], {"sector": "skipped: not an S&P 500 constituent"})[
        "catalyst_kind"] == "unmatched_dated"


def test_the_verdict_day_is_the_et_day():
    late = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)       # 22:00 ET on Fri 11-20
    out = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "earnings", "date": "2026-11-21"}}, late,
                                  read_dates=lambda *a: NVDA_DATES, read_events=lambda *a: [])
    assert out["catalyst_kind_detail"]["verdict_day_et"] == "2026-11-20"
    assert out["catalyst_kind_detail"]["verdict_time_et"] == "22:00"
    assert out["catalyst_days_ahead"] == 1
    assert out["catalyst_kind"] == "calendar_matched"


def test_the_check_reads_once_and_records_kind_on_every_shape():
    fake = types.SimpleNamespace(get_sp500_constituents=lambda: {"NVDA": {"gics_sector": "Information Technology"}},
                                 SECTOR_ETF_MAP={"Information Technology": "XLK"})
    now = datetime(2026, 11, 19, 16, 0, tzinfo=timezone.utc)
    rd, re_ = MagicMock(return_value=NVDA_DATES), MagicMock(return_value=[XLK])
    with patch.dict(sys.modules, {"ingestion.sp500_universe": fake}):
        out = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "NVIDIA Q4 earnings", "date": "2026-11-30"}},
                                      now, read_dates=rd, read_events=re_)
    rd.assert_called_once_with("NVDA", now)
    re_.assert_called_once_with("NVDA", "XLK", now)
    assert out["catalyst_kind"] == "unmatched_dated"
    assert out["matches_sector_cluster_date"] is True                 # the earnings field, same read
    assert out["catalyst_kind_detail"]["read_events"] == "ok"
    assert out["catalyst_kind_detail"]["read_sector"] == "ok"
    # No day date: no reads at all.
    rd.reset_mock()
    re_.reset_mock()
    out = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "probe", "date": "ongoing"}}, now,
                                  read_dates=rd, read_events=re_)
    assert rd.call_count == 0
    assert re_.call_count == 0
    assert out["catalyst_kind"] == "ongoing"
    # Every input shape carries the five fields.
    for bad in (None, {"catalyst": "x"}, object()):
        r = ec.earnings_claim_check("NVDA", bad, now, read_dates=rd, read_events=re_)
        assert set(r) >= {"catalyst_kind", "catalyst_days_ahead", "catalyst_matched_source",
                          "catalyst_matched_event_type", "catalyst_kind_detail"}
    with patch.object(ec, "catalyst_kind_of", side_effect=RuntimeError("boom")):
        r = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "e", "date": "2026-11-30"}}, now,
                                    read_dates=rd, read_events=re_)
    assert r["catalyst_kind"] is None
    assert "boom" in r["catalyst_kind_detail"]["error"]
    assert "earnings_claim_unmatched" in r                            # the earnings fields stand


def test_the_store_read_is_one_read_only_query_as_of(tmp_path):
    path = tmp_path / "cal.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE catalyst_events (scope TEXT, ticker TEXT, event_type TEXT, event_date TEXT, "
              "window_start TEXT, window_end TEXT, created_at TEXT, thesis_note TEXT)")
    old = "2026-10-01T00:00:00+00:00"
    c.executemany("INSERT INTO catalyst_events VALUES (?,?,?,?,?,?,?,?)", [
        ("ticker", "NVDA", "earnings", "2026-11-19", None, None, old, ""),
        ("ticker", "AMD", "earnings", "2026-11-03", None, None, old, ""),
        ("sector", "XLK", "sector_earnings_cluster", "2026-11-30", "2026-11-30", "2026-12-06", old, "IT"),
        ("sector", "XLF", "sector_earnings_cluster", "2026-11-30", None, None, old, ""),
        ("macro", "MARKET", "cpi_release", "2026-12-10", None, None, old, ""),
        ("macro", "MARKET", "fomc_meeting", "2027-01-27", None, None, "2026-12-01T00:00:00+00:00", ""),
    ])
    c.commit()
    c.close()
    now = datetime(2026, 11, 19, 16, 0, tzinfo=timezone.utc)
    got = ec.catalyst_events_as_of("NVDA", "XLK", now, path=path)
    assert [(e["scope"], e["ticker"]) for e in got] == [
        ("ticker", "NVDA"), ("sector", "XLK"), ("macro", "MARKET")]   # created after now is left out
    no_etf = ec.catalyst_events_as_of("NVDA", None, now, path=path)
    assert [e["scope"] for e in no_etf] == ["ticker", "macro"]
    assert ec.sector_cluster_windows("XLK", now, path=path)[0]["window_end"] == "2026-12-06"
    missing = tmp_path / "none.db"
    with pytest.raises(sqlite3.OperationalError):
        ec.catalyst_events_as_of("NVDA", "XLK", now, path=missing)
    assert not missing.exists()                                       # read-only: never created
    src = inspect.getsource(ec.catalyst_events_as_of)
    assert "mode=ro" in src
    assert src.count(".execute(") == 1


def test_batch_verify_check():
    from scripts import batch_verify as bv
    good = {"catalyst_kind": "unmatched_dated", "catalyst_days_ahead": 11, "catalyst_matched_source": None,
            "catalyst_matched_event_type": None, "catalyst_kind_detail": {"read_events": "ok"}}
    c = bv.check_catalyst_kind([{"id": 1, "catalyst_check": good}])
    assert c["status"] == "OK"
    assert c["reads"]["catalyst_kind"]["unmatched_dated"] == 1
    assert bv.check_catalyst_kind([])["status"] == "WARN"
    for k in bv.CATALYST_KIND_REQUIRED:
        bad = {kk: v for kk, v in good.items() if kk != k}
        assert bv.check_catalyst_kind([{"id": 2, "catalyst_check": bad}])["status"] == "FAIL", k
    odd = bv.check_catalyst_kind([{"id": 3, "catalyst_check": {**good, "catalyst_kind": "soon"}}])
    assert odd["status"] == "FAIL"
    assert "not a kind" in odd["why"]
    errs = [{"id": i, "catalyst_check": {**good, "catalyst_kind": None,
                                         "catalyst_kind_detail": {"read_events": "error: OperationalError"}}}
            for i in range(3)] + [{"id": 9, "catalyst_check": good}]
    c = bv.check_catalyst_kind(errs)
    assert c["status"] == "FAIL"
    assert "read_events" in c["why"]
    # The writer's own output passes, a day date or not.
    for raw in ("2026-11-30", "ongoing"):
        out = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "earnings", "date": raw}},
                                      datetime(2026, 11, 19, 16, tzinfo=timezone.utc),
                                      read_dates=lambda *a: {}, read_events=lambda *a: [])
        assert bv.check_catalyst_kind([{"id": 4, "catalyst_check": out}])["status"] == "OK", raw
    assert set(bv.CATALYST_KIND_VALUES) == set(ec.CATALYST_KINDS) | {None}


def test_registry_stage_is_always_on_full_verdicts():
    from scripts import shadow_registry_check as src
    st = next(s for s in src.STAGES if s["stage"] == "catalyst_kind")
    assert st["expect_rows"] == "always"
    assert st["switch"] is None
    assert src.writer_present(src.ROOT, st["writer"]) is True
    assert st["sources"][0][0] == "signals"
    assert "catalyst_kind" in st["sources"][0][1]
    assert "%" not in st["sources"][0][1]                             # psycopg2 params
