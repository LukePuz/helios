"""Record-only earnings-claim check (earnings_claim.py): does a verdict
catalyst that names earnings have a date any earnings source knows for the
ticker, and does it fall in a sector earnings-cluster window? Merged into
decision.catalyst_check; nothing in the prompt, gates or routing reads it."""
import inspect
import json
import sqlite3
import sys
import types
from contextlib import ExitStack
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agents import analyst
import earnings_claim as ec

NOW = datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc)
_XLK_1130 = {"event_date": "2026-11-30", "window_start": "2026-11-30", "window_end": "2026-12-06",
             "note": "3 Information Technology constituents report earnings this week: CRM, CRWD, MRVL"}
_XLK_EVENT = {"scope": "sector", "ticker": "XLK", "event_type": "sector_earnings_cluster", **_XLK_1130}
_NVDA_DATES = {"earnings_calendar": [date(2026, 8, 27), date(2026, 11, 19)],
               "earnings_calendar_history": [date(2026, 11, 19)],
               "earnings_consensus_history": [date(2026, 11, 19)]}


def test_the_audit_shape_is_unmatched_and_on_the_cluster_week():
    out = ec.evaluate("NVIDIA's Q4 earnings report", "2026-11-30", _NVDA_DATES, [_XLK_1130])
    assert out["earnings_claim_unmatched"] is True
    assert out["matches_sector_cluster_date"] is True
    assert out["calendar_source_used"] is None
    d = out["earnings_claim"]
    assert d["names_earnings"] is True and d["names_sector_or_cluster"] is False
    assert d["nearest_days"] == {"earnings_calendar": -11, "earnings_calendar_history": -11,
                                 "earnings_consensus_history": -11}
    assert d["n_dates"] == {"earnings_calendar": 2, "earnings_calendar_history": 1,
                            "earnings_consensus_history": 1}
    assert d["sector_window"]["on_event_date"] is True and d["tol_days"] == 3


def test_a_date_within_three_days_matches_the_first_source_in_order():
    out = ec.evaluate("NVDA earnings report", "2026-11-17", _NVDA_DATES, [_XLK_1130])
    assert out["earnings_claim_unmatched"] is False
    assert out["calendar_source_used"] == "earnings_calendar"
    assert out["matches_sector_cluster_date"] is False
    only_hist = {**_NVDA_DATES, "earnings_calendar": []}
    out = ec.evaluate("NVDA earnings report", "2026-11-22", only_hist, [])
    assert out["calendar_source_used"] == "earnings_calendar_history"
    only_yf = {"earnings_calendar": [], "earnings_calendar_history": [],
               "earnings_consensus_history": [date(2026, 11, 19)]}
    assert ec.evaluate("earnings", "2026-11-16", only_yf, [])["calendar_source_used"] == \
        "earnings_consensus_history"
    # Four days off is outside the window.
    assert ec.evaluate("earnings", "2026-11-23", only_yf, [])["earnings_claim_unmatched"] is True
    # A date in the cluster week that is also the real date: matched, and in the window.
    real = ec.evaluate("CRM earnings", "2026-12-02", {"earnings_calendar": [date(2026, 12, 2)]},
                       [_XLK_1130])
    assert real["earnings_claim_unmatched"] is False and real["matches_sector_cluster_date"] is True
    assert real["earnings_claim"]["sector_window"]["on_event_date"] is False


def test_no_known_date_at_all_is_unmatched_with_zero_counts():
    out = ec.evaluate("earnings report", "2026-11-30", {}, None)
    assert out["earnings_claim_unmatched"] is True
    assert out["earnings_claim"]["n_dates"] == {s: 0 for s in ec.SOURCES}
    assert out["matches_sector_cluster_date"] is None          # the store was not read


def test_which_texts_name_earnings():
    yes = ("NVDA earnings report", "Q3 earnings report and guidance", "EPS beat expected",
           "Q4 FY27 results", "fiscal Q3 report", "results for the quarter",
           "XLK sector earnings cluster with CRM, CRWD, MRVL reporting",
           "Nvidia earnings call where the CEO transition is discussed")
    no = ("CPI report", "jobs report on Friday", "Phase 3 trial results", "AMD Partnership Announcement",
          "TD Cowen raised price target", "", None)
    for t in yes:
        assert ec.evaluate(t, "2026-11-30", {}, [])["earnings_claim"]["names_earnings"], t
    for t in no:
        out = ec.evaluate(t, "2026-11-30", {}, [])
        assert not out["earnings_claim"]["names_earnings"], t
        assert out["earnings_claim_unmatched"] is None, t
    assert ec.evaluate(yes[6], "2026-11-30", {}, [])["earnings_claim"]["names_sector_or_cluster"]


def test_dates_that_are_not_a_day_give_none_with_a_reason():
    for raw, why in (("2026-12", "month-only catalyst date"), ("ongoing", "ongoing catalyst"),
                     ("soon", "unparseable catalyst date"), ("2026-13-40", "unparseable catalyst date"),
                     (None, "no catalyst date")):
        out = ec.evaluate("earnings report", raw, _NVDA_DATES, [_XLK_1130])
        assert out["earnings_claim_unmatched"] is None and out["matches_sector_cluster_date"] is None
        assert out["earnings_claim"]["why_no_day"] == why


def test_the_check_reads_only_what_it_needs_and_fails_open():
    rd, rw = MagicMock(return_value=_NVDA_DATES), MagicMock(return_value=[_XLK_EVENT])
    fake = types.SimpleNamespace(get_sp500_constituents=lambda: {"NVDA": {"gics_sector": "Information Technology"}},
                                 SECTOR_ETF_MAP={"Information Technology": "XLK"})
    parsed = {"catalyst": {"event": "NVDA earnings report", "date": "2026-11-30"}}
    with patch.dict(sys.modules, {"ingestion.sp500_universe": fake}):
        out = ec.earnings_claim_check("NVDA", parsed, NOW, read_dates=rd, read_events=rw)
        assert out["earnings_claim_unmatched"] is True and out["matches_sector_cluster_date"] is True
        assert out["earnings_claim"]["sector_etf"] == "XLK"
        assert out["earnings_claim"]["read_dates"] == out["earnings_claim"]["read_sector"] == "ok"
        rd.assert_called_once_with("NVDA", NOW)
        rw.assert_called_once_with("NVDA", "XLK", NOW)
        # No earnings wording: no Postgres read, the sector window still read.
        rd.reset_mock()
        out = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "GTC keynote", "date": "2026-11-30"}},
                                      NOW, read_dates=rd, read_events=rw)
        assert rd.call_count == 0 and out["earnings_claim_unmatched"] is None
        assert out["earnings_claim"]["read_dates"] == "skipped: catalyst names no earnings"
        assert out["matches_sector_cluster_date"] is True
        # Failing reads become error labels, never an exception.
        out = ec.earnings_claim_check("NVDA", parsed, NOW,
                                      read_dates=MagicMock(side_effect=TimeoutError("x")),
                                      read_events=MagicMock(side_effect=sqlite3.OperationalError("x")))
        assert out["earnings_claim_unmatched"] is None and out["matches_sector_cluster_date"] is None
        assert out["earnings_claim"]["read_dates"] == "error: TimeoutError"
        assert out["earnings_claim"]["read_sector"] == "error: OperationalError"
        # Not a constituent: the store is read for the ticker and macro rows only.
        rw.reset_mock()
        out = ec.earnings_claim_check("ZS", parsed, NOW, read_dates=rd, read_events=rw)
        rw.assert_called_once_with("ZS", None, NOW)
        assert out["matches_sector_cluster_date"] is None
        assert out["earnings_claim"]["read_sector"] == "skipped: not an S&P 500 constituent"
    # Sector map not loaded in this process: never imported (that import can fetch).
    with patch.dict(sys.modules, {"ingestion.sp500_universe": None}):
        assert ec.sector_etf("NVDA") == (None, "sector map not loaded in this process")
    no_day = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "earnings", "date": "2026-12"}},
                                     NOW, read_dates=rd, read_events=rw)
    assert no_day["earnings_claim"]["read_dates"] == "skipped: no day date"
    for bad in (None, {"catalyst": "x"}, object()):
        r = ec.earnings_claim_check("NVDA", bad, NOW, read_dates=rd, read_events=rw)
        assert set(r) >= {"earnings_claim_unmatched", "matches_sector_cluster_date",
                          "calendar_source_used", "earnings_claim"}
    with patch.object(ec, "evaluate", side_effect=RuntimeError("boom")):
        r = ec.earnings_claim_check("NVDA", parsed, NOW, read_dates=rd, read_events=rw)
    assert r["earnings_claim_unmatched"] is None and "boom" in r["earnings_claim"]["error"]


def test_sector_windows_from_the_store_read_only_and_as_of(tmp_path):
    path = tmp_path / "cal.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE catalyst_events (scope TEXT, ticker TEXT, event_type TEXT, event_date TEXT, "
              "window_start TEXT, window_end TEXT, created_at TEXT, thesis_note TEXT)")
    rows = [("sector", "XLK", "sector_earnings_cluster", "2026-11-30", "2026-11-30", "2026-12-06",
             "2026-10-11T20:32:25+00:00", "3 IT constituents"),
            ("sector", "XLK", "sector_earnings_cluster", "2027-01-25", "2027-01-25", "2027-01-31",
             "2026-12-01T00:00:00+00:00", "later"),
            ("sector", "XLK", "sector_regulatory", "2026-11-30", None, None, "2026-10-01T00:00:00+00:00", ""),
            ("sector", "XLF", "sector_earnings_cluster", "2026-11-30", None, None, "2026-10-01T00:00:00+00:00", ""),
            ("ticker", "XLK", "sector_earnings_cluster", "2026-11-30", None, None, "2026-10-01T00:00:00+00:00", "")]
    c.executemany("INSERT INTO catalyst_events VALUES (?,?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()
    got = ec.sector_cluster_windows("XLK", NOW, path=path)
    assert [w["event_date"] for w in got] == ["2026-11-30"]      # created after NOW is left out
    assert got[0]["window_end"] == "2026-12-06"
    missing = tmp_path / "none.db"
    with pytest.raises(sqlite3.OperationalError):
        ec.sector_cluster_windows("XLK", NOW, path=missing)
    assert not missing.exists()                                   # read-only: never created


def test_the_db_read_is_bounded_and_read_only():
    import db
    src = inspect.getsource(db.get_earnings_claim_dates)
    assert "SET LOCAL statement_timeout" in src and "rollback()" in src and "_put_conn" in src
    for word in ("INSERT", "UPDATE ", "DELETE", "commit("):
        assert word not in src, word
    assert ec.EARNINGS_CLAIM_READ_TIMEOUT_MS == 500
    msrc = inspect.getsource(ec)
    for word in ("INSERT", "UPDATE ", "DELETE", "commit(", "requests", "yfinance",
                 "import ingestion", "from ingestion"):
        assert word not in msrc, word


def _analyze(check=None):
    parsed = {
        "action": "HOLD", "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {},
        "catalyst": {"event": "NVIDIA's Q4 earnings report", "date": "2026-11-30"},
        "valuation": {"bull_case_target": "$130", "downside_scenario": "$90"},
        "expected_return_pct": 8.0, "key_risks": [], "pass_to_risk": False,
        "upside_downside_ratio": 2.4, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}],
    }
    log_signal = MagicMock()
    ollama = MagicMock(return_value=json.dumps(parsed))
    patches = [
        patch("agents.analyst._call_ollama", ollama),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.get_trade_ladder", return_value=None),
        patch("db.get_next_earnings", return_value=None),
        patch("db.get_earnings_claim_dates", return_value=_NVDA_DATES),
        patch("earnings_claim.catalyst_events_as_of", return_value=[_XLK_EVENT]),
        patch("earnings_claim.sector_etf", return_value=("XLK", None)),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
    ]
    if check is not None:
        patches.append(patch("earnings_claim.earnings_claim_check", check))
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "NVDA", "role": "Compounder", "tier": "Core",
                                  "sector": "Semis", "held": True},
                                 situation_brief="Current Price: $100.00\nBRIEF",
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"], ollama.call_args


_FIELDS = ("earnings_claim_unmatched", "matches_sector_cluster_date", "calendar_source_used",
           "catalyst_kind", "catalyst_days_ahead", "catalyst_matched_source",
           "catalyst_matched_event_type", "catalyst_kind_detail")


def _without_clock(d):
    if isinstance(d, dict):
        return {k: _without_clock(v) for k, v in d.items() if not k.endswith("_at")}
    return d


def test_merged_into_catalyst_check_and_nothing_else_changes():
    result, decision, sent = _analyze()
    cc = decision["catalyst_check"]
    assert cc["earnings_claim_unmatched"] is True and cc["matches_sector_cluster_date"] is True
    assert cc["calendar_source_used"] is None
    assert cc["earnings_claim"]["sector_etf"] == "XLK"
    assert cc["catalyst_kind"] in ("unmatched_dated", "already_released")   # 11-30 vs the clock
    assert cc["catalyst_kind_detail"]["read_events"] == "ok"
    assert "model_date" in cc and "days_model_vs_calendar" in cc    # the existing fields stay
    off, decision_off, sent_off = _analyze(check=MagicMock(side_effect=RuntimeError("x")))
    cc_off = decision_off["catalyst_check"]
    assert cc_off["earnings_claim"] == {"error": "RuntimeError"}
    assert not any(k in cc_off for k in _FIELDS)
    assert sent == sent_off                                       # the prompt sent is the same
    for r in (result, off):
        r.pop("timestamp")
    assert _without_clock(result) == _without_clock(off)
    assert not any(k in result for k in _FIELDS + ("earnings_claim",))
    rest = {k: v for k, v in decision.items() if k != "catalyst_check"}
    rest_off = {k: v for k, v in decision_off.items() if k != "catalyst_check"}
    assert rest == rest_off


def test_not_in_the_prompt_path():
    for f in (analyst._build_prompt, analyst._prompt_template_full_text):
        assert "earnings_claim" not in inspect.getsource(f)
    import context_engine
    import knowledge_loader
    for m in (context_engine, knowledge_loader):
        assert "earnings_claim" not in inspect.getsource(m)
    assert "earnings_claim" not in analyst._BRIEF_MODULES


def test_batch_verify_check():
    from scripts import batch_verify as bv
    good = {"earnings_claim_unmatched": True, "matches_sector_cluster_date": True,
            "calendar_source_used": None,
            "earnings_claim": {"read_dates": "ok", "read_sector": "ok"}}
    c = bv.check_earnings_claim([{"id": 1, "ticker": "NVDA", "catalyst_check": good}])
    assert c["status"] == "OK" and c["reads"]["earnings_claim_unmatched"]["true"] == 1
    assert bv.check_earnings_claim([])["status"] == "WARN"
    for k in bv.EARNINGS_CLAIM_REQUIRED:
        bad = {kk: v for kk, v in good.items() if kk != k}
        assert bv.check_earnings_claim([{"id": 2, "catalyst_check": bad}])["status"] == "FAIL", k
    old = {"model_date": "2026-11-30"}                           # a row from before the change
    assert bv.check_earnings_claim([{"id": 3, "catalyst_check": old}])["status"] == "FAIL"
    errs = [{"id": i, "catalyst_check": {**good, "earnings_claim": {"read_dates": "error: TimeoutError",
                                                                   "read_sector": "ok"}}}
            for i in range(3)] + [{"id": 9, "catalyst_check": good}]
    c = bv.check_earnings_claim(errs)
    assert c["status"] == "FAIL" and "read_dates" in c["why"]
    # The writer's own output passes.
    out = ec.earnings_claim_check("NVDA", {"catalyst": {"event": "earnings", "date": "2026-11-30"}},
                                  NOW, read_dates=lambda *a: {}, read_events=lambda *a: [])
    assert bv.check_earnings_claim([{"id": 4, "catalyst_check": out}])["status"] == "OK"
    src = inspect.getsource(bv._fetch_verdicts)
    assert "SET TRANSACTION READ ONLY" in src and "rollback()" in src and "commit" not in src


def test_registry_stage_is_conditional_on_earnings_verdicts():
    from scripts import shadow_registry_check as src
    st = next(s for s in src.STAGES if s["stage"] == "earnings_claim")
    assert st["expect_rows"] == "conditional" and st["switch"] is None
    assert src.writer_present(src.ROOT, st["writer"]) is True
    assert "earnings" in st["trigger"][1] and st["trigger"][0] == "signals"
    assert "earnings_claim_unmatched" in st["sources"][0][1]
    assert "%" not in st["trigger"][1] + st["sources"][0][1]    # psycopg2 params
