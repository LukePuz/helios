"""scripts/catalyst_claim_readout.py: the claim counts and the migration
categories (pure parts), the read-only fetch, and the port guard."""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import catalyst_claim_readout as ccr  # noqa: E402

AT = datetime(2026, 12, 2, 15, 0, tzinfo=timezone.utc)          # 10:00 ET on 12-02
OLD = "2026-09-11T20:59:16+00:00"


def _cc(unmatched=None, sector=None, names=True, etf="XLK", **extra):
    return {"earnings_claim_unmatched": unmatched, "matches_sector_cluster_date": sector,
            "earnings_claim": {"names_earnings": names, "sector_etf": etf}, **extra}


def _row(ticker="NVDA", date="2026-12-31", event="Q4 Earnings Report", ts=AT, cc=None, id_=1):
    return {"id": id_, "ticker": ticker, "signal": "HOLD", "ts": ts,
            "catalyst_check": _cc() if cc is None else cc,
            "catalyst": {"date": date, "event": event}}


def _ev(scope="ticker", ticker="CRWD", kind="earnings", d="2026-12-01", ws=None, we=None,
        created=OLD, superseded=None, conf="estimated"):
    return {"scope": scope, "ticker": ticker, "event_type": kind, "event_date": d,
            "window_start": ws, "window_end": we, "confidence": conf, "created_at": created,
            "superseded_by": superseded}


CAL = [_ev(), _ev(ticker="CRM", d="2026-12-02"),
       _ev(scope="sector", ticker="XLK", kind="sector_earnings_cluster", d="2026-11-30",
           ws="2026-11-30", we="2026-12-06", conf="confirmed"),
       _ev(scope="macro", ticker="MARKET", kind="jobs_report", d="2026-12-04", conf="confirmed"),
       _ev(scope="macro", ticker="MARKET", kind="fomc_meeting", d="2026-12-09", conf="confirmed"),
       _ev(ticker="AVGO", d="2026-12-09")]


# ---- claims ---------------------------------------------------------------

def test_claims_counts_each_flag_per_ticker_and_in_total():
    rows = [_row(cc=_cc(True, True)), _row(cc=_cc(True, False)), _row(cc=_cc(None, None)),
            _row(ticker="anet", cc=_cc(False, True))]
    c = ccr.claims(rows)
    n = c["tickers"]["NVDA"]
    assert n["verdicts"] == 3
    assert n["earnings_claim_unmatched"] == {"true": 2, "false": 0, "none": 1, "not recorded": 0}
    assert n["matches_sector_cluster_date"] == {"true": 1, "false": 1, "none": 1, "not recorded": 0}
    assert n["both_true"] == 1
    assert c["tickers"]["ANET"]["earnings_claim_unmatched"]["false"] == 1
    assert c["total"]["verdicts"] == 4 and c["total"]["both_true"] == 1


def test_claims_reads_the_stored_flags_and_never_recomputes():
    # A date the calendar would match still counts as the record stored it.
    c = ccr.claims([_row(date="2026-12-01", cc=_cc(True, False))])
    assert c["total"]["earnings_claim_unmatched"]["true"] == 1


def test_a_row_without_the_keys_is_not_recorded():
    c = ccr.claims([_row(cc={}), _row(cc={"x": 1})])
    assert c["total"]["earnings_claim_unmatched"]["not recorded"] == 2
    assert c["total"]["both_true"] == 0


def test_a_non_bool_flag_is_not_recorded():
    c = ccr.claims([_row(cc=_cc("yes", 1))])
    assert c["total"]["earnings_claim_unmatched"]["not recorded"] == 1
    assert c["total"]["matches_sector_cluster_date"]["not recorded"] == 1


def test_claims_tally_the_catalyst_dates_most_named_first():
    rows = [_row(date="2026-11-30"), _row(date="2026-11-30"), _row(date="ongoing"), _row(date=None)]
    assert list(ccr.claims(rows)["total"]["dates"].items()) == [
        ("2026-11-30", 2), ("none", 1), ("ongoing", 1)]


def test_claims_date_falls_back_to_the_records_model_date():
    r = _row(date=None, cc=_cc(model_date="2026-11-30"))
    assert ccr.claims([r])["total"]["dates"] == {"2026-11-30": 1}


# ---- migration categories -------------------------------------------------

@pytest.mark.parametrize("raw,why", [("ongoing", "ongoing catalyst"), ("2026-12", "month-only catalyst date"),
                                     (None, "no catalyst date"), ("Q4", "unparseable catalyst date"),
                                     ("2026-02-30", "unparseable catalyst date")])
def test_no_day_date(raw, why):
    x = ccr.classify(_row(date=raw), CAL)
    assert x["category"] == "no_day_date" and x["why_no_day"] == why and x["lines"] == []


def test_past_date_is_before_the_verdicts_et_day_even_with_a_line_on_it():
    x = ccr.classify(_row(ticker="CRWD", date="2026-12-01"), CAL)
    assert x["category"] == "past_date"
    assert [ln["kind"] for ln in x["lines"]] == ["own_earnings_line", "sector_window"]


def test_the_verdicts_own_day_is_not_past():
    x = ccr.classify(_row(ticker="CRM", date="2026-12-02"), CAL)
    assert x["category"] == "own_earnings_line"


def test_et_day_not_utc_day_sets_the_verdict_day():
    late = datetime(2026, 12, 3, 2, 0, tzinfo=timezone.utc)      # 21:00 ET on 12-02
    x = ccr.classify(_row(ticker="CRM", date="2026-12-02", ts=late), CAL)
    assert x["verdict_day"] == "2026-12-02" and x["category"] == "own_earnings_line"


def test_own_earnings_line_needs_the_verdicts_own_ticker():
    x = ccr.classify(_row(ticker="NVDA", date="2026-12-09", cc=_cc(etf=None)), CAL)
    # AVGO's row is not NVDA's; the FOMC macro row is on the date.
    assert x["category"] == "macro_line"
    assert [ln["kind"] for ln in x["lines"]] == ["macro_line"]


def test_own_earnings_line_wins_over_a_macro_row_on_the_same_date():
    x = ccr.classify(_row(ticker="AVGO", date="2026-12-09", cc=_cc(etf=None)), CAL)
    assert x["category"] == "own_earnings_line"
    assert {ln["kind"] for ln in x["lines"]} == {"own_earnings_line", "macro_line"}


def test_sector_window_uses_the_window_not_only_the_event_date():
    x = ccr.classify(_row(date="2026-12-03"), CAL)
    assert x["category"] == "sector_window"
    assert x["lines"][0]["date"] == "2026-11-30" and x["lines"][0]["days_off"] == 0


def test_sector_window_needs_the_records_sector_etf():
    assert ccr.classify(_row(date="2026-12-03", cc=_cc(etf="XLF")), CAL)["category"] == "invented_earnings"
    assert ccr.classify(_row(date="2026-12-03", cc=_cc(etf=None)), CAL)["category"] == "invented_earnings"


def test_sector_window_wins_over_a_macro_row():
    x = ccr.classify(_row(date="2026-12-04"), CAL)
    assert x["category"] == "sector_window"
    assert {ln["kind"] for ln in x["lines"]} == {"sector_window", "macro_line"}


def test_own_other_line():
    cal = CAL + [_ev(ticker="NVDA", kind="product_launch", d="2026-12-15")]
    assert ccr.classify(_row(date="2026-12-15", cc=_cc(etf=None)), cal)["category"] == "own_other_line"


def test_invented_earnings_and_unmatched_date_split_on_the_wording():
    a = ccr.classify(_row(date="2026-12-31"), CAL)
    b = ccr.classify(_row(date="2026-12-31", event="Product launch", cc=_cc(names=False)), CAL)
    assert (a["category"], b["category"]) == ("invented_earnings", "unmatched_date")


def test_names_earnings_falls_back_to_the_earnings_claim_regex():
    cc = {"earnings_claim_unmatched": None}
    assert ccr.classify(_row(date="2026-12-31", event="Q4 FY27 earnings", cc=cc), CAL)["names_earnings"] is True
    assert ccr.classify(_row(date="2026-12-31", event="CPI report", cc=cc), CAL)["names_earnings"] is False


def test_a_row_created_after_the_verdict_does_not_count():
    cal = [_ev(ticker="NVDA", d="2026-12-31", created="2026-12-02T16:00:00+00:00")]
    x = ccr.classify(_row(date="2026-12-31"), cal)
    assert x["category"] == "invented_earnings" and x["own_earnings_rows"] == 0


def test_superseded_rows_count_and_are_flagged():
    cal = [_ev(ticker="NVDA", d="2026-12-31", superseded="abc")]
    x = ccr.classify(_row(date="2026-12-31"), cal)
    assert x["category"] == "own_earnings_line" and x["lines"][0]["superseded"] is True
    assert x["own_earnings_rows"] == 0


def test_own_earnings_rows_counts_the_tickers_unsuperseded_rows():
    cal = [_ev(ticker="ZS", d="2026-11-24"), _ev(ticker="ZS", d="2026-12-02"), _ev(ticker="CRM")]
    assert ccr.classify(_row(ticker="ZS", date="2026-12-02"), cal)["own_earnings_rows"] == 2


def test_tolerance_widens_the_match_and_records_days_off():
    assert ccr.classify(_row(ticker="CRM", date="2026-12-03"), CAL)["category"] == "sector_window"
    x = ccr.classify(_row(ticker="CRM", date="2026-12-03", cc=_cc(etf=None)), CAL, tol=1)
    assert x["category"] == "own_earnings_line" and x["lines"][0]["days_off"] == -1
    assert ccr.classify(_row(ticker="CRM", date="2026-12-03", cc=_cc(etf=None)), CAL)["category"] == \
        "invented_earnings"


def test_migration_tallies_by_category_and_by_catalyst_date():
    rows = [_row(date="2026-12-31", id_=1), _row(date="2026-12-31", id_=2),
            _row(ticker="CRM", date="2026-12-02", cc=_cc(etf=None), id_=3),
            _row(date="2026-11-30", id_=4), _row(date="ongoing", id_=5)]
    m = ccr.migration(rows, CAL)
    assert m["verdicts"] == 5 and m["tol_days"] == 0
    assert list(m["by_category"]) == list(ccr.CATEGORIES)
    assert m["by_category"]["invented_earnings"] == {"verdicts": 2, "names_earnings": 2,
                                                     "tickers": {"NVDA": 2}}
    assert m["by_category"]["past_date"]["verdicts"] == 1
    assert m["by_catalyst_date"]["2026-12-02"]["line_kinds"] == ["own_earnings_line:earnings"]
    assert m["by_catalyst_date"]["2026-12-31"]["category"] == {"invented_earnings": 2}
    assert sum(v["verdicts"] for v in m["by_category"].values()) == 5
    assert [r["id"] for r in m["rows"]] == [1, 2, 3, 4, 5]


# ---- render, windows, fetch, calendar, guard ------------------------------

def test_render_has_both_sections_and_a_total_row():
    res = {"env": "live", "claim_window": ["2026-11-24", "2026-11-30"],
           "claims": ccr.claims([_row()]), "migration_window": ["2026-12-01", "2026-12-04"],
           "migration": ccr.migration([_row()], CAL)}
    text = ccr.render(res)
    assert "## Claims, 2026-11-24 to 2026-11-30 ET" in text
    assert "| all | 1 |" in text
    assert "## Where the claim goes, 2026-12-01 to 2026-12-04 ET" in text
    assert "| invented_earnings | 1 | 1 | NVDA 1 |" in text


def test_bounds_are_et_days_inclusive():
    lo, hi = ccr._bounds("2026-11-24", "2026-11-30")
    assert lo == datetime(2026, 11, 24, 5, 0, tzinfo=timezone.utc)
    assert hi == datetime(2026, 12, 1, 5, 0, tzinfo=timezone.utc)


class _Cur:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows):
        self.log, self.rows, self.rolled_back = [], rows, 0

    def cursor(self):
        return _Cur(self.log, self.rows)

    def rollback(self):
        self.rolled_back += 1


def test_fetch_is_read_only_rolled_back_and_bounded():
    conn = _Conn([(7, "NVDA", "HOLD", AT, _cc(), {"date": "2026-12-31"})])
    rows = ccr.fetch_verdicts(conn, "live", "2026-12-01", "2026-12-04", ("SCREENED_OUT",))
    assert conn.log[0][0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1][0].startswith("SET LOCAL statement_timeout")
    sql, p = conn.log[2]
    assert sql == ccr.VERDICTS_SQL and p["env"] == "live" and p["non"] == ["SCREENED_OUT"]
    assert (p["lo"], p["hi"]) == ccr._bounds("2026-12-01", "2026-12-04")
    assert conn.rolled_back == 1
    assert rows == [{"id": 7, "ticker": "NVDA", "signal": "HOLD", "ts": AT, "catalyst_check": _cc(),
                     "catalyst": {"date": "2026-12-31"}}]


def test_run_splits_one_read_into_the_two_windows(tmp_path):
    path = tmp_path / "cal.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE catalyst_events (scope, ticker, event_type, event_date, window_start, "
              "window_end, confidence, created_at, superseded_by)")
    c.execute("INSERT INTO catalyst_events VALUES ('ticker','CRM','earnings','2026-12-02',NULL,NULL,"
              "'estimated',?,NULL)", (OLD,))
    c.commit()
    c.close()
    nov = datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc)
    conn = _Conn([(1, "NVDA", "HOLD", nov, _cc(True, True), {"date": "2026-11-30"}),
                  (2, "CRM", "HOLD", AT, _cc(False, None, etf=None), {"date": "2026-12-02"})])
    res = ccr.run(conn, "live", "both", ("2026-11-24", "2026-11-30"), ("2026-12-01", "2026-12-04"),
                  0, path, ())
    assert res["claims"]["total"]["verdicts"] == 1
    assert res["migration"]["verdicts"] == 1
    assert res["migration"]["rows"][0]["category"] == "own_earnings_line"
    p = conn.log[2][1]
    assert (p["lo"], p["hi"]) == (ccr._bounds("2026-11-24", "2026-11-24")[0],
                                  ccr._bounds("2026-12-04", "2026-12-04")[1])


def test_run_claims_only_never_opens_the_calendar(tmp_path):
    res = ccr.run(_Conn([]), "live", "claims", ("2026-11-24", "2026-11-30"),
                  ("2026-12-01", "2026-12-04"), 0, tmp_path / "missing.db", ())
    assert "migration" not in res and res["claims"]["total"]["verdicts"] == 0
    assert not (tmp_path / "missing.db").exists()


def test_read_calendar_never_creates_the_store(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        ccr.read_calendar(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_main_refuses_without_the_pooler_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert ccr.main([]) == 2
    assert "6543" in capsys.readouterr().err


def test_default_windows_are_the_audits():
    assert ccr.CLAIM_WINDOW == ("2026-11-24", "2026-11-30")
    assert ccr.MIGRATION_WINDOW == ("2026-12-01", "2026-12-04")


# ---- named cases ------------------------------------------------------------

def test_msft_plan_3_is_a_named_case():
    (case,) = [c for c in ccr.NAMED_CASES if c["name"] == "MSFT plan 3"]
    assert case["ticker"] == "MSFT" and case["plan_id"] == 3 and case["date"] == "2026-11-30"
    assert "MSFT plan 3" in ccr.__doc__


def test_the_named_case_counts_its_tickers_verdicts_in_the_window():
    """A 12-02 MSFT verdict still dating Q1 FY27 earnings to 11-30 is past_date;
    one that moved to 12-03 lands in the XLK cluster window (11-30..12-06)."""
    rows = [_row(ticker="MSFT", date="2026-11-30", event="Q1 FY27 earnings", id_=11),
            _row(ticker="MSFT", date="2026-12-03", event="Q1 FY27 earnings", id_=12),
            _row(ticker="MSFT", date="Q1 2027", event="Azure growth", id_=13),
            _row(ticker="NVDA", date="2026-11-30", id_=14)]
    m = ccr.migration(rows, CAL)
    (c,) = m["named_cases"]
    assert c["verdicts"] == 3 and c["naming_the_date"] == 1 and c["ids"] == [11, 12, 13]
    assert c["category"] == {"no_day_date": 1, "past_date": 1, "sector_window": 1}
    assert c["catalyst_dates"] == {"2026-11-30": 1, "2026-12-03": 1, "Q1 2027": 1}
    assert sum(v["verdicts"] for v in m["by_category"].values()) == 4    # tallies unchanged


def test_a_named_case_with_no_verdicts_still_shows():
    m = ccr.migration([_row()], CAL)
    assert m["named_cases"][0]["verdicts"] == 0 and m["named_cases"][0]["category"] == {}
    assert ccr.migration([_row()], CAL, cases=())["named_cases"] == []


def test_render_lists_the_named_cases():
    res = {"env": "live", "migration_window": ["2026-12-01", "2026-12-04"],
           "migration": ccr.migration([_row(ticker="MSFT", date="2026-11-30", id_=11)], CAL)}
    text = ccr.render(res)
    assert "Named cases:" in text
    assert "| MSFT plan 3 (MSFT) | 2026-11-30 | 1 | 1 | past_date 1 | 2026-11-30 1 |" in text
