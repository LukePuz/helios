"""The weekly candidate scan says how old its price bars are (audit A1031-0453-5)."""
from datetime import date
from unittest.mock import patch

from agents import candidate_scanner as cs


def test_bars_age_is_per_ticker():
    """SEF 10-31: a few fresh names must not hide a mostly stale universe."""
    fresh = [{"date": "2026-10-30"}]
    stale = [{"date": "2026-10-16"}]
    bars = {"A": fresh, "B": stale, "C": stale, "D": stale}
    newest, median, share = cs._bars_age(bars, today=date(2026, 11, 2))      # Mon
    assert newest == "2026-10-30" and median == 11 and share == 0.75
    assert cs._bars_age({"A": fresh}, today=date(2026, 11, 2)) == ("2026-10-30", 1, 0.0)
    assert cs._bars_age({}, today=date(2026, 11, 2)) == (None, None, 1.0)


def test_alert_names_the_bar_date_and_flags_stale_bars():
    import tasks
    result = {"scan_notes": "1 candidate", "newest_bar": "2026-10-30", "median_bar_age": 12,
              "stale_ticker_share": 0.75, "stale_bars": True,
              "candidates": [{"ticker": "AAA", "company": "A Co", "composite_score": 70,
                              "gics_sector": "Tech"}]}
    with patch("agents.candidate_scanner.run_candidate_scan", return_value=result), \
         patch("tasks._heartbeat_ok"), patch("alerts._send") as send:
        tasks.run_candidate_scan.run()
    text = send.call_args[0][0]
    assert "median 12 weekdays old, 75% of tickers stale (newest 2026-10-30)" in text and "STALE" in text


def test_scan_reports_its_funnel_before_the_cap():
    """Audit 11-07 01:08 #4: the cap, not the threshold, decided every run."""
    bars = [{"date": f"2026-10-{d:02d}", "close": 1.0} for d in range(1, 31)] * 3
    cons = {f"T{i}": {"company": "c", "gics_sector": "s", "gics_sub_industry": "x"} for i in range(5)}
    cons["SHORT"] = {"company": "c", "gics_sector": "s", "gics_sub_industry": "x"}
    hist = {**{f"T{i}": bars for i in range(5)}, "SHORT": bars[:10]}
    moves = iter([(30, "m"), (30, "m"), (30, "m"), (5, "m"), (30, "m")])
    with patch("agents.candidate_scanner.seed_thesis_collection"), \
         patch("knowledge_loader.get_all_tickers", return_value=["WL1", "T9"]), \
         patch("ingestion.sp500_universe.get_sp500_constituents", return_value=cons), \
         patch("db.get_price_history_bulk", return_value=hist), \
         patch("agents.candidate_scanner._score_movement", side_effect=lambda t, b: next(moves)), \
         patch("agents.candidate_scanner._score_thesis_match", return_value=(20, "t", "x")), \
         patch("db.log_watchlist_candidate"), patch("db.log_activity") as act:
        out = cs.run_candidate_scan(min_composite_score=40, max_results=2)
    stage, _, _, _, extra = act.call_args.args
    assert stage == "candidate_scan_funnel" and extra["cap"] == 2 and extra["above_threshold"] == 4
    f = out["funnel"]
    assert (f["screened"], f["short_history"], f["below_movement"], f["scored"]) == (6, 1, 1, 4)
    assert (f["above_threshold"], f["returned"]) == (4, 2)
    assert f["watchlist_outside_universe"] == 2
