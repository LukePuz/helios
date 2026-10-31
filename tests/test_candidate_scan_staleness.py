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
