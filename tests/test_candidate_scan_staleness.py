"""The weekly candidate scan says how old its price bars are (audit A1031-0453-5)."""
from datetime import date
from unittest.mock import patch

from agents import candidate_scanner as cs


def test_bars_age_counts_weekdays_since_the_newest_bar():
    bars = {"AAA": [{"date": "2026-10-15"}, {"date": "2026-10-16"}], "BBB": [{"date": "2026-10-14"}]}
    assert cs._bars_age(bars, today=date(2026, 10, 19)) == ("2026-10-16", 1)    # Fri -> Mon
    assert cs._bars_age(bars, today=date(2026, 10, 26)) == ("2026-10-16", 6)
    assert cs._bars_age({}, today=date(2026, 10, 26)) == (None, None)


def test_alert_names_the_bar_date_and_flags_stale_bars():
    import tasks
    result = {"scan_notes": "1 candidate", "newest_bar": "2026-10-16", "stale_bars": True,
              "candidates": [{"ticker": "AAA", "company": "A Co", "composite_score": 70,
                              "gics_sector": "Tech"}]}
    with patch("agents.candidate_scanner.run_candidate_scan", return_value=result), \
         patch("tasks._heartbeat_ok"), patch("alerts._send") as send:
        tasks.run_candidate_scan.run()
    text = send.call_args[0][0]
    assert "Price bars as of 2026-10-16" in text and "STALE" in text
