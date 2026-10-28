"""Held names reporting earnings get a dated earnings review first in the
daily rotation (roadmap 0e: held-name earnings review T-2/T+1)."""
from datetime import date
from unittest.mock import MagicMock, patch

from agents import crew


def test_event_text_states_the_date_before_on_and_after():
    d = date(2026, 11, 2)
    pre = crew._earnings_review_event("PLTR", d, today=date(2026, 10, 30))
    assert pre["type"] == "earnings review"
    assert "2026-11-02 (in 3 calendar days)" in pre["text"]
    assert "hold through the print" in pre["text"]
    assert "today" in crew._earnings_review_event("PLTR", d, today=d)["text"]
    post = crew._earnings_review_event("PLTR", d, today=date(2026, 11, 3))
    assert "1 calendar day ago" in post["text"] and "confirmed or broke" in post["text"]


def test_reporting_names_go_first_and_get_the_earnings_event():
    positions = {t: {} for t in ["AMD", "NVDA", "PLTR"]}
    seen = []
    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=positions), \
         patch("agents.market_regime.get_cached_regime", return_value={}), \
         patch("db.get_state", return_value='{"PLTR": "2026-10-27", "AMD": "2026-10-01", "NVDA": "2026-10-02"}'), \
         patch("db.set_state"), \
         patch("db.get_earnings_around", return_value={"PLTR": date(2026, 11, 2)}), \
         patch("agents.crew.review_held_position",
               side_effect=lambda t, *a, earnings_date=None: seen.append((t, earnings_date)) or {}):
        crew.review_all_held_positions(max_per_run=2, priority_tickers=["NVDA"])
    assert seen == [("PLTR", date(2026, 11, 2)), ("NVDA", None)]


def test_earnings_lookup_failure_falls_back_to_normal_rotation():
    positions = {"AMD": {}}
    with patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=positions), \
         patch("agents.market_regime.get_cached_regime", return_value={}), \
         patch("db.get_state", return_value="{}"), patch("db.set_state"), \
         patch("db.get_earnings_around", side_effect=RuntimeError("db down")), \
         patch("agents.crew.review_held_position", return_value={}) as review:
        crew.review_all_held_positions()
    review.assert_called_once()
    assert review.call_args.kwargs["earnings_date"] is None


def test_review_passes_the_earnings_event_to_the_analysis():
    with patch("agents.crew._analyze_held_position", return_value={}) as analyze, \
         patch("agents.crew._capture_latest_thesis"):
        crew.review_held_position("PLTR", MagicMock(), {}, {}, earnings_date=date(2026, 11, 2))
    assert analyze.call_args.kwargs["event"]["type"] == "earnings review"
