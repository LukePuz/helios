"""POSITION CONTEXT block for held tickers (stop, thesis, recent verdicts)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import context_engine


def test_held_ticker_shows_stop_thesis_and_verdicts():
    stop = {"entry_price": 100.0, "stop_price": 90.0, "stop_method": "2.5x ATR",
            "opened_at": datetime.now(timezone.utc) - timedelta(days=10),
            "thesis_bucket": "Compounder", "conviction": 7, "trailing_active": False}
    ctx = {"original_thesis": {"thesis_summary": "AI leader"}, "latest_thesis": {"thesis_summary": "still AI leader"},
           "thesis_updated_at": "2026-10-27", "profit_target_taken": False,
           "recent_verdicts": [{"timestamp": "2026-10-28 09:00", "signal": "HOLD", "conviction": 6}]}
    with patch("db.get_position_stop", return_value=stop), patch("db.get_position_context", return_value=ctx):
        text = context_engine._position_context_section("XYZ", 100.0)
    assert "protective stop $90.00 (-10.0% from current price)" in text
    assert "Held 10 days" in text and "Thesis at entry: AI leader" in text
    assert "Recent verdicts: 2026-10-28 HOLD c6" in text


def test_not_held_is_empty():
    with patch("db.get_position_stop", return_value=None):
        assert context_engine._position_context_section("XYZ", 100.0) == ""
