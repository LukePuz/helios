"""PRICE HISTORY brief section, default off (Luke 11-08)."""
from datetime import date, timedelta
from unittest.mock import patch

import config
import context_engine


def _bars(n=60, start=100.0, step=1.0):
    d0 = date(2026, 8, 3)
    out = []
    for i in range(n):
        c = start + step * i
        out.append({"date": (d0 + timedelta(days=i)).isoformat(), "open": c, "high": c + 1,
                    "low": c - 1, "close": c, "volume": 1000 + (5000 if i == n - 1 else 0)})
    return out


def test_off_returns_nothing_and_reads_nothing():
    with patch("db.get_price_history") as gh:
        assert context_engine._price_history_section("NVDA", 150.0) == ""
    assert not gh.called


def test_on_shows_the_path(monkeypatch):
    monkeypatch.setattr(config, "BRIEF_PRICE_HISTORY", True)
    bars = _bars()
    with patch("db.get_price_history", return_value=bars):
        s = context_engine._price_history_section("NVDA", 140.0)
    assert s.startswith("=== PRICE HISTORY — NVDA")
    assert s.count("$") > 20
    assert "Last sessions:" in s and s.split("Last sessions: ")[1].split("\n")[0].count(";") == 9
    assert "from the 60-session high" in s
    assert "Weekly closes, last" in s
    assert "Now: $140.00 (-11.9% vs the last close $159.00" in s
    assert "From the 20-day high ($160.00" in s
    assert "Volume, last session: 6.0x its 20-day average" in s


def test_on_with_no_history(monkeypatch):
    monkeypatch.setattr(config, "BRIEF_PRICE_HISTORY", True)
    with patch("db.get_price_history", return_value=[]):
        assert context_engine._price_history_section("X", 1.0) == ""
    with patch("db.get_price_history", side_effect=RuntimeError("db")):
        assert context_engine._price_history_section("X", 1.0) == ""
