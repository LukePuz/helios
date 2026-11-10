"""Digest "Book" block: theme totals, AI-infra total against its cap, cash
against the regime target (audit 2026-11-07 19:07 #7). Report-only."""
from types import SimpleNamespace
from unittest.mock import patch

import tasks

WL = {"NVDA": {"sector": "AI Infrastructure — Compute / Accelerators"},
      "ANET": {"sector": "AI Infrastructure — Networking"},
      "JPM": {"sector": "Financials — Banks"}}


def _pos(sym, mv):
    return SimpleNamespace(symbol=sym, market_value=mv)


def test_lines():
    positions = [_pos("NVDA", 30), _pos("ANET", 20), _pos("JPM", 10), _pos("XYZ", 5)]
    reg = {"regime": "Neutral", "portfolio_impact_json": {"cash_target_pct": 20}}
    with patch("knowledge_loader.load_watchlist", return_value=WL), \
         patch("db.get_latest_regime", return_value=reg):
        lines = tasks._digest_book_lines(positions, 100.0, 35.0)
    assert lines[0] == "Themes: AI Infrastructure 50%, Financials 10%, Off watchlist 5%"
    assert lines[1] == "AI infrastructure: 50% of a 60% cap"
    assert lines[2] == "Cash: 35% (target 20% in Neutral)"


def test_parts_that_fail_are_left_out():
    with patch("knowledge_loader.load_watchlist", side_effect=RuntimeError("x")), \
         patch("db.get_latest_regime", side_effect=RuntimeError("db")):
        lines = tasks._digest_book_lines([_pos("NVDA", 40)], 100.0, 60.0)
    assert lines == ["AI infrastructure: 40% of a 60% cap"]
    assert tasks._digest_book_lines([_pos("NVDA", 40)], 0.0, 1.0) == []


def test_block_is_rendered():
    import alerts
    with patch("alerts._send") as send:
        alerts.send_daily_digest(portfolio_value=1.0, day_pl=0.0, day_pl_pct=0.0, spy_pl_pct=None,
                                 positions_text=[], catalysts_text=[], pending_count=0,
                                 halted=False, ingestion_issues=[], book_text=["Cash: 35%"])
    assert "<b>Book:</b>\n  Cash: 35%" in send.call_args[0][0]
