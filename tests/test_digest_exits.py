"""The daily digest lists today's exits (audit 2026-11-07 03:08 #3): a broker
stop fill is booked by reconcile with no Telegram message."""
from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import alerts
import tasks

_ET = ZoneInfo("America/New_York")


def _trade(sym, side, ts, price=100.0, qty=5, reason="Reconciled untracked exit"):
    return {"symbol": sym, "side": side, "quantity": qty, "price": price,
            "timestamp": ts, "reason": reason, "price_source": "fill"}


def test_lists_only_todays_sells_oldest_first():
    today = date(2026, 11, 5)
    rows = [  # get_recent_trades is newest first
        _trade("PLTR", "SELL", datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc), 180.5),
        _trade("AMD", "BUY", datetime(2026, 11, 5, 14, 50, tzinfo=timezone.utc)),
        _trade("NVDA", "SELL", datetime(2026, 11, 5, 14, 45, tzinfo=timezone.utc), 190.0, 3),
        # 23:30 ET on 11-04 is 04:30 UTC on 11-05: yesterday in ET.
        _trade("MSFT", "SELL", datetime(2026, 11, 5, 4, 30, tzinfo=timezone.utc)),
        _trade("ANET", "SELL", datetime(2026, 11, 4, 15, 0)),  # naive = UTC
    ]
    with patch("db.get_recent_trades", return_value=rows):
        lines = tasks._digest_exits_today(today, _ET)
    assert lines == ["NVDA: 3sh @ $190.00 (Reconciled untracked exit)",
                     "PLTR: 5sh @ $180.50 (Reconciled untracked exit)"]


def test_unknown_price_blank_reason_and_html_are_handled():
    ts = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)
    rows = [_trade("X", "sell", ts, price=None, reason=""), _trade("Y", "SELL", ts, reason="a<b & c")]
    with patch("db.get_recent_trades", return_value=rows):
        lines = tasks._digest_exits_today(date(2026, 11, 5), _ET)
    assert lines == ["Y: 5sh @ $100.00 (a&lt;b &amp; c)", "X: 5sh"]


def test_read_failure_gives_no_lines():
    with patch("db.get_recent_trades", side_effect=RuntimeError("down")):
        assert tasks._digest_exits_today(date(2026, 11, 5), _ET) == []


def _digest(**kw):
    args = dict(portfolio_value=1.0, day_pl=0.0, day_pl_pct=0.0, spy_pl_pct=None,
                positions_text=[], catalysts_text=[], pending_count=0, halted=False,
                ingestion_issues=[])
    args.update(kw)
    with patch("alerts._send") as send:
        alerts.send_daily_digest(**args)
    return send.call_args.args[0]


def test_digest_shows_the_exits_section_only_when_there_are_exits():
    text = _digest(exits_text=["NVDA: 3sh @ $190.00 (stop)"])
    assert "<b>Exits booked today:</b>" in text and "  NVDA: 3sh @ $190.00 (stop)" in text
    assert "Exits booked today" not in _digest()
    many = _digest(exits_text=[f"T{i}: 1sh" for i in range(12)])
    assert "  +2 more" in many and "T10" not in many
