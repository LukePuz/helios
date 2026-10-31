"""A gics_sector regime row records how old its breadth sample's bars are
(SEF 10-31: last_bar_date is the sector ETF's, so a stale S&P 500 cache
didn't show)."""
from datetime import date, timedelta
from unittest.mock import patch

from agents import market_regime as mr


def _bars(n, last):
    out = []
    for i in range(n):
        d = last - timedelta(days=n - 1 - i)
        out.append({"date": d.isoformat(), "open": 100 + i * 0.1, "high": 101 + i * 0.1,
                    "low": 99 + i * 0.1, "close": 100 + i * 0.1, "volume": 1_000_000})
    return out


def test_sector_packet_carries_breadth_freshness():
    today = date.today()
    old = today - timedelta(days=30)
    etf = _bars(220, today)
    bulk = {"A": _bars(210, today), "B": _bars(210, old), "C": _bars(210, old)}
    with patch("agents.market_regime._fetch_bars", return_value=etf), \
         patch("agents.market_regime.completed_bars", side_effect=lambda b: b), \
         patch("ingestion.macro.get_macro_snapshot", return_value={}), \
         patch("db.get_price_history_bulk", return_value=bulk), \
         patch("db.log_regime"):
        packet = mr.assess_sector_regime("XLF", breadth_tickers=["A", "B", "C", "D"],
                                         regime_type="gics_sector")
    f = packet["signals"]["breadth_freshness"]
    assert f["sample"] == 4 and f["with_bars"] == 3 and f["newest_bar"] == today.isoformat()
    assert f["stale_share"] == round(2 / 3, 3)
