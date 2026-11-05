"""The regime's "200DMA" averages every bar it has below 200 (about 170 from
the 220+30 calendar-day fetch). Log-only: the count averaged is recorded in
signals.trend.ma200_bars_used; the computation is unchanged."""
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from agents import market_regime as mr


def _bars(n):
    last = date.today()
    return [{"date": (last - timedelta(days=n - 1 - i)).isoformat(), "open": 100 + i * 0.1,
             "high": 101 + i * 0.1, "low": 99 + i * 0.1, "close": 100 + i * 0.1,
             "volume": 1_000_000} for i in range(n)]


@pytest.mark.parametrize("n, expected", [(171, 171), (250, 200), (49, None)])
def test_ma200_bars_used(n, expected):
    assert mr._ma200_bars_used(_bars(n)) == expected


@pytest.mark.parametrize("n, expected", [(171, 171), (230, 200)])
def test_assess_regime_records_count(n, expected):
    bars = _bars(n)
    with patch("agents.market_regime._fetch_spy_bars", return_value=bars), \
         patch("agents.market_regime.completed_bars", side_effect=lambda b: b), \
         patch("ingestion.macro.get_macro_snapshot", return_value={}), \
         patch("agents.market_regime._fetch_watchlist_bars_for_breadth", return_value=[]), \
         patch("ingestion.options_flow.get_etf_flows", return_value={}), \
         patch("agents.markov_regime.get_markov_packet", return_value={"available": False, "notes": "x"}), \
         patch("db.log_regime") as log_regime:
        packet = mr.assess_regime()
    assert packet["signals"]["trend"]["ma200_bars_used"] == expected
    assert log_regime.call_args.args[3]["trend"]["ma200_bars_used"] == expected
    trend_raw, _ = mr._score_trend(bars)
    assert packet["signals"]["trend"]["score"] == trend_raw        # scoring unchanged


def test_sector_regime_records_count():
    with patch("agents.market_regime._fetch_bars", return_value=_bars(171)), \
         patch("agents.market_regime.completed_bars", side_effect=lambda b: b), \
         patch("ingestion.macro.get_macro_snapshot", return_value={}), \
         patch("db.get_price_history_bulk", return_value={}), \
         patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
         patch("agents.markov_regime.get_markov_packet", return_value={"available": False, "notes": "x"}), \
         patch("db.log_regime") as log_regime:
        packet = mr.assess_sector_regime("XLK", breadth_tickers=["A"], regime_type="sector")
    assert packet["signals"]["trend"]["ma200_bars_used"] == 171
    assert log_regime.call_args.args[3]["trend"]["ma200_bars_used"] == 171
