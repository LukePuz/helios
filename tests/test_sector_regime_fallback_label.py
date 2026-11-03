"""gics_sector cache-miss rows (watchlist breadth) are logged under their
own regime_type, rows record the breadth source and n_used, and the empty
sample proxy note names the ETF actually used."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from agents import market_regime as mr


def _bars(n, last, step=0.1):
    out = []
    for i in range(n):
        d = last - timedelta(days=n - 1 - i)
        out.append({"date": d.isoformat(), "open": 100 + i * step, "high": 101 + i * step,
                    "low": 99 + i * step, "close": 100 + i * step, "volume": 1_000_000})
    return out


def _assess(call, watchlist_bars=(), bulk=None):
    etf = _bars(220, date.today())
    log_regime = MagicMock()
    with patch("agents.market_regime._fetch_bars", return_value=etf), \
         patch("agents.market_regime.completed_bars", side_effect=lambda b: b), \
         patch("ingestion.macro.get_macro_snapshot", return_value={}), \
         patch("agents.market_regime._fetch_watchlist_bars_for_breadth", return_value=list(watchlist_bars)), \
         patch("db.get_price_history_bulk", return_value=bulk or {}), \
         patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
         patch("agents.markov_regime.get_markov_packet", return_value={"available": False}), \
         patch("db.log_regime", log_regime):
        packet = call()
    return packet, log_regime.call_args.kwargs


def test_cache_miss_gics_row_is_labelled_fallback_but_cached_as_gics_sector():
    mr._sector_regime_cache.pop("XLV:gics_sector", None)
    packet, kw = _assess(lambda: mr.get_cached_sector_regime("XLV", regime_type="gics_sector"),
                         watchlist_bars=[_bars(210, date.today())])
    assert kw["regime_type"] == "gics_sector_fallback" and kw["sector_etf"] == "XLV"
    assert len("gics_sector_fallback") <= 20                 # VARCHAR(20) column
    assert packet["signals"]["breadth_sample"] == {"source": "watchlist", "n_used": 1}
    assert mr._sector_regime_cache["XLV:gics_sector"] is packet
    mr._sector_regime_cache.pop("XLV:gics_sector", None)


def test_sector_type_cache_miss_keeps_its_label():
    mr._sector_regime_cache.pop("XLK:sector", None)
    _, kw = _assess(lambda: mr.get_cached_sector_regime("XLK", regime_type="sector"))
    assert kw["regime_type"] == "sector"
    mr._sector_regime_cache.pop("XLK:sector", None)


def test_sweep_row_keeps_gics_sector_and_records_n_used():
    today = date.today()
    bulk = {"A": _bars(210, today), "B": _bars(150, today)}
    packet, kw = _assess(lambda: mr.assess_sector_regime("XLF", breadth_tickers=["A", "B"],
                                                         regime_type="gics_sector"), bulk=bulk)
    assert kw["regime_type"] == "gics_sector"
    assert packet["signals"]["breadth_sample"] == {"source": "explicit_sample", "n_used": 1}


def test_empty_sample_proxy_note_names_the_sector_etf():
    packet, _ = _assess(lambda: mr.assess_sector_regime("XLE", breadth_tickers=["A"],
                                                        regime_type="gics_sector"), bulk={})
    note = packet["signals"]["breadth"]["notes"]
    assert note.startswith("XLE proxy breadth") and "SPY" not in note
    assert packet["signals"]["breadth_sample"]["n_used"] == 0


def test_overall_breadth_proxy_note_still_says_spy():
    _, note = mr._score_breadth(_bars(220, date.today()), [])
    assert note.startswith("SPY proxy breadth")
