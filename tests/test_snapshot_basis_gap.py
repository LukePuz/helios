"""Portfolio snapshots record the stop record's entry beside the broker's
average cost, and their gap (audit 11-06 05:33 #4). Record only."""
from unittest.mock import MagicMock, patch

import tasks


def _snapshot(stops):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1000.0, cash=500.0)
    client.get_all_positions.return_value = [
        MagicMock(symbol="NVDA", qty=1, market_value=100.0, avg_entry_price=112.5, unrealized_pl=-12.5),
        MagicMock(symbol="MSFT", qty=1, market_value=100.0, avg_entry_price=100.0, unrealized_pl=0.0)]
    stops_patch = (patch("db.get_all_position_stops", side_effect=stops) if isinstance(stops, Exception)
                   else patch("db.get_all_position_stops", return_value=stops))
    with patch("ibkr.get_client", return_value=client), stops_patch, patch("db.get_entry_price_sources", return_value={}), \
         patch("db.log_snapshot") as snap, patch("yfinance.Ticker", side_effect=RuntimeError("offline")):
        tasks.take_portfolio_snapshot.run()
    return snap.call_args.kwargs["positions"]


def test_gap_is_recorded_against_the_stop_record():
    pos = _snapshot([{"ticker": "NVDA", "entry_price": 100.0}])
    assert pos["NVDA"]["record_entry_price"] == 100.0 and pos["NVDA"]["basis_gap_frac"] == 0.125
    assert pos["MSFT"]["record_entry_price"] is None and pos["MSFT"]["basis_gap_frac"] is None
    assert pos["NVDA"]["avg_entry_price"] == 112.5       # the broker figure is unchanged


def test_a_failed_record_read_still_writes_the_snapshot():
    pos = _snapshot(RuntimeError("pool exhausted"))
    assert pos["NVDA"]["record_entry_price"] is None and pos["NVDA"]["qty"] == 1.0
