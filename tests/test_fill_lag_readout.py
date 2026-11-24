"""Fill-lag readout: one exit per fill, lags, decisions inside the window, no prices."""
import importlib.util
import os
from datetime import date, datetime, timedelta, timezone

_spec = importlib.util.spec_from_file_location(
    "fill_lag_readout", os.path.join(os.path.dirname(__file__), "..", "scripts", "fill_lag_readout.py"))
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)

FILL = datetime(2026, 11, 20, 16, 32, tzinfo=timezone.utc)
BOOK = FILL + timedelta(minutes=29, seconds=6)


def _ex():
    outcomes = [("MSFT", FILL, BOOK), ("MSFT", FILL, BOOK),          # two lots, one fill
                ("ANET", None, datetime(2026, 10, 22, 20, 0, tzinfo=timezone.utc))]
    trades = [("MSFT", BOOK), ("MSFT", FILL - timedelta(days=3))]
    seen = [("MSFT", FILL - timedelta(hours=3), (FILL - timedelta(hours=3, minutes=2)).isoformat()),  # other order
            ("MSFT", FILL + timedelta(minutes=2), FILL.isoformat()),
            ("MSFT", FILL + timedelta(minutes=9), None)]                               # no exec time
    signals = [("MSFT", FILL + timedelta(minutes=5), "HOLD"), ("MSFT", FILL + timedelta(minutes=6), "SCREENED_OUT"),
               ("MSFT", BOOK + timedelta(minutes=1), "HOLD"), ("NVDA", FILL + timedelta(minutes=5), "OPEN")]
    approvals = [("MSFT", FILL + timedelta(minutes=10), None), ("MSFT", FILL - timedelta(hours=1), BOOK + timedelta(hours=1))]
    return fl.attach(fl.exits(outcomes), trades, seen, signals, approvals)


def test_lags_and_window_counts():
    ex = _ex()
    assert len(ex) == 2
    m = next(e for e in ex if e["ticker"] == "MSFT")
    assert abs(m["lag_booked_min"] - 29.1) < 1e-9 and abs(m["lag_seen_min"] - 2.0) < 1e-9
    assert m["signals_in_window"] == {"HOLD": 1, "SCREENED_OUT": 1}
    assert m["approvals_in_window"] == 1
    a = next(e for e in ex if e["ticker"] == "ANET")
    assert a["fill"] is None and a["lag_booked_min"] is None


def test_readout_is_times_and_counts_only():
    text = fl.readout(_ex(), date(2026, 8, 16), "live", "abc")
    assert "| MSFT | 2026-11-20 11:32 ET | 2026-11-20 11:34 ET | 2026-11-20 12:01 ET | 2.0 | 29.1 | HOLD 1 (+1 screened out) | 1 |" in text
    assert "fill -> booked: n 1, median 29.1 min" in text
    assert "1 exit(s) have no broker fill time" in text


def test_seen_row_outside_the_tolerance_is_not_matched():
    outcomes = [("MSFT", FILL, BOOK)]
    seen = [("MSFT", FILL + timedelta(minutes=50), (FILL + timedelta(minutes=45)).isoformat())]
    e = fl.attach(fl.exits(outcomes), [("MSFT", BOOK)], seen, [], [])[0]
    assert e["seen"] is None and e["lag_seen_min"] is None
