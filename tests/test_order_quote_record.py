"""Each market order records the quote it was sent on: data type and age
(audit 2026-11-07 01:08 #6)."""
import datetime as dt
import math
from types import SimpleNamespace
from unittest.mock import patch

import ibkr


def test_snapshot_reads_type_prices_and_age():
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=900)
    q = ibkr._quote_snapshot(SimpleNamespace(marketDataType=3, bid=10.0, ask=10.2,
                                             last=math.nan, time=t))
    assert q["md_type"] == 3 and q["md_type_label"] == "delayed"
    assert q["bid"] == 10.0 and q["ask"] == 10.2 and q["last"] is None
    assert 899 <= q["quote_age_s"] <= 905 and q["quote_time"] == t.isoformat()


def test_snapshot_with_no_ticks_is_all_none():
    q = ibkr._quote_snapshot(SimpleNamespace(marketDataType=None, bid=-1, ask=0, last=None, time=None))
    assert all(v is None for v in q.values())


def test_snapshot_never_raises():
    class Bad:
        @property
        def marketDataType(self):
            raise RuntimeError("x")
    assert ibkr._quote_snapshot(Bad())["md_type"] is None


def test_record_writes_an_order_quote_activity_row():
    result = {"order_id": 42, "quote": {"md_type": 3, "md_type_label": "delayed", "quote_age_s": 1.5}}
    with patch("db.log_activity_direct") as act:
        ibkr._record_order_quote("NVDA", "BUY", result).join(5)
    stage, ticker, msg = act.call_args.args[:3]
    extra = act.call_args.args[4]
    assert stage == "order_quote" and ticker == "NVDA" and "delayed" in msg
    assert extra["order_id"] == 42 and extra["md_type"] == 3 and extra["side"] == "BUY"


def test_record_never_raises():
    with patch("db.log_activity_direct", side_effect=RuntimeError("down")):
        ibkr._record_order_quote("NVDA", "BUY", {"order_id": 1}).join(5)
    with patch("threading.Thread", side_effect=RuntimeError("no threads")):
        assert ibkr._record_order_quote("NVDA", "BUY", {"order_id": 1}) is None


def test_record_does_not_wait_for_the_db_write():
    import threading
    import time
    gate = threading.Event()
    with patch("db.log_activity_direct", side_effect=lambda *a, **k: gate.wait(5)):
        start = time.monotonic()
        t = ibkr._record_order_quote("NVDA", "BUY", {"order_id": 1})
        assert time.monotonic() - start < 1.0 and t.daemon
        gate.set()
        t.join(5)


def test_market_order_result_carries_the_quote_and_records_it():
    from unittest.mock import MagicMock
    fake_ib = MagicMock()
    fake_ib.reqMktData.return_value = SimpleNamespace(
        marketDataType=3, bid=100.0, ask=100.5, last=100.2,
        time=dt.datetime.now(dt.timezone.utc))
    trade = SimpleNamespace(order=SimpleNamespace(orderId=999, tif=None, permId=0),
                            orderStatus=SimpleNamespace(status="Filled", filled=5.0, avgFillPrice=100.4),
                            isDone=lambda: True, fills=[])
    fake_ib.placeOrder.return_value = trade
    with patch("ibkr.get_ib", return_value=fake_ib), patch("ibkr._stock", return_value=MagicMock()), \
         patch("ibkr.ibi.MarketOrder", return_value=SimpleNamespace(orderId=999, tif=None)), \
         patch("ibkr._record_order_quote") as rec:
        result = ibkr.place_market_order("NVDA", "BUY", 5, fill_timeout=1.0)
    assert result["quote"]["md_type_label"] == "delayed" and result["quote"]["ask"] == 100.5
    assert rec.call_args.args[:2] == ("NVDA", "BUY") and rec.call_args.args[2] is result
