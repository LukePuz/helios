"""Fill times when TWS sends execution.time as a UTC wall clock with no zone:
ib_async labels it with the TWS/local zone, so 20:07Z became 01:07Z (MSFT
BUY 2026-11-03, orders.filled_at 5 h late)."""
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import ibkr
import test_order_row_before_fill_wait as _fw   # its fake orders table (a module import: no re-collection)

ET = ZoneInfo("America/New_York")
_real_tws_zone = ibkr._tws_zone   # the fixture below patches it


@pytest.fixture(autouse=True)
def _tws_in_et():
    ibkr._relabel_warned.clear()
    with patch("ibkr._tws_zone", return_value=ET):
        yield


def _true_time(ago_s=120):
    return (datetime.now(timezone.utc) - timedelta(seconds=ago_s)).replace(microsecond=0)


def _mislabelled(t):
    """What ib_async makes of a suffix-less UTC wall clock: read as ET."""
    return t.replace(tzinfo=None).replace(tzinfo=ET).astimezone(timezone.utc)


def _exec(t, **kw):
    return SimpleNamespace(time=t, execId=kw.pop("exec_id", "e1"), orderId=kw.pop("order_id", 79),
                           shares=kw.pop("shares", 1.0), price=kw.pop("price", 522.55),
                           side=kw.pop("side", "BOT"), permId=kw.pop("perm_id", 0), **kw)


def _req_fill(t, symbol="MSFT", **kw):
    """A reqExecutions Fill: its time is execution.time."""
    return SimpleNamespace(contract=SimpleNamespace(symbol=symbol), execution=_exec(t, **kw), time=t)


def _live_fill(exec_t, receipt_t, symbol="MSFT", **kw):
    """A live Fill: its time is the message receipt time."""
    return SimpleNamespace(contract=SimpleNamespace(symbol=symbol), execution=_exec(exec_t, **kw),
                           time=receipt_t)


class TestExecTime:
    def test_live_fill_uses_the_receipt_time(self):
        t = _true_time()
        receipt = t + timedelta(milliseconds=145)
        assert ibkr._exec_time(_live_fill(_mislabelled(t), receipt)) == receipt

    def test_req_executions_time_5h_ahead_is_read_as_utc(self, caplog):
        t = _true_time()
        rec = _req_fill(_mislabelled(t))
        assert rec.execution.time - t == timedelta(hours=5) or rec.execution.time - t == timedelta(hours=4)
        with caplog.at_level(logging.WARNING, logger="ibkr"):
            assert ibkr._exec_time(rec) == t
            assert ibkr._exec_time(rec) == t
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert rec.execution.time.isoformat() in warnings[0].getMessage()
        assert t.isoformat() in warnings[0].getMessage()

    def test_record_without_its_own_time_is_treated_like_req_executions(self):
        t = _true_time()
        rec = SimpleNamespace(execution=_exec(_mislabelled(t)))
        assert ibkr._exec_time(rec) == t

    def test_correct_past_time_is_unchanged(self, caplog):
        t = _true_time()
        with caplog.at_level(logging.WARNING, logger="ibkr"):
            assert ibkr._exec_time(_req_fill(t)) == t
        assert not caplog.records

    def test_time_within_the_slack_is_unchanged(self):
        t = datetime.now(timezone.utc) + timedelta(seconds=30)
        assert ibkr._exec_time(_req_fill(t)) == t

    def test_future_time_that_relabelling_cannot_fix_is_none(self, caplog):
        t = datetime.now(timezone.utc) + timedelta(days=3)
        with caplog.at_level(logging.WARNING, logger="ibkr"):
            assert ibkr._exec_time(_req_fill(t)) is None
        assert "no fill time recorded" in caplog.records[0].getMessage()

    def test_shifted_time_read_6h_later_is_returned_unchanged(self, caplog):
        """The known limit: once the shifted time is in the past it can't be
        told from a correct one."""
        t = _true_time(6 * 3600 + 120)
        shifted = _mislabelled(t)
        assert shifted < datetime.now(timezone.utc)
        with caplog.at_level(logging.WARNING, logger="ibkr"):
            assert ibkr._exec_time(_req_fill(shifted)) == shifted
        assert not caplog.records

    def test_with_fill_time_logs_receipt_and_fallback_counts_once(self, caplog):
        t = _true_time()
        recs = [_live_fill(_mislabelled(t), t + timedelta(milliseconds=100)), _req_fill(t),
                _req_fill(t - timedelta(seconds=5))]
        with caplog.at_level(logging.INFO, logger="ibkr"):
            ibkr._with_fill_time({}, recs)
        info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert info == ["fill_time from 3 record(s): 1 receipt time, 2 execution-time fallback"]

    def test_naive_time_is_utc(self):
        t = _true_time().replace(tzinfo=None)
        assert ibkr._exec_time(_req_fill(t)) == t.replace(tzinfo=timezone.utc)

    def test_unset_receipt_time_falls_back_to_the_execution_time(self):
        t = _true_time()
        assert ibkr._exec_time(_live_fill(t, datetime.min)) == t

    def test_no_datetime_means_none(self):
        assert ibkr._exec_time(SimpleNamespace(execution=SimpleNamespace(time=None))) is None
        assert ibkr._exec_time(SimpleNamespace(execution=MagicMock())) is None

    def test_with_fill_time_takes_the_latest_corrected_time(self):
        t1, t2 = _true_time(300), _true_time(120)
        out = ibkr._with_fill_time({}, [_req_fill(_mislabelled(t2)), _req_fill(t1)])
        assert out["fill_time"] == t2


class TestIbkrCallers:
    def test_place_market_order_passes_the_live_receipt_time(self):
        t = _true_time(5)
        receipt = t + timedelta(milliseconds=145)
        trade = MagicMock()
        trade.isDone.return_value = True
        trade.order.orderId, trade.order.permId = 79, 1903836920
        trade.orderStatus.status, trade.orderStatus.filled = "Filled", 1.0
        trade.orderStatus.avgFillPrice = 522.55
        trade.fills = [_live_fill(_mislabelled(t), receipt)]
        fake_ib = MagicMock()
        fake_ib.placeOrder.return_value = trade
        with patch("ibkr.get_ib", return_value=fake_ib), patch("ibkr._stock", return_value=MagicMock()), \
             patch("ibkr.ibi.MarketOrder", return_value=MagicMock(orderId=79)):
            assert ibkr.place_market_order("MSFT", "BUY", 1, fill_timeout=1.0)["fill_time"] == receipt

    def test_fill_for_order_corrects_and_still_matches_the_placement_time(self):
        t = _true_time()
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [_req_fill(_mislabelled(t))]
        with patch("ibkr.get_ib", return_value=fake_ib):
            fill = ibkr.get_fill_for_order(79, ticker="MSFT", side="BUY",
                                           placed_at=t - timedelta(seconds=10))
        assert fill["fill_time"] == t

    def test_recent_sell_fill_corrects_the_total_and_each_order(self, caplog):
        t = _true_time()
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [_req_fill(_mislabelled(t), side="SLD", order_id=34)]
        with patch("ibkr.get_ib", return_value=fake_ib), caplog.at_level(logging.WARNING, logger="ibkr"):
            fill = ibkr.get_recent_sell_fill("MSFT")
        assert fill["fill_time"] == t and fill["by_order"][0]["fill_time"] == t
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def _broker_with_live_fill(t, receipt, filled_qty=42.0, price=101.5):
    """A place_market_order stand-in whose fill_time comes from the real
    ibkr._with_fill_time over a live Fill with a mislabelled execution.time."""
    def place(ticker, side, qty, fill_timeout=30.0, on_submitted=None):
        if on_submitted is not None:
            on_submitted(501, None)
        return ibkr._with_fill_time(
            {"order_id": 501, "perm_id": 9001, "ticker": ticker, "side": side, "qty": qty,
             "fill_price": price, "filled_qty": filled_qty, "status": "Filled"},
            [_live_fill(_mislabelled(t), receipt, symbol=ticker)])
    return place


class TestCallerPaths:
    def test_execution_exit_writes_the_receipt_time_to_the_order_and_outcome(self):
        t = _true_time(5)
        receipt = t + timedelta(milliseconds=145)
        fake = _fw.FakeOrders()
        real_mark = fake.mark_submitted_order_fill
        fake.mark_submitted_order_fill = MagicMock(side_effect=real_mark)
        with patch("execution._remaining_after_sell", return_value=0.0):
            _fw._run_exit(fake, _broker_with_live_fill(t, receipt))
        assert fake.mark_submitted_order_fill.call_args.kwargs["filled_at"] == receipt
        assert fake.mocks["close_trade_outcome"].call_args.kwargs["exit_fill_at"] == receipt

    def test_reconcile_writes_the_corrected_req_executions_time(self):
        t = _true_time()
        fake = _fw.FakeOrders()
        fake.log_order("NVDA", "SELL", "MARKET", 42.0, broker_order_id=501, status="unconfirmed")
        fake.rows[1]["created_at"] = t - timedelta(minutes=1)
        real_update = fake.update_order_fill
        fake.update_order_fill = MagicMock(side_effect=real_update)
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [
            _req_fill(_mislabelled(t), symbol="NVDA", side="SLD", order_id=501, shares=42.0, price=99.8)]
        with patch("ibkr.get_ib", return_value=fake_ib):
            fill = ibkr.get_fill_for_order(501, ticker="NVDA", side="SELL")
        assert fill["fill_time"] == t
        _fw._reconcile(fake, fill)
        assert fake.update_order_fill.call_args.kwargs["filled_at"] == t

    def test_partial_sale_writes_the_receipt_time(self, monkeypatch):
        import market_data
        import risk_manager
        monkeypatch.setattr(market_data, "is_market_hours", lambda *a, **k: True)
        t = _true_time(5)
        receipt = t + timedelta(milliseconds=145)
        client = MagicMock()
        client.get_open_position.return_value = MagicMock(qty=70)
        with patch("execution._stops_still_resting", return_value=0.0), \
             patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
             patch("db.set_stop_broker_order_id"), patch("tasks.manage_cancel_stop_orders_for"), \
             patch("ibkr.place_market_order", side_effect=_broker_with_live_fill(t, receipt, 30, 125.5)), \
             patch("ibkr.get_client", return_value=client), \
             patch("tasks.manage_stop_place", return_value={"order_id": 8}), patch("alerts._send"), \
             patch("db.log_trade"), patch("db.log_order", return_value=11), \
             patch("db.update_order_fill") as fill:
            object.__new__(risk_manager.RiskManager)._partial_sell_and_resize_stop(
                "NVDA", 100, 30, 100.0, decision_price=126.0)
        assert fill.call_args.kwargs["filled_at"] == receipt


class TestPreviewScript:
    def test_prints_both_tables_and_never_commits(self):
        from scripts import fix_filled_at_preview as p
        sql = p.build_sql("America/New_York")
        assert "UPDATE orders SET filled_at" in sql and "UPDATE trade_outcomes SET exit_fill_at" in sql
        assert "(filled_at - created_at)" in sql and "(exit_fill_at - exit_date)" in sql
        assert "\nCOMMIT" not in sql

    def test_rejects_a_zone_that_is_not_a_name(self):
        from scripts import fix_filled_at_preview as p
        with pytest.raises(SystemExit):
            p.main(["--tz", "UTC'; DROP TABLE orders; --"])


def test_unset_tws_zone_means_the_host_zone():
    with patch.object(ibkr, "_ib", None):
        assert _real_tws_zone() is None
    with patch.object(ibkr, "_ib", SimpleNamespace(TimezoneTWS="US/Eastern")):
        assert _real_tws_zone() == ZoneInfo("US/Eastern")
