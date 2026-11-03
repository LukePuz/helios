"""tasks._log_stop_slippage: a log-only activity_feed row (stage
stop_slippage) comparing a protective stop's stop price with its fill, and
the stale-stop reconcile path that calls it."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import tasks
import test_tasks_dedup as _dedup   # its stale-stop harness (a module import: no re-collection)


def _call(fill, stop_price=100.0, done=frozenset(), order_id=11, prev_close=None):
    with patch("db.get_activity_shadow_keys", return_value=set(done) if done is not None else None), \
         patch("db.log_activity", return_value=True) as act:
        ok = tasks._log_stop_slippage("DDOG", stop_price, fill, "reconcile_stale_stop",
                                      order_id, prev_close)
    return ok, act


def _extra(act):
    act.assert_called_once()
    args = act.call_args.args
    assert args[0] == "stop_slippage" and args[1] == "DDOG"
    return args[4]


class TestLogStopSlippage:
    def test_gap_down_fill_at_the_open(self):
        # 14:31 UTC on 2026-11-02 (EST) = 09:31 ET: a fill in the first 5 minutes.
        t = datetime(2026, 11, 2, 14, 31, tzinfo=timezone.utc)
        ok, act = _call({"filled_qty": 84.0, "fill_price": 92.0, "fill_time": t}, prev_close=104.0)
        assert ok
        x = _extra(act)
        assert x["stop_price"] == 100.0 and x["fill_price"] == 92.0
        assert x["slippage_pct"] == pytest.approx(-8.0)
        assert x["shares"] == 84.0
        assert x["fill_time"] == t.isoformat()
        assert x["at_open"] is True and x["outside_rth"] is False
        assert x["prev_close"] == 104.0
        assert x["source"] == "reconcile_stale_stop"
        assert x["stop_order_id"] == 11
        assert x["shadow_key"].endswith(":DDOG:11")
        assert x["code_sha"] == "test" and "env_sha" in x

    def test_normal_midday_fill(self):
        t = datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc)   # 12:00 ET
        ok, act = _call({"filled_qty": 10.0, "fill_price": 99.9, "fill_time": t})
        x = _extra(act)
        assert x["slippage_pct"] == pytest.approx(-0.1)
        assert x["at_open"] is False and x["outside_rth"] is False
        assert x["prev_close"] is None

    def test_premarket_fill_is_outside_rth(self):
        t = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)   # 07:00 ET
        _, act = _call({"filled_qty": 10.0, "fill_price": 95.0, "fill_time": t})
        x = _extra(act)
        assert x["outside_rth"] is True and x["at_open"] is False

    def test_no_fill_time_gives_null_session_flags_and_a_key(self):
        _, act = _call({"filled_qty": 10.0, "fill_price": 95.0}, order_id=None)
        x = _extra(act)
        assert x["fill_time"] is None and x["outside_rth"] is None and x["at_open"] is None
        assert x["shadow_key"].count(":") == 2

    @pytest.mark.parametrize("stop, fill", [
        (None, {"filled_qty": 1.0, "fill_price": 95.0}),
        (100.0, {"filled_qty": 1.0, "fill_price": None}),
        (100.0, {"filled_qty": 1.0}),
        (0.0, {"filled_qty": 1.0, "fill_price": 95.0}),
        (100.0, {"filled_qty": 1.0, "fill_price": float("nan")}),
        (100.0, None),
    ])
    def test_missing_prices_write_no_row(self, stop, fill):
        ok, act = _call(fill, stop_price=stop)
        assert ok is False
        act.assert_not_called()

    def test_dedup_on_shadow_key(self):
        from config import TRADING_ENV
        ok, act = _call({"filled_qty": 1.0, "fill_price": 95.0}, done={f"{TRADING_ENV}:DDOG:11"})
        assert ok is False
        act.assert_not_called()

    def test_failed_dedup_read_still_writes(self):
        ok, act = _call({"filled_qty": 1.0, "fill_price": 95.0}, done=None)
        assert ok
        act.assert_called_once()

    def test_errors_are_swallowed(self):
        with patch("db.get_activity_shadow_keys", side_effect=RuntimeError("db down")), \
             patch("db.log_activity", side_effect=RuntimeError("db down")):
            assert tasks._log_stop_slippage("DDOG", 100.0, {"fill_price": 95.0}, "x", 11) is False
        with patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.log_activity", side_effect=RuntimeError("db down")):
            assert tasks._log_stop_slippage("DDOG", 100.0, {"fill_price": 95.0}, "x", 11) is False


class TestReconcilePathUnchanged:
    """The stale-stop reconcile path does exactly the same with and without
    the slippage row (helper writing, helper raising)."""

    @pytest.fixture(autouse=True)
    def _no_pending_sells(self):
        with patch("db.awaiting_fill_sell_tickers", return_value=set()), \
                patch("db.trade_pending_sell_tickers", return_value=set()):
            yield

    FILL = {"filled_qty": 84.0, "fill_price": 220.00,
            "fill_time": datetime(2026, 11, 2, 14, 31, tzinfo=timezone.utc)}

    def _two_cycles(self, helper_patch):
        runner = _dedup.TestStaleStopReconciliation()
        state = {}
        with helper_patch as helper:
            runner._run(state, fill_return=None)    # arms the marker
            out = runner._run(state, fill_return=dict(self.FILL))
        return out, helper, state

    def _calls(self, out):
        mock_fill, log_trade, close_outcome, delete, alert = out
        return (log_trade.call_args_list, close_outcome.call_args_list,
                delete.call_args_list, alert.call_args_list)

    def test_outcome_identical_with_writing_and_raising_helper(self):
        with patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.log_activity", return_value=True) as act:
            out_ok, _, st_ok = self._two_cycles(patch("tasks._log_stop_slippage",
                                                      wraps=tasks._log_stop_slippage))
        rows = [c for c in act.call_args_list if c.args[0] == "stop_slippage"]
        assert len(rows) == 1
        x = rows[0].args[4]
        assert x["stop_price"] == 228.74 and x["fill_price"] == 220.00
        assert x["stop_order_id"] == 11 and x["at_open"] is True

        out_err, helper, st_err = self._two_cycles(
            patch("tasks._log_stop_slippage", side_effect=RuntimeError("boom")))
        helper.assert_called_once()
        out_none, _, st_none = self._two_cycles(patch("tasks._log_stop_slippage", return_value=False))

        assert self._calls(out_ok) == self._calls(out_err) == self._calls(out_none)
        assert st_ok == st_err == st_none
        log_trade = out_ok[1]
        log_trade.assert_called_once()
        assert log_trade.call_args.kwargs["price"] == 220.00
        out_ok[3].assert_called_once_with("DDOG")

    def test_helper_not_called_when_the_trade_row_was_not_written(self):
        runner = _dedup.TestStaleStopReconciliation()
        state = {}
        with patch("tasks._log_stop_slippage") as helper:
            runner._run(state, fill_return=None)
            _, log_trade, close_outcome, delete, _ = runner._run(
                state, fill_return=dict(self.FILL), log_trade_ok=False)
        log_trade.assert_called_once()
        helper.assert_not_called()
        close_outcome.assert_called_once()           # the rest of the path is unchanged
        delete.assert_called_once_with("DDOG")


class TestFillMatch:
    def test_fill_match_is_always_present(self):
        _, act = _call({"filled_qty": 1.0, "fill_price": 95.0})
        x = _extra(act)
        assert x["fill_match"] == "symbol_vwap_unrecorded_sells"
        assert x["fill_order_ids"] is None and x["fill_is_stop_order"] is None

    @pytest.mark.parametrize("ids, stop_id, expect", [
        ([11], 11, True),
        ([11, 847], 11, False),        # a manual TWS sale blended in
        ([847], 11, False),
        ([11], None, None),            # stop order id unknown
    ])
    def test_fill_is_stop_order(self, ids, stop_id, expect):
        _, act = _call({"filled_qty": 1.0, "fill_price": 95.0, "order_ids": ids}, order_id=stop_id)
        x = _extra(act)
        assert x["fill_order_ids"] == sorted(ids)
        assert x["fill_is_stop_order"] is expect

    def test_get_recent_sell_fill_returns_the_matched_order_ids(self):
        from unittest.mock import MagicMock
        import ibkr
        from test_ibkr import _FakeFillRecord
        fake_ib = MagicMock()
        fake_ib.reqExecutions.return_value = [
            _FakeFillRecord("DDOG", order_id=847, shares=34.0, avg_price=226.50, side="SLD"),
            _FakeFillRecord("DDOG", order_id=11, shares=50.0, avg_price=228.00, side="SLD"),
            _FakeFillRecord("DDOG", order_id=11, shares=10.0, avg_price=228.00, side="SLD"),
        ]
        with patch("ibkr.get_ib", return_value=fake_ib):
            r = ibkr.get_recent_sell_fill("DDOG")
        assert r["order_ids"] == [11, 847]
        assert r["filled_qty"] == 94.0
