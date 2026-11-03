"""tasks._log_stop_slippage (professor M229): the log-only stop_slippage
row uses the filled order's own fill and last-seen broker trigger where it
can be identified, records trailing_active and stop_method, says how the
fill was matched, and fills prev_close or says why it is null.
tasks._cache_stop_triggers keeps the last-seen triggers per order."""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import tasks
import test_tasks_dedup as _dedup   # its stale-stop harness (a module import: no re-collection)


def _extra(act):
    act.assert_called_once()
    args = act.call_args.args
    assert args[0] == "stop_slippage" and args[1] == "DDOG"
    return args[4]


def _cache(*entries):
    return json.dumps({f"{e['order_id']}|{e.get('perm_id') or ''}": e for e in entries})


def _entry(order_id, trigger, perm_id=None, order_type="TRAIL", ticker="DDOG",
           seen_at="2026-11-02T14:15:00+00:00"):
    return {"ticker": ticker, "order_id": order_id, "perm_id": perm_id, "order_type": order_type,
            "trigger": trigger, "seen_at": seen_at}


def _call(fill, cache=None, stop_price=100.0, order_id=11, bars=(), **kw):
    with patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_state", return_value=cache or ""), \
         patch("db.get_price_history", return_value=list(bars)) as hist, \
         patch("db.log_activity", return_value=True) as act:
        ok = tasks._log_stop_slippage("DDOG", stop_price, fill, "reconcile_stale_stop",
                                      order_id, **kw)
    return ok, act, hist


T_FILL = datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc)   # 12:00 ET, Monday


def _key():
    from config import TRADING_ENV
    return tasks._STOP_TRIGGER_CACHE_KEY.format(env=TRADING_ENV)


_KEY = _key()
STALE_FILL = {"filled_qty": 84.0, "fill_price": 220.00,
              "fill_time": datetime(2026, 11, 2, 14, 31, tzinfo=timezone.utc)}


OLD_FIELDS = ("stop_price", "fill_price", "slippage_pct", "shares")


class TestFilledOrder:
    def test_trail_fill_trigger_goes_in_new_fields_only(self):
        # Record 100 (sampled RTH prices), the broker TRAIL last seen at 104;
        # a manual sale at 94 is blended into the symbol VWAP.
        fill = {"filled_qty": 60.0, "fill_price": 101.5, "fill_time": T_FILL, "order_ids": [11, 847],
                "by_order": [{"order_id": 11, "perm_id": 900, "qty": 50.0, "price": 103.0},
                             {"order_id": 847, "perm_id": 901, "qty": 10.0, "price": 94.0}]}
        ok, act, _ = _call(fill, _cache(_entry(11, 104.0, perm_id=900)),
                           trailing_active=True, stop_method="atr")
        assert ok
        x = _extra(act)
        # The original fields keep their meaning: record stop, symbol VWAP.
        assert x["stop_price"] == 100.0 and x["fill_price"] == 101.5 and x["shares"] == 60.0
        assert x["slippage_pct"] == pytest.approx(1.5)
        assert x["stop_price_source"] == "db_record"
        assert x["broker_trigger_price"] == 104.0 and x["broker_order_type"] == "TRAIL"
        assert x["broker_trigger_seen_at"] == "2026-11-02T14:15:00+00:00"
        assert x["matched_fill_price"] == 103.0 and x["matched_shares"] == 50.0
        assert x["slippage_vs_broker_trigger_pct"] == pytest.approx((103.0 - 104.0) / 104.0 * 100,
                                                                     abs=1e-4)
        assert x["fill_match"] == "stop_order_perm_id"
        assert x["matched_order_id"] == 11 and x["matched_perm_id"] == 900
        assert x["trailing_active"] is True and x["stop_method"] == "atr"
        assert x["fill_is_stop_order"] is False          # unchanged meaning
        for gone in ("record_stop_price", "slippage_vs_record_pct", "symbol_vwap_fill_price",
                     "symbol_shares"):
            assert gone not in x

    @pytest.mark.parametrize("cache", [None, _cache(_entry(11, 104.0, perm_id=900))])
    def test_original_fields_match_a_row_with_no_order_data(self, cache):
        """Rows written now read like old rows on every original field."""
        base = {"filled_qty": 60.0, "fill_price": 101.5, "fill_time": T_FILL, "order_ids": [11]}
        _, plain, _ = _call(dict(base))
        _, rich, _ = _call(dict(base, by_order=[{"order_id": 11, "perm_id": 900, "qty": 60.0,
                                                 "price": 101.5}]), cache)
        p, r = _extra(plain), _extra(rich)
        for k in OLD_FIELDS + ("fill_time", "outside_rth", "at_open", "source", "fill_order_ids",
                               "fill_is_stop_order", "stop_order_id", "shadow_key"):
            assert p[k] == r[k], k
        assert plain.call_args.args[2] == rich.call_args.args[2]   # the message too

    def test_record_order_filled_but_never_seen_resting(self):
        fill = {"filled_qty": 60.0, "fill_price": 101.5, "fill_time": T_FILL,
                "by_order": [{"order_id": 11, "perm_id": None, "qty": 50.0, "price": 99.0},
                             {"order_id": 847, "perm_id": None, "qty": 10.0, "price": 94.0}]}
        _, act, _ = _call(fill)
        x = _extra(act)
        assert x["fill_match"] == "stop_order_id" and x["matched_fill_price"] == 99.0
        assert x["fill_price"] == 101.5 and x["stop_price"] == 100.0
        assert x["broker_trigger_price"] is None and x["broker_order_type"] is None
        assert x["slippage_vs_broker_trigger_pct"] is None

    def test_perm_id_mismatch_is_not_the_cached_order(self):
        # orderId 11 reused by another client (perm 777): the cached order is
        # perm 900, so the fill is not ours. No match at all: symbol VWAP only.
        fill = {"filled_qty": 50.0, "fill_price": 99.0, "fill_time": T_FILL,
                "by_order": [{"order_id": 11, "perm_id": 777, "qty": 50.0, "price": 99.0}]}
        _, act, _ = _call(fill, _cache(_entry(11, 104.0, perm_id=900)))
        x = _extra(act)
        assert x["fill_match"] == "symbol_vwap_perm_id_mismatch"
        assert x["matched_order_id"] is None and x["matched_fill_price"] is None
        assert x["broker_trigger_price"] is None and x["slippage_vs_broker_trigger_pct"] is None
        assert x["fill_price"] == 99.0 and x["stop_price"] == 100.0

    def test_perm_id_mismatch_without_a_record_order_id(self):
        fill = {"filled_qty": 50.0, "fill_price": 99.0, "fill_time": T_FILL,
                "by_order": [{"order_id": 11, "perm_id": 777, "qty": 50.0, "price": 99.0}]}
        _, act, _ = _call(fill, _cache(_entry(11, 104.0, perm_id=900)), order_id=None)
        assert _extra(act)["fill_match"] == "symbol_vwap_perm_id_mismatch"

    def test_a_different_cached_stop_filled(self):
        # The record still names order 11, but a stacked stop 55 filled.
        fill = {"filled_qty": 50.0, "fill_price": 97.0, "fill_time": T_FILL,
                "by_order": [{"order_id": 55, "perm_id": 5500, "qty": 50.0, "price": 97.0}]}
        _, act, _ = _call(fill, _cache(_entry(55, 98.0, perm_id=5500, order_type="STP")))
        x = _extra(act)
        assert x["matched_order_id"] == 55 and x["fill_match"] == "stop_order_perm_id"
        assert x["broker_trigger_price"] == 98.0 and x["broker_order_type"] == "STP"
        assert x["stop_price"] == 100.0

    def test_two_cached_orders_filled_is_ambiguous(self):
        fill = {"filled_qty": 60.0, "fill_price": 98.0, "fill_time": T_FILL,
                "by_order": [{"order_id": 55, "perm_id": 1, "qty": 30.0, "price": 97.0},
                             {"order_id": 56, "perm_id": 2, "qty": 30.0, "price": 99.0}]}
        _, act, _ = _call(fill, _cache(_entry(55, 98.0, perm_id=1), _entry(56, 99.5, perm_id=2)),
                          order_id=None)
        x = _extra(act)
        assert x["fill_match"] == "symbol_vwap_ambiguous_orders"
        assert x["fill_price"] == 98.0 and x["broker_trigger_price"] is None
        assert x["matched_order_id"] is None

    def test_other_tickers_cache_entries_are_ignored(self):
        fill = {"filled_qty": 50.0, "fill_price": 97.0, "fill_time": T_FILL,
                "by_order": [{"order_id": 55, "perm_id": 5500, "qty": 50.0, "price": 97.0}]}
        _, act, _ = _call(fill, _cache(_entry(55, 98.0, perm_id=5500, ticker="NVDA")), order_id=None)
        x = _extra(act)
        assert x["fill_match"] == "symbol_vwap_unrecorded_sells"
        assert x["broker_trigger_price"] is None

    def test_no_by_order_keeps_the_symbol_vwap(self):
        _, act, _ = _call({"filled_qty": 1.0, "fill_price": 95.0, "fill_time": T_FILL})
        x = _extra(act)
        assert x["fill_match"] == "symbol_vwap_unrecorded_sells" and x["fill_price"] == 95.0
        assert x["trailing_active"] is None and x["stop_method"] is None


class TestPrevClose:
    BARS = [{"date": "2026-10-29", "close": 101.0}, {"date": "2026-10-30", "close": 102.0},
            {"date": "2026-11-02", "close": 90.0}]   # the fill day's own bar is not a prev close

    def test_prev_close_from_price_history_before_the_fill_date(self):
        _, act, hist = _call({"filled_qty": 1.0, "fill_price": 95.0, "fill_time": T_FILL},
                             bars=self.BARS)
        x = _extra(act)
        assert x["prev_close"] == 102.0 and x["prev_close_date"] == "2026-10-30"
        assert x["prev_close_source"] == "price_history" and x["prev_close_null_reason"] is None
        hist.assert_called_once_with("DDOG", days=5)

    def test_null_prev_close_says_why(self):
        _, act, _ = _call({"filled_qty": 1.0, "fill_price": 95.0, "fill_time": T_FILL},
                          bars=[{"date": "2026-11-02", "close": 90.0}])
        x = _extra(act)
        assert x["prev_close"] is None and x["prev_close_date"] is None
        assert x["prev_close_null_reason"] == "no_price_history_close_before_fill_date"

    def test_price_history_raising_is_recorded(self):
        with patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.get_state", return_value=""), \
             patch("db.get_price_history", side_effect=RuntimeError("down")), \
             patch("db.log_activity", return_value=True) as act:
            assert tasks._log_stop_slippage("DDOG", 100.0, {"fill_price": 95.0, "fill_time": T_FILL},
                                            "x", 11)
        x = _extra(act)
        assert x["prev_close"] is None and x["prev_close_null_reason"] == "price_history_read_failed"

    def test_caller_value_wins(self):
        _, act, hist = _call({"filled_qty": 1.0, "fill_price": 95.0, "fill_time": T_FILL},
                             bars=self.BARS, prev_close=104.0)
        x = _extra(act)
        assert x["prev_close"] == 104.0 and x["prev_close_source"] == "caller"
        hist.assert_not_called()


class TestStopTriggerCache:
    NOW = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)

    def _run(self, working, existing=""):
        state = {_KEY: existing}
        with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
            tasks._cache_stop_triggers(working, now=self.NOW)
        return json.loads(state[_KEY])

    def test_stores_trigger_type_and_perm(self):
        c = self._run([{"order_id": 11, "perm_id": 900, "ticker": "DDOG", "order_type": "TRAIL",
                        "stop_price": 104.0, "qty": 50.0}])
        assert c == {"11|900": {"ticker": "DDOG", "order_id": 11, "perm_id": 900,
                                "order_type": "TRAIL", "trigger": 104.0,
                                "seen_at": self.NOW.isoformat()}}

    def test_keeps_an_order_no_longer_resting_and_drops_old_ones(self):
        old = _cache(_entry(11, 104.0, perm_id=900, seen_at="2026-11-02T14:45:00+00:00"),
                     _entry(7, 50.0, seen_at="2026-10-28T14:45:00+00:00"))
        c = self._run([], existing=old)
        assert set(c) == {"11|900"} and c["11|900"]["trigger"] == 104.0

    def test_never_raises(self):
        with patch("db.get_state", side_effect=RuntimeError("down")), \
             patch("db.set_state", side_effect=RuntimeError("down")):
            tasks._cache_stop_triggers([{"order_id": 1, "ticker": "X"}])

    def test_a_failed_read_writes_nothing(self):
        """A read error must not be taken as an empty cache and overwrite it."""
        with patch("db.get_state_strict", side_effect=RuntimeError("down")), \
             patch("db.set_state") as set_state:
            tasks._cache_stop_triggers([{"order_id": 11, "perm_id": 900, "ticker": "DDOG",
                                         "order_type": "TRAIL", "stop_price": 104.0}],
                                       now=self.NOW)
        set_state.assert_not_called()

    def test_an_unparseable_cache_is_not_overwritten(self):
        with patch("db.get_state_strict", return_value="[1, 2]"), \
             patch("db.set_state") as set_state:
            tasks._cache_stop_triggers([{"order_id": 11, "ticker": "DDOG"}], now=self.NOW)
        set_state.assert_not_called()

    def test_reads_through_the_strict_getter(self):
        with patch("db.get_state_strict", return_value="") as strict, \
             patch("db.set_state") as set_state:
            tasks._cache_stop_triggers([], now=self.NOW)
        strict.assert_called_once_with(_KEY, "")
        set_state.assert_called_once_with(_KEY, "{}")


class TestReconcileWiring:
    @pytest.fixture(autouse=True)
    def _no_pending_sells(self):
        with patch("db.awaiting_fill_sell_tickers", return_value=set()), \
                patch("db.trade_pending_sell_tickers", return_value=set()):
            yield

    def test_reconcile_caches_the_working_stops_each_cycle(self):
        with patch("tasks._cache_stop_triggers") as cache:
            _dedup.TestStaleStopReconciliation()._run({}, fill_return=None)
        cache.assert_called_once_with([])

    def test_reconcile_passes_trailing_active_and_stop_method(self):
        runner = _dedup.TestStaleStopReconciliation()
        state = {}
        with patch("tasks._log_stop_slippage", return_value=True) as helper:
            runner._run(state, fill_return=None)
            runner._run(state, fill_return=dict(STALE_FILL))
        helper.assert_called_once()
        assert "trailing_active" in helper.call_args.kwargs
        assert "stop_method" in helper.call_args.kwargs

    def test_a_cached_trigger_reaches_the_reconciled_row(self):
        runner = _dedup.TestStaleStopReconciliation()
        # seen_at is real "now": each cycle's _cache_stop_triggers prunes by age.
        state = {_KEY: _cache(_entry(11, 230.10, perm_id=None,
                                                     seen_at=datetime.now(timezone.utc).isoformat()))}
        fill = dict(STALE_FILL, by_order=[{"order_id": 11, "perm_id": None, "qty": 84.0,
                                           "price": 220.00}])
        with patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.get_price_history", return_value=[]), \
             patch("db.log_activity", return_value=True) as act:
            runner._run(state, fill_return=None)
            _, log_trade, *_ = runner._run(state, fill_return=fill)
        rows = [c for c in act.call_args_list if c.args[0] == "stop_slippage"]
        assert len(rows) == 1
        x = rows[0].args[4]
        assert x["stop_price"] == 228.74 and x["broker_trigger_price"] == 230.10
        assert x["slippage_vs_broker_trigger_pct"] == pytest.approx(
            (220.00 - 230.10) / 230.10 * 100, abs=1e-4)
        assert x["fill_match"] == "stop_order_id"
        assert log_trade.call_args.kwargs["price"] == 220.00   # the trade row is unchanged


def test_get_recent_sell_fill_splits_executions_by_order_and_perm():
    import ibkr
    from test_ibkr import _FakeFillRecord
    fake_ib = MagicMock()
    fake_ib.reqExecutions.return_value = [
        _FakeFillRecord("DDOG", order_id=11, shares=50.0, avg_price=228.00, side="SLD", perm_id=900),
        _FakeFillRecord("DDOG", order_id=11, shares=10.0, avg_price=227.00, side="SLD", perm_id=900),
        _FakeFillRecord("DDOG", order_id=11, shares=34.0, avg_price=226.50, side="SLD", perm_id=777),
    ]
    with patch("ibkr.get_ib", return_value=fake_ib):
        r = ibkr.get_recent_sell_fill("DDOG")
    by = {(g["order_id"], g["perm_id"]): g for g in r["by_order"]}
    assert set(by) == {(11, 777), (11, 900)}
    assert by[(11, 900)]["qty"] == 60.0
    assert by[(11, 900)]["price"] == pytest.approx((50 * 228.0 + 10 * 227.0) / 60)
    assert by[(11, 777)]["price"] == 226.50
    assert r["filled_qty"] == 94.0     # the symbol-level result is unchanged
