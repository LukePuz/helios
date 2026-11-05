from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import execution
import tasks


def _approval(action="ADD", price_at_sizing=100.0):
    return {"id": 1, "ticker": "NVDA", "action": action, "conviction": 7,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 100, "price_at_sizing": price_at_sizing,
                          "stop_loss_price": 90.0}}


_STALE = {"price": 100.0, "is_stale": True, "age_minutes": 2251.0, "market_open": True}


def test_stale_buy_queues_the_shadow_on_ops():
    approval = _approval(action="ADD", price_at_sizing=100.0)
    stale = _STALE
    with patch("ibkr.get_ib"), patch("ibkr.get_client"), \
         patch("risk_manager.RiskManager") as rm, patch("ibkr.place_market_order"), \
         patch("market_data.get_priced_quote", return_value=stale), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks.retry_deferred_entry"), patch("time.sleep"), patch("alerts._send"), \
         patch("tasks.stale_quote_fallback_shadow") as shadow:
        rm.return_value.is_halted.return_value = False
        execution.execute_approved(approval, price_retry_attempt=1)
    _, kw = shadow.apply_async.call_args
    assert kw["args"][:4] == [approval["ticker"], approval["id"], 1, 2251.0]
    assert datetime.fromisoformat(kw["args"][4]).tzinfo is not None
    assert kw["queue"] == "ops" and kw["retry"] is False


def test_a_queue_failure_does_not_change_the_outcome():
    approval = _approval(action="ADD", price_at_sizing=100.0)
    stale = _STALE
    with patch("ibkr.get_ib"), patch("ibkr.get_client"), \
         patch("risk_manager.RiskManager") as rm, patch("ibkr.place_market_order") as order, \
         patch("market_data.get_priced_quote", return_value=stale), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks.retry_deferred_entry") as retry, patch("time.sleep"), patch("alerts._send"), \
         patch("tasks.stale_quote_fallback_shadow") as shadow:
        shadow.apply_async.side_effect = RuntimeError("redis down")
        rm.return_value.is_halted.return_value = False
        assert execution.execute_approved(approval, price_retry_attempt=1) is False
    order.assert_not_called()
    retry.apply_async.assert_called_once()


def test_shadow_row_records_the_yahoo_bar_age_from_seen_at():
    seen = datetime(2026, 11, 4, 14, 33, tzinfo=timezone.utc)          # 09:33 ET
    bar_at = seen - timedelta(minutes=2)                                # 09:31 ET bar start
    df = pd.DataFrame({"Close": [178.0]}, index=[pd.Timestamp(bar_at)])
    with patch("yfinance.Ticker") as T, patch("db.log_activity") as act:
        T.return_value.history.return_value = df
        tasks.stale_quote_fallback_shadow("ANET", 7, 1, 2310.2, seen.isoformat())
    meta = act.call_args.args[4]
    assert meta["yf_price_found"] is True and meta["yf_age_min"] == 2.0
    assert meta["bar_in_rth"] is True and meta["seen_at"] == seen.isoformat()
    assert meta["ibkr_age_min"] == 2310.2 and meta["error"] is None
    assert "price" not in meta    # no price stored, only whether one was found


def test_shadow_row_on_a_yahoo_error():
    with patch("yfinance.Ticker", side_effect=RuntimeError("yf")), patch("db.log_activity") as act:
        tasks.stale_quote_fallback_shadow("ANET", 7, 2, 2310.2)
    meta = act.call_args.args[4]
    assert meta["yf_price_found"] is False and meta["error"] == "RuntimeError"


def test_a_premarket_bar_is_not_in_rth():
    seen = datetime(2026, 11, 4, 14, 30, 30, tzinfo=timezone.utc)      # 09:30:30 ET
    df = pd.DataFrame({"Close": [178.0]}, index=[pd.Timestamp(seen - timedelta(minutes=1))])
    with patch("yfinance.Ticker") as T, patch("db.log_activity") as act:
        T.return_value.history.return_value = df
        tasks.stale_quote_fallback_shadow("ANET", 7, 1, 2310.2, seen.isoformat())
    assert act.call_args.args[4]["bar_in_rth"] is False


def test_task_has_time_limits():
    t = tasks.stale_quote_fallback_shadow
    assert t.soft_time_limit == 20 and t.time_limit == 30


def test_an_after_hours_bar_is_not_in_rth():
    seen = datetime(2026, 11, 4, 21, 5, tzinfo=timezone.utc)           # 16:05 ET
    df = pd.DataFrame({"Close": [178.0]}, index=[pd.Timestamp(seen - timedelta(minutes=1))])
    with patch("yfinance.Ticker") as T, patch("db.log_activity") as act:
        T.return_value.history.return_value = df
        tasks.stale_quote_fallback_shadow("ANET", 7, 1, 2310.2, seen.isoformat())
    assert act.call_args.args[4]["bar_in_rth"] is False


def test_a_bar_after_seen_at_is_ignored():
    seen = datetime(2026, 11, 4, 14, 33, tzinfo=timezone.utc)
    idx = [pd.Timestamp(seen - timedelta(minutes=2)), pd.Timestamp(seen + timedelta(minutes=3))]
    df = pd.DataFrame({"Close": [178.0, 179.0]}, index=idx)
    with patch("yfinance.Ticker") as T, patch("db.log_activity") as act:
        T.return_value.history.return_value = df
        tasks.stale_quote_fallback_shadow("ANET", 7, 1, 2310.2, seen.isoformat())
    assert act.call_args.args[4]["yf_age_min"] == 2.0
