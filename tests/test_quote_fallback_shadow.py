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
    # the sizing price and the IBKR bar source ride along: the shadow makes no IBKR call
    assert kw["args"][5:] == [100.0, None]
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
    # without a sizing price (an older 5-argument message) drift is unknown
    assert meta["yf_price"] == 178.0 and meta["drift_pct"] is None
    assert meta["would_pass_drift"] is None and meta["ibkr_price_source"] is None


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


# ── the counterfactual a Yahoo fallback would have run (record only) ─────────

import inspect

import pytest

import market_data
import quote_fallback

_ET = "America/New_York"


def _minute_frame(day, closes, opens=None, start="09:30", pre=()):
    """Yahoo-shaped 1-min frame: optional pre-market rows, then RTH bars from start."""
    rth = pd.date_range(f"{day} {start}", periods=len(closes), freq="1min", tz=_ET)
    idx = list(pd.DatetimeIndex([pd.Timestamp(f"{day} {t}", tz=_ET) for t in pre])) + list(rth)
    o = list(opens or closes)
    return pd.DataFrame({"Open": [100.0] * len(pre) + o,
                         "Close": [100.0] * len(pre) + list(closes)}, index=pd.DatetimeIndex(idx))


def _daily_frame(day_closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz=_ET) for d, _ in day_closes])
    return pd.DataFrame({"Close": [c for _, c in day_closes]}, index=idx)


def test_shadow_row_records_drift_source_and_gap_without_an_ibkr_call():
    seen = datetime(2026, 11, 4, 14, 33, 30, tzinfo=timezone.utc)      # 09:33:30 ET
    minute = _minute_frame("2026-11-04", [104.0, 105.0, 106.0, 106.5], opens=[106.0] * 4)
    daily = _daily_frame([("2026-11-03", 100.0), ("2026-11-04", 106.5)])
    with patch("quote_fallback.minute_bars", return_value=minute), \
         patch("quote_fallback.daily_bars", return_value=daily), \
         patch("market_data.get_historical_bars", side_effect=AssertionError("IBKR call")), \
         patch("ibkr.get_ib", side_effect=AssertionError("IBKR call")), \
         patch("db.log_activity") as act:
        tasks.stale_quote_fallback_shadow("QUBT", 9, 2, 2251.0, seen.isoformat(), 100.0,
                                          "daily_fallback")
    meta = act.call_args.args[4]
    assert meta["yf_price"] == 106.5 and meta["yf_price_found"] is True
    assert meta["drift_pct"] == 6.5 and meta["would_pass_drift"] is True
    assert meta["ibkr_price_source"] == "daily_fallback"
    assert meta["yf_gap_status"] == "pending"          # a 6% gap, 4 bars since the open
    assert meta["bar_in_rth"] is True and meta["yf_age_min"] == 0.5


def test_drift_beyond_the_abort_does_not_pass():
    assert quote_fallback.drift_record(108.01, 100.0) == {"drift_pct": 8.01, "would_pass_drift": False}
    assert quote_fallback.drift_record(92.0, 100.0) == {"drift_pct": -8.0, "would_pass_drift": True}
    assert quote_fallback.drift_record(None, 100.0)["would_pass_drift"] is None
    assert quote_fallback.drift_record(100.0, "x")["drift_pct"] is None
    assert quote_fallback.drift_record(100.0, 0)["drift_pct"] is None


def test_drift_threshold_is_executions_abort():
    assert quote_fallback.DRIFT_ABORT_FRAC == 0.08
    assert "if drift > 0.08:" in inspect.getsource(execution.execute_approved)


def test_fallback_quote_fields_and_errors():
    seen = datetime(2026, 11, 4, 14, 33, tzinfo=timezone.utc)
    minute = _minute_frame("2026-11-04", [10.0, 11.0], pre=("09:28",))
    q = quote_fallback.get_fallback_quote("X", seen, bars=minute)
    assert q["price"] == 11.0 and q["in_rth"] is True and q["source"] == "yahoo_1m"
    assert q["age_min"] == 2.0 and q["error"] is None
    early = datetime(2026, 11, 4, 14, 29, 30, tzinfo=timezone.utc)   # 09:29:30 ET
    q = quote_fallback.get_fallback_quote("X", early, bars=minute)
    assert q["price"] == 100.0 and q["in_rth"] is False                 # the pre-market row
    with patch("yfinance.Ticker", side_effect=RuntimeError("yf")):
        q = quote_fallback.get_fallback_quote("X", seen)
    assert q["price"] is None and q["error"] == "RuntimeError"


def _ibkr_status(day, minute, daily):
    """check_gap_confirmation on IBKR-shaped bars carrying the same numbers."""
    rth = minute[[quote_fallback._in_rth(ts) for ts in minute.index]]
    intraday = [{"time": str(ts), "open": float(o), "close": float(c)}
                for ts, o, c in zip(rth.index, rth["Open"], rth["Close"])]
    dly = [{"time": ts.date().isoformat(), "close": float(c)}
           for ts, c in zip(daily.index, daily["Close"])]

    def bars(ticker, duration, bar_size, **kw):
        return dly if bar_size == "1 day" else intraday
    with patch("market_data.get_historical_bars", side_effect=bars):
        return market_data.check_gap_confirmation("X", today=day)["status"]


@pytest.mark.parametrize("closes, opens, want", [
    ([101.0] * 3, [101.0] * 3, "no_gap"),                     # 1% open
    ([106.0] * 4, [106.0] * 4, "pending"),                    # 6% gap, 4 bars
    ([106.0] * 12, [106.0] * 12, "confirmed"),                # 6% gap holding, 12 bars
    ([106.0, 104.0, 102.5], [106.0] * 3, "failed"),           # gave back over half
    ([94.0] * 12, [94.0] * 12, "confirmed"),                  # gap down holding
])
def test_gap_status_matches_check_gap_confirmation_on_the_same_bars(closes, opens, want):
    minute = _minute_frame("2026-11-04", closes, opens=opens, pre=("08:00", "09:15"))
    daily = _daily_frame([("2026-11-02", 99.0), ("2026-11-03", 100.0), ("2026-11-04", closes[-1])])
    at = datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)
    got = quote_fallback.fallback_gap_status("X", at, bars=minute, daily=daily)
    assert got == want == _ibkr_status("2026-11-04", minute, daily)


def test_gap_status_only_counts_bars_before_seen_and_never_raises():
    minute = _minute_frame("2026-11-04", [106.0] * 12)
    daily = _daily_frame([("2026-11-03", 100.0)])
    at = datetime(2026, 11, 4, 14, 34, 30, tzinfo=timezone.utc)      # 5 bars started by 09:34:30
    assert quote_fallback.fallback_gap_status("X", at, bars=minute, daily=daily) == "pending"
    before_open = datetime(2026, 11, 4, 14, 29, tzinfo=timezone.utc)
    assert quote_fallback.fallback_gap_status("X", before_open, bars=minute, daily=daily) == "unreadable"
    assert quote_fallback.fallback_gap_status("X", at, bars=minute,
                                              daily=_daily_frame([("2026-11-04", 1.0)])) == "unreadable"
    with patch("quote_fallback.daily_bars", side_effect=RuntimeError("yf")):
        assert quote_fallback.fallback_gap_status("X", at, bars=minute) == "error:RuntimeError"


def test_yf_last_minute_unchanged_through_quote_fallback():
    seen = datetime(2026, 11, 4, 14, 33, tzinfo=timezone.utc)
    idx = [pd.Timestamp(seen - timedelta(minutes=2)), pd.Timestamp(seen + timedelta(minutes=3))]
    df = pd.DataFrame({"Close": [178.0, 179.0]}, index=idx)
    with patch("yfinance.Ticker") as T:
        T.return_value.history.return_value = df
        assert tasks._yf_last_minute("ANET", seen) == (178.0, idx[0].isoformat())
        assert tasks._yf_last_minute("ANET") == (179.0, idx[1].isoformat())
    T.return_value.history.assert_called_with(period="1d", interval="1m", prepost=True)


@pytest.mark.parametrize("source, text", [
    ("daily_fallback", "stale IBKR quote (check 1)"),
    (None, "stale IBKR quote (check 1)"),
    ("none", "no IBKR quote (check 1)"),
    ("error: TimeoutError", "IBKR quote check raised (check 1)"),
])
def test_shadow_row_text_names_the_case(source, text):
    with patch("quote_fallback.minute_bars", return_value=None), \
         patch("quote_fallback.daily_bars", return_value=None), \
         patch("db.log_activity") as act:
        tasks.stale_quote_fallback_shadow("QUBT", 9, 1, None, None, 100.0, source)
    assert act.call_args.args[2].startswith(text)
    assert act.call_args.args[4]["ibkr_price_source"] == source
