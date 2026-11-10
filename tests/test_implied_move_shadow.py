"""Log-only implied-move shadow: the nearest-expiry ATM straddle computed
from the chain poll_options_flow already downloads (ingestion.options_free),
logged to one system_state map and one activity_feed row per ticker per ET
date. No DB, Redis or network: everything is patched."""
import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from ingestion import options_free
import tasks

ET = ZoneInfo("America/New_York")
TODAY = date(2026, 11, 3)


def _leg(strikes, bid, ask, last, iv, vol=10, oi=100):
    n = len(strikes)
    return pd.DataFrame({"strike": strikes, "bid": [bid] * n, "ask": [ask] * n,
                         "lastPrice": [last] * n, "impliedVolatility": [iv] * n,
                         "volume": [vol] * n, "openInterest": [oi] * n})


# ── the computation ──────────────────────────────────────────────────────────

def test_atm_is_nearest_common_strike_and_mid_used():
    calls = _leg([95.0, 100.0, 105.0], 2.0, 2.2, 9.0, 0.4)
    puts = _leg([95.0, 100.0, 105.0], 1.8, 2.0, 9.0, 0.45)
    m = options_free._implied_move(calls, puts, {"regularMarketPrice": 101.0}, "2026-11-07", TODAY)
    assert m["atm_strike"] == 100.0
    assert m["straddle"] == 4.0                     # 2.1 + 1.9
    assert m["price_source"] == "mid"
    assert m["implied_move_pct"] == round(4.0 / 101.0 * 100, 4)
    assert m["underlying"] == 101.0
    assert m["expiry"] == "2026-11-07" and m["dte"] == 4
    assert m["call_iv"] == 0.4 and m["put_iv"] == 0.45


def test_atm_tie_goes_to_lower_strike_and_needs_both_sides():
    calls = _leg([95.0, 100.0, 105.0], 1, 1.2, 1, 0.3)
    puts = _leg([95.0, 105.0], 1, 1.2, 1, 0.3)       # 100 only quoted on the call side
    m = options_free._implied_move(calls, puts, {"regularMarketPrice": 100.0}, "2026-11-03", TODAY)
    assert m["atm_strike"] == 95.0 and m["dte"] == 0


def test_no_bid_ask_falls_back_to_last():
    calls = _leg([100.0], 0.0, 0.0, 3.0, 0.5)
    puts = _leg([100.0], float("nan"), float("nan"), 2.0, float("nan"))
    m = options_free._implied_move(calls, puts, {"regularMarketPrice": 100.0}, "2026-11-07", TODAY)
    assert m["straddle"] == 5.0 and m["price_source"] == "last"
    assert m["call_price_source"] == "last" and m["put_price_source"] == "last"
    assert m["put_iv"] is None


def test_mixed_sources_are_recorded():
    calls = _leg([100.0], 2.0, 2.2, 9.0, 0.5)
    puts = _leg([100.0], 0.0, 0.0, 2.0, 0.5)
    m = options_free._implied_move(calls, puts, {"regularMarketPrice": 100.0}, "2026-11-07", TODAY)
    assert m["price_source"] == "mixed"
    assert (m["call_price_source"], m["put_price_source"]) == ("mid", "last")


def test_crossed_quote_uses_last():
    calls = _leg([100.0], 2.5, 2.0, 2.2, 0.5)
    puts = _leg([100.0], 2.0, 2.2, 2.2, 0.5)
    m = options_free._implied_move(calls, puts, {"regularMarketPrice": 100.0}, "2026-11-07", TODAY)
    assert m["call_price_source"] == "last"


def test_missing_inputs_give_none():
    calls = _leg([100.0], 2.0, 2.2, 2.0, 0.5)
    empty = calls.iloc[0:0]
    q = {"regularMarketPrice": 100.0}
    assert options_free._implied_move(empty, empty, q, "2026-11-07", TODAY) is None
    assert options_free._implied_move(None, None, q, "2026-11-07", TODAY) is None
    assert options_free._implied_move(calls, calls, {}, "2026-11-07", TODAY) is None
    assert options_free._implied_move(calls, calls, None, "2026-11-07", TODAY) is None
    no_price = _leg([100.0], 0.0, 0.0, 0.0, 0.5)
    assert options_free._implied_move(no_price, calls, q, "2026-11-07", TODAY) is None
    assert options_free._implied_move("garbage", calls, q, "x", TODAY) is None   # swallowed


# ── get_flow_summary: same return, one chain request ─────────────────────────

def _fake_ticker():
    calls = _leg([95.0, 100.0, 105.0], 2.0, 2.2, 2.1, 0.4, vol=30)
    puts = _leg([95.0, 100.0, 105.0], 1.8, 2.0, 1.9, 0.45, vol=10)
    t = MagicMock()
    t.options = ("2026-11-07", "2026-11-14")
    t.option_chain.return_value = SimpleNamespace(
        calls=calls, puts=puts, underlying={"regularMarketPrice": 101.0})
    return t


def test_flow_summary_unchanged_and_no_extra_yfinance_calls():
    t1, t2 = _fake_ticker(), _fake_ticker()
    with patch("ingestion.options_free.yf.Ticker", return_value=t1) as yt1:
        plain = options_free.get_flow_summary("NVDA")
    sink = {}
    with patch("ingestion.options_free.yf.Ticker", return_value=t2) as yt2:
        with_sink = options_free.get_flow_summary("NVDA", sink)
    assert plain == with_sink
    assert yt1.call_count == yt2.call_count == 1
    assert t1.option_chain.call_count == t2.option_chain.call_count == 1
    t2.option_chain.assert_called_once_with("2026-11-07")
    assert sink["NVDA"]["atm_strike"] == 100.0 and sink["NVDA"]["expiry"] == "2026-11-07"


def test_flow_summary_survives_shadow_failure():
    t = _fake_ticker()
    sink = {}
    with patch("ingestion.options_free.yf.Ticker", return_value=t), \
            patch("ingestion.options_free._implied_move", side_effect=RuntimeError("boom")):
        out = options_free.get_flow_summary("NVDA", sink)
    assert out["sentiment"] == "BULLISH" and sink == {}


def test_all_flow_summaries_fills_sink_and_keeps_return():
    with patch("knowledge_loader.get_all_tickers", return_value=["NVDA", "AMD"]), \
            patch("ingestion.options_free.yf.Ticker", side_effect=lambda _t: _fake_ticker()), \
            patch("ingestion.options_free.time.sleep"):
        plain = options_free.get_all_flow_summaries()
        sink = {}
        same = options_free.get_all_flow_summaries(implied_sink=sink)
    assert plain == same and set(sink) == {"NVDA", "AMD"}


# ── the logger ──────────────────────────────────────────────────────────────

_MOVE = {"implied_move_pct": 4.0, "straddle": 4.0, "price_source": "mid",
         "call_price_source": "mid", "put_price_source": "mid", "underlying": 100.0,
         "atm_strike": 100.0, "expiry": "2026-11-07", "dte": 4, "call_iv": 0.4, "put_iv": 0.45}


class _FakeDB:
    def __init__(self):
        self.state = {}
        self.rows = []
        self.calls = {"get": 0, "set": 0, "keys": 0, "stops": 0}

    def get_state_strict(self, k, d=None):
        self.calls["get"] += 1
        return self.state.get(k, d)

    def set_state(self, k, v):
        self.calls["set"] += 1
        self.state[k] = v

    def get_activity_shadow_keys(self, stage):
        self.calls["keys"] += 1
        return {r[4]["shadow_key"] for r in self.rows if r[0] == stage}

    def get_all_position_stops(self):
        self.calls["stops"] += 1
        return [{"ticker": "NVDA", "stop_price": 92.0}]

    def log_activity(self, *a):
        self.rows.append(a)
        return True


def _run(fake, moves, at, held=frozenset({"NVDA"})):
    with patch.multiple("db", get_state_strict=fake.get_state_strict, set_state=fake.set_state,
                        get_activity_shadow_keys=fake.get_activity_shadow_keys,
                        get_all_position_stops=fake.get_all_position_stops,
                        log_activity=fake.log_activity), \
            patch("tasks._held_tickers", return_value=set(held)), \
            patch("agents.analyst.shadow_versions",
                  return_value={"code_sha": "abc", "env_sha": "def"}):
        return tasks._log_implied_move_shadow(moves, now=at)


def test_one_row_per_ticker_per_day_after_0945():
    fake = _FakeDB()
    moves = {"NVDA": dict(_MOVE), "AMD": dict(_MOVE, implied_move_pct=6.0)}
    assert _run(fake, moves, datetime(2026, 11, 3, 9, 40, tzinfo=ET)) == 0   # before 09:45
    assert fake.rows == [] and fake.calls["set"] == 1
    assert _run(fake, moves, datetime(2026, 11, 3, 9, 45, tzinfo=ET)) == 2
    assert _run(fake, moves, datetime(2026, 11, 3, 9, 50, tzinfo=ET)) == 0
    assert _run(fake, moves, datetime(2026, 11, 3, 15, 55, tzinfo=ET)) == 0
    assert len(fake.rows) == 2
    # Later same-day cycles: one state read + one state write each, nothing else.
    assert fake.calls == {"get": 4, "set": 4, "keys": 1, "stops": 1}
    assert _run(fake, moves, datetime(2026, 11, 4, 9, 46, tzinfo=ET)) == 2
    assert len(fake.rows) == 4

    by = {r[1]: r for r in fake.rows[:2]}
    stage, _t, _msg, _sent, extra = by["NVDA"]
    assert stage == "options_shadow"
    for k in ("ticker", "implied_move_pct", "straddle", "price_source", "underlying",
              "atm_strike", "expiry", "dte", "call_iv", "put_iv", "et_date",
              "code_sha", "env_sha", "shadow_key"):
        assert k in extra
    assert extra["et_date"] == "2026-11-03" and extra["held"] is True
    assert extra["stop_distance_pct"] == 8.0 and extra["implied_move_over_stop"] == 0.5
    amd = by["AMD"][4]
    assert amd["held"] is False and "stop_distance_pct" not in amd
    assert "implied_move_over_stop" not in amd
    # Extra expiries (audit 2026-11-07 19:07 #4): held names only here, as
    # the earnings read fails closed to {} without a DB.
    assert extra["extra_expiries"] == {"fetched": True, "stub": True, "earnings_window_days": 14}
    assert amd["extra_expiries"]["fetched"] is False

    book = json.loads(next(iter(fake.state.values())))
    assert book["NVDA"]["logged_et_date"] == "2026-11-04"
    assert book["AMD"]["implied_move_pct"] == 6.0


def test_shadow_key_dedups_after_a_lost_state_write():
    fake = _FakeDB()
    moves = {"NVDA": dict(_MOVE)}
    _run(fake, moves, datetime(2026, 11, 3, 10, 0, tzinfo=ET))
    fake.state.clear()                                    # state lost
    assert _run(fake, moves, datetime(2026, 11, 3, 10, 5, tzinfo=ET)) == 0
    assert len(fake.rows) == 1


def test_held_without_stop_record_gets_null_keys():
    fake = _FakeDB()
    fake.get_all_position_stops = lambda: []
    _run(fake, {"NVDA": dict(_MOVE)}, datetime(2026, 11, 3, 10, 0, tzinfo=ET))
    extra = fake.rows[0][4]
    assert extra["stop_distance_pct"] is None and extra["implied_move_over_stop"] is None


def test_failed_shadow_key_read_skips_logging():
    fake = _FakeDB()
    fake.get_activity_shadow_keys = lambda stage: None
    assert _run(fake, {"NVDA": dict(_MOVE)}, datetime(2026, 11, 3, 10, 0, tzinfo=ET)) == 0
    assert fake.rows == []
    assert json.loads(next(iter(fake.state.values())))["NVDA"]["logged_et_date"] is None


def test_failed_activity_write_retries_next_cycle():
    fake = _FakeDB()
    ok = fake.log_activity
    fake.log_activity = lambda *a: False
    _run(fake, {"NVDA": dict(_MOVE)}, datetime(2026, 11, 3, 10, 0, tzinfo=ET))
    fake.log_activity = ok
    assert _run(fake, {"NVDA": dict(_MOVE)}, datetime(2026, 11, 3, 10, 5, tzinfo=ET)) == 1


def test_errors_are_swallowed():
    fake = _FakeDB()

    def boom(*a, **k):
        raise RuntimeError("db down")
    fake.get_state_strict = boom
    assert _run(fake, {"NVDA": dict(_MOVE)}, datetime(2026, 11, 3, 10, 0, tzinfo=ET)) == 0
    fake = _FakeDB()
    fake.log_activity = boom
    fake.set_state = boom
    assert _run(fake, {"NVDA": dict(_MOVE)}, datetime(2026, 11, 3, 10, 0, tzinfo=ET)) == 0
    assert tasks._log_implied_move_shadow({}) == 0


def test_poll_options_flow_shadow_failure_does_not_change_dispatch():
    summaries = {"NVDA": {"unusual": True, "summary": "x"}}

    def fake_all(implied_sink=None):
        if implied_sink is not None:
            implied_sink["NVDA"] = dict(_MOVE)
        return summaries

    with patch("market_data._is_market_hours", return_value=True), \
            patch("ingestion.options_flow._KEY_CONFIGURED", False), \
            patch("ingestion.options_free.get_all_flow_summaries", side_effect=fake_all), \
            patch("db.is_seen", return_value=False), patch("db.mark_seen"), \
            patch("db.get_state_strict", side_effect=RuntimeError("db down")), \
            patch("tasks._dispatch_event") as disp, patch("tasks._heartbeat_ok") as hb:
        tasks.poll_options_flow.run()
    disp.assert_called_once()
    hb.assert_called_once_with("options_flow", 1)


def test_poll_uses_the_options_flow_key_test_so_an_empty_key_takes_the_free_path():
    """Audit 2026-11-07 05:07 #6: tasks tested only startswith("your_"), so an
    empty key took the paid path and the implied-move shadow went off."""
    import importlib
    from ingestion import options_flow
    with patch.dict("os.environ", {"UNUSUAL_WHALES_API_KEY": ""}):
        assert importlib.reload(options_flow)._KEY_CONFIGURED is False
    importlib.reload(options_flow)
    with patch("market_data._is_market_hours", return_value=True), \
            patch("ingestion.options_flow._KEY_CONFIGURED", False), \
            patch("ingestion.options_free.get_all_flow_summaries", return_value={}) as free, \
            patch("ingestion.options_flow.get_all_flow_summaries") as paid, \
            patch("tasks._heartbeat_ok"):
        tasks.poll_options_flow.run()
    free.assert_called_once()
    paid.assert_not_called()
    assert "implied_sink" in free.call_args.kwargs

