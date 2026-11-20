"""The regime's "200DMA" averages about 173 bars, because the live request
spans 250 calendar days. With REGIME_SHADOW_200_ENABLED each assessment makes
a second, wider request, re-scores trend, the ETF-proxy breadth fallback and
sector price action on it, and stores the score and label they imply under
signals.trend.shadow_200.

Record-only. These tests pin that the live packet, the analyst prompt and the
panel's macro slice are the same with the flag on and off, that a failed,
slow or short fetch only sets a status, and that the shadow's 200DMA is the
last 200 completed bars. No network: every bar fetch here is a fake."""
import contextlib
import copy
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from agents import analyst
from agents import market_regime as mr
from helios3d.panel import roles

# The real request function, taken at import: the conftest stubs it per test.
_REAL_FETCH_SHADOW_BARS = mr._fetch_shadow_bars

NOW = datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc)    # a Monday, 15:00 ET: session open
TODAY = "2026-11-16"
LAST_SESSION = date(2026, 11, 13)

_CAND = {"ticker": "NVDA", "role": "Accelerator", "tier": "Core", "sector": "Semis", "held": True}
_MARKOV_OFF = {"available": False, "notes": "x"}


class _Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _frozen_clock_and_caches(monkeypatch):
    """15:00 ET on TODAY, so a bar dated TODAY is unfinished and
    completed_bars drops it; the module caches are restored afterwards."""
    monkeypatch.setattr(mr, "datetime", _Clock)
    monkeypatch.setattr(mr, "_regime_cache", None)
    monkeypatch.setattr(mr, "_cache_ts", 0.0)
    monkeypatch.setattr(mr, "_sector_regime_cache", {})
    monkeypatch.setattr(mr, "_sector_cache_ts", {})


def _bars(closes, in_progress=None):
    """One bar per calendar day ending on LAST_SESSION. in_progress appends
    today's unfinished bar with that close."""
    n = len(closes)
    out = [{"date": (LAST_SESSION - timedelta(days=n - 1 - i)).isoformat(), "open": c,
            "high": c + 1, "low": c - 1, "close": c, "volume": 1_000_000}
           for i, c in enumerate(closes)]
    if in_progress is not None:
        out.append({"date": TODAY, "open": in_progress, "high": in_progress + 1,
                    "low": in_progress - 1, "close": in_progress, "volume": 5_000})
    return out


RISING = [100 + 0.1 * i for i in range(173)]
LIVE = _bars(RISING, in_progress=1.0)                         # the live window: 173 completed bars
# 226 completed bars each. The 53 older ones decide where the true 200DMA sits.
WIDE_HIGH = _bars([200.0] * 53 + RISING, in_progress=1.0)     # above the price: trend 0, live says 5
WIDE_LOW = _bars([50.0] * 53 + RISING, in_progress=1.0)       # below the price: trend 5, as live
WIDE_SAME = _bars([100 + 0.1 * i for i in range(-53, 173)], in_progress=1.0)

FALLING = [150 - 0.1 * i for i in range(173)]
LIVE_FALLING = _bars(FALLING, in_progress=1.0)                # Death Cross on 173 bars: trend -5
WIDE_FALLING = _bars([10.0] * 53 + FALLING, in_progress=1.0)  # true 200DMA below the price: trend -2
WEAK_NAME = _bars([200 - 0.1 * i for i in range(210)])        # a breadth-sample name under its 200DMA


def _shadow_fetch(shadow):
    """A fake _fetch_shadow_bars: returns bars (or None), raises an exception
    instance, or runs a callable."""
    if callable(shadow) or isinstance(shadow, BaseException):
        return MagicMock(side_effect=shadow)
    return MagicMock(return_value=shadow)


def _overall(flag, shadow, *, live=LIVE, macro=None, flows_error=None, watchlist=(), **kw):
    """assess_regime() with every input faked. Returns (packet, the fake
    shadow fetch, the db.log_regime mock)."""
    fetch = _shadow_fetch(shadow)
    with patch("config.REGIME_SHADOW_200_ENABLED", flag), \
         patch.object(mr, "_fetch_spy_bars", return_value=list(live)), \
         patch.object(mr, "_fetch_shadow_bars", fetch), \
         patch("ingestion.macro.get_macro_snapshot", return_value=macro or {}), \
         patch.object(mr, "_fetch_watchlist_bars_for_breadth", return_value=list(watchlist)), \
         patch("ingestion.options_flow.get_etf_flows", return_value={}, side_effect=flows_error), \
         patch("agents.markov_regime.get_markov_packet", return_value=_MARKOV_OFF), \
         patch("db.log_regime") as log_regime:
        packet = mr.assess_regime(**kw)
    return packet, fetch, log_regime


def _sector(flag, shadow, *, live=LIVE, macro=None, bulk=None, price_action=None, **kw):
    """assess_sector_regime("XLK") with every input faked. bulk is the breadth
    sample's bars; empty means the live breadth is the ETF proxy."""
    fetch = _shadow_fetch(shadow)
    pa = (patch.object(mr, "_score_price_action_volume", side_effect=price_action)
          if price_action else contextlib.nullcontext())
    with patch("config.REGIME_SHADOW_200_ENABLED", flag), \
         patch.object(mr, "_fetch_bars", return_value=list(live)), \
         patch.object(mr, "_fetch_shadow_bars", fetch), \
         patch("ingestion.macro.get_macro_snapshot", return_value=macro or {}), \
         patch("db.get_price_history_bulk", return_value=bulk or {}), \
         patch("agents.candidate_scanner._bars_age", return_value=(None, None, 1.0)), \
         patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
         patch("agents.markov_regime.get_markov_packet", return_value=_MARKOV_OFF), \
         pa, \
         patch("db.log_regime") as log_regime:
        packet = mr.assess_sector_regime("XLK", breadth_tickers=["A"], regime_type="sector", **kw)
    return packet, fetch, log_regime


def _without_shadow(packet):
    out = copy.deepcopy(packet)
    out["signals"]["trend"].pop("shadow_200", None)
    return out


def _implied(packet, weights, normalize, **swapped):
    """The live weighted sum with some raw scores swapped."""
    raws = {name: packet["signals"][name]["score"] for name in weights}
    raws.update(swapped)
    return normalize(sum(raws[name] * w for name, w in weights.items()))


# ---------------------------------------------------------------------------
# (a) The live packet is the same with the flag on and off
# ---------------------------------------------------------------------------

_SHADOW_CASES = [
    pytest.param(WIDE_HIGH, "ok", id="scored-and-differs"),
    pytest.param(WIDE_SAME, "ok", id="scored-and-agrees"),
    pytest.param(_bars(RISING + RISING[:17], in_progress=1.0), "too_few_bars", id="190-bars"),
    pytest.param([], "too_few_bars", id="no-bars"),
    pytest.param(RuntimeError("boom"), "fetch_failed", id="fetch-raises"),
    pytest.param(requests.exceptions.ReadTimeout("slow"), "timeout", id="request-times-out"),
    pytest.param(None, "no_credentials", id="no-credentials"),
    pytest.param([{"date": "2026-01-02"}] * 210, "error", id="malformed-bars"),
]


@pytest.mark.parametrize("run, ticker", [(_overall, "SPY"), (_sector, "XLK")])
@pytest.mark.parametrize("shadow, status", _SHADOW_CASES)
def test_live_packet_is_identical_flag_on_and_off(run, ticker, shadow, status):
    off, off_fetch, off_log = run(False, shadow)
    on, on_fetch, on_log = run(True, shadow)

    assert "shadow_200" not in off["signals"]["trend"]
    off_fetch.assert_not_called()                       # flag off: no second request
    on_fetch.assert_called_once_with(ticker)
    assert on["signals"]["trend"]["shadow_200"]["status"] == status

    assert (on["regime"], on["score"]) == (off["regime"], off["score"])
    assert on["reasoning"] == off["reasoning"]
    assert set(on["signals"]) == set(off["signals"])
    for key in off["signals"]["trend"]:
        assert on["signals"]["trend"][key] == off["signals"]["trend"][key]
    assert _without_shadow(on) == off                   # every other key, whole packet

    # The DB row: same label, score, confidence, impact and reasoning.
    a, b = on_log.call_args, off_log.call_args
    assert a.args[:3] == b.args[:3] and a.args[4:] == b.args[4:] and a.kwargs == b.kwargs
    assert a.args[3]["trend"]["shadow_200"]["status"] == status
    assert "shadow_200" not in b.args[3]["trend"]


def test_shadow_disagrees_without_moving_the_live_score():
    off, _, _ = _overall(False, WIDE_HIGH)
    on, _, _ = _overall(True, WIDE_HIGH)
    rec = on["signals"]["trend"]["shadow_200"]
    live_trend = on["signals"]["trend"]["score"]
    assert live_trend == 5 and rec["trend_score"] == 0  # 173-bar mean below price, true 200DMA above
    assert rec["score"] == _implied(on, mr._WEIGHTS, mr._normalize, trend=0)
    assert rec["regime"] == mr._regime_from_score(rec["score"])
    assert rec["differs"] is True and rec["score"] != on["score"]
    assert rec["regime_differs"] is (rec["regime"] != on["regime"])
    assert (on["score"], on["regime"]) == (off["score"], off["regime"])
    assert on["signals"]["trend"]["ma200_bars_used"] == 173      # the live window is untouched
    assert "200DMA" in on["signals"]["trend"]["notes"]
    assert on["signals"]["trend"]["notes"] == off["signals"]["trend"]["notes"]


def test_shadow_that_agrees_reports_no_difference():
    on, _, _ = _overall(True, WIDE_SAME)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "ok" and rec["trend_score"] == on["signals"]["trend"]["score"]
    assert rec["score"] == on["score"] and rec["regime"] == on["regime"]
    assert rec["differs"] is False and rec["regime_differs"] is False
    assert rec["breadth_rescored"] is False
    assert rec["breadth_score"] == on["signals"]["breadth"]["score"]
    assert rec["price_action_score"] is None            # sector-only input
    assert rec["price_action_rescored"] is None
    assert rec["live_bars"] == 173 and rec["same_last_bar"] is True


# A failed live request (_fetch_bars returns []) or a short one leaves the live
# trend unscored; 50 bars is where _score_trend starts scoring.
@pytest.mark.parametrize("run", [_overall, _sector])
@pytest.mark.parametrize("live, status", [
    pytest.param([], "live_unscored", id="live-request-failed"),
    pytest.param(_bars(RISING[:49], in_progress=1.0), "live_unscored", id="49-live-bars"),
    pytest.param(_bars(RISING[:50], in_progress=1.0), "ok", id="50-live-bars"),
])
def test_unscored_live_trend_is_not_compared(run, live, status):
    kw = {"flows_error": RuntimeError("flows down")} if run is _overall else {}
    off, _, _ = run(False, WIDE_HIGH, live=live, **kw)
    on, fetch, _ = run(True, WIDE_HIGH, live=live, **kw)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == status
    assert rec["live_bars"] == len(mr.completed_bars(list(live)))
    if status == "live_unscored":
        fetch.assert_not_called()                       # no second request after a failed one
        for key in ("bars_fetched", "same_last_bar", "trend_score", "score", "regime",
                    "differs", "regime_differs"):
            assert rec[key] is None
    else:
        fetch.assert_called_once()
        assert rec["score"] is not None and rec["differs"] is not None
    assert _without_shadow(on) == off


def test_windows_ending_on_different_bars_are_marked():
    # The two requests straddle the close: the shadow's last completed bar is
    # not the live one, so its trend reads a different current price.
    on, _, _ = _overall(True, WIDE_SAME[:-2])           # ends a session before the live window
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "ok" and rec["same_last_bar"] is False
    assert rec["last_bar_date"] != on["signals"]["last_bar_date"]
    on, _, _ = _overall(True, WIDE_SAME)
    assert on["signals"]["trend"]["shadow_200"]["same_last_bar"] is True


def test_record_stays_out_of_notes_reasoning_and_missing_inputs():
    for run in (_overall, _sector):
        on, _, _ = run(True, RuntimeError("boom"))
        text = repr(_without_shadow(on))
        assert "shadow" not in text and "fetch_failed" not in text
        assert on["signals"]["trend"]["shadow_200"]["status"] == "fetch_failed"


def test_one_shadow_log_line_and_the_live_bar_count_line_is_unchanged(caplog):
    def lines(flag):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=mr.log.name):
            _overall(flag, WIDE_HIGH)
            _sector(flag, RuntimeError("boom"))
        msgs = [r.getMessage() for r in caplog.records]
        return ([m for m in msgs if "shadow_200" in m], [m for m in msgs if "200DMA averaged" in m])

    off_shadow, off_live = lines(False)
    on_shadow, on_live = lines(True)
    assert off_shadow == [] and len(on_shadow) == 2
    assert on_shadow[0].startswith("SPY regime shadow_200: 226 bars")
    assert on_shadow[1] == "XLK regime shadow_200: not scored, fetch_failed (RuntimeError)"
    assert on_live == off_live == ["SPY trend: 200DMA averaged 173 bars",
                                   "XLK trend: 200DMA averaged 173 bars"]


# ---------------------------------------------------------------------------
# The inputs the shadow re-scores
# ---------------------------------------------------------------------------

def test_overall_breadth_proxy_is_rescored_only_when_live_used_it():
    # The ETF-flow call raising is the one path to the proxy in assess_regime.
    on, _, _ = _overall(True, WIDE_LOW, flows_error=RuntimeError("flows down"))
    off, _, _ = _overall(False, WIDE_LOW, flows_error=RuntimeError("flows down"))
    rec = on["signals"]["trend"]["shadow_200"]
    assert on["signals"]["breadth"] == off["signals"]["breadth"] == {
        "score": 0, "notes": "Breadth data unavailable"}            # 173 bars: no proxy
    assert rec["breadth_rescored"] is True and rec["breadth_score"] == 2
    assert rec["trend_score"] == on["signals"]["trend"]["score"] == 5
    assert rec["score"] == _implied(on, mr._WEIGHTS, mr._normalize, breadth=2)
    assert rec["differs"] is True

    # ETF flows answered: the live breadth is kept as it is.
    on, _, _ = _overall(True, WIDE_LOW)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["breadth_rescored"] is False
    assert rec["breadth_score"] == on["signals"]["breadth"]["score"]
    assert rec["differs"] is False


def test_overall_volatility_is_rescored_from_the_shadow_trend():
    # VIX above 40 scores -2 only while trend and breadth are both -5.
    kw = dict(live=LIVE_FALLING, macro={"VIXCLS": {"value": 45.0}}, watchlist=[WEAK_NAME] * 4)
    off, _, _ = _overall(False, WIDE_FALLING, **kw)
    on, _, _ = _overall(True, WIDE_FALLING, **kw)
    rec = on["signals"]["trend"]["shadow_200"]
    assert on["signals"]["trend"]["score"] == -5 and on["signals"]["breadth"]["score"] == -5
    assert on["signals"]["volatility"]["score"] == -2
    assert rec["trend_score"] == -2 and rec["breadth_score"] == -5 and rec["volatility_score"] == 5
    assert rec["score"] == _implied(on, mr._WEIGHTS, mr._normalize, trend=-2, volatility=5)
    assert rec["score"] == pytest.approx(on["score"] + 17)   # trend +3, volatility +7 at weight 2
    assert _without_shadow(on) == off


def test_sector_rescores_trend_breadth_proxy_and_price_action():
    def by_window(bars, ticker=""):
        return (2, "wide") if len(bars) >= 200 else (-2, "live")

    off, _, _ = _sector(False, WIDE_LOW, price_action=by_window)
    on, _, _ = _sector(True, WIDE_LOW, price_action=by_window)
    rec = on["signals"]["trend"]["shadow_200"]
    assert on["signals"]["price_action_volume"] == {"score": -2, "notes": "live"}
    assert on["signals"]["breadth"]["notes"] == "Breadth data unavailable"
    assert rec["price_action_score"] == 2 and rec["price_action_rescored"] is True
    assert rec["breadth_rescored"] is True and rec["breadth_score"] == 2
    assert rec["trend_score"] == 5 and rec["volatility_score"] == on["signals"]["volatility"]["score"]
    assert rec["score"] == _implied(on, mr._SECTOR_WEIGHTS, mr._normalize_sector,
                                    breadth=2, price_action_volume=2)
    assert rec["regime"] == mr._regime_from_score(rec["score"]) and rec["differs"] is True
    assert _without_shadow(on) == off


def test_sector_keeps_the_live_breadth_when_the_sample_had_bars():
    on, _, _ = _sector(True, WIDE_LOW, bulk={"A": WEAK_NAME, "B": WEAK_NAME})
    rec = on["signals"]["trend"]["shadow_200"]
    assert on["signals"]["breadth_sample"]["n_used"] == 2
    assert rec["breadth_rescored"] is False
    assert rec["breadth_score"] == on["signals"]["breadth"]["score"] == -5


def test_sector_price_action_uses_the_wide_bars():
    on, _, _ = _sector(True, WIDE_HIGH)
    rec = on["signals"]["trend"]["shadow_200"]
    wide = mr.completed_bars(list(WIDE_HIGH))
    assert len(wide) == 226
    assert rec["price_action_score"] == mr._score_price_action_volume(wide, "XLK")[0]
    assert rec["trend_score"] == mr._score_trend(wide)[0] == 0


def test_price_action_floor_is_a_full_200_bar_slope():
    # The slope compares ma200 now with ma200 TREND_LOOKBACK bars back; the
    # older mean averages 200 closes only from 199 + TREND_LOOKBACK bars on.
    from agents import price_action
    assert mr._SHADOW_PRICE_ACTION_MIN_BARS == 200 + price_action.TREND_LOOKBACK - 1 == 219


@pytest.mark.parametrize("completed, rescored", [(200, False), (218, False), (219, True)])
def test_sector_price_action_is_rescored_only_on_a_full_window(completed, rescored):
    def by_window(bars, ticker=""):
        return (2, "wide") if len(bars) >= 200 else (-2, "live")

    wide = _bars([100 + 0.1 * i for i in range(completed)], in_progress=1.0)
    on, _, _ = _sector(True, wide, price_action=by_window)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "ok" and rec["bars_fetched"] == completed
    assert rec["trend_score"] == mr._score_trend(mr.completed_bars(list(wide)))[0]
    assert rec["price_action_rescored"] is rescored
    assert rec["price_action_score"] == (2 if rescored else -2)   # else the live raw score
    assert rec["score"] == _implied(on, mr._SECTOR_WEIGHTS, mr._normalize_sector,
                                    trend=rec["trend_score"], breadth=rec["breadth_score"],
                                    price_action_volume=rec["price_action_score"])


def test_sector_volatility_rescore_keeps_the_live_arguments():
    # The sector rule calls _score_volatility(vix, trend, None): above VIX 40
    # it is +5 even with trend and breadth both -5, where the overall rule
    # (breadth passed) gives -2.
    wide_falling = _bars([150 - 0.1 * i for i in range(-53, 173)], in_progress=1.0)
    on, _, _ = _sector(True, wide_falling, live=LIVE_FALLING, macro={"VIXCLS": {"value": 45.0}},
                       bulk={"A": WEAK_NAME, "B": WEAK_NAME})
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "ok"
    assert rec["trend_score"] == -5 and rec["breadth_score"] == -5
    assert rec["breadth_rescored"] is False
    assert on["signals"]["volatility"]["score"] == 5
    assert rec["volatility_score"] == 5
    assert mr._score_volatility(45.0, -5, -5)[0] == -2     # what passing breadth would give


# ---------------------------------------------------------------------------
# (b) What the models read
# ---------------------------------------------------------------------------

def test_analyst_prompt_is_byte_identical_flag_on_and_off():
    off, _, _ = _overall(False, WIDE_HIGH)
    on, _, _ = _overall(True, WIDE_HIGH)
    assert on["signals"]["trend"]["shadow_200"]["differs"] is True      # the shadow disagrees here
    with patch.object(analyst, "_build_track_record_text", return_value="TRACK RECORD"):
        p_off = analyst._build_prompt(_CAND, "Current Price: $100.00\nBRIEF", "THESIS", off, "PHILO")
        p_on = analyst._build_prompt(_CAND, "Current Price: $100.00\nBRIEF", "THESIS", on, "PHILO")
    assert p_on.encode() == p_off.encode()
    assert "shadow" not in p_on and off["reasoning"][:500] in p_on


def test_panel_macro_signals_are_identical_flag_on_and_off():
    for run in (_overall, _sector):
        off, _, _ = run(False, WIDE_HIGH)
        on, _, _ = run(True, WIDE_HIGH)
        flat = roles.flatten_signals(on["signals"])
        assert flat == roles.flatten_signals(off["signals"])
        assert not any("shadow" in k for k in flat)
        assert flat["trend.ma200_bars_used"] == 173


# ---------------------------------------------------------------------------
# (c) A failed, slow or short fetch never raises and records a status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("run", [_overall, _sector])
@pytest.mark.parametrize("shadow, status, error", [
    (RuntimeError("boom"), "fetch_failed", "RuntimeError"),
    (requests.exceptions.ConnectionError("down"), "fetch_failed", "ConnectionError"),
    (requests.exceptions.ConnectTimeout("slow"), "timeout", "ConnectTimeout"),
    (None, "no_credentials", None),
    (_bars(RISING + RISING[:17]), "too_few_bars", None),
    ([{"date": "2026-01-02"}] * 210, "error", "KeyError"),
])
def test_failed_or_short_fetch_records_a_status(run, shadow, status, error):
    on, _, _ = run(True, shadow)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == status and rec.get("error") == error
    assert rec["live_bars"] == 173
    assert rec["bars_fetched"] == {"too_few_bars": 190, "error": 210}.get(status)
    for key in ("bars_used", "same_last_bar", "ma200", "trend_score", "breadth_score",
                "price_action_score", "price_action_rescored", "volatility_score", "score",
                "regime", "differs", "regime_differs"):
        assert rec[key] is None
    assert on["regime"] and isinstance(on["score"], float)


def test_slow_fetch_is_abandoned_at_the_timeout(monkeypatch):
    monkeypatch.setattr(mr, "REGIME_SHADOW_TIMEOUT_S", 0.05)
    release = threading.Event()

    def slow(ticker):
        release.wait(30)
        return WIDE_HIGH

    off, _, _ = _overall(False, slow)
    try:
        t0 = time.monotonic()
        on, fetch, _ = _overall(True, slow)
        waited = time.monotonic() - t0
    finally:
        release.set()
    assert waited < 10                                   # did not wait for the fetch
    assert on["signals"]["trend"]["shadow_200"]["status"] == "timeout"
    assert _without_shadow(on) == off

    # The late answer goes nowhere.
    for th in threading.enumerate():
        if th.name == "regime-shadow-SPY":
            th.join(5)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "timeout" and rec["score"] is None and rec["bars_fetched"] is None


def test_shadow_helper_swallows_its_own_bugs():
    with patch.object(mr, "_shadow_bars_bounded", side_effect=ValueError("bug")):
        rec = mr._regime_shadow_200("SPY", {"trend": 5}, 55.0, "Risk-On", live_bars=LIVE,
                                    vix=None, breadth_proxy=False)
    assert rec["status"] == "error" and rec["error"] == "ValueError" and rec["score"] is None


# ---------------------------------------------------------------------------
# The two requests
# ---------------------------------------------------------------------------

def _alpaca(bars):
    return [{"t": b["date"] + "T05:00:00Z", "o": b["open"], "h": b["high"], "l": b["low"],
             "c": b["close"], "v": b["volume"]} for b in bars]


def _real_requests(monkeypatch, flag, wide_response=None):
    """assess_regime() through the real _fetch_bars and _fetch_shadow_bars
    with requests.get faked. Returns (packet, the requests made)."""
    monkeypatch.setattr(mr, "_fetch_shadow_bars", _REAL_FETCH_SHADOW_BARS)
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params), "timeout": timeout,
                      "headers": sorted(headers)})
        wide = params["limit"] == 330
        if wide and wide_response is not None:
            return wide_response()
        resp = MagicMock()
        resp.json.return_value = {"bars": _alpaca(WIDE_HIGH if wide else LIVE)}
        return resp

    with patch("config.REGIME_SHADOW_200_ENABLED", flag), \
         patch.object(mr.requests, "get", fake_get), \
         patch("ingestion.macro.get_macro_snapshot", return_value={}), \
         patch.object(mr, "_fetch_watchlist_bars_for_breadth", return_value=[]), \
         patch("ingestion.options_flow.get_etf_flows", return_value={}), \
         patch("agents.markov_regime.get_markov_packet", return_value=_MARKOV_OFF), \
         patch("db.log_regime"):
        packet = mr.assess_regime()
    return packet, calls


def test_live_request_is_unchanged_and_the_shadow_request_is_separate(monkeypatch):
    assert _REAL_FETCH_SHADOW_BARS.__name__ == "_fetch_shadow_bars"
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    off, off_calls = _real_requests(monkeypatch, False)
    on, on_calls = _real_requests(monkeypatch, True)

    live_request = {"url": "https://data.alpaca.markets/v2/stocks/SPY/bars",
                    "params": {"timeframe": "1Day", "start": "2026-03-11", "end": TODAY,
                               "limit": 250, "feed": "iex"},
                    "timeout": 15, "headers": ["APCA-API-KEY-ID", "APCA-API-SECRET-KEY"]}
    assert off_calls == [live_request]                   # flag off: today's one request
    assert on_calls[0] == live_request and len(on_calls) == 2
    assert on_calls[1] == {**live_request, "timeout": 5.0,
                           "params": {**live_request["params"], "start": "2025-12-21", "limit": 330}}

    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "ok" and rec["bars_fetched"] == 226 and rec["calendar_days"] == 330
    assert on["signals"]["trend"]["ma200_bars_used"] == 173
    assert _without_shadow(on) == off


def test_real_fetch_failures_are_recorded(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    off, _ = _real_requests(monkeypatch, False)

    def http_500():
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        return resp

    def read_timeout():
        raise requests.exceptions.ReadTimeout("read timed out")

    def null_bars():
        resp = MagicMock()
        resp.json.return_value = {"bars": None}
        return resp

    for response, status, error in ((http_500, "fetch_failed", "HTTPError"),
                                    (read_timeout, "timeout", "ReadTimeout"),
                                    (null_bars, "too_few_bars", None)):
        on, calls = _real_requests(monkeypatch, True, wide_response=response)
        rec = on["signals"]["trend"]["shadow_200"]
        assert (rec["status"], rec.get("error")) == (status, error)
        assert len(calls) == 2 and _without_shadow(on) == off


def test_no_credentials_makes_no_request(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    on, calls = _real_requests(monkeypatch, True)
    assert calls == []
    # The live request has no credentials either, so its trend is unscored
    # and the shadow does not try.
    assert on["signals"]["trend"]["shadow_200"]["status"] == "live_unscored"
    with patch.object(mr.requests, "get") as get:
        assert _REAL_FETCH_SHADOW_BARS("SPY") is None
    get.assert_not_called()


# ---------------------------------------------------------------------------
# (d) The shadow's 200DMA is the last 200 completed bars
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("run", [_overall, _sector])
def test_shadow_averages_exactly_the_last_200_completed_bars(run):
    window = [100 + 0.1 * i for i in range(200)]
    # 26 outliers just outside the window, then today's unfinished bar at 1.0.
    wide = _bars([5000.0] * 26 + window, in_progress=1.0)
    on, _, _ = run(True, wide)
    rec = on["signals"]["trend"]["shadow_200"]
    assert rec["status"] == "ok" and rec["bars_fetched"] == 226 and rec["bars_used"] == 200
    assert rec["last_bar_date"] == LAST_SESSION.isoformat()             # today's bar dropped
    assert rec["ma200"] == round(sum(window) / 200, 4)
    assert rec["ma200"] != round(sum([5000.0] * 26 + window) / 226, 4)
    # Golden Cross on those 200 alone. One outlier in the average puts the
    # 200DMA above the price (trend 0); today's bar as the price gives -5.
    assert rec["trend_score"] == 5
    assert mr._score_trend(_bars([5000.0] + window[1:]))[0] == 0
    assert mr._score_trend(wide)[0] == -5


@pytest.mark.parametrize("completed, status", [(200, "ok"), (199, "too_few_bars")])
def test_200_completed_bars_is_the_floor(completed, status):
    # The unfinished bar does not count toward the 200.
    wide = _bars([100 + 0.1 * i for i in range(completed)], in_progress=1.0)
    on, _, _ = _overall(True, wide)
    rec = on["signals"]["trend"]["shadow_200"]
    assert len(wide) == completed + 1
    assert rec["status"] == status and rec["bars_fetched"] == completed
    assert rec["bars_used"] == (200 if status == "ok" else None)


# --- Nothing else picks the flag or the record up ---

def test_flag_is_not_part_of_the_treatment_config():
    assert "REGIME_SHADOW_200_ENABLED" not in analyst._treatment_config()


def test_brief_modules_read_no_regime_record():
    # The analyst brief is built without a regime packet: none of the modules
    # behind it runs an assessment or loads a stored one, so the record has no
    # way into the brief text. (The prompt, which does take a packet, is
    # rendered with the flag on and off above.)
    readers = ("agents.market_regime", "import market_regime", "assess_regime",
               "assess_sector_regime", "get_cached_sector_regime", "get_latest_regime",
               "get_latest_sector_regime", "signals_json", "shadow_200")
    for name in analyst._BRIEF_MODULES:
        src = analyst._module_source(name)
        assert src, name
        assert [r for r in readers if r in src] == [], name


# -- On-demand paths skip the shadow ------------------------------------------
# A sector cache miss runs inside live loops (risk_manager.check_profit_targets
# via get_trade_style_for_ticker, setup_gates); get_cached_regime's last
# resort runs on demand too. Neither may wait on the log-only request.

@pytest.mark.parametrize("run", [_overall, _sector])
def test_record_shadow_false_makes_no_request_and_leaves_the_packet_alone(run):
    on, _, _ = run(True, _bars([100.0] * 260))
    off, fetch, log_regime = run(True, _bars([100.0] * 260), record_shadow=False)
    fetch.assert_not_called()
    assert "shadow_200" not in off["signals"]["trend"]
    assert "shadow_200" in on["signals"]["trend"]
    drop = lambda p: {k: v for k, v in _without_shadow(p).items() if k != "timestamp"}
    assert drop(off) == drop(on)
    log_regime.assert_called_once()


def test_sector_cache_miss_recomputes_without_the_shadow():
    with patch.object(mr, "assess_sector_regime", return_value={"score": 50.0}) as assess:
        mr.get_cached_sector_regime("XLK", regime_type="gics_sector")
    assert assess.call_args.kwargs["record_shadow"] is False


def test_regime_cache_fallback_recomputes_without_the_shadow():
    with patch("db.get_latest_regime", return_value=None), \
         patch.object(mr, "assess_regime", return_value={"score": 50.0}) as assess:
        mr.get_cached_regime()
    assess.assert_called_once_with(record_shadow=False)


def test_scheduled_sector_sweep_still_records_the_shadow():
    calls = []
    with patch.object(mr, "_fetch_bars", return_value=[]), \
         patch("ingestion.sp500_universe.get_tickers_by_gics_sector", return_value=[]), \
         patch.object(mr, "assess_sector_regime",
                      side_effect=lambda *a, **k: calls.append(k) or {"score": 50.0}):
        mr.assess_market_sector_map()
    assert calls and all(k.get("record_shadow", True) for k in calls)
