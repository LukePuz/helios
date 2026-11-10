"""The options shadow also prices the expiry spanning the next print and the
~30-DTE expiry, for held names and prints within 14 days (audit 2026-11-07
19:07 #4). Log-only; no network."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import ingestion.options_free as m
from ingestion.options_free import extra_expiry_moves as REAL   # bound before the autouse stub

TODAY = date(2026, 11, 3)


def _leg(strikes, bid, ask):
    n = len(strikes)
    return pd.DataFrame({"strike": strikes, "bid": [bid] * n, "ask": [ask] * n,
                         "lastPrice": [bid] * n, "impliedVolatility": [0.5] * n})


class _Ticker:
    options = ("2026-11-07", "2026-11-14", "2026-11-21", "2026-12-04", "2026-12-19")

    def __init__(self, *a):
        self.calls = []

    def option_chain(self, exp):
        self.calls.append(exp)
        return SimpleNamespace(calls=_leg([100.0], 3.0, 3.2), puts=_leg([100.0], 2.8, 3.0),
                               underlying={"regularMarketPrice": 100.0})


def test_event_spanning_and_dte30_expiries():
    t = _Ticker()
    with patch.object(m.yf, "Ticker", return_value=t):
        out = REAL("NVDA", "2026-11-19", TODAY)
        on_day = REAL("NVDA", "2026-11-21", TODAY)
    assert out["event_spanning"]["expiry"] == "2026-11-21"
    assert on_day["event_spanning"]["expiry"] == "2026-12-04"   # an expiry on the print date doesn't span it
    assert out["dte30"]["expiry"] == "2026-12-04"            # 31 DTE is nearest 30
    assert out["earnings_date"] == "2026-11-19" and t.calls[:2] == ["2026-11-21", "2026-12-04"]


def test_no_earnings_date_prices_dte30_only_and_shares_a_chain():
    t = _Ticker()
    t.options = ("2026-11-07", "2026-12-03")
    with patch.object(m.yf, "Ticker", return_value=t):
        out = REAL("NVDA", None, TODAY)
        out2 = REAL("NVDA", "2026-11-20", TODAY)
    assert out["event_spanning"] is None and out["dte30"]["expiry"] == "2026-12-03"
    assert out2["event_spanning"]["expiry"] == out2["dte30"]["expiry"] == "2026-12-03"
    assert t.calls == ["2026-12-03", "2026-12-03"]           # one call per invocation


def test_failures_never_raise():
    with patch.object(m.yf, "Ticker", side_effect=RuntimeError("down")):
        out = REAL("NVDA", "2026-11-19", TODAY)
    assert out["error"] == "RuntimeError" and out["dte30"] is None
    t = _Ticker()
    t.options = ()
    with patch.object(m.yf, "Ticker", return_value=t):
        assert REAL("NVDA", None, TODAY)["error"] == "no_expirations"


def test_the_extras_are_bounded_by_the_cycle_budget():
    import threading
    import time
    import tasks
    gate = threading.Event()
    with patch("ingestion.options_free.extra_expiry_moves", side_effect=lambda *a: gate.wait(2) and {}):
        t0 = time.monotonic()
        out = tasks._bounded_extra_expiries("NVDA", None, TODAY, time.monotonic() + 0.1)
        assert time.monotonic() - t0 < 1.0 and out == {"fetched": False, "reason": "budget"}
        assert tasks._bounded_extra_expiries("NVDA", None, TODAY, time.monotonic() - 1) == \
            {"fetched": False, "reason": "budget"}
    gate.set()


def test_the_shadow_runs_after_the_dispatch_loop():
    import inspect
    import tasks
    src = inspect.getsource(tasks.poll_options_flow)
    assert src.index("_dispatch_event(event)") < src.index("_log_implied_move_shadow(implied)")
