"""Each price-monitor action is recorded with how far into the session it ran
(audit 2026-11-07 07:07 #6). Log-only."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks
from market_data import minutes_since_open

_ET = ZoneInfo("America/New_York")


def test_minutes_since_open():
    assert minutes_since_open(datetime(2026, 11, 6, 9, 31, 30, tzinfo=_ET)) == 1.5
    assert minutes_since_open(datetime(2026, 11, 6, 15, 50, tzinfo=_ET)) == 380.0
    assert minutes_since_open(datetime(2026, 11, 6, 9, 0, tzinfo=_ET)) is None     # before the open
    assert minutes_since_open(datetime(2026, 11, 7, 10, 0, tzinfo=_ET)) is None    # Saturday


def test_one_row_per_action_with_the_timing():
    results = [{"ticker": "NVDA", "action": "PARTIAL_SELL"}, {"ticker": "AMD", "action": "BREAKEVEN_STOP"}]
    with patch("market_data.minutes_since_open", return_value=1.2), \
         patch("market_data.market_session", return_value="rth"), \
         patch("db.log_activity_direct") as act:
        tasks._record_monitor_actions("check_profit_targets", results)
        _join()
    assert [c.args[1] for c in act.call_args_list] == ["NVDA", "AMD"]
    extra = act.call_args_list[0].args[4]
    assert extra["task"] == "check_profit_targets" and extra["action"] == "PARTIAL_SELL"
    assert extra["minutes_since_open"] == 1.2 and extra["market_session"] == "rth"


def test_no_results_no_rows_and_failures_never_raise():
    with patch("db.log_activity_direct") as act:
        tasks._record_monitor_actions("check_trailing_stops", [])
        tasks._record_monitor_actions("check_trailing_stops", None)
    act.assert_not_called()
    with patch("db.log_activity_direct", side_effect=RuntimeError("db")):
        tasks._record_monitor_actions("check_trailing_stops", [{"ticker": "X"}])
        _join()


def test_the_caller_does_not_wait_on_the_db():
    """Professor 65a: the trailing-stop backstop runs right after this call."""
    import threading
    import time
    gate = threading.Event()
    with patch("db.log_activity_direct", side_effect=lambda *a, **k: gate.wait(5)):
        start = time.monotonic()
        tasks._record_monitor_actions("check_trailing_stops", [{"ticker": "X"}, {"ticker": "Y"}])
        assert time.monotonic() - start < 1.0
        gate.set()
        _join()


def _join():
    import threading
    for t in threading.enumerate():
        if t.name == "monitor-action":
            t.join(5)


def test_a_pre_open_breach_carries_its_distance_below_the_stop():
    """audit 2026-11-07 11:07 #2: a stop seen breached before the open."""
    results = [{"ticker": "NVDA", "action": "EXIT", "current_price": 190.0, "stop_price": 200.0,
                "rth": False}]
    with patch("market_data.minutes_since_open", return_value=None), \
         patch("market_data.market_session", return_value="pre"), \
         patch("db.log_activity_direct") as act:
        tasks._record_monitor_actions("check_trailing_stops", results)
        _join()
    extra = act.call_args.args[4]
    assert extra["market_session"] == "pre" and extra["rth"] is False
    assert extra["current_price"] == 190.0 and extra["stop_price"] == 200.0
    assert extra["pct_vs_stop"] == -5.0


def test_price_fields_are_none_when_missing_or_bad():
    assert tasks._monitor_price_fields({"action": "PARTIAL_SELL"}) == {
        "current_price": None, "stop_price": None, "rth": None, "pct_vs_stop": None}
    f = tasks._monitor_price_fields({"current_price": "nan", "stop_price": 0, "rth": "yes"})
    assert f == {"current_price": None, "stop_price": 0.0, "rth": None, "pct_vs_stop": None}
