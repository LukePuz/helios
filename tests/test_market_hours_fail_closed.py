"""execute_approved's market-hours gate for new entries fails closed.

It used to wrap the whole market-closed deferral in one try whose except
logged "Market-hours check failed, proceeding" and fell through to placing
the order. Now:
- an errored check (is_market_hours or its import) places no order:
  "failed: market-hours check errored" and one 🛑 alert;
- a closed market whose next open can't be worked out (next_market_open
  raises or returns a non-datetime) places no order and schedules no retry:
  "failed: market closed, market-open retry not scheduled" and one 🛑 alert;
- once a retry is scheduled, a failing deferral alert still returns the
  deferred outcome, so the order isn't placed now and again at the open.
The clean closed and open paths are unchanged; sells are not touched.
"""
import builtins
import datetime
from unittest.mock import MagicMock, call, patch

import execution

ETA = datetime.datetime(2026, 11, 24, 14, 30, tzinfo=datetime.timezone.utc)
_FRESH = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}


def _approval(action="ADD", shares_initial=10):
    return {"id": 7, "ticker": "NVDA", "action": action, "conviction": 7,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": shares_initial, "price_at_sizing": 100.0,
                          "stop_loss_price": 90.0}}


def _run(action="ADD", hours=None, nmo=None, send=None, retry=None, extra=()):
    """execute_approved with the gate's collaborators faked. hours/nmo are
    the patch kwargs for is_market_hours / next_market_open. The later BUY
    gates (fresh quote, no gap, no bracket) are passed, so a fall-through
    past the market-hours gate really reaches place_market_order."""
    send = send or MagicMock()
    retry = retry or MagicMock()
    ps = [patch("ibkr.get_ib"), patch("ibkr.get_client"),
          patch("risk_manager.RiskManager"),
          patch("ibkr.place_market_order",
                return_value={"order_id": 55, "fill_price": None, "filled_qty": 0.0}),
          patch("market_data.is_market_hours", **(hours or {"return_value": False})),
          patch("market_data.next_market_open", **(nmo or {"return_value": ETA})),
          patch("tasks.retry_deferred_entry", retry),
          patch("tasks.stale_quote_fallback_shadow"),
          patch("alerts._send", send),
          patch("ibkr.get_latest_price", return_value=100.0),
          patch("market_data.get_priced_quote", return_value=dict(_FRESH)),
          patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}),
          patch("config.USE_BRACKET_ORDERS", False),
          patch("db.log_trade"), patch("db.log_order"), patch("db.upsert_position_stop"),
          *extra]
    mocks = [p.start() for p in ps]
    try:
        mocks[2].return_value.is_halted.return_value = False
        result = execution.execute_approved(_approval(action))
    finally:
        for p in reversed(ps):
            p.stop()
    return result, {"order": mocks[3], "hours": mocks[4], "nmo": mocks[5],
                    "retry": retry, "send": send}


# --- (a) the check errors ---------------------------------------------------

def test_a_raising_market_hours_check_places_no_order():
    result, m = _run(hours={"side_effect": RuntimeError("calendar")})
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == "failed: market-hours check errored"
    m["retry"].apply_async.assert_not_called()
    m["nmo"].assert_not_called()


def test_a_raising_check_sends_one_not_sent_alert():
    result, m = _run(hours={"side_effect": RuntimeError("calendar")})
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert m["send"].call_count == 1
    text = m["send"].call_args[0][0]
    assert text.startswith("🛑 ORDER NOT SENT — market-hours check errored\n")
    assert "<b>NVDA</b> ADD" in text
    assert "The approval is not retried; a new verdict will route again." in text
    assert "$" not in text
    assert "%" not in text


def test_a_failing_alert_after_a_raising_check_still_fails_closed():
    send = MagicMock(side_effect=RuntimeError("telegram"))
    result, m = _run(hours={"side_effect": RuntimeError("calendar")}, send=send)
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == "failed: market-hours check errored"


def test_a_failed_import_of_the_check_places_no_order():
    real_import = builtins.__import__

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        # Only the gate's own import fails; the later quote/gap imports work,
        # so a fall-through would reach the order.
        if name == "market_data" and fromlist and "next_market_open" in fromlist:
            raise ImportError("market_data")
        return real_import(name, globals, locals, fromlist, level)
    result, m = _run(hours={"return_value": True},
                     extra=[patch("builtins.__import__", side_effect=_imp)])
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == "failed: market-hours check errored"


# --- (c) closed, the next open can't be found -----------------------------

def test_a_raising_next_open_places_no_order_and_schedules_nothing():
    result, m = _run(nmo={"side_effect": RuntimeError("calendar")})
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == \
        "failed: market closed, market-open retry not scheduled"
    m["retry"].apply_async.assert_not_called()
    assert m["send"].call_count == 1
    text = m["send"].call_args[0][0]
    assert text.startswith("🛑 ORDER NOT SENT — market closed, no retry scheduled\n")
    assert "The approval is not retried; a new verdict will route again." in text


def test_a_non_datetime_next_open_places_no_order():
    for bad in (None, "2026-11-24T14:30:00Z"):
        result, m = _run(nmo={"return_value": bad})
        m["order"].assert_not_called()   # first: a fall-through reaches the order
        assert result is False
        assert execution.last_outcome_reason() == \
            "failed: market closed, market-open retry not scheduled"
        m["retry"].apply_async.assert_not_called()


def test_a_failing_alert_after_a_raising_next_open_still_fails_closed():
    send = MagicMock(side_effect=RuntimeError("telegram"))
    result, m = _run(nmo={"side_effect": RuntimeError("calendar")}, send=send)
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == \
        "failed: market closed, market-open retry not scheduled"


def test_the_not_sent_reasons_are_not_deferred():
    for reason in ("failed: market-hours check errored",
                   "failed: market closed, market-open retry not scheduled"):
        assert execution.outcome_deferred(reason) is False


# --- (b) a retry is scheduled, then the alert fails ------------------------

def test_a_failing_deferral_alert_after_a_scheduled_retry_stays_deferred():
    send = MagicMock(side_effect=RuntimeError("telegram"))
    result, m = _run(send=send)
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == \
        "deferred: market closed, retry scheduled for the open"
    m["retry"].apply_async.assert_called_once_with(args=[7], eta=ETA, queue="ops")


def test_a_failing_eta_format_after_a_scheduled_retry_stays_deferred():
    class _Eta(datetime.datetime):
        def astimezone(self, tz=None):
            raise ValueError("tz")
    eta = _Eta(2026, 11, 24, 14, 30, tzinfo=datetime.timezone.utc)
    result, m = _run(nmo={"return_value": eta})
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == \
        "deferred: market closed, retry scheduled for the open"
    m["retry"].apply_async.assert_called_once_with(args=[7], eta=eta, queue="ops")
    m["send"].assert_not_called()


def test_an_unscheduled_retry_with_a_failing_alert_is_still_failed():
    retry = MagicMock()
    retry.apply_async.side_effect = RuntimeError("broker down")
    send = MagicMock(side_effect=RuntimeError("telegram"))
    result, m = _run(retry=retry, send=send)
    m["order"].assert_not_called()   # first: a fall-through reaches the order
    assert result is False
    assert execution.last_outcome_reason() == \
        "failed: market closed, market-open retry not scheduled"


# --- the clean paths, unchanged ---------------------------------------------

def test_the_clean_closed_path_defers_with_todays_alert():
    result, m = _run()
    assert result is False
    assert execution.last_outcome_reason() == \
        "deferred: market closed, retry scheduled for the open"
    m["order"].assert_not_called()
    m["retry"].apply_async.assert_called_once_with(args=[7], eta=ETA, queue="ops")
    assert m["nmo"].call_args_list == [call()]
    assert m["send"].call_count == 1
    text = m["send"].call_args[0][0]
    assert text == ("⏸️ ORDER DEFERRED — Market Closed\n"
                    "<b>NVDA</b> ADD was approved but the market is closed.\n"
                    "Not submitting now — a stale recommendation shouldn't fill blind. "
                    "Will retry automatically at the next open "
                    "(Tue 09:30 AM ET) with a fresh price/drift check.")


def test_the_clean_closed_path_with_no_schedule_is_failed():
    retry = MagicMock()
    retry.apply_async.side_effect = RuntimeError("broker down")
    result, m = _run(retry=retry)
    assert result is False
    assert execution.last_outcome_reason() == \
        "failed: market closed, market-open retry not scheduled"
    m["order"].assert_not_called()
    assert m["send"].call_count == 1
    assert m["send"].call_args[0][0].startswith("⏸️ ORDER DEFERRED — Market Closed\n")


def test_the_open_path_makes_todays_calls_and_places_the_order():
    """The gate reads is_market_hours() once with no arguments, never calls
    next_market_open, schedules nothing and goes on to the order."""
    hours = MagicMock(return_value=True)
    result, m = _run(hours={"new": hours})
    assert result is False   # the fake order reports no fill
    assert hours.call_args_list[0] == call()
    m["nmo"].assert_not_called()
    m["retry"].apply_async.assert_not_called()
    m["order"].assert_called_once()
    assert m["order"].call_args[0][:3] == ("NVDA", "BUY", 10)
    assert not any("ORDER NOT SENT" in str(c) or "ORDER DEFERRED" in str(c)
                   for c in m["send"].call_args_list)


# --- sells, untouched ----------------------------------------------------

def test_a_sell_never_reaches_the_entry_gate():
    """TRIM/EXIT take the elif branch: next_market_open is never called and
    no market-hours ORDER NOT SENT alert is sent, even when the check would
    raise."""
    with patch("tasks.manage_cancel_stop_orders_for"), patch("db.log_signal"):
        result, m = _run(action="EXIT", hours={"side_effect": RuntimeError("calendar")})
    m["nmo"].assert_not_called()
    m["retry"].apply_async.assert_not_called()
    assert not any("market-hours check errored" in str(c) for c in m["send"].call_args_list)
    assert execution.last_outcome_reason() != "failed: market-hours check errored"
