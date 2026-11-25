"""Above-market guard on whole-position stops (risk_manager._stop_below_market),
here on the ADD merge (execution._whole_position_stop).

config.STOP_GUARD_MODE "shadow" (the default): the guard writes a row
(outcome would_*) and an info alert and returns the level unchanged, so the
stop is the same as with no guard. "enforce": a merged SELL stop at or above
the price is replaced by the highest of the new shares' own stop and the
resting stops below the price, with a row and a 🚨 alert.

Prices are scaled as in test_merge_after_lagging_add: L197 = 89.3 (lot 197's
own stop, kept on the record), L198 10.7% below it (the remaining share's
resting stop after lot 197's stop filled on 11-12), the 11-13 ADD filling
0.6% below L197 with its own stop L199 6.2% below L198."""
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

import config
import execution
import risk_manager

L197 = 89.3
L198 = round(L197 * (1 - 0.107), 2)
L199 = round(L198 * (1 - 0.062), 2)
FILL_1113 = round(L197 * (1 - 0.006), 2)


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(config, "STOP_GUARD_MODE", "enforce")


def _stp(price, qty=1.0, status="Submitted", order_type="STP"):
    return {"ticker": "NVDA", "action": "SELL", "order_type": order_type, "qty": qty,
            "stop_price": price, "status": status}


def _add(record_stop, resting, market, new_stop=L199, cache=2, fresh=(2.0, 95.0), held_before=1,
         resting_effect=None, price_is_fill=True, rth=True, guard_off=False):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=cache, avg_entry_price=95.0)
    read = (patch("tasks._resting_sell_stops", side_effect=resting_effect) if resting_effect
            else patch("tasks._resting_sell_stops", return_value=list(resting)))
    guard = (patch("risk_manager._stop_below_market", side_effect=lambda t, level, *a, **k: level)
             if guard_off else nullcontext())
    status: dict = {}
    with patch("db.get_position_stop", return_value={"stop_price": record_stop}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=fresh), \
         read as reads, guard, \
         patch("market_data.is_market_hours", return_value=rth), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("db.log_activity_queued") as queued, \
         patch("alerts._send") as send, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        out = execution._whole_position_stop("NVDA", new_stop, 1, status=status,
                                             market_price=market, action="ADD",
                                             held_before=held_before, price_is_fill=price_is_fill)
    rows = [c.args for c in queued.call_args_list if c.args[0] == "stop_above_market_guard"]
    _add.status = status
    _add.info = [c.args[0] for c in send.call_args_list if "stop guard (shadow)" in c.args[0]]
    alerts = [c.args[0] for c in send.call_args_list if "🚨" in c.args[0]]
    return out, status, rows, alerts, cancel, reads


# ── shadow (the default): what it would place, the stop unchanged ────────────

def _read_fails_second(ticker, *a, **k):
    _read_fails_second.n = getattr(_read_fails_second, "n", 0) + 1
    if _read_fails_second.n % 2 == 1:
        return [_stp(L198)]          # the floor's read
    raise TimeoutError("read")       # the guard's read


SHADOW_CASES = {
    # name: (_add kwargs, expected would_ outcome or None for no row)
    "kept_record_above_the_price": (dict(record_stop=L197, resting=[_stp(L198)], market=FILL_1113),
                                    "would_placed_below_market"),
    "no_resting_below": (dict(record_stop=L197, resting=[], market=FILL_1113),
                         "would_placed_below_market"),
    "nothing_below": (dict(record_stop=L197, resting=[], market=FILL_1113, new_stop=FILL_1113),
                      "would_no_level_below_market"),
    "stale_price": (dict(record_stop=L197, resting=[_stp(L197), _stp(L198)], market=FILL_1113,
                         price_is_fill=False), "would_resting_at_or_above_price"),
    "price_unread": (dict(record_stop=L197, resting=[_stp(L198)], market=None),
                     "would_price_unread"),
    "normal_merge": (dict(record_stop=L198, resting=[_stp(L198)], market=100.0), None),
}


@pytest.mark.parametrize("name", list(SHADOW_CASES))
def test_shadow_returns_the_level_with_no_guard(name):
    kw, outcome = SHADOW_CASES[name]
    _read_fails_second.n = 0
    off, _, _, _, off_cancel, _ = _add(**kw, guard_off=True)
    on, status, rows, alerts, cancel, _ = _add(**kw)
    assert on == off                                  # identical stop and quantity
    assert cancel.call_args_list == off_cancel.call_args_list
    assert alerts == [] and "record_stop" not in status
    if outcome is None:
        assert rows == [] and _add.info == []
        return
    assert len(rows) == 1
    extra = rows[0][4]
    assert extra["outcome"] == outcome and extra["mode"] == "shadow" and rows[0][3] == "neutral"
    assert len(_add.info) == (0 if outcome == "would_price_unread" else 1)


def test_shadow_row_says_what_it_would_place():
    (stop, _), _, rows, _, _, _ = _add(L197, [_stp(L198)], FILL_1113)
    assert stop == L197                               # placed as before
    extra = rows[0][4]
    assert extra["level_vs_price_pct"] == pytest.approx((L197 / FILL_1113 - 1) * 100, abs=1e-3)
    assert extra["placed_vs_price_pct"] == pytest.approx((L198 / FILL_1113 - 1) * 100, abs=1e-3)
    assert "would place" in _add.info[0] and "%" in _add.info[0]


def test_shadow_execute_path_places_the_unguarded_level():
    """execute_approved places what _whole_position_stop returns: in shadow
    that is the unguarded merged level, so manage_stop_place's args match a
    run with no guard at all."""
    on, _, _, _, _, _ = _add(L197, [_stp(L198)], FILL_1113)
    off, _, _, _, _, _ = _add(L197, [_stp(L198)], FILL_1113, guard_off=True)
    assert on == off == (L197, (95.0, 2.0))


# ── enforce ──────────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("enforce")
def test_kept_record_above_the_market_no_longer_places_above_it():
    """11-13 ADD #173 under a kept record: max(L199, L197) = L197 sits above
    the fill; the remaining share's resting L198 goes on instead."""
    (stop, whole), _, rows, alerts, cancel, _ = _add(L197, [_stp(L198)], FILL_1113)
    assert stop == L198 and stop < FILL_1113
    assert whole == (95.0, 2.0)
    cancel.assert_called_once_with("NVDA", stuck_out=[])
    assert len(rows) == 1
    _, ticker, _, sentiment, extra = rows[0]
    assert ticker == "NVDA" and sentiment == "negative" and extra["mode"] == "enforce"
    assert extra["outcome"] == "placed_below_market" and extra["path"] == "add_whole_position"
    assert extra["level_vs_price_pct"] == pytest.approx((L197 / FILL_1113 - 1) * 100, abs=1e-3)
    assert extra["placed_vs_price_pct"] == pytest.approx((L198 / FILL_1113 - 1) * 100, abs=1e-3)
    assert extra["n_resting_below"] == 1 and extra["resting_read"] == "ok"
    assert len(alerts) == 1 and "REFUSED" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_no_resting_stop_below_falls_back_to_the_new_shares_stop():
    (stop, _), _, rows, alerts, _, _ = _add(L197, [], FILL_1113)
    assert stop == L199
    assert rows[0][4]["outcome"] == "placed_below_market" and rows[0][4]["n_resting_below"] == 0
    assert len(alerts) == 1


@pytest.mark.usefixtures("enforce")
def test_resting_read_failure_falls_back_to_the_new_shares_stop():
    """The floor's read fails and the guard's one bounded re-read
    (risk_manager._reread_resting_stops) fails too, as it does when the
    failed read dropped the session: it works from the new shares' own stop.
    The plain read is not repeated (test_guard_reread has the re-read cases)."""
    with patch("risk_manager._reread_resting_stops",
               side_effect=ConnectionError("IBKR not connected")) as reread:
        (stop, _), _, rows, _, _, reads = _add(L197, None, FILL_1113,
                                               resting_effect=TimeoutError("read"))
    assert stop == L199 and rows[0][4]["resting_read"] == "reread_failed"
    assert rows[0][4]["reread_error"] == "ConnectionError"
    assert reads.call_count == 1
    reread.assert_called_once_with("NVDA")


@pytest.mark.usefixtures("enforce")
def test_nothing_below_the_price_keeps_the_level_and_alerts():
    (stop, _), _, rows, alerts, _, _ = _add(L197, [], FILL_1113, new_stop=FILL_1113)
    assert stop == L197
    assert rows[0][4]["outcome"] == "no_level_below_market"
    assert len(alerts) == 1 and "may sell at once" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_normal_merge_is_unchanged():
    (stop, whole), _, rows, alerts, cancel, reads = _add(L198, [_stp(L198)], 100.0, new_stop=L199)
    assert stop == L198 and whole == (95.0, 2.0)
    assert rows == [] and alerts == []
    assert reads.call_count == 1          # the floor's read only
    cancel.assert_called_once_with("NVDA", stuck_out=[])


@pytest.mark.usefixtures("enforce")
def test_normal_merge_with_a_higher_resting_stop_still_floors():
    (stop, _), _, rows, _, _, _ = _add(L198, [_stp(L197), _stp(L198)], 100.0, new_stop=L199)
    assert stop == L197 and rows == []


@pytest.mark.usefixtures("enforce")
def test_price_unread_places_as_before_and_records_it():
    (stop, _), _, rows, alerts, _, _ = _add(L197, [_stp(L198)], None)
    assert stop == L197                   # today's behaviour
    assert rows[0][4]["outcome"] == "price_unread" and rows[0][3] == "neutral"
    assert alerts == []


def test_ambiguous_path_is_unchanged():
    """Reads disagree: only the filled shares are stopped at their own level;
    no guard read, no row."""
    out, status, rows, alerts, cancel, reads = _add(L197, [_stp(L198)], FILL_1113,
                                                    cache=2, fresh=(1.0, 95.0), held_before=None)
    assert out == (L199, None) and status.get("prior_unreadable") is True
    assert rows == [] and alerts == []
    reads.assert_not_called()
    cancel.assert_not_called()


def test_guard_crash_places_the_level(enforce):
    with patch("risk_manager._plausible_price", side_effect=ValueError("boom")):
        assert risk_manager._stop_below_market("NVDA", 90.0, 80.0, 88.0, "x", "p") == 90.0


@pytest.mark.parametrize("market", [None, float("nan"), 0, -1, 2e9])
def test_implausible_price_counts_as_unread(enforce, market):
    with patch("db.log_activity_queued") as queued, patch("alerts._send") as send:
        assert risk_manager._stop_below_market("NVDA", 90.0, 80.0, market, "x", "p") == 90.0
    assert queued.call_args.args[4]["outcome"] == "price_unread"
    send.assert_not_called()


def test_below_the_price_returns_at_once():
    with patch("tasks._resting_sell_stops") as reads, patch("db.log_activity_queued") as queued:
        assert risk_manager._stop_below_market("NVDA", 87.0, 80.0, 88.0, "x", "p") == 87.0
    reads.assert_not_called()
    queued.assert_not_called()


# ── enforce: the old-price inference (non-fill price, STP/TRAIL, in RTH) ─────

@pytest.mark.usefixtures("enforce")
def test_quote_price_under_a_working_l197_keeps_l197():
    """A non-fill price under a still-working L197 in regular hours: the
    price is old, so the kept L197 is not lowered."""
    (stop, _), _, rows, alerts, cancel, _ = _add(L197, [_stp(L197), _stp(L198)], FILL_1113,
                                                 price_is_fill=False)
    assert stop == L197
    cancel.assert_called_once_with("NVDA", stuck_out=[])       # the merge itself goes ahead as before
    assert rows[0][4]["outcome"] == "resting_at_or_above_price"
    assert rows[0][4]["n_resting_at_or_above"] == 1 and rows[0][4]["stale_check"] == "made"
    assert len(alerts) == 1 and "PRICE LOOKS OLD" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_fill_price_is_never_read_as_old():
    """An ADD filling under a working L197 is a real print: no old-price
    inference, so the merged level is lowered (SHOULD-FIX 1)."""
    (stop, _), _, rows, alerts, _, _ = _add(L197, [_stp(L197), _stp(L198)], FILL_1113,
                                            price_is_fill=True)
    assert stop == L198 and stop < FILL_1113
    assert rows[0][4]["outcome"] == "placed_below_market"
    assert rows[0][4]["stale_check"] == "skipped_fill_price"


@pytest.mark.usefixtures("enforce")
def test_outside_regular_hours_a_resting_stop_is_no_evidence():
    (stop, _), _, rows, _, _, _ = _add(L197, [_stp(L197), _stp(L198)], FILL_1113,
                                       price_is_fill=False, rth=False)
    assert stop == L198 and rows[0][4]["stale_check"] == "skipped_outside_rth"


@pytest.mark.usefixtures("enforce")
@pytest.mark.parametrize("resting_l197", [_stp(L197, status="PendingCancel"),
                                          _stp(L197, order_type="STP LMT")])
def test_a_cancelling_or_stop_limit_order_is_not_evidence(resting_l197):
    (stop, _), _, rows, _, _, _ = _add(L197, [resting_l197, _stp(L198)], FILL_1113,
                                       price_is_fill=False)
    assert stop == L198 and rows[0][4]["outcome"] == "placed_below_market"


# ── enforce: the record follows a lowered level only on a fill price ─────────

@pytest.mark.usefixtures("enforce")
def test_fill_price_lets_the_record_follow():
    (stop, _), _, rows, alerts, _, _ = _add(L197, [_stp(L198)], FILL_1113, price_is_fill=True)
    assert stop == L198 and "record_stop" not in _add.status
    assert rows[0][4]["record"] == "moved_to_placed" and "is moved to" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_quote_price_keeps_the_record_higher():
    (stop, _), _, rows, alerts, _, _ = _add(L197, [_stp(L198)], FILL_1113, price_is_fill=False)
    assert stop == L198 and _add.status["record_stop"] == L197
    assert rows[0][4]["record"] == "kept_higher" and "still shows" in alerts[0]


@pytest.mark.usefixtures("enforce")
def test_unlowered_merge_sets_no_record_stop():
    _add(L198, [_stp(L198)], 100.0, price_is_fill=False)
    assert "record_stop" not in _add.status


# ── ibkr.get_fresh_minute_price ──────────────────────────────────────────────

def _bars(end, close=88.0):
    return [MagicMock(date=end, close=close)]


def test_fresh_minute_price_reads_a_bounded_minute_bar():
    import datetime as dt
    import ibkr
    now = dt.datetime.now(dt.timezone.utc)
    ib = MagicMock()
    with patch("ibkr.get_ib", return_value=ib), patch("ibkr._stock", return_value=MagicMock()):
        ib.reqHistoricalData.return_value = _bars(now - dt.timedelta(minutes=2))
        assert ibkr.get_fresh_minute_price("NVDA", max_age_s=300, timeout_s=5) == 88.0
        kw = ib.reqHistoricalData.call_args.kwargs
        assert kw["barSizeSetting"] == "1 min" and kw["timeout"] == 5 and kw["formatDate"] == 2
        ib.reqHistoricalData.return_value = _bars(now - dt.timedelta(minutes=30))
        assert ibkr.get_fresh_minute_price("NVDA", max_age_s=300) is None          # old bar
        ib.reqHistoricalData.return_value = _bars(now.date())
        assert ibkr.get_fresh_minute_price("NVDA") is None                         # a daily bar
        ib.reqHistoricalData.return_value = []
        assert ibkr.get_fresh_minute_price("NVDA") is None                         # timeout / none
        ib.reqHistoricalData.return_value = _bars(now, close=float("nan"))
        assert ibkr.get_fresh_minute_price("NVDA") is None
        ib.reqHistoricalData.side_effect = RuntimeError("down")
        assert ibkr.get_fresh_minute_price("NVDA") is None
