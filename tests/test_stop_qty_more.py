"""Position-count sizing (this process's ib.positions(), via
ibkr.get_broker_stock_qty, or ibkr.cached_stock_qty for the profit-target
remainder), extended:
- the stop after a profit-target partial sale (risk_manager._pt_stop_qty,
  config PT_STOP_QTY_FROM_BROKER): the portfolio cache read seconds after
  the sale can still show the shares just sold, and a stop for them is
  refused (SellExceedsPosition). The count is read after one more event-loop
  run (settle);
- a BUY fill reconcile books late (tasks._late_fill_expect, config
  LATE_FILL_EXPECT_MODE, "enforce" here unless a test says shadow): a fresh deduplicated reqPositions read
  first, then ib.positions(); the whole-position stop waits until the cache
  shows that count (expect_at_least), as the manual ADD does.
Unusable counts keep the old behaviour, with an alert or a row."""
from unittest.mock import DEFAULT, MagicMock, call, patch

import pytest

import config
import risk_manager
import tasks


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "PT_STOP_QTY_FROM_BROKER", True)
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "enforce")


# ── profit-target remainder ──────────────────────────────────────────────────

def _pt_sale(broker, cache=2, sold=1, held=2, cache_effect=None, place_effect=None):
    """A profit-target sale of `sold` of `held` shares filling at 120; the
    cache read after it shows `cache` (or raises cache_effect)."""
    client = MagicMock()
    if cache_effect is not None:
        client.get_open_position.side_effect = cache_effect
    else:
        client.get_open_position.return_value = MagicMock(qty=cache)
    bkw = {"side_effect": broker} if isinstance(broker, Exception) else {"return_value": broker}
    reads = [[], []]
    events = []
    _pt_sale.events = events
    with patch("market_data.is_market_hours", return_value=True), \
         patch("risk_manager._pt_hold_still_blocked", return_value=False), \
         patch("db.get_position_stop", return_value={"stop_price": 100.0, "entry_price": 100.0}), \
         patch("tasks._resting_sell_stops", side_effect=lambda *a, **k: reads.pop(0) if reads else []), \
         patch("tasks._reconcile_alert_due", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("stop_lock.assert_held"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": sold, "fill_price": 120.0,
                                                        "order_id": 5}), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.cached_stock_qty", **bkw) as bread, \
         patch("db.log_trade"), patch("db.log_order", return_value=0), \
         patch("db.set_stop_broker_order_id"), \
         patch("db.log_activity_queued") as queued, \
         patch("tasks.manage_stop_place", side_effect=place_effect or (lambda *a, **k: (
                   events.append("place"), {"order_id": 8})[1])) as place, \
         patch("alerts._send", side_effect=lambda t, *a, **k: events.append(t)) as send:
        rm = object.__new__(risk_manager.RiskManager)
        rm._client = client
        with patch.object(risk_manager.RiskManager, "_changed_while_waiting", return_value=None):
            rm._partial_sell_and_resize_stop("NVDA", held, sold, 100.0, decision_price=120.0)
    _pt_sale.sends = send.call_args_list
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "pt_stop_qty"]
    alerts = [c.args[0] for c in send.call_args_list if "PT STOP" in c.args[0]]
    return place, rows, alerts, bread


def test_stale_cache_after_the_sale_sizes_from_the_broker(on):
    """The cache still shows both shares after selling one; the broker
    shows one: the stop is for one, not refused for two."""
    place, rows, alerts, bread = _pt_sale(1.0, cache=2)
    bread.assert_called_once_with("NVDA")      # cached_stock_qty runs the loop once first
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["source"] == "broker" and rows[0]["read_qty"] == 2.0
    assert rows[0]["cache_qty"] == 2.0 and rows[0]["expected_qty"] == 1.0 and rows[0]["used_qty"] == 1.0
    assert alerts == []


@pytest.mark.switches(PT_STOP_QTY_FROM_BROKER=False)
def test_switch_off_sizes_from_the_cache():
    place, rows, alerts, bread = _pt_sale(1.0, cache=2)
    assert place.call_args_list == [call("NVDA", 2.0, 100.0, market_price=120.0)]   # today
    bread.assert_not_called()
    assert rows == [] and alerts == []


@pytest.mark.parametrize("broker,read", [(TimeoutError("ib"), "error: TimeoutError"), (0.0, "zero"),
                                         (-1.0, "short"), (float("nan"), "not_finite")])
def test_unusable_count_falls_back_to_the_cache_capped_and_alerts(on, broker, read):
    """The cache still shows both shares after selling one: the fallback is
    capped at held minus fill (1), not today's 2, which would be refused."""
    place, rows, alerts, _ = _pt_sale(broker, cache=2)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["source"] == "position_reads" and rows[0]["broker_read"] == read
    assert rows[0]["read_qty"] == 2.0 and rows[0]["used_qty"] == 1.0 and rows[0]["capped"] is True
    assert len(alerts) == 1 and "UNCONFIRMED" in alerts[0] and "capped" in alerts[0]


def test_unusable_count_with_the_cache_within_held_minus_fill_is_uncapped(on):
    place, rows, alerts, _ = _pt_sale(TimeoutError("ib"), cache=1)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["capped"] is False and len(alerts) == 1


@pytest.mark.parametrize("expected", [0.0, -1.0, float("nan"), None])
def test_no_positive_expected_keeps_todays_fallback(on, expected):
    with patch("ibkr.cached_stock_qty", side_effect=TimeoutError("ib")), \
         patch("db.log_activity_queued") as queued, patch("alerts._send"):
        assert risk_manager._pt_stop_qty("NVDA", 2.0, 2.0, expected, 1.0)[0] == 2.0
    assert queued.call_args.args[4]["capped"] is False


def test_failed_cache_read_falls_back_to_expected_then_the_broker(on):
    place, rows, _, _ = _pt_sale(1.0, cache_effect=RuntimeError("cache"))
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["cache_qty"] is None and rows[0]["read_qty"] == 1.0


def test_agreeing_counts_change_nothing(on):
    place, rows, alerts, _ = _pt_sale(1.0, cache=1)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["source"] == "broker" and alerts == []


def test_count_below_the_expected_remainder_is_used_and_alerted(on):
    # held 3, sold 1: 2 expected; the count shows 1
    place, rows, alerts, _ = _pt_sale(1.0, cache=3, held=3, sold=1)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["below_expected"] is True
    assert len(alerts) == 1 and "MAY NOT COVER" in alerts[0]


def test_stale_high_count_is_capped_at_held_minus_fill(on):
    """positions() still shows both shares (the sale not yet applied): the
    stop is for held minus fill, never above it."""
    place, rows, alerts, _ = _pt_sale(2.0, cache=2)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["above_expected"] is True and rows[0]["capped"] is True
    assert rows[0]["broker_qty"] == 2.0 and rows[0]["used_qty"] == 1.0
    assert rows[0]["capped_ambiguous"] is False           # stale by exactly the sale
    assert alerts == []


def test_pt_count_uses_the_cached_read_with_the_same_settle(on):
    """Review S2: the remainder's count is cached_stock_qty: one event-loop
    run (ib.sleep(0)) as settle=True did, no request and no get_ib() (its
    heartbeat or a reconnect sat between the sale and the place)."""
    import ibkr
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.positions.return_value = [MagicMock(contract=MagicMock(symbol="NVDA", secType="STK"),
                                           position=1.0)]
    with patch.object(ibkr, "_ib", ib), patch("ibkr.get_ib", side_effect=AssertionError("get_ib")), \
         patch("db.log_activity_queued"):
        assert risk_manager._pt_stop_qty("NVDA", 2.0, 2.0, 1.0, 1.0)[0] == 1.0
    ib.sleep.assert_called_once_with(0)
    with patch.object(ibkr, "_ib", None), patch("ibkr.get_ib", side_effect=AssertionError("get_ib")), \
         patch("db.log_activity_queued") as queued:
        qty, text, _ = risk_manager._pt_stop_qty("NVDA", 2.0, 2.0, 1.0, 1.0)
    assert qty == 1.0 and "UNCONFIRMED" in text            # not connected: the capped fallback
    assert queued.call_args.args[4]["broker_read"] == "error: ConnectionError"


def test_settle_runs_the_event_loop_before_reading():
    import ibkr
    ib = MagicMock()
    ib.positions.return_value = [MagicMock(contract=MagicMock(symbol="NVDA", secType="STK"),
                                           position=1.0)]
    with patch("ibkr.get_ib", return_value=ib):
        assert ibkr.get_broker_stock_qty("NVDA", settle=True) == 1.0
        ib.sleep.assert_called_once_with(0)
        ib.sleep.reset_mock()
        assert ibkr.get_broker_stock_qty("NVDA") == 1.0
        ib.sleep.assert_not_called()


def test_pt_helper_never_raises(on):
    with patch("ibkr.cached_stock_qty", return_value=1.0), \
         patch("db.log_activity_queued", side_effect=RuntimeError("q")):
        assert risk_manager._pt_stop_qty("NVDA", 2.0, 2.0, 1.0, 1.0) == (1.0, None, None)
    with patch("ibkr.cached_stock_qty", side_effect=RuntimeError("ib")), \
         patch("db.log_activity_queued"), patch("alerts._send") as send:
        qty, text, _ = risk_manager._pt_stop_qty("NVDA", 2.0, 2.0, 1.0, None)
    assert qty == 1.0 and "UNCONFIRMED" in text      # fallback capped at held minus fill
    send.assert_not_called()                          # the caller sends it after the place


def test_alert_failure_after_the_place_does_not_raise(on):
    with patch("alerts._send", side_effect=RuntimeError("tg")):
        place, rows, _, _ = _pt_sale(TimeoutError("ib"), cache=2)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]


@pytest.mark.parametrize("broker,cache,held,label", [
    (TimeoutError("ib"), 2, 2, "UNCONFIRMED"),        # fallback
    (1.0, 3, 3, "MAY NOT COVER"),                     # below held − fill
    (3.0, 2, 2, "PT STOP CAPPED")])                   # capped, cause unclear
def test_every_pt_alert_is_sent_after_the_stop_is_placed(on, broker, cache, held, label):
    """The stop was cancelled before the sale: a Telegram send between the
    cancel and the place would widen the unprotected window."""
    place, _, alerts, _ = _pt_sale(broker, cache=cache, held=held)
    assert len(alerts) == 1 and label in alerts[0]
    assert _pt_sale.events == ["place", alerts[0]]


@pytest.mark.switches(PT_STOP_QTY_FROM_BROKER=False)
def test_switch_off_returns_the_read_and_no_alert():
    with patch("ibkr.cached_stock_qty") as bread, patch("db.log_activity_queued") as queued:
        assert risk_manager._pt_stop_qty("NVDA", 2.0, 2.0, 1.0, 1.0) == (2.0, None, None)
    bread.assert_not_called(); queued.assert_not_called()


def test_cap_beyond_the_sale_flags_ambiguous_and_sends_info(on):
    """positions() shows 3 after selling 1 of 2 held: 2 above held − fill,
    not the 1 sold, so the cause is unclear (e.g. a held count missing an
    untracked ADD). Capped anyway, flagged, with an info alert."""
    place, rows, alerts, _ = _pt_sale(3.0, cache=2)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["capped"] is True and rows[0]["capped_ambiguous"] is True
    assert rows[0]["fill_qty"] == 1.0
    assert len(alerts) == 1 and alerts[0].startswith("ℹ️ PT STOP CAPPED")
    assert "capped at 1 shares" in alerts[0] and "broker count 3" in alerts[0]


def test_fallback_cap_beyond_the_sale_is_flagged_ambiguous(on):
    place, rows, alerts, _ = _pt_sale(TimeoutError("ib"), cache=3)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["capped_ambiguous"] is True and len(alerts) == 1 and "UNCONFIRMED" in alerts[0]


# ── late BUY fill: expect_at_least ───────────────────────────────────────────

def _expect(broker, fresh=(144.0, 181.0)):
    bkw = {"side_effect": broker} if isinstance(broker, Exception) else {"return_value": broker}
    fkw = {"side_effect": fresh} if isinstance(fresh, Exception) else {"return_value": fresh}
    order = []
    with patch("ibkr.get_broker_stock_qty", side_effect=lambda *a, **k: (
                   order.append("positions"), _value(broker))[1]) as bread, \
         patch("ibkr.get_fresh_stock_position", side_effect=lambda *a, **k: (
                   order.append("reqPositions"), _value(fresh))[1]) as fread, \
         patch("db.log_activity_queued") as queued:
        out = tasks._late_fill_expect("CRWD", 44.0, "reconcile_late_buy")
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "late_fill_expect"]
    _expect.order = order
    return out, rows, bread


def _value(v):
    if isinstance(v, Exception):
        raise v
    return v


def test_expect_reads_fresh_first_then_positions(on):
    out, rows, _ = _expect(144.0, fresh=(144.0, 181.0))
    assert _expect.order == ["reqPositions", "positions"]
    assert out == 144.0 and rows[0]["source"] == "broker_after_fresh" and rows[0]["fill_qty"] == 44.0


@pytest.mark.parametrize("broker", [TimeoutError("ib"), 0.0, float("nan")])
def test_unusable_positions_count_falls_back_to_the_fresh_read(on, broker):
    out, rows, _ = _expect(broker, fresh=(144.0, 181.0))
    assert out == 144.0 and rows[0]["source"] == "fresh_read"


@pytest.mark.parametrize("fresh", [TimeoutError("ib"), (0.0, None)])
def test_no_fresh_read_means_no_expectation_and_no_positions_read(on, fresh):
    out, rows, bread = _expect(144.0, fresh=fresh)
    assert out is None and rows[0]["source"] == "none"
    bread.assert_not_called()                     # the cache alone is not trusted


@pytest.mark.switches(LATE_FILL_EXPECT_MODE="off")
def test_switch_off_reads_nothing():
    with patch("ibkr.get_broker_stock_qty") as bread, patch("db.log_activity_queued") as queued:
        assert tasks._late_fill_expect("CRWD", 44.0, "x") is None
    bread.assert_not_called()
    queued.assert_not_called()


def _order():
    return {"id": 44, "ticker": "CRWD", "side": "BUY", "qty": 144.0, "broker_order_id": 764,
            "decision_price": 180.1, "approval_id": 59}


def _approval():
    return {"id": 59, "ticker": "CRWD", "status": "FAILED", "thesis_bucket": "Compounder",
            "conviction": 7, "stop_loss_price": 153.61,
            "risk_json": {"stop_method": "2.5x ATR", "trailing_stop_distance_pct": 0.175}}


def _fake_ib(qty, connected=True, events=None, symbol="CRWD"):
    """This process's IB object as _late_fill_cache_qty reads it (ibkr._ib)."""
    ib = MagicMock()
    ib.isConnected.return_value = connected
    item = MagicMock(position=qty)
    item.contract.secType, item.contract.symbol = "STK", symbol

    def portfolio():
        if events is not None:
            events.append("snapshot")
        return [item] if qty else []
    ib.portfolio.side_effect = portfolio
    return ib


def _late_buy(cache, broker, order=None, approval=None):
    events = _late_buy.events = []

    def ev(name, value):
        def f(*a, **k):
            events.append(name)
            return value
        return f
    client = MagicMock()
    client.get_all_positions.side_effect = ev("cache", [MagicMock(symbol="CRWD", qty=cache)])
    with patch("db.get_unconfirmed_orders", return_value=[order or _order()]), \
         patch("ibkr._ib", _fake_ib(cache, events=events)), \
         patch("ibkr.get_fill_for_order", return_value={"filled_qty": 144.0, "fill_price": 181.3}), \
         patch.multiple("db", log_trade=DEFAULT, update_order_fill=DEFAULT,
                        upsert_position_stop=DEFAULT, set_stop_broker_order_id=DEFAULT,
                        update_approval_status=DEFAULT), \
         patch("db.get_approval_by_id", return_value=approval if order else _approval()), \
         patch("tasks.manage_stop_place", side_effect=ev("place", {"order_id": 900})) as place, \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_broker_stock_qty", side_effect=ev("positions", broker)), \
         patch("ibkr.get_fresh_stock_position", side_effect=ev("reqPositions", (broker, 181.3))), \
         patch("ibkr.get_latest_price", return_value=None), \
         patch("ibkr.get_fresh_minute_price", return_value=None), \
         patch("db.get_position_stop", side_effect=[None, {"stop_price": 153.61}]), \
         patch("tasks._retry_pending_reprotects"), \
         patch("tasks._queue_reprotect") as queue, \
         patch("db.log_activity_queued") as queued, \
         patch("alerts._send") as send, \
         patch.object(tasks.recheck_stop_coverage, "apply_async") as recheck:
        # The re-protect's fresh read is empty here (the fake session shows
        # no positions), so a place is sized from the cache and queues the
        # coverage re-check (_late_buy.recheck).
        tasks.reconcile_unconfirmed_orders.run()
    _late_buy.recheck = recheck
    rows = [c.args[4] for c in queued.call_args_list if c.args[0] == "late_fill_expect"]
    _late_buy.sent = [c.args[0] for c in send.call_args_list]
    return place, queue, rows


def test_late_buy_waits_while_the_cache_lags_the_broker(on):
    """The cache still shows the shares held before the late fill; the
    broker shows the fill: no stop sized to the cache, the retry is queued."""
    place, queue, rows = _late_buy(cache=100, broker=144.0)
    place.assert_not_called()
    queue.assert_called_once_with("CRWD", {"intent": "BUY", "expect_at_least": 144.0}, None)
    assert rows[0]["expect_at_least"] == 144.0
    late = [t for t in _late_buy.sent if "LATE FILL" in t]
    assert len(late) == 1 and "⏳ late BUY fill: stop deferred until the position cache shows 144" in late[0]
    assert "Could not place" not in late[0]


def test_late_buy_real_failure_keeps_the_check_tws_wording(on):
    with patch("tasks._reprotect_after_late_sell", return_value=False):
        _late_buy(cache=144, broker=144.0)
    late = [t for t in _late_buy.sent if "LATE FILL" in t]
    assert len(late) == 1 and "Could not place a protective stop" in late[0] and "deferred" not in late[0]


def _manual_order():
    return {"id": 45, "ticker": "CRWD", "side": "BUY", "qty": 144.0, "broker_order_id": 765,
            "decision_price": 180.1, "approval_id": None, "intent": "MANUAL_OPEN"}


def test_manual_late_buy_waits_while_the_cache_lags_the_broker(on):
    place, queue, rows = _late_buy(cache=100, broker=144.0, order=_manual_order())
    place.assert_not_called()
    queue.assert_called_once_with("CRWD", {"intent": "BUY", "expect_at_least": 144.0}, None)
    assert rows[0]["path"] == "reconcile_late_manual_buy" and rows[0]["expect_at_least"] == 144.0
    assert any("stop deferred until the position cache shows 144" in t for t in _late_buy.sent)


def test_manual_late_buy_with_the_cache_caught_up_places(on):
    place, queue, rows = _late_buy(cache=144, broker=144.0, order=_manual_order())
    assert len(place.call_args_list) == 1 and place.call_args.args[:2] == ("CRWD", 144.0)
    queue.assert_not_called()
    assert rows[0]["path"] == "reconcile_late_manual_buy"
    _late_buy.recheck.assert_called_once()              # sized from the cache (I6)


# ── queued BUY retry keeps expect_at_least for 2 tries ──────────────────────

def test_queue_entry_carries_expect_at_least():
    import json
    state = {}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        tasks._queue_reprotect("CRWD", {"intent": "BUY", "expect_at_least": 144.0}, None)
    assert json.loads(state["reprotect_pending"])["CRWD"]["expect_at_least"] == 144.0


def test_retry_keeps_the_expectation_two_tries_then_drops_it_and_still_escalates(on):
    """Cache stuck below the fill: tries 1-2 wait for 144, tries 3-4 run
    without it (today), and a failed 4th still gives up with the alert."""
    import json
    state = {"reprotect_pending": json.dumps({"CRWD": {"intent": "BUY", "stop_price": None,
                                                      "fill_price": None,
                                                      "expect_at_least": 144.0}})}
    seen = []
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reprotect_after_late_sell",
               side_effect=lambda t, *a, **k: (seen.append(k.get("expect_at_least")), False)[1]), \
         patch("tasks.open_manual_action") as manual, patch("alerts._send") as send:
        for _ in range(4):
            tasks._retry_pending_reprotects()
    assert seen == [144.0, 144.0, None, None]
    assert json.loads(state["reprotect_pending"]) == {}
    assert any("STOP NOT RESTORED" in c.args[0] for c in send.call_args_list)
    manual.assert_called_once()


def test_late_buy_with_the_cache_caught_up_places_as_before(on):
    place, queue, _ = _late_buy(cache=144, broker=144.0)
    assert place.call_args_list == [call("CRWD", 144.0, 153.61)]
    queue.assert_not_called()
    _late_buy.recheck.assert_called_once()              # sized from the cache (I6)


@pytest.mark.switches(LATE_FILL_EXPECT_MODE="off")
def test_late_buy_switch_off_places_to_the_cache():
    place, _, rows = _late_buy(cache=100, broker=144.0)
    assert place.call_args_list == [call("CRWD", 100.0, 153.61)]        # today
    assert rows == []
    _late_buy.recheck.assert_called_once()              # sized from the cache (I6)


# ── LATE_FILL_EXPECT_MODE = "shadow": reads and a row, today's stop ─────────

def _late_buy_calls(monkeypatch, mode, cache, order=None):
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", mode)
    with patch("tasks._reprotect_after_late_sell", wraps=tasks._reprotect_after_late_sell) as rp:
        place, queue, rows = _late_buy(cache=cache, broker=144.0, order=order)
    return (rp.call_args_list, place.call_args_list, queue.call_args_list, _late_buy.sent), rows


@pytest.mark.parametrize("manual", [False, True])
def test_shadow_places_exactly_as_off_and_records_would_defer(monkeypatch, manual):
    """The cache lags the fill: enforce would defer. Shadow passes the same
    args, places the same stop, queues nothing and words the alert as off."""
    order = _manual_order() if manual else None
    off, off_rows = _late_buy_calls(monkeypatch, "off", 100, order)
    shadow, rows = _late_buy_calls(monkeypatch, "shadow", 100, order)
    assert shadow == off and off_rows == []
    assert shadow[1] and shadow[1][0].args[:2] == ("CRWD", 100.0)           # placed now, as today
    assert all("expect_at_least" not in c.kwargs for c in shadow[0])
    assert not any("deferred" in t for t in shadow[3])
    assert "CRWD" not in tasks._REPROTECT_DEFERRED
    assert rows[0]["mode"] == "shadow" and rows[0]["would_defer"] is True
    assert rows[0]["cache_qty"] == 100.0 and rows[0]["cache_read"] == "ok"
    assert rows[0]["cache_at"] == "before_place" and rows[0]["expect_at_least"] == 144.0
    assert rows[0]["path"] == ("reconcile_late_manual_buy" if manual else "reconcile_late_buy")


@pytest.mark.parametrize("manual", [False, True])
def test_shadow_places_the_stop_before_any_broker_read(monkeypatch, manual):
    """The record-only reads can take over a minute on a reconnect: in
    shadow they run after the stop, and the cache snapshot (local) before."""
    _, rows = _late_buy_calls(monkeypatch, "shadow", 100, _manual_order() if manual else None)
    ev = _late_buy.events
    assert "place" in ev and "reqPositions" in ev and "positions" in ev
    assert ev.index("place") < ev.index("reqPositions") < ev.index("positions")
    assert ev[0] == "snapshot"                        # the local snapshot, before the reprotect
    assert rows[0]["would_defer"] is True


def test_shadow_row_is_written_even_when_the_place_fails(monkeypatch):
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "shadow")
    with patch("tasks._reprotect_after_late_sell", side_effect=RuntimeError("ib")):
        _, queue, rows = _late_buy(cache=100, broker=144.0)
    queue.assert_called_once_with("CRWD", {"intent": "BUY"}, None)
    assert rows[0]["would_defer"] is True and rows[0]["cache_at"] == "before_place"
    assert rows[0]["stop_placed"] == "raised"


def test_shadow_row_failure_never_raises(monkeypatch):
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "shadow")
    with patch("tasks._late_fill_expect", side_effect=RuntimeError("x")):
        place, _, _ = _late_buy(cache=100, broker=144.0)
    assert place.call_args_list == [call("CRWD", 100.0, 153.61)]


def test_enforce_reads_before_the_place(on):
    _late_buy(cache=144, broker=144.0)
    ev = _late_buy.events
    assert ev.index("reqPositions") < ev.index("positions") < ev.index("place")


def test_shadow_with_the_cache_caught_up_would_not_defer(monkeypatch):
    _, rows = _late_buy_calls(monkeypatch, "shadow", 144)
    assert rows[0]["would_defer"] is False and rows[0]["cache_qty"] == 144.0


def test_shadow_flat_cache_would_not_defer(monkeypatch):
    """A flat cache read takes the flat-BUY retry path, deferral or not."""
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "shadow")
    with patch("ibkr._ib", _fake_ib(0)):
        out, rows, _ = _expect(144.0)
    assert out is None and rows[0]["cache_qty"] == 0.0 and rows[0]["would_defer"] is False


def test_shadow_unreadable_cache_records_none_and_returns_none(monkeypatch):
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "shadow")
    ib = _fake_ib(100)
    ib.portfolio.side_effect = RuntimeError("ib")
    with patch("ibkr._ib", ib):
        out, rows, _ = _expect(144.0)
    assert out is None and rows[0]["would_defer"] is None
    assert rows[0]["cache_read"] == "error: RuntimeError" and rows[0]["expect_at_least"] == 144.0


def test_shadow_reads_in_order_fresh_positions_then_cache(monkeypatch):
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "shadow")
    seen = []
    with patch("ibkr._ib", _fake_ib(100, events=seen)), \
         patch("ibkr.get_fresh_stock_position",
               side_effect=lambda *a, **k: (seen.append("reqPositions"), (144.0, 181.0))[1]), \
         patch("ibkr.get_broker_stock_qty",
               side_effect=lambda *a, **k: (seen.append("positions"), 144.0)[1]), \
         patch("db.log_activity_queued") as queued:
        assert tasks._late_fill_expect("CRWD", 44.0, "reconcile_late_buy") is None
    assert seen == ["reqPositions", "positions", "snapshot"]
    assert queued.call_args.args[4]["would_defer"] is True


@pytest.mark.parametrize("mode", ["off", "OFF", "bogus", None, False])
def test_off_or_unknown_mode_reads_nothing(monkeypatch, mode):
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", mode)
    with patch("ibkr.get_broker_stock_qty") as bread, patch("ibkr.get_fresh_stock_position") as fread, \
         patch("ibkr.get_client") as client, patch("db.log_activity_queued") as queued:
        assert tasks._late_fill_expect("CRWD", 44.0, "x") is None
    bread.assert_not_called(); fread.assert_not_called(); client.assert_not_called()
    queued.assert_not_called()


def test_shipped_default_is_shadow():
    import importlib
    src = open(importlib.util.find_spec("config").origin).read()
    assert 'LATE_FILL_EXPECT_MODE = "shadow"' in src


# ── review fixes ────────────────────────────────────────────────────────────

def test_retry_in_shadow_never_passes_a_queued_expectation(monkeypatch):
    import json
    monkeypatch.setattr(config, "LATE_FILL_EXPECT_MODE", "shadow")
    state = {"reprotect_pending": json.dumps({"CRWD": {"intent": "BUY", "stop_price": None,
                                                      "fill_price": None,
                                                      "expect_at_least": 144.0}})}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reprotect_after_late_sell", return_value=True) as rp:
        tasks._retry_pending_reprotects()
    assert "expect_at_least" not in rp.call_args.kwargs


@pytest.mark.parametrize("ib,read", [(None, "not connected"), ("down", "not connected")])
def test_cache_snapshot_never_reconnects(ib, read):
    fake = _fake_ib(100, connected=False) if ib == "down" else None
    with patch("ibkr._ib", fake), patch("ibkr.get_ib", side_effect=AssertionError("reconnect")), \
         patch("ibkr.get_client", side_effect=AssertionError("reconnect")):
        assert tasks._late_fill_cache_qty("CRWD") == (None, read)


def test_cache_snapshot_reads_the_portfolio_directly():
    with patch("ibkr._ib", _fake_ib(100)), patch("ibkr.get_ib", side_effect=AssertionError("x")):
        assert tasks._late_fill_cache_qty("CRWD") == (100.0, "ok")
        assert tasks._late_fill_cache_qty("AMD") == (0.0, "ok")


def test_shadow_rows_wait_for_the_end_of_the_reconcile_run():
    """Order A's record-only broker reads must not delay order B's stop."""
    seen = []
    with patch("tasks._late_fill_expect",
               side_effect=lambda t, *a, **k: seen.append(("row", t))):
        def run():
            tasks._late_fill_shadow_add(("A", 1.0, "p", (1.0, "ok"), True))
            seen.append(("stop", "B"))
            tasks._late_fill_shadow_add(("B", 1.0, "p", (1.0, "ok"), True))
            return "done"
        assert tasks._late_fill_shadow_buffered(run) == "done"
    assert seen == [("stop", "B"), ("row", "A"), ("row", "B")]


def test_shadow_rows_are_written_when_the_run_raises_and_outside_a_run_at_once():
    seen = []
    with patch("tasks._late_fill_expect", side_effect=lambda t, *a, **k: seen.append(t)):
        def run():
            tasks._late_fill_shadow_add(("A", 1.0, "p", (1.0, "ok"), False))
            raise RuntimeError("boom")
        with pytest.raises(RuntimeError):
            tasks._late_fill_shadow_buffered(run)
        assert seen == ["A"]
        tasks._late_fill_shadow_add(("C", 1.0, "p", (1.0, "ok"), True))   # no run: now
        assert seen == ["A", "C"]


def test_a_failing_shadow_row_does_not_stop_the_others():
    seen = []

    def write(t, *a, **k):
        seen.append(t)
        if t == "A":
            raise RuntimeError("db")
    with patch("tasks._late_fill_expect", side_effect=write):
        tasks._late_fill_shadow_buffered(lambda: [tasks._late_fill_shadow_add((t, 1.0, "p", None, True))
                                                  for t in ("A", "B")])
    assert seen == ["A", "B"]


def test_nested_buffer_is_written_by_the_outer_run():
    seen = []
    with patch("tasks._late_fill_expect", side_effect=lambda t, *a, **k: seen.append(t)):
        def inner():
            tasks._late_fill_shadow_add(("A", 1.0, "p", None, True))
        def outer():
            tasks._late_fill_shadow_buffered(inner)
            seen.append("outer_end")
        tasks._late_fill_shadow_buffered(outer)
    assert seen == ["outer_end", "A"]


def test_reconcile_run_buffers_the_shadow_rows():
    import inspect
    assert "_late_fill_shadow_buffered(_reconcile_unconfirmed_orders_locked" in \
        inspect.getsource(tasks.reconcile_unconfirmed_orders)


def test_shadow_row_records_a_placed_stop(monkeypatch):
    _, rows = _late_buy_calls(monkeypatch, "shadow", 100)
    assert rows[0]["stop_placed"] is True


def test_place_failure_sends_no_pt_alert_and_flags_the_count(on):
    def boom(*a, **k):
        raise RuntimeError("ib")
    place, rows, alerts, _ = _pt_sale(TimeoutError("ib"), cache=2, place_effect=boom)
    assert alerts == []                                     # no "stop for the rest" text
    texts = [c.args[0] for c in _pt_sale.sends]
    assert len(texts) == 1 and texts[0].startswith("🚨 UNPROTECTED POSITION")
    assert "flagged by the sizing check" in texts[0] and "for 1.0 shares" in texts[0]


def test_place_failure_without_a_pt_flag_keeps_the_old_text(on):
    def boom(*a, **k):
        raise RuntimeError("ib")
    _pt_sale(1.0, cache=1, place_effect=boom)
    texts = [c.args[0] for c in _pt_sale.sends]
    assert len(texts) == 1 and "flagged" not in texts[0]


def test_capped_info_alert_is_sent_as_a_report():
    import alert_policy
    with patch.object(config, "PT_STOP_QTY_FROM_BROKER", True):
        _pt_sale(3.0, cache=2)
    capped = [c for c in _pt_sale.sends if "PT STOP CAPPED" in c.args[0]]
    assert len(capped) == 1 and capped[0].kwargs.get("severity") == alert_policy.REPORT
    others = [c for c in _pt_sale.sends if "PT STOP CAPPED" not in c.args[0]]
    assert all(c.kwargs.get("severity") is None for c in others)


@pytest.mark.parametrize("broker,ambiguous", [
    (3.0, False),     # held 3, sold 2, nothing applied: count − exp = 2 = fill
    (2.0, False),     # half applied: count − exp = 1 ≤ fill
    (4.0, True)])     # 3 above held − fill: more than the sale
def test_partly_applied_sale_is_capped_silently(on, broker, ambiguous):
    place, rows, alerts, _ = _pt_sale(broker, cache=3, held=3, sold=2)
    assert place.call_args_list == [call("NVDA", 1.0, 100.0, market_price=120.0)]
    assert rows[0]["capped"] is True and rows[0]["capped_ambiguous"] is ambiguous
    assert (len(alerts) == 1 and "PT STOP CAPPED" in alerts[0]) if ambiguous else alerts == []
