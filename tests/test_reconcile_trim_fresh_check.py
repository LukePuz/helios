"""Reconcile section 2b's over-cover trim checks a fresh positions read
(RECONCILE_TRIM_FRESH_CHECK_MODE; 08 Helios Engineering/reconcile-trim-under-lag.md G1).

Before: 2b took a fresh read only when some ticker looked flat. With none
flat, a cache reading a held ticker short on two cycles had _fix_overstop
cancel every stop and re-place one for the short count, so the rest of the
real shares went bare.

Now, for that trim, one bounded fresh read per cycle (no reconnect; the
cycle's own fresh read when it took one) is compared with the cache:
- shadow (shipped default): the trim runs as before. After the cycle's order
  work (a read that times out drops the session, so it never runs before a
  fix), each trim that sent a cancel is checked; a fresh read that disagrees
  or failed writes one reconcile_trim_fresh_check row.
- enforce: read before the trim; on a disagreeing or failed read the
  add-only branch (allow_cancel False, capped at the smaller read) and a
  throttled alert run instead, plus the same row.
- off: no read, no row, today's trim.
A fresh read that agrees changes nothing and writes no row."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, MagicMock, call, patch

import pytest

import config
import tasks

OLD = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()   # marker ready
ROW_KEYS = {"mode", "would_block", "blocked", "read_status", "fresh_vs_cache",
            "covered_vs_cache", "minutes_since_last_buy_fill", "read_source", "read_lag_s",
            "mkt_sell_working", "buy_fill_after_trim", "env"}


def _stop(oid, qty, ticker="AMD"):
    return {"order_id": oid, "perm_id": None, "ticker": ticker, "action": "SELL",
            "order_type": "STP", "qty": float(qty), "stop_price": 150.0, "status": "Submitted",
            "remaining": float(qty)}


def _fake_fix(fix_msg, cancel_sent, events, sized=5.0):
    """_fix_overstop_safe stand-in; a sent cancel fills outcome_out as the
    real one does (sized: the signed count it sized from)."""
    def fix(ticker, **kw):
        events.append(("fix", ticker))
        if cancel_sent and kw.get("outcome_out") is not None:
            kw["outcome_out"].update(cancel_sent=True, sized_signed=sized, mkt_sell_working=False,
                                     cancel_at=datetime.now(timezone.utc))
        return fix_msg
    return fix


def _run(monkeypatch, mode, fresh=None, fresh_error=None, cache=5, stops=None, armed=True,
         fix_msg="🔧 AMD: stops covered 10 shares but 5 held; placed a stop for 5 @ $150.00",
         records=(), last_buy=None, cancel_sent=True, state=None, real_fix=False, sized=5.0,
         cancel_error=None):
    """One reconcile cycle with no ticker looking flat. real_fix: the real
    _fix_overstop with the order calls patched (place / cancel mocks
    returned in `orders`)."""
    monkeypatch.setattr(config, "RECONCILE_TRIM_FRESH_CHECK_MODE", mode, raising=False)
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=cache)]
    state = state if state is not None else {}
    if armed:
        state.setdefault("overstop_pending_AMD", OLD)
    stops = stops if stops is not None else [_stop(1, 5), _stop(2, 5)]   # the two-stop pair
    events: list = []
    cancelled = {"all": False}

    def open_orders(*a, **k):
        events.append(("open_orders",))
        return [] if cancelled["all"] else list(stops)

    def read(*a, **k):
        events.append(("read", k))
        if fresh_error is not None:
            raise fresh_error
        return fresh

    orders = {}
    with ExitStack() as st:
        st.enter_context(patch("ibkr.get_client", return_value=client))
        st.enter_context(patch("ibkr.get_open_orders", side_effect=open_orders))
        rd = st.enter_context(patch("ibkr.get_fresh_stock_positions", side_effect=read))
        st.enter_context(patch("db.get_all_position_stops", return_value=list(records)))
        st.enter_context(patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)))
        st.enter_context(patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)))
        act = st.enter_context(patch("db.log_activity_direct", return_value=True))
        st.enter_context(patch("db.last_buy_fill_at", return_value=last_buy))
        for name in ("get_unconfirmed_orders", "awaiting_fill_sell_tickers",
                     "trade_pending_sell_tickers"):     # sections 1-2's pending-SELL reads
            st.enter_context(patch(f"db.{name}", return_value=[]))
        if real_fix:
            def cancel_all(ticker, *a, **k):
                if cancel_error is not None:
                    raise cancel_error
                cancelled["all"] = True
            orders["cancel_all"] = st.enter_context(
                patch("tasks.manage_cancel_stop_orders_for", side_effect=cancel_all))
            orders["cancel_one"] = st.enter_context(patch("tasks.manage_cancel_order"))
            orders["place"] = st.enter_context(
                patch("tasks.manage_stop_place", return_value={"order_id": 9}))
            st.enter_context(patch("db.get_position_stop", return_value={"stop_price": 150.0}))
            st.enter_context(patch("db.set_stop_broker_order_id"))
            st.enter_context(patch("ibkr.get_latest_price", return_value=200.0))
            st.enter_context(patch("ibkr.get_fresh_minute_price", return_value=None))
            fix = st.enter_context(patch("tasks._fix_overstop_safe",
                                         side_effect=tasks._fix_overstop_safe))
        else:
            fix = st.enter_context(patch("tasks._fix_overstop_safe",
                                         side_effect=_fake_fix(fix_msg, cancel_sent, events,
                                                               sized)))
        send = st.enter_context(patch("alerts._send"))
        tasks.reconcile_stop_orders.run()
    rows = [c.args for c in act.call_args_list if c.args[0] == "reconcile_trim_fresh_check"]
    return dict(fix=fix, read=rd, rows=rows, send=send, state=state, events=events, orders=orders)


def _sent(r):
    return [c.args[0] for c in r["send"].call_args_list]


# ── shadow: the open case (no flat ticker, the cache short on both cycles) ──

def test_shadow_trims_as_before_and_records_the_would_block(monkeypatch):
    last = datetime.now(timezone.utc) - timedelta(minutes=12)
    r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0}, last_buy=last)
    assert r["fix"].call_args_list == [call("AMD", outcome_out=ANY)]   # today's trim
    r["read"].assert_called_once_with(timeout=tasks._TRIM_FRESH_TIMEOUT_S, reconnect=False)
    assert len(r["rows"]) == 1
    stage, ticker, message, _, extra = r["rows"][0]
    assert (stage, ticker) == ("reconcile_trim_fresh_check", "AMD")
    assert set(extra) == ROW_KEYS                                     # no share counts
    assert extra["mode"] == "shadow" and extra["would_block"] is True and extra["blocked"] is False
    assert extra["read_status"] == "mismatch"
    assert extra["fresh_vs_cache"] == 1.0 and extra["covered_vs_cache"] == 2.0   # fractions
    assert 11.5 <= extra["minutes_since_last_buy_fill"] <= 13
    assert extra["read_source"] == "own" and 0 <= extra["read_lag_s"] < 5
    assert extra["buy_fill_after_trim"] is False and extra["mkt_sell_working"] is False
    assert "ran" in message and "disagrees" in message


def test_shadow_compares_with_the_count_the_trim_sized_from(monkeypatch):
    """The cycle-start cache read 5, but the trim's own under-lock read sized
    from 10 and the fresh read shows 10: the trim used the right count, no row."""
    r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0}, sized=10.0)
    r["read"].assert_called_once()
    assert r["rows"] == []


def test_shadow_flags_a_buy_fill_after_the_trim(monkeypatch):
    later = datetime.now(timezone.utc) + timedelta(seconds=30)
    r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0}, last_buy=later)
    assert r["rows"][0][4]["buy_fill_after_trim"] is True


def test_shadow_records_a_trim_whose_cancel_raised(monkeypatch):
    """A cancel call that raises may still land at the broker: the trim
    counts as sent, so the shadow checks it."""
    import execution
    err = getattr(execution, "OrdersCallFailed", RuntimeError)("boom")
    r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0}, real_fix=True, cancel_error=err)
    r["orders"]["cancel_all"].assert_called_once_with("AMD")
    assert len(r["rows"]) == 1 and r["rows"][0][4]["read_status"] == "mismatch"


def test_a_failing_shadow_check_never_reaches_the_retry(monkeypatch):
    with patch("tasks._trim_fresh_check", side_effect=RuntimeError("check")):
        r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0})   # .run() would raise Retry
    assert r["fix"].call_args_list == [call("AMD", outcome_out=ANY)]
    assert r["rows"] == []


def test_shadow_reads_only_after_the_cycles_order_work(monkeypatch):
    """A read that times out drops the session (ibkr._bounded_req), so it must
    not run before the trim: shadow's read comes after every fix."""
    r = _run(monkeypatch, "shadow", fresh_error=TimeoutError())
    kinds = [e[0] for e in r["events"]]
    assert "fix" in kinds and "read" in kinds
    assert kinds.index("read") > max(i for i, k in enumerate(kinds) if k == "fix")
    assert r["rows"][0][4]["read_status"] == "failed: TimeoutError"


def test_shadow_writes_no_row_when_the_fix_sent_no_cancel(monkeypatch):
    r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0}, cancel_sent=False,
             fix_msg="🚨 AMD: stops cover 10 of 5 shares; recorded stop $150.00 is not below "
                     "the last price (unknown), so nothing was placed — set the stop in TWS")
    r["read"].assert_not_called()
    assert r["rows"] == []


def test_shadow_with_the_real_fix_orders_as_off(monkeypatch):
    """The cancel and place calls in shadow equal those with the mode off."""
    calls = {}
    for mode in ("off", "shadow"):
        r = _run(monkeypatch, mode, fresh={"AMD": 10.0}, real_fix=True)
        calls[mode] = (r["orders"]["cancel_all"].call_args_list,
                       r["orders"]["place"].call_args_list,
                       r["orders"]["cancel_one"].call_args_list, _sent(r))
        if mode == "shadow":
            assert len(r["rows"]) == 1 and r["rows"][0][4]["read_status"] == "mismatch"
    assert calls["shadow"] == calls["off"]
    assert calls["off"][0] == [call("AMD")]
    assert calls["off"][1] == [call("AMD", 5.0, 150.0, market_price=200.0)]
    assert any(s.startswith("STOP RECONCILE — 🔧 AMD") for s in calls["off"][3])


# ── enforce ──────────────────────────────────────────────────────────────────

def test_enforce_takes_the_add_only_branch_and_alerts(monkeypatch):
    r = _run(monkeypatch, "enforce", fresh={"AMD": 10.0},
             fix_msg="🚨 AMD: resting stops for 10 shares exceed the smaller position read (5) "
                     "while the reads disagree; nothing added or cancelled — check the position "
                     "and its stops in TWS")
    assert r["fix"].call_args_list == [call("AMD", allow_cancel=False, max_qty=5.0)]
    assert r["read"].call_count == 1
    assert len(r["rows"]) == 1 and r["rows"][0][4]["blocked"] is True
    assert r["rows"][0][4]["mode"] == "enforce" and "blocked" in r["rows"][0][2]
    assert set(r["rows"][0][4]) == ROW_KEYS
    assert any(t.startswith("STOP RECONCILE — 🚨 AMD") for t in _sent(r))


def test_enforce_alert_is_throttled(monkeypatch):
    state = {"reconcile_alert_overstop_trim_fresh_AMD": datetime.now(timezone.utc).isoformat()}
    r = _run(monkeypatch, "enforce", fresh={"AMD": 10.0}, state=state,
             fix_msg="🚨 AMD: resting stops for 10 shares exceed the smaller position read (5)")
    assert r["fix"].call_args_list == [call("AMD", allow_cancel=False, max_qty=5.0)]
    assert not any(t.startswith("STOP RECONCILE") for t in _sent(r))
    assert len(r["rows"]) == 1                                         # still recorded


def test_enforce_throttle_never_holds_back_other_alerts(monkeypatch):
    """Only the add-only report is throttled: a short, a failed fix or an
    unusable stop level alerts every time, as with the mode off."""
    state = {"reconcile_alert_overstop_trim_fresh_AMD": datetime.now(timezone.utc).isoformat()}
    r = _run(monkeypatch, "enforce", fresh={"AMD": 4.0}, state=state,
             fix_msg="🚨 AMD: broker shows a SHORT of -5; no stop placed — check TWS now")
    assert any(t.startswith("STOP RECONCILE — 🚨 AMD: broker shows a SHORT") for t in _sent(r))


@pytest.mark.parametrize("fresh, status, cap, ratio", [
    ({"NVDA": 3.0}, "missing", 0.0, -1.0),           # the fresh read leaves it out
    ({"AMD": -5.0}, "mismatch", 0.0, -2.0),          # a sign flip: no SELL stop added
    ({"AMD": 4.0}, "mismatch", 4.0, -0.2),           # fresh below the cache: capped there
], ids=["missing", "sign_flip", "fresh_lower"])
def test_enforce_caps_at_the_smaller_read(monkeypatch, fresh, status, cap, ratio):
    r = _run(monkeypatch, "enforce", fresh=fresh)
    assert r["fix"].call_args_list == [call("AMD", allow_cancel=False, max_qty=cap)]
    assert r["rows"][0][4]["read_status"] == status
    assert r["rows"][0][4]["fresh_vs_cache"] == ratio


@pytest.mark.parametrize("kw", [
    {"fresh": {"AMD": 10.0}}, {"fresh": {"AMD": 4.0}}, {"fresh": {"AMD": -5.0}},
    {"fresh": {"NVDA": 3.0}}, {"fresh": {}}, {"fresh_error": TimeoutError()},
], ids=["higher", "lower", "sign_flip", "missing", "empty", "timeout"])
def test_enforce_with_the_real_fix_cancels_nothing(monkeypatch, kw):
    r = _run(monkeypatch, "enforce", real_fix=True, **kw)
    r["orders"]["cancel_all"].assert_not_called()
    r["orders"]["cancel_one"].assert_not_called()
    r["orders"]["place"].assert_not_called()
    assert any(t.startswith("STOP RECONCILE — 🚨 AMD") for t in _sent(r))


def test_enforce_failed_read_alert_names_the_read(monkeypatch):
    r = _run(monkeypatch, "enforce", real_fix=True, fresh_error=TimeoutError())
    text = next(t for t in _sent(r) if t.startswith("STOP RECONCILE"))
    assert "fresh positions read failed: TimeoutError" in text
    assert "reads disagree" not in text


# ── the fresh read agrees: today's trim, no row ──────────────────────────────

@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_an_agreeing_fresh_read_changes_nothing(monkeypatch, mode):
    r = _run(monkeypatch, mode, fresh={"AMD": 5.0})
    assert [c.args for c in r["fix"].call_args_list] == [("AMD",)]
    r["read"].assert_called_once()
    assert r["rows"] == []


# ── a failed or empty read ───────────────────────────────────────────────────

@pytest.mark.parametrize("kw, status", [
    ({"fresh_error": TimeoutError()}, "failed: TimeoutError"),
    ({"fresh_error": ConnectionError("IBKR not connected")}, "failed: ConnectionError"),
    ({"fresh": {}}, "empty"),
], ids=["timeout", "no_session", "empty"])
def test_a_failed_read_records_in_shadow(monkeypatch, kw, status):
    r = _run(monkeypatch, "shadow", **kw)
    assert r["fix"].call_args_list == [call("AMD", outcome_out=ANY)]
    assert len(r["rows"]) == 1
    extra = r["rows"][0][4]
    assert extra["read_status"] == status and extra["blocked"] is False
    assert extra["fresh_vs_cache"] is None and "failed" in r["rows"][0][2]


def test_a_failed_read_is_add_only_in_enforce(monkeypatch):
    r = _run(monkeypatch, "enforce", fresh_error=TimeoutError())
    assert r["fix"].call_args_list == [call("AMD", allow_cancel=False, max_qty=None)]
    assert r["rows"][0][4]["blocked"] is True


# ── off, and what the check never touches ────────────────────────────────────

@pytest.mark.parametrize("mode", ["off", "", None, "bogus"])
def test_off_reads_nothing_and_trims_as_today(monkeypatch, mode):
    r = _run(monkeypatch, mode, fresh={"AMD": 10.0})
    assert r["fix"].call_args_list == [call("AMD")]
    r["read"].assert_not_called()
    assert r["rows"] == []


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_under_cover_is_not_checked(monkeypatch, mode):
    r = _run(monkeypatch, mode, fresh={"AMD": 10.0}, cache=15)
    assert r["fix"].call_args_list == [call("AMD")]          # adds the missing stop as today
    r["read"].assert_not_called()
    assert r["rows"] == []


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_one_fresh_read_per_cycle_for_several_tickers(monkeypatch, mode):
    nvda = MagicMock(symbol="NVDA", qty=2)
    stops = [_stop(1, 5), _stop(2, 5), _stop(3, 2, "NVDA"), _stop(4, 2, "NVDA")]
    monkeypatch.setattr(config, "RECONCILE_TRIM_FRESH_CHECK_MODE", mode, raising=False)
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=5), nvda]
    state = {"overstop_pending_AMD": OLD, "overstop_pending_NVDA": OLD}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=stops), \
         patch("ibkr.get_fresh_stock_positions", return_value={"AMD": 10.0, "NVDA": 4.0}) as read, \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("db.last_buy_fill_at", return_value=None), \
         patch("db.get_unconfirmed_orders", return_value=[]), \
         patch("db.awaiting_fill_sell_tickers", return_value=[]), \
         patch("db.trade_pending_sell_tickers", return_value=[]), \
         patch("tasks._fix_overstop_safe", side_effect=_fake_fix("🔧 x", True, [])) as fix, \
         patch("alerts._send"):
        tasks.reconcile_stop_orders.run()
    read.assert_called_once()
    assert sorted(c.args[1] for c in act.call_args_list
                  if c.args[0] == "reconcile_trim_fresh_check") == ["AMD", "NVDA"]
    assert sorted(c.args[0] for c in fix.call_args_list) == ["AMD", "NVDA"]


def test_enforce_after_a_timed_out_read_blocks_every_trim_without_a_second_read(monkeypatch):
    stops = [_stop(1, 5), _stop(2, 5), _stop(3, 2, "NVDA"), _stop(4, 2, "NVDA")]
    monkeypatch.setattr(config, "RECONCILE_TRIM_FRESH_CHECK_MODE", "enforce", raising=False)
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=5),
                                             MagicMock(symbol="NVDA", qty=2)]
    state = {"overstop_pending_AMD": OLD, "overstop_pending_NVDA": OLD}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", return_value=stops), \
         patch("ibkr.get_fresh_stock_positions", side_effect=TimeoutError()) as read, \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_activity_direct", return_value=True), \
         patch("db.last_buy_fill_at", return_value=None), \
         patch("db.get_unconfirmed_orders", return_value=[]), \
         patch("db.awaiting_fill_sell_tickers", return_value=[]), \
         patch("db.trade_pending_sell_tickers", return_value=[]), \
         patch("tasks._fix_overstop_safe", return_value="🚨 x") as fix, \
         patch("alerts._send"):
        tasks.reconcile_stop_orders.run()
    read.assert_called_once()
    assert sorted(fix.call_args_list) == sorted([call("AMD", allow_cancel=False, max_qty=None),
                                                 call("NVDA", allow_cancel=False, max_qty=None)])


def test_the_cycles_own_fresh_read_is_reused_in_the_body(monkeypatch):
    """Section 1 took a fresh read (MSFT has a stop record the cache doesn't
    hold, so it looks flat; the fresh read doesn't hold it either, so nothing
    is in disagree): the trim check reuses that read, with no second call."""
    rec = {"ticker": "MSFT", "stop_price": 300.0, "qty": 3.0, "broker_order_id": None}
    r = _run(monkeypatch, "shadow", fresh={"AMD": 10.0}, records=[rec])
    r["read"].assert_called_once_with()                  # section 1's own read, the defaults
    assert [x[4]["read_status"] for x in r["rows"] if x[1] == "AMD"] == ["mismatch"]


def test_a_first_cycle_only_arms_the_marker(monkeypatch):
    r = _run(monkeypatch, "enforce", fresh={"AMD": 10.0}, armed=False)
    r["fix"].assert_not_called()
    r["read"].assert_not_called()                        # no read before the marker is ready
    assert r["rows"] == [] and r["state"].get("overstop_pending_AMD")


# ── the helpers ──────────────────────────────────────────────────────────────

def test_the_cycles_own_fresh_read_is_reused():
    box: dict = {}
    with patch("config.RECONCILE_TRIM_FRESH_CHECK_MODE", "shadow"), \
         patch("ibkr.get_fresh_stock_positions") as read:
        out = tasks._trim_fresh_check("AMD", 5.0, {"AMD": 5.0}, box)
    read.assert_not_called()
    assert out["block"] is False and out["status"] == "agrees"


def test_fix_overstop_reports_a_sent_cancel():
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=5)]
    out: dict = {}
    with patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_open_orders", side_effect=[[_stop(1, 5), _stop(2, 5)], []]), \
         patch("db.get_position_stop", return_value={"stop_price": 150.0}), \
         patch("ibkr.get_latest_price", return_value=200.0), \
         patch("ibkr.get_fresh_minute_price", return_value=None), \
         patch("tasks.manage_cancel_stop_orders_for") as cancel, \
         patch("tasks.manage_stop_place", return_value={"order_id": 9}), \
         patch("db.set_stop_broker_order_id"):
        msg = tasks._fix_overstop("AMD", outcome_out=out)
    cancel.assert_called_once_with("AMD")
    assert out["cancel_sent"] is True and out["sized_signed"] == 5.0
    assert out["mkt_sell_working"] is False and out["cancel_at"].tzinfo is not None
    assert msg.startswith("🔧")


def test_a_failing_row_write_never_raises():
    with patch("db.last_buy_fill_at", side_effect=RuntimeError("db")), \
         patch("db.log_activity_direct") as act:
        tasks._record_trim_fresh_check("AMD", {"mode": "shadow", "status": "mismatch",
                                               "fresh": 10.0}, 10.0, 5.0)
    act.assert_not_called()


def test_last_buy_fill_at_returns_none_on_a_failed_read():
    import db
    with patch("db._get_conn", side_effect=RuntimeError("pool")):
        assert db.last_buy_fill_at("AMD") is None
