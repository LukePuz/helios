"""Record-only fill watch (tasks.watch_broker_sells): one broker_sell_seen row
the first time a SELL the bot did not log appears in today's executions.
Reads only; never places, cancels or books; no price or quantity."""
import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import config
import db
import ibkr
import tasks

NOW = datetime(2026, 11, 20, 16, 40, tzinfo=timezone.utc)        # Fri 11:40 ET


def _fill(sym, perm, oid, t, side="SLD", shares=7, price=412.37):
    return SimpleNamespace(contract=SimpleNamespace(symbol=sym), time=None,
                           execution=SimpleNamespace(side=side, permId=perm, orderId=oid, time=t,
                                                     execId=f"e{perm}{oid}{t.minute}",
                                                     shares=shares, price=price))


def test_today_sell_executions_group_per_order_and_carry_no_price(monkeypatch):
    t1, t2 = NOW - timedelta(minutes=8), NOW - timedelta(minutes=7)
    fills = [_fill("MSFT", 853, 34, t1), _fill("MSFT", 853, 34, t2),       # one order, two execs
             _fill("NVDA", 0, 50, t1),                                     # no permId: orderId
             _fill("AMD", 900, 9, t1, side="BOT")]                         # a buy: ignored
    monkeypatch.setattr(ibkr, "get_ib", lambda: SimpleNamespace(reqExecutions=lambda: fills))
    monkeypatch.setattr(ibkr, "_bounded_req", lambda ib, what, fn, *a, **k: fn())
    out = {(g["ticker"], g["perm_id"], g["order_id"]): g for g in ibkr.get_today_sell_executions()}
    assert set(out) == {("MSFT", 853, 34), ("NVDA", None, 50)}
    m = out[("MSFT", 853, 34)]
    assert m["first_exec_at"] == t1 and m["last_exec_at"] == t2 and m["n_execs"] == 2
    assert m["time_source"] == "execution"
    for g in out.values():
        assert not ({"shares", "qty", "price", "fill_price"} & set(g))


@pytest.fixture
def env(monkeypatch):
    state, rows = {}, []
    monkeypatch.setattr(config, "BROKER_SELL_WATCH_ENABLED", True)
    monkeypatch.setattr(db, "get_state", lambda k, d=None: state.get(k, d))
    monkeypatch.setattr(db, "get_state_strict", lambda k, d=None: state.get(k, d))
    monkeypatch.setattr(db, "set_state", lambda k, v: state.__setitem__(k, v))
    monkeypatch.setattr(db, "awaiting_fill_sell_tickers", lambda: set())
    monkeypatch.setattr(db, "trade_pending_sell_tickers", lambda: set())
    monkeypatch.setattr(db, "get_unconfirmed_orders", lambda *a, **k: [])
    monkeypatch.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: (set(), set()))
    monkeypatch.setattr(tasks, "_broker_held", lambda t, *a, **k: (False, "snapshot", {}))
    monkeypatch.setattr(db, "log_activity_direct",
                        lambda stage, ticker, msg, sent="neutral", extra=None: rows.append(
                            (stage, ticker, msg, extra)) or True)
    execs = [{"ticker": "MSFT", "perm_id": 853, "order_id": 34,
              "first_exec_at": NOW - timedelta(minutes=8), "last_exec_at": NOW - timedelta(minutes=7),
              "time_source": "execution", "n_execs": 2}]
    monkeypatch.setattr(ibkr, "get_today_sell_executions", lambda: execs)
    return SimpleNamespace(state=state, rows=rows, execs=execs, mp=monkeypatch)


def test_writes_one_row_per_new_unlogged_sell(env):
    out = tasks._watch_broker_sells_body(now=NOW)
    assert len(out) == 1 and len(env.rows) == 1
    stage, ticker, msg, rec = env.rows[0]
    assert stage == "broker_sell_seen" and ticker == "MSFT"
    assert rec["perm_id"] == 853 and rec["seen_lag_s"] == 480.0
    assert rec["first_exec_at"] == (NOW - timedelta(minutes=8)).isoformat()
    assert "bot_sell_pending" not in rec and rec["held_in_snapshot"] is False
    assert "record only" in msg
    # the next run sees it as already recorded
    assert tasks._watch_broker_sells_body(now=NOW + timedelta(minutes=2)) == []
    assert len(env.rows) == 1


def test_row_has_no_price_or_quantity(env):
    tasks._watch_broker_sells_body(now=NOW)
    _, _, msg, rec = env.rows[0]
    text = msg + json.dumps(rec)
    for bad in ("412", "shares", "qty", "price", "\"7\""):
        assert bad not in text


def test_the_bots_own_logged_sell_is_skipped_and_remembered(env):
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: (set(), {853}))
    assert tasks._watch_broker_sells_body(now=NOW) == [] and env.rows == []
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: (set(), set()))
    assert tasks._watch_broker_sells_body(now=NOW) == []                    # already marked seen


def test_no_perm_id_matches_on_order_id(env):
    env.execs[0]["perm_id"] = None
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: ({34}, set()))
    assert tasks._watch_broker_sells_body(now=NOW) == []


def test_unknown_logged_sells_wait_for_the_next_run(env):
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: None)
    assert tasks._watch_broker_sells_body(now=NOW) == []
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: (set(), set()))
    assert len(tasks._watch_broker_sells_body(now=NOW)) == 1                 # not marked seen


def test_bots_early_row_without_perm_id_is_matched_on_order_id(env):
    """_early_order_row logs a bot SELL before its fill wait with no permId:
    recorded_sell_order_ids then holds its orderId only."""
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: ({34}, set()))
    assert tasks._watch_broker_sells_body(now=NOW) == [] and env.rows == []


@pytest.mark.parametrize("which", ["awaiting", "trade_pending", "unconfirmed_today"])
def test_pending_bot_sell_waits_and_is_classified_next_run(env, which):
    if which == "awaiting":
        env.mp.setattr(db, "awaiting_fill_sell_tickers", lambda: {"MSFT"})
    elif which == "trade_pending":
        env.mp.setattr(db, "trade_pending_sell_tickers", lambda: {"MSFT"})
    else:
        env.mp.setattr(db, "get_unconfirmed_orders", lambda *a, **k: [
            {"ticker": "MSFT", "side": "SELL", "created_at": NOW - timedelta(minutes=9)}])
    assert tasks._watch_broker_sells_body(now=NOW) == [] and env.rows == []
    assert not env.state                                        # not marked seen
    # next run: the fill is logged with its permId -> the bot's own order, no row
    env.mp.setattr(db, "awaiting_fill_sell_tickers", lambda: set())
    env.mp.setattr(db, "trade_pending_sell_tickers", lambda: set())
    env.mp.setattr(db, "get_unconfirmed_orders", lambda *a, **k: [])
    env.mp.setattr(db, "recorded_sell_order_ids", lambda t, *a, **k: (set(), {853}))
    assert tasks._watch_broker_sells_body(now=NOW + timedelta(minutes=2)) == [] and env.rows == []


def test_unconfirmed_sell_from_an_earlier_day_does_not_hold_the_row(env):
    env.mp.setattr(db, "get_unconfirmed_orders", lambda *a, **k: [
        {"ticker": "MSFT", "side": "SELL", "created_at": NOW - timedelta(days=1)}])
    assert len(tasks._watch_broker_sells_body(now=NOW)) == 1


def test_pending_read_failure_waits(env):
    env.mp.setattr(db, "get_unconfirmed_orders", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert tasks._watch_broker_sells_body(now=NOW) == [] and not env.state


def test_failed_seen_state_read_writes_nothing(env):
    tasks._watch_broker_sells_body(now=NOW)
    assert len(env.rows) == 1
    before = dict(env.state)
    env.mp.setattr(db, "get_state_strict", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    assert tasks._watch_broker_sells_body(now=NOW + timedelta(minutes=2)) == []
    assert len(env.rows) == 1 and env.state == before            # no duplicate, no state write


def test_seen_keys_keep_insertion_order_under_the_cap(env, monkeypatch):
    monkeypatch.setattr(tasks, "_SELL_SEEN_MAX", 2)
    base = env.execs[0]
    env.execs[:] = [dict(base, ticker=t, perm_id=p) for t, p in (("ZZZ", 1), ("AAA", 2), ("MMM", 3))]
    tasks._watch_broker_sells_body(now=NOW)
    kept = json.loads(next(iter(env.state.values())))["seen"]
    assert kept == ["AAA:perm:2", "MMM:perm:3"]                 # newest two, recorded order


def test_failed_write_is_retried_next_run(env):
    env.mp.setattr(db, "log_activity_direct", lambda *a, **k: False)
    assert tasks._watch_broker_sells_body(now=NOW) == []
    rows = []
    env.mp.setattr(db, "log_activity_direct", lambda stage, t, m, s="neutral", extra=None: rows.append(t) or True)
    assert len(tasks._watch_broker_sells_body(now=NOW)) == 1 and rows == ["MSFT"]


def test_new_day_resets_the_seen_set(env):
    tasks._watch_broker_sells_body(now=NOW)
    nxt = NOW + timedelta(days=3)                                            # Mon 11:40 ET
    assert len(tasks._watch_broker_sells_body(now=nxt)) == 1


@pytest.mark.parametrize("when", [datetime(2026, 11, 20, 13, 0, tzinfo=timezone.utc),   # 08:00 ET
                                  datetime(2026, 11, 20, 21, 45, tzinfo=timezone.utc),  # 16:45 ET
                                  datetime(2026, 11, 21, 16, 40, tzinfo=timezone.utc)]) # Saturday
def test_outside_market_hours_no_broker_read(env, when):
    env.mp.setattr(ibkr, "get_today_sell_executions", lambda: pytest.fail("broker read off-hours"))
    assert tasks._watch_broker_sells_body(now=when) == []


def test_disabled_does_nothing(env):
    env.mp.setattr(config, "BROKER_SELL_WATCH_ENABLED", False)
    env.mp.setattr(ibkr, "get_today_sell_executions", lambda: pytest.fail("read while disabled"))
    assert tasks._watch_broker_sells_body(now=NOW) == []


def test_never_raises(env):
    env.mp.setattr(ibkr, "get_today_sell_executions", lambda: (_ for _ in ()).throw(TimeoutError()))
    assert tasks._watch_broker_sells_body(now=NOW) == []
    env.mp.setattr(ibkr, "get_today_sell_executions", lambda: [{"junk": 1}, None, {"ticker": "X"}])
    tasks._watch_broker_sells_body(now=NOW)                    # odd rows: no raise
    env.mp.setattr(db, "get_state_strict", lambda *a, **k: "{not json")
    env.mp.setattr(ibkr, "get_today_sell_executions", lambda: env.execs)
    assert len(tasks._watch_broker_sells_body(now=NOW)) == 1


def test_reads_only_nothing_placed_cancelled_or_booked():
    src = inspect.getsource(tasks._watch_broker_sells_body) + inspect.getsource(ibkr.get_today_sell_executions)
    for call in ("place_", "placeOrder", "cancel_", "cancelOrder", "reqGlobalCancel", "log_trade(",
                 "close_trade_outcome(", "delete_position_stop(", "upsert_position_stop(",
                 "send_task(", "apply_async(", ".delay("):
        assert call not in src, call


def test_beat_runs_it_every_two_minutes():
    entry = tasks.app.conf.beat_schedule["watch-broker-sells"]
    assert entry["task"] == "tasks.watch_broker_sells" and entry["schedule"] == 120
    assert entry["options"]["expires"] == 120


def test_ticker_with_its_stop_lock_held_waits_for_the_next_run(env):
    """A profit-target sale holds the ticker's stop lock from the cancel
    through its order row: an untracked-looking SELL then may be the bot's."""
    from conftest import _FakeLock
    _FakeLock.held.add("helios:stop_lock:MSFT")
    assert tasks._watch_broker_sells_body(now=NOW) == [] and env.rows == []
    _FakeLock.held.discard("helios:stop_lock:MSFT")
    assert len(tasks._watch_broker_sells_body(now=NOW + timedelta(minutes=2))) == 1


def test_unreadable_stop_lock_waits_too(env):
    import stop_lock
    env.mp.setattr(stop_lock, "held_elsewhere", lambda t: None)
    assert tasks._watch_broker_sells_body(now=NOW) == [] and env.rows == []
