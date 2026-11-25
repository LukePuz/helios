"""Reclaim shadow for gate-blocked buys (agents/crew.py:_entry_gate_reclaim_record,
_reclaim_gates_pass, _check_entry_gate_reclaims). Record only: a block stores one
record per ticker; the nightly check logs the first session every blocking gate
would pass again and the 5/10/20-session forward moves, then expires the
record. Synthetic data only: system_state, the bars and the activity writes are
in memory; no DB, broker or Redis."""
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
from agents import crew

_REAL_BOOK_EDIT = crew._reclaim_book_edit   # conftest stubs it per test; restored below
_KEY = "entry_gate_reclaim_shadow_" + config.TRADING_ENV
_NOW = datetime(2026, 11, 23, 15, 40, tzinfo=timezone.utc)     # 10:40 ET


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS", 5)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "SPY")
    monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_SESSIONS", 20)
    monkeypatch.setattr(crew, "_reclaim_book_edit", _REAL_BOOK_EDIT)


@pytest.fixture
def state():
    """In-memory system_state; Redis down, so the edit runs unlocked."""
    st = {}
    with patch("db.get_state_strict", side_effect=lambda k, d=None: st.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: st.__setitem__(k, v)), \
         patch("stop_lock._redis", side_effect=RuntimeError("no redis")):
        yield st


def _book(st):
    return json.loads(st.get(_KEY) or "{}")


def _result(action="ADD", sid=4242):
    return {"ticker": "NVDA", "action": action, "conviction": 8, "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m", "signal_id": sid}}


def _trend(below50=True, below200=True, read="ok", age=1, last_close=100.0, bar_date="2026-11-20",
           suspect=False):
    return {"read": read, "bar_age_days": age, "below_sma50": below50, "below_sma200": below200,
            "sma50": 120.0, "sma200": 125.0, "last_close": last_close, "last_bar_date": bar_date,
            "pct_vs_sma50": -16.67 if below50 else 4.0, "pct_vs_sma200": -20.0 if below200 else 9.0,
            "split_suspect": suspect}


def _spy(below200=True, read="ok", age=1):
    return {"read": read, "bar_age_days": age, "below_sma200": below200,
            "pct_vs_sma200": -2.5 if below200 else 3.1, "split_suspect": False}


def _block(state, trend=None, stop_rec=None, result=None, now=_NOW, spy=None):
    """One routed buy through the live gate; returns the gate's answer."""
    real = crew._entry_gate_reclaim_record
    with patch("db.log_signal"), \
         patch.object(crew, "_entry_trend_check", return_value=spy or _spy()), \
         patch.object(crew, "_entry_gate_reclaim_record",
                      side_effect=lambda *a, **k: real(*a, now=now, **k)):
        return crew._entry_gate("NVDA", result or _result(), stop_rec,
                                _trend() if trend is None else trend)


# ── storing ──────────────────────────────────────────────────────────────────

class TestStore:
    def test_a_trend_block_stores_one_record_with_the_levels_it_tested(self, state):
        assert _block(state) is True
        rec = _book(state)["NVDA"]
        assert rec["gates"] == ["trend"] and rec["action"] == "ADD" and rec["signal_id"] == 4242
        assert rec["blocked_at"] == rec["first_blocked_at"] == _NOW.isoformat()
        assert rec["et_date"] == "2026-11-23" and rec["blocks"] == 1
        lv = rec["levels"]
        assert (lv["sma50"], lv["sma200"], lv["pct_vs_sma50"], lv["pct_vs_sma200"]) == \
            (120.0, 125.0, -16.67, -20.0)
        assert (lv["market_ticker"], lv["market_below_sma200"], lv["market_pct_vs_sma200"]) == \
            ("SPY", True, -2.5)
        assert rec["base_close"] == 100.0 and rec["base_bar_date"] == "2026-11-20"
        assert rec["logged"] == {}

    def test_a_stopout_block_keeps_the_window_end(self, state):
        stop = {"read": "ok", "hours_since": 17.5, "exit_at": "2026-11-19T20:56:32+00:00"}
        assert _block(state, trend=_trend(False, False), stop_rec=stop) is True
        rec = _book(state)["NVDA"]
        assert rec["gates"] == ["stopout"]
        assert rec["levels"]["stopout_window_end"] == "2026-11-29T20:56:32+00:00"
        assert rec["levels"]["stopout_days"] == 10

    def test_a_newer_block_replaces_the_record_and_keeps_first_blocked_at(self, state):
        _block(state)
        later = _NOW + timedelta(days=2)
        _block(state, trend=_trend(last_close=90.0, bar_date="2026-11-24"), now=later,
               result=_result(sid=5151))
        rec = _book(state)["NVDA"]
        assert rec["first_blocked_at"] == _NOW.isoformat()
        assert rec["blocked_at"] == later.isoformat() and rec["signal_id"] == 5151
        assert rec["blocks"] == 2 and rec["base_close"] == 90.0
        (old,) = rec["superseded_pending"]
        assert old["blocked_at"] == _NOW.isoformat() and old["gates"] == ["trend"]
        assert old["base_close"] == 100.0 and old["base_bar_date"] == "2026-11-20"
        assert old["signal_id"] == 4242 and old["logged"] == {}
        assert old["superseded_at"] == later.isoformat() and old["superseded_et_date"] == "2026-11-25"

    def test_superseded_snapshots_carry_over_and_are_capped(self, state):
        for i in range(crew._RECLAIM_SUPERSEDED_MAX + 3):
            _block(state, now=_NOW + timedelta(hours=i))
        rec = _book(state)["NVDA"]
        assert rec["blocks"] == crew._RECLAIM_SUPERSEDED_MAX + 3
        assert len(rec["superseded_pending"]) == crew._RECLAIM_SUPERSEDED_MAX
        assert rec["superseded_dropped"] == 2
        assert rec["superseded_pending"][-1]["blocked_at"] == (
            _NOW + timedelta(hours=crew._RECLAIM_SUPERSEDED_MAX + 1)).isoformat()

    def test_the_block_path_waits_one_second_for_the_lock_and_the_nightly_five(self, state):
        lock = MagicMock()
        lock.acquire.return_value = True
        redis = MagicMock()
        redis.lock.return_value = lock
        with patch("stop_lock._redis", return_value=redis):
            assert _block(state) is True
        assert redis.lock.call_args.kwargs["blocking_timeout"] == 1.0
        assert "NVDA" in _book(state)
        lock.release.assert_called_once()
        with patch("stop_lock._redis", return_value=redis):
            _night(state, _bars(6))
        assert redis.lock.call_args.kwargs["blocking_timeout"] == 5
        assert crew._RECLAIM_BLOCK_WAIT_S == 1.0 and crew._RECLAIM_LOCK_WAIT_S == 5

    @pytest.mark.parametrize("raw", ["{not json", "[1, 2]", "null", "7"])
    def test_a_corrupt_book_is_reset_with_one_warning(self, state, raw, caplog):
        state[_KEY] = raw
        with caplog.at_level("WARNING"):
            n, la = _night(state, _bars(6))
        assert n == 0 and not la.called
        assert state[_KEY] == "{}"
        assert sum("not a JSON object" in r.getMessage() for r in caplog.records) == 1
        assert _block(state) is True              # the shadow records again
        assert _book(state)["NVDA"]["blocks"] == 1

    def test_a_corrupt_book_is_reset_on_the_block_path_too(self, state, caplog):
        state[_KEY] = "{not json"
        with caplog.at_level("WARNING"):
            assert _block(state) is True
        assert list(_book(state)) == ["NVDA"]
        assert sum("not a JSON object" in r.getMessage() for r in caplog.records) == 1

    def test_nothing_is_stored_when_the_gate_passes_or_the_shadow_is_off(self, state, monkeypatch):
        assert _block(state, trend=_trend(False, False)) is False
        assert _KEY not in state
        monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_ENABLED", False)
        assert _block(state) is True
        assert _KEY not in state

    def test_a_failed_read_writes_nothing(self):
        with patch("db.get_state_strict", side_effect=RuntimeError("db down")), \
             patch("db.set_state") as ss, patch("stop_lock._redis", side_effect=RuntimeError("x")):
            assert crew._reclaim_book_edit(lambda b: {**b, "X": {}}) is False
        ss.assert_not_called()

    def test_a_busy_lock_skips_the_edit(self):
        lock = MagicMock()
        lock.acquire.return_value = False
        redis = MagicMock()
        redis.lock.return_value = lock
        with patch("stop_lock._redis", return_value=redis), \
             patch("db.get_state_strict") as gs, patch("db.set_state") as ss:
            assert crew._reclaim_book_edit(lambda b: b) is False
        gs.assert_not_called(); ss.assert_not_called()


class TestNeverRaisesIntoTheGate:
    def test_a_raising_record_never_lets_the_buy_through(self, state):
        with patch.object(crew, "_entry_gate_reclaim_record", side_effect=RuntimeError("boom")):
            assert _block(state) is True

    def test_a_raising_store_returns_false(self, state):
        with patch.object(crew, "_reclaim_book_edit", side_effect=RuntimeError("boom")):
            assert crew._entry_gate_reclaim_record("NVDA", _result(), {"trend": {"blocked": True}},
                                                   _trend()) is False

    def test_the_nightly_check_swallows_a_failed_read(self):
        with patch("db.get_state_strict", side_effect=RuntimeError("db down")):
            assert crew._check_entry_gate_reclaims() == 0

    def test_the_nightly_check_swallows_a_per_ticker_error(self, state):
        state[_KEY] = json.dumps({"NVDA": {"blocked_at": "x", "et_date": "2026-11-23",
                                            "base_bar_date": "2026-11-20", "base_close": "bad"}})
        with patch("db.get_price_history_bulk", side_effect=lambda *a, **k: {"NVDA": [{"date": "2026-11-23", "close": 1.0}]}), \
             patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.log_activity", side_effect=RuntimeError("write")):
            assert crew._check_entry_gate_reclaims(today=date(2026, 11, 24)) == 0

    def test_the_nightly_hook_runs_beside_the_pending_entries_check(self):
        import tasks
        src = inspect.getsource(tasks.refresh_price_history)
        assert src.index("_check_pending_entries()") < src.index("_check_entry_gate_reclaims()")

    def test_a_raising_reclaim_check_is_swallowed_inside_the_refresh_task(self):
        import tasks
        with patch("ingestion.price_history.refresh_all_price_history", return_value=3), \
             patch.object(tasks, "_heartbeat_ok") as hb, \
             patch.object(tasks, "_heartbeat_fail") as hf, \
             patch.object(tasks, "_check_pending_entries") as pe, \
             patch.object(crew, "_check_entry_gate_reclaims", side_effect=RuntimeError("boom")) as rc, \
             patch.object(tasks, "_log_plan_cross_section") as cs, \
             patch.object(tasks, "_check_tranche2_shadow") as t2, \
             patch.object(tasks, "_check_c6_bar_outcomes") as c6:
            tasks.refresh_price_history.run()       # no raise, no retry
        pe.assert_called_once_with()
        rc.assert_called_once_with()
        hb.assert_called_once_with("price_history", 3)
        hf.assert_not_called()
        cs.assert_called_once(); t2.assert_called_once(); c6.assert_called_once()


# ── nightly: forward moves, reclaim, expiry ──────────────────────────────────

def _bars(n, start=date(2026, 11, 20), base=100.0, step=1.0):
    """The base bar (start) and n sessions after it, closes base + i*step."""
    out, d = [], start
    for i in range(n + 1):
        out.append({"date": d.isoformat(), "open": 1.0, "high": 1.0, "low": 1.0,
                    "close": round(base + i * step, 4), "volume": 1})
        d += timedelta(days=1)
    return out


def _stored(state, gates=("trend",), **kw):
    rec = {"gates": list(gates), "action": "ADD", "signal_id": 4242, "path": None,
           "verdict_at": _NOW.isoformat(), "blocked_at": _NOW.isoformat(),
           "first_blocked_at": _NOW.isoformat(), "blocks": 1, "et_date": "2026-11-23",
           "levels": {}, "base_close": 100.0, "base_bar_date": "2026-11-20", "logged": {}, **kw}
    state[_KEY] = json.dumps({"NVDA": rec})


def _night(state, bars, trend=None, spy=None, done=None, exit_=None, today=date(2026, 12, 30)):
    """One nightly run; returns (rows written, the activity writes)."""
    checks = {"NVDA": trend or _trend(), "SPY": spy or _spy()}
    with patch("db.get_price_history_bulk", return_value={"NVDA": bars}), \
         patch("db.get_activity_shadow_keys", return_value=set() if done is None else done), \
         patch("db.log_activity", return_value=True) as la, \
         patch("db.get_last_broker_exit", return_value=exit_), \
         patch.object(crew, "_entry_trend_check", side_effect=lambda t: checks[t]):
        n = crew._check_entry_gate_reclaims(today=today)
    return n, la


def _events(la):
    return {c.args[4]["event"]: c.args[4] for c in la.call_args_list}


class TestForwardMoves:
    def test_5_10_20_session_moves_are_logged_once_each_in_percent(self, state):
        _stored(state)
        n, la = _night(state, _bars(12))          # still below both: no reclaim
        ev = _events(la)
        assert set(ev) == {"forward_5", "forward_10"}
        assert ev["forward_5"]["move_pct"] == 5.0 and ev["forward_5"]["bar_date"] == "2026-11-25"
        assert ev["forward_10"]["move_pct"] == 10.0 and ev["forward_10"]["sessions"] == 10
        for e in ev.values():
            assert e["shadow_key"].endswith(":" + e["event"]) and e["gates"] == ["trend"]
            assert "$" not in str(e)
        n2, la2 = _night(state, _bars(12))
        assert n2 == 0 and not la2.called          # each move once
        n3, la3 = _night(state, _bars(20))
        assert set(_events(la3)) == {"forward_20", "expired"}
        assert _events(la3)["forward_20"]["move_pct"] == 20.0

    def test_a_row_already_in_activity_feed_is_not_written_again(self, state):
        _stored(state)
        sk = f"{config.TRADING_ENV}:NVDA:{_NOW.isoformat()}:forward_5"
        n, la = _night(state, _bars(6), done={sk})
        assert "forward_5" not in _events(la)
        assert _book(state)["NVDA"]["logged"]["forward_5"] == "already written"

    def test_a_record_without_a_base_takes_the_last_close_before_the_block(self, state):
        _stored(state, base_close=None, base_bar_date=None)
        bars = _bars(12, start=date(2026, 11, 18))  # 11-18 .. 11-30; block ET date 11-23
        n, la = _night(state, bars)
        rec = _book(state)["NVDA"]
        assert rec["base_bar_date"] == "2026-11-22" and rec["base_close"] == 104.0
        assert rec["base_source"].startswith("last cached close before")
        ev = _events(la)["forward_5"]
        assert ev["bar_date"] == "2026-11-27" and ev["move_pct"] == round((109 / 104 - 1) * 100, 2)


class TestReclaim:
    def test_fires_once_when_the_trend_gate_would_pass(self, state):
        _stored(state)
        up = _trend(below50=False, below200=True, bar_date="2026-11-26")
        n, la = _night(state, _bars(6), trend=up)
        ev = _events(la)["reclaim"]
        assert ev["reclaim_date"] == "2026-11-26" and ev["sessions"] == 6 and ev["move_pct"] == 6.0
        assert ev["check"]["pass"] is True and ev["check"]["per_gate"]["trend"]["pass"] is True
        n2, la2 = _night(state, _bars(7), trend=_trend(below50=False, bar_date="2026-11-27"))
        assert "reclaim" not in _events(la2)

    def test_the_market_reclaiming_its_200_day_counts(self, state):
        _stored(state)
        n, la = _night(state, _bars(3), trend=_trend(bar_date="2026-11-23"), spy=_spy(below200=False))
        assert _events(la)["reclaim"]["check"]["per_gate"]["trend"]["market_pct_vs_sma200"] == 3.1

    @pytest.mark.parametrize("trend,spy", [
        (_trend(), _spy()),                                    # still blocked
        (_trend(below50=False, age=9), _spy()),                # stale bars: the gate stands down, not judged
        (_trend(below50=False, read="insufficient_bars"), _spy()),
        (_trend(below50=False, suspect=True), _spy()),         # split-suspect bars
        (_trend(), _spy(read="error: OperationalError")),      # market unreadable
    ])
    def test_no_reclaim_while_blocked_or_unjudged(self, state, trend, spy):
        _stored(state)
        n, la = _night(state, _bars(3), trend=trend, spy=spy)
        assert "reclaim" not in _events(la)

    def test_stopout_reclaims_once_the_newest_exit_is_out_of_the_window(self, state):
        _stored(state, gates=("stopout",))
        n, la = _night(state, _bars(3), exit_={"hours_since": 30.0})
        assert "reclaim" not in _events(la)
        n, la = _night(state, _bars(4), exit_={"hours_since": 241.0})
        assert _events(la)["reclaim"]["check"]["per_gate"]["stopout"]["pass"] is True
        _stored(state, gates=("stopout",))
        n, la = _night(state, _bars(3), exit_=None)      # no exit in the lookback at all
        assert "reclaim" in _events(la)

    def test_both_gates_must_pass(self, state):
        _stored(state, gates=("stopout", "trend"))
        n, la = _night(state, _bars(3), trend=_trend(below50=False), exit_={"hours_since": 30.0})
        assert "reclaim" not in _events(la)

    def test_add_loser_is_never_judged_on_bars(self, state):
        _stored(state, gates=("add_loser",))
        n, la = _night(state, _bars(6))
        ev = _events(la)
        assert "reclaim" not in ev and "forward_5" in ev


class TestGateOffAndOddReads:
    @pytest.mark.parametrize("gates,cfg", [
        (("trend",), ("ENTRY_TREND_GATE_ENABLED", False)),
        (("stopout",), ("STOPOUT_REENTRY_GATE_DAYS", 0)),
        (("stopout", "trend"), ("STOPOUT_REENTRY_GATE_DAYS", 0)),
    ])
    def test_a_gate_turned_off_since_the_block_never_reclaims_and_expires_as_usual(
            self, state, monkeypatch, gates, cfg):
        monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_SESSIONS", 5)
        monkeypatch.setattr(config, *cfg)
        _stored(state, gates=gates)
        chk = crew._reclaim_gates_pass("NVDA", list(gates))
        off = "trend" if cfg[0] == "ENTRY_TREND_GATE_ENABLED" else "stopout"
        assert chk["per_gate"][off] == {"pass": None, "read": "gate off"}
        assert chk["pass"] is None
        n, la = _night(state, _bars(5), trend=_trend(below50=False, below200=False))
        ev = _events(la)
        assert "reclaim" not in ev and set(ev) == {"forward_5", "expired"}
        assert ev["expired"]["reclaimed"] is False
        assert _book(state) == {}

    @pytest.mark.parametrize("hrs", [-1.0, -0.01, "12", None, float("nan"), True])
    def test_a_negative_or_non_numeric_exit_age_is_not_judged(self, state, hrs):
        with patch("db.get_last_broker_exit", return_value={"hours_since": hrs}):
            chk = crew._reclaim_gates_pass("NVDA", ["stopout"])
        assert chk["per_gate"]["stopout"]["pass"] is None and chk["pass"] is None
        _stored(state, gates=("stopout",))
        n, la = _night(state, _bars(3), exit_={"hours_since": hrs})
        assert "reclaim" not in _events(la)


class TestSuperseded:
    def _two_blocks(self, state):
        """Block 11-23 (base 11-20); one night logs its forward_5; a newer block on 11-27."""
        _block(state)
        _night(state, _bars(5), today=date(2026, 11, 25))
        assert "forward_5" in _book(state)["NVDA"]["logged"]
        later = _NOW + timedelta(days=4)
        _block(state, trend=_trend(last_close=106.0, bar_date="2026-11-26"), now=later,
               result=_result(sid=5151))
        return later

    def test_the_replaced_record_gets_one_superseded_row_with_what_it_had(self, state):
        later = self._two_blocks(state)
        n, la = _night(state, _bars(12), today=date(2026, 12, 2))
        sup = [c.args for c in la.call_args_list if c.args[4]["event"] == "superseded"]
        assert len(sup) == 1
        stage, ticker, msg, sent, x = sup[0]
        assert stage == "entry_gate_reclaim_shadow" and ticker == "NVDA" and "record only" in msg
        assert x["shadow_key"] == f"{config.TRADING_ENV}:NVDA:{_NOW.isoformat()}:superseded"
        assert x["blocked_at"] == _NOW.isoformat() and x["gates"] == ["trend"]
        assert x["signal_id"] == 4242 and x["base_close"] == 100.0
        assert x["base_bar_date"] == "2026-11-20"
        assert x["base_source"] == "verdict-time last completed close"
        assert x["forward_logged"] == ["forward_5"] and x["reclaimed"] is False
        assert x["sessions_elapsed"] == 6          # 11-21 .. 11-26, before the 11-27 block
        assert x["superseded_at"] == later.isoformat() and x["superseded_et_date"] == "2026-11-27"
        assert x["first_blocked_at"] == _NOW.isoformat() and x["blocks"] == 2
        assert "$" not in str(x)
        rec = _book(state)["NVDA"]
        assert "superseded_pending" not in rec
        assert rec["first_blocked_at"] == _NOW.isoformat() and rec["blocks"] == 2
        assert rec["blocked_at"] == later.isoformat()
        n2, la2 = _night(state, _bars(12), today=date(2026, 12, 2))
        assert "superseded" not in {c.args[4]["event"] for c in la2.call_args_list}

    def test_a_superseded_row_already_in_activity_feed_is_not_written_again(self, state):
        self._two_blocks(state)
        sk = f"{config.TRADING_ENV}:NVDA:{_NOW.isoformat()}:superseded"
        n, la = _night(state, _bars(12), done={sk}, today=date(2026, 12, 2))
        assert "superseded" not in {c.args[4]["event"] for c in la.call_args_list}
        assert "superseded_pending" not in _book(state)["NVDA"]

    def test_a_failed_superseded_write_stays_pending_and_holds_the_expiry(self, state, monkeypatch):
        self._two_blocks(state)
        monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_SESSIONS", 5)
        with patch("db.get_price_history_bulk", return_value={"NVDA": _bars(12)}), \
             patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.log_activity", side_effect=lambda *a: a[4]["event"] != "superseded"), \
             patch.object(crew, "_entry_trend_check", side_effect=lambda t: _trend()):
            crew._check_entry_gate_reclaims(today=date(2026, 12, 2))
        rec = _book(state)["NVDA"]
        assert len(rec["superseded_pending"]) == 1 and rec["logged"].get("expired")
        n, la = _night(state, _bars(12), today=date(2026, 12, 3))
        assert [c.args[4]["event"] for c in la.call_args_list] == ["superseded"]
        assert _book(state) == {}                    # written, then dropped as expired

    def test_a_record_without_a_base_takes_it_from_the_bars(self, state):
        _stored(state, base_close=None, base_bar_date=None, superseded_pending=[
            {"blocked_at": "2026-11-18T15:00:00+00:00", "gates": ["stopout"], "et_date": "2026-11-18",
             "base_close": None, "base_bar_date": None, "logged": {},
             "superseded_at": _NOW.isoformat(), "superseded_et_date": "2026-11-23"}])
        n, la = _night(state, _bars(12, start=date(2026, 11, 14)))
        x = _events(la)["superseded"]
        assert x["base_bar_date"] == "2026-11-17" and x["base_close"] == 103.0
        assert x["base_source"].startswith("last cached close before")
        assert x["sessions_elapsed"] == 5             # 11-18 .. 11-22


class TestExpiry:
    def test_expires_after_the_configured_sessions_and_drops_the_record(self, state, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_SESSIONS", 5)
        _stored(state)
        n, la = _night(state, _bars(5))
        ev = _events(la)
        assert set(ev) == {"forward_5", "expired"}
        assert ev["expired"]["window_complete"] is True and ev["expired"]["reclaimed"] is False
        assert _book(state) == {}

    def test_a_record_whose_bars_never_arrive_expires_incomplete(self, state):
        _stored(state)
        n, la = _night(state, [], today=date(2027, 2, 1))
        ev = _events(la)["expired"]
        assert ev["window_complete"] is False and ev["sessions"] == 0
        assert _book(state) == {}

    def test_a_failed_expired_write_keeps_the_record_for_the_next_night(self, state, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_SESSIONS", 5)
        _stored(state)
        with patch("db.get_price_history_bulk", return_value={"NVDA": _bars(5)}), \
             patch("db.get_activity_shadow_keys", return_value=set()), \
             patch("db.log_activity", return_value=False), \
             patch.object(crew, "_entry_trend_check", side_effect=lambda t: _trend()):
            crew._check_entry_gate_reclaims(today=date(2026, 12, 30))
        assert "NVDA" in _book(state)

    def test_a_newer_block_during_the_night_is_kept(self, state, monkeypatch):
        monkeypatch.setattr(config, "ENTRY_GATE_RECLAIM_SHADOW_SESSIONS", 5)
        _stored(state)
        newer = {**_book(state)["NVDA"], "blocked_at": "2026-12-29T15:00:00+00:00"}
        real = crew._reclaim_book_edit

        def edit(fn):     # a block lands between the night's read and its write
            state[_KEY] = json.dumps({"NVDA": newer})
            return real(fn)
        with patch.object(crew, "_reclaim_book_edit", side_effect=edit):
            _night(state, _bars(5))
        assert _book(state)["NVDA"]["blocked_at"] == "2026-12-29T15:00:00+00:00"


# ── the gate's own rules, shared; nothing the analyst sees changes ──────────

@pytest.mark.parametrize("s,days,hit", [
    ({"read": "ok", "hours_since": 0.5}, 10, True), ({"read": "ok", "hours_since": 240.0}, 10, False),
    ({"read": "ok", "hours_since": -1.0}, 10, False), ({"read": "error: X", "hours_since": 1.0}, 10, False),
    ({"read": "ok", "hours_since": True}, 10, False), ({"read": "ok"}, 10, False),
])
def test_stopout_rule_is_the_gates(s, days, hit):
    assert crew._stopout_gate_hit(s, days) is hit


def test_template_shas_unchanged():
    from agents import analyst
    assert analyst._template_shas() == {
        "addon_template_sha": "ea7b89255103", "brief_template_full_sha": "a994de527dcd",
        "brief_template_sha": "2d7efec589d0", "prompt_db_queries_sha": "1ce9c6ccff32",
        "prompt_template_full_sha": "c553694b23b4", "prompt_template_sha": "6b93fe2037b7"}
