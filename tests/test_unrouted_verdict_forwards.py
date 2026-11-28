"""Forward closes after non-routed verdicts (agents/crew.py:
_unrouted_verdict_forward_record, _check_unrouted_verdict_forwards). Record
only: the day's first non-routed verdict per ticker is stored with a count of
the day's later ones; the nightly check logs the close 5/10/20 sessions after
the verdict-time last completed close, then expires the record. Synthetic
data only: system_state, the bars and the activity writes are in memory; no
DB, broker or Redis."""
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
from agents import crew

_REAL_BOOK_EDIT = crew._unrouted_forward_book_edit   # conftest stubs it per test; restored below
_KEY = "unrouted_verdict_forward_shadow_" + config.TRADING_ENV
_MON_1040 = datetime(2026, 11, 23, 15, 40, tzinfo=timezone.utc)   # Mon 10:40 ET
_MON_1630 = datetime(2026, 11, 23, 21, 30, tzinfo=timezone.utc)   # Mon 16:30 ET, after the close


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(config, "UNROUTED_VERDICT_FORWARD_SHADOW_ENABLED", True)
    monkeypatch.setattr(crew, "_unrouted_forward_book_edit", _REAL_BOOK_EDIT)


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


def _verdict(action="WATCH", ticker="NVDA", conviction=5, sid=4242, path="daily_scan"):
    return {"ticker": ticker, "action": action, "conviction": conviction, "conviction_raw": conviction,
            "pass_to_risk": False,
            "provenance": {"signal_id": sid, "trigger": {"path": path, "source": "scanner"}}}


def _rec(st, result, now=_MON_1040):
    return crew._unrouted_verdict_forward_record(result, now=now)


# ── storing ──────────────────────────────────────────────────────────────────

class TestStore:
    def test_the_days_first_non_routed_verdict_is_kept_and_later_ones_counted(self, state):
        assert _rec(state, _verdict("WATCH", conviction=5)) is True
        assert _rec(state, _verdict("HOLD", conviction=7, sid=4243), now=_MON_1040 + timedelta(hours=2))
        assert _rec(state, _verdict("WATCH", sid=4244), now=_MON_1040 + timedelta(hours=3))
        book = _book(state)
        assert list(book) == ["NVDA:2026-11-23"]
        r = book["NVDA:2026-11-23"]
        assert (r["action"], r["conviction"], r["signal_id"], r["path"]) == ("WATCH", 5, 4242, "daily_scan")
        assert r["verdict_at"] == _MON_1040.isoformat() and r["et_date"] == "2026-11-23"
        assert r["verdicts"] == 3 and r["actions_seen"] == {"WATCH": 2, "HOLD": 1}

    def test_a_new_et_date_or_another_ticker_is_its_own_record(self, state):
        _rec(state, _verdict())
        _rec(state, _verdict(ticker="AMD"))
        _rec(state, _verdict(), now=_MON_1040 + timedelta(days=1))
        assert sorted(_book(state)) == ["AMD:2026-11-23", "NVDA:2026-11-23", "NVDA:2026-11-24"]

    def test_the_base_session_is_the_last_close_completed_at_verdict_time(self, state):
        _rec(state, _verdict(), now=_MON_1040)                        # Monday intraday: Friday
        _rec(state, _verdict(ticker="AMD"), now=_MON_1630)            # Monday after the close
        _rec(state, _verdict(ticker="MU"), now=datetime(2026, 11, 22, 15, tzinfo=timezone.utc))   # Sunday
        _rec(state, _verdict(ticker="KLAC"),                          # the early close: done at 13:00
             now=datetime(2026, 11, 27, 18, 5, tzinfo=timezone.utc))
        _rec(state, _verdict(ticker="ORCL"),                          # Thanksgiving: Wednesday
             now=datetime(2026, 11, 26, 18, tzinfo=timezone.utc))
        b = _book(state)
        assert b["NVDA:2026-11-23"]["base_session"] == "2026-11-20"
        assert b["AMD:2026-11-23"]["base_session"] == "2026-11-23"
        assert b["MU:2026-11-22"]["base_session"] == "2026-11-20"
        assert b["KLAC:2026-11-27"]["base_session"] == "2026-11-27"
        assert b["ORCL:2026-11-26"]["base_session"] == "2026-11-25"

    def test_placeholders_and_the_switch_store_nothing(self, state, monkeypatch):
        for r in ({"ticker": "NVDA", "action": "SKIP", "conviction": 0}, {"ticker": "NVDA"},
                  {"action": "WATCH"}, None, "x"):
            assert _rec(state, r) is False
        monkeypatch.setattr(config, "UNROUTED_VERDICT_FORWARD_SHADOW_ENABLED", False)
        assert _rec(state, _verdict()) is False
        assert _book(state) == {}

    def test_the_book_is_capped_oldest_verdict_first(self, state, monkeypatch):
        monkeypatch.setattr(crew, "_UVF_BOOK_MAX", 3)
        for i, t in enumerate(("A", "B", "C", "D")):
            _rec(state, _verdict(ticker=t), now=_MON_1040 + timedelta(minutes=i))
        assert sorted(_book(state)) == ["B:2026-11-23", "C:2026-11-23", "D:2026-11-23"]

    def test_failures_never_raise(self, state):
        with patch("db.get_state_strict", side_effect=RuntimeError("db down")):
            assert _rec(state, _verdict()) is False
        state[_KEY] = "[1, 2]"                              # not a JSON object: reset
        assert _rec(state, _verdict()) is True and list(_book(state)) == ["NVDA:2026-11-23"]
        with patch.object(crew, "_last_completed_session_date", side_effect=ValueError("cal")):
            assert _rec(state, _verdict(ticker="AMD")) is False

    def test_a_busy_lock_skips_the_edit(self, state):
        class _Lock:
            def acquire(self):
                return False
        with patch("stop_lock._redis") as r:
            r.return_value.lock.return_value = _Lock()
            assert _rec(state, _verdict()) is False
        assert _book(state) == {}


def test_every_non_routed_verdict_reaches_the_record():
    with patch("agents.c6_shadow.observe_unrouted", side_effect=RuntimeError("c6 down")), \
         patch.object(crew, "_unrouted_verdict_forward_record") as rec:
        crew._c6_bar_unrouted(_verdict(), {"regime": "Neutral"}, None)
    rec.assert_called_once()
    with patch("agents.c6_shadow.observe_unrouted"), \
         patch.object(crew, "_unrouted_verdict_forward_record", side_effect=RuntimeError("x")):
        crew._c6_bar_unrouted(_verdict(), {"regime": "Neutral"}, None)     # never raises


# ── the nightly check ────────────────────────────────────────────────────────

def _sessions_from(start, n):
    import market_data
    out, d = [], start
    while len(out) < n:
        if market_data._session(d) is not None:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _bars(dates, base=100.0, step=1.0):
    return [{"date": d, "close": round(base + i * step, 2)} for i, d in enumerate(dates)]


def _check(state, bars, today, done=None):
    rows = []

    def log_activity(stage, ticker, msg, tone, extra):
        rows.append({"stage": stage, "ticker": ticker, "msg": msg, **extra})
        return True
    with patch("db.get_price_history_bulk", return_value=bars) as bulk, \
         patch("db.get_activity_shadow_keys", return_value=set(done or ())), \
         patch("db.log_activity", side_effect=log_activity):
        n = crew._check_unrouted_verdict_forwards(today=today)
    return n, rows, bulk


class TestCheck:
    def test_forward_rows_on_completed_bars_after_the_base_then_expiry(self, state):
        _rec(state, _verdict("HOLD", conviction=6))         # base 2026-11-20
        _rec(state, _verdict("WATCH"), now=_MON_1040 + timedelta(hours=1))
        dates = _sessions_from(date(2026, 11, 18), 30)
        bars = {"NVDA": _bars(dates)}
        base_i = dates.index("2026-11-20")
        n, rows, bulk = _check(state, {"NVDA": _bars(dates[:base_i + 6])}, date(2026, 12, 1))
        assert bulk.call_args.args[0] == ["NVDA"]
        assert [r["event"] for r in rows] == ["forward_5"]
        r = rows[0]
        assert r["stage"] == "unrouted_verdict_forward_shadow" and n == 1
        assert r["base_bar_date"] == "2026-11-20" and r["base_bar_matches"] is True
        assert r["bar_date"] == dates[base_i + 5] and r["sessions"] == 5
        assert r["move_pct"] == round(((100 + base_i + 5) / (100 + base_i) - 1) * 100, 2)
        assert (r["action"], r["conviction"], r["verdicts"], r["actions_seen"]) == \
            ("HOLD", 6, 2, {"HOLD": 1, "WATCH": 1})
        assert r["shadow_key"] == f"{config.TRADING_ENV}:NVDA:2026-11-23:forward_5"
        assert "$" not in r["msg"]
        n, rows, _ = _check(state, bars, date(2026, 12, 30))
        assert [r["event"] for r in rows] == ["forward_10", "forward_20", "expired"]
        assert rows[-1]["window_complete"] is True and rows[-1]["forward_logged"] == \
            ["forward_10", "forward_20", "forward_5"]
        assert _book(state) == {}                          # dropped after expiry
        assert _check(state, bars, date(2026, 12, 31))[0] == 0

    def test_no_base_bar_yet_waits_and_a_gap_takes_the_last_close_before(self, state):
        _rec(state, _verdict(), now=_MON_1630)              # base 2026-11-23
        n, rows, _ = _check(state, {"NVDA": _bars(["2026-11-20"])}, date(2026, 11, 23))
        assert n == 0 and "base_bar_date" not in _book(state)["NVDA:2026-11-23"]
        dates = ["2026-11-20"] + _sessions_from(date(2026, 11, 24), 6)    # 11-23 missing
        n, rows, _ = _check(state, {"NVDA": _bars(dates)}, date(2026, 12, 3))
        assert rows[0]["base_bar_date"] == "2026-11-20" and rows[0]["base_bar_matches"] is False

    def test_a_name_without_bars_expires_incomplete(self, state):
        _rec(state, _verdict(ticker="ZZZZ"))
        assert _check(state, {}, date(2026, 12, 20))[0] == 0
        n, rows, _ = _check(state, {}, date(2027, 1, 20))
        assert [r["event"] for r in rows] == ["expired"]
        assert rows[0]["window_complete"] is False and rows[0]["base_bar_date"] is None
        assert _book(state) == {}

    def test_rows_already_written_are_not_written_again(self, state):
        _rec(state, _verdict())
        dates = _sessions_from(date(2026, 11, 20), 8)
        sk = f"{config.TRADING_ENV}:NVDA:2026-11-23:forward_5"
        n, rows, _ = _check(state, {"NVDA": _bars(dates)}, date(2026, 12, 2), done={sk})
        assert n == 0 and rows == []
        assert _book(state)["NVDA:2026-11-23"]["logged"]["forward_5"] == "already written"

    def test_verdicts_counted_during_the_check_are_kept(self, state):
        _rec(state, _verdict())
        dates = _sessions_from(date(2026, 11, 20), 8)
        real_edit = crew._unrouted_forward_book_edit
        first = [True]

        def edit_after_a_new_verdict(fn, wait_s=None):
            if first[0]:      # the check's own edit: a verdict lands first
                first[0] = False
                _rec(state, _verdict("HOLD"), now=_MON_1040 + timedelta(hours=5))
            return real_edit(fn)
        with patch.object(crew, "_unrouted_forward_book_edit", side_effect=edit_after_a_new_verdict):
            _check(state, {"NVDA": _bars(dates)}, date(2026, 12, 2))
        r = _book(state)["NVDA:2026-11-23"]
        assert r["verdicts"] == 2 and r["logged"]["forward_5"] == "2026-12-02"

    def test_off_empty_or_failing_never_raises(self, state, monkeypatch):
        assert _check(state, {}, date(2026, 12, 2))[0] == 0          # empty book
        _rec(state, _verdict())
        with patch("db.get_price_history_bulk", side_effect=RuntimeError("db")):
            assert crew._check_unrouted_verdict_forwards(today=date(2026, 12, 2)) == 0
        state[_KEY] = '"x"'
        assert crew._check_unrouted_verdict_forwards(today=date(2026, 12, 2)) == 0
        assert _book(state) == {}                                    # reset to {}
        monkeypatch.setattr(config, "UNROUTED_VERDICT_FORWARD_SHADOW_ENABLED", False)
        with patch("db.get_state_strict") as g:
            assert crew._check_unrouted_verdict_forwards() == 0
        g.assert_not_called()


def test_the_nightly_refresh_runs_the_check_after_the_reclaim_check():
    import tasks
    src = inspect.getsource(tasks.refresh_price_history)
    assert src.index("_check_entry_gate_reclaims()") < src.index("_check_unrouted_verdict_forwards()")


def test_record_only():
    src = inspect.getsource(crew._unrouted_verdict_forward_record) + \
        inspect.getsource(crew._check_unrouted_verdict_forwards)
    for word in ("threshold", "score", "evaluate(", "route", "approval", "ibkr", "place_"):
        assert word not in src.replace("routed", "")
