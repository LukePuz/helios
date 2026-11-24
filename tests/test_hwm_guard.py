"""Three record-first checks (portfolio-risk follow-up d5/A3 and the
data-pipeline gap hunt's smaller items, 2026-11-21):

1. RiskManager.is_halted with an unusable stored high-water mark
   (config.HWM_GUARD_MODE): recorded and alerted; in shadow each case does
   exactly what it did at ad52500 (checked side by side).
2. tasks.refresh_fundamentals' change count (config FUNDAMENTALS_FREEZE_*):
   targets and forward EPS unchanged for all 75 tickers 10-26..10-29, unflagged.
3. source_yield's per-feed news breakdown: yfinance news was down 11-09/11-10
   while google_news kept the "news" source level."""
import os
import re
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import alert_policy
import config
import risk_manager
import tasks
from risk_manager import RiskManager

ENV = risk_manager.TRADING_ENV
ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).date().isoformat()


# ---- 1. the high-water-mark guard ----

def _rm(pv=1000.0, start=None, simulation=False):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=pv)
    return RiskManager(client, start_value=start, simulation=simulation)


class _SyncThread:
    """threading.Thread stand-in: runs the target at start(), so the alert
    the guard hands to a daemon thread is visible as soon as is_halted returns."""
    made = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.target, self.args, self.kwargs, self.daemon = target, args, kwargs or {}, daemon
        _SyncThread.made.append(self)

    def start(self):
        self.target(*self.args, **self.kwargs)


def _halt(rm, hwm, state=None, mode="shadow", strict_error=None, verify_fails=False,
          sync=True):
    """is_halted with the stored mark hwm (None: missing). state: the plain
    system_state (alert dedupe), shared across calls. Returns (result or the
    exception raised, rec, state)."""
    state = {} if state is None else state
    strict = {f"high_water_mark_{ENV}": hwm} if hwm is not None else {}
    rec = MagicMock()

    def verified(writes):
        rec.set_state_verified(writes)
        if verify_fails:
            return [k for k, _ in writes]
        for k, v in writes:
            state[k] = v
        return []

    def set_state(k, v):
        rec.set_state(k, v)
        state[k] = v
    with patch.object(config, "HWM_GUARD_MODE", mode), \
         patch("db.get_state_strict", side_effect=lambda k, d=None: strict.get(k, d)), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=set_state), \
         patch("db.set_state_verified", side_effect=verified), \
         patch("db.log_halt_event", rec.log_halt_event), \
         patch("db.log_activity_queued", rec.row), \
         patch("alerts._send", rec.send), \
         patch.object(risk_manager.threading, "Thread", _SyncThread) if sync else _nullctx():
        if strict_error:
            rec.row.side_effect = strict_error
        try:
            out = rm.is_halted()
        except Exception as e:
            out = e
    return out, rec, state


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rows(rec):
    return [c.args for c in rec.row.call_args_list if c.args[0] == "hwm_guard"]


def _latched(rec):
    return any(k == f"trading_halted_{ENV}" for c in rec.set_state_verified.call_args_list
               for k, _ in c.args[0])


def _hwm_writes(rec):
    return [c.args for c in rec.set_state.call_args_list
            if c.args[0] == f"high_water_mark_{ENV}"]


def _no_money(text):
    assert "$" not in text and not re.search(r"\d{3,}", text.replace(ENV, "")), text


# RiskManager.is_halted exactly as it was at ad52500 (git show
# ad52500:risk_manager.py), frozen here so a later edit to the latch, the
# plausibility guard or the ratchet shows up as a difference rather than
# being copied into the reference. Do not edit it.
_AD52500_IS_HALTED = r'''
    def is_halted(self) -> bool:
        """
        Halt if (a) the persisted manual/auto halt flag is set, or
        (b) drawdown from the persisted high-water mark exceeds the limit.
        Baselines live in the system_state table so every process agrees.

        Keys are trading_env-scoped (high_water_mark_paper vs
        high_water_mark_live, etc.) -- found 2026-08-14: a shared
        unscoped "high_water_mark" key held paper's ~$1.09M baseline, and
        reconnecting to the live account (~$2k) would have hit the
        implausible-value guard below on EVERY check, permanently
        (not just on one transient bad read, which is what that guard is
        actually for) -- silently disabling the drawdown halt, the single
        most important safety mechanism in the system, for live trading.
        """
        self.halt_unknown = False
        # Persisted halt flag (set manually via /halt or by the drawdown task)
        try:
            import db
            if db.get_state_strict(f"trading_halted_{TRADING_ENV}", "false") == "true":
                log.warning(f"HALT — trading_halted_{TRADING_ENV} flag is set in system_state")
                return True
            hwm_str = db.get_state_strict(f"high_water_mark_{TRADING_ENV}")
        except Exception as e:
            # Fail closed (audit A1031-1053-2): an unread flag may be a /halt,
            # and the fallback baseline is the account value at construction,
            # so the drawdown would read as ~0. Not latched: this call only.
            log.warning(f"Halt state read failed ({e}) — treating as halted for this check")
            self.halt_unknown = True
            return True

        current  = self.portfolio_value()
        baseline = float(hwm_str) if hwm_str else self.start_value
        if baseline <= 0:
            return False

        # Plausibility guard: a transient broker read of $0 (clientId
        # collision, cold-morning connection race) would otherwise compute
        # as a 100% drawdown and latch trading_halted until a manual
        # /resume. An implausible read blocks this one buy without latching
        # (A1031-1053-2): sizing on it would be wrong anyway, and a
        # persistent one halts via check_portfolio_drawdown.
        if current <= 0 or current < 0.5 * baseline:
            log.warning(f"Implausible portfolio value ${current:,.2f} "
                       f"(baseline ${baseline:,.2f}) — blocking this buy, not latching a halt")
            self.halt_unknown = True
            return True

        drawdown = (baseline - current) / baseline
        if drawdown >= DRAWDOWN_HALT_PCT and self.simulation:
            return True
        if drawdown >= DRAWDOWN_HALT_PCT:
            log.warning(
                f"DRAWDOWN HALT — drawdown {drawdown:.1%} exceeds {DRAWDOWN_HALT_PCT:.0%} "
                f"(high-water=${baseline:,.2f}, current=${current:,.2f})"
            )
            halt_saved = False
            try:
                import db
                not_saved = db.set_state_verified([
                    (f"trading_halted_{TRADING_ENV}", "true"),
                    (f"halt_reason_{TRADING_ENV}", f"Auto: drawdown {drawdown:.1%} from HWM ${baseline:,.2f}")])
                halt_saved = f"trading_halted_{TRADING_ENV}" not in not_saved
                # Marked when the flag couldn't be confirmed saved (M149); this buy is blocked either way.
                db.log_halt_event("auto_halt", "per_buy_check", "drawdown",
                                  drawdown_pct=round(drawdown, 4), verified=halt_saved)
            except Exception:
                pass
            # The latch used to be silent: only the 30-min task alerted, at its
            # own MAX_DRAWDOWN_PCT. One alert per latch: a set flag returns
            # above, before this branch. Percentages only. Never raises.
            try:
                import alerts
                alerts._send(
                    f"🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b> ({TRADING_ENV}, per-buy check)\n"
                    f"Drawdown: -{drawdown:.1%} from the high-water mark "
                    f"(limit {DRAWDOWN_HALT_PCT:.0%}).\n"
                    f"No new buys. Sells and stops are unaffected. /resume clears the flag, "
                    f"but each buy re-checks and halts again while the drawdown is at or past "
                    f"the limit."
                    + ("" if halt_saved else "\n⚠️ The halt flag could not be confirmed saved; "
                       "check /status and send /halt if it isn't set.")
                )
            except Exception as e:
                log.warning(f"Per-buy halt alert failed: {e}")
            return True

        # Ratchet the high-water mark up as the portfolio grows
        try:
            import db
            if current > baseline and not self.simulation:
                db.set_state(f"high_water_mark_{TRADING_ENV}", str(current))
        except Exception:
            pass
        return False
'''


def _is_halted_ad52500():
    """The frozen ad52500 is_halted, exec'd in risk_manager's namespace."""
    import textwrap
    ns = {}
    exec(textwrap.dedent(_AD52500_IS_HALTED), dict(vars(risk_manager)), ns)
    return ns["is_halted"]


def test_the_frozen_reference_is_ad52500_verbatim():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        src = subprocess.run(["git", "-C", root, "show", "ad52500:risk_manager.py"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        pytest.skip("git unavailable")
    if src.returncode != 0:
        pytest.skip("ad52500 not in this clone")
    text = src.stdout
    a = text.index("    def is_halted(self) -> bool:")
    b = text.index("    # ---", a)
    assert _AD52500_IS_HALTED.lstrip("\n") == text[a:b].rstrip() + "\n"


# Account cases: (reads of the account value: at construction when no start
# is given, then is_halted's read; the explicit start value or None).
_ACCOUNTS = {
    "flat": ([1000.0], 1000.0),
    "drawdown_11pct": ([890.0], 1000.0),
    "new_high": ([1200.0], 1000.0),
    "start_0_falls_to_account": ([1000.0], 0.0),
    "nan_start": ([1000.0], float("nan")),
    "inf_start": ([1000.0], float("inf")),
    "ticks_up_between_reads": ([1000.0, 1010.0], None),
    "ticks_down_between_reads": ([1000.0, 990.0], None),
    "explicit_start_15pct_drawdown": ([850.0], 1000.0),
}
_MARKS = ["abc", "1,000", "inf", "-inf", "nan", "0", "-5", "1000", None,
          "", "NaN", "1e999", " 1800 "]


def _rm_reads(reads, start, simulation=False):
    client = MagicMock()
    seq = list(reads)

    def account():
        v = seq.pop(0) if len(seq) > 1 else seq[0]
        return MagicMock(portfolio_value=v)
    client.get_account.side_effect = account
    return RiskManager(client, start_value=start, simulation=simulation)


def _full_outcome(out, rm, rec):
    """Everything is_halted decides: the result (or the exception type),
    halt_unknown, every latch write (keys and values), every high-water-mark
    write, the halt events and the drawdown alerts (the guard's own alert and
    dedupe key left out)."""
    kind = type(out).__name__ if isinstance(out, Exception) else out
    latch = [c.args[0] for c in rec.set_state_verified.call_args_list
             if not any(k.startswith("hwm_guard_alerted_") for k, _ in c.args[0])]
    sends = [c.args for c in rec.send.call_args_list
             if "HIGH-WATER MARK UNUSABLE" not in c.args[0]]
    return (kind, rm.halt_unknown, latch, _hwm_writes(rec),
            [(c.args, c.kwargs) for c in rec.log_halt_event.call_args_list], sends)


@pytest.mark.parametrize("simulation", [False, True], ids=["live", "simulation"])
@pytest.mark.parametrize("hwm", _MARKS, ids=lambda m: repr(m))
@pytest.mark.parametrize("account", list(_ACCOUNTS))
def test_shadow_does_exactly_what_ad52500_did(hwm, account, simulation):
    """Allowed, blocked or raised, the same halt_unknown, latch writes,
    high-water-mark writes, halt events and drawdown alerts as the frozen
    ad52500 is_halted, for each stored mark (None: missing) and account case."""
    reads, start = _ACCOUNTS[account]
    rm_new, rm_old = _rm_reads(reads, start, simulation), _rm_reads(reads, start, simulation)
    new, rec_new, _ = _halt(rm_new, hwm)
    with patch.object(RiskManager, "is_halted", _is_halted_ad52500()):
        old, rec_old, _ = _halt(rm_old, hwm)
    assert _full_outcome(new, rm_new, rec_new) == _full_outcome(old, rm_old, rec_old)
    if hwm in ("abc", "1,000"):
        assert isinstance(new, ValueError)                          # raises, as before
    if hwm in ("inf", "1e999"):
        assert new is True and rm_new.halt_unknown                  # implausible: blocked
    if hwm in ("-inf", "nan", "NaN", "0", "-5"):
        assert new is False and not _hwm_writes(rec_new)


def test_the_cases_reach_every_branch():
    """The account cases cover the latch, the ratchet and the plausibility
    block, so the comparison is not only of allowed buys."""
    seen = set()
    for account, (reads, start) in _ACCOUNTS.items():
        for hwm in _MARKS:
            rm = _rm_reads(reads, start)
            with patch.object(RiskManager, "is_halted", _is_halted_ad52500()):
                out, rec, _ = _halt(rm, hwm)
            if isinstance(out, Exception):
                seen.add("raises")
            elif _latched(rec):
                seen.add("latch")
            elif _hwm_writes(rec):
                seen.add("ratchet")
            elif out is True:
                seen.add("blocked")
            else:
                seen.add("allowed")
    assert seen == {"raises", "latch", "ratchet", "blocked", "allowed"}


@pytest.mark.parametrize("hwm", ["0", "-5"])
def test_non_positive_in_shadow_allows_the_buy_as_before_and_records(hwm):
    rm = _rm(890.0)
    out, rec, _ = _halt(rm, hwm)
    assert out is False and rm.halt_unknown is False                 # as before
    (stage, ticker, msg, sentiment, extra), = _rows(rec)
    assert extra == {"reason": "non_positive", "env": ENV, "mode": "shadow",
                     "action": "as_before", "stored": True}
    assert sentiment == "neutral"
    text = rec.send.call_args.args[0]
    assert "HIGH-WATER MARK UNUSABLE" in text and "does not block buys, as before" in text
    _no_money(text), _no_money(msg)
    assert not _latched(rec)


@pytest.mark.parametrize("hwm", ["abc", "1,000"])
def test_unparseable_in_shadow_records_and_raises_as_before(hwm):
    """A stored mark that is not a number raises out of is_halted, as it did
    at ad52500, so every caller fails closed; the guard only records it."""
    rm = _rm(1000.0, 1000.0)
    out, rec, _ = _halt(rm, hwm)
    assert isinstance(out, ValueError)
    extra = _rows(rec)[0][4]
    assert extra["reason"] == "unparseable" and extra["action"] == "as_before"
    text = rec.send.call_args.args[0]
    assert "is not a number" in text and "fails closed, as before" in text
    _no_money(text)
    assert not _hwm_writes(rec) and not _latched(rec)


@pytest.mark.parametrize("hwm, blocked, says", [
    ("inf", True, "implausible and blocks buys, as before"),
    ("-inf", False, "does not block buys, as before"),
    ("nan", False, "cannot compare against it and does not block buys, as before")])
def test_non_finite_in_shadow_records_and_runs_the_old_path(hwm, blocked, says):
    rm = _rm(1000.0, 1000.0)
    out, rec, _ = _halt(rm, hwm)
    assert out is blocked and rm.halt_unknown is blocked
    rows = _rows(rec)
    assert [r[4]["reason"] for r in rows] == ["non_finite"]          # -inf: not also non_positive
    text = rec.send.call_args.args[0]
    assert "is not finite" in text and says in text
    assert not _hwm_writes(rec) and not _latched(rec)


def test_no_stored_mark_and_a_non_finite_start_value_says_so():
    rm = _rm(1000.0, float("nan"))
    out, rec, _ = _halt(rm, None)
    assert out is False
    assert _rows(rec)[0][4]["stored"] is False
    text = rec.send.call_args.args[0]
    assert "No high-water mark is stored, and the start value it falls back to is not finite" in text
    assert "stored high-water mark is" not in text


@pytest.mark.parametrize("hwm, reason", [("0", "non_positive"), ("abc", "unparseable"),
                                         ("1,000", "unparseable"), ("inf", "non_finite"),
                                         ("-inf", "non_finite"), ("nan", "non_finite")])
def test_enforce_blocks_this_buy_without_latching(hwm, reason):
    rm = _rm(1000.0)
    out, rec, _ = _halt(rm, hwm, mode="enforce")
    assert out is True and rm.halt_unknown is True
    assert not _latched(rec) and not rec.log_halt_event.called and not _hwm_writes(rec)
    extra = _rows(rec)[0][4]
    assert extra["reason"] == reason and extra["action"] == "blocked"
    assert _rows(rec)[0][3] == "negative"
    assert "This buy was blocked (not latched)" in rec.send.call_args.args[0]


@pytest.mark.parametrize("mode", ["ENFORCE", " Enforce "])
def test_the_mode_is_read_case_insensitively(mode):
    out, _, _ = _halt(_rm(1000.0), "0", mode=mode)
    assert out is True


def test_an_unknown_mode_logs_and_runs_as_shadow(caplog):
    with caplog.at_level("WARNING"):
        out, rec, _ = _halt(_rm(1000.0), "0", mode="enforced")
    assert out is False and _rows(rec)[0][4]["mode"] == "shadow"
    assert "is not shadow or enforce" in caplog.text


def test_one_alert_per_reason_per_day():
    state = {}
    _, r1, _ = _halt(_rm(), "0", state)
    _, r2, _ = _halt(_rm(), "0", state)                              # same reason, same day
    _, r3, _ = _halt(_rm(), "abc", state)                            # another reason
    assert r1.send.call_count == 1 and r2.send.call_count == 0 and r3.send.call_count == 1
    assert len(_rows(r2)) == 1                                        # the row is still written
    assert state == {f"hwm_guard_alerted_{ENV}_non_positive": TODAY,
                     f"hwm_guard_alerted_{ENV}_unparseable": TODAY}
    state[f"hwm_guard_alerted_{ENV}_non_positive"] = "2026-01-01"    # a new day
    _, r4, _ = _halt(_rm(), "0", state)
    assert r4.send.call_count == 1


def test_the_dedupe_key_is_written_first_and_a_failed_write_sends_nothing():
    order = []
    _SyncThread.made.clear()
    out, rec, state = _halt(_rm(), "0", verify_fails=True)
    assert out is False and not rec.send.called and not _SyncThread.made
    assert rec.set_state_verified.call_args.args[0] == [(f"hwm_guard_alerted_{ENV}_non_positive",
                                                          TODAY)]
    assert len(_rows(rec)) == 1                                       # still recorded
    # On success the write comes before the send.
    rec2 = MagicMock()
    rec2.set_state_verified.side_effect = lambda w: order.append("write") or []
    rec2.send.side_effect = lambda *a, **k: order.append("send")
    with patch.object(config, "HWM_GUARD_MODE", "shadow"), \
         patch("db.get_state_strict", side_effect=lambda k, d=None: {
             f"high_water_mark_{ENV}": "0"}.get(k, d)), \
         patch("db.get_state", return_value=None), patch("db.set_state"), \
         patch("db.set_state_verified", rec2.set_state_verified), \
         patch("db.log_activity_queued"), patch("alerts._send", rec2.send), \
         patch.object(risk_manager.threading, "Thread", _SyncThread):
        _rm().is_halted()
    assert order == ["write", "send"]


def test_the_alert_goes_on_a_daemon_thread_and_a_hanging_send_does_not_delay_the_check():
    import threading
    import time
    release = threading.Event()
    started = threading.Event()

    def hang(*a, **k):
        started.set()
        release.wait(10)
    try:
        with patch.object(config, "HWM_GUARD_MODE", "shadow"), \
             patch("db.get_state_strict", side_effect=lambda k, d=None: {
                 f"high_water_mark_{ENV}": "0"}.get(k, d)), \
             patch("db.get_state", return_value=None), patch("db.set_state"), \
             patch("db.set_state_verified", return_value=[]), \
             patch("db.log_activity_queued"), patch("alerts._send", side_effect=hang) as send:
            t0 = time.monotonic()
            out = _rm().is_halted()
            took = time.monotonic() - t0
            assert started.wait(5)
        assert out is False and took < 2
        assert send.call_args.kwargs == {"severity": alert_policy.CRITICAL}
        th = [t for t in threading.enumerate() if t.name.startswith("hwm-guard-alert-")]
        assert th and all(t.daemon for t in th)
    finally:
        release.set()


def test_the_alert_is_critical_and_has_no_dollar_values():
    _, rec, _ = _halt(_rm(123456.78), "0")
    text = rec.send.call_args.args[0]
    assert alert_policy.classify(text)[0] == alert_policy.CRITICAL
    assert rec.send.call_args.kwargs == {"severity": alert_policy.CRITICAL}
    _no_money(text)
    for args in _rows(rec):
        _no_money(args[2])
        assert all(not isinstance(v, float) for v in args[4].values())


def test_no_exception_escapes_the_guard():
    """A failing row write: the guard logs and the check answers as it would."""
    out, _, _ = _halt(_rm(1000.0), "0", strict_error=RuntimeError("db"))
    assert out is False
    rm = _rm(1000.0)
    out, _, _ = _halt(rm, "0", mode="enforce", strict_error=RuntimeError("db"))
    assert out is True and rm.halt_unknown is True
    out, _, _ = _halt(_rm(1000.0), "abc", strict_error=RuntimeError("db"))
    assert isinstance(out, ValueError)                               # the old raise, not the db error


def test_simulation_writes_and_sends_nothing():
    out, rec, _ = _halt(_rm(simulation=True), "0")
    assert out is False and not rec.row.called and not rec.send.called


def test_a_usable_mark_is_untouched():
    out, rec, _ = _halt(_rm(1000.0), "1000")
    assert out is False and not rec.row.called and not rec.send.called


def test_hwm_guard_ships_in_shadow():
    import ast
    tree = ast.parse(open(config.__file__).read())
    shipped = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert shipped["HWM_GUARD_MODE"] == "shadow"
    assert shipped["FUNDAMENTALS_FREEZE_REFRESHES"] == 2
    assert shipped["FUNDAMENTALS_FREEZE_MAX_CHANGED"] == 0


# ---- 2. fundamentals freeze counter ----

def _skey():
    import db
    return f"fundamentals_unchanged_streak_{db.TRADING_ENV}"


def _akey():
    import db
    return f"fundamentals_freeze_alerted_{db.TRADING_ENV}"


def _streak(state):
    import json
    return json.loads(state[_skey()])["streak"]


class _Day(datetime):
    day_value = datetime(2026, 11, 20, 22, 0, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.day_value.astimezone(tz) if tz else cls.day_value


def _record(changes, state=None, on=None, send_returns=101):
    """One _fundamentals_change_record run on the ET day `on` (a date)."""
    state = {} if state is None else state
    rec = MagicMock()
    rec.send.return_value = send_returns
    _Day.day_value = datetime.combine(on or date(2026, 11, 20), datetime.min.time(),
                                      ET).replace(hour=22)
    with patch("db.count_fundamentals_changes", return_value=changes), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_activity_direct", rec.row), patch("alerts._send", rec.send), \
         patch("datetime.datetime", _Day):
        tasks._fundamentals_change_record(_Day.day_value)
    return rec, state


FROZEN = {"tickers": 75, "changed": 0, "no_prior": 0}
MOVING = {"tickers": 75, "changed": 19, "no_prior": 0}
D1, D2, D3 = date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 20)


def test_one_unchanged_refresh_is_recorded_not_alerted():
    rec, state = _record(FROZEN, on=D1)
    assert _streak(state) == 1 and not rec.send.called
    extra = rec.row.call_args.args[4]
    assert rec.row.call_args.args[0] == "fundamentals_changes"
    assert extra["changed"] == 0 and extra["judged"] == 75 and extra["flagged"] is False


def test_two_unchanged_days_warn_and_alert_once_a_day():
    state = {}
    _record(FROZEN, state, on=D1)
    rec, state = _record(FROZEN, state, on=D2)
    assert _streak(state) == 2
    text = rec.send.call_args.args[0]
    assert "Fundamentals unchanged" in text and "0 of 75" in text and "2 refreshes" in text
    assert rec.row.call_args.args[3] == "negative" and rec.row.call_args.args[4]["alerted"] is True
    assert state[_akey()] == D2.isoformat()
    rec3, state = _record(FROZEN, state, on=D2)                       # a re-run the same day
    assert not rec3.send.called and rec3.row.call_args.args[4]["flagged"] is True


def test_a_retry_or_rerun_the_same_day_is_not_a_second_unchanged_refresh():
    """One frozen day, then the same day's refresh again (a retry 5 min
    later, or a manual re-run): still a run of 1, and no alert."""
    state = {}
    _record(FROZEN, state, on=D1)
    rec, state = _record(FROZEN, state, on=D1)
    assert _streak(state) == 1 and not rec.send.called
    assert rec.row.call_args.args[4]["unchanged_streak"] == 1
    # The day's step is replaced, not added: a re-run that saw changes resets it.
    _record(FROZEN, state, on=D2)
    assert _streak(state) == 2
    _record(MOVING, state, on=D2)
    assert _streak(state) == 0
    _record(FROZEN, state, on=D2)                                     # and back: 1 + 1
    assert _streak(state) == 2


def test_a_failed_send_is_recorded_as_not_alerted():
    state = {}
    _record(FROZEN, state, on=D1)
    rec, state = _record(FROZEN, state, on=D2, send_returns=None)    # _send gave up
    assert rec.send.called and rec.row.call_args.args[4]["alerted"] is False


def test_the_keys_are_env_scoped():
    import db
    _, state = _record(FROZEN, {}, on=D1)
    _, state = _record(FROZEN, state, on=D2)
    assert set(state) == {_skey(), _akey()}
    assert all(k.endswith(f"_{db.TRADING_ENV}") for k in state)
    assert "fundamentals_unchanged_streak" not in state and "fundamentals_freeze_alerted" not in state


def test_a_change_resets_the_run():
    import json
    state = {_skey(): json.dumps({"streak": 4, "day": D1.isoformat(), "before": 3})}
    rec, state = _record(MOVING, state, on=D2)
    assert _streak(state) == 0 and not rec.send.called


def test_no_comparable_ticker_leaves_the_run_and_unreadable_counts_record_nothing():
    import json
    state = {_skey(): json.dumps({"streak": 1, "day": D1.isoformat(), "before": 0})}
    _, state = _record({"tickers": 3, "changed": 0, "no_prior": 3}, state, on=D2)
    assert _streak(state) == 1
    rec, _ = _record(None)
    assert not rec.row.called and not rec.send.called


def test_an_unreadable_stored_run_starts_from_zero():
    _, state = _record(FROZEN, {_skey(): "not json"}, on=D1)
    assert _streak(state) == 1


def test_the_record_runs_after_a_successful_refresh_only_and_never_raises():
    with patch("ingestion.fundamentals.refresh_all_fundamentals", return_value=75), \
         patch("tasks._heartbeat_ok"), patch("tasks._fundamentals_change_record") as record:
        tasks.refresh_fundamentals.run()
    since = record.call_args.args[0]
    assert since.tzinfo is not None
    with patch("db.count_fundamentals_changes", side_effect=RuntimeError("db")):
        tasks._fundamentals_change_record(datetime(2026, 11, 20, tzinfo=ET))     # no raise


class _Cur:
    def __init__(self, seen):
        self.seen = seen

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.seen.setdefault("sql", []).append((sql, args))

    def fetchone(self):
        return (75, 19, 1)


def _count(since, **kw):
    import db
    seen = {}
    conn = MagicMock()
    conn.cursor.return_value = _Cur(seen)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.count_fundamentals_changes(since, **kw)
    return out, seen["sql"][-1]


def test_count_fundamentals_changes_compares_with_the_previous_et_day():
    since = datetime(2026, 11, 20, 22, 0, tzinfo=ET)
    out, (sql, args) = _count(since)
    assert out == {"tickers": 75, "changed": 19, "no_prior": 1}
    assert "IS DISTINCT FROM" in sql and "target_mean_price" in sql and "forward_eps" in sql
    assert "recorded_at >= %s" in sql and "recorded_at < %s" in sql
    assert args == (since, datetime(2026, 11, 20, tzinfo=ET))      # prior rows: before 11-20 ET
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b", sql)


def test_count_fundamentals_changes_on_a_retry_skips_the_run_it_repeats():
    """A retry at 22:05 ET: the 22:00 run's rows are after the start of the
    day, so they are not the prior rows; a UTC since past midnight UTC still
    takes the ET day."""
    retry = datetime(2026, 11, 21, 3, 5, tzinfo=ZoneInfo("UTC"))        # 22:05 ET on 11-20
    _, (_, args) = _count(retry)
    assert args[1] == datetime(2026, 11, 20, tzinfo=ET)
    assert args[1] < datetime(2026, 11, 20, 22, 0, tzinfo=ET)
    explicit = datetime(2026, 11, 19, tzinfo=ET)
    _, (_, args) = _count(retry, prior_before=explicit)
    assert args == (retry, explicit)


# ---- 3. per-feed counts in source_yield ----

class _Now(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 11, 18, 20, tzinfo=ET)


def _feeds(per_day: dict):
    return [{"source": f, "day": d, "n": n} for d, fs in per_day.items() for f, n in fs.items()]


def test_a_dead_feed_shows_while_the_other_carries_the_source():
    """11-10: yfinance news 0, google_news up, so "news" as a whole looks
    normal; the feed breakdown shows the outage."""
    days = [date(2026, 11, d) for d in (3, 4, 5, 6, 9)] + [date(2026, 10, 30)]
    news = [{"source": "news", "day": d, "n": 1100} for d in days] + \
           [{"source": "news", "day": date(2026, 11, 10), "n": 1000}]
    feeds = _feeds({d: {"google_news": 400, "yfinance": 700} for d in days}
                   | {date(2026, 11, 10): {"google_news": 1000}})
    with patch("datetime.datetime", _Now), \
         patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_seen_item_counts", return_value=news), \
         patch("db.get_news_feed_counts", return_value=feeds), \
         patch("db.TRADING_ENV", "live"), patch("db.log_activity_direct") as write:
        tasks.source_yield(days_back=1)
    extra = write.call_args.args[4]
    assert extra["day"] == "2026-11-10"
    assert extra["drops"] == []                                          # the source looks fine
    assert extra["feed_drops"] == ["news:yfinance"]
    yf = extra["feeds"]["news"]["yfinance"]
    assert yf["n"] == 0 and yf["baseline_mean"] == 700 and yf["drop"] is True
    assert extra["feeds"]["news"]["google_news"]["drop"] is False
    # the existing keys are unchanged
    assert set(extra) == {"day", "shadow_key", "sources", "drops", "computed_at", "env",
                          "feeds", "feed_drops"}
    assert set(extra["sources"]["news"]) == {"n", "baseline_mean", "baseline_days",
                                             "baseline_skipped_days", "ratio", "drop"}


def test_an_unread_breakdown_still_writes_the_row():
    with patch("datetime.datetime", _Now), \
         patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_seen_item_counts", return_value=[{"source": "news", "day": date(2026, 11, 10),
                                                         "n": 5}]), \
         patch("db.get_news_feed_counts", return_value=None), \
         patch("db.TRADING_ENV", "live"), patch("db.log_activity_direct") as write:
        tasks.source_yield(days_back=1)
    extra = write.call_args.args[4]
    assert extra["feeds"] is None and extra["feed_drops"] is None and extra["sources"]["news"]["n"] == 5


def test_news_feed_counts_split_by_the_seen_id():
    import db
    seen = {}

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, args=None):
            seen["sql"] = sql

        def fetchall(self):
            return [("google_news", date(2026, 11, 10), 400), ("yfinance", date(2026, 11, 10), 700)]
    conn = MagicMock()
    conn.cursor.return_value = Cur()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.get_news_feed_counts(datetime(2026, 11, 1), datetime(2026, 11, 11))
    assert out == [{"source": "google_news", "day": date(2026, 11, 10), "n": 400},
                   {"source": "yfinance", "day": date(2026, 11, 10), "n": 700}]
    assert "https://news.google.com/%%" in seen["sql"] and "source = 'news'" in seen["sql"]
