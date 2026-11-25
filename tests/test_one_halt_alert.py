"""A buy that execute_approved blocks on a drawdown halt sends one alert.

is_halted(alert=False, latch_out=...) skips only its own latch alert and
hands its numbers to the caller; execute_approved's ORDER BLOCKED carries
them (percentages only). Every other caller keeps the default alert=True.
"""
from unittest.mock import MagicMock, patch

import execution
import risk_manager
from risk_manager import RiskManager

ENV = risk_manager.TRADING_ENV
LIMIT = risk_manager.DRAWDOWN_HALT_PCT
HWM = 1000.0
PV_PAST = HWM * (1 - LIMIT - 0.02)    # 2 points past the limit, still plausible
PV_OK = HWM * (1 - LIMIT / 2)         # half the limit


def _client(pv):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=pv)
    return client


def _state(halted=False):
    return {f"trading_halted_{ENV}": "true" if halted else "false",
            f"high_water_mark_{ENV}": str(HWM)}


class _Db:
    """The db calls is_halted makes, on one state dict; a verified write
    lands in it, so a second check reads the latched flag."""
    def __init__(self, state, not_saved=()):
        self.state = state
        self.not_saved = list(not_saved)
        self.rows = []

    def get_state_strict(self, k, d=None):
        return self.state.get(k, d)

    def set_state_verified(self, pairs):
        for k, v in pairs:
            if k not in self.not_saved:
                self.state[k] = v
        return list(self.not_saved)

    def log_halt_event(self, *a, **k):
        self.rows.append((a, k))


def _patched(fake, send):
    return [patch("db.get_state_strict", side_effect=fake.get_state_strict),
            patch("db.set_state_verified", side_effect=fake.set_state_verified),
            patch("db.log_halt_event", side_effect=fake.log_halt_event),
            patch("db.set_state"),
            patch("alerts._send", send)]


def _check(pv, state, **kw):
    fake, send = _Db(state, kw.pop("not_saved", ())), MagicMock()
    rm = RiskManager(_client(pv), simulation=kw.pop("simulation", False))
    ps = _patched(fake, send)
    for p in ps:
        p.start()
    try:
        out = rm.is_halted(**kw)
    finally:
        for p in reversed(ps):
            p.stop()
    return out, fake, send, rm


# --- is_halted ------------------------------------------------------------

def test_alert_false_skips_only_the_send():
    latch = {}
    out, fake, send, _ = _check(PV_PAST, _state(), alert=False, latch_out=latch)
    assert out is True
    send.assert_not_called()
    assert fake.state[f"trading_halted_{ENV}"] == "true"
    assert len(fake.rows) == 1
    assert fake.rows[0][0] == ("auto_halt", "per_buy_check", "drawdown")


def test_latch_out_carries_this_calls_numbers():
    latch = {}
    _check(PV_PAST, _state(), alert=False, latch_out=latch)
    assert set(latch) == {"drawdown_pct", "limit_pct", "saved"}
    assert abs(latch["drawdown_pct"] - (LIMIT + 0.02)) < 1e-9
    assert latch["limit_pct"] == LIMIT
    assert latch["saved"] is True


def test_latch_out_says_when_the_flag_did_not_save():
    latch = {}
    _check(PV_PAST, _state(), alert=False, latch_out=latch,
           not_saved=[f"trading_halted_{ENV}"])
    assert latch["saved"] is False


def test_the_default_still_sends_the_halt_alert():
    latch = {}
    out, _, send, _ = _check(PV_PAST, _state(), latch_out=latch)
    assert out is True
    assert send.call_count == 1
    assert "TRADING HALTED — DRAWDOWN LIMIT" in send.call_args[0][0]
    assert latch["saved"] is True


def test_latch_out_is_untouched_below_the_limit():
    latch = {}
    out, _, send, _ = _check(PV_OK, _state(), alert=False, latch_out=latch)
    assert out is False
    assert latch == {}
    send.assert_not_called()


def test_latch_out_is_untouched_when_already_halted():
    latch = {}
    out, fake, _, _ = _check(PV_PAST, _state(halted=True), alert=False, latch_out=latch)
    assert out is True
    assert latch == {}
    assert fake.rows == []


def test_latch_out_is_untouched_on_an_unknown_check():
    latch = {}
    out, _, _, rm = _check(0.0, _state(), alert=False, latch_out=latch)   # implausible read
    assert out is True
    assert rm.halt_unknown is True
    assert latch == {}


def test_latch_out_is_untouched_in_a_simulation():
    latch = {}
    out, fake, _, _ = _check(PV_PAST, _state(), alert=False, latch_out=latch, simulation=True)
    assert out is True
    assert latch == {}
    assert fake.rows == []


def test_latch_out_is_untouched_on_the_hwm_guard_enforce_path():
    latch = {}
    state = dict(_state(), **{f"high_water_mark_{ENV}": "abc"})
    with patch("config.HWM_GUARD_MODE", "enforce"), \
         patch("db.log_activity_queued"), patch("db.get_state", return_value=None), \
         patch("threading.Thread"):
        out, fake, _, rm = _check(PV_PAST, state, alert=False, latch_out=latch)
    assert out is True
    assert rm.halt_unknown is True
    assert latch == {}
    assert fake.rows == []


def test_latch_out_is_untouched_when_the_state_read_raises():
    latch = {}
    rm = RiskManager(_client(PV_PAST))
    with patch("db.get_state_strict", side_effect=RuntimeError("db down")), \
         patch("db.set_state_verified") as write, patch("alerts._send") as send:
        out = rm.is_halted(alert=False, latch_out=latch)
    assert out is True
    assert rm.halt_unknown is True
    assert latch == {}
    write.assert_not_called()
    send.assert_not_called()


# --- the ORDER BLOCKED text ---------------------------------------------

# execute_approved's drawdown ORDER BLOCKED before this change, verbatim.
_OLD_BLOCKED_TEXT = (
    "🛑 ORDER BLOCKED — Drawdown Halt\n"
    "<b>NVDA</b> OPEN order was approved but cannot execute.\n"
    "Portfolio drawdown limit is active. No new BUY entries allowed.\n"
    "Review and reset the halt flag when drawdown is resolved.")


def test_an_empty_latch_is_byte_identical_to_the_old_text():
    assert execution._drawdown_blocked_text("NVDA", "OPEN", {}) == _OLD_BLOCKED_TEXT
    assert execution._drawdown_blocked_text("NVDA", "OPEN", None) == _OLD_BLOCKED_TEXT


def test_a_latch_leads_with_the_halt_and_is_critical():
    import alert_policy
    text = execution._drawdown_blocked_text(
        "NVDA", "OPEN", {"drawdown_pct": LIMIT + 0.02, "limit_pct": LIMIT, "saved": True})
    assert text.split("\n", 1)[0] == "🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b> · ORDER BLOCKED — NVDA"
    assert alert_policy.classify(text)[0] == alert_policy.CRITICAL


def test_the_old_text_is_still_critical():
    import alert_policy
    assert alert_policy.classify(_OLD_BLOCKED_TEXT)[0] == alert_policy.CRITICAL


def test_a_latch_with_unusable_numbers_falls_back():
    for bad in ({"drawdown_pct": None, "limit_pct": None, "saved": True},
                {"drawdown_pct": "n/a", "limit_pct": LIMIT, "saved": True},
                {"drawdown_pct": LIMIT, "limit_pct": "x", "saved": True},
                {"saved": True}):
        text = execution._drawdown_blocked_text("NVDA", "OPEN", bad)
        assert text.startswith("🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b> · ORDER BLOCKED — NVDA\n")
        assert "This buy's check latched the halt (per-buy check)." in text
        assert "Drawdown: -" not in text
        assert "could not be confirmed saved" not in text


def test_a_latch_without_a_saved_true_names_the_flag():
    for saved in (False, None):
        text = execution._drawdown_blocked_text(
            "NVDA", "OPEN", {"drawdown_pct": LIMIT, "limit_pct": LIMIT, "saved": saved})
        assert "could not be confirmed saved" in text


# --- execute_approved -----------------------------------------------------

def _approval():
    return {"id": 1, "ticker": "NVDA", "action": "OPEN", "conviction": 7,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 10, "price_at_sizing": 100.0,
                          "stop_loss_price": 90.0}}


def _execute(fake, send, pv):
    ps = _patched(fake, send) + [patch("ibkr.get_ib"),
                                 patch("ibkr.get_client", return_value=_client(pv)),
                                 patch("ibkr.place_market_order")]
    started = [p.start() for p in ps]
    try:
        result = execution.execute_approved(_approval())
    finally:
        for p in reversed(ps):
            p.stop()
    return result, started[-1]


def test_a_buy_that_latches_sends_one_alert_with_the_numbers():
    fake, send = _Db(_state()), MagicMock()
    result, order = _execute(fake, send, PV_PAST)
    assert result is False
    assert execution.last_outcome_reason() == "blocked: drawdown halt active"
    order.assert_not_called()
    assert send.call_count == 1
    text = send.call_args[0][0]
    assert text.startswith("🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b> · ORDER BLOCKED — NVDA\n")
    assert "<b>NVDA</b> OPEN" in text
    assert f"-{LIMIT + 0.02:.1%} from the high-water mark" in text
    assert f"(limit {LIMIT:.0%})" in text
    assert "latched the halt" in text
    assert "could not be confirmed saved" not in text
    assert "$" not in text


def test_the_latch_still_writes_the_flag_and_the_row():
    fake, send = _Db(_state()), MagicMock()
    _execute(fake, send, PV_PAST)
    assert fake.state[f"trading_halted_{ENV}"] == "true"
    assert len(fake.rows) == 1


def test_an_unsaved_flag_is_named_in_the_one_alert():
    fake, send = _Db(_state(), not_saved=[f"trading_halted_{ENV}"]), MagicMock()
    _execute(fake, send, PV_PAST)
    assert send.call_count == 1
    assert "could not be confirmed saved" in send.call_args[0][0]


def test_an_already_set_flag_keeps_todays_text():
    fake, send = _Db(_state(halted=True)), MagicMock()
    result, order = _execute(fake, send, PV_PAST)
    assert result is False
    order.assert_not_called()
    assert send.call_count == 1
    text = send.call_args[0][0]
    assert text == _OLD_BLOCKED_TEXT
    assert "Drawdown: -" not in text
    assert "latched the halt" not in text


def test_a_failure_after_a_latch_sends_the_drawdown_text():
    """The first send raises after the latch: the except branch's alert is
    the drawdown text, not "No halt is set"."""
    fake, send = _Db(_state()), MagicMock(side_effect=[RuntimeError("telegram"), None])
    result, order = _execute(fake, send, PV_PAST)
    assert result is False
    assert execution.last_outcome_reason() == "blocked: halt check failed"
    order.assert_not_called()
    assert send.call_count == 2
    text = send.call_args_list[1][0][0]
    assert text.startswith("🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b>")
    assert "No halt is set" not in text
    assert f"-{LIMIT + 0.02:.1%} from the high-water mark" in text


def test_a_failure_with_no_latch_keeps_the_failed_text():
    fake, send = _Db(_state()), MagicMock()
    with patch("risk_manager.RiskManager", side_effect=RuntimeError("init")):
        result, order = _execute(fake, send, PV_PAST)
    assert result is False
    assert execution.last_outcome_reason() == "blocked: halt check failed"
    order.assert_not_called()
    assert send.call_count == 1
    assert send.call_args[0][0].startswith("🛑 ORDER BLOCKED — halt check failed\n")
    assert "No halt is set" in send.call_args[0][0]


def test_the_rm_is_built_per_call():
    """latch_out is a fresh dict per execute_approved; the RiskManager is too
    (no module-level or cached instance to carry one call's latch)."""
    fake, send = _Db(_state()), MagicMock()
    with patch("risk_manager.RiskManager", wraps=RiskManager) as cls:
        _execute(fake, send, PV_PAST)
        _execute(_Db(_state()), MagicMock(), PV_PAST)
    assert cls.call_count == 2


# --- the deferred-retry path, pinned as it is ---------------------------

def test_deferred_retry_latch_then_order_blocked_is_two_alerts_today():
    """Left as is: when tasks._deferred_buy_shares_now's re-size latches the
    halt first, size_position's is_halted (default alert) sends the halt
    alert; execute_approved then finds the flag set and sends ORDER BLOCKED
    without numbers. Two alerts, from two separate checks."""
    import tasks
    fake, send = _Db(_state()), MagicMock()
    ps = _patched(fake, send) + [patch("ibkr.get_client", return_value=_client(PV_PAST)),
                                 patch("agents.market_regime.get_cached_regime",
                                       return_value={"regime": "Neutral"})]
    for p in ps:
        p.start()
    try:
        tasks._deferred_buy_shares_now(_approval())
    except Exception:
        pass
    finally:
        for p in reversed(ps):
            p.stop()
    assert fake.state[f"trading_halted_{ENV}"] == "true"
    assert send.call_count == 1
    assert "TRADING HALTED — DRAWDOWN LIMIT" in send.call_args[0][0]

    _execute(fake, send, PV_PAST)
    assert send.call_count == 2
    second = send.call_args_list[1][0][0]
    assert "ORDER BLOCKED — Drawdown Halt" in second
    assert "Drawdown: -" not in second
