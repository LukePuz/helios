"""RiskManager(trading_client) raising in _route_to_approval leaves a
ROUTE_RISK_INIT_FAILED row and a throttled alert. No live DB/IBKR."""
from unittest.mock import MagicMock, patch

import pytest

from agents import crew


class _FakeTimer:
    """threading.Timer stand-in: records the delay, never runs on its own."""
    made: list = []

    def __init__(self, delay, fn):
        self.delay, self.fn, self.daemon, self.started = delay, fn, False, False
        _FakeTimer.made.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        pass


@pytest.fixture(autouse=True)
def _reset_throttle():
    crew._RISK_INIT_ALERT.update(sent_at=None, held=[], timer=None)
    _FakeTimer.made = []
    with patch("db.is_near_earnings", return_value=False), \
         patch("db.get_next_earnings", return_value=None), \
         patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("catalyst_service.gates.catalyst_gate_check", return_value={"approved": True}), \
         patch("threading.Timer", _FakeTimer):
        yield
    crew._RISK_INIT_ALERT.update(sent_at=None, held=[], timer=None)


def _result(ticker="NVDA", action="OPEN", conviction=7):
    return {"ticker": ticker, "action": action, "conviction": conviction,
            "thesis_bucket": "Accelerator", "reasoning": "test",
            "provenance": {"model": "m", "prompt_sha": "p"}}


def _boom():
    return ConnectionError("Could not connect to IBKR TWS at host:port after 3 attempts acct U123")


def _route(result, log_signal, send):
    with patch("db.log_signal", log_signal), patch("alerts._send", send), \
         patch("risk_manager.RiskManager", side_effect=_boom()), \
         patch("agents.crew._get_price") as get_price:
        crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
    get_price.assert_not_called()


def _rows(log_signal, kind="ROUTE_RISK_INIT_FAILED"):
    return [c for c in log_signal.call_args_list if c.args[1] == kind]


def test_risk_init_failure_writes_one_row_and_one_alert():
    log_signal, send = MagicMock(), MagicMock(return_value=1)
    _route(_result(), log_signal, send)
    rows = _rows(log_signal)
    assert len(rows) == 1 and len(log_signal.call_args_list) == 1
    args, kw = rows[0].args, rows[0].kwargs
    assert args[0] == "NVDA" and args[2] == 0.7
    assert "ConnectionError" in args[3] and "U123" not in args[3] and "TWS" not in args[3]
    assert kw["model"] == "m"
    assert kw["decision"] == {"route_exit": "ROUTE_RISK_INIT_FAILED", "verdict_action": "OPEN",
                              "verdict_conviction": 7, "error_class": "ConnectionError",
                              "would_retry": True}
    send.assert_called_once()
    assert send.call_args.args[0] == ("Routing blocked: RiskManager init failed when sizing NVDA OPEN c7 (ConnectionError); "
                                      "verdict not executed")


def test_would_retry_false_outside_new_entries():
    log_signal, send = MagicMock(), MagicMock(return_value=1)
    _route(_result(action="WATCH"), log_signal, send)
    assert _rows(log_signal)[0].kwargs["decision"]["would_retry"] is False


def test_throttle_holds_and_batches_into_next_alert():
    log_signal, send = MagicMock(), MagicMock(return_value=1)
    with patch("time.monotonic", return_value=1000.0):
        _route(_result("NVDA"), log_signal, send)
        _route(_result("AMD", "ADD", 8), log_signal, send)
        _route(_result("KLAC"), log_signal, send)
    assert len(_rows(log_signal)) == 3          # every verdict still gets its row
    assert send.call_count == 1
    with patch("time.monotonic", return_value=1000.0 + crew._RISK_INIT_ALERT_WINDOW_S - 1):
        _route(_result("ORCL"), log_signal, send)
    assert send.call_count == 1
    with patch("time.monotonic", return_value=1000.0 + crew._RISK_INIT_ALERT_WINDOW_S):
        _route(_result("MU"), log_signal, send)
    assert send.call_count == 2
    text = send.call_args.args[0]
    assert text.startswith("Routing blocked: RiskManager init failed when sizing MU OPEN c7")
    assert "(3): AMD ADD c8 (ConnectionError), KLAC OPEN c7 (ConnectionError), ORCL OPEN c7 (ConnectionError)" in text
    assert crew._RISK_INIT_ALERT["held"] == []


def test_row_write_failure_does_not_raise():
    log_signal = MagicMock(side_effect=RuntimeError("db down"))
    send = MagicMock(return_value=1)
    _route(_result(), log_signal, send)          # must not raise
    send.assert_called_once()


def test_alert_failure_does_not_raise():
    log_signal, send = MagicMock(), MagicMock(side_effect=RuntimeError("telegram down"))
    _route(_result(), log_signal, send)
    assert len(_rows(log_signal)) == 1


def test_healthy_path_unchanged():
    """RiskManager starts: no new row, no alert, and sizing runs as before."""
    log_signal, send = MagicMock(), MagicMock(return_value=1)
    rm = MagicMock()
    rm.size_position.return_value = {"shares_initial": 0, "flags": ["c7 below floor"]}
    with patch("db.log_signal", log_signal), patch("alerts._send", send), \
         patch("risk_manager.RiskManager", return_value=rm), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("agents.crew._attach_shadows"), \
         patch("agents.crew._shadow_swap", return_value=None), \
         patch.object(rm, "find_swap_candidate", return_value=None, create=True):
        crew._route_to_approval(_result(), {"regime": "Neutral"}, MagicMock())
    rm.size_position.assert_called_once_with("NVDA", 7, "Accelerator", "Neutral", 100.0)
    assert _rows(log_signal) == []
    assert crew._RISK_INIT_ALERT["sent_at"] is None
    assert all("Routing blocked" not in str(c.args[0]) for c in send.call_args_list)


def test_route_exit_without_extra_is_unchanged():
    with patch("db.log_signal") as log_signal:
        crew._log_route_exit("META", "ROUTE_NO_PRICE", {"action": "OPEN", "conviction": 6}, "no price")
    assert log_signal.call_args.kwargs["decision"] == {
        "route_exit": "ROUTE_NO_PRICE", "verdict_action": "OPEN", "verdict_conviction": 6}


def test_route_exit_extra_cannot_override_existing_keys():
    with patch("db.log_signal") as log_signal:
        crew._log_route_exit("META", "ROUTE_NO_PRICE", {"action": "OPEN", "conviction": 6}, "no price",
                             decision_extra={"route_exit": "X", "verdict_action": "EXIT", "extra": 1})
    assert log_signal.call_args.kwargs["decision"] == {
        "route_exit": "ROUTE_NO_PRICE", "verdict_action": "OPEN", "verdict_conviction": 6, "extra": 1}


def test_new_kind_is_not_read_as_a_verdict():
    import db
    assert "ROUTE_RISK_INIT_FAILED" in db._NON_ANALYST_SIGNALS
    assert "ROUTE_RISK_INIT_FAILED" in db._RECORD_ONLY_SIGNALS


def test_held_hits_are_flushed_by_a_timer_when_no_later_failure_comes():
    """Professor m302: hits held inside the window reached Telegram only with
    the next failure; with none, they never did."""
    log_signal, send = MagicMock(), MagicMock(return_value=1)
    with patch("time.monotonic", return_value=1000.0):
        _route(_result("NVDA"), log_signal, send)
    with patch("time.monotonic", return_value=1600.0):
        _route(_result("AMD", "ADD", 8), log_signal, send)
        _route(_result("KLAC"), log_signal, send)
    assert send.call_count == 1
    assert len(_FakeTimer.made) == 1 and _FakeTimer.made[0].started
    assert _FakeTimer.made[0].daemon is True
    assert _FakeTimer.made[0].delay == crew._RISK_INIT_ALERT_WINDOW_S - 600
    with patch("alerts._send", send):
        assert _FakeTimer.made[0].fn(1000.0 + crew._RISK_INIT_ALERT_WINDOW_S) is True
    assert send.call_count == 2
    text = send.call_args.args[0]
    assert "2 more verdict(s)" in text
    assert "AMD ADD c8 (ConnectionError), KLAC OPEN c7 (ConnectionError)" in text
    assert crew._RISK_INIT_ALERT["held"] == [] and crew._RISK_INIT_ALERT["timer"] is None
    # The flush counts as the window's alert: the next hit is held again.
    with patch("time.monotonic", return_value=1000.0 + crew._RISK_INIT_ALERT_WINDOW_S + 5):
        _route(_result("MU"), log_signal, send)
    assert send.call_count == 2 and len(_FakeTimer.made) == 2


def test_timer_finding_nothing_held_sends_nothing():
    send = MagicMock(return_value=1)
    with patch("alerts._send", send):
        assert crew._flush_risk_init_held(5000.0) is False
    send.assert_not_called()


def test_timer_inside_a_newer_window_rearms_instead_of_sending():
    send = MagicMock(return_value=1)
    crew._RISK_INIT_ALERT.update(sent_at=2000.0, held=["MU OPEN c7 (ConnectionError)"])
    with patch("alerts._send", send):
        assert crew._flush_risk_init_held(2100.0) is False
    send.assert_not_called()
    assert crew._RISK_INIT_ALERT["held"] == ["MU OPEN c7 (ConnectionError)"]
    assert _FakeTimer.made[-1].delay == crew._RISK_INIT_ALERT_WINDOW_S - 100
