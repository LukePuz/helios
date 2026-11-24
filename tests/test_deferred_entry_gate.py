"""The stop-out entry gate re-runs on a deferred OPEN/ADD retry
(tasks._deferred_entry_gate): a broker exit since the verdict expires the
approval and writes ENTRY_GATE_BLOCKED with entry_gate.path deferred_retry;
any failure fails open; the trend half is not re-run."""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
import tasks

APPROVED_AT = datetime.now(timezone.utc) - timedelta(hours=3)
PROV = {"prompt_sha": "abc123", "model": "qwen2.5:72b", "code_sha": "x"}


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
    monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", True)   # must not apply on retry
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS", 5)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "")


def _approval(action="OPEN"):
    return {"id": 7, "ticker": "NVDA", "action": action, "status": "FAILED", "conviction": 7,
            "timestamp": APPROVED_AT, "thesis_bucket": "Compounder",
            "analyst_json": {"ticker": "NVDA", "action": action, "conviction": 7, "provenance": PROV},
            "risk_json": {"shares_initial": 1, "price_at_sizing": 100.0}}


def _exit(hours=1.5):
    return {"exit_at": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
            "hours_since": hours, "lots": 1, "lot_outcome_pct": [-6.1], "lot_alpha_pct": [0.6],
            "lot_convictions": [7], "entries_since_exit": 0}


def _run(approval, exit_patch, expire=True):
    with patch("db.get_approval_by_id", return_value=approval), \
         patch("db.has_newer_buy_approval", return_value=False), \
         patch("db.get_latest_signal_for", return_value=None), \
         patch("tasks._deferred_buy_shares_now", return_value=None), \
         patch("db.claim_deferred_entry_for_retry", return_value=True) as claim, \
         patch("db.get_last_broker_exit", **exit_patch) as broker_exit, \
         patch("db.expire_failed_entry", return_value=expire) as expire_mock, \
         patch("db.update_approval_status"), \
         patch("db.log_signal", return_value=1) as log_signal, \
         patch("agents.crew._entry_trend_check") as trend_read, \
         patch("execution.execute_approved", return_value=True) as execute:
        tasks.retry_deferred_entry.run(7)
    return {"execute": execute, "claim": claim, "expire": expire_mock, "log_signal": log_signal,
            "broker_exit": broker_exit, "trend_read": trend_read}


def _rows(log_signal, kind):
    return [json.loads(json.dumps(c.kwargs, default=str)) for c in log_signal.call_args_list
            if c.args[1] == kind]


@pytest.mark.parametrize("action", ["OPEN", "ADD"])
def test_stop_fill_since_the_verdict_blocks_the_retry(gate_on, action):
    m = _run(_approval(action), {"return_value": _exit(1.5)})
    m["execute"].assert_not_called()
    m["claim"].assert_not_called()
    m["expire"].assert_called_once()
    assert m["expire"].call_args.args[0] == 7
    assert m["expire"].call_args.args[1].startswith("entry gate at retry")
    blocked = _rows(m["log_signal"], "ENTRY_GATE_BLOCKED")
    assert len(blocked) == 1
    gate = blocked[0]["decision"]["entry_gate"]
    assert gate["path"] == "deferred_retry" and gate["approval_id"] == 7
    assert gate["approval_at"] == APPROVED_AT.isoformat()
    assert gate["stopout"]["blocked"] is True and gate["stopout"]["hours_since"] == 1.5
    assert "trend" not in gate                          # trend half not re-run
    m["trend_read"].assert_not_called()
    assert blocked[0]["decision"]["verdict_action"] == action
    assert blocked[0]["provenance"]["prompt_sha"] == "abc123"   # the verdict's provenance
    shadow = _rows(m["log_signal"], "STOPOUT_REENTRY_SHADOW")
    assert len(shadow) == 1 and shadow[0]["decision"]["stopout_reentry"]["path"] == "deferred_retry"


def _execute_args(m):
    m["execute"].assert_called_once()
    return m["execute"].call_args


@pytest.mark.parametrize("action", ["OPEN", "ADD"])
def test_no_stop_fill_executes_with_identical_args(gate_on, action, monkeypatch):
    on = _run(_approval(action), {"return_value": None})
    on_args = _execute_args(on)
    assert _rows(on["log_signal"], "ENTRY_GATE_BLOCKED") == []
    on["broker_exit"].assert_called_once()
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 0)
    off = _run(_approval(action), {"return_value": None})
    assert _execute_args(off) == on_args
    on["expire"].assert_not_called()


def test_old_exit_outside_the_window_does_not_block(gate_on):
    m = _run(_approval(), {"return_value": _exit(11 * 24)})
    _execute_args(m)
    assert _rows(m["log_signal"], "ENTRY_GATE_BLOCKED") == []


def test_raising_read_fails_open(gate_on):
    m = _run(_approval(), {"side_effect": RuntimeError("db down")})
    _execute_args(m)
    assert _rows(m["log_signal"], "ENTRY_GATE_BLOCKED") == []
    m["expire"].assert_not_called()


def test_gate_internal_error_fails_open(gate_on):
    with patch("agents.crew._stopout_reentry_shadow", side_effect=ValueError("boom")):
        m = _run(_approval(), {"return_value": _exit(1.5)})
    _execute_args(m)


@pytest.mark.parametrize("switch,value", [("STOPOUT_REENTRY_GATE_DAYS", 0),
                                          ("STOPOUT_REENTRY_SHADOW_ENABLED", False)])
def test_switch_off_skips_the_gate(gate_on, monkeypatch, switch, value):
    monkeypatch.setattr(config, switch, value)
    with patch("agents.crew._entry_gate") as gate, patch("agents.crew._stopout_reentry_shadow") as shadow:
        m = _run(_approval(), {"return_value": _exit(1.5)})
    gate.assert_not_called()
    shadow.assert_not_called()
    m["broker_exit"].assert_not_called()
    _execute_args(m)


def test_block_still_holds_when_the_expiry_does_not_apply(gate_on):
    m = _run(_approval(), {"return_value": _exit(1.5)}, expire=False)
    m["execute"].assert_not_called()
    m["claim"].assert_not_called()


@pytest.mark.parametrize("action", ["TRIM", "EXIT"])
def test_sells_are_untouched(gate_on, action):
    with patch("db.get_last_broker_exit") as broker_exit, patch("agents.crew._entry_gate") as gate:
        assert tasks._deferred_entry_gate(_approval(action)) is False
    broker_exit.assert_not_called()
    gate.assert_not_called()


def test_verdict_time_gate_unchanged_without_path(gate_on):
    """The verdict-time call (no path) still records and applies the trend half."""
    from agents import crew
    trend = {"read": "ok", "bar_age_days": 1, "below_sma50": True, "below_sma200": True,
             "pct_vs_sma50": -5.0, "pct_vs_sma200": -9.0}
    with patch("db.log_signal") as ls:
        assert crew._entry_gate("NVDA", {"action": "OPEN", "conviction": 7}, None, trend) is True
    gate = ls.call_args.kwargs["decision"]["entry_gate"]
    assert "path" not in gate and gate["trend"]["blocked"] is True


def test_retry_path_cannot_block_on_trend(gate_on):
    from agents import crew
    trend = {"read": "ok", "bar_age_days": 1, "below_sma50": True, "below_sma200": True}
    with patch("db.log_signal") as ls:
        assert crew._entry_gate("NVDA", {"action": "OPEN"}, None, trend, path="deferred_retry") is False
        assert crew._entry_gate("NVDA", {"action": "OPEN"}, None, None, path="deferred_retry") is False
    ls.assert_not_called()


@pytest.mark.parametrize("expire,alerted", [(True, False), (False, True)])
def test_unexpired_block_is_marked_so_no_stuck_alert(gate_on, expire, alerted):
    with patch("tasks._mark_deferred_stuck_alerted") as mark:
        m = _run(_approval("OPEN"), {"return_value": _exit(1.5)}, expire=expire)
    m["execute"].assert_not_called()
    assert mark.called is alerted
    assert "1.5 h ago" in m["expire"].call_args.args[1]
