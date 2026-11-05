"""Every routing exit row carries the verdict's model, provenance and a
decision naming the exit (as EARNINGS_BLOCKED did), and a failed write is
logged at WARNING instead of dropped silently."""
import logging
from unittest.mock import MagicMock, patch

import pytest

from agents import crew

_PROV = {"model": "qwen72b", "prompt_sha": "abc", "plan": {"x": 1}}


def _result(action):
    return {"ticker": "NVDA", "action": action, "conviction": 6, "thesis_bucket": "Accelerator",
            "reasoning": "r", "pass_to_risk": True, "provenance": dict(_PROV)}


def _cooldown():
    with patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval",
               return_value={"conviction": 6, "decided_at": "2026-11-03T12:00:00Z"}), \
         patch("feed.emit"):
        crew._route_to_approval(_result("ADD"), {"regime": "Neutral"}, MagicMock())


def _no_position():
    client = MagicMock()
    client.get_all_positions.return_value = []
    with patch("db.has_active_approval", return_value=False):
        crew._route_to_approval(_result("EXIT"), {"regime": "Neutral"}, client)


def _unreadable():
    client = MagicMock()
    client.get_all_positions.side_effect = ConnectionError("gateway down")
    with patch("db.has_active_approval", return_value=False), patch("alerts._send"):
        crew._route_to_approval(_result("EXIT"), {"regime": "Neutral"}, client)


def _catalyst_blocked():
    with patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("catalyst_service.gates.catalyst_gate_check",
               return_value={"approved": False, "reason": "FDA date unresolved"}), \
         patch("alerts._send"), patch("feed.emit"):
        crew._route_to_approval(_result("OPEN"), {"regime": "Neutral"}, MagicMock())


def _catalyst_gate_error():
    with patch("db.has_active_approval", return_value=False), \
         patch("db.get_last_executed_approval", return_value=None), \
         patch("catalyst_service.gates.catalyst_gate_check", side_effect=RuntimeError("calendar")), \
         patch("config.EARNINGS_ENTRY_BLACKOUT_DAYS", 5), \
         patch("db.is_near_earnings", return_value=True), \
         patch("db.get_next_earnings", return_value="2026-11-05"), \
         patch("feed.emit"):
        crew._route_to_approval(_result("OPEN"), {"regime": "Neutral"}, MagicMock())


_CASES = [("COOLDOWN_SKIPPED", _cooldown), ("SELL_NO_POSITION", _no_position),
          ("SELL_POSITION_UNREADABLE", _unreadable), ("CATALYST_BLOCKED", _catalyst_blocked),
          ("CATALYST_GATE_ERROR", _catalyst_gate_error)]


@pytest.mark.parametrize("kind, run", _CASES)
def test_route_exit_row_fields(kind, run):
    with patch("db.log_signal") as log_signal:
        run()
    rows = [c for c in log_signal.call_args_list if c.args[1] == kind]
    assert len(rows) == 1
    kw = rows[0].kwargs
    assert kw["model"] == "qwen72b"
    assert kw["provenance"] == {"model": "qwen72b", "prompt_sha": "abc"}   # verdict-only keys dropped
    assert kw["decision"]["route_exit"] == kind and kw["decision"]["verdict_conviction"] == 6


@pytest.mark.parametrize("kind, run", _CASES)
def test_route_exit_write_failure_warns(kind, run, caplog):
    with patch("db.log_signal", side_effect=RuntimeError("db down")), \
         caplog.at_level(logging.WARNING, logger="agents.crew"):
        run()
    assert any(f"{kind} signal log failed" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)
