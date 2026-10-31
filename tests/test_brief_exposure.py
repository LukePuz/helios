"""What the brief showed the model about its position and past verdicts
(audit A1031-0453-1/-2): log-only provenance."""
from unittest.mock import patch

from agents import analyst

_PROMPT = ("intro\n=== POSITION CONTEXT — NVDA (held) ===\nEntry $180.00; protective stop $160.00\n"
           "Recent verdicts: 2026-10-30 HOLD c6; 2026-10-30 HOLD c7; 2026-10-30 HOLD c6\nmore")


def test_records_the_verdicts_shown_and_broker_held():
    with patch("db.get_position_stop", return_value={"stop_price": 160.0}), \
         patch("tasks._snapshot_positions_or_none", return_value={"NVDA": {"qty": 1, "market_value": 200}}):
        out = analyst._brief_exposure("NVDA", _PROMPT)
    assert out == {"position_context_shown": True,
                   "prior_verdicts_shown": ["2026-10-30 HOLD c6", "2026-10-30 HOLD c7", "2026-10-30 HOLD c6"],
                   "has_stop_row": True, "held_broker": True, "held_broker_reason": "snapshot"}


def test_broker_position_without_a_stop_row_is_visible():
    with patch("db.get_position_stop", return_value=None), \
         patch("tasks._snapshot_positions_or_none", return_value={"AMD": {"qty": 3}}):
        out = analyst._brief_exposure("AMD", "no position section")
    assert out["position_context_shown"] is False and out["prior_verdicts_shown"] == []
    assert out["has_stop_row"] is False and out["held_broker"] is True


def test_unknown_when_no_fresh_snapshot_or_reads_fail():
    with patch("db.get_position_stop", side_effect=RuntimeError("db down")), \
         patch("tasks._snapshot_positions_or_none", return_value=None):
        out = analyst._brief_exposure("PLTR", "")
    assert out["has_stop_row"] is None and out["held_broker"] is None
    assert out["held_broker_reason"] == "no_snapshot"
    with patch("db.get_position_stop", return_value=None), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        out = analyst._brief_exposure("PLTR", "")
    assert out["held_broker"] is False and out["held_broker_reason"] == "empty"
    with patch("db.get_position_stop", return_value=None), \
         patch("tasks._snapshot_positions_or_none", return_value={"NVDA": {"qty": 1}}):
        assert analyst._brief_exposure("PLTR", "")["held_broker"] is False


def test_a_news_line_outside_the_position_section_is_not_read_as_verdicts():
    prompt = "=== EVENT ===\nRecent verdicts: analysts upgrade\n=== MEMORY ===\nx"
    with patch("db.get_position_stop", return_value=None), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        out = analyst._brief_exposure("NVDA", prompt)
    assert out["position_context_shown"] is False and out["prior_verdicts_shown"] == []


def test_broker_held_reasons():
    import tasks
    with patch("tasks._snapshot_positions_or_none", return_value=None):
        assert tasks._broker_held("NVDA") == (None, "no_snapshot", {})
    with patch("tasks._snapshot_positions_or_none", return_value={}):
        assert tasks._broker_held("NVDA") == (False, "empty", {})
    with patch("tasks._snapshot_positions_or_none", return_value={"NVDA": {"qty": 1}}):
        assert tasks._broker_held("nvda")[:2] == (True, "snapshot")
        assert tasks._broker_held("AMD")[:2] == (False, "snapshot")
    with patch("tasks._snapshot_positions_or_none", side_effect=RuntimeError("db down")):
        assert tasks._broker_held("NVDA")[:2] == (None, "error")


def test_gate_flags():
    """Audits A1031-0253-2, A1031-0453-3/-4: log-only counters."""
    from datetime import datetime, timezone
    now = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)          # 13:00 ET
    earlier = [{"timestamp": datetime(2026, 11, 2, 14, 0, tzinfo=timezone.utc)},   # today ET
               {"timestamp": datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)}]    # 11-01 ET
    with patch("db.get_recent_verdicts", return_value=earlier):
        f = analyst._gate_flags("NVDA", "EXIT", 6, 2, True, now=now)
    assert f == {"model_sell_pass_unconfirmed": True, "rubric_gate_conflict": False, "verdicts_today": 1}
    with patch("db.get_recent_verdicts", return_value=[]):
        assert analyst._gate_flags("NVDA", "TRIM", 4, 0, True, now=now)["model_sell_pass_unconfirmed"] is False
        # the model's own c7 OPEN citing 2 sources counts even if a gate later capped it
        assert analyst._gate_flags("NVDA", "OPEN", 6, 2, True, "OPEN", 7, now=now)["rubric_gate_conflict"] is True
        assert analyst._gate_flags("NVDA", "WATCH", 7, 1, False, "OPEN", 7, now=now)["rubric_gate_conflict"] is False
        assert analyst._gate_flags("NVDA", "OPEN", 8, 2, True, "OPEN", 8, now=now)["rubric_gate_conflict"] is False
    with patch("db.get_recent_verdicts", side_effect=RuntimeError("db down")):
        assert analyst._gate_flags("NVDA", "HOLD", 6, 2, False, now=now)["verdicts_today"] is None
