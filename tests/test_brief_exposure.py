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
    assert isinstance(out.pop("read_at"), str)
    assert out == {"position_context_shown": True, "position_context_status": None,
                   "stop_shown": 160.0, "stop_shown_pct": None,
                   "prior_verdicts_shown": ["2026-10-30 HOLD c6", "2026-10-30 HOLD c7", "2026-10-30 HOLD c6"],
                   "has_stop_row": True, "prior_verdicts_before_open": None,
                   "held_broker": True, "held_broker_reason": "snapshot"}


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


def test_held_at_gate_records_the_read_time_and_age():
    """Professor M138: how old the broker read was when the gates used it."""
    from datetime import datetime, timezone
    now = datetime(2026, 11, 2, 14, 5, tzinfo=timezone.utc)
    cand = {"held": True, "held_as_of": "2026-11-02T14:01:30+00:00"}
    assert analyst._held_at_gate(cand, now=now) == {
        "held_at_gate": True, "held_at_gate_as_of": "2026-11-02T14:01:30+00:00", "held_at_gate_age_s": 210}
    assert analyst._held_at_gate({"held": None}, now=now) == {
        "held_at_gate": None, "held_at_gate_as_of": None, "held_at_gate_age_s": None}


def test_counts_shown_verdicts_from_before_the_position_opened():
    """Audit 11-06 05:33 #2d: after a re-entry the section can show the
    earlier position's verdicts."""
    from datetime import datetime, timezone
    opened = datetime(2026, 11, 5, 16, 8, tzinfo=timezone.utc)
    verdicts = [{"timestamp": datetime(2026, 11, 6, 11, 38, tzinfo=timezone.utc)},
                {"timestamp": datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)},
                {"timestamp": datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)}]
    with patch("db.get_position_stop", return_value={"stop_price": 160.0, "opened_at": opened}), \
         patch("db.get_position_context", return_value={"recent_verdicts": verdicts}), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        assert analyst._brief_exposure("NVDA", _PROMPT)["prior_verdicts_before_open"] == 2


def test_before_open_count_is_none_when_unreadable_and_absent_without_a_section():
    from datetime import datetime, timezone
    rec = {"stop_price": 160.0, "opened_at": datetime(2026, 11, 5, tzinfo=timezone.utc)}
    with patch("db.get_position_stop", return_value=rec), \
         patch("db.get_position_context", side_effect=RuntimeError("pool exhausted")), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        assert analyst._brief_exposure("NVDA", _PROMPT)["prior_verdicts_before_open"] is None
        assert "prior_verdicts_before_open" not in analyst._brief_exposure("NVDA", "no section")
    with patch("db.get_position_stop", side_effect=RuntimeError("db down")), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        assert analyst._brief_exposure("NVDA", _PROMPT)["prior_verdicts_before_open"] is None


def test_before_open_count_is_none_when_the_context_query_failed():
    """get_position_context returns {} on a query error (professor 58m)."""
    from datetime import datetime, timezone
    rec = {"stop_price": 160.0, "opened_at": datetime(2026, 11, 5, tzinfo=timezone.utc)}
    with patch("db.get_position_stop", return_value=rec), \
         patch("db.get_position_context", return_value={}), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        assert analyst._brief_exposure("NVDA", _PROMPT)["prior_verdicts_before_open"] is None
    with patch("db.get_position_stop", return_value=rec), \
         patch("db.get_position_context", return_value={"recent_verdicts": []}), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        assert analyst._brief_exposure("NVDA", _PROMPT)["prior_verdicts_before_open"] == 0


def test_records_the_stop_shown_and_the_downside_against_it():
    prompt = ("=== POSITION CONTEXT — NVDA (held) ===\n"
              "Entry $180.00; protective stop $165.50 (-8.1% from current price) [atr]\n=== X ===")
    with patch("db.get_position_stop", return_value=None), \
         patch("tasks._snapshot_positions_or_none", return_value={}):
        out = analyst._brief_exposure("NVDA", prompt)
    assert out["stop_shown"] == 165.5 and out["stop_shown_pct"] == -8.1
    assert analyst._downside_vs_stop_shown({"valuation": {"downside_scenario": "$165.50"}}, 165.5) == 0.0
    assert analyst._downside_vs_stop_shown({"valuation": {"downside_scenario": "$150"}}, 165.5) == -0.0937
    assert analyst._downside_vs_stop_shown({"valuation": {"downside_scenario": "Unknown"}}, 165.5) is None
    assert analyst._downside_vs_stop_shown({}, None) is None


def test_position_context_status_says_why_the_section_is_out():
    """Audit 11-06 23:24 #10: a DB error no longer reads as "not held"; the
    brief is unchanged and the status records why."""
    from unittest.mock import patch
    import context_engine
    context_engine._POSITION_CONTEXT_STATUS.clear()
    with patch("config.BRIEF_POSITION_CONTEXT", True):
        with patch("db.get_position_stop", side_effect=RuntimeError("pool")) as gs:
            assert context_engine._position_context_section("NVDA", 100.0) == ""
        assert gs.call_args.kwargs == {"strict": True}
        assert context_engine.position_context_status("NVDA") == "error:RuntimeError"
        with patch("db.get_position_stop", return_value=None):
            context_engine._position_context_section("AMD", 100.0)
        assert context_engine.position_context_status("AMD") == "not_held"
        stop = {"entry_price": 90.0, "stop_price": 80.0, "opened_at": None}
        with patch("db.get_position_stop", return_value=stop), \
             patch("db.get_position_context", side_effect=RuntimeError("pool")):
            assert "protective stop $80.00" in context_engine._position_context_section("TSM", 100.0)
        assert context_engine.position_context_status("TSM") == "shown_partial:RuntimeError"
        import pytest
        with patch("db.get_position_stop", return_value={"entry_price": None, "stop_price": 80.0}), \
             patch("db.get_position_context", return_value={}):
            with pytest.raises(TypeError):
                context_engine._position_context_section("ARM", 100.0)
        assert context_engine.position_context_status("ARM") == "error:TypeError"
    with patch("config.BRIEF_POSITION_CONTEXT", False):
        context_engine._position_context_section("MU", 100.0)
    assert context_engine.position_context_status("MU") == "off"
    assert context_engine.position_context_status("QQQ") is None
    from agents.analyst import _brief_exposure
    with patch("db.get_position_stop", return_value=None):
        assert _brief_exposure("NVDA", "")["position_context_status"] == "error:RuntimeError"
    context_engine._POSITION_CONTEXT_STATUS.clear()
