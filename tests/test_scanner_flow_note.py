"""agents.scanner._score_momentum_with_flow: the unusual-flow note survives a
missing premium. The free flow proxy returns total_premium None; formatting
it raised after the +8 was added, so the score kept the bonus and the note
was lost. The score and the bonus rule are unchanged."""
from unittest.mock import patch

from agents import scanner


def _run(flow, base=(10, "Momentum: 10/25 — x")):
    with patch("agents.scanner._score_momentum", return_value=base), \
         patch("ingestion.options_flow.get_flow_summary", return_value=flow):
        return scanner._score_momentum_with_flow([], "NVDA")


def test_a_missing_premium_keeps_the_note():
    score, notes = _run({"unusual": True, "total_premium": None})
    assert score == 18
    assert notes.endswith(" | Unusual options flow (premium not reported)")


def test_an_absent_premium_key_reads_as_zero_as_before():
    score, notes = _run({"unusual": True})
    assert score == 18
    assert notes.endswith(" | Unusual options flow ($0 premium)")


def test_a_reported_premium_is_formatted_as_before():
    score, notes = _run({"unusual": True, "total_premium": 1234567})
    assert score == 18
    assert notes.endswith(" | Unusual options flow ($1,234,567 premium)")


def test_the_bonus_still_caps_at_25():
    score, _ = _run({"unusual": True, "total_premium": None}, base=(22, "m"))
    assert score == 25


def test_the_skew_branch_is_unchanged():
    score, notes = _run({"unusual": False, "call_ratio": 0.7})
    assert score == 13
    assert notes.endswith(" | Bullish options skew")
