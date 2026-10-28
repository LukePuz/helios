"""PRICE STRUCTURE block in the brief (audit 2026-10-28 09:00, #2a)."""
from unittest.mock import patch

import context_engine


def test_section_lists_levels_around_price():
    ladder = {"atr": 5.0, "base_floor": 90.0, "base_ceiling": 110.0, "fair_value_blended": 120.0,
              "valuation_confidence": "medium", "street_consensus_target": 130.0,
              "rungs": [{"price": 95.0, "detail": "ma50"}, {"price": 80.0, "detail": "ma200"},
                        {"price": 105.0, "detail": "swing_high"}],
              "computed_at": "2026-10-27 21:00"}
    with patch("agents.price_action.compute_price_action", return_value=None), \
         patch("agents.price_action.compute_trend_template", return_value={"pass": True, "reasoning": "8/8"}), \
         patch("agents.chase_risk.compute_chase_risk", return_value={"pct_from_kijun": 4.0, "score": 3}), \
         patch("db.get_trade_ladder", return_value=ladder):
        text = context_engine._price_structure_section("XYZ", 100.0)
    assert "Trend Template: PASS" in text and "+4.0%" in text and "ATR(14): $5.00 (5.0% of price)" in text
    assert "Support: $95.00 (ma50); $80.00 (ma200)" in text
    assert "Resistance: $105.00 (swing_high)" in text


def test_empty_when_nothing_available():
    with patch("agents.price_action.compute_price_action", return_value=None), \
         patch("agents.price_action.compute_trend_template", return_value=None), \
         patch("agents.chase_risk.compute_chase_risk", return_value=None), \
         patch("db.get_trade_ladder", return_value=None):
        assert context_engine._price_structure_section("XYZ", 100.0) == ""
