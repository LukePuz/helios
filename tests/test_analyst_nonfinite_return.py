"""A NaN/inf expected_return_pct or expected_downside_pct (json.loads reads
them) is read as missing, and the approval insert drops any non-finite
value left in analyst_json: Postgres jsonb rejects NaN, and a failed insert
left a signals row with pass_to_risk True and no approval."""
import json
import math
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import db


def _analyze(payload: dict) -> dict:
    from agents import analyst
    with ExitStack() as stack:
        for p in (patch("agents.analyst._call_ollama", return_value=json.dumps(payload)),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("db.log_signal", return_value=7), patch("feed.emit")):
            stack.enter_context(p)
        return analyst.analyze({"ticker": "SNOW", "role": "Compounder"}, situation_brief="b",
                               regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})


def _approval_params(analyst_json: dict) -> tuple:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (41,)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_pending_approval(
            ticker="SNOW", action="OPEN", conviction=8, bucket="Compounder", size_pct=0.05,
            stop_price=150.0, entry_strategy="s", reasoning="r",
            analyst_json=analyst_json, risk_json={"shares_initial": 1}) == 41
    return cur.execute.call_args.args[1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_return_on_an_undated_open_is_missing_and_the_approval_has_no_nan(bad):
    payload = {
        "action": "OPEN", "conviction": 8, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "catalyst": {"event": "earnings", "is_priced_in": False},
        "valuation": {"fair_value": bad}, "expected_return_pct": bad,
        "expected_downside_pct": bad, "upside_downside_ratio": 3.0, "key_risks": [],
        "pass_to_risk": True, "reasoning": "per Q2 fundamentals, real evidence",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    }
    result = _analyze(payload)
    assert result["action"] == "OPEN" and result["pass_to_risk"] is True
    assert result["expected_return_pct"] is None and result["expected_downside_pct"] is None
    # A non-finite value the analyst passes through untouched (valuation)
    # is dropped at the approval insert, so jsonb accepts the row.
    assert math.isnan(result["valuation"]["fair_value"]) or math.isinf(result["valuation"]["fair_value"])
    analyst_param = _approval_params(result)[9]
    stored = analyst_param.adapted
    json.dumps(stored, allow_nan=False, default=str)        # raises on any NaN/inf
    assert stored["expected_return_pct"] is None and stored["valuation"]["fair_value"] is None
    assert "NaN" not in analyst_param.dumps(stored) and "Infinity" not in analyst_param.dumps(stored)


def test_finite_return_is_unchanged():
    payload = {"action": "WATCH", "conviction": 5, "reasoning": "r", "sources": [],
               "expected_return_pct": "12.5", "expected_downside_pct": -4}
    result = _analyze(payload)
    assert result["expected_return_pct"] == 12.5 and result["expected_downside_pct"] == -4.0


def test_a_non_finite_sizing_value_is_dropped_from_the_approval_payload():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (41,)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_pending_approval(
            ticker="SNOW", action="OPEN", conviction=8, bucket="Compounder", size_pct=0.05,
            stop_price=150.0, entry_strategy="s", reasoning="r", analyst_json={},
            risk_json={"shares_initial": 1, "atr": float("nan"), "levels": [float("inf")]}) == 41
    risk_param = cur.execute.call_args.args[1][10]
    json.dumps(risk_param.adapted, allow_nan=False)
    assert risk_param.adapted == {"shares_initial": 1, "atr": None, "levels": [None]}
