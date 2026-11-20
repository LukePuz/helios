"""Record-only buy_confirm on OPEN/ADD verdicts: the verdict before this one
(action, conviction, gap, same item), read with _sell_confirm_record's
fields. Nothing gates on it; it stays on the verdict row."""
import json
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agents import analyst, crew

ITEM = {"path": "event", "url": "https://x/a", "title": "NVDA headline"}


def _prev(signal, conviction, seconds_ago=99, trigger=ITEM):
    return {"signal": signal, "confidence": conviction / 10.0, "id": 41,
            "timestamp": datetime.now(timezone.utc) - timedelta(seconds=seconds_ago),
            "trigger": trigger}


@pytest.mark.parametrize("prev_action", ["HOLD", "ADD"])
def test_add_after_hold_and_after_add(prev_action):
    with patch("db.get_latest_signal_for", return_value=_prev(prev_action, 7)) as get, \
         patch("db.get_position_stop", return_value=None):
        rec = analyst._buy_confirm_record("NVDA", dict(ITEM))
    assert get.call_args.kwargs == {"with_trigger": True, "strict": True}
    assert rec["prev_read"] == "ok"
    assert rec["prev_action"] == prev_action and rec["prev_conviction"] == 7
    assert 95 <= rec["confirm_gap_s"] <= 130
    assert rec["confirm_same_item"] is True and rec["prev_signal_id"] == 41
    assert rec["prev_trigger_path"] == "event"


def test_different_item_and_no_previous_verdict():
    with patch("db.get_latest_signal_for",
               return_value=_prev("HOLD", 6, trigger={"path": "event", "url": "https://x/b"})), \
         patch("db.get_position_stop", return_value=None):
        rec = analyst._buy_confirm_record("NVDA", dict(ITEM))
    assert rec["confirm_same_item"] is False and rec["prev_conviction"] == 6
    with patch("db.get_latest_signal_for", return_value=None):
        rec = analyst._buy_confirm_record("NVDA", dict(ITEM))
    assert rec["prev_read"] == "none" and rec["prev_action"] is None
    assert rec["confirm_gap_s"] is None


def test_failed_read_never_raises():
    with patch("db.get_latest_signal_for", side_effect=RuntimeError("pool")):
        rec = analyst._buy_confirm_record("NVDA", dict(ITEM))
    assert rec == {"prev_read": "error: RuntimeError", "prev_action": None, "prev_conviction": None}
    with patch("db.get_latest_signal_for", return_value={"signal": "HOLD", "confidence": "x"}):
        rec = analyst._buy_confirm_record("NVDA", None)
    assert rec["prev_read"] == "ok" and rec["error"] == "ValueError"


def _analyze(action, prev):
    response = json.dumps({
        "action": action, "conviction": 7, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "x", "expectation_gap": {}, "catalyst": {}, "valuation": {},
        "upside_downside_ratio": 3.0, "key_risks": [], "pass_to_risk": action in ("OPEN", "ADD"),
        "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"},
                    {"type": "price_data", "detail": "c"}]})
    cand = {"ticker": "NVDA", "role": "Compounder", "tier": "Core", "sector": "Semis",
            "held": action == "ADD"}
    with ExitStack() as st:
        for p in (patch("agents.analyst._call_ollama", return_value=response),
                  patch("knowledge_loader.load_thesis", return_value=""),
                  patch("knowledge_loader.load_philosophy", return_value=""),
                  patch("agents.market_regime.get_cached_regime",
                        return_value={"regime": "Neutral", "score": 50, "reasoning": ""}),
                  patch("agents.chase_risk.compute_chase_risk", return_value=None),
                  patch("agents.price_action.compute_price_action", return_value=None),
                  patch("agents.price_action.compute_trend_template", return_value=None),
                  patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}),
                  patch("db.get_latest_signal_for", return_value=prev),
                  patch("db.get_position_stop", return_value=None),
                  patch("db.get_conviction_calibration", return_value=[]),
                  patch("feed.emit")):
            st.enter_context(p)
        log_signal = st.enter_context(patch("db.log_signal"))
        analyst.analyze(cand, situation_brief="Current Price: $100.00\nbrief",
                        trigger=analyst.trigger_record("event", {"source": "news",
                                                                 "title": "NVDA headline"}))
    return log_signal.call_args.kwargs


@pytest.mark.parametrize("action", ["OPEN", "ADD"])
def test_buy_verdict_carries_buy_confirm(action):
    kw = _analyze(action, _prev("HOLD", 7))
    rec = kw["provenance"]["buy_confirm"]
    assert rec["prev_action"] == "HOLD" and rec["prev_conviction"] == 7
    assert rec["prev_read"] == "ok"


def test_hold_verdict_has_no_buy_confirm():
    kw = _analyze("HOLD", _prev("ADD", 7))
    assert "buy_confirm" not in kw["provenance"]


def test_buy_confirm_stays_on_the_verdict_row():
    fields = crew._route_row_fields({"provenance": {"buy_confirm": {"confirm_gap_s": 1.0},
                                                    "model": "m"}}, "ROUTE_ZERO_SHARES")
    assert "buy_confirm" not in (fields.get("provenance") or {})
    assert "buy_confirm" in crew._VERDICT_ONLY_PROVENANCE
