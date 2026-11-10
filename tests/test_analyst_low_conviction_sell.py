"""Low-conviction sells (2026-10-27). The conviction >= 5 pass_to_risk rule
used to drop every TRIM/EXIT whose thesis was weakening or broken. Now a
confirmed low reading (previous verdict also <= 4) passes; EXIT needs <= 2,
and an EXIT at 3-4 is reduced to TRIM."""
import json
from contextlib import ExitStack
from unittest.mock import patch

from agents.analyst import _low_conviction_sell

_CANDIDATE = {"ticker": "QBTS", "role": "Moonshot", "tier": "Watch", "sector": "Quantum"}


def _prev(conviction):
    return {"signal": "TRIM", "confidence": conviction / 10, "timestamp": None}


def _analyze(action, conviction, prev, llm_pass=False):
    response = json.dumps({
        "action": action, "conviction": conviction, "thesis_strength": 2, "catalyst_quality": 2,
        "valuation_risk_reward": 2, "data_completeness": 3, "thesis_bucket": "Moonshot",
        "thesis_summary": "test", "expectation_gap": {}, "catalyst": {}, "valuation": {},
        "upside_downside_ratio": 0.5, "key_risks": [], "pass_to_risk": llm_pass,
        "reasoning": "per Q3 fundamentals, backlog fell 30%",
        "sources": [{"type": "news", "detail": "a"}, {"type": "fundamentals_data", "detail": "b"}],
    })
    patches = [
        patch("agents.analyst._call_ollama", return_value=response),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("agents.market_regime.get_cached_regime",
              return_value={"regime": "Neutral", "score": 50, "reasoning": ""}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("agents.price_action.compute_trend_template", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}),
        patch("db.get_latest_signal_for", return_value=prev),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.log_signal"),
        patch("feed.emit"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from agents.analyst import analyze
        return analyze(_CANDIDATE, situation_brief="brief")


class TestLowConvictionSellRule:
    def test_trim_passes_when_previous_verdict_was_also_low(self):
        with patch("db.get_latest_signal_for", return_value=_prev(3)):
            assert _low_conviction_sell("QBTS", "TRIM", 4)[:2] == ("TRIM", True)

    def test_trim_held_when_previous_verdict_was_high(self):
        with patch("db.get_latest_signal_for", return_value=_prev(7)):
            assert _low_conviction_sell("QBTS", "TRIM", 3)[:2] == ("TRIM", False)

    def test_trim_held_with_no_previous_verdict(self):
        with patch("db.get_latest_signal_for", return_value=None):
            assert _low_conviction_sell("QBTS", "TRIM", 3)[:2] == ("TRIM", False)

    def test_exit_at_three_becomes_trim(self):
        with patch("db.get_latest_signal_for", return_value=_prev(4)):
            assert _low_conviction_sell("QBTS", "EXIT", 3)[:2] == ("TRIM", True)

    def test_exit_at_two_stays_exit(self):
        with patch("db.get_latest_signal_for", return_value=_prev(3)):
            assert _low_conviction_sell("QBTS", "EXIT", 2)[:2] == ("EXIT", True)


class TestAnalyzeWiring:
    def test_confirmed_low_trim_passes_to_risk_even_if_llm_said_false(self):
        result = _analyze("TRIM", 3, _prev(4))
        assert result["action"] == "TRIM"
        assert result["pass_to_risk"] is True

    def test_unconfirmed_low_exit_does_not_pass(self):
        result = _analyze("EXIT", 2, _prev(6))
        assert result["pass_to_risk"] is False

    def test_open_rules_unchanged(self):
        result = _analyze("OPEN", 4, _prev(4), llm_pass=True)
        assert result["pass_to_risk"] is False


def test_stale_confirming_verdict_does_not_count():
    from datetime import datetime, timedelta, timezone
    old = {"signal": "TRIM", "confidence": 0.3,
           "timestamp": datetime.now(timezone.utc) - timedelta(days=20)}
    with patch("db.get_latest_signal_for", return_value=old):
        assert _low_conviction_sell("QBTS", "TRIM", 3)[:2] == ("TRIM", False)


def test_sell_confirm_record_measures_spacing_item_and_position():
    """Audit 11-06 23:24 #4: record-only fields on the confirming verdict."""
    from datetime import datetime, timedelta, timezone
    from agents.analyst import _sell_confirm_record
    ts = datetime.now(timezone.utc) - timedelta(seconds=90)
    prev = {"timestamp": ts, "id": 77, "confidence": 0.3,
            "trigger": {"path": "headline", "url": "https://x/a", "title": "T"}}
    with patch("db.get_position_stop", return_value={"opened_at": ts + timedelta(seconds=30)}):
        rec = _sell_confirm_record("QBTS", prev, {"path": "headline", "url": "https://x/a", "title": "U"})
    assert rec["prev_signal_id"] == 77 and rec["prev_trigger_path"] == "headline"
    assert 85 <= rec["confirm_gap_s"] <= 120
    assert rec["confirm_same_item"] is True and rec["confirm_before_open"] is True
    with patch("db.get_position_stop", side_effect=RuntimeError("pool")):
        rec = _sell_confirm_record("QBTS", {**prev, "trigger": None}, None)
    assert rec["confirm_same_item"] is None and rec["confirm_before_open"] is None
    assert _sell_confirm_record("QBTS", None, None)["confirm_gap_s"] is None


def test_low_conviction_sell_fills_the_record_and_reads_the_trigger():
    rec = {}
    with patch("db.get_latest_signal_for", return_value=_prev(3)) as get, \
         patch("db.get_position_stop", return_value=None):
        _low_conviction_sell("QBTS", "TRIM", 3, {"path": "scheduled"}, rec)
    assert get.call_args.kwargs == {"with_trigger": True}
    assert set(rec) >= {"confirm_gap_s", "confirm_same_item", "confirm_before_open"}


def test_sell_confirm_stays_on_the_verdict_row():
    from agents import crew
    fields = crew._route_row_fields({"provenance": {"sell_confirm": {"confirm_gap_s": 1.0}, "model": "m"}},
                                    "SELL_NO_POSITION")
    assert "sell_confirm" not in (fields.get("provenance") or {})
