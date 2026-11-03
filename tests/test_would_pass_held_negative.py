"""Log-only would_pass_held_negative shadow (audit 2026-11-02 19:27 #4,
roadmap 01(c)): a held name's negative event that the screener still drops
is recorded in the screener provenance; the decision is unchanged."""
from unittest.mock import patch

from agents import screener

_NO = '{"relevant": false, "urgency": "eod", "reason": "no new fact"}'


def _screen(title, source="news", held=True, reply=_NO, **event):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value={"entry_price": 1.0} if held else None):
        return screener.screen({"ticker": "NOW", "source": source, "title": title, "text": "x",
                                **event})


def test_held_negative_dropped_by_the_override_cooldown_is_recorded():
    first = _screen("ServiceNow shares tumble on weak guidance")
    assert first["relevant"] and first["provenance"]["would_pass_held_negative"] is None
    r = _screen("ServiceNow slides after peer warns")
    assert not r["relevant"] and r["override_skipped"] == "cooldown"
    s = r["provenance"]["would_pass_held_negative"]
    assert s["would_pass"] is True and s["negative_signals"] == ["tone_neg"]
    assert s["override_skipped"] == "cooldown"


def test_bearish_options_flow_on_a_held_name_is_a_would_pass():
    r = _screen("NOW options flow: BEARISH", source="options_flow", sentiment="BEARISH")
    assert not r["relevant"]
    s = r["provenance"]["would_pass_held_negative"]
    assert s["would_pass"] is True and s["negative_signals"] == ["sentiment_bearish"]


def test_rating_down_and_pt_cut_are_read_from_the_ratings_title():
    hits = screener._would_pass_held_negative(
        {"relevant": False}, {}, "analyst_ratings",
        "Mizuho down NOW: Buy → Neutral, price target lowers to $150 (from $180)", True)
    assert hits["would_pass"] and {"rating_down", "pt_lowered", "tone_neg"} <= set(hits["negative_signals"])


def test_neutral_held_events_record_false_and_unheld_or_passed_record_none():
    r = _screen("ServiceNow to present at conference")
    assert not r["relevant"]
    assert r["provenance"]["would_pass_held_negative"]["would_pass"] is False
    r = _screen("ServiceNow shares tumble", held=False)
    assert not r["relevant"] and r["provenance"]["would_pass_held_negative"] is None
    r = _screen("ServiceNow shares tumble",
                reply='{"relevant": true, "urgency": "eod", "reason": "guidance"}')
    assert r["relevant"] and r["provenance"]["would_pass_held_negative"] is None


def test_the_screener_decision_and_prompt_do_not_change():
    with patch.object(screener, "_would_pass_held_negative", return_value={"would_pass": True}):
        with_shadow = _screen("ServiceNow to present at conference")
    without = _screen("ServiceNow to present at conference")
    for k in ("relevant", "urgency", "affected_tickers", "reason", "model_used"):
        assert with_shadow.get(k) == without.get(k)
    assert with_shadow["provenance"]["prompt_sha"] == without["provenance"]["prompt_sha"]


def test_the_decision_does_not_change_when_the_shadow_hits():
    """m271: the case above never runs the real shadow. Here the real shadow
    takes its hit path (a dropped bearish flow on a held name) and the
    decision matches a run with the shadow switched off."""
    kw = dict(title="NOW options flow: BEARISH", source="options_flow", sentiment="BEARISH")
    hit = _screen(**kw)
    assert hit["provenance"]["would_pass_held_negative"]["would_pass"] is True
    with patch.object(screener, "_would_pass_held_negative", return_value=None):
        off = _screen(**kw)
    for k in ("relevant", "urgency", "affected_tickers", "reason", "model_used", "override_skipped"):
        assert hit.get(k) == off.get(k), k
    assert hit["provenance"]["prompt_sha"] == off["provenance"]["prompt_sha"]


def test_swallows_its_own_errors():
    class Bad(dict):
        def get(self, *a):
            raise ValueError("boom")
    assert "error" in screener._would_pass_held_negative(Bad(), {}, "news", "t", True)
