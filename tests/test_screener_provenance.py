"""SCREENED_OUT rows carry the screener model and code version (SEF run 7)."""
from unittest.mock import patch

from agents import crew, screener


def test_screen_result_has_provenance():
    with patch("agents.screener._call_ollama", return_value='{"relevant": false, "urgency": "low", "reason": "noise"}'), \
         patch("feed.emit"):
        result = screener.screen({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"})
    prov = result["provenance"]
    assert prov["model"] == screener.SCREENER_MODEL and prov["code_sha"] == "test"
    assert prov["head_sha"] == "testhead"  # SEF run 12: was missing on screener rows
    assert prov["model_digest"] == "testdigest"
    assert len(prov["prompt_template_sha"]) == 12 and len(prov["prompt_sha"]) == 12


def test_screened_out_signal_logs_model_and_provenance():
    prov = {"code_sha": "abc", "model": "qwen2.5:7b"}
    with patch("vector_store.store"), \
         patch("agents.screener.screen", return_value={"relevant": False, "reason": "noise", "provenance": prov}), \
         patch("db.log_signal") as log_signal:
        crew.screen_event_only({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"})
    assert log_signal.call_args.kwargs == {"model": "qwen2.5:7b", "provenance": prov}


def test_title_tone_tags_cuts_and_raises():
    """A1853-1: log-only tone tag, so pass rates can be split by tone."""
    assert screener.title_tone("BTIG main NOW: Buy, price target lowers to $180") == "neg"
    assert screener.title_tone("JP Morgan main NOW: Overweight, price target raises to $160") == "pos"
    assert screener.title_tone("Why is ServiceNow stock falling today") == "neg"
    assert screener.title_tone("Stock beats estimates but guidance cuts") == "mixed"
    assert screener.title_tone("ServiceNow to present at conference") == "neutral"
    assert screener.title_tone(None) == "neutral"


def test_screen_provenance_records_title_tone():
    with patch("agents.screener._call_ollama", return_value='{"relevant": false, "urgency": "low", "reason": "noise"}'), \
         patch("feed.emit"):
        result = screener.screen({"ticker": "NOW", "source": "analyst_ratings",
                                  "title": "price target lowers to $180", "text": "x"})
    assert result["provenance"]["title_tone"] == "neg"


def _screen(title, source="analyst_ratings", held=True, reply='{"relevant": false, "urgency": "low", "reason": "no new fact"}'):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value={"entry_price": 1.0} if held else None):
        return screener.screen({"ticker": "NOW", "source": source, "title": title, "text": "x"})


def test_held_rating_change_or_bad_news_overrides_once_per_cooldown():
    """Luke D7: held names get analysed on PT cuts / bad news even when the 7B says no."""
    r = _screen("BTIG main NOW: Buy, price target lowers to $180")
    assert r["relevant"] and r["override"] == "held_negative_or_rating"
    r2 = _screen("Barclays main NOW: price target lowers to $170")
    assert not r2["relevant"] and r2["override_skipped"] == "cooldown"


def test_no_override_for_unheld_or_neutral_news():
    assert not _screen("BTIG main NOW: price target lowers to $180", held=False)["relevant"]
    assert not _screen("ServiceNow to present at conference", source="news")["relevant"]


def test_prompt_tells_the_screener_about_holdings_and_symmetry():
    with patch("agents.screener._call_ollama", return_value='{"relevant": true, "urgency": "eod", "reason": "x"}') as call, \
         patch("feed.emit"), patch("db.get_position_stop", return_value=None):
        screener.screen({"ticker": "NOW", "source": "news", "title": "t", "text": "x"})
    prompt = call.call_args.args[0]
    assert "We currently hold this stock: no" in prompt and "price-target CUT" in prompt


def test_recap_headline_does_not_use_up_a_rating_changes_override():
    """Review 4 #3: separate cooldowns for headlines and rating changes; the
    override is recorded in provenance (#4)."""
    r = _screen("Why Is ServiceNow Stock Falling Today?", source="news")
    assert r["relevant"] and r["provenance"]["override"] == "held_negative_or_rating"
    assert _screen("BTIG main NOW: Buy, price target lowers to $180")["relevant"]
