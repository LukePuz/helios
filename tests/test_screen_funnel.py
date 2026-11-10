"""Daily screen_funnel rows: screener pass counts by source, tier, held and
prompt version (audit 2026-11-07 15:07 #1). Log-only."""
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks
from agents import screener


def _row(sig, tier, n, source="news", held=False, sha="abc", model="qwen2.5:7b", override=None):
    return {"signal": sig, "source": source, "tier": tier, "held": held,
            "model": model, "prompt_template_sha": sha, "override": override, "n": n}


def test_screen_provenance_records_tier_and_held():
    with patch("agents.screener._call_ollama", return_value='{"relevant": false, "urgency": "eod", "reason": "x"}'), \
         patch("feed.emit"), patch("db.get_position_stop", return_value={"entry_price": 1.0}), \
         patch("agents.screener._get_tier", return_value="Speculative"):
        prov = screener.screen({"ticker": "RGTI", "source": "news", "title": "t", "text": "x"})["provenance"]
    assert prov["tier"] == "Speculative" and prov["held"] is True


def test_summary_and_swings():
    day = tasks._funnel_summary([_row("SCREENED_IN", "Core", 30), _row("SCREENED_OUT", "Core", 70),
                                 _row("SCREENED_OUT", None, 5)])
    assert day["Core"] == {"in": 30, "out": 70, "rate": 0.3, "fallback": 0, "override": 0}
    assert day["unknown"] == {"in": 0, "out": 5, "rate": 0.0, "fallback": 0, "override": 0}
    prior = {"Core": {"in": 80, "out": 920, "rate": 0.08}, "unknown": {"in": 0, "out": 100, "rate": 0.0}}
    assert tasks._funnel_swings(day, prior) == [{"tier": "Core", "rate": 0.3, "prior_rate": 0.08}]
    # Small samples and small moves are not flagged.
    assert tasks._funnel_swings({"Core": {"in": 3, "out": 7, "rate": 0.3}}, prior) == []
    assert tasks._funnel_swings({"Core": {"in": 9, "out": 91, "rate": 0.09}}, prior) == []


def test_summary_counts_7b_verdicts_only():
    """Professor m622: keyword-fallback screens and held overrides are counted
    apart from the 7B's own pass rate."""
    day = tasks._funnel_summary([
        _row("SCREENED_IN", "Speculative", 1), _row("SCREENED_OUT", "Speculative", 9),
        _row("SCREENED_OUT", "Speculative", 40, model="keyword_fallback"),
        _row("SCREENED_IN", "Speculative", 2, model="keyword_fallback"),
        _row("SCREENED_IN", "Speculative", 3, override="held_negative_or_rating")])
    assert day["Speculative"] == {"in": 1, "out": 9, "rate": 0.1, "fallback": 42, "override": 3}


class _Now(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 9, 5, 40, tzinfo=ZoneInfo("America/New_York"))


def test_task_writes_unwritten_days_once():
    calls = []
    with patch("datetime.datetime", _Now), \
         patch("db.get_activity_shadow_keys", return_value={"paper:2026-11-07"}) as keys, \
         patch("db.get_screen_funnel", side_effect=lambda s, e: calls.append((s, e)) or
               [_row("SCREENED_IN", "Core", 2), _row("SCREENED_OUT", "Core", 8)]), \
         patch("db.TRADING_ENV", "paper"), \
         patch("db.log_activity_direct") as write:
        tasks.screen_funnel(days_back=3)
    keys.assert_called_once_with("screen_funnel")
    days = [c.args[4]["day"] for c in write.call_args_list]
    assert days == ["2026-11-06", "2026-11-08"]
    extra = write.call_args_list[0].args[4]
    assert extra["shadow_key"] == "paper:2026-11-06" and extra["by_tier"]["Core"]["rate"] == 0.2
    assert datetime.fromisoformat(extra["computed_at"]) == datetime(2026, 11, 9, 10, 40, tzinfo=ZoneInfo("UTC"))
    assert write.call_args_list[0].args[0] == "screen_funnel"
    start, end = calls[0]
    assert start.date() == date(2026, 11, 6) and (end - start).days == 1


def test_task_skips_on_unreadable_counts():
    with patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_screen_funnel", return_value=None), \
         patch("db.log_activity_direct") as write:
        tasks.screen_funnel(days_back=2)
        assert not write.called
    with patch("db.get_activity_shadow_keys", return_value=None), \
         patch("db.log_activity_direct") as write:
        tasks.screen_funnel()
        assert not write.called


def test_beat_entry():
    e = tasks.app.conf.beat_schedule["screen-funnel"]
    assert e["task"] == "tasks.screen_funnel" and e["options"]["queue"] == "ops"


def test_spilled_day_is_not_written_again():
    tasks._FUNNEL_SPILLED.clear()
    rows = [_row("SCREENED_IN", "Core", 1)]
    with patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_screen_funnel", return_value=rows), \
         patch("db.log_activity_direct", return_value=False) as write:
        tasks.screen_funnel(days_back=1)
        tasks.screen_funnel(days_back=1)
    assert write.call_count == 1
    tasks._FUNNEL_SPILLED.clear()
