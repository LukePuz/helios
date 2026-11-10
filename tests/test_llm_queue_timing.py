"""The verdict's trigger record carries when the event joined the LLM queue
and when its analysis started (audit 2026-11-07 05:07 #5). Log-only."""
from unittest.mock import patch

import tasks
from agents import analyst


def test_trigger_record_reads_the_queue_times_and_wait():
    rec = analyst.trigger_record("event", {"title": "t", "_llm_enqueued_at": "2026-11-05T14:00:00+00:00",
                                           "_llm_started_at": "2026-11-05T14:11:48+00:00"})
    assert rec["llm_enqueued_at"] == "2026-11-05T14:00:00+00:00"
    assert rec["llm_started_at"] == "2026-11-05T14:11:48+00:00"
    assert rec["llm_queue_wait_s"] == 708.0


def test_no_queue_keys_off_the_queue_and_bad_times_give_no_wait():
    assert "llm_queue_wait_s" not in analyst.trigger_record("event", {"title": "t"})
    rec = analyst.trigger_record("event", {"_llm_started_at": "not a time", "_llm_enqueued_at": "x"})
    assert rec["llm_queue_wait_s"] is None
    rec = analyst.trigger_record("event", {"_llm_started_at": "2026-11-05T14:00:00+00:00"})
    assert rec["llm_enqueued_at"] is None and rec["llm_queue_wait_s"] is None


def test_screen_pass_stamps_the_enqueue_time():
    with patch("agents.crew.screen_event_only", return_value={"relevant": True}), \
         patch("tasks._held_tickers", return_value=set()), \
         patch.object(tasks.deep_analyze_event, "apply_async") as send:
        tasks.screen_event.run({"ticker": "NVDA", "title": "t"})
    ev = send.call_args.kwargs["args"][0]
    assert ev["_llm_enqueued_at"] and ev["ticker"] == "NVDA"


def test_deep_analyze_stamps_the_start_time_without_touching_the_caller_dict():
    original = {"ticker": "NVDA", "_llm_enqueued_at": "2026-11-05T14:00:00+00:00"}
    with patch("agents.crew.deep_analyze") as deep:
        tasks.deep_analyze_event.run(original, None)
    ev = deep.call_args.args[0]
    assert ev["_llm_started_at"] and "_llm_started_at" not in original
    assert ev["_llm_attempt"] == 0


def test_trigger_record_carries_the_attempt():
    rec = analyst.trigger_record("event", {"_llm_started_at": "2026-11-05T14:00:00+00:00", "_llm_attempt": 1})
    assert rec["llm_attempt"] == 1


def test_an_expired_event_keeps_its_enqueue_time():
    with patch("db.log_signal") as log_signal, \
         patch("agents.analyst.shadow_versions", return_value={}):
        assert tasks._log_event_lost("tasks.deep_analyze_event", "expired",
                                     [{"ticker": "NVDA", "title": "t",
                                       "_llm_enqueued_at": "2026-11-05T14:00:00+00:00"}])
        tasks._log_event_lost("tasks.screen_event", "expired", [{"ticker": "NVDA", "title": "t"}])
    first, second = log_signal.call_args_list
    assert first.kwargs["decision"]["llm_enqueued_at"] == "2026-11-05T14:00:00+00:00"
    assert "llm_enqueued_at" not in second.kwargs["decision"]
