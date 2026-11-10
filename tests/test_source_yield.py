"""Daily source_yield rows: new items per ingestion source against the prior
weekdays (audit 2026-11-07 17:17 #6). Log-only."""
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks


def _counts(src, per_day: dict):
    return [{"source": src, "day": d, "n": n} for d, n in per_day.items()]


def test_summary_baseline_is_prior_weekdays_and_flags_a_collapse():
    # 2026-11-06 is a Friday; prior weekdays 11-05 .. 10-30.
    days = {date(2026, 11, d): 100 for d in (2, 3, 4, 5)} | {date(2026, 10, 30): 100}
    counts = _counts("news", days | {date(2026, 11, 6): 40}) + _counts("edgar", {date(2026, 11, 6): 3})
    s = tasks._yield_summary(date(2026, 11, 6), counts)
    assert s["news"]["baseline_mean"] == 100 and s["news"]["n"] == 40 and s["news"]["drop"] is True
    assert s["news"]["baseline_days"][0] == "2026-11-05" and "2026-11-01" not in s["news"]["baseline_days"]
    # A small source is never flagged.
    assert s["edgar"]["baseline_mean"] == 0 and s["edgar"]["drop"] is False and s["edgar"]["ratio"] is None


def test_weekends_are_not_flagged():
    days = {date(2026, 11, d): 100 for d in (2, 3, 4, 5, 6)}
    s = tasks._yield_summary(date(2026, 11, 7), _counts("news", days))
    assert s["news"]["n"] == 0 and s["news"]["drop"] is False


class _Now(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 9, 18, 20, tzinfo=ZoneInfo("America/New_York"))


def test_task_writes_each_unwritten_day_once():
    with patch("datetime.datetime", _Now), \
         patch("db.get_activity_shadow_keys", return_value={"live:2026-11-07"}), \
         patch("db.get_seen_item_counts", return_value=_counts("news", {date(2026, 11, 6): 5})), \
         patch("db.TRADING_ENV", "live"), \
         patch("db.log_activity_direct") as write:
        tasks.source_yield(days_back=3)
    days = [c.args[4]["day"] for c in write.call_args_list]
    assert days == ["2026-11-06", "2026-11-08"]
    assert write.call_args_list[0].args[0] == "source_yield"
    assert write.call_args_list[0].args[4]["sources"]["news"]["n"] == 5


def test_task_skips_on_failed_reads():
    with patch("db.get_activity_shadow_keys", return_value=None), patch("db.log_activity_direct") as w:
        tasks.source_yield()
        assert not w.called
    with patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.get_seen_item_counts", return_value=None), patch("db.log_activity_direct") as w:
        tasks.source_yield()
        assert not w.called


def test_beat_entry():
    e = tasks.app.conf.beat_schedule["source-yield"]
    assert e["task"] == "tasks.source_yield" and e["options"]["queue"] == "ops"


def test_a_zero_day_is_skipped_from_the_baseline():
    """Professor 68a: a stack-down weekday would pull the baseline down."""
    days = {date(2026, 11, d): 100 for d in (2, 3, 5)} | {date(2026, 10, 30): 100, date(2026, 10, 29): 100}
    s = tasks._yield_summary(date(2026, 11, 6), _counts("news", days | {date(2026, 11, 6): 90}))
    assert s["news"]["baseline_skipped_days"] == ["2026-11-04"]
    assert s["news"]["baseline_mean"] == 100 and "2026-10-29" in s["news"]["baseline_days"]
