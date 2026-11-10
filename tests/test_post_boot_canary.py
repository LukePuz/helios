"""Post-boot canary: one row at 10:30 ET on NYSE days saying whether the
booted code is the code on disk and whether screening and verdicts ran
since 05:00 (audit 2026-11-07 19:07 #6). Log-only."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import tasks

RUN = {"code_sha": "abc", "head_sha": "abc"}
OK_COUNTS = {"screens": 40, "verdicts": 3, "analyst_failed": 0,
             "boots": [{"service": "llm", "code_sha": "abc"}, {"service": "ops", "code_sha": "abc"}]}


def test_ok_and_each_failure():
    assert tasks._canary_verdict(RUN, RUN, OK_COUNTS) == {"ok": True, "failed": [], "warnings": []}
    v = tasks._canary_verdict(RUN, {"code_sha": "def", "head_sha": "def"}, OK_COUNTS)
    assert v["failed"] == ["code_on_disk_differs_from_running"]
    v = tasks._canary_verdict(RUN, {"code_sha": "abc-dirty", "head_sha": "abc"}, OK_COUNTS)
    assert v["failed"] == ["dirty_tree"]
    v = tasks._canary_verdict(RUN, RUN, {**OK_COUNTS, "screens": 0, "verdicts": 0, "analyst_failed": 2,
                                         "boots": [{"code_sha": "abc"}, {"code_sha": "x9"}]})
    assert v["failed"] == ["no_screens_since_0500", "mixed_boot_code_sha"]
    assert v["warnings"] == ["no_verdicts_since_0500", "analyst_failed"]
    assert tasks._canary_verdict(RUN, RUN, None)["failed"] == ["counts_unreadable"]
    # A research-log-only commit moves head_sha but not code_sha.
    assert tasks._canary_verdict(RUN, {"code_sha": "abc", "head_sha": "rl1"}, OK_COUNTS)["ok"]


class _Cur:
    def __init__(self):
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, params=None):
        self.sql.append((q, params))

    def fetchone(self):
        return (5, 0, 0)

    def fetchall(self):
        return []


class _Conn:
    def __init__(self):
        self.cur = _Cur()

    def cursor(self):
        return self.cur


def test_verdict_count_excludes_screens_and_degraded_rows():
    import db
    conn = _Conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.get_canary_counts(datetime(2026, 11, 9, 5, tzinfo=ZoneInfo("America/New_York")))
    q, params = conn.cur.sql[1]
    assert "signal <> 'SCREENED_IN'" in q and "NOT LIKE" in q
    assert params[1] == db.DEGRADED_SIGNAL_PREFIX + "%"
    assert "? %s" in q and params[2] == db._VERDICT_COPY_KEY == "route_exit"
    # Only screens since 05:00: verdicts is 0, so the warning fires.
    assert out["verdicts"] == 0
    assert "no_verdicts_since_0500" in tasks._canary_verdict(RUN, RUN, out)["warnings"]


class _Now(datetime):
    day = (2026, 11, 9)

    @classmethod
    def now(cls, tz=None):
        return datetime(*cls.day, 10, 30, tzinfo=ZoneInfo("America/New_York"))


def test_task_writes_a_row_and_logs_an_error_on_failure(caplog):
    with patch("datetime.datetime", _Now), \
         patch("agents.analyst._git_code_sha", return_value={"code_sha": "zzz", "head_sha": "zzz"}), \
         patch("agents.analyst._CODE_SHA", RUN), \
         patch("db.get_canary_counts", return_value=OK_COUNTS), \
         patch("db.log_activity_direct") as write:
        tasks.post_boot_canary()
    stage, _, msg, sentiment, extra = write.call_args.args
    assert stage == "post_boot_canary" and sentiment == "alert"
    assert extra["failed"] == ["code_on_disk_differs_from_running"] and extra["day"] == "2026-11-09"
    assert any("post-boot canary failed" in r.message and r.levelname == "ERROR" for r in caplog.records)


def test_holidays_and_weekends_are_skipped():
    _Now.day = (2026, 11, 26)          # Thanksgiving
    try:
        with patch("datetime.datetime", _Now), patch("db.log_activity_direct") as write:
            tasks.post_boot_canary()
        assert not write.called
    finally:
        _Now.day = (2026, 11, 9)


def test_never_raises():
    with patch("datetime.datetime", _Now), \
         patch("db.get_canary_counts", side_effect=RuntimeError("db")), \
         patch("db.log_activity_direct") as write:
        tasks.post_boot_canary()
    assert not write.called


def test_beat_entry():
    e = tasks.app.conf.beat_schedule["post-boot-canary"]
    assert e["task"] == "tasks.post_boot_canary" and e["options"]["queue"] == "ops"


def _stop(t, qty, status="Submitted", remaining=None, otype="STP"):
    return {"ticker": t, "action": "SELL", "order_type": otype, "qty": qty,
            "status": status, "remaining": remaining}


def test_stop_coverage_ratios():
    pos = {"NVDA": {"qty": 10}, "msft": {"qty": 4}, "ANET": {"qty": 5}, "OLD": {"qty": 0}}
    orders = [_stop("NVDA", 10), _stop("MSFT", 4, remaining=2), _stop("ANET", 5, status="Cancelled"),
              {"ticker": "NVDA", "action": "BUY", "order_type": "LMT", "qty": 3, "status": "Submitted"}]
    c = tasks._stop_coverage(pos, orders)
    assert c["ratio"] == {"NVDA": 1.0, "MSFT": 0.5, "ANET": 0.0}
    assert c["uncovered"] == ["ANET", "MSFT"]
    assert tasks._stop_coverage(None, orders) == {"readable": False}
    assert tasks._stop_coverage(pos, None) == {"readable": False}


def test_uncovered_holding_fails_the_canary():
    with patch("datetime.datetime", _Now), \
         patch("agents.analyst._git_code_sha", return_value=RUN), \
         patch("agents.analyst._CODE_SHA", RUN), \
         patch("db.get_canary_counts", return_value=OK_COUNTS), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("tasks._snapshot_positions_or_none", return_value={"NVDA": {"qty": 10}}), \
         patch("db.log_activity_direct") as write:
        tasks.post_boot_canary()
    extra = write.call_args.args[4]
    assert extra["ok"] is False and "holdings_without_full_stop" in extra["failed"]
    assert extra["stop_coverage"]["ratio"] == {"NVDA": 0.0}
