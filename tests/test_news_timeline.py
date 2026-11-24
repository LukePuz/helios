"""Record-only news timeline (agents/news_timeline.py, analyst._news_timeline_enqueue):
after each event/scan verdict, a side thread records the ticker's last 7
days of news_log headlines with their screener decision, tone and whether
each reached the prompt. The verdict path only queues a job (no I/O); the
verdict, its decision row and the prompt are unchanged with the record on
and off; the template hashes do not read the new code. Synthetic data."""
import importlib.util
import inspect
import json
import pathlib
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
from agents import analyst
from agents import news_timeline as nt

VERDICT_AT = datetime(2026, 11, 18, 16, 43, tzinfo=timezone.utc)      # 11:43 ET
GN = "https://news.google.com/rss/articles/"


def _n(hours_ago, title, url):
    return (VERDICT_AT - timedelta(hours=hours_ago), title, url)


def _s(hours_ago, signal, url, title, reason):
    return (VERDICT_AT - timedelta(hours=hours_ago), signal, url, title, reason)


CFO_YF = "https://finance.yahoo.com/markets/stocks/articles/ionq-sinks-8-inder-singh-151502791.html"
NEWS = [
    _n(1, "How Could IonQ Change Critical Communications Security?",
       "https://finance.yahoo.com/a/qkd.html"),
    _n(20, "IonQ: Quantum Progress Meets Unnecessary Shareholder Dilution (Rating Downgrade) "
           "- Seeking Alpha", GN + "dil"),
    _n(30, "Ionq stock hits 52-week low at 24.91 USD - Investing.com", GN + "low"),
    _n(120, "IonQ Sinks 8% as Inder Singh Exits Effective Immediately", CFO_YF),
    _n(121, "IonQ Sinks 8% as Inder Singh Exits Effective Immediately - Benzinga", GN + "cfo"),
    _n(122, "IonQ Wins DARPA Stage C", "https://finance.yahoo.com/a/darpa.html"),
    _n(123, "", "https://x/empty"),
]
SCREENS = [
    _s(0.9, "SCREENED_IN", "https://finance.yahoo.com/a/qkd.html",
       "How Could IonQ Change Critical Communications Security?", "Interoperability result"),
    _s(19.9, "SCREENED_OUT", GN + "dil", "IonQ: Quantum Progress Meets Unnecessary Shareholder "
       "Dilution (Rating Downgrade) - Seeking Alpha", "Opinion piece ... rating downgrade"),
    _s(119.9, "SCREENED_OUT", CFO_YF, "IonQ Sinks 8% as Inder Singh Exits Effective Immediately",
       "CFO exit, no specific impact on financials"),
    _s(118, "SCREENED_OUT", None, "IonQ Sinks 8% as Inder Singh Exits Effective Immediately",
       "generic management change"),
]
TRIGGER = {"path": "event", "url": "https://finance.yahoo.com/a/qkd.html",
           "title": "How Could IonQ Change Critical Communications Security?"}
PROMPT = ("...\n=== EVENT ===\nHow could IonQ change critical communications security? ...\n"
          "=== MEMORY ===\nIonQ Wins DARPA Stage C (2026-11-13)\n")


def _rec(**kw):
    return nt.build_record(kw.get("news", NEWS), kw.get("screens", SCREENS),
                           kw.get("prompt", PROMPT), kw.get("trigger", TRIGGER), cap=kw.get("cap", 10),
                           prompt_at=kw.get("prompt_at"))


class TestBuild:
    def test_counts_over_the_window(self):
        rec = _rec()
        assert rec["n_news_rows_7d"] == 7 and rec["n_screen_rows_7d"] == 4
        assert rec["n_items_7d"] == 5                    # the Benzinga copy folds, no-title row skipped
        assert rec["n_passed_7d"] == 1 and rec["n_rejected_7d"] == 2
        assert rec["n_no_screen_row_7d"] == 2
        assert rec["n_negative_7d"] == 2                 # downgrade, sinks; "52-week low" not in lexicon
        assert rec["n_negative_rejected_7d"] == 2 and rec["n_negative_in_prompt_7d"] == 0
        assert rec["n_in_prompt_7d"] == 2 and rec["n_positive_7d"] == 1

    def test_items_newest_first_with_their_fields(self):
        items = _rec()["items"]
        assert [i["date_et"] for i in items] == ["2026-11-18", "2026-11-17", "2026-11-17",
                                                 "2026-11-13", "2026-11-13"]
        trig, dil = items[0], items[1]
        assert trig == {"date_et": "2026-11-18",
                        "title": "How Could IonQ Change Critical Communications Security?",
                        "source": "finance.yahoo.com", "screen": "passed",
                        "screen_reason": "Interoperability result", "n_screens": 1,
                        "tone": "neutral", "in_prompt": True, "after_prompt": None,
                        "is_trigger": True, "n_copies": 1}
        assert dil["source"] == "gn:seeking alpha" and dil["screen"] == "rejected"
        assert dil["tone"] == "neg" and dil["in_prompt"] is False and dil["is_trigger"] is False

    def test_a_story_folds_its_copies_and_their_screens(self):
        cfo = next(i for i in _rec()["items"] if "Inder Singh" in i["title"])
        assert cfo["n_copies"] == 2 and cfo["n_screens"] == 2      # by url and by title
        assert cfo["screen"] == "rejected"
        assert cfo["screen_reason"] == "generic management change"   # the newest screen
        assert cfo["title"].endswith("Effective Immediately")      # the newest copy (yahoo)

    def test_a_pass_on_any_copy_reads_passed(self):
        screens = SCREENS + [_s(117, "SCREENED_IN", GN + "cfo", "x", "relevant now")]
        cfo = next(i for i in _rec(screens=screens)["items"] if "Inder Singh" in i["title"])
        assert cfo["screen"] == "passed" and cfo["screen_reason"] == "relevant now"

    def test_in_prompt_is_the_first_60_characters_any_case(self):
        long = "A" * 59 + "B tail that is not in the prompt"
        rec = _rec(news=[_n(1, long, "https://a/1")], screens=[], prompt="x " + "a" * 59 + "b y")
        assert rec["items"][0]["in_prompt"] is True
        rec = _rec(news=[_n(1, long, "https://a/1")], screens=[], prompt="a" * 59)
        assert rec["items"][0]["in_prompt"] is False

    def test_items_logged_after_the_prompt_are_flagged_and_counted_apart(self):
        """The window ends at the verdict; a headline first logged during
        inference is in it but could not have been in the prompt."""
        prompt_at = VERDICT_AT - timedelta(minutes=6)
        news = NEWS + [_n(0.05, "IonQ Falls 5% After $2 Billion Stock Sale", "https://a/sale")]
        rec = _rec(news=news, prompt_at=prompt_at)
        sale = next(i for i in rec["items"] if "Stock Sale" in i["title"])
        assert sale["after_prompt"] is True and sale["tone"] == "neg"
        assert all(i["after_prompt"] is False for i in rec["items"] if i is not sale)
        assert rec["n_items_7d"] == 6 and rec["n_items_before_prompt"] == 5
        assert rec["n_negative_7d"] == 3 and rec["n_negative_before_prompt"] == 2
        assert rec["n_negative_rejected_before_prompt"] == 2
        assert rec["n_in_prompt_7d"] == 2                  # unchanged

    def test_a_story_first_logged_before_the_prompt_is_before_it(self):
        """A re-logged copy after the prompt does not move the story."""
        news = [_n(0.01, "Same story - Benzinga", GN + "1"), _n(5, "Same story", "https://a/1")]
        rec = _rec(news=news, screens=[], prompt_at=VERDICT_AT - timedelta(minutes=5))
        assert rec["items"][0]["after_prompt"] is False and rec["n_items_before_prompt"] == 1

    def test_without_a_prompt_time_the_before_prompt_fields_are_none(self):
        rec = _rec()
        assert rec["items"][0]["after_prompt"] is None
        assert rec["n_items_before_prompt"] is None and rec["n_negative_before_prompt"] is None
        assert rec["n_negative_rejected_before_prompt"] is None

    def test_in_prompt_also_tries_the_title_without_its_publisher(self):
        title = "Nvidia Supplier Warns On Orders As Demand Cools Into The Holiday Quarter - Reuters"
        body = title.rsplit(" - ", 1)[0]
        short = "Supplier Warns On Orders - Reuters"
        rec = _rec(news=[_n(1, title, GN + "r"), _n(2, short, GN + "s")], screens=[],
                   prompt="memory: supplier warns on orders (11-17) ... " + body[:30].lower())
        flags = {i["title"]: i["in_prompt"] for i in rec["items"]}
        assert flags[short] is True                     # "- Reuters" is not in the prompt
        assert flags[title] is False                    # its first 60 chars are not either

    def test_in_prompt_reads_every_copy(self):
        news = [_n(1, "Fresh Rewrite Of The Story - CNBC", GN + "x"),
                _n(2, "fresh rewrite of the story", "https://a/x")]
        rec = _rec(news=news, screens=[], prompt="... Fresh rewrite of the story ...")
        assert rec["items"][0]["in_prompt"] is True

    def test_the_cap_keeps_the_newest(self):
        news = [_n(h, f"Story number {h}", f"https://a/{h}") for h in range(1, 31)]
        rec = _rec(news=news, screens=[], cap=10)
        assert rec["n_items_7d"] == 30 and len(rec["items"]) == 10
        assert rec["items"][0]["title"] == "Story number 1"
        assert rec["items"][-1]["title"] == "Story number 10"

    def test_titles_are_truncated(self):
        rec = _rec(news=[_n(1, "T" * 400, "https://a/1")], screens=[])
        assert len(rec["items"][0]["title"]) == 160

    def test_empty_and_bad_input(self):
        rec = _rec(news=[], screens=[], prompt=None, trigger=None)
        assert rec["n_items_7d"] == 0 and rec["items"] == []
        assert _rec(news=[(None, None, None), (1, "x", None)], screens=[(None,) * 5],
                    prompt=None)["n_items_7d"] == 1
        assert _rec(news=[("a", "b")])["build"].startswith("error: ")

    def test_record_is_plain_json(self):
        rec = _rec()
        assert json.loads(json.dumps(rec)) == rec


class TestJob:
    def _job(self, **kw):
        return {"ticker": "IONQ", "signal_id": 94689, "env": "live", "path": "event",
                "verdict_at": VERDICT_AT, "prompt_at": VERDICT_AT - timedelta(minutes=6),
                "prompt": PROMPT, "trigger": TRIGGER, **kw}

    def test_reads_the_closed_window_and_writes_one_row(self):
        read, write = MagicMock(return_value=(NEWS, SCREENS)), MagicMock()
        extra = nt.run_job(self._job(), read=read, write=write)
        assert read.call_args.args == ("IONQ", "live", VERDICT_AT - timedelta(days=7), VERDICT_AT)
        (stage, ticker, msg, sentiment, row), = [c.args for c in write.call_args_list]
        assert stage == "news_timeline" and ticker == "IONQ" and row is extra
        assert row["shadow_key"] == "live:94689" and row["signal_id"] == 94689
        assert row["read"] == "ok" and row["limit_hit"] is False and row["n_items_7d"] == 5
        assert row["tone_lexicon"] == "agents.screener.title_tone"
        from agents.screener import TONE_SHA
        assert row["tone_sha"] == TONE_SHA and row["window_days"] == 7 and row["cap"] == 10
        assert row["prompt_at"] == (VERDICT_AT - timedelta(minutes=6)).isoformat()
        assert row["n_items_before_prompt"] == 5
        assert "2 negative (2 screener-rejected)" in msg

    def test_the_row_goes_through_the_activity_writer(self):
        import db
        with patch.object(db, "log_activity_queued") as q, \
                patch.object(db, "log_activity_direct", side_effect=AssertionError("direct")):
            nt.run_job(self._job(), read=MagicMock(return_value=(NEWS, SCREENS)))
        (stage, ticker, msg, sentiment, extra), = [c.args for c in q.call_args_list]
        assert stage == "news_timeline" and extra["signal_id"] == 94689

    def test_a_job_without_a_prompt_time(self):
        extra = nt.run_job(self._job(prompt_at=None), read=MagicMock(return_value=(NEWS, SCREENS)),
                           write=MagicMock())
        assert extra["prompt_at"] is None and extra["n_items_before_prompt"] is None

    def test_a_failed_read_writes_a_row_with_the_error(self):
        write = MagicMock()
        extra = nt.run_job(self._job(), read=MagicMock(side_effect=RuntimeError("down")),
                           write=write)
        assert extra["read"] == "error: RuntimeError" and "n_items_7d" not in extra
        assert write.call_count == 1

    def test_the_default_read_uses_its_own_connection_and_is_read_only(self):
        cur = MagicMock()
        cur.fetchall.side_effect = [[("n",)], [("s",)]]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        import db
        with patch.object(db.psycopg2, "connect", return_value=conn) as connect, \
                patch.object(db, "_get_conn", side_effect=AssertionError("not the pool")):
            assert nt._read("IONQ", "live", VERDICT_AT - timedelta(days=7), VERDICT_AT) == \
                ([("n",)], [("s",)])
        assert connect.called and conn.rollback.called and conn.close.called
        kw = connect.call_args.kwargs
        assert kw["port"] == 6543 == db._TRANSACTION_POOLER_PORT       # never the session pooler
        assert kw["connect_timeout"] == 5 and kw["keepalives"] == 1
        assert (kw["keepalives_idle"], kw["keepalives_interval"], kw["keepalives_count"]) == (10, 5, 3)
        assert kw["tcp_user_timeout"] == 10000 and kw["sslmode"] == "require"
        screen_args = cur.execute.call_args_list[3].args[1]
        assert screen_args[0] == "IONQ"
        assert cur.execute.call_args_list[1].args == ("SET LOCAL statement_timeout = %s",
                                                      (nt.READ_TIMEOUT_MS,))
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert sqls[0] == "SET TRANSACTION READ ONLY"
        assert not any(w in s.upper() for s in sqls for w in ("INSERT", "UPDATE ", "DELETE"))
        news_args = cur.execute.call_args_list[2].args[1]
        assert news_args[0] == "IONQ" and news_args[-1] == nt.READ_LIMIT

    def test_both_reads_upper_case_the_ticker(self):
        cur = MagicMock()
        cur.fetchall.side_effect = [[], []]
        nt.read_inputs(cur, "ionq", "live", VERDICT_AT - timedelta(days=7), VERDICT_AT)
        assert cur.execute.call_args_list[1].args[1][0] == "IONQ"
        assert cur.execute.call_args_list[2].args[1][0] == "IONQ"

    @pytest.mark.parametrize("value, port", [(None, 6543), ("", 6543), ("5433", 5433),
                                             ("x", 6543)])
    def test_the_read_port_override(self, monkeypatch, value, port):
        if value is None:
            monkeypatch.delenv("NEWS_TIMELINE_READ_PORT", raising=False)
        else:
            monkeypatch.setenv("NEWS_TIMELINE_READ_PORT", value)
        assert nt.read_port() == port

    def test_the_read_ignores_the_bots_port(self, monkeypatch):
        """A bot on the session pooler (5432) still reads on 6543."""
        import db
        monkeypatch.setenv("SUPABASE_PORT", "5432")
        monkeypatch.delenv("NEWS_TIMELINE_READ_PORT", raising=False)
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchall.side_effect = [[], []]
        with patch.object(db.psycopg2, "connect", return_value=conn) as connect:
            nt._read("IONQ", "live", VERDICT_AT - timedelta(days=7), VERDICT_AT)
        assert connect.call_args.kwargs["port"] == 6543

    def test_never_raises(self):
        assert "job" in nt.run_job({"ticker": "X"}, read=MagicMock(), write=MagicMock())
        assert nt.run_job(self._job(), read=MagicMock(return_value=(NEWS, SCREENS)),
                          write=MagicMock(side_effect=RuntimeError("x")))["job"] == \
            "error: RuntimeError"


@pytest.mark.news_timeline_lane
class TestLane:
    def test_enqueue_returns_at_once_and_the_job_runs_on_its_own_thread(self, monkeypatch):
        started, release, seen = threading.Event(), threading.Event(), {}

        def slow(job):
            seen["thread"] = threading.current_thread().name
            started.set()
            release.wait(2)
        monkeypatch.setattr(nt, "run_job", slow)
        monkeypatch.setattr(nt, "_QUEUE", None)
        t0 = time.monotonic()
        assert nt.enqueue({"ticker": "X"}) is True
        assert time.monotonic() - t0 < 0.5
        assert started.wait(2) and seen["thread"] == "news-timeline"
        release.set()
        nt._QUEUE.join()

    def test_a_full_queue_drops_and_counts(self, monkeypatch):
        import queue
        monkeypatch.setattr(nt, "_QUEUE", queue.Queue(maxsize=1))
        monkeypatch.setattr(nt, "_WORKER", (__import__("os").getpid(), MagicMock(is_alive=lambda: True)))
        monkeypatch.setitem(nt.DROPPED, "queue_full", 0)
        assert nt.enqueue({"a": 1}) is True and nt.enqueue({"a": 2}) is False
        assert nt.DROPPED["queue_full"] == 1

    def test_a_dead_threads_queue_is_counted_when_replaced(self, monkeypatch):
        import os
        import queue
        old = queue.Queue()
        old.put({"a": 1})
        old.put({"a": 2})
        monkeypatch.setattr(nt, "_QUEUE", old)
        monkeypatch.setattr(nt, "_WORKER", (os.getpid(), MagicMock(is_alive=lambda: False)))
        monkeypatch.setitem(nt.DROPPED, "dead_thread", 0)
        monkeypatch.setattr(nt, "run_job", lambda job: None)
        assert nt.enqueue({"a": 3}) is True
        assert nt.DROPPED["dead_thread"] == 2 and nt._QUEUE is not old
        nt._QUEUE.join()

    def test_a_forked_child_does_not_count_the_parents_queue(self, monkeypatch):
        import queue
        old = queue.Queue()
        old.put({"a": 1})
        monkeypatch.setattr(nt, "_QUEUE", old)
        monkeypatch.setattr(nt, "_WORKER", (-1, MagicMock(is_alive=lambda: True)))
        monkeypatch.setitem(nt.DROPPED, "dead_thread", 0)
        monkeypatch.setattr(nt, "run_job", lambda job: None)
        assert nt.enqueue({"a": 2}) is True and nt.DROPPED["dead_thread"] == 0
        nt._QUEUE.join()

    def test_drain_drops_what_is_left_and_counts(self, monkeypatch):
        import os
        import queue
        q = queue.Queue()
        q.put({"a": 1})
        monkeypatch.setattr(nt, "_QUEUE", q)
        monkeypatch.setattr(nt, "_WORKER", (os.getpid(), MagicMock()))
        monkeypatch.setitem(nt.DROPPED, "at_exit", 0)
        assert nt.drain_now(timeout_s=0.05) == 1 and nt.DROPPED["at_exit"] == 1
        assert nt.drain_now() == 1 - 1                     # empty now

    def test_the_celery_child_exit_hook_drains_it_before_the_activity_writer(self):
        import tasks
        order = []
        with patch("agents.news_timeline.drain_now", side_effect=lambda: order.append("nt")), \
                patch("db.drain_activity_queue_now", side_effect=lambda: order.append("db")), \
                patch("ibkr.drain_commission_queue"):
            tasks._drain_on_child_exit()
        assert order == ["nt", "db"]

    def test_the_exit_hook_does_not_import_it(self, monkeypatch):
        import sys
        import tasks
        monkeypatch.delitem(sys.modules, "agents.news_timeline")
        with patch("db.drain_activity_queue_now") as dq, patch("ibkr.drain_commission_queue"):
            tasks._drain_on_child_exit()
        assert "agents.news_timeline" not in sys.modules and dq.called


# ---- the hook on the verdict path ----

_FUND_BRIEF = "FUNDAMENTALS:\n  Revenue growth (YoY): +20.0%\n  Forward P/E: 8.79x\n"


def _analyze(trigger=TRIGGER, signal_id=94689):
    parsed = {
        "action": "OPEN", "conviction": 7, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Moonshot",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-12-20"},
        "valuation": {"current_multiple": "8.79x", "peer_range": "6x to 12x",
                      "bull_case_target": "$130", "downside_scenario": "$90"},
        "expected_return_pct": 30.0, "expected_downside_pct": 10.0, "key_risks": [],
        "pass_to_risk": True, "upside_downside_ratio": 2.4, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a - Reuters"}, {"type": "news", "detail": "b"}],
    }
    log_signal = MagicMock(return_value=signal_id)
    ollama = MagicMock(return_value=json.dumps(parsed))
    enq = MagicMock(return_value=True)
    patches = [
        patch("agents.analyst._call_ollama", ollama),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Moonshot"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.get_trade_ladder", return_value=None),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
        patch("agents.news_timeline.enqueue", enq),
        patch("agents.news_timeline.run_job", side_effect=AssertionError("ran on the verdict path")),
        patch("agents.news_timeline._read", side_effect=AssertionError("read on the verdict path")),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "IONQ", "role": "Moonshot", "tier": "Speculative",
                                  "sector": "Quantum", "held": False},
                                 situation_brief="Current Price: $25.00\n" + _FUND_BRIEF,
                                 regime_packet={"regime": "Risk-Off", "score": 36, "reasoning": ""},
                                 trigger=dict(trigger) if trigger else None)
    return result, log_signal.call_args.kwargs, ollama.call_args, enq


def _clockless(d):
    if isinstance(d, dict):
        return {k: _clockless(v) for k, v in d.items() if not k.endswith("_at") and k != "timestamp"}
    if isinstance(d, list):
        return [_clockless(v) for v in d]
    if isinstance(d, float):
        return round(d, 4)              # weeks_to_catalyst and the like run off the wall clock
    return d


@pytest.mark.switches(NEWS_TIMELINE_RECORD_ENABLED=True)
def test_the_hook_queues_the_sent_prompt_and_reads_nothing():
    _, _, sent, enq = _analyze()
    (job,), _ = enq.call_args
    assert job["ticker"] == "IONQ" and job["signal_id"] == 94689 and job["path"] == "event"
    assert job["prompt"] == sent.args[0]                  # the exact string _call_ollama got
    assert job["trigger"] == {"url": TRIGGER["url"], "title": TRIGGER["title"]}
    assert job["env"] == config.TRADING_ENV
    assert abs((job["verdict_at"] - datetime.now(timezone.utc)).total_seconds()) < 60
    assert isinstance(job["prompt_at"], datetime) and job["prompt_at"] <= job["verdict_at"]


def test_other_tests_queue_to_the_inert_list():
    """conftest: outside @news_timeline_lane, a job is recorded, never run."""
    with patch.object(config, "NEWS_TIMELINE_RECORD_ENABLED", True):
        analyst._news_timeline_enqueue("IONQ", 7, TRIGGER, "p", VERDICT_AT)
    assert [j["signal_id"] for j in nt._TEST_JOBS] == [7] and nt._QUEUE is None


@pytest.mark.switches(NEWS_TIMELINE_RECORD_ENABLED=True)
@pytest.mark.parametrize("trigger, sid, queued", [
    ({"path": "daily_scan", "source": "scanner"}, 5, True),
    ({"path": "scheduled_review"}, 5, False), ({"path": "earnings_review"}, 5, False),
    (None, 5, False), (TRIGGER, None, False)])
def test_which_verdicts_queue(trigger, sid, queued):
    assert _analyze(trigger=trigger, signal_id=sid)[3].called is queued


@pytest.mark.switches(NEWS_TIMELINE_RECORD_ENABLED=False)
def test_off_queues_nothing():
    assert not _analyze()[3].called


def test_the_verdict_and_its_row_are_the_same_on_and_off():
    with patch.object(config, "NEWS_TIMELINE_RECORD_ENABLED", True):
        on = _analyze()
    with patch.object(config, "NEWS_TIMELINE_RECORD_ENABLED", False):
        off = _analyze()
    assert on[3].called and not off[3].called
    assert on[2] == off[2]                                # the prompt sent
    assert _clockless(on[0]) == _clockless(off[0])        # the verdict routing reads
    a, b = on[1], off[1]
    assert set(a) == set(b) and a["prompt"] == b["prompt"]
    skip = {"catalyst_check"}
    assert _clockless({k: v for k, v in a["decision"].items() if k not in skip}) == \
        _clockless({k: v for k, v in b["decision"].items() if k not in skip})
    assert "news_timeline" not in json.dumps(a["decision"], default=str)


@pytest.mark.switches(NEWS_TIMELINE_RECORD_ENABLED=True)
def test_a_failing_enqueue_does_not_stop_the_verdict():
    with patch("agents.news_timeline.enqueue", side_effect=RuntimeError("x")):
        analyst._news_timeline_enqueue("IONQ", 1, TRIGGER, "p")      # never raises


def test_not_in_the_prompt_path():
    for f in (analyst._build_prompt,):
        assert "news_timeline" not in inspect.getsource(f)
    assert "news_timeline" not in analyst._prompt_template_full_text()
    assert "news_timeline" not in analyst._prompt_db_queries_text()
    for m in analyst._BRIEF_MODULES:
        assert "news_timeline" not in (analyst._module_source(m) or "")
    assert "agents.news_timeline" not in analyst._BRIEF_MODULES


def test_shipped_on_for_event_and_scan():
    import importlib
    src = (pathlib.Path(config.__file__).read_text())
    assert "NEWS_TIMELINE_RECORD_ENABLED = True" in src
    assert 'NEWS_TIMELINE_PATHS = ("event", "daily_scan")' in src
    importlib.import_module("agents.news_timeline")


# ---- batch_verify and the registry ----

def _script(name):
    path = pathlib.Path(analyst.__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_ntl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_verify_stage():
    bv = _script("batch_verify")
    now = VERDICT_AT + timedelta(hours=1)
    extra = nt.run_job({"ticker": "IONQ", "signal_id": 1, "env": "live", "path": "event",
                        "verdict_at": VERDICT_AT, "prompt": PROMPT, "trigger": TRIGGER},
                       read=MagicMock(return_value=(NEWS, SCREENS)), write=MagicMock())
    assert set(bv.NEWS_TIMELINE_REQUIRED) <= set(extra)
    row = {"id": 50, "stage": "news_timeline", "ticker": "IONQ", "extra": extra}
    verdicts = [{"id": 1, "ts": VERDICT_AT, "path": "event"},
                {"id": 2, "ts": VERDICT_AT, "path": "scheduled_review"},       # not a path
                {"id": 3, "ts": now - timedelta(minutes=2), "path": "event"}]   # not due yet
    c = bv.check_news_timeline(verdicts, [row], "live", now)
    assert c["status"] == "OK" and c["rows"] == 1 and c["reads"]["read"] == {"ok": 1}
    assert "not yet due: 1" in " ".join(c["notes"])
    c = bv.check_news_timeline(verdicts + [{"id": 4, "ts": VERDICT_AT, "path": "daily_scan"}],
                               [row], "live", now)
    assert c["status"] == "FAIL" and c["missing"] == [4]
    bad = dict(extra)
    bad.pop("items")
    c = bv.check_news_timeline(verdicts[:1], [dict(row, extra=bad)], "live", now)
    assert c["status"] == "FAIL" and "required key" in c["why"]
    err = {k: extra[k] for k in bv.NEWS_TIMELINE_BASE}
    err["read"] = "error: OperationalError"
    c = bv.check_news_timeline(verdicts[:1], [dict(row, extra=err)], "live", now)
    assert c["status"] == "FAIL" and "errors" in c["why"]
    assert bv.check_news_timeline([], [], "live", now)["status"] == "WARN"
    off = bv.check_news_timeline(verdicts + [{"id": 4, "ts": VERDICT_AT, "path": "event"}],
                                 [], "live", now, enabled=False)
    assert off["status"] == "OK"
    paper = dict(row, extra=dict(extra, env="paper"))
    assert bv.check_news_timeline(verdicts[:1], [paper], "live", now)["missing"] == [1]


def test_registry_stage():
    rc = _script("shadow_registry_check")
    st = {s["stage"]: s for s in rc.STAGES}["news_timeline"]
    assert st["switch"] == "NEWS_TIMELINE_RECORD_ENABLED" and st["expect_rows"] == "conditional"
    assert st["trigger"][0] == "signals" and "'event', 'daily_scan'" in st["trigger"][1]
    assert st["sources"] == (("activity_feed", "stage = 'news_timeline'"),)
