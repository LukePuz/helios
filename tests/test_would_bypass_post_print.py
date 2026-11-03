"""Log-only would_bypass_post_print shadow: a held name's earnings item that
the screener drops in the post-print window gets one screener_shadow row per
(ticker, earnings date), later ones only a count. The decision is unchanged."""
import json
from contextlib import ExitStack
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from agents import screener

AFTER_CLOSE = datetime(2026, 11, 2, 21, 30, tzinfo=timezone.utc)   # Mon 16:30 ET
PRE_OPEN = datetime(2026, 11, 3, 13, 0, tzinfo=timezone.utc)       # Tue 08:00 ET
MIDDAY = datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc)         # Mon 12:00 ET
_NO = {"relevant": False, "urgency": "eod", "reason": "no new fact", "model_used": "7b"}
_8K = {"ticker": "PLTR", "source": "sec_filing", "form": "8-K", "type": "8-K",
       "title": "8-K - PALANTIR TECHNOLOGIES INC.", "url": "https://sec/x-index.htm",
       "_seen_key": "urn:filing:1",
       "text": "8-K items: 2.02 (results of operations), 9.01 (exhibits)\n\nEX-99.1: ..."}


class _Env:
    def __init__(self, earnings=date(2026, 11, 2), write_ok=True):
        self.state = {}
        self.log_activity = MagicMock(return_value=write_ok)
        self.around = MagicMock(return_value={"PLTR": earnings} if earnings else {})
        self.set_state = MagicMock(side_effect=lambda k, v: self.state.__setitem__(k, v))

    def __enter__(self):
        self._stack = ExitStack()
        for p in (patch("db.get_earnings_around", self.around),
                  patch("db.get_state", side_effect=lambda k, d=None: self.state.get(k, d)),
                  patch("db.set_state", self.set_state),
                  patch("db.log_activity", self.log_activity)):
            self._stack.enter_context(p)
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def rows(self):
        return [c.args for c in self.log_activity.call_args_list if c.args[0] == "screener_shadow"]


def _run(event=None, result=None, held=True, now=AFTER_CLOSE):
    event = dict(event or _8K)
    screener._log_would_bypass_post_print(dict(result or _NO), event, event["ticker"],
                                          event["source"], event["title"], held, now=now)


def test_item_202_after_the_close_logs_one_row():
    with _Env() as env:
        _run()
    (stage, ticker, _msg, _sent, extra), = env.rows()
    assert (stage, ticker) == ("screener_shadow", "PLTR")
    assert extra["would_bypass_post_print"] is True
    assert extra["shadow_key"] == "PLTR:2026-11-02" and extra["earnings_date"] == "2026-11-02"
    assert extra["earnings_like"] == "8k_item_2_02" and extra["window"] == "after_close"
    assert extra["item_id"] == "urn:filing:1" and extra["source"] == "sec_filing"
    assert extra["screener_reason"] == "no new fact" and extra["count_at_log"] == 1
    assert {"code_sha", "env_sha"} <= set(extra)


def test_titles_before_the_next_open_log_including_across_a_weekend_and_holiday():
    ev = {"ticker": "PLTR", "source": "news", "title": "Palantir Posts Q3, raises outlook"}
    with _Env() as env:
        _run(ev, now=PRE_OPEN)
    assert env.rows()[0][4]["window"] == "pre_open" and env.rows()[0][4]["earnings_like"] == "title"
    with _Env(earnings=date(2026, 10, 30)) as env:          # Fri print, Mon 08:00 ET
        _run(ev, now=datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc))
    assert len(env.rows()) == 1
    with _Env(earnings=date(2026, 11, 25)) as env:          # Wed print, Thanksgiving, Fri 08:00
        _run(ev, now=datetime(2026, 11, 27, 13, 0, tzinfo=timezone.utc))
    assert len(env.rows()) == 1


def test_the_window_covers_whole_weekend_days_and_ends_at_the_next_open():
    ev = {"ticker": "PLTR", "source": "news", "title": "Palantir Posts Q3, raises outlook"}
    with _Env(earnings=date(2026, 10, 30)) as env:          # Fri print, Sat 14:00 ET
        _run(ev, now=datetime(2026, 10, 31, 18, 0, tzinfo=timezone.utc))
    assert len(env.rows()) == 1
    with _Env(earnings=date(2026, 10, 30)) as env:          # Mon 10:00 ET: past the next open
        _run(ev, now=datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc))
    assert env.rows() == [] and not env.around.called      # session hours: no DB read
    with _Env(earnings=date(2026, 10, 29)) as env:          # Thu print, Fri 17:00 ET is past it
        _run(ev, now=datetime(2026, 10, 30, 21, 0, tzinfo=timezone.utc))
    assert env.rows() == []


def test_a_corrupt_book_starts_over():
    with _Env() as env:
        env.state[screener._POST_PRINT_KEY.format(env=__import__("config").TRADING_ENV)] = "{not json"
        _run()
    assert len(env.rows()) == 1


def test_later_items_only_raise_the_count():
    with _Env() as env:
        _run()
        _run({"ticker": "PLTR", "source": "news", "title": "Palantir beats on EPS", "url": "u2"})
        _run({"ticker": "PLTR", "source": "news", "title": "Palantir revenue tops", "url": "u3"})
    assert len(env.rows()) == 1
    book = json.loads(env.state["would_bypass_post_print_" + __import__("config").TRADING_ENV])
    entry = book["PLTR:2026-11-02"]
    assert entry["count"] == 3 and entry["logged"] and entry["first_item_id"] == "urn:filing:1"


def test_a_failed_write_lets_the_next_item_log():
    with _Env(write_ok=False) as env:
        _run()
        env.log_activity.return_value = True
        _run({"ticker": "PLTR", "source": "news", "title": "Palantir Q3 results", "url": "u2"})
        _run({"ticker": "PLTR", "source": "news", "title": "Palantir Q3 EPS", "url": "u3"})
    assert len(env.rows()) == 2
    assert env.rows()[1][4]["count_at_log"] == 2


def test_non_triggers_write_nothing():
    cases = [
        dict(held=False),
        dict(result={**_NO, "relevant": True}),
        dict(now=MIDDAY),
        dict(event={"ticker": "PLTR", "source": "news", "title": "Palantir CEO to speak"}),
        dict(event={**_8K, "text": "8-K items: 5.02 (officer or director change)"}),
    ]
    for kw in cases:
        with _Env() as env:
            _run(**kw)
        assert env.rows() == [] and env.set_state.call_count == 0, kw
    with _Env() as env:          # mid-session never reads the earnings calendar
        _run(now=MIDDAY)
    assert env.around.call_count == 0
    with _Env(earnings=date(2026, 11, 3)) as env:     # reports tomorrow
        _run()
    with _Env(earnings=date(2026, 11, 2)) as env2:    # report day, before the open
        _run(now=datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc))
    with _Env(earnings=date(2026, 10, 30)) as env3:   # two sessions back
        _run(now=PRE_OPEN)
    with _Env(earnings=None) as env4:
        _run()
    assert env.rows() == env2.rows() == env3.rows() == env4.rows() == []


def test_swallows_its_own_errors():
    with patch("db.get_earnings_around", side_effect=RuntimeError("db down")), \
         patch("db.log_activity") as la:
        _run()
    la.assert_not_called()
    with _Env() as env, patch("db.get_state", side_effect=RuntimeError("timeout")):
        _run()
    assert env.rows() == []
    screener._log_would_bypass_post_print(None, None, "PLTR", "news", "Q3", True)   # no raise


def test_the_screener_decision_and_prompt_do_not_change():
    real = screener._log_would_bypass_post_print
    ev = {"ticker": "PLTR", "source": "news", "title": "Palantir Posts Q3, raises outlook",
          "text": "x"}

    def run(shadow):
        with _Env() as env, \
             patch("agents.screener._call_ollama", return_value=json.dumps(_NO)), \
             patch("db.get_position_stop", return_value={"entry_price": 1.0}), \
             patch("stop_lock._redis", side_effect=RuntimeError("no redis")), \
             patch.object(screener, "_log_would_bypass_post_print", shadow):
            return screener.screen(dict(ev)), env

    hit, env = run(lambda *a: real(*a, now=AFTER_CLOSE))
    assert len(env.rows()) == 1
    off, _ = run(lambda *a: None)
    assert hit == off
