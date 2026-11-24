"""Ingestion yield watch (tasks.feed_health, config FEED_HEALTH_*) and the
brief price's source and age on trigger provenance (crew._get_price info,
crew._brief_price_record). Gap hunt 2026-11-21, data pipeline items 1-3: the
analyst-ratings feed returned 0 items from 11-13 while every poll succeeded,
and source_yield's trailing mean absorbed the outage within 2 days."""
import inspect
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import alert_policy
import config
import tasks
from agents import crew


def _sessions(end: date, n: int) -> list[date]:
    """The n weekday sessions (no NYSE holiday) ending at end, oldest first."""
    from market_calendar import NYSE_HOLIDAYS
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5 and d not in NYSE_HOLIDAYS:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


TODAY = date(2026, 11, 20)        # a Friday


def _counts(series: dict, end=TODAY, news=500) -> list[dict]:
    """series: {source: [n per session, oldest first]} on the sessions ending
    at end; a busy news feed keeps every session "up" unless news=0."""
    m = max(len(v) for v in series.values())
    days = _sessions(end, m)
    rows = [{"source": "news", "day": d, "n": news} for d in days if news]
    for src, ns in series.items():
        for d, n in zip(days[m - len(ns):], ns):
            if n:
                rows.append({"source": src, "day": d, "n": n})
    return rows


# ---- the summary (pure) ----

def test_the_analyst_ratings_freeze_is_flagged_zero_against_its_own_baseline():
    s = tasks._feed_health_summary(_counts({"analyst_ratings": [25, 23, 30, 28, 27, 35, 25, 23, 27, 25,
                                                                0, 0, 0, 0, 0, 0]}), TODAY)
    r = s["analyst_ratings"]
    assert r["flag"] == "zero" and r["zero_sessions"] == 6 and r["n"] == 0
    assert r["baseline_median"] == 26.0                       # the 10 sessions before the zeros
    assert len(r["baseline_days"]) == 10


def test_one_zero_session_is_not_flagged():
    s = tasks._feed_health_summary(_counts({"arxiv": [14, 13, 15, 0, 16, 17, 12, 14, 0, 15, 0]}), TODAY)
    assert s["arxiv"]["flag"] is None and s["arxiv"]["zero_sessions"] == 1


def test_two_zero_sessions_are_flagged():
    s = tasks._feed_health_summary(_counts({"edgar": [25, 30, 22, 28, 26, 24, 0, 0]}), TODAY)
    assert s["edgar"]["flag"] == "zero" and s["edgar"]["baseline_median"] == 25.5


def test_the_baseline_leaves_out_zero_sessions():
    """A source with quiet days: its baseline is the median of the non-zero
    ones, not dragged down by the zeros."""
    s = tasks._feed_health_summary(_counts({"arxiv": [0, 14, 0, 16, 0, 12, 0, 18, 0, 20, 15]}), TODAY)
    # unflagged: the baseline before today, from 14, 16, 12, 18, 20
    assert s["arxiv"]["baseline_median"] == 16.0 and s["arxiv"]["flag"] is None


def test_three_low_sessions_are_flagged_and_two_are_not():
    base = [100] * 10
    s = tasks._feed_health_summary(_counts({"insider": base + [10, 12, 9]}), TODAY)
    assert s["insider"]["flag"] == "low" and s["insider"]["low_sessions"] == 3
    assert s["insider"]["baseline_median"] == 100.0          # the low sessions are not in it
    s = tasks._feed_health_summary(_counts({"insider": base + [10, 12]}), TODAY)
    assert s["insider"]["flag"] is None and s["insider"]["low_sessions"] == 2
    s = tasks._feed_health_summary(_counts({"insider": base + [10, 25, 9]}), TODAY)   # 25 is 25%
    assert s["insider"]["low_sessions"] == 1 and s["insider"]["flag"] is None


def test_a_slow_collapse_keeps_its_pre_collapse_baseline():
    """Low sessions never become the baseline: 8 sessions at about 10% of
    100 still read against 100."""
    s = tasks._feed_health_summary(_counts({"edgar": [100] * 10 + [10] * 8}), TODAY)
    assert s["edgar"]["flag"] == "low" and s["edgar"]["low_sessions"] == 8
    assert s["edgar"]["baseline_median"] == 100.0


def test_too_little_history_is_not_judged():
    s = tasks._feed_health_summary(_counts({"options_flow_dispatch": [41, 49, 52, 0, 0]}), TODAY)
    assert s["options_flow_dispatch"]["flag"] is None
    assert s["options_flow_dispatch"]["baseline_median"] is None


def test_a_stack_down_day_and_a_holiday_are_not_sessions():
    """Nothing at all from any source is the stack down, not a quiet feed;
    Thanksgiving (11-26) is no session either."""
    days = _sessions(date(2026, 11, 25), 12)
    rows = [{"source": "news", "day": d, "n": 500} for d in days]
    rows += [{"source": "edgar", "day": d, "n": 25} for d in days[:-1]]
    rows = [r for r in rows if r["day"] != days[-2]]           # the stack down on day -2
    s = tasks._feed_health_summary(rows, days[-1])
    assert s["edgar"]["zero_sessions"] == 1 and s["edgar"]["flag"] is None
    assert tasks._feed_health_sessions(rows + [{"source": "news", "day": date(2026, 11, 26), "n": 9}],
                                       date(2026, 11, 26))[-1] == days[-1]


def test_a_day_that_is_not_a_session_judges_nothing():
    rows = _counts({"edgar": [25] * 10})
    assert tasks._feed_health_summary(rows, date(2026, 11, 21)) == {}          # Saturday
    assert tasks._feed_health_summary(rows, TODAY, excluded={"edgar"}).keys() == {"news"}


# ---- the task ----

def _run(rows, state=None, today=TODAY, quiver=False):
    state = {} if state is None else state
    db = MagicMock()
    db.TRADING_ENV = "live"
    db.get_seen_item_counts.return_value = rows
    db.get_state.side_effect = lambda k, d=None: state.get(k, d)
    db.set_state.side_effect = lambda k, v: state.__setitem__(k, v)
    sent = []
    fake_now = datetime.combine(today, datetime.min.time()).replace(hour=18, minute=40)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fake_now.replace(tzinfo=tz)
    with patch.dict("sys.modules", {"db": db}), \
         patch("datetime.datetime", _DT), \
         patch.object(config, "QUIVER_ENABLED", quiver), \
         patch("alerts._send", side_effect=lambda t, *a, **k: sent.append(t)):
        tasks.feed_health.run()
    return db, sent, state


FROZEN = {"analyst_ratings": [25] * 10 + [0] * 6, "edgar": [25] * 16,
          "news_title": [0] * 16, "congressional": [3] * 10 + [0] * 6}


def test_the_task_alerts_once_per_source_per_day_and_writes_a_row():
    rows = _counts(FROZEN)
    db, sent, state = _run(rows)
    assert len(sent) == 1 and "analyst_ratings: 0 new items for 6 sessions" in sent[0]
    assert "congressional" not in sent[0] and "news_title" not in sent[0]   # skipped
    assert state == {"feed_health_alerted_analyst_ratings": TODAY.isoformat()}
    row = db.log_activity_direct.call_args.args
    assert row[0] == "feed_health" and row[3] == "negative"
    extra = row[4]
    assert extra["flagged"] == ["analyst_ratings"] and extra["alerted"] == ["analyst_ratings"]
    assert {"congressional", "news_title"} <= set(extra["excluded"])
    # the same day again: a row, no second alert
    db2, sent2, _ = _run(rows, state)
    assert sent2 == [] and db2.log_activity_direct.call_args.args[4]["alerted"] == []
    # the next session alerts again
    nxt = date(2026, 11, 23)
    _, sent3, _ = _run(_counts({**FROZEN, "analyst_ratings": [25] * 10 + [0] * 7}, end=nxt), state,
                       today=nxt)
    assert len(sent3) == 1


def test_congressional_is_judged_when_quiver_is_on():
    _, sent, _ = _run(_counts(FROZEN), quiver=True)
    assert "congressional" in sent[0]


def test_the_alert_is_critical_like_the_ingestion_health_check():
    _, sent, _ = _run(_counts(FROZEN))
    assert alert_policy.classify(sent[0])[0] == alert_policy.CRITICAL


def test_nothing_flagged_sends_nothing_and_writes_a_neutral_row():
    db, sent, _ = _run(_counts({"edgar": [25] * 12}))
    assert sent == [] and db.log_activity_direct.call_args.args[3] == "neutral"


def test_the_task_reads_seen_items_only_and_never_raises():
    db, _, _ = _run(_counts(FROZEN))
    used = {c[0].split(".")[0] for c in db.mock_calls}
    assert used <= {"get_seen_item_counts", "get_state", "set_state", "log_activity_direct"}
    db, sent, _ = _run(None)                                   # counts unreadable
    assert sent == [] and not db.log_activity_direct.called
    with patch("db.get_seen_item_counts", side_effect=RuntimeError("db")):
        tasks.feed_health.run()                                # no raise


def test_switch_off_reads_nothing():
    with patch.object(config, "FEED_HEALTH_ENABLED", False), \
         patch("db.get_seen_item_counts") as read:
        tasks.feed_health.run()
    read.assert_not_called()


def test_beat_entry_and_shipped_defaults():
    e = tasks.app.conf.beat_schedule["feed-health"]
    assert e["task"] == "tasks.feed_health" and e["options"]["queue"] == "ops"
    assert (config.FEED_HEALTH_ENABLED, config.FEED_HEALTH_BASELINE_SESSIONS,
            config.FEED_HEALTH_ZERO_SESSIONS, config.FEED_HEALTH_LOW_RATIO,
            config.FEED_HEALTH_LOW_SESSIONS) == (True, 10, 2, 0.2, 3)


# ---- brief price source and age (record only) ----

BAR = datetime(2026, 11, 20, 13, 0, tzinfo=timezone.utc)


def _legs(session=None, snap=None, quote=None, yf_price=None):
    yf = MagicMock()
    yf.Ticker.return_value.info = {"regularMarketPrice": yf_price, "postMarketPrice": None,
                                   "preMarketPrice": None}
    yf.Ticker.return_value.fast_info = MagicMock(last_price=None)
    return [patch("market_data.get_session_quote", return_value=session),
            patch("ingestion.polygon_stream.get_snapshot", return_value=snap or {}),
            patch("market_data.get_latest_quotes", return_value={"NVDA": quote} if quote else {}),
            patch.dict("sys.modules", {"yfinance": yf})]


@pytest.mark.parametrize("legs, price, source, as_of, stale", [
    (dict(session={"price": 101.0, "bar_time": BAR, "session": "after-hours"}), 101.0,
     "session_quote", BAR, None),
    (dict(snap={"price": 102.0}), 102.0, "polygon_snapshot", None, None),
    (dict(quote={"mid": 103.0, "is_stale": True, "bar_time": BAR}), 103.0, "latest_quote", BAR, True),
    (dict(yf_price=104.0), 104.0, "yfinance_info", None, None),
    (dict(), None, None, None, None),
])
def test_get_price_says_which_source_answered_and_the_price_is_unchanged(legs, price, source,
                                                                          as_of, stale):
    from contextlib import ExitStack
    with ExitStack() as st:
        for p in _legs(**legs):
            st.enter_context(p)
        plain = crew._get_price("NVDA", None)
        info = {}
        assert crew._get_price("NVDA", None, info=info) == plain == price
    utc = info.pop("as_of_utc")   # the sizing record's zone-known time (aware sources only)
    assert info == {"source": source, "as_of": as_of, "is_stale": stale}
    assert utc == (crew._aware_utc(as_of) if source == "session_quote" else None)


def test_brief_price_record_ages_from_the_source_time():
    r = crew._brief_price_record({"source": "latest_quote", "as_of": BAR, "is_stale": True},
                                 (BAR + timedelta(minutes=3, seconds=4)).isoformat())
    assert r == {"brief_price_source": "latest_quote", "brief_price_as_of": BAR.isoformat(),
                 "brief_price_age_s": 184.0, "brief_price_is_stale": True}
    # A naive session_quote time is ET, as market_data reads that source's bars.
    naive = crew._brief_price_record({"source": "session_quote",
                                      "as_of": datetime(2026, 11, 20, 8, 0)}, BAR.isoformat())
    assert naive["brief_price_as_of"] == BAR.isoformat() and naive["brief_price_age_s"] == 0.0
    assert crew._brief_price_record(None, None) == {
        "brief_price_source": None, "brief_price_as_of": None, "brief_price_age_s": None,
        "brief_price_is_stale": None}
    assert crew._brief_price_record({"as_of": "not a time"}, "x")["brief_price_age_s"] is None


def test_deep_analyze_puts_the_record_on_the_trigger_only():
    src = inspect.getsource(crew.deep_analyze)
    assert "_get_price(ticker, client, info=price_info)" in src
    assert "trigger.update(_brief_price_record(price_info, brief_built_at))" in src
    # the brief itself still gets the price alone
    assert "assemble_brief(event, positions, client, current_price=price)" in src


def test_latest_quotes_pass_the_bar_time_through():
    import market_data
    q = {"price": 50.0, "is_stale": False, "bar_time": BAR, "price_source": "1min"}
    with patch("market_data.get_priced_quote", return_value=q):
        out = market_data.get_latest_quotes(["NVDA"])["NVDA"]
    assert out["mid"] == 50.0 and out["bar_time"] == BAR and out["price_source"] == "1min"


def test_rth_latest_quote_naive_et_label_is_not_read_as_utc():
    """An RTH latest_quote: get_priced_quote's bar_time is the naive ET
    wall-clock label (13:04 ET = 18:04 UTC in November). Read as UTC it aged
    the brief by +5 h. With the bar's offset known (as_of_utc) the age is
    right; without it the time is unknown, not guessed."""
    label = datetime(2026, 11, 20, 13, 4)                        # naive, ET wall clock
    utc = datetime(2026, 11, 20, 18, 4, tzinfo=timezone.utc)
    built = (utc + timedelta(seconds=50)).isoformat()
    r = crew._brief_price_record({"source": "latest_quote", "as_of": label, "is_stale": False,
                                  "as_of_utc": utc}, built)
    assert r["brief_price_as_of"] == utc.isoformat() and r["brief_price_age_s"] == 50.0
    r = crew._brief_price_record({"source": "latest_quote", "as_of": label, "is_stale": False},
                                 built)
    assert r["brief_price_as_of"] is None and r["brief_price_age_s"] is None
    assert r["brief_price_source"] == "latest_quote" and r["brief_price_is_stale"] is False
    # End to end through _get_price: the quote's bar carried its offset.
    from contextlib import ExitStack
    info = {}
    with ExitStack() as st:
        for p in _legs(quote={"mid": 103.0, "is_stale": False, "bar_time": label,
                              "bar_time_utc": utc}):
            st.enter_context(p)
        assert crew._get_price("NVDA", None, info=info) == 103.0
    assert crew._brief_price_record(info, built)["brief_price_age_s"] == 50.0


def test_every_source_goes_through_the_same_rule():
    """session_quote (aware ET bar; a naive one is ET), latest_quote (naive
    without its UTC form: unknown), the sources with no time (None), a naive
    ISO string from an unknown source (unknown), and the sizing record too."""
    from zoneinfo import ZoneInfo
    et = datetime(2026, 11, 20, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert crew._as_of_utc({"source": "session_quote", "as_of": et}) == BAR
    assert crew._as_of_utc({"source": "session_quote", "as_of": et.replace(tzinfo=None)}) == BAR
    assert crew._as_of_utc({"source": "latest_quote", "as_of": et.replace(tzinfo=None)}) is None
    assert crew._as_of_utc({"source": "latest_quote", "as_of": "2026-11-20T08:00:00"}) is None
    assert crew._as_of_utc({"source": "x", "as_of": "2026-11-20T13:00:00Z"}) == BAR
    for src in ("polygon_snapshot", "yfinance_info", None):
        assert crew._as_of_utc({"source": src, "as_of": None}) is None
    # as_of_utc wins over a naive as_of, and a naive as_of_utc is never used.
    assert crew._as_of_utc({"source": "latest_quote", "as_of": et.replace(tzinfo=None),
                            "as_of_utc": BAR}) == BAR
    assert crew._as_of_utc({"source": "latest_quote", "as_of_utc": BAR.replace(tzinfo=None)}) is None
    a = crew._arrival_record({}, 1.0, {"source": "latest_quote", "as_of": et.replace(tzinfo=None)})
    assert a["sizing_bar_time"] is None and a["sizing_price_age_s"] is None
    a = crew._arrival_record({}, 1.0, {"source": "session_quote", "as_of": et.replace(tzinfo=None)})
    assert a["sizing_bar_time"] == BAR.isoformat()
