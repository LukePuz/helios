"""Drop watch: shadow rows by default, a review per ticker per day when on."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import config
import tasks


def _hist(n=30, c=100.0):
    return [{"date": f"2026-10-{i + 1:02d}", "open": c, "high": c + 1.5, "low": c - 1.5,
             "close": c, "volume": 1} for i in range(n)]


def test_metrics():
    m = tasks._dip_metrics(92.0, 100.0, _hist())
    assert m["move_pct"] == -8.0 and m["atr_pct"] == 3.0 and m["atr_mult"] == round(-8 / 3, 2)
    assert m["from_20d_high_pct"] == round((92 / 101.5 - 1) * 100, 2)
    assert tasks._dip_metrics(99.0, 100.0, [])["atr_mult"] is None


class _DT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 9, 12, 30, tzinfo=ZoneInfo("America/New_York"))


def _run(prices, held=("NVDA",), watch=("AMD", "MSFT"), redis_set=True, shadow_prev=None):
    snap = {t: {"qty": 1} for t in held}
    r = MagicMock()
    r.set.return_value = redis_set
    r.get.return_value = shadow_prev
    with patch("tasks._snapshot_positions_or_none", return_value=snap), \
         patch("tasks._get_client", side_effect=AssertionError("no broker read")), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("knowledge_loader.load_watchlist", return_value={t: {} for t in watch}), \
         patch("tasks._today_prices", return_value=dict(prices)), \
         patch("db.get_recent_news_for_ticker", return_value=[]), \
         patch("db.get_price_history", return_value=_hist()), \
         patch("db.log_signal") as log_signal, \
         patch("redis.Redis.from_url", return_value=r), \
         patch("tasks.deep_analyze_event") as dae, \
         patch("datetime.datetime", _DT):
        tasks.dip_watch()
    return log_signal, dae


PRICES = {"NVDA": {"last": 93.0, "prev_close": 100.0},      # -7%: would review
          "AMD": {"last": 96.5, "prev_close": 100.0},       # -3.5%: shadow only
          "MSFT": {"last": 99.5, "prev_close": 100.0}}      # -0.5%: nothing


def test_off_writes_shadow_rows_and_queues_nothing():
    log_signal, dae = _run(PRICES)
    rows = {c.args[0]: c.kwargs["provenance"] for c in log_signal.call_args_list}
    assert set(rows) == {"NVDA", "AMD"}
    assert rows["NVDA"]["would_review"] is True and rows["NVDA"]["review_queued"] is False
    assert rows["AMD"]["would_review"] is False and rows["NVDA"]["held"] is True
    assert all(c.args[1] == "DIP_WATCH_SHADOW" for c in log_signal.call_args_list)
    assert not dae.apply_async.called


def test_on_queues_one_review_with_the_held_queue(monkeypatch):
    monkeypatch.setattr(config, "DIP_REVIEW_ENABLED", True)
    log_signal, dae = _run(PRICES)
    assert dae.apply_async.call_count == 1
    kw = dae.apply_async.call_args.kwargs
    ev = kw["args"][0]
    assert ev["ticker"] == "NVDA" and ev["type"] == "price drop" and ev["source"] == "dip_watch"
    assert "no headline trigger" in ev["title"] and "You hold this position" in ev["text"]
    assert kw["queue"] == tasks.LLM_QUEUE_HELD


def test_on_but_already_reviewed_today(monkeypatch):
    monkeypatch.setattr(config, "DIP_REVIEW_ENABLED", True)
    log_signal, dae = _run(PRICES, redis_set=False)
    assert not dae.apply_async.called


def test_not_held_drop_goes_to_the_bulk_queue(monkeypatch):
    monkeypatch.setattr(config, "DIP_REVIEW_ENABLED", True)
    _, dae = _run({"AMD": {"last": 90.0, "prev_close": 100.0}})
    assert dae.apply_async.call_args.kwargs["queue"] == tasks.LLM_QUEUE_BULK
    assert "You do not hold" in dae.apply_async.call_args.kwargs["args"][0]["text"]


def test_scheduled_and_filtered_from_verdict_reads():
    import db
    assert tasks.app.conf.beat_schedule["dip-watch"]["options"]["queue"] == tasks.INGESTION_QUEUE
    assert "DIP_WATCH_SHADOW" in db._NON_ANALYST_SIGNALS


def test_single_ticker_multiindex_frame_is_read():
    import pandas as pd
    idx = pd.to_datetime(["2026-11-06", "2026-11-09"])
    cols = pd.MultiIndex.from_tuples([("NVDA", "Close"), ("NVDA", "Open")])
    df = pd.DataFrame([[100.0, 99.0], [93.0, 98.0]], index=idx, columns=cols)
    with patch("yfinance.download", return_value=df):
        assert tasks._today_prices(["NVDA"], "2026-11-09") == {"NVDA": {"last": 93.0, "prev_close": 100.0}}
    flat = pd.DataFrame({"Close": [100.0, 93.0]}, index=idx)
    with patch("yfinance.download", return_value=flat):
        assert tasks._today_prices(["NVDA"], "2026-11-09")["NVDA"]["last"] == 93.0
    with patch("yfinance.download", return_value=df):
        assert tasks._today_prices(["NVDA"], "2026-11-10") == {}      # no bar dated today


def test_a_lingering_stop_row_is_not_held(monkeypatch):
    monkeypatch.setattr(config, "DIP_REVIEW_ENABLED", True)
    r = MagicMock()
    r.set.return_value = True
    r.get.return_value = None
    with patch("tasks._snapshot_positions_or_none", return_value={}), \
         patch("db.get_recent_news_for_ticker", return_value=[]), \
         patch("db.get_all_position_stops", return_value=[{"ticker": "QBTS"}]), \
         patch("knowledge_loader.load_watchlist", return_value={}), \
         patch("tasks._today_prices", return_value={"QBTS": {"last": 90.0, "prev_close": 100.0}}), \
         patch("db.get_price_history", return_value=_hist()), \
         patch("db.log_signal") as log_signal, \
         patch("redis.Redis.from_url", return_value=r), \
         patch("tasks.deep_analyze_event") as dae, \
         patch("datetime.datetime", _DT):
        tasks.dip_watch()
    ev = dae.apply_async.call_args.kwargs
    assert "You do not hold" in ev["args"][0]["text"] and ev["queue"] == tasks.LLM_QUEUE_BULK
    assert log_signal.call_args.kwargs["provenance"]["held"] is False


def test_new_fact_gate_for_a_drop_review(monkeypatch):
    from agents import crew
    monkeypatch.setattr(crew, "_new_fact_settings", lambda: (True, "news_only", "24h"))
    trig = {"path": "event", "source": "dip_watch"}
    with patch("db.recent_new_fact_items", return_value=[]):
        g = crew.new_fact_gate("NVDA", "OPEN", trig)
    assert g["passed"] is False and g["dip_review"] is True and "item_new_fact_missing" not in g
    monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", True)
    g = crew.new_fact_gate("NVDA", "OPEN", trig)
    assert g["passed"] is True and "drop-watch" in g["reason"]
    assert crew.new_fact_gate("NVDA", "HOLD", trig) is None


def test_spy_relative_move_and_one_row_per_session():
    prices = {**PRICES, "SPY": {"last": 98.0, "prev_close": 100.0}}
    log_signal, _ = _run(prices)
    rows = {c.args[0]: c.kwargs["provenance"] for c in log_signal.call_args_list}
    assert "SPY" not in rows
    assert rows["NVDA"]["spy_move_pct"] == -2.0 and rows["NVDA"]["rel_move_pct"] == -5.0
    assert rows["NVDA"]["first_crossing"] is True and rows["NVDA"]["news_items_24h"] == 0
    log_signal, _ = _run(PRICES, shadow_prev="-6.0")        # already logged at -6%: -7% is not 2 pp deeper
    assert {c.args[0] for c in log_signal.call_args_list} == set()
    log_signal, _ = _run(PRICES, shadow_prev="-4.5")        # -7% is 2.5 pp deeper
    rows = {c.args[0]: c.kwargs["provenance"] for c in log_signal.call_args_list}
    assert set(rows) == {"NVDA"} and rows["NVDA"]["first_crossing"] is False


def test_event_text_names_only_sections_that_exist(monkeypatch):
    m = {"move_pct": -7.0, "atr_mult": -2.3, "from_20d_high_pct": -9.0, "spy_move_pct": -1.0}
    assert "PRICE HISTORY" not in tasks._dip_event("NVDA", m, True)["text"]
    monkeypatch.setattr(config, "BRIEF_PRICE_HISTORY", True)
    t = tasks._dip_event("NVDA", m, True)["text"]
    assert "PRICE HISTORY" in t and "the market moved -1.0% today" in t


def test_unreadable_snapshot_falls_back_to_stop_rows(monkeypatch):
    monkeypatch.setattr(config, "DIP_REVIEW_ENABLED", True)
    r = MagicMock()
    r.set.return_value = True
    r.get.return_value = None
    with patch("tasks._snapshot_positions_or_none", return_value=None), \
         patch("db.get_recent_news_for_ticker", return_value=[]), \
         patch("db.get_all_position_stops", return_value=[{"ticker": "NVDA"}]), \
         patch("knowledge_loader.load_watchlist", return_value={}), \
         patch("tasks._today_prices", return_value={"NVDA": {"last": 90.0, "prev_close": 100.0}}), \
         patch("db.get_price_history", return_value=_hist()), \
         patch("db.log_signal"), patch("redis.Redis.from_url", return_value=r), \
         patch("tasks.deep_analyze_event") as dae, patch("datetime.datetime", _DT):
        tasks.dip_watch()
    assert "You hold this position" in dae.apply_async.call_args.kwargs["args"][0]["text"]
