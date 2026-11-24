"""Story de-dup and staleness shadow (agents/story_dedup.py): record only, on
the event verdict's provenance.trigger.story_dedup. Never raises, never
changes what the analyst sees."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agents import crew, story_dedup as sd

NOW = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)
YAHOO = ("https://finance.yahoo.com/markets/article/"
         "nvidia-stock-extends-25-trillion-rout-as-sec-probe-reportedly-investigates-its-accounting-194945274.html")


@pytest.fixture
def shadow_on(monkeypatch):
    import config
    monkeypatch.setattr(config, "STORY_DEDUP_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "STORY_DEDUP_WINDOW_HOURS", 24)
    monkeypatch.setattr(config, "STORY_STALE_CUTOFF_HOURS", 24)


def _trig(title, url=None, source="news", published=None):
    return {"path": "event", "source": source, "title": title, "url": url,
            "event_time": published.isoformat() if published else None,
            "event_time_source": "published_at" if published else None}


def _prior(sid, hours_ago, title, url=None, source="news"):
    return {"signal_id": sid, "timestamp": NOW - timedelta(hours=hours_ago),
            "title": title, "url": url, "source": source}


# ── keys ────────────────────────────────────────────────────────────────────

def test_story_key_strips_publisher_and_punctuation():
    k = sd.story_key("Nvidia Shares Sink to 52-Week Low!", "https://www.reuters.com/x", "news")
    assert k == "reuters.com|nvidia shares sink to 52 week low"


def test_google_news_domain_is_the_publisher_in_the_title():
    gn = "https://news.google.com/rss/articles/CBMiabc?oc=5"
    k = sd.story_key("Nvidia stock plunges 12% - CNBC", gn, "news")
    assert k == "gn:cnbc|nvidia stock plunges 12"
    # Same headline from another publisher is a different story key.
    assert sd.story_key("Nvidia stock plunges 12% - Yahoo Finance", gn, "news") != k


def test_no_url_falls_back_to_source():
    assert sd.story_key("Mizuho raises NVDA target", None, "analyst_ratings") == \
        "analyst_ratings|mizuho raises nvda target"
    assert sd.story_key(None, None, "news") is None and sd.story_key("", "x", "news") is None


def test_url_key_drops_scheme_www_query_fragment():
    assert sd.url_key("https://www.x.com/a/b/?utm=1#f") == sd.url_key("http://x.com/a/b") == "x.com/a/b"
    assert sd.url_key(None) is None and sd.url_key("not a url") is None


@pytest.mark.parametrize("value", ["2026-11-20T19:42:54+00:00", "2026-11-20T14:42:54-05:00",
                                   "2026-11-20T19:42:54Z", "Fri, 20 Nov 2026 19:42:54 GMT",
                                   "2026-11-20T19:42:54"])
def test_parse_time_formats(value):
    assert sd.parse_time(value) == datetime(2026, 11, 20, 19, 42, 54, tzinfo=timezone.utc)


def test_parse_time_bad_values():
    assert sd.parse_time(None) is None and sd.parse_time("") is None and sd.parse_time("soon") is None


# ── classify ────────────────────────────────────────────────────────────────

def test_retitled_article_is_a_url_duplicate():
    """The audit's SEC story: same Yahoo URL, title edited from $2.5T to $2.7T."""
    t = _trig("Nvidia stock extends $2.7 trillion rout as SEC probe reportedly investigates its accounting",
              YAHOO, published=NOW - timedelta(hours=2.3))
    prior = [_prior(98569, 1.75, "Nvidia stock extends $2.5 trillion rout as SEC probe reportedly "
                    "investigates its accounting", YAHOO)]
    rec = sd.classify(t, prior, NOW, 24, 24)
    assert rec["duplicate_of"] == {"signal_id": 98569, "timestamp": "2026-11-20T20:15:00+00:00",
                                   "hours_before": 1.75, "match": "url", "title_changed": True}
    assert rec["would_skip_duplicate"] is True and rec["duplicate_count"] == 1
    assert rec["published_age_hours"] == 2.3 and rec["would_skip_stale"] is False


def test_same_story_key_different_url_is_a_duplicate():
    t = _trig("Nvidia stock plunges 12% - CNBC", "https://news.google.com/rss/articles/B?oc=5")
    prior = [_prior(1, 3, "Nvidia stock plunges 12% - CNBC", "https://news.google.com/rss/articles/A?oc=5")]
    rec = sd.classify(t, prior, NOW, 24, 24)
    assert rec["duplicate_of"]["match"] == "story_key" and rec["would_skip_duplicate"] is True
    assert rec["duplicate_of"]["title_changed"] is False


def test_newest_match_is_reported_and_window_applies():
    t = _trig("Same story", "https://a.com/s")
    prior = [_prior(1, 30, "Same story", "https://a.com/s"),        # outside 24 h
             _prior(2, 10, "Same story", "https://a.com/s"),
             _prior(3, 2, "Same story", "https://a.com/s"),
             _prior(4, 1, "Other story", "https://a.com/o")]
    rec = sd.classify(t, prior, NOW, 24, 24)
    assert rec["duplicate_of"]["signal_id"] == 3 and rec["duplicate_count"] == 2
    assert rec["prior_event_verdicts"] == 4


def test_prior_at_or_after_decision_is_ignored():
    t = _trig("Same story", "https://a.com/s")
    rec = sd.classify(t, [_prior(1, 0, "Same story", "https://a.com/s"),
                          _prior(2, -1, "Same story", "https://a.com/s")], NOW, 24, 24)
    assert rec["duplicate_of"] is None and rec["would_skip_duplicate"] is False


def test_different_story_not_duplicate():
    rec = sd.classify(_trig("A", "https://a.com/1"), [_prior(1, 1, "B", "https://a.com/2")], NOW, 24, 24)
    assert rec["duplicate_of"] is None and rec["would_skip_duplicate"] is False


def test_stale_label_at_cutoff():
    rec = sd.classify(_trig("A", published=NOW - timedelta(hours=36)), [], NOW, 24, 24)
    assert rec["published_age_hours"] == 36.0 and rec["would_skip_stale"] is True
    assert rec["published_time_source"] == "published_at"
    rec = sd.classify(_trig("A", published=NOW - timedelta(hours=24)), [], NOW, 24, 24)
    assert rec["would_skip_stale"] is False      # at the cut-off is not past it


def test_unknown_time_and_no_key_give_none_labels():
    rec = sd.classify({"path": "event", "source": "options"}, [], NOW, 24, 24)
    assert rec["story_key"] is None and rec["would_skip_duplicate"] is None
    assert rec["published_age_hours"] is None and rec["would_skip_stale"] is None


def test_recheck_is_exempt():
    t = _trig("re-analysis", None, source="stale_entry_recheck", published=NOW - timedelta(hours=40))
    rec = sd.classify(t, [_prior(1, 1, "re-analysis", None, "stale_entry_recheck")], NOW, 24, 24)
    assert rec["exempt"] == "recheck"
    assert rec["would_skip_duplicate"] is False and rec["would_skip_stale"] is False


def test_classify_tolerates_bad_prior_rows():
    t = _trig("A", "https://a.com/1")
    rec = sd.classify(t, [{"timestamp": None}, {"timestamp": "bad"}, {}], NOW, 24, 24)
    assert rec["duplicate_of"] is None


# ── live record ─────────────────────────────────────────────────────────────

@pytest.mark.switches(STORY_DEDUP_SHADOW_ENABLED=False)
def test_record_off_returns_none():
    assert sd.story_dedup_record("NVDA", _trig("A")) is None


def test_record_reads_window_strictly(shadow_on):
    with patch("db.recent_event_verdicts", return_value=[]) as rv:
        rec = sd.story_dedup_record("NVDA", _trig("A", "https://a.com/1"))
    rv.assert_called_once_with("NVDA", 24 * 60, strict=True)
    assert rec["read"] == "ok" and rec["would_skip_duplicate"] is False
    datetime.fromisoformat(rec["decided_at"])


def test_record_read_failure_gives_none_dup_labels(shadow_on):
    published = datetime.now(timezone.utc) - timedelta(hours=30)
    with patch("db.recent_event_verdicts", side_effect=RuntimeError("db down")):
        rec = sd.story_dedup_record("NVDA", _trig("A", "https://a.com/1", published=published))
    assert rec["read"] == "error: RuntimeError"
    assert rec["would_skip_duplicate"] is None and rec["duplicate_of"] is None
    assert rec["would_skip_stale"] is True                    # age needs no read


def test_record_never_raises(shadow_on):
    with patch("agents.story_dedup.classify", side_effect=ValueError("boom")), \
         patch("db.recent_event_verdicts", return_value=[]):
        rec = sd.story_dedup_record("NVDA", _trig("A"))
    assert rec == {"version": sd.STORY_DEDUP_VERSION, "read": "error: ValueError"}


def test_db_strict_raises_and_default_returns_empty():
    import db
    with patch("db._get_conn", side_effect=RuntimeError("x")):
        with pytest.raises(RuntimeError):
            db.recent_event_verdicts("NVDA", 60, strict=True)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("q")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.recent_event_verdicts("NVDA", 60) == []
        with pytest.raises(RuntimeError):
            db.recent_event_verdicts("NVDA", 60, strict=True)


def test_db_rows_carry_signal_id_and_source():
    import db
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [(NOW, "T", "U", "HOLD", 7, "news")]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        rows = db.recent_event_verdicts("NVDA", 60)
    assert rows == [{"timestamp": NOW, "title": "T", "url": "U", "model_action": "HOLD",
                     "signal_id": 7, "source": "news"}]


# ── deep_analyze integration ────────────────────────────────────────────────

_EVENT = {"ticker": "NVDA", "source": "news", "title": "Nvidia stock plunges 12% - CNBC",
          "url": "https://news.google.com/rss/articles/B?oc=5", "text": "x",
          "published_at": "2026-11-20T19:26:29+00:00"}


def _run_deep():
    seen = {}

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        seen["trigger"], seen["brief"] = trigger, situation_brief
        return {"action": "HOLD", "ticker": "NVDA", "conviction": 5, "pass_to_risk": False}
    with patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch("context_engine.assemble_brief", return_value="full brief"), \
         patch("agents.crew._get_positions", return_value={}), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("agents.analyst.analyze", fake_analyze), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("agents.crew._c6_bar_unrouted"):
        result = crew.deep_analyze(dict(_EVENT), {"relevant": True})
    return seen, result


def test_deep_analyze_puts_record_on_trigger_only(shadow_on):
    prior = [{"signal_id": 5, "timestamp": datetime.now(timezone.utc) - timedelta(minutes=1),
              "title": "Nvidia stock plunges 12% - CNBC", "url": "https://news.google.com/rss/articles/A?oc=5",
              "source": "news", "model_action": "HOLD"}]
    with patch("db.recent_event_verdicts", return_value=prior):
        seen, result = _run_deep()
    rec = seen["trigger"]["story_dedup"]
    assert rec["would_skip_duplicate"] is True and rec["duplicate_of"]["signal_id"] == 5
    assert rec["published_time_source"] == "published_at" and rec["published_age_hours"] is not None
    assert seen["brief"] == "full brief"            # the brief is untouched
    assert result["action"] == "HOLD"               # the analysis still ran


@pytest.mark.switches(STORY_DEDUP_SHADOW_ENABLED=False)
def test_deep_analyze_without_shadow_has_no_key():
    seen, _ = _run_deep()
    assert "story_dedup" not in seen["trigger"]


def test_deep_analyze_survives_shadow_crash(shadow_on):
    with patch("agents.story_dedup.classify", side_effect=RuntimeError("boom")), \
         patch("db.recent_event_verdicts", return_value=[]):
        seen, result = _run_deep()
    assert seen["trigger"]["story_dedup"]["read"] == "error: RuntimeError"
    assert result["action"] == "HOLD"


# ── review fixes: suffix only on Google News, date-only times, prompt identity ─

def test_dash_clause_outside_google_news_is_part_of_the_headline():
    beats = sd.story_key("NVDA Q3 - revenue beats", "https://www.reuters.com/a", "news")
    misses = sd.story_key("NVDA Q3 - revenue misses", "https://www.reuters.com/b", "news")
    assert beats != misses and beats == "reuters.com|nvda q3 revenue beats"
    t = _trig("NVDA Q3 - revenue misses", "https://www.reuters.com/b")
    rec = sd.classify(t, [_prior(1, 1, "NVDA Q3 - revenue beats", "https://www.reuters.com/a")], NOW, 24, 24)
    assert rec["duplicate_of"] is None and rec["would_skip_duplicate"] is False


def test_google_news_suffix_is_still_stripped():
    gn = "https://news.google.com/rss/articles/X?oc=5"
    assert sd.story_key("NVDA Q3 - revenue beats - Reuters", gn, "news") == "gn:reuters|nvda q3 revenue beats"
    assert sd.normalise_title("A - Bloomberg") == "a bloomberg"
    assert sd.normalise_title("A - Bloomberg", strip_publisher=True) == "a"


def test_title_changed_uses_each_rows_own_suffix_rule():
    url = "https://finance.yahoo.com/x.html"
    t = _trig("Stocks rise - live updates", url)
    rec = sd.classify(t, [_prior(1, 1, "Stocks rise - live coverage", url)], NOW, 24, 24)
    assert rec["duplicate_of"]["match"] == "url" and rec["duplicate_of"]["title_changed"] is True


@pytest.mark.parametrize("value", ["2026-11-18", date(2026, 11, 18)])
def test_date_only_time_gives_lower_bound_not_midnight_utc(value):
    t = {"path": "event", "source": "congress", "title": "Rep. buys NVDA",
         "event_time": value, "event_time_source": "date"}
    rec = sd.classify(t, [], NOW, 24, 24)       # NOW = 11-20 22:00 UTC (17:00 ET)
    assert rec["published_time_precision"] == "date" and rec["published_age_reason"] == "date_only"
    assert rec["published_age_hours"] is None
    # From 11-18 23:59:59 ET (11-19 04:59:59 UTC) to 11-20 22:00 UTC.
    assert rec["published_age_hours_min"] == pytest.approx(41.0, abs=0.01)
    assert rec["would_skip_stale"] is True       # even the lower bound is past 24 h


def test_date_only_within_cutoff_is_unknown():
    t = {"path": "event", "source": "congress", "title": "x", "event_time": "2026-11-20"}
    rec = sd.classify(t, [], NOW, 24, 24)
    assert rec["published_age_hours_min"] < 0 and rec["would_skip_stale"] is None


def test_full_timestamps_keep_exact_age():
    rec = sd.classify(_trig("A", published=NOW - timedelta(hours=5)), [], NOW, 24, 24)
    assert rec["published_time_precision"] == "datetime" and rec["published_age_reason"] is None
    assert rec["published_age_hours"] == 5.0 and rec["published_age_hours_min"] is None
    none = sd.classify({"path": "event", "title": "A"}, [], NOW, 24, 24)
    assert none["published_age_reason"] == "no_item_time" and none["published_time_precision"] is None


def test_time_precision():
    assert sd.time_precision("2026-11-20") == "date"
    assert sd.time_precision("2026-11-20T00:00:00+00:00") == "datetime"
    assert sd.time_precision(datetime(2026, 11, 20)) == "datetime"
    assert sd.time_precision("junk") is None and sd.time_precision(None) is None


def _prompt_through_real_analyze(monkeypatch, enabled: bool):
    """crew.deep_analyze -> the real analyst.analyze, with only the model call
    and the prompt's own knowledge/db reads stubbed; returns the prompt sent
    and the trigger logged on the verdict row."""
    import json
    import config
    monkeypatch.setattr(config, "STORY_DEDUP_SHADOW_ENABLED", enabled)
    sent = []
    reply = json.dumps({"action": "WATCH", "conviction": 4, "reasoning": "r", "sources": []})
    prior = [{"signal_id": 5, "timestamp": datetime.now(timezone.utc) - timedelta(minutes=5),
              "title": _EVENT["title"], "url": _EVENT["url"], "source": "news", "model_action": "HOLD"}]
    with patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch("context_engine.assemble_brief", return_value="full brief"), \
         patch("agents.crew._get_positions", return_value={}), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("agents.market_regime.get_cached_regime",
               return_value={"regime": "Neutral", "score": 50, "reasoning": ""}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("knowledge_loader.load_thesis", return_value=""), \
         patch("knowledge_loader.load_philosophy", return_value=""), \
         patch("agents.chase_risk.compute_chase_risk", return_value=None), \
         patch("agents.price_action.compute_price_action", return_value=None), \
         patch("db.get_fundamentals", return_value={"market_cap": 1_000_000_000}), \
         patch("db.get_conviction_calibration", return_value=[]), \
         patch("db.recent_event_verdicts", return_value=prior), \
         patch("db.log_signal", return_value=7) as log_signal, patch("feed.emit"), \
         patch("agents.crew._c6_bar_unrouted"), \
         patch("agents.analyst._call_ollama", side_effect=lambda p, *a, **k: sent.append(p) or reply):
        crew.deep_analyze(dict(_EVENT), {"relevant": True})
    assert len(sent) == 1
    triggers = [(c.kwargs.get("provenance") or {}).get("trigger") for c in log_signal.call_args_list]
    triggers = [t for t in triggers if isinstance(t, dict) and t.get("path") == "event"]
    trig = triggers[0] if triggers else None
    return sent[0], trig


def test_prompt_is_byte_identical_with_shadow_on_and_off(monkeypatch):
    on_prompt, on_trig = _prompt_through_real_analyze(monkeypatch, True)
    off_prompt, off_trig = _prompt_through_real_analyze(monkeypatch, False)
    assert "full brief" in on_prompt               # the real prompt builder ran
    assert on_prompt.encode() == off_prompt.encode()
    assert on_trig is not None and on_trig["story_dedup"]["would_skip_duplicate"] is True
    assert "story_dedup" not in off_trig
    assert "story_dedup" not in on_prompt and "would_skip" not in on_prompt
