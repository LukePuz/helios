"""Per-source data health (ingestion/source_health.py, roadmap 04e4): each
poll noted ok / empty / error in Redis, one daily source_health row, and one
digest line for sources dead 2+ sessions. Record and report only."""
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from ingestion import source_health as sh

_REAL_REDIS = sh._redis
_REAL_DAILY = sh.daily_record
# 2026-11-24 21:15 UTC = 16:15 EST, the digest's time
_T = datetime(2026, 11, 24, 21, 15, tzinfo=timezone.utc)


class _Pipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def hset(self, key, field=None, value=None, mapping=None):
        self.ops.append(("hset", key, dict(mapping) if mapping else {field: value}))
        return self

    def hincrby(self, key, field, n):
        self.ops.append(("hincrby", key, field, n))
        return self

    def expire(self, key, s):
        self.ops.append(("expire", key, s))
        return self

    def execute(self):
        if self.r.fail:
            raise self.r.fail
        for op in self.ops:
            h = self.r.h.setdefault(op[1], {})
            if op[0] == "hset":
                h.update({k: str(v) for k, v in op[2].items()})
            elif op[0] == "hincrby":
                h[op[2]] = str(int(h.get(op[2], 0)) + op[3])
            else:
                self.r.ttl[op[1]] = op[2]
        return []


class _FakeRedis:
    def __init__(self):
        self.h, self.ttl, self.fail, self.calls = {}, {}, None, 0

    def pipeline(self, transaction=False):
        self.calls += 1
        return _Pipe(self)

    def scan_iter(self, match, count=None):
        if self.fail:
            raise self.fail
        return [k for k in self.h if k.startswith(match.rstrip("*"))]

    def hgetall(self, key):
        return dict(self.h.get(key, {}))

    def hdel(self, key, *fields):
        for f in fields:
            self.h.get(key, {}).pop(f, None)


@pytest.fixture
def fake(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(sh, "_redis", lambda: r)
    monkeypatch.setattr(sh, "_down_until", 0.0)
    monkeypatch.setattr(sh, "daily_record", _REAL_DAILY)
    monkeypatch.setattr("config.SOURCE_HEALTH_RECORD_ENABLED", True)
    monkeypatch.setattr("config.TRADING_ENV", "live")
    return r


def _h(r, source):
    return r.h.get(f"helios:source_health:live:{source}", {})


class TestNote:
    def test_ok_empty_error_streaks(self, fake):
        assert sh.note("fred:VIXCLS", sh.EMPTY, now=_T) is True
        sh.note("fred:VIXCLS", sh.ERROR, now=_T)
        sh.note("fred:VIXCLS", sh.ERROR, now=_T)
        h = _h(fake, "fred:VIXCLS")
        assert h["consec_error"] == "2"
        assert h["consec_empty"] == "0"
        assert h["consec_not_ok"] == "3"
        assert h["last_outcome"] == "error"
        assert h["day:2026-11-24:error"] == "2"
        assert h["day:2026-11-24:empty"] == "1"
        assert "last_ok_at" not in h
        sh.note("fred:VIXCLS", sh.OK, now=_T)
        h = _h(fake, "fred:VIXCLS")
        assert h["consec_not_ok"] == "0"
        assert h["consec_error"] == "0"
        assert h["last_ok_at"] == _T.isoformat()
        assert fake.ttl

    def test_blocked_is_counted_and_leaves_the_streaks(self, fake):
        sh.note("polygon_snapshot", sh.ERROR, now=_T)
        assert sh.note("polygon_snapshot", sh.BLOCKED, now=_T) is True
        h = _h(fake, "polygon_snapshot")
        assert h["day:2026-11-24:blocked"] == "1"
        assert h["last_outcome"] == "blocked"
        assert h["last_blocked_at"] == _T.isoformat()
        assert h["consec_error"] == "1"
        assert h["consec_not_ok"] == "1"

    def test_off_or_unknown_outcome_writes_nothing(self, fake, monkeypatch):
        assert sh.note("x", "maybe") is False
        monkeypatch.setattr("config.SOURCE_HEALTH_RECORD_ENABLED", False)
        assert sh.note("x", sh.OK) is False
        assert fake.calls == 0

    def test_a_connection_error_backs_off_and_never_raises(self, fake):
        import redis
        fake.fail = redis.exceptions.ConnectionError("down")
        assert sh.note("edgar", sh.OK) is False
        calls = fake.calls
        assert sh.note("edgar", sh.OK) is False
        assert fake.calls == calls          # skipped while backing off

    def test_another_error_is_swallowed_without_backoff(self, fake):
        fake.fail = ValueError("odd")
        assert sh.note("edgar", sh.OK) is False
        fake.fail = None
        assert sh.note("edgar", sh.OK) is True

    def test_the_client_has_short_timeouts(self, monkeypatch):
        seen = {}

        def from_url(url, **kw):
            seen.update(kw)
            return MagicMock()
        monkeypatch.setattr(sh, "_client", None)
        with patch("redis.Redis.from_url", side_effect=from_url):
            _REAL_REDIS()
        monkeypatch.setattr(sh, "_client", None)
        assert seen["socket_timeout"] <= 0.2
        assert seen["socket_connect_timeout"] <= 0.2


def _day(r, source, day_t, outcomes):
    for o in outcomes:
        sh.note(source, o, now=day_t)


class TestDailyRecord:
    def _run(self, now, state):
        rows = []
        with patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("db.log_activity_direct",
                   side_effect=lambda *a, **k: rows.append(a) or True):
            dead = sh.daily_record(now=now)
        return dead, rows

    def test_dead_two_sessions_is_named_on_the_second(self, fake):
        state = {}
        _day(fake, "analyst_ratings", _T - timedelta(days=1), [sh.EMPTY, sh.EMPTY])
        _day(fake, "edgar", _T - timedelta(days=1), [sh.OK])
        dead, rows = self._run(_T - timedelta(days=1), state)
        assert dead == []
        assert rows[0][4]["sources"]["analyst_ratings"]["dead_sessions"] == 1
        _day(fake, "analyst_ratings", _T, [sh.EMPTY])
        _day(fake, "edgar", _T, [sh.OK])
        dead, rows = self._run(_T, state)
        assert dead == ["analyst_ratings"]
        row = rows[0][4]
        assert row["dead"] == ["analyst_ratings"]
        assert row["sources"]["analyst_ratings"]["dead_sessions"] == 2
        assert row["sources"]["analyst_ratings"]["polls"] == 1
        assert row["sources"]["edgar"]["dead_sessions"] == 0
        assert rows[0][3] == "negative"

    def test_a_rerun_the_same_day_does_not_count_twice(self, fake):
        state = {}
        _day(fake, "polygon_snapshot", _T, [sh.ERROR])
        self._run(_T, state)
        dead, rows = self._run(_T, state)
        assert rows[0][4]["sources"]["polygon_snapshot"]["dead_sessions"] == 1

    def test_an_ok_poll_resets_and_a_day_not_polled_keeps_the_streak(self, fake):
        state = {}
        for d in (2, 1):
            _day(fake, "fred:VIXCLS", _T - timedelta(days=d), [sh.ERROR])
            self._run(_T - timedelta(days=d), state)
        dead, rows = self._run(_T, state)    # not polled today
        assert dead == ["fred:VIXCLS"]
        assert rows[0][4]["sources"]["fred:VIXCLS"]["judged"] is False
        _day(fake, "fred:VIXCLS", _T + timedelta(days=1), [sh.ERROR, sh.OK])
        dead, rows = self._run(_T + timedelta(days=1), state)
        assert dead == []
        assert rows[0][4]["sources"]["fred:VIXCLS"]["dead_sessions"] == 0

    def test_days_since_success_and_old_day_fields_dropped(self, fake):
        state = {}
        _day(fake, "news:yfinance", _T - timedelta(days=3), [sh.OK])
        _day(fake, "news:yfinance", _T, [sh.EMPTY])
        dead, rows = self._run(_T, state)
        assert rows[0][4]["sources"]["news:yfinance"]["days_since_success"] == 3
        assert not any(f.startswith("day:2026-11-21") for f in _h(fake, "news:yfinance"))

    def test_a_failed_redis_read_writes_a_row_and_leaves_the_book(self, fake):
        state = {"source_health_live": json.dumps({"edgar": {"dead_sessions": 5}})}
        fake.fail = RuntimeError("down")
        dead, rows = self._run(_T, state)
        assert dead == []
        assert rows[0][4]["read"] == "RuntimeError"
        assert json.loads(state["source_health_live"]) == {"edgar": {"dead_sessions": 5}}

    def test_off_writes_nothing(self, fake, monkeypatch):
        monkeypatch.setattr("config.SOURCE_HEALTH_RECORD_ENABLED", False)
        dead, rows = self._run(_T, {})
        assert dead == []
        assert rows == []

    def test_the_row_has_no_values_only_counts_and_times(self, fake):
        _day(fake, "options", _T, [sh.EMPTY])
        dead, rows = self._run(_T, {})
        src = rows[0][4]["sources"]["options"]
        assert set(src) == {"polls", "ok", "empty", "error", "blocked", "judged", "dead_today",
                            "blocked_only", "last_blocked_at", "consec_empty",
                            "consec_error", "consec_not_ok", "last_poll_at", "last_success_at",
                            "last_error_at", "days_since_success", "dead_sessions"}

    def test_a_blocked_source_is_never_dead_and_never_named(self, fake):
        state = {}
        for d in (3, 2, 1, 0):
            _day(fake, "polygon_snapshot", _T - timedelta(days=d), [sh.BLOCKED, sh.BLOCKED])
            dead, rows = self._run(_T - timedelta(days=d), state)
            assert dead == []
        src = rows[0][4]["sources"]["polygon_snapshot"]
        assert src["blocked"] == 2
        assert src["polls"] == 0
        assert src["judged"] is False
        assert src["blocked_only"] is True
        assert src["dead_sessions"] == 0
        assert rows[0][4]["dead"] == []

    def test_a_blocked_only_day_keeps_a_streak_but_leaves_the_line(self, fake):
        state = {}
        for d in (2, 1):
            _day(fake, "polygon", _T - timedelta(days=d), [sh.ERROR])
            self._run(_T - timedelta(days=d), state)
        _day(fake, "polygon", _T, [sh.BLOCKED])
        dead, rows = self._run(_T, state)
        assert dead == []
        assert rows[0][4]["sources"]["polygon"]["dead_sessions"] == 2
        _day(fake, "polygon", _T + timedelta(days=1), [sh.BLOCKED, sh.ERROR])
        dead, rows = self._run(_T + timedelta(days=1), state)
        assert dead == ["polygon"]                 # an error that day judges it
        assert rows[0][4]["sources"]["polygon"]["dead_sessions"] == 3


class TestDigest:
    def test_one_line_names_the_dead_sources(self):
        import alerts
        with patch("alerts._send") as send:
            alerts.send_daily_digest(portfolio_value=1.0, day_pl=0.0, day_pl_pct=0.0, spy_pl_pct=None,
                                     positions_text=[], catalysts_text=[], pending_count=0,
                                     halted=False, ingestion_issues=[],
                                     dead_sources=["analyst_ratings", "polygon_snapshot"])
        text = send.call_args[0][0]
        assert "\nData sources dead 2+ sessions: analyst_ratings, polygon_snapshot" in text
        assert text.count("Data sources dead") == 1

    def test_no_dead_source_no_line(self):
        import alerts
        with patch("alerts._send") as send:
            alerts.send_daily_digest(portfolio_value=1.0, day_pl=0.0, day_pl_pct=0.0, spy_pl_pct=None,
                                     positions_text=[], catalysts_text=[], pending_count=0,
                                     halted=False, ingestion_issues=[], dead_sources=[])
        assert "Data sources dead" not in send.call_args[0][0]

    def _task(self, daily):
        import tasks
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=1100.0)
        client.get_all_positions.return_value = []
        digest = MagicMock()
        with patch.object(tasks, "_get_client", return_value=client), \
             patch.object(tasks, "_get_stale_ingestion_sources", return_value=[]), \
             patch("db.get_portfolio_history", return_value=[]), \
             patch("db.get_all_position_stops", return_value=[]), \
             patch("db.get_verdict_day_summary", return_value={}), \
             patch("db.get_pending_approvals", return_value=[]), \
             patch("db.get_state", return_value="false"), \
             patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
             patch("catalyst_service.client.query_events", return_value=[]), \
             patch("ingestion.source_health.daily_record", daily), \
             patch("alerts.send_daily_digest", digest):
            tasks.send_daily_digest.run()
        return digest

    def test_the_task_passes_the_dead_sources(self):
        digest = self._task(MagicMock(return_value=["fred:VIXCLS"]))
        assert digest.call_args.kwargs["dead_sources"] == ["fred:VIXCLS"]

    def test_a_record_that_raises_never_stops_the_digest(self, caplog):
        with caplog.at_level(logging.WARNING):
            digest = self._task(MagicMock(side_effect=RuntimeError("boom")))
        digest.assert_called_once()
        assert digest.call_args.kwargs["dead_sources"] == []
        assert any("source health record failed" in r.getMessage() for r in caplog.records)


class TestHooks:
    @pytest.fixture
    def noted(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sh, "note", lambda source, outcome, now=None: calls.append((source, outcome)) or True)
        return calls

    @pytest.mark.parametrize("stats, expect", [
        ({"tickers": 3, "failed": 3, "empty": 0, "recent": 0, "returned": 0}, "error"),
        ({"tickers": 3, "failed": 1, "empty": 2, "recent": 0, "returned": 0}, "empty"),
        ({"tickers": 3, "failed": 0, "empty": 0, "recent": 2, "returned": 4}, "ok"),
    ])
    def test_analyst_ratings(self, noted, stats, expect):
        from ingestion import analyst_ratings as ar
        ar._STATS.clear()
        ar._STATS.update({**stats, "first_error": None})
        ar._log_stats()
        assert noted == [("analyst_ratings", expect)]

    def test_news_per_feed(self, noted, monkeypatch):
        from ingestion import news
        monkeypatch.setattr(news, "_track_feed_outage", lambda *a, **k: False)
        monkeypatch.setattr(news, "_FEED_STATS", {
            "google_news": {"tickers": 2, "raw": 5, "returned": 3, "failed": 0, "first_error": None},
            "yfinance": {"tickers": 2, "raw": 0, "returned": 0, "failed": 0, "first_error": None}})
        news._log_feed_stats()
        assert sorted(noted) == [("news:google_news", "ok"), ("news:yfinance", "empty")]

    @pytest.fixture
    def wrapped(self, monkeypatch):
        """instrument() as at worker boot; monkeypatch restores the originals."""
        from ingestion import macro, polygon_stream as ps
        monkeypatch.setattr(ps, "_get", ps._get)
        monkeypatch.setattr(macro, "_fetch_series", macro._fetch_series)
        monkeypatch.setattr("config.SOURCE_HEALTH_RECORD_ENABLED", True)
        assert sh.instrument() == ["ingestion.polygon_stream._get", "ingestion.macro._fetch_series"]
        return ps, macro

    def test_instrument_wraps_once_and_not_when_off(self, wrapped, monkeypatch):
        ps, macro = wrapped
        first = ps._get
        sh.instrument()
        assert ps._get is first
        assert ps._get.__wrapped__.__name__ == "_get"
        monkeypatch.setattr(ps, "_get", first.__wrapped__)
        monkeypatch.setattr("config.SOURCE_HEALTH_RECORD_ENABLED", False)
        assert sh.instrument() == []
        assert ps._get is first.__wrapped__

    def test_the_brief_modules_are_not_edited(self):
        """brief_template_full_sha hashes these modules whole; the hooks live
        outside them."""
        import inspect
        from ingestion import macro, polygon_stream as ps
        for mod in (macro, ps):
            assert "source_health" not in inspect.getsource(mod)

    def test_the_worker_boot_installs_it(self, monkeypatch):
        import tasks
        calls = []
        monkeypatch.setattr(sh, "instrument", lambda: calls.append(1) or [])
        tasks._instrument_source_health(sender=None)
        assert calls == [1]
        monkeypatch.setattr(sh, "instrument", MagicMock(side_effect=RuntimeError("x")))
        tasks._instrument_source_health(sender=None)     # never raises

    def test_polygon_counts_each_call_without_a_log_line(self, wrapped, noted, monkeypatch, caplog):
        import requests
        ps, _ = wrapped
        monkeypatch.setattr(ps, "API_KEY", "k")
        resp = MagicMock(status_code=403, text="NOT_AUTHORIZED")
        err = requests.exceptions.HTTPError(response=resp)
        bad = MagicMock()
        bad.raise_for_status.side_effect = err
        good = MagicMock()
        good.json.return_value = {"x": 1}
        monkeypatch.setattr(ps, "_ENTITLEMENT_BLOCKED", set())
        other = MagicMock()
        other.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=500, text="oops"))
        with patch("requests.get", side_effect=[bad, bad, other, good]), caplog.at_level(logging.DEBUG):
            for _ in range(2):
                ps.get_snapshot("NVDA")
            assert ps._get("/v2/aggs/ticker/NVDA") == {}
            assert ps._get("/v2/aggs/ticker/NVDA") == {"x": 1}
        assert noted == [("polygon_snapshot", "blocked"), ("polygon_snapshot", "blocked"),
                         ("polygon", "error"), ("polygon", "ok")]
        assert not [r for r in caplog.records if "health" in r.getMessage().lower()]

    def test_polygon_without_a_key_is_an_error(self, wrapped, noted, monkeypatch):
        ps, _ = wrapped
        monkeypatch.setattr(ps, "API_KEY", "")
        assert ps._get(endpoint="/v2/snapshot/locale/us/markets/stocks/tickers/NVDA") == {}
        assert noted == [("polygon_snapshot", "error")]

    @pytest.mark.parametrize("obs, raise_, expect", [
        ([{"value": "1"}], None, "ok"), ([], None, "empty"), (None, RuntimeError("500"), "empty")])
    def test_fred(self, wrapped, noted, monkeypatch, obs, raise_, expect):
        _, macro = wrapped
        monkeypatch.setattr(macro, "FRED_API_KEY", "k")
        resp = MagicMock()
        if raise_:
            resp.raise_for_status.side_effect = raise_
        resp.json.return_value = {"observations": obs or []}
        with patch("requests.get", return_value=resp):
            assert macro._fetch_series("VIXCLS") == (obs or [])
        assert noted == [("fred:VIXCLS", expect)]

    def test_fred_without_a_key_is_an_error(self, wrapped, noted, monkeypatch):
        _, macro = wrapped
        monkeypatch.setattr(macro, "FRED_API_KEY", "")
        macro._fetch_series(series_id="VIXCLS")
        assert noted == [("fred:VIXCLS", "error")]

    def test_the_wrapped_functions_own_exception_is_noted_then_re_raised(self, noted):
        class Boom(Exception):
            pass
        err = Boom("x")

        def fn(endpoint):
            raise err
        w = sh._wrap(fn, sh._polygon_source, sh._polygon_outcome)
        with pytest.raises(Boom) as got:
            w("/v2/snapshot/locale/us/markets/stocks/tickers/NVDA")
        assert got.value is err
        assert noted == [("polygon_snapshot", "error")]

    def test_a_note_that_raises_never_changes_the_return(self, wrapped, monkeypatch):
        ps, _ = wrapped
        monkeypatch.setattr(sh, "note", MagicMock(side_effect=RuntimeError("x")))
        monkeypatch.setattr(ps, "API_KEY", "")
        assert ps._get("/v2/aggs/ticker/NVDA") == {}

    @pytest.mark.parametrize("results, expect", [({"A": [], "B": []}, "ok"),
                                                 ({"A": [{"form": "8-K"}], "B": None}, "ok"),
                                                 ({"A": None, "B": None}, "error")])
    def test_edgar_pass(self, noted, results, expect):
        from ingestion import edgar
        with patch.object(edgar, "TICKER_TO_CIK", {"A": "1", "B": "2"}), \
             patch.object(edgar, "_get_recent_filings", side_effect=lambda t, c: results[t]), \
             patch.object(edgar, "_get_state", return_value=None), \
             patch.object(edgar, "_set_state"), \
             patch.object(edgar, "_aged_out_to_name", return_value=([], 0)), \
             patch("ingestion.edgar.time.sleep"):
            edgar._poll_all()
        assert noted == [("edgar", expect)]

    @pytest.mark.parametrize("results, expect", [({"A": [], "B": []}, "ok"),
                                                 ({"A": None, "B": None}, "error")])
    def test_insider_pass(self, noted, results, expect):
        from ingestion import edgar, insider
        with patch.object(insider, "TICKER_TO_CIK", {"A": "1", "B": "2"}), \
             patch("ingestion.insider._search_form4", side_effect=lambda t, c: results[t]), \
             patch.object(edgar, "_backoff_until", return_value=None), \
             patch.object(edgar, "_get_state", return_value=None), \
             patch.object(edgar, "_set_state"), \
             patch("ingestion.insider.time.sleep"):
            insider.poll_once()
        assert noted == [("insider", expect)]

    @pytest.mark.parametrize("summaries, expect", [
        ({"A": {"summary": "No free options data available (yfinance chain empty)."}}, "empty"),
        ({"A": {"summary": "No free options data available (yfinance chain empty)."},
          "B": {"summary": "calls 10 puts 5"}}, "ok")])
    def test_options(self, noted, monkeypatch, summaries, expect):
        from ingestion import options_free as of
        monkeypatch.setattr("knowledge_loader.get_all_tickers", lambda: list(summaries))
        monkeypatch.setattr(of, "get_flow_summary", lambda t, *a: summaries[t])
        monkeypatch.setattr(of.time, "sleep", lambda s: None)
        of.get_all_flow_summaries()
        assert noted == [("options", expect)]

    @pytest.mark.parametrize("vix, expect", [(18.5, "ok"), (None, "error")])
    def test_yfinance_vix(self, noted, vix, expect):
        import tasks
        with patch("agents.market_regime.get_intraday_vix", return_value=vix), \
             patch("redis.Redis.from_url", return_value=MagicMock()):
            tasks.cache_live_vix()
        assert noted == [("yfinance_vix", expect)]
