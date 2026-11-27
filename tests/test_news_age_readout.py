"""scripts/news_age_readout.py: an event item's age at its screen and at its
verdict (time minus provenance.trigger.published_at, hours) per day, feed,
ticker, held and day x ticker, with fixed buckets, published_at none /
unparseable / future counts and unmatched screen rows; screened-in items
that reached a verdict only; counts, hours and ids only; the read is
read-only on the pooler port and skips the stack's start."""
import importlib.util
import json
import os
import sys
import types
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "news_age_readout.py")
_spec = importlib.util.spec_from_file_location("news_age_readout", _PATH)
na = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(na)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 27, 10, 0, tzinfo=NY)
HEAD = "pinned0"
V_TS = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
TITLE = "Nvidia faces credit warning from lenders"   # never in the output


def _row(sid=1, ts=V_TS, sym="NVDA", source="news", host="news.google.com", has_url=True,
         pub="2026-11-27T13:00:00+00:00", plan_held="true", held_review="true", screen_id=11,
         screened=datetime(2026, 11, 27, 14, 30, tzinfo=UTC)):
    return (sid, ts, sym, source, host, has_url, pub, plan_held, held_review, screen_id, screened)


# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("v, want", [
    ("2026-11-27T13:00:00+00:00", datetime(2026, 11, 27, 13, tzinfo=UTC)),
    ("2026-11-27T08:00:00-05:00", datetime(2026, 11, 27, 13, tzinfo=UTC)),
    ("2026-11-27T13:00:00Z", datetime(2026, 11, 27, 13, tzinfo=UTC)),
    ("Fri, 27 Nov 2026 13:00:00 GMT", datetime(2026, 11, 27, 13, tzinfo=UTC)),
    ("Fri, 27 Nov 2026 08:00:00 -0500", datetime(2026, 11, 27, 13, tzinfo=UTC)),
])
def test_parse_published_aware_forms(v, want):
    assert na.parse_published(v) == want


@pytest.mark.parametrize("v, want", [(None, "none"), ("", "none"), ("  ", "none"),
                                     ("2026-11-27", "unparseable"),
                                     ("2026-11-27T13:00:00", "unparseable"),
                                     ("yesterday", "unparseable"), ("None", "unparseable")])
def test_parse_published_none_and_unparseable(v, want):
    assert na.parse_published(v) == want


@pytest.mark.parametrize("ph, hr, want", [("true", None, "held"), ("false", "true", "not held"),
                                          (None, "true", "held"), (None, "false", "not held"),
                                          (None, None, "unknown"), (True, None, "held"),
                                          ("x", None, "unknown")])
def test_held_label(ph, hr, want):
    assert na.held_label(ph, hr) == want


@pytest.mark.parametrize("h, b", [(-0.01, "negative"), (0.0, "<15 min"), (0.2499, "<15 min"),
                                  (0.25, "15-60 min"), (0.99, "15-60 min"), (1.0, "1-4 h"),
                                  (3.99, "1-4 h"), (4.0, "4-24 h"), (23.99, "4-24 h"),
                                  (24.0, ">24 h"), (100.0, ">24 h")])
def test_buckets_include_their_lower_edge(h, b):
    assert na.bucket(h) == b


# --- classify ---------------------------------------------------------------

def test_classify_gives_ids_labels_and_hours_only():
    r = na.classify(_row())
    assert r == {"id": 1, "day": "2026-11-27", "ticker": "NVDA", "feed": "news news.google.com",
                 "has_url": True, "held": "held", "published": "ok", "screen_id": 11,
                 "screen_h": 1.5, "verdict_h": 2.0}


def test_no_screen_match_has_no_screen_age():
    r = na.classify(_row(screen_id=None, screened=None))
    assert r["screen_h"] is None and r["verdict_h"] == 2.0


@pytest.mark.parametrize("pub, label", [(None, "none"), ("2026-11-27", "unparseable")])
def test_no_published_has_no_ages(pub, label):
    r = na.classify(_row(pub=pub))
    assert r["published"] == label and r["screen_h"] is None and r["verdict_h"] is None


def test_feed_labels():
    assert na.classify(_row(host=None, has_url=False))["feed"] == "news no url"
    assert na.classify(_row(host=None, has_url=True))["feed"] == "news no host"
    assert na.classify(_row(source=None))["feed"] == "unknown news.google.com"


def test_naive_timestamps_are_utc_and_the_day_is_new_york():
    r = na.classify(_row(ts=datetime(2026, 11, 27, 3, 0), pub="2026-11-27T01:00:00+00:00",
                         screened=datetime(2026, 11, 27, 2, 0)))
    assert r["day"] == "2026-11-26" and r["verdict_h"] == 2.0 and r["screen_h"] == 1.0


# --- tally and build --------------------------------------------------------

def _rows():
    mk = lambda sid, vh, sh, **kw: _row(sid, pub=(V_TS - timedelta(hours=vh)).isoformat(),
                                        screened=(V_TS - timedelta(hours=vh - sh)) if sh is not None else None,
                                        screen_id=100 + sid if sh is not None else None, **kw)
    raw = [mk(1, 0.2, 0.1), mk(2, 0.5, 0.4), mk(3, 2.0, 1.9), mk(4, 6.0, 5.9),
           mk(5, 31.5, None), mk(6, -0.5, -0.6, sym="AMD", plan_held="false", host="finance.yahoo.com"),
           _row(7, sym="AMD", pub=None, plan_held="false", host="finance.yahoo.com", screen_id=None,
                screened=None),
           _row(8, sym="AMD", pub="2026-11-27", plan_held=None, held_review=None, host=None,
                has_url=False, screen_id=None, screened=None)]
    return [na.classify(r) for r in raw]


def test_tally_distribution_buckets_and_counts():
    t = na.tally(_rows())
    assert t["verdicts"] == 8 and t["published"] == {"none": 1, "unparseable": 1}
    assert t["screen_matched"] == 5
    v = t["verdict"]
    assert v["n"] == 6 and v["median"] == 1.25 and v["max"] == 31.5 and v["min"] == -0.5
    assert v["buckets"] == {"negative": 1, "<15 min": 1, "15-60 min": 1, "1-4 h": 1, "4-24 h": 1,
                            ">24 h": 1}
    assert v["future_over_5min"] == 1
    s = t["screen"]
    assert s["n"] == 5 and s["max"] == 5.9 and s["future_over_5min"] == 1
    assert s["buckets"]["<15 min"] == 1 and s["buckets"]["15-60 min"] == 1


def test_the_future_count_is_strictly_beyond_5_minutes():
    rows = [na.classify(_row(pub=(V_TS + timedelta(minutes=m)).isoformat(), screen_id=None,
                             screened=None)) for m in (4, 5, 6)]
    t = na.tally(rows)
    assert t["verdict"]["future_over_5min"] == 1 and t["verdict"]["buckets"]["negative"] == 3


def test_p90_is_inclusive_quantiles_and_one_value_is_itself():
    assert na._dist([1.0])["p90"] == 1.0
    assert na._dist([float(x) for x in range(1, 11)])["p90"] == 9.1
    assert na._dist([]) == {"n": 0, "median": None, "p90": None, "max": None, "min": None}


def test_build_keys_and_unmatched_counts():
    out = na.build(_rows())
    assert list(out["by_ticker"]) == ["AMD", "NVDA"]
    assert list(out["by_held"]) == ["held", "not held", "unknown"]
    assert list(out["by_feed"]) == ["news finance.yahoo.com", "news news.google.com", "news no url"]
    assert list(out["by_day_ticker"]) == [("2026-11-27", "AMD"), ("2026-11-27", "NVDA")]
    assert out["unmatched"] == {"with_url": 2, "no_url": 1}
    assert out["by_ticker"]["NVDA"]["verdict"]["max"] == 31.5


# --- render -----------------------------------------------------------------

def _render(rows=None, screens=(1058, 27, 131)):
    out = na.build(_rows() if rows is None else rows)
    return na.render(out, screens, "live", date(2026, 11, 9), date(2026, 11, 28), HEAD, NOW)


def test_render_states_the_scope_notes_and_counts():
    text = _render()
    assert text.startswith("# News age readout (live, event verdicts 2026-11-09 to before "
                           "2026-11-28 ET): screened-in items that reached a verdict only")
    assert "Generated 2026-11-27 10:00 ET. Code version: pinned0." in text
    assert "no screen row (SCREENED_IN or SCREENED_OUT) stores published_at" in text
    assert "news_log.published_at is the time the headline was logged" in text
    assert "there is no relevant / not-relevant split" in text
    assert "- event verdicts: 8; published_at none: 1, unparseable: 1; with a matched screen row: 5" in text
    assert "- verdicts with no matched SCREENED_IN row: 3 (2 with a url, 1 without one)" in text
    assert ("- SCREENED_IN rows in the window: 1058; without a url: 27; with a url and no verdict "
            "on it within 7 days after: 131") in text
    assert "## Age at the screen" in text and "## Age at the verdict" in text
    assert "published more than 5 min after the verdict: 1" in text
    assert "| NVDA | 5 | 5 | 2.00 | " in text
    assert TITLE not in text and "Nvidia" not in text


def test_every_table_row_has_its_header_width():
    for rows in (None, []):
        text = _render(rows=rows)
        width = None
        for line in text.splitlines():
            if line.startswith("| ") and width is None:
                width = line.count("|")
            elif line.startswith("|") and width is not None:
                assert line.count("|") == width, line
            else:
                width = None


def test_render_has_no_guard_or_label_words():
    text = _render().lower()
    for word in ("stale", "too old", "should", "recommend", "late item"):
        assert word not in text


# --- main -------------------------------------------------------------------

class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append((sql, params))

    def fetchall(self):
        return self.conn.rows

    def fetchone(self):
        return self.conn.screens


class _Conn:
    def __init__(self, rows=(), screens=(0, 0, 0)):
        self.log, self.rows, self.screens, self.rolled = [], list(rows), screens, 0

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled += 1


def _fake_db(monkeypatch, conn):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append, __file__="fake")
    monkeypatch.setitem(sys.modules, "db", mod)
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("SUPABASE_HOST", "fake-host")
    monkeypatch.setattr(na, "code_version", lambda: HEAD)
    return put


def test_main_reads_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    conn = _Conn([_row(), _row(2, screen_id=None, screened=None)], (5, 1, 2))
    put = _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert na.main(["--since", "2026-11-20", "--until", "2026-11-28", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2:] == [na._VERDICTS_SQL, na._SCREENS_SQL]
    p = {"env": "live", "start": datetime(2026, 11, 20, tzinfo=NY), "end": datetime(2026, 11, 28, tzinfo=NY)}
    assert conn.log[2][1] == p and conn.log[3][1] == p
    assert conn.rolled == 1 and put == [conn]
    text, data = out.read_text(), json.loads(js.read_text())
    assert "Code version: pinned0." in text and "- event verdicts: 2;" in text
    assert "- SCREENED_IN rows in the window: 5; without a url: 1;" in text
    assert data["screen_rows"] == {"n": 5, "no_url": 1, "url_no_verdict": 2}
    assert data["generated"] == NOW.isoformat() and data["code_version"] == HEAD
    assert set(data["verdicts"][0]) == {"id", "day", "ticker", "feed", "has_url", "held",
                                        "published", "screen_id", "screen_h", "verdict_h"}


def test_main_defaults_the_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert na.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["start"] == datetime(2026, 11, 9, tzinfo=NY)
    assert conn.log[2][1]["end"] == datetime(2026, 11, 28, tzinfo=NY)


def test_the_sql_returns_no_title_url_or_text():
    sql = na._VERDICTS_SQL
    assert "s.provenance->'trigger'->>'path' = 'event'" in sql
    assert "s.decision ? 'model_output'" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    assert "x.signal = 'SCREENED_IN'" in sql
    assert "sc.timestamp <= v.timestamp AND sc.timestamp > v.timestamp - interval '7 days'" in sql
    final = sql.split("\nSELECT v.id")[1]
    assert "substring(v.url from" in final and "v.url IS NOT NULL" in final
    assert final.replace("substring(v.url from", "").replace("v.url IS NOT NULL", "").count("url") == 0
    for both in (sql, na._SCREENS_SQL):
        for col in ("title", "reasoning", "prompt", "text"):
            assert col not in both
    assert na._SCREENS_SQL.split("\nSELECT count(*),")[1].count("url") == 4   # filters only


def test_the_sources_are_trigger_record_and_the_screen_rows():
    an = open(os.path.join(na.ROOT, "agents", "analyst.py")).read()
    assert '"published_at": str(e.get("timestamp") or e.get("published_at") or "") or None,' in an
    assert '"published_at": trigger.get("published_at"), "prior_events": len(prior),' in an
    crew = open(os.path.join(na.ROOT, "agents", "crew.py")).read()
    assert 'db.log_signal(ticker, "SCREENED_IN", 0.0,' in crew
    assert '"url": str(event.get("url") or "")[:300] or None,' in crew
    fields = crew.split("def event_item_fields")[1].split("\ndef ")[0]
    assert "published" not in fields and "timestamp" not in fields
    db_src = open(os.path.join(na.ROOT, "db.py")).read()
    assert "published_at  TIMESTAMPTZ DEFAULT NOW()" in db_src
    assert '"INSERT INTO news_log (ticker, title, url, image_url) VALUES (%s,%s,%s,%s)",' in db_src


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert na.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = na.main(["--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert na.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--since", "2026-11-26", "--until", "2026-11-26"],
                                  ["--since", "11-26"], ["--until", "x"]])
def test_main_rejects_bad_arguments(monkeypatch, argv):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert na.main(argv, now=NOW) == 2 and conn.log == []


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        na.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert na.db_root(str(tmp_path)) == na.ROOT
    (tmp_path / "db.py").write_text("")
    assert na.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "json", "os", "statistics", "subprocess", "sys",
                       "datetime", "email", "zoneinfo"}
    assert "    import db\n" in src and "import agents" not in src


def test_code_version_is_the_short_head(monkeypatch):
    monkeypatch.setattr(na.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="abc1234\n"))
    assert na.code_version() == "abc1234"
    monkeypatch.setattr(na.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert na.code_version() == "unknown"
