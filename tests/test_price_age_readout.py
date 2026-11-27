"""scripts/price_age_readout.py: verdicts whose brief price is older than
their trigger item, counted per day, ticker, path and held x session; the
read is read-only on the pooler port and skips the stack's start."""
import importlib.util
import os
import sys
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "price_age_readout.py")
_spec = importlib.util.spec_from_file_location("price_age_readout", _PATH)
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 26, 22, 0, tzinfo=NY)


def _row(sid, ts, ticker="NVDA", path="event", as_of="2026-11-26T00:59:00+00:00",
         ev="2026-11-26T05:00:19+00:00", built=None, held="true"):
    return (sid, ts, ticker, path, as_of, ev, built, held)


# --- parsing and the gap ----------------------------------------------------

@pytest.mark.parametrize("value, expect", [
    ("2026-11-26T05:00:19+00:00", datetime(2026, 11, 26, 5, 0, 19, tzinfo=UTC)),
    ("2026-11-26T05:00:19Z", datetime(2026, 11, 26, 5, 0, 19, tzinfo=UTC)),
    ("2026-11-26T00:00:19-05:00", datetime(2026, 11, 26, 5, 0, 19, tzinfo=UTC)),
    ("Thu, 26 Nov 2026 05:00:19 GMT", datetime(2026, 11, 26, 5, 0, 19, tzinfo=UTC)),
    ("2026-11-26 05:00:19", datetime(2026, 11, 26, 5, 0, 19, tzinfo=UTC)),   # naive = UTC
])
def test_times_parse_as_crew_does(value, expect):
    assert pa.parse_when(value) == expect


def test_unparseable_and_empty_times_are_none():
    assert pa.parse_when("yesterday") is None
    assert pa.parse_when("") is None and pa.parse_when(None) is None


def test_the_gap_is_item_minus_price_in_minutes():
    # The audit's case: the price is Wed 19:59 ET, the item Thu 00:00:19 ET.
    assert pa.gap_minutes("2026-11-26T00:59:00+00:00", "2026-11-26T05:00:19+00:00") == (241.3, None)
    # Price after the item: negative, not older.
    assert pa.gap_minutes("2026-11-26T15:00:00+00:00", "2026-11-26T14:30:00+00:00") == (-30.0, None)
    assert pa.gap_minutes("2026-11-26T15:00:00+00:00", "2026-11-26T15:00:00+00:00") == (0.0, None)


@pytest.mark.parametrize("as_of, ev, reason", [
    (None, "2026-11-26T05:00:19+00:00", "no price time"),
    (None, None, "no price time"),
    ("2026-11-26T00:59:00+00:00", None, "no item time"),
    ("2026-11-26T00:59:00+00:00", "2026-11-25", "bare date"),
    ("2026-11-26T00:59:00+00:00", " 2026-11-25 ", "bare date"),
    ("2026-11-26T00:59:00+00:00", "last week", "unparsed"),
    ("garbage", "2026-11-26T05:00:19+00:00", "unparsed"),
])
def test_untimed_verdicts_get_a_reason_and_no_gap(as_of, ev, reason):
    assert pa.gap_minutes(as_of, ev) == (None, reason)


# --- session by market_calendar --------------------------------------------

def test_rth_uses_the_market_calendar():
    import market_calendar as mc
    assert date(2026, 11, 26) in mc.NYSE_HOLIDAYS and date(2026, 11, 27) in mc.NYSE_EARLY_CLOSES
    assert pa.in_rth(datetime(2026, 11, 25, 10, 0, tzinfo=NY)) is True      # Wed session
    assert pa.in_rth(datetime(2026, 11, 25, 9, 29, tzinfo=NY)) is False     # before the open
    assert pa.in_rth(datetime(2026, 11, 25, 16, 0, tzinfo=NY)) is False     # at the close
    assert pa.in_rth(datetime(2026, 11, 26, 11, 0, tzinfo=NY)) is False     # Thanksgiving
    assert pa.in_rth(datetime(2026, 11, 27, 12, 59, tzinfo=NY)) is True     # half day, before 13:00
    assert pa.in_rth(datetime(2026, 11, 27, 13, 0, tzinfo=NY)) is False     # half day, after
    assert pa.in_rth(datetime(2026, 11, 28, 11, 0, tzinfo=NY)) is False     # Saturday


def test_rth_reads_a_utc_time_in_new_york():
    assert pa.in_rth(datetime(2026, 11, 25, 15, 0, tzinfo=UTC)) is True     # 10:00 ET
    assert pa.in_rth(datetime(2026, 11, 25, 14, 0, tzinfo=UTC)) is False    # 09:00 ET


def test_session_is_taken_from_the_brief_build_time_else_the_row_time():
    ts = datetime(2026, 11, 25, 22, 0, tzinfo=UTC)                          # 17:00 ET
    assert pa.classify(_row(1, ts))["session"] == "outside RTH"
    built = "2026-11-25T15:30:00+00:00"                                     # 10:30 ET
    assert pa.classify(_row(1, ts, built=built))["session"] == "RTH"


# --- classify and tally -----------------------------------------------------

def test_classify_labels_day_held_and_path():
    ts = datetime(2026, 11, 26, 3, 0, tzinfo=UTC)                           # 11-25 22:00 ET
    c = pa.classify(_row(7, ts, path=None, held=None))
    assert c["day"] == "2026-11-25" and c["path"] == "none" and c["held"] == "unknown"
    assert pa.classify(_row(7, ts, held="false"))["held"] == "not held"
    assert pa.classify(_row(7, ts, held="true"))["held"] == "held"


def test_tally_counts_older_not_older_and_the_reasons():
    ts = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
    rows = [pa.classify(r) for r in (
        _row(1, ts),                                                         # +241.3
        _row(2, ts, ev="2026-11-26T01:59:00+00:00"),                         # +60
        _row(3, ts, ev="2026-11-26T00:29:00+00:00"),                         # -30
        _row(4, ts, as_of=None),
        _row(5, ts, ev="2026-11-25"),
        _row(6, ts, ev=None),
        _row(7, ts, ev="soon"))]
    t = pa.tally(rows)
    assert (t["verdicts"], t["timed"], t["older"], t["not_older"]) == (7, 3, 2, 1)
    assert t["older_pct"] == 66.7
    assert t["reasons"] == {"no price time": 1, "no item time": 1, "bare date": 1, "unparsed": 1}
    assert t["gap_timed"] == {"n": 3, "min": -30.0, "p25": 15.0, "median": 60.0, "p75": 150.7,
                              "max": 241.3}
    assert t["gap_older"]["n"] == 2 and t["gap_older"]["median"] == 150.7


def test_a_price_at_the_items_own_time_is_not_older():
    ts = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
    same = "2026-11-26T05:00:19+00:00"
    t = pa.tally([pa.classify(_row(1, ts, as_of=same, ev=same))])
    assert (t["timed"], t["older"], t["not_older"], t["older_pct"]) == (1, 0, 1, 0.0)
    assert t["gap_older"] == {"n": 0}


def test_an_empty_cell_has_no_percent_and_no_distribution():
    t = pa.tally([])
    assert t["older_pct"] is None and t["gap_timed"] == {"n": 0} and t["gap_older"] == {"n": 0}


def test_a_single_timed_row_has_a_flat_distribution():
    ts = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
    d = pa.tally([pa.classify(_row(1, ts))])["gap_timed"]
    assert d == {"n": 1, "min": 241.3, "p25": 241.3, "median": 241.3, "p75": 241.3, "max": 241.3}


def test_build_groups_by_day_ticker_path_and_held_session():
    rows = [pa.classify(r) for r in (
        _row(1, datetime(2026, 11, 25, 15, 0, tzinfo=UTC), ticker="ANET"),
        _row(2, datetime(2026, 11, 26, 15, 0, tzinfo=UTC), held="false", path="daily_scan"))]
    out = pa.build(rows)
    assert list(out["by_day"]) == ["2026-11-25", "2026-11-26"]
    assert list(out["by_ticker"]) == ["ANET", "NVDA"]
    assert set(out["by_path"]) == {"event", "daily_scan"}
    assert set(out["by_held_session"]) == {"held / RTH", "not held / outside RTH"}
    assert out["total"]["verdicts"] == 2


# --- render -----------------------------------------------------------------

def test_render_has_every_section_and_no_threshold_words():
    rows = [pa.classify(_row(1, datetime(2026, 11, 26, 15, 0, tzinfo=UTC)))]
    text = pa.render(pa.build(rows), "live", date(2026, 11, 24), date(2026, 11, 27), "abc", NOW)
    for h in ("## By day", "## Held x session", "## By trigger path", "## By ticker"):
        assert h in text
    assert "- verdicts: 1; timed: 1; older than the item: 1 (100.0% of timed); not older: 0" in text
    assert "| 2026-11-26 | 1 | 1 | 1 | 100.0 | 0 | 0 | 0 | 0 | 0 | 241.3 [241.3, 241.3], 241.3..241.3 | 241.3 [241.3, 241.3], 241.3 |" in text
    body = text.lower().replace("no threshold, exclusion or stale label is applied", "")
    assert "stale" not in body and "threshold" not in body


def test_every_table_row_has_the_header_width():
    rows = [pa.classify(_row(1, datetime(2026, 11, 26, 15, 0, tzinfo=UTC)))]
    text = pa.render(pa.build(rows), "live", date(2026, 11, 24), date(2026, 11, 27), "abc", NOW)
    table = [l for l in text.splitlines() if l.startswith("|")]
    assert len({l.count("|") for l in table}) == 1


# --- the read ---------------------------------------------------------------

class _Cur:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows=()):
        self.log, self.rows, self.rolled = [], list(rows), 0

    def cursor(self):
        return _Cur(self.log, self.rows)

    def rollback(self):
        self.rolled += 1


def _fake_db(monkeypatch, conn):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append, __file__="fake")
    monkeypatch.setitem(sys.modules, "db", mod)
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("SUPABASE_HOST", "fake-host")
    return put


def test_main_reads_once_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    conn = _Conn([_row(1, datetime(2026, 11, 26, 15, 0, tzinfo=UTC))])
    put = _fake_db(monkeypatch, conn)
    out = tmp_path / "r.md"
    assert pa.main(["--since", "2026-11-24", "--until", "2026-11-27", "--out", str(out)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2] == pa._VERDICTS_SQL and len(sqls) == 3
    params = conn.log[2][1]
    assert params["env"] == "live"
    assert params["start"] == datetime(2026, 11, 24, tzinfo=NY)
    assert params["end"] == datetime(2026, 11, 27, tzinfo=NY)
    assert conn.rolled == 1 and put == [conn]
    assert "older than the item: 1" in out.read_text()


def test_the_sql_reads_only_verdicts_and_no_price():
    sql = pa._VERDICTS_SQL
    assert "s.decision ? 'model_output'" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    assert "s.timestamp >= %(start)s AND s.timestamp < %(end)s" in sql
    assert "brief_price'" not in sql and "price_history" not in sql
    for key in ("brief_price_as_of", "event_time", "brief_built_at"):
        assert f"s.provenance->'trigger'->>'{key}'" in sql
    assert "s.provenance->'plan'->>'held'" in sql


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert pa.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = pa.main(["--since", "2026-11-24", "--until", "2026-11-27", "--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert pa.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


def test_main_rejects_an_empty_window(monkeypatch, capsys):
    _fake_db(monkeypatch, _Conn())
    assert pa.main(["--since", "2026-11-26", "--until", "2026-11-26"], now=NOW) == 2


def test_main_defaults_until_to_tomorrow(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert pa.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["end"] == datetime(2026, 11, 27, tzinfo=NY)
    assert conn.log[2][1]["start"] == datetime(2026, 11, 24, tzinfo=NY)


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        pa.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert pa.db_root(str(tmp_path)) == pa.ROOT
    (tmp_path / "db.py").write_text("")
    assert pa.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_market_calendar_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "os", "re", "statistics", "subprocess", "sys", "collections",
                       "datetime", "email", "zoneinfo"}
    assert "    import db\n" in src and "        import market_calendar as mc\n" in src
