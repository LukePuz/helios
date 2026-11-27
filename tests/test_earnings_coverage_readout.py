"""scripts/earnings_coverage_readout.py: per watchlist-or-held ticker, the
next earnings_calendar row, its days ahead and whether the brief's window
shows it, the last past row, and for held names the active plan's
catalyst_date and its gap to the next row; tickers, dates, days, counts and
ids only; the read is read-only on the pooler port and skips the stack's
start."""
import importlib.util
import json
import os
import re
import sys
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "earnings_coverage_readout.py")
_spec = importlib.util.spec_from_file_location("earnings_coverage_readout", _PATH)
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 27, 11, 30, tzinfo=NY)
TODAY = date(2026, 11, 27)

WATCHLIST = """\
### AI Infrastructure — Compute / Accelerators
| Ticker | Company | Role | Tier |
|---|---|---|---|
| NVDA | Nvidia | Compounder | Core |
| ANET | Arista Networks | Compounder | Core |
| MSFT | Microsoft | Compounder | Core |
| DRAM | Roundhill Memory ETF | Accelerator | Watch |
| AMD | AMD | Accelerator | Watch |
| ODDX | Odd Co | Sidekick | Core |
"""


def _cal(tk, d, confirmed=False, refreshed=None):
    return (tk, d, confirmed, refreshed or datetime(2026, 11, 27, 13, 0, tzinfo=UTC))


CAL = [
    _cal("NVDA", date(2026, 11, 19), True),                 # past only
    _cal("ANET", date(2026, 11, 4), True),
    _cal("ANET", date(2027, 2, 17)),                        # 82 days: outside 45
    _cal("MSFT", date(2027, 1, 6), True),                   # 40 days: inside
    _cal("MSFT", date(2027, 1, 28)),                        # a second future row
    _cal("AMD", date(2026, 11, 27), None),                  # today counts as next
    _cal("IONQ", date(2026, 11, 5), True),                  # held, off the watchlist
    _cal("ZZZZ", date(2027, 1, 2)),                         # neither
]
PLANS = [
    (63, "NVDA", date(2027, 1, 30), datetime(2026, 11, 27, 13, 42, tzinfo=UTC)),
    (60, "ANET", date(2027, 2, 28), datetime(2026, 11, 26, 15, 0, tzinfo=UTC)),
    (55, "ANET", date(2026, 12, 31), datetime(2026, 11, 20, 15, 0, tzinfo=UTC)),
]
HELD = {"NVDA", "ANET", "IONQ", "MSFT"}


def _built(cutoff=45):
    watch, _ = ec.parse_watchlist(WATCHLIST)
    return ec.build(watch, HELD, CAL, PLANS, TODAY, cutoff)


def _by(recs):
    return {r["ticker"]: r for r in recs}


def test_parse_watchlist_uses_the_loaders_rows_and_marks_etfs():
    watch, unparsed = ec.parse_watchlist(WATCHLIST)
    assert sorted(watch) == ["AMD", "ANET", "DRAM", "MSFT", "NVDA"]
    assert watch["DRAM"]["etf"] is True and watch["NVDA"]["etf"] is False
    assert unparsed == ["ODDX"]        # "Ticker" is mixed case: not ticker-like


def test_the_row_pattern_is_knowledge_loaders():
    src = open(os.path.join(ec.ROOT, "knowledge_loader.py")).read()
    m = re.search(r'row_match = re\.match\(\s*r"(.+?)",', src, re.S)
    assert m and ec.WATCHLIST_ROW.pattern == m.group(1)
    assert 're.search(r"\\bETF\\b", d["company"])' in src


def test_the_live_watchlist_parses_like_the_loader():
    text = open(os.path.join(ec.ROOT, "knowledge", "watchlist.md"), encoding="utf-8").read()
    watch, _ = ec.parse_watchlist(text)
    import knowledge_loader
    assert set(watch) == set(knowledge_loader.load_watchlist())
    assert {t for t, v in watch.items() if v["etf"]} == knowledge_loader.get_etf_tickers()


def test_read_window_from_the_real_context_engine():
    w = ec.read_window(open(os.path.join(ec.ROOT, "context_engine.py")).read())
    assert w["own_cutoff_days"] == 45 and w["calendar_days_ahead"] == 45
    src = open(os.path.join(ec.ROOT, "context_engine.py")).read().splitlines()
    assert src[w["own_line"] - 1].startswith("def _own_earnings_line(")
    assert src[w["calendar_line"] - 1].startswith("def _catalyst_calendar_section(")


def test_read_window_follows_the_text_and_scopes_by_function():
    src = ("def _own_earnings_line(t):\n    if (row[0] - d).days > 30:\n        return ''\n\n\n"
           "def other():\n    x.days > 99\n    days_ahead=7\n\n\n"
           "def _catalyst_calendar_section(t):\n    \"\"\"Doc.\"\"\"\n"
           "    ev = upcoming_events(ticker=t, days_ahead=60)\n")
    w = ec.read_window(src)
    assert (w["own_cutoff_days"], w["own_line"]) == (30, 1)
    assert (w["calendar_days_ahead"], w["calendar_line"]) == (60, 11)
    assert ec.read_window("def unrelated():\n    pass\n") == {
        "own_cutoff_days": None, "own_line": None,
        "calendar_days_ahead": None, "calendar_line": None}


def test_the_next_row_rule_is_db_get_next_earnings_rows():
    src = open(os.path.join(ec.ROOT, "db.py")).read()
    i = src.index("def get_next_earnings_row(")
    body = src[i:i + 800]
    assert "earnings_date >= CURRENT_DATE" in body and "ORDER BY earnings_date ASC LIMIT 1" in body
    ce = open(os.path.join(ec.ROOT, "context_engine.py")).read()
    assert "row = db.get_next_earnings_row(ticker)" in ce
    tasks = open(os.path.join(ec.ROOT, "tasks.py")).read()
    assert "SELECT positions_json FROM portfolio_snapshots " in tasks   # _held_tickers
    assert "return {t.upper() for t in (row[0] or {})} if row else set()" in tasks


def test_build_per_ticker():
    recs, _ = _built()
    r = _by(recs)
    assert sorted(r) == ["AMD", "ANET", "DRAM", "IONQ", "MSFT", "NVDA"]
    n = r["NVDA"]
    assert (n["future_rows"], n["next_date"], n["next_days_ahead"], n["inside_window"]) == \
        (0, "none", None, "no row")
    assert (n["last_past_date"], n["last_past_days_ago"]) == ("2026-11-19", 8)
    a = r["ANET"]
    assert (a["next_date"], a["next_days_ahead"], a["inside_window"], a["next_confirmed"]) == \
        ("2027-02-17", 82, "no", False)
    assert a["last_past_date"] == "2026-11-04" and a["next_refreshed"] == "2026-11-27"
    m = r["MSFT"]
    assert (m["next_date"], m["next_days_ahead"], m["inside_window"], m["future_rows"]) == \
        ("2027-01-06", 40, "yes", 2)
    d = r["AMD"]
    assert (d["next_date"], d["next_days_ahead"], d["inside_window"], d["next_confirmed"]) == \
        ("2026-11-27", 0, "yes", None)
    assert d["last_past_date"] == "none" and "plan_id" not in d       # not held
    z = r["DRAM"]
    assert z["rows"] == 0 and z["etf"] and z["inside_window"] == "no row"
    i = r["IONQ"]
    assert i["held"] and not i["watchlist"] and i["active_plans"] == 0 and i["plan_id"] is None


def test_build_held_plan_columns():
    r = _by(_built()[0])
    assert (r["NVDA"]["plan_id"], r["NVDA"]["plan_catalyst_date"],
            r["NVDA"]["plan_catalyst_days_ahead"], r["NVDA"]["catalyst_minus_next_days"]) == \
        (63, "2027-01-30", 64, None)
    # The newest active plan is taken; both are counted.
    assert (r["ANET"]["plan_id"], r["ANET"]["active_plans"],
            r["ANET"]["catalyst_minus_next_days"]) == (60, 2, 11)
    assert r["MSFT"]["plan_id"] is None and r["MSFT"]["catalyst_minus_next_days"] is None


def test_build_summaries():
    s = _built()[1]
    assert s["held_without_next_row"] == ["IONQ", "NVDA"]
    assert s["held_next_outside_window"] == ["ANET"]
    assert s["held_without_active_plan"] == ["IONQ", "MSFT"]
    assert s["held_not_on_watchlist"] == ["IONQ"]
    assert s["without_next_row"] == ["DRAM", "IONQ", "NVDA"] and s["no_row_at_all"] == ["DRAM"]
    assert s["several_future_rows"] == ["MSFT"]
    assert s["inside_window"] == {"no row": 3, "no": 1, "yes": 2}
    assert s["next_rows_by_confirmed"] == {"estimated": 1, "confirmed": 1, "null": 1}
    assert s["calendar_tickers_not_watchlist_or_held"] == 1
    assert (s["tickers"], s["watchlist"], s["held"], s["with_next_row"]) == (6, 5, 4, 3)


def test_an_unread_cutoff_is_unknown_not_a_guess():
    r = _by(_built(cutoff=None)[0])
    assert r["MSFT"]["inside_window"] == "unknown" and r["NVDA"]["inside_window"] == "no row"


def test_the_cutoff_is_inclusive_like_the_brief():
    cal = [_cal("NVDA", date(2027, 1, 11)), _cal("ANET", date(2027, 1, 12))]   # 45, 46
    r = _by(ec.build({"NVDA": {}, "ANET": {}}, set(), cal, [], TODAY, 45)[0])
    assert r["NVDA"]["inside_window"] == "yes" and r["ANET"]["inside_window"] == "no"


def test_render_lists_the_summaries_and_tables():
    recs, summ = _built()
    w = ec.read_window(open(os.path.join(ec.ROOT, "context_engine.py")).read())
    text = ec.render(recs, summ, w, "live", TODAY, date(2026, 11, 27),
                     datetime(2026, 11, 27, 16, 25, tzinfo=UTC), "headfix", ["ODDX"])
    assert "code headfix." in text and ec.NOTE in text
    assert "days ahead <= 45" in text and "Postgres CURRENT_DATE 2026-11-27" in text
    assert "- Held without a next row: 2 (IONQ, NVDA)." in text
    assert "- Held whose next row is outside the window: 1 (ANET)." in text
    assert "confirmed 1, estimated 1, null 1." in text
    assert "| NVDA | none | none | no row | 2026-11-19 | 8 | 63 | 1 | 2027-01-30 | 64 | no row |" in text
    assert "| ANET | 2027-02-17 | 82 | no | 2026-11-04 | 23 | 60 | 2 | 2027-02-28 | 93 | 11 |" in text
    assert "not parsed: 1 (ODDX)" in text


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append((sql, params))

    def fetchone(self):
        sql = self.conn.log[-1][0]
        return (self.conn.db_today,) if sql == ec._TODAY_SQL else self.conn.snap

    def fetchall(self):
        sql = self.conn.log[-1][0]
        return self.conn.cal if sql == ec._CAL_SQL else self.conn.plans


class _Conn:
    def __init__(self, cal=(), plans=(), snap=None, db_today=TODAY):
        self.log, self.cal, self.plans, self.snap = [], list(cal), list(plans), snap
        self.db_today, self.rolled = db_today, 0

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled += 1


def _fake_db(monkeypatch, conn, tmp_path):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append, __file__="fake")
    monkeypatch.setitem(sys.modules, "db", mod)
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("SUPABASE_HOST", "fake-host")
    monkeypatch.setattr(ec.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="headfix\n"))
    (tmp_path / "knowledge").mkdir(exist_ok=True)
    (tmp_path / "knowledge" / "watchlist.md").write_text(WATCHLIST)
    (tmp_path / "context_engine.py").write_text(
        open(os.path.join(ec.ROOT, "context_engine.py")).read())
    monkeypatch.setattr(ec, "db_root", lambda cwd=None: str(tmp_path))
    return put


def test_main_reads_read_only_and_rolls_back(monkeypatch, tmp_path):
    snap = (datetime(2026, 11, 27, 16, 25, tzinfo=UTC), ["nvda", "ANET", "IONQ", "MSFT"])
    conn = _Conn(CAL, PLANS, snap)
    put = _fake_db(monkeypatch, conn, tmp_path)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert ec.main(["--out", str(out), "--json", str(js)], now=NOW) == 0
    assert [s for s, _ in conn.log] == ["SET TRANSACTION READ ONLY",
                                        "SET LOCAL statement_timeout = %s", ec._TODAY_SQL,
                                        ec._HELD_SQL, ec._CAL_SQL, ec._PLANS_SQL]
    assert conn.log[1][1] == (60000,)
    assert conn.log[3][1] == {"env": "live"} and conn.log[5][1] == {"env": "live"}
    assert conn.rolled == 1 and put == [conn]
    text = out.read_text()
    assert "code headfix." in text and "- Held without a next row: 2 (IONQ, NVDA)." in text
    d = json.loads(js.read_text())
    assert d["code_version"] == "headfix" and d["today"] == "2026-11-27"
    assert d["summary"]["held_next_outside_window"] == ["ANET"]
    assert d["window"]["own_cutoff_days"] == 45 and d["watchlist_unparsed"] == ["ODDX"]


def test_main_without_a_snapshot_reads_no_held(monkeypatch, tmp_path):
    conn = _Conn(CAL, PLANS, None)
    _fake_db(monkeypatch, conn, tmp_path)
    out = tmp_path / "r.md"
    assert ec.main(["--out", str(out), "--env", "paper"], now=NOW) == 0
    text = out.read_text()
    assert "none (no snapshot)" in text and "| none | | | | | | | | | | |" in text
    assert conn.log[3][1] == {"env": "paper"}


def test_the_evening_uses_the_new_york_date(monkeypatch, tmp_path):
    conn = _Conn(CAL, PLANS, None, db_today=date(2026, 11, 28))
    _fake_db(monkeypatch, conn, tmp_path)
    out = tmp_path / "r.md"
    assert ec.main(["--out", str(out)], now=datetime(2026, 11, 27, 20, 30, tzinfo=NY)) == 0
    text = out.read_text()
    assert "today 2026-11-27 (New York), Postgres CURRENT_DATE 2026-11-28" in text
    assert "| AMD | yes | no | no | 1 | 1 | 2026-11-27 | 0 |" in text


def test_the_sql_selects_no_amount():
    assert "jsonb_object_keys(positions_json)" in ec._HELD_SQL
    assert "trading_env = %(env)s" in ec._HELD_SQL
    assert "status = 'active' AND (trading_env = %(env)s OR trading_env IS NULL)" in ec._PLANS_SQL
    for sql in (ec._HELD_SQL, ec._CAL_SQL, ec._PLANS_SQL):
        sel = sql.upper().split("FROM")[0]
        for col in ("TOTAL_VALUE", "CASH", "PRICE", "T1", "STOP", "TARGET", "EVENT", "THESIS"):
            assert col not in sel
        assert not any(w in sql.upper() for w in ("INSERT", "UPDATE ", "DELETE", "ALTER"))
    assert "SELECT timestamp,\n       CASE" in ec._HELD_SQL      # no positions_json values


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert ec.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, tmp_path, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn, tmp_path)
    rc = ec.main(["--out", os.devnull], now=datetime(2026, 11, 27, *hm, tzinfo=NY))
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys, tmp_path):
    conn = _Conn()
    _fake_db(monkeypatch, conn, tmp_path)
    monkeypatch.delenv("SUPABASE_HOST")
    assert ec.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


def test_main_stops_without_the_files(monkeypatch, capsys, tmp_path):
    conn = _Conn()
    _fake_db(monkeypatch, conn, tmp_path)
    (tmp_path / "knowledge" / "watchlist.md").unlink()
    assert ec.main([], now=NOW) == 2 and conn.log == []
    assert "watchlist" in capsys.readouterr().err


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch, tmp_path):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn, tmp_path)
    with pytest.raises(RuntimeError):
        ec.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert ec.db_root(str(tmp_path)) == ec.ROOT
    (tmp_path / "db.py").write_text("")
    assert ec.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "json", "os", "re", "subprocess", "sys", "collections",
                       "datetime", "zoneinfo"}
    assert "    import db\n" in src and "agents" not in imports
    assert "earnings_coverage_readout" not in open(os.path.join(ec.ROOT, "weekly_readouts.py")).read()
