"""scripts/add_trigger_readout.py: OPEN/ADD verdicts against their plan's
add_trigger (add_under_hold_plan, the trigger side and the
price_vs_add_trigger_pct distribution, relation counts) per day, ticker and
day x ticker, percentages only; the FOLLOW line points at classify's rule;
the read is read-only on the pooler port and skips the stack's start."""
import importlib.util
import json
import os
import sys
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "add_trigger_readout.py")
_spec = importlib.util.spec_from_file_location("add_trigger_readout", _PATH)
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 27, 9, 30, tzinfo=NY)
HEAD = "pinned0"
TS = datetime(2026, 11, 24, 15, 0, tzinfo=UTC)


def _row(sid=1, ts=TS, sym="NVDA", action="ADD", rel="FOLLOW", says="hold", held="true",
         has_rec=True, under="true", trig="number", pct="-26.4"):
    return (sid, ts, sym, action, rel, says, held, has_rec, under, trig, pct)


# --- classify ---------------------------------------------------------------

def test_classify_reads_the_keys_and_the_side():
    r = at.classify(_row())
    assert r == {"id": 1, "day": "2026-11-24", "ticker": "NVDA", "action": "ADD",
                 "relation": "FOLLOW", "plan_says": "hold", "held": True, "add_record": True,
                 "add_under_hold_plan": True, "side": "below", "pct": -26.4}


@pytest.mark.parametrize("kw, side, pct", [
    ({"pct": "0"}, "at or above", 0.0), ({"pct": "3.2"}, "at or above", 3.2),
    ({"pct": "-0.01"}, "below", -0.01),
    ({"trig": "null", "pct": None}, "null trigger", None),
    ({"trig": None, "pct": None}, "null trigger", None),
    ({"pct": None}, "no pct", None), ({"pct": "NaN"}, "no pct", None),
    ({"has_rec": False, "under": None, "trig": None, "pct": None}, "no add record", None),
])
def test_trigger_sides(kw, side, pct):
    r = at.classify(_row(**kw))
    assert r["side"] == side and r["pct"] == pct


def test_no_record_has_no_under_hold_value_and_missing_relation_is_none():
    r = at.classify(_row(has_rec=False, rel=None, under="true"))
    assert r["add_under_hold_plan"] is None and r["relation"] == "none" and not r["add_record"]


def test_the_day_is_new_york_and_a_naive_timestamp_is_utc():
    assert at.classify(_row(ts=datetime(2026, 11, 25, 3, 0, tzinfo=UTC)))["day"] == "2026-11-24"
    assert at.classify(_row(ts=datetime(2026, 11, 25, 3, 0)))["day"] == "2026-11-24"


@pytest.mark.parametrize("v, out", [("true", True), ("false", False), (None, None), (True, True),
                                    ("x", None)])
def test_bool_text(v, out):
    assert at._bool(v) is out


# --- tally and build --------------------------------------------------------

def _rows():
    d1 = datetime(2026, 11, 23, 15, 0, tzinfo=UTC)
    d2 = datetime(2026, 11, 24, 15, 0, tzinfo=UTC)
    raw = [_row(1, d1, pct="-10"), _row(2, d1, rel="REVISE", pct="3.2"),
           _row(3, d2, pct="-30"), _row(4, d2, rel="REVISE", pct="-20"),
           _row(5, d2, sym="AMD", rel="DIVERGE", held="false", under="false", says="buy", pct="-5"),
           _row(6, d2, sym="AMD", rel="FOLLOW", held="false", under="false", says="buy",
                trig="null", pct=None),
           _row(7, d2, sym="ANET", action="OPEN", rel="NEW_PLAN", has_rec=False, under=None,
                trig=None, pct=None),
           _row(8, d2, sym="ANET", action="OPEN", rel="FOLLOW", held="false", under="false",
                says="buy", pct="-1")]
    return [at.classify(r) for r in raw]


def test_tally_counts_sides_distribution_and_relations():
    t = at.tally([r for r in _rows() if r["action"] == "ADD"])
    assert t["verdicts"] == 6 and t["under_hold"] == 4
    assert t["sides"] == {"below": 4, "at or above": 1, "null trigger": 1, "no pct": 0,
                          "no add record": 0}
    assert t["pct"] == {"n": 5, "median": -10.0, "min": -30.0, "max": 3.2}
    assert t["relations"] == {"FOLLOW": 3, "REVISE": 2, "DIVERGE": 1}
    assert t["follow_held_below"] == 2      # ids 1 and 3; id 6 is not held, no trigger


def test_follow_below_needs_follow_add_held_and_below():
    base = at.classify(_row())
    assert at.follow_below(base)
    for kw in ({"rel": "REVISE"}, {"action": "OPEN"}, {"held": "false"}, {"held": None},
               {"pct": "0"}, {"trig": "null", "pct": None}):
        assert not at.follow_below(at.classify(_row(**kw))), kw


def test_an_empty_tally():
    t = at.tally([])
    assert t["verdicts"] == 0 and t["pct"] == {"n": 0, "median": None, "min": None, "max": None}


def test_build_splits_actions_and_keys_the_tables():
    out = at.build(_rows(), "ADD")
    assert out["by_action"]["ADD"]["verdicts"] == 6 and out["by_action"]["OPEN"]["verdicts"] == 2
    assert out["by_action"]["OPEN"]["sides"]["no add record"] == 1
    assert list(out["by_day"]) == ["2026-11-23", "2026-11-24"]
    assert out["by_day"]["2026-11-23"]["pct"]["median"] == -3.4
    assert list(out["by_ticker"]) == ["AMD", "NVDA"]
    assert list(out["by_day_ticker"]) == [("2026-11-23", "NVDA"), ("2026-11-24", "AMD"),
                                          ("2026-11-24", "NVDA")]
    assert out["relations"] == ["FOLLOW", "REVISE", "DIVERGE", "NEW_PLAN"]
    assert list(at.build(_rows(), "OPEN")["by_ticker"]) == ["ANET"]


# --- the FOLLOW rule --------------------------------------------------------

def test_follow_line_is_classifys_held_add_rule():
    n = at.follow_line_no()
    lines = open(os.path.join(at.ROOT, "agents", "plans.py")).read().splitlines()
    assert n is not None and "adding within plan" in lines[n - 1]
    assert 'says == _HOLD and out["verdict_dir"] == _BUY and held' in lines[n - 1]
    before = "\n".join(lines[:n])
    assert before.rfind("\ndef ") == before.rfind("\ndef classify(")   # inside classify


def test_follow_line_missing_file_or_mark(tmp_path):
    assert at.follow_line_no(str(tmp_path / "none.py")) is None
    p = tmp_path / "plans.py"
    p.write_text("x = 1\n# adding within plan, but not the rule\n")
    assert at.follow_line_no(str(p)) is None


def test_the_record_keys_are_add_plan_records():
    src = open(os.path.join(at.ROOT, "agents", "plans.py")).read()
    assert 'return {"add_under_hold_plan": action == "ADD" and says == _HOLD,' in src
    assert '"add_trigger": trig,' in src
    assert '"price_vs_add_trigger_pct": (round((price - trig) / trig * 100, 2)' in src
    assert 'out.update(_add_plan_record(plan, action, says, price))' in src


# --- ADD share cells --------------------------------------------------------

def _srow(sid=1, ts=TS, sym="NVDA", sig="ADD", model=None, conv="7", udr="3.36",
          at_gate="true", review="false"):
    return (sid, ts, sym, sig, model or sig, conv, udr, at_gate, review)


HOL, EARLY = {date(2026, 11, 26)}, {date(2026, 11, 27)}


@pytest.mark.parametrize("hm, d, out", [
    ((9, 29), 24, "pre-market"), ((9, 30), 24, "RTH"), ((15, 59), 24, "RTH"),
    ((16, 0), 24, "after-hours"), ((4, 0), 24, "pre-market"), ((20, 0), 24, "after-hours"),
    ((12, 59), 27, "RTH"), ((13, 0), 27, "after-hours"), ((10, 0), 26, "non-trading day"),
    ((10, 0), 28, "non-trading day"), ((10, 0), 29, "non-trading day"),
])
def test_session_by_new_york_clock_and_calendar(hm, d, out):
    ts = datetime(2026, 11, d, *hm, tzinfo=NY).astimezone(UTC)
    assert at.session_of(ts, HOL, EARLY) == out


def test_the_repo_calendar_has_the_early_close_and_the_holiday():
    mc = at.calendar()
    assert date(2026, 11, 27) in mc.NYSE_EARLY_CLOSES and date(2026, 11, 26) in mc.NYSE_HOLIDAYS
    ts = datetime(2026, 11, 27, 13, 30, tzinfo=NY)
    assert at.session_of(ts, mc.NYSE_HOLIDAYS, mc.NYSE_EARLY_CLOSES) == "after-hours"


@pytest.mark.parametrize("v, band", [("1.99", "<2"), ("2", "2-3"), ("2.99", "2-3"), ("3", "3-4"),
                                     ("4", ">=4"), ("12.5", ">=4"), ("-1", "<2"), (None, "missing"),
                                     ("NaN", "missing"), ("x", "missing")])
def test_udr_bands_include_their_lower_edge(v, band):
    assert at.udr_band(v) == band


@pytest.mark.parametrize("v, out", [("7", 7), ("7.0", 7), ("6.5", 6.5), (None, "missing"),
                                    ("", "missing"), ("high", "high")])
def test_conviction_values(v, out):
    assert at.conviction_of(v) == out


@pytest.mark.parametrize("g, r, out", [("true", "false", "held"), (None, "true", "held"),
                                       ("false", "true", "held"), ("false", "false", "not held"),
                                       ("false", None, "not held"), (None, "false", "held unknown"),
                                       (None, None, "held unknown")])
def test_held_state(g, r, out):
    assert at.held_state(g, r) == out


def test_share_record_fields_and_naive_utc():
    r = at.share_record(_srow(ts=datetime(2026, 11, 27, 12, 16)), HOL, EARLY)
    assert r == {"id": 1, "day": "2026-11-27", "ticker": "NVDA", "signal": "ADD",
                 "model_action": "ADD", "conviction": 7, "udr_band": "3-4",
                 "session": "pre-market", "held": "held"}


def _share_recs():
    pm = datetime(2026, 11, 24, 13, 0, tzinfo=UTC)    # 08:00 ET
    rth = datetime(2026, 11, 24, 16, 0, tzinfo=UTC)   # 11:00 ET
    raw = [_srow(1, pm), _srow(2, pm, sig="HOLD"), _srow(3, rth, sig="HOLD", udr="1.5"),
           _srow(4, rth, sig="HOLD", conv="6", udr=None),
           _srow(5, rth, sig="WATCH", model="ADD"),
           _srow(6, rth, sym="IONQ", sig="ADD", conv="8", udr="4.2", at_gate=None, review="true"),
           _srow(7, rth, sym="AMD", sig="OPEN", model="ADD", at_gate="false"),
           _srow(8, rth, sym="AMD", sig="HOLD", at_gate=None, review="false"),
           _srow(9, rth, sym="MU", sig="HOLD", at_gate="false")]
    return [at.share_record(r, HOL, EARLY) for r in raw]


def test_share_build_by_signal():
    sh = at.share_build(_share_recs(), "signal")
    assert sh["all"] == {"add": 2, "n": 5}
    assert sh["excluded"] == {"held unknown": 1, "not held": 1}
    assert list(sh["grid"]) == [6, 7, 8]
    assert sh["grid"][7]["3-4"] == {"add": 1, "n": 2} and sh["grid"][7]["<2"] == {"add": 0, "n": 1}
    assert sh["grid"][6]["missing"] == {"add": 0, "n": 1}
    assert sh["by_udr"][">=4"] == {"add": 1, "n": 1} and sh["by_udr"]["2-3"] == {"add": 0, "n": 0}
    assert sh["by_ticker"] == {"IONQ": {"add": 1, "n": 1}, "NVDA": {"add": 1, "n": 4}}
    assert sh["by_session"]["pre-market"] == {"add": 1, "n": 2}
    assert sh["by_session"]["RTH"] == {"add": 1, "n": 3}
    assert sh["by_session"]["non-trading day"] == {"add": 0, "n": 0}
    assert sh["pairs"] == {"ADD -> WATCH": 1}


def test_share_build_by_model_takes_the_downgraded_add_and_not_the_unheld_open():
    sh = at.share_build(_share_recs(), "model")
    assert sh["field"] == "model_action" and sh["all"] == {"add": 3, "n": 6}
    assert sh["by_ticker"]["NVDA"] == {"add": 2, "n": 5} and "AMD" not in sh["by_ticker"]


def test_share_build_empty():
    sh = at.share_build([], "signal")
    assert sh["all"] == {"add": 0, "n": 0} and sh["grid"] == {} and sh["pairs"] == {}
    text = "\n".join(at.share_render(sh))
    assert "| all | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |" in text and "| none | | | |" in text
    assert "| RTH | 0 | 0 | n/a |" in text


def test_share_render_cells():
    text = "\n".join(at.share_render(at.share_build(_share_recs(), "signal")))
    assert "## ADD share cells (held names, signal ADD or HOLD)" in text
    assert "- held verdicts: 2/5 (40.0%); left out (not held / held unknown): 1 / 1" in text
    assert "(model -> signal): ADD -> WATCH 1" in text
    assert "| 7 | 0/1 (0.0%) | 0/0 | 1/2 (50.0%) | 0/0 | 0/0 | 1/3 (33.3%) |" in text
    assert "| all | 0/1 (0.0%) | 0/0 | 1/2 (50.0%) | 1/1 (100.0%) | 0/1 (0.0%) | 2/5 (40.0%) |" in text
    assert "| NVDA | 1 | 4 | 25.0% |" in text and "| pre-market | 1 | 2 | 50.0% |" in text


def test_share_sql_reads_no_prices_or_text():
    sql = at._SHARE_SQL
    assert "s.decision ? 'model_output'" in sql
    assert "(s.signal IN ('ADD', 'HOLD') OR s.decision->>'model_action' IN ('ADD', 'HOLD'))" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    for key in ("price", "prompt", "qty", "reasoning", "headline", "model_output'->"):
        assert key not in sql


# --- render -----------------------------------------------------------------

def _render(rows=None, action="ADD", line_no=336, share=None):
    out = at.build(_rows() if rows is None else rows, action)
    return at.render(out, "live", date(2026, 11, 23), date(2026, 11, 28), HEAD, NOW, line_no,
                     share)


def test_render_has_the_follow_line_totals_and_tables():
    text = _render()
    assert text.startswith("# Add trigger readout (live, OPEN/ADD verdicts 2026-11-23 to before "
                           "2026-11-28 ET)")
    assert "Generated 2026-11-27 09:30 ET. Code version: pinned0." in text
    assert ("- FOLLOW includes held ADDs below the trigger: classify (agents/plans.py:336, "
            "\"adding within plan\")") in text
    assert "Here 2 of 3 FOLLOW ADD verdicts were held and below the trigger." in text
    assert ("- ADD verdicts: 6; add_under_hold_plan true: 4; below: 4; at or above: 1; null "
            "trigger: 1; no pct: 0; no add record: 0; pct median -10.0%, min -30.0%, max +3.2% "
            "(n 5)") in text
    assert "- relations (ADD): FOLLOW 3, REVISE 2, DIVERGE 1" in text
    for title in ("## Totals by action", "## ADD verdicts by day", "## ADD verdicts by ticker",
                  "## ADD verdicts by day and ticker"):
        assert title in text
    assert "| 2026-11-24 | NVDA | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 2 | -25.0% | -30.0% | -20.0% | 1 | 1 | 0 | 0 |" in text


def test_render_without_the_line_or_rows():
    text = _render(rows=[], line_no=None)
    assert "agents/plans.py classify (line not found)" in text
    assert "- ADD verdicts: 0" in text and "Here 0 of 0 FOLLOW" in text
    assert "| none |" in text


def test_every_table_row_has_its_header_width():
    for rows, recs in ((None, None), ([], None), (None, _share_recs()), ([], [])):
        share = None if recs is None else at.share_build(recs, "signal")
        text = _render(rows=rows, share=share)
        width = None
        for line in text.splitlines():
            if line.startswith("| ") and width is None:
                width = line.count("|")
            elif line.startswith("|") and width is not None:
                assert line.count("|") == width, line
            else:
                width = None


def test_render_has_no_label_or_verdict_words():
    text = _render(share=at.share_build(_share_recs(), "signal")).lower()
    for word in ("should", "recommend", "violat", "wrong", "contradict"):
        assert word not in text


# --- main -------------------------------------------------------------------

class _Cur:
    def __init__(self, log, rows, share_rows):
        self.log, self.rows, self.share_rows = log, rows, share_rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchall(self):
        return self.share_rows if self.log[-1][0] == at._SHARE_SQL else self.rows


class _Conn:
    def __init__(self, rows=(), share_rows=()):
        self.log, self.rows, self.share_rows, self.rolled = [], list(rows), list(share_rows), 0

    def cursor(self):
        return _Cur(self.log, self.rows, self.share_rows)

    def rollback(self):
        self.rolled += 1


def _fake_db(monkeypatch, conn):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append, __file__="fake")
    monkeypatch.setitem(sys.modules, "db", mod)
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("SUPABASE_HOST", "fake-host")
    monkeypatch.setattr(at, "code_version", lambda: HEAD)
    return put


def test_main_reads_once_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    conn = _Conn([_row(), _row(2, pct="3.2", rel="REVISE")], [_srow(), _srow(2, sig="HOLD")])
    put = _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert at.main(["--since", "2026-11-20", "--until", "2026-11-28", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2] == at._VERDICTS_SQL and sqls[3] == at._SHARE_SQL and len(sqls) == 4
    assert conn.log[2][1] == conn.log[3][1] == {
        "env": "live", "start": datetime(2026, 11, 20, tzinfo=NY),
        "end": datetime(2026, 11, 28, tzinfo=NY)}
    assert conn.rolled == 1 and put == [conn]
    text, data = out.read_text(), json.loads(js.read_text())
    assert "Code version: pinned0." in text and "- ADD verdicts: 2;" in text
    assert data["code_version"] == HEAD and [v["pct"] for v in data["verdicts"]] == [-26.4, 3.2]
    assert data["generated"] == NOW.isoformat()
    assert "- held verdicts: 1/2 (50.0%);" in text and data["share_basis"] == "signal"
    assert [v["signal"] for v in data["share_verdicts"]] == ["ADD", "HOLD"]


def test_main_share_basis_model(monkeypatch, tmp_path):
    conn = _Conn([], [_srow(sig="WATCH", model="ADD"), _srow(2, sig="HOLD", model="HOLD")])
    _fake_db(monkeypatch, conn)
    out = tmp_path / "r.md"
    assert at.main(["--share-basis", "model", "--out", str(out)], now=NOW) == 0
    text = out.read_text()
    assert "## ADD share cells (held names, model_action ADD or HOLD)" in text
    assert "- held verdicts: 1/2 (50.0%);" in text and "ADD -> WATCH 1" in text


def test_main_defaults_the_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert at.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["start"] == datetime(2026, 11, 23, tzinfo=NY)
    assert conn.log[2][1]["end"] == datetime(2026, 11, 28, tzinfo=NY)


def test_the_sql_reads_the_keys_not_the_prices():
    sql = at._VERDICTS_SQL
    assert "s.decision ? 'model_output' AND s.signal IN ('OPEN', 'ADD')" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    assert "s.timestamp >= %(start)s AND s.timestamp < %(end)s" in sql
    assert "jsonb_typeof(s.provenance->'plan'->'add_trigger')" in sql
    rest = sql.replace("jsonb_typeof(s.provenance->'plan'->'add_trigger')", "")
    assert "add_trigger'" not in rest           # the trigger's value is never read
    for key in ("'price'", "entry_price", "prompt", "qty"):
        assert key not in sql


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert at.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = at.main(["--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert at.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--since", "2026-11-26", "--until", "2026-11-26"],
                                  ["--since", "11-26"], ["--until", "x"]])
def test_main_rejects_bad_arguments(monkeypatch, argv):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert at.main(argv, now=NOW) == 2 and conn.log == []


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        at.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert at.db_root(str(tmp_path)) == at.ROOT
    (tmp_path / "db.py").write_text("")
    assert at.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "importlib", "json", "math", "os", "statistics", "subprocess",
                       "sys", "collections", "datetime", "zoneinfo"}
    assert "    import db\n" in src and "import agents" not in src


def test_code_version_is_the_short_head(monkeypatch):
    monkeypatch.setattr(at.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="abc1234\n"))
    assert at.code_version() == "abc1234"
    monkeypatch.setattr(at.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert at.code_version() == "unknown"
