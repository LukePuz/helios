"""scripts/plan_revision_readout.py: plan creation and revision per ticker
and day from trade_plans and signals.provenance.plan; relation and
revise_triggers counts, lifetimes, horizon spans, catalyst bins and the
level moves between revisions in percent and ATRs, no prices; the read is
read-only on the pooler port and skips the stack's start."""
import importlib.util
import json
import os
import sys
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "plan_revision_readout.py")
_spec = importlib.util.spec_from_file_location("plan_revision_readout", _PATH)
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 27, 9, 30, tzinfo=NY)
# Distinctive made-up levels, so a leak into the output is detectable.
T1, ADD, STOP = 123.45, 131.17, 98.76
T1B, ADDB, STOPB = 135.795, 131.17, 93.822   # +10%, 0%, -5%
ATR = 2.469


def _ts(d, h=15):
    return datetime(2026, 11, d, h, 0, tzinfo=UTC)


def _plan(pid, tk="ANET", status="active", created=None, closed=None, revision=0,
          replaces=None, cat=None, horizon=None, t1=T1, add=ADD, stop=STOP, in_window=True,
          source="analyst", entry=None, lsrc=None, built=None):
    return (pid, tk, status, source, created or _ts(20), closed, revision, replaces, cat,
            horizon, t1, add, stop, entry, lsrc, built, in_window)


def _verdict(sid, rel, ts=None, sym="ANET", trig=None, pair=None, new_id=None, atr=None):
    return (sid, ts or _ts(20), sym, rel, trig, pair,
            str(new_id) if new_id is not None else None, str(atr) if atr is not None else None)


def _raw():
    plans = [
        _plan(1, status="superseded", created=_ts(19), closed=_ts(20), cat=date(2026, 11, 18),
              horizon=date(2027, 2, 17), in_window=False),
        # Fri 11-20: catalyst the same ISO week (Sun 11-22), then a later week.
        _plan(2, status="superseded", created=_ts(20), closed=_ts(21, 21), revision=1, replaces=1,
              cat=date(2026, 11, 22), horizon=date(2026, 11, 27), t1=T1B, add=ADDB, stop=STOPB),
        _plan(3, created=_ts(21, 21), revision=2, replaces=2, cat=date(2026, 12, 31),
              horizon=date(2027, 1, 5), t1=None, add=ADDB, stop=STOPB),
        _plan(4, tk="NVDA", created=_ts(21), cat=None, horizon=date(2027, 2, 19),
              t1=None, add=None, stop=None, source="held_seed"),
    ]
    verdicts = [
        _verdict(10, "REVISE", trig=["target", "catalyst_date"], pair=False, new_id=2, atr=ATR),
        _verdict(11, "FOLLOW"),
        _verdict(12, "REVISE", ts=_ts(21, 21), trig=json.dumps(["catalyst_date"]), pair=True,
                 new_id=3, atr=None),
        _verdict(13, "SEEDED", ts=_ts(21), sym="NVDA"),
        _verdict(14, "REVISE", ts=_ts(21), sym="NVDA"),          # no trigger keys
        _verdict(15, "SHADOW_ODD", ts=_ts(21), sym="NVDA"),
    ]
    return plans, verdicts


@pytest.mark.parametrize("cat, created, want", [
    (date(2026, 11, 19), date(2026, 11, 20), "past"),
    (date(2026, 11, 20), date(2026, 11, 20), "same week"),
    (date(2026, 11, 22), date(2026, 11, 20), "same week"),     # Sunday, the same ISO week
    (date(2026, 11, 23), date(2026, 11, 20), "later"),         # the next Monday
    (None, date(2026, 11, 20), "none"),
    (date(2026, 11, 20), None, "none"),
])
def test_catalyst_bin(cat, created, want):
    assert pr.catalyst_bin(cat, created) == want


@pytest.mark.parametrize("old, new, atr, want", [
    (100.0, 110.0, 2.0, (10.0, 5.0)),
    (100.0, 95.0, None, (-5.0, None)),
    (100.0, 95.0, 0, (-5.0, None)),
    (None, 95.0, 2.0, (None, None)),
    (0, 95.0, 2.0, (None, None)),
    (100.0, float("nan"), 2.0, (None, None)),
    ("100", "101", "2", (1.0, 0.5)),
])
def test_move(old, new, atr, want):
    assert pr.move(old, new, atr) == want


def _build():
    plans_raw, verdicts_raw = _raw()
    plans, levels = pr.build_plans(plans_raw)
    verdicts = pr.build_verdicts(verdicts_raw)
    return plans, verdicts, pr.build_moves(plans, levels, verdicts)


def test_build_plans_keeps_no_levels():
    plans, _, _ = _build()
    assert [p["plan_id"] for p in plans] == [2, 3, 4]
    p2 = plans[0]
    assert p2["day"] == "2026-11-20" and p2["lifetime_h"] == 30.0 and p2["horizon_days"] == 7
    assert p2["catalyst"] == "same week" and plans[1]["catalyst"] == "later"
    assert plans[2]["catalyst"] == "none" and plans[1]["lifetime_h"] is None
    blob = json.dumps(plans)
    for v in ("123.45", "131.17", "98.76", "135.79", "93.82"):
        assert v not in blob


def test_build_moves_in_percent_and_atr():
    _, _, moves = _build()
    m2, m3 = moves
    assert m2["plan_id"] == 2 and m2["replaces_id"] == 1 and m2["prior_found"]
    assert m2["t1_pct"] == 10.0 and m2["add_trigger_pct"] == 0.0 and m2["structural_stop_pct"] == -5.0
    assert m2["t1_atr"] == round((T1B - T1) / ATR, 3) and m2["add_trigger_atr"] == 0.0
    assert m3["t1_pct"] is None and m3["structural_stop_pct"] == 0.0
    assert m3["structural_stop_atr"] is None and not m3["atr_recorded"]


def test_a_revision_whose_prior_is_missing_gives_none():
    plans, levels = pr.build_plans([_plan(9, replaces=99)])
    m = pr.build_moves(plans, levels, [])
    assert m[0]["prior_found"] is False and m[0]["t1_pct"] is None


def test_build_verdicts_reads_trigger_lists_and_strings():
    _, verdicts, _ = _build()
    assert verdicts[0]["triggers"] == ["target", "catalyst_date"]
    assert verdicts[2]["triggers"] == ["catalyst_date"] and verdicts[2]["pair"] is True
    assert verdicts[4]["triggers"] is None and verdicts[4]["pair"] is None
    assert verdicts[0]["new_plan_id"] == 2 and verdicts[0]["ladder_atr"] == ATR


def test_render_counts_without_levels():
    plans, verdicts, moves = _build()
    text = pr.render(plans, verdicts, moves, "live", date(2026, 11, 20), date(2026, 11, 28),
                     "headfix")
    assert "code headfix." in text and pr.NOTE in text and "item 37, Luke's" in text
    assert ("Plans created: 3 (revisions 2, superseded 1). Verdicts with a plan record: 6 "
            "(REVISE 3). Revisions with the replaced plan read: 2 of 2; with a recorded ATR: 1."
            ) in text
    # plans per ticker: ANET created 2, revisions 2, past 0, same week 1, later 1, none 0,
    # life n 1 (30 h), horizon n 2 (7 and 45 days).
    assert "| ANET | 2 | 2 | 0 | 1 | 1 | 0 | 1 | 30.00 | 30.00 | 30.00 | 2 | 26.00 | 7.00 | 45.00 |" in text
    assert "| NVDA | 1 | 0 | 0 | 0 | 0 | 1 | 0 | - | - | - | 1 | 90.00 | 90.00 | 90.00 |" in text
    # verdicts per ticker: relations in RELATIONS order then the other one, triggers, pair.
    assert "| ANET | 3 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 1 | 1 | 0 |" in text
    assert "| NVDA | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |" in text
    assert "SHADOW_ODD" in text
    assert "| 2026-11-21 | NVDA | 3 |" in text
    assert "| ANET | t1 | 1 | 10.00 | 10.00 | 10.00 | 1 | 5.00 | 5.00 | 5.00 |" in text
    assert "| ANET | structural_stop | 2 | -2.50 | -5.00 | 0.00 | 1 | -2.00 | -2.00 | -2.00 |" in text
    assert "| 2026-11-20 | ANET | 2 | 1 | 1 | 10.00 | 5.000 | 0.00 | 0.000 | -5.00 | -2.000 |" in text
    assert "| 2026-11-21 | ANET | 3 | 2 | 2 | none | none | 0.00 | none | 0.00 | none |" in text
    for v in ("123.45", "131.17", "98.76", "135.79", "93.82", "2.469"):
        assert v not in text


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
        sql = self.conn.log[-1][0]
        return self.conn.plans if sql == pr._PLANS_SQL else self.conn.verdicts


class _Conn:
    def __init__(self, plans=(), verdicts=()):
        self.log, self.plans, self.verdicts, self.rolled = [], list(plans), list(verdicts), 0

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
    monkeypatch.setattr(pr.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="headfix\n"))
    return put


def test_main_reads_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    conn = _Conn(*_raw())
    put = _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert pr.main(["--since", "2026-11-20", "--until", "2026-11-28", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls == ["SET TRANSACTION READ ONLY", "SET LOCAL statement_timeout = %s",
                    pr._PLANS_SQL, pr._VERDICTS_SQL]
    assert conn.log[1][1] == (60000,)
    p = conn.log[2][1]
    assert p == {"env": "live", "start": datetime(2026, 11, 20, tzinfo=NY),
                 "end": datetime(2026, 11, 28, tzinfo=NY)} and conn.log[3][1] == p
    assert conn.rolled == 1 and put == [conn]
    text, data = out.read_text(), js.read_text()
    assert "code headfix." in text
    d = json.loads(data)
    assert d["code_version"] == "headfix" and len(d["moves"]) == 2
    assert all("ladder_atr" not in v for v in d["verdicts"])
    for v in ("123.45", "131.17", "98.76", "135.79", "93.82", "2.469"):
        assert v not in text and v not in data


def test_the_sql_is_env_bounded_and_reads_the_prior_plan():
    assert "(tp.trading_env = %(env)s OR tp.trading_env IS NULL)" in pr._PLANS_SQL
    assert "tp.created_at >= %(start)s AND tp.created_at < %(end)s" in pr._PLANS_SQL
    assert "r.id IN (SELECT replaces_id FROM w WHERE replaces_id IS NOT NULL)" in pr._PLANS_SQL
    assert "s.provenance ? 'plan'" in pr._VERDICTS_SQL
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in pr._VERDICTS_SQL
    for sql in (pr._PLANS_SQL, pr._VERDICTS_SQL):
        assert not any(w in sql.upper() for w in ("INSERT", "UPDATE ", "DELETE", "ALTER"))


def test_the_record_keys_are_plans():
    src = open(os.path.join(pr.ROOT, "agents", "plans.py")).read()
    assert 'return {"revise_triggers": fired, "catalyst_month_only_pair": pair}' in src
    assert 'out["new_plan_id"] = db.insert_plan(' in src
    assert '"ladder_atr": a}' in src
    for t in pr.TRIGGERS:
        assert f'fired.append("{t}")' in src
    db_src = open(os.path.join(pr.ROOT, "db.py")).read()
    assert "status='superseded', closed_at=NOW()" in db_src
    assert '_PLAN_LEVEL_COLS = ("structural_stop", "t1", "t2", "add_trigger",' in db_src


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert pr.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    rc = pr.main(["--out", os.devnull], now=datetime(2026, 11, 27, *hm, tzinfo=NY))
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert pr.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--since", "2026-11-26", "--until", "2026-11-26"],
                                  ["--since", "11/26/2026"]])
def test_main_rejects_bad_windows(monkeypatch, argv):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert pr.main(argv, now=NOW) == 2 and conn.log == []


def test_main_defaults_the_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert pr.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["start"] == datetime(2026, 11, 9, tzinfo=NY)
    assert conn.log[2][1]["end"] == datetime(2026, 11, 28, tzinfo=NY)


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        pr.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert pr.db_root(str(tmp_path)) == pr.ROOT
    (tmp_path / "db.py").write_text("")
    assert pr.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "json", "math", "os", "statistics", "subprocess", "sys",
                       "collections", "datetime", "zoneinfo"}
    assert "    import db\n" in src and "agents" not in {i for i in imports}
    assert "plan_revision_readout" not in open(os.path.join(pr.ROOT, "weekly_readouts.py")).read()


# --- Code-level flips -------------------------------------------------------
B0 = datetime(2026, 11, 25, 17, 16, 59, tzinfo=NY)
B1 = datetime(2026, 11, 26, 17, 16, 59, tzinfo=NY)


def _flip_raw():
    lad = dict(tk="NVDA", lsrc="ladder")
    plans = [
        _plan(57, created=_ts(26, 14), entry=200.0, t1=230.0, add=231.0, stop=180.0, built=B0,
              status="superseded", closed=_ts(27, 10), **lad),
        # A new build: t1 on the far rung.
        _plan(58, created=_ts(27, 10), replaces=57, revision=1, entry=210.0, t1=250.0, add=251.0,
              stop=185.0, built=B1, status="superseded", closed=_ts(27, 12), **lad),
        # The same build, a lower price: t1 re-picks the nearer rung.
        _plan(59, created=_ts(27, 12), replaces=58, revision=2, entry=203.0, t1=208.56,
              add=209.56, stop=185.0, built=B1, status="superseded", closed=_ts(27, 13), **lad),
        _plan(60, created=_ts(27, 13), replaces=59, revision=3, entry=203.5, t1=208.56,
              add=209.56, stop=185.0, built=B1, **lad),
        _plan(61, tk="NVDA", created=_ts(27, 13), source="held_seed", entry=150.0, t1=None,
              add=None, stop=None),
    ]
    verdicts = [
        _verdict(1, "REVISE", ts=_ts(27, 10), sym="NVDA", trig=["catalyst_date"], pair=False,
                 new_id=58, atr=4.0),
        _verdict(2, "REVISE", ts=_ts(27, 12), sym="NVDA", trig=["catalyst_date", "target"],
                 pair=False, new_id=59),
        _verdict(3, "REVISE", ts=_ts(27, 13), sym="NVDA", trig=["catalyst_date"], pair=True,
                 new_id=60),
        _verdict(4, "SEEDED", ts=_ts(27, 13), sym="NVDA"),
    ]
    return plans, verdicts


def _flips():
    plans_raw, verdicts_raw = _flip_raw()
    plans, levels = pr.build_plans(plans_raw)
    verdicts = pr.build_verdicts(verdicts_raw)
    flips = pr.build_flips(plans, levels, verdicts)
    flips["distinct_t1"] = pr.distinct_t1(plans, levels)
    return plans, verdicts, pr.build_moves(plans, levels, verdicts), flips


@pytest.mark.parametrize("old, new, changes, want", [
    ("b1", "b1", {"t1": "changed", "add_trigger": "changed", "structural_stop": "unchanged"},
     "same-build flip"),
    ("b1", "b1", {"t1": "unchanged", "add_trigger": "missing", "structural_stop": "unchanged"},
     "same-build unchanged"),
    ("b1", "b1", {"t1": "missing", "add_trigger": "missing", "structural_stop": "missing"},
     "same-build no levels"),
    ("b0", "b1", {"t1": "changed", "add_trigger": "unchanged", "structural_stop": "unchanged"},
     "new build changed"),
    ("b0", "b1", {"t1": "unchanged", "add_trigger": "unchanged", "structural_stop": "unchanged"},
     "new build unchanged"),
    (None, "b1", {"t1": "changed", "add_trigger": "changed", "structural_stop": "changed"},
     "build unknown"),
])
def test_flip_kind(old, new, changes, want):
    assert pr.flip_kind(old, new, changes) == want


def test_changed_uses_the_rounding_tolerance():
    assert pr.FLIP_TOL_PCT == 0.01
    assert pr._changed(100.0, 100.009) == "unchanged"
    assert pr._changed(100.0, 100.011) == "changed"
    assert pr._changed(None, 100.0) == "missing"


def test_basis_and_atr_source():
    assert pr.basis({"source": "analyst", "levels_source": "ladder"}) == "match"
    for lv in ({"source": "held_seed", "levels_source": None},
               {"source": "analyst", "levels_source": "analyst_only"}, {}):
        assert pr.basis(lv) == "basis unknown"
    lv = {"t1": 10.0, "add_trigger": 10.5}
    assert pr.plan_atr(lv, 1.9) == (1.9, "recorded")
    assert pr.plan_atr(lv, None) == (2.0, "derived") and pr.plan_atr(lv, 0) == (2.0, "derived")
    assert pr.plan_atr({"t1": 10.0, "add_trigger": 10.0}, None) == (None, "none")
    assert pr.plan_atr({}, "nan") == (None, "none")


def test_build_flips_labels_the_nvda_shape():
    _, _, _, flips = _flips()
    pairs = {r["plan_id"]: r for r in flips["pairs"]}
    assert pairs[58]["kind"] == "new build changed" and pairs[58]["atr_source"] == "recorded"
    assert pairs[59]["kind"] == "same-build flip"
    assert pairs[59]["changes"] == {"t1": "changed", "add_trigger": "changed",
                                    "structural_stop": "unchanged"}
    assert pairs[59]["triggers"] == "target+catalyst_date" and pairs[59]["atr_source"] == "derived"
    assert pairs[59]["old_t1_from_new_entry_atr"] == 11.75
    assert pairs[60]["kind"] == "same-build unchanged"
    assert pairs[60]["old_t1_from_new_entry_atr"] is None
    plans = {r["plan_id"]: r for r in flips["plans"]}
    assert plans[57]["t1_edge_margin_atr"] == 6.5 and plans[58]["t1_edge_margin_atr"] == 9.0
    assert plans[59]["t1_edge_margin_atr"] == pytest.approx(0.39)
    assert plans[60]["t1_edge_margin_atr"] == pytest.approx(0.265)
    assert plans[61]["basis"] == "basis unknown" and plans[61]["t1_edge_margin_atr"] is None
    assert plans[61]["atr_source"] == "none" and plans[58]["derived_vs_recorded_pct"] == 0.0
    assert flips["distinct_t1"] == {("NVDA", "2026-11-25 17:16"): 1,
                                    ("NVDA", "2026-11-26 17:16"): 2, ("NVDA", "none"): 0}
    for v in ("208.56", "209.56", "203.5", "251.0", "231.0"):
        assert v not in json.dumps(flips["plans"]) + json.dumps(flips["pairs"])


def test_render_flips_without_levels():
    plans, verdicts, moves, flips = _flips()
    text = pr.render(plans, verdicts, moves, "live", date(2026, 11, 26), date(2026, 11, 28),
                     "headfix", flips)
    assert "## Code-level flips" in text and pr.FLIP_NOTE in text and "are Luke's" in text
    assert ("Revision pairs: 3 (same-build flip 1, same-build unchanged 1, same-build no levels 0, "
            "new build changed 1, new build unchanged 0, build unknown 0). Plans by basis: "
            "match 4, basis unknown 1; ATR source: recorded 1, derived 3, none 1.") in text
    assert "Derived vs recorded ATR, percent (plans with both): n | median | min | max: " \
           "1 | 0.00 | 0.00 | 0.00." in text
    assert "| NVDA | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |" in text
    assert "| NVDA | 2026-11-26 17:16 | 3 | 2 | 1 |" in text
    assert "| NVDA | 2026-11-25 17:16 | 1 | 1 | 0 |" in text and "| NVDA | none | 1 | 0 | 0 |" in text
    assert "| catalyst_date | 0 | 1 | 0 | 1 | 0 | 0 | 2 |" in text
    assert "| target+catalyst_date | 1 | 0 | 0 | 0 | 0 | 0 | 1 |" in text
    assert "| 1 | 11.75 | 11.75 | 11.75 | 0 |" in text
    assert ("| 2026-11-27 | NVDA | 59 | 58 | 2026-11-26 17:16 | same-build flip | changed | changed "
            "| unchanged | target+catalyst_date | derived | 0.390 | 11.750 |") in text
    for v in ("208.56", "209.56", "203.5", "251.0", "231.0", "210.0", "200.0"):
        assert v not in text


def test_main_writes_the_flips_without_levels(monkeypatch, tmp_path):
    conn = _Conn(*_flip_raw())
    _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert pr.main(["--since", "2026-11-26", "--until", "2026-11-28", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    text, data = out.read_text(), js.read_text()
    assert "## Code-level flips" in text and "code headfix." in text
    d = json.loads(data)
    assert len(d["flip_pairs"]) == 3 and len(d["flip_plans"]) == 5
    for v in ("208.56", "209.56", "203.5", "251.0", "231.0", "210.0", "200.0"):
        assert v not in text and v not in data


def test_the_plan_sql_reads_the_flip_columns():
    for alias in ("p", "r"):
        assert (f"{alias}.entry_price, {alias}.levels_source, {alias}.ladder_computed_at"
                in pr._PLANS_SQL)


def test_the_entry_and_band_rules_are_the_codes():
    plans = open(os.path.join(pr.ROOT, "agents", "plans.py")).read()
    assert "        price = brief_price(brief)\n" in plans
    assert "new = (plan_from_verdict(ticker, parsed, conviction, price, today)" in plans
    assert '"entry_price": price,' in plans
    assert plans.count("db.insert_plan(_with_levels(new, parsed, price, ticker, today") == 2
    assert "lv = compile_plan(parsed, price, ladder, None, today)" in plans
    assert 'seed.update(source="held_seed", entry_price=stop["entry_price"],' in plans
    assert '"plan_id": db.insert_plan(seed)' in plans
    assert 't1 = resist[0]["price"] if resist else None' in plans
    assert 'add_trigger=resist[0].get("band_high") if resist else None,' in plans
    assert 'resist = [r for r in rungs if atr and r["price"] >= price + atr]' in plans
    assert 'atr = float((ladder or {}).get("atr") or 0) or None' in plans
    assert 'seen["atr"] = (ladder or {}).get("atr")' in plans
    lad = open(os.path.join(pr.ROOT, "agents", "trade_ladder.py")).read()
    assert f"RUNG_BAND_ATR_FRACTION = {pr.RUNG_BAND_ATR_FRACTION}" in lad
    assert "band = atr * RUNG_BAND_ATR_FRACTION" in lad
    assert '"price": round(anchor, 2),' in lad and '"band_high": round(anchor + band, 2),' in lad
    assert "rungs = _merge_rungs(raw_levels, atr) if atr" in lad
    assert '"atr": round(atr, 4) if atr else None,' in lad
