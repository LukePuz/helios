"""scripts/plan_level_readout.py on fixture plans and bars: compiled level
crossings per plan and in force at the open, vs the exit, and the forward
table; read-only, no prices in the output."""
import importlib.util
import inspect
import os
from datetime import date, datetime, timedelta, timezone

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "plan_level_readout.py")
_spec = importlib.util.spec_from_file_location("plan_level_readout", _PATH)
plr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plr)

NY = plr.NY


def _days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _days(date(2026, 11, 2), 40)


def _bar(d, close, hi=None, lo=None):
    return {"date": d, "high": hi or close * 1.01, "low": lo or close * 0.99, "close": close}


def _et(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=NY)


def _plan(i, created, closed=None, **kw):
    p = {"id": i, "ticker": "NVDA", "status": "active", "source": "analyst", "created_at": created,
         "closed_at": closed, "entry_price": 100.0, "structural_stop": 90.0, "t1": 110.0,
         "t2": 130.0, "add_trigger": 111.5, "horizon_end": DAYS[30], "levels_source": "ladder"}
    p.update(kw)
    return p


def test_window_takes_full_sessions_only():
    bars = [_bar(d, 100.0) for d in DAYS[:5]]
    w = plr.window(bars, _et(DAYS[0], 11), _et(DAYS[3], 12))
    assert [b["date"] for b in w] == [DAYS[1], DAYS[2]]          # day 0 opened before; day 3 not closed
    assert [b["date"] for b in plr.window(bars, _et(DAYS[0], 8), _et(DAYS[0], 16))] == [DAYS[0]]


def test_first_crossings_intraday_and_close():
    sess = [_bar(DAYS[0], 100.0), _bar(DAYS[1], 105.0, hi=110.5), _bar(DAYS[2], 111.0),
            _bar(DAYS[3], 95.0, lo=89.0)]
    c = plr.first_crossings(_plan(1, _et(DAYS[0], 8)), sess)
    ok = {"intraday": False, "close": False}
    assert c["t1"] == {"intraday": (1, DAYS[1]), "close": (2, DAYS[2]), "nonfinite": ok}
    assert c["structural_stop"] == {"intraday": (3, DAYS[3]), "close": None, "nonfinite": ok}
    assert c["t2"] == {"intraday": None, "close": None, "nonfinite": ok}
    assert c["time_stop"]["intraday"] is None
    c = plr.first_crossings(_plan(1, _et(DAYS[0], 8), t1=None, horizon_end=DAYS[1]), sess)
    assert "t1" not in c and c["time_stop"]["intraday"] == (2, DAYS[2])


def test_exit_and_vs_exit():
    outcomes = [("NVDA", _et(DAYS[1], 15)), ("NVDA", _et(DAYS[5], 15)), ("AMD", _et(DAYS[0], 10))]
    ex = plr.exit_after(outcomes, "NVDA", _et(DAYS[2], 9))
    assert ex == _et(DAYS[5], 15)
    assert plr._vs_exit((0, DAYS[3]), ex) == "before exit"
    assert plr._vs_exit((0, DAYS[5]), ex) == "exit day"
    assert plr._vs_exit(None, ex) == "not crossed" and plr._vs_exit((0, DAYS[3]), None) == "no exit"


def test_in_force_uses_the_newest_plan_at_the_open():
    bars = {"NVDA": [_bar(d, 100.0) for d in DAYS[:4]]}
    plans = [_plan(1, _et(DAYS[0], 8), closed=_et(DAYS[1], 12)),
             _plan(2, _et(DAYS[1], 12), closed=_et(DAYS[1], 13), t1=100.5),   # revised within a day
             _plan(3, _et(DAYS[1], 13), t1=100.5)]
    view = plr.in_force(plans, bars, [], _et(DAYS[3], 17))
    assert [(d["date"], d["plan_id"]) for d in view["NVDA"]] == [
        (DAYS[0], 1), (DAYS[1], 1), (DAYS[2], 3), (DAYS[3], 3)]
    assert view["NVDA"][2]["crossed"]["t1"] is True and view["NVDA"][1]["crossed"]["t1"] is False
    # a session that has not closed yet is left out
    assert len(plr.in_force(plans, bars, [], _et(DAYS[3], 12))["NVDA"]) == 3


def test_in_force_summary_first_crossing_vs_exit():
    bars = {"NVDA": [_bar(DAYS[0], 100.0), _bar(DAYS[1], 105.0, hi=110.5), _bar(DAYS[2], 112.0)]}
    view = plr.in_force([_plan(1, _et(DAYS[0], 8))], bars, [], _et(DAYS[2], 17))
    s = {r["level"]: r for r in plr.in_force_summary(view, [("NVDA", _et(DAYS[2], 11))])}
    assert s["t1"]["sessions"] == 3 and s["t1"]["crossed"] == 2 and s["t1"]["firsts"] == {"NVDA": DAYS[1]}
    assert s["t1"]["vs_exit"] == ["before exit"] and s["structural_stop"]["crossed"] == 0


def test_build_end_to_end_has_no_prices():
    closes = [123.45 * (1 + 0.004 * i) for i in range(len(DAYS))]
    bars = {"NVDA": [_bar(d, c) for d, c in zip(DAYS, closes)],
            "SPY": [_bar(d, 400.0 + i) for i, d in enumerate(DAYS)]}
    plans = [_plan(1, _et(DAYS[0], 8), entry_price=123.45, structural_stop=111.11, t1=128.0,
                   add_trigger=129.0, t2=160.0),
             _plan(2, _et(DAYS[0], 9), levels_source=None)]
    text = plr.build(plans, [("NVDA", _et(DAYS[20], 15))], bars, DAYS[0], "live", "abc1234",
                     _et(DAYS[-1], 17))
    assert "Plans with compiled levels: 1 on 1 names (1 plans" in text
    assert "## Levels in force at the open" in text and "t1 crossed (in force, intraday)" in text
    assert "plan created (baseline)" in text and "### Forward-return readout" in text
    for price in ("123.45", "111.11", "128.0", "160"):
        assert price not in text, price


def test_fetch_only_reads():
    src = inspect.getsource(plr._fetch)
    assert "SET TRANSACTION READ ONLY" in src and "conn.rollback()" in src
    for word in ("INSERT", "UPDATE ", "DELETE", "commit"):
        assert word not in src, word


# --- non-finite values (audit 2026-11-27 16:53 #3 follow-up) ------------------

NAN, INF = float("nan"), float("inf")


def test_nonfinite_values():
    for v in (NAN, INF, -INF, "NaN"):
        assert plr.nonfinite(v)
    for v in (None, True, 0, 1.5, "x"):
        assert not plr.nonfinite(v)


def test_a_nan_bar_before_the_first_crossing_makes_that_rule_unknown():
    sess = [_bar(DAYS[0], 100.0), {"date": DAYS[1], "high": NAN, "low": 99.0, "close": 105.0},
            _bar(DAYS[2], 111.0)]
    c = plr.first_crossings(_plan(1, _et(DAYS[0], 8)), sess)["t1"]
    # intraday reads the NaN high before any crossing: unknown; close crosses on day 2
    assert c["nonfinite"] == {"intraday": True, "close": False}
    assert c["intraday"] is None and c["close"] == (2, DAYS[2])


def test_a_nan_bar_after_the_first_crossing_changes_nothing():
    sess = [_bar(DAYS[0], 100.0, hi=110.5), {"date": DAYS[1], "high": NAN, "low": NAN, "close": NAN}]
    c = plr.first_crossings(_plan(1, _et(DAYS[0], 8)), sess)["t1"]
    assert c["intraday"] == (0, DAYS[0]) and c["nonfinite"]["intraday"] is False
    assert c["nonfinite"]["close"] is True and c["close"] is None   # no close crossing before the NaN


def test_a_nonfinite_level_is_unknown_for_both_rules_and_has_no_pct():
    sess = [_bar(DAYS[0], 100.0), _bar(DAYS[1], 120.0)]
    c = plr.first_crossings(_plan(1, _et(DAYS[0], 8), t1=NAN), sess)["t1"]
    assert c["nonfinite"] == {"intraday": True, "close": True} and c["intraday"] is None
    rec = plr.analyse([_plan(1, _et(DAYS[0], 8), t1=INF, entry_price=100.0)],
                      {"NVDA": sess}, [], _et(DAYS[1], 17))[0]
    assert rec["pct_from_entry"]["t1"] is None and rec["pct_from_entry"]["t2"] == 30.0
    rec = plr.analyse([_plan(1, _et(DAYS[0], 8), entry_price=NAN)], {"NVDA": sess}, [],
                      _et(DAYS[1], 17))[0]
    assert all(v is None for v in rec["pct_from_entry"].values())


def test_in_force_counts_nonfinite_sessions_and_an_unknown_first_crossing():
    bars = {"NVDA": [_bar(DAYS[0], 100.0), {"date": DAYS[1], "high": NAN, "low": 99.0, "close": 100.0},
                     _bar(DAYS[2], 112.0)],
            "AMD": [_bar(DAYS[0], 112.0), {"date": DAYS[1], "high": NAN, "low": 99.0, "close": 100.0}]}
    plans = [_plan(1, _et(DAYS[0], 8)), _plan(2, _et(DAYS[0], 8), ticker="AMD")]
    view = plr.in_force(plans, bars, [], _et(DAYS[2], 17))
    assert view["NVDA"][1]["crossed"]["t1"] is None and view["NVDA"][1]["crossed"]["structural_stop"] is False
    s = {r["level"]: r for r in plr.in_force_summary(view, [])}["t1"]
    assert s["sessions"] == 3 and s["crossed"] == 2 and s["nonfinite_sessions"] == 2
    assert s["nonfinite_names"] == ["AMD", "NVDA"]
    assert s["firsts"] == {"AMD": DAYS[0]} and s["uncertain_first"] == ["NVDA"]   # AMD crossed first
    assert all(t != "NVDA" or "t1" not in lab for t, lab, _, _ in plr.forward_rows([], [s]))


def test_readout_columns_and_widths_with_nonfinite():
    nan_day = {"date": DAYS[3], "high": NAN, "low": NAN, "close": NAN}
    bars = {"NVDA": [_bar(d, 100.0) for d in DAYS[:3]] + [nan_day]
            + [_bar(d, 100.0) for d in DAYS[4:30]],
            "SPY": [_bar(d, 400.0) for d in DAYS[:30]]}
    plans = [_plan(1, _et(DAYS[0], 8))]
    text = plr.build(plans, [], bars, DAYS[0], "live", "pinned0", _et(DAYS[29], 17))
    assert "Generated " in text and "at code pinned0" in text
    # no level crossed, so the NaN day is before any crossing for every price level
    assert "| t1 | 1 | 1 (NVDA) | 1 (NVDA) | 0 | 0 | - | 0 | 0 | 0 | 0 |" in text
    assert "| t1 | 29 | 1 (NVDA) | 0 (0%) | 0 | 1 (NVDA) | 0 | 0 | 0 |" in text
    assert "| time_stop | 1 | 0 | 0 | 0 (0%) | 0 (0%) | - | 0 | 0 | 0 | 0 |" in text
    for head in ("| level | plans with it |", "| level | sessions in force |"):
        block = text.split(head)[1].split("\n\n")[0]
        widths = {l.count("|") for l in (head + block).splitlines() if l.startswith("|")}
        assert len(widths) == 1, widths
    assert "nan" not in text.replace("non-finite", "").replace("NaN", "").lower().replace("financ", "")
