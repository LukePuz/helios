"""scripts/compile_plan_readout.py on fixture lots, approvals, plans and
bars: which compile_plan record a lot gets, the stop gap, the structural
stop before the later high, t1 vs the stop, add_trigger firing, and the
markdown (insufficient under n=5, no prices, read-only fetch)."""
import importlib.util
import inspect
import os
import re
from datetime import date, datetime, timedelta, timezone

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "compile_plan_readout.py")
_spec = importlib.util.spec_from_file_location("compile_plan_readout", _PATH)
cpr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpr)

NY = cpr.NY


def _days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _days(date(2026, 11, 2), 30)


def _et(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=NY)


def _bar(d, close, hi=None, lo=None):
    return {"date": d, "high": hi or close * 1.01, "low": lo or close * 0.99, "close": close}


def _lot(i=1, ticker="NVDA", entry=DAYS[0], exit_=DAYS[5], source="broker_exit", **kw):
    lot = {"id": i, "ticker": ticker, "action": "OPEN", "approval_id": 100 + i,
           "entry_date": _et(entry, 11), "entry_price": 100.0,
           "exit_date": _et(exit_, 15) if exit_ else None,
           "exit_at": _et(exit_, 15) if exit_ else None,
           "exit_price": 93.0 if exit_ else None, "exit_source": source if exit_ else None}
    lot.update(kw)
    return lot


def _approval(i=101, ts=None, price=100.0, **pc):
    plan = {"structural_stop": 90.0, "t1": 110.0, "t2": 130.0, "add_trigger": 111.0,
            "time_stop": str(DAYS[20]), "actual_stop": 94.0, "stop_gap_pct": 4.0,
            "structural_stop_pct": 10.0}
    plan.update(pc)
    return {"id": i, "timestamp": ts or _et(DAYS[0], 10, 55),
            "risk_json": {"price_at_sizing": price, "stop_loss_price": 94.0, "plan_compiled": plan}}


def _plan(i, created, **kw):
    p = {"id": i, "ticker": "NVDA", "created_at": created, "entry_price": 100.0,
         "structural_stop": 88.0, "t1": 112.0, "t2": 125.0, "add_trigger": 113.0,
         "horizon_end": DAYS[25], "levels_source": "ladder"}
    p.update(kw)
    return p


# --- pick_record -----------------------------------------------------------

def test_sizing_record_wins_and_uses_price_at_sizing():
    rec = cpr.pick_record(_lot(), _approval(price=101.0), [_plan(1, _et(DAYS[0], 9))])
    assert rec["source"] == "sizing"
    assert rec["price"] == 101.0
    assert rec["structural_stop"] == 90.0
    assert rec["actual_stop"] == 94.0
    assert rec["stop_gap_pct"] == 4.0
    assert rec["time_stop"] == DAYS[20]


def test_sizing_record_price_backs_out_of_structural_stop_pct():
    a = _approval()
    del a["risk_json"]["price_at_sizing"]
    rec = cpr.pick_record(_lot(), a, [])
    assert abs(rec["price"] - 100.0) < 1e-9          # 90 / (1 - 0.10)


def test_sizing_gap_computed_when_not_recorded():
    rec = cpr.pick_record(_lot(), _approval(stop_gap_pct=None), [])
    assert rec["stop_gap_pct"] == 4.0                # (94 - 90) / 100 * 100


def test_errored_sizing_record_falls_back_to_newest_plan_before_entry():
    a = {"id": 101, "timestamp": _et(DAYS[0], 10), "risk_json": {
        "stop_loss_price": 95.0, "plan_compiled": {"error": "boom"}}}
    plans = [_plan(1, _et(DAYS[0], 8)), _plan(2, _et(DAYS[0], 10), structural_stop=91.0),
             _plan(3, _et(DAYS[0], 12), structural_stop=99.0),          # after entry
             _plan(4, _et(DAYS[0], 9), ticker="AMD"),                    # other name
             _plan(5, _et(DAYS[0], 10, 30), levels_source=None)]         # no compiled levels
    rec = cpr.pick_record(_lot(), a, plans)
    assert rec["source"] == "trade_plans"
    assert rec["structural_stop"] == 91.0
    assert rec["actual_stop"] == 95.0
    assert rec["stop_gap_pct"] == 4.0                # (95 - 91) / 100 * 100
    assert rec["time_stop"] == DAYS[25]


def test_no_record_when_nothing_before_entry():
    assert cpr.pick_record(_lot(), None, [_plan(1, _et(DAYS[1], 9))]) is None
    assert cpr.pick_record(_lot(), {"id": 101, "risk_json": None}, []) is None


def test_trade_plans_gap_none_without_placed_stop():
    rec = cpr.pick_record(_lot(), {"id": 101, "risk_json": {}}, [_plan(1, _et(DAYS[0], 9))])
    assert rec["actual_stop"] is None
    assert rec["stop_gap_pct"] is None


# --- analyse_lot -----------------------------------------------------------

def _rec(**kw):
    r = cpr.pick_record(_lot(), _approval(), [])
    r.update(kw)
    return r


def test_life_excludes_entry_day_and_includes_exit_day():
    bars = [_bar(d, 100.0) for d in DAYS[:8]]
    r = cpr.analyse_lot(_lot(), _rec(), bars)
    assert r["life_sessions"] == 5                   # DAYS[1]..DAYS[5]
    assert r["stopped"] is True
    assert r["bars_cover_exit"] is True


def test_t1_first_then_stopped():
    bars = [_bar(DAYS[0], 100.0, hi=120.0),          # entry day: ignored
            _bar(DAYS[1], 105.0, hi=110.5), _bar(DAYS[2], 100.0), _bar(DAYS[3], 95.0, lo=93.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[3]), _rec(), bars)
    assert r["t1_vs_stopout"] == "t1 first"


def test_t1_on_stopout_session_is_same_session():
    bars = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 100.0, hi=111.0, lo=92.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[2]), _rec(), bars)
    assert r["t1_vs_stopout"] == "same session"


def test_no_t1_touch_before_stopout():
    bars = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 95.0, lo=93.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[2]), _rec(), bars)
    assert r["t1_vs_stopout"] == "no t1 touch"


def test_stopped_lot_without_exit_day_bar():
    bars = [_bar(DAYS[1], 100.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[2]), _rec(), bars)
    assert r["t1_vs_stopout"] == "no bar for the exit day"
    assert r["bars_cover_exit"] is False


def test_not_stopped_lot_t1_labels():
    bars = [_bar(DAYS[1], 105.0, hi=111.0), _bar(DAYS[2], 104.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[2], source="model_exit"), _rec(), bars)
    assert r["stopped"] is False
    assert r["t1_vs_stopout"] == "t1 touched, not stopped"
    r2 = cpr.analyse_lot(_lot(exit_=DAYS[2], source="model_exit"), _rec(t1=150.0),
                         [_bar(DAYS[1], 100.0), _bar(DAYS[2], 100.0)])
    assert r2["t1_vs_stopout"] == "no t1 touch, not stopped"


def test_t1_vs_structural_orders():
    rec = _rec()
    first_struct = [_bar(DAYS[1], 95.0, lo=89.0), _bar(DAYS[2], 105.0, hi=111.0)]
    assert cpr.analyse_lot(_lot(exit_=DAYS[2]), rec, first_struct)["t1_vs_structural"] == \
        "structural first"
    first_t1 = [_bar(DAYS[1], 105.0, hi=111.0), _bar(DAYS[2], 95.0, lo=89.0)]
    assert cpr.analyse_lot(_lot(exit_=DAYS[2]), rec, first_t1)["t1_vs_structural"] == "t1 first"
    both = [_bar(DAYS[1], 100.0, hi=111.0, lo=89.0)]
    assert cpr.analyse_lot(_lot(exit_=DAYS[1]), rec, both)["t1_vs_structural"] == "same session"
    neither = [_bar(DAYS[1], 100.0)]
    assert cpr.analyse_lot(_lot(exit_=DAYS[1]), rec, neither)["t1_vs_structural"] == "neither"


def test_structural_window_runs_past_the_exit():
    # t1 is touched only after the stop-out: still "t1 first" vs structural.
    bars = [_bar(DAYS[1], 95.0, lo=93.0), _bar(DAYS[2], 105.0, hi=110.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[1]), _rec(), bars)
    assert r["t1_vs_stopout"] == "no t1 touch"
    assert r["t1_vs_structural"] == "t1 first"


def test_later_high_structural_held():
    bars = [_bar(DAYS[1], 97.0), _bar(DAYS[2], 94.0, lo=92.0),           # exit day
            _bar(DAYS[3], 96.0, lo=91.0), _bar(DAYS[4], 104.0, hi=108.0),  # the later high
            _bar(DAYS[5], 100.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[2]), _rec(), bars)
    lh = r["later_high"]
    assert lh["date"] == DAYS[4]
    assert lh["sessions_after_exit"] == 2
    assert lh["pct_vs_price"] == 8.0
    assert lh["above_t1"] is False
    assert lh["structural_breached_before"] is False
    assert lh["structural_breached_on_high_day"] is False


def test_later_high_structural_breached_after_exit():
    bars = [_bar(DAYS[1], 94.0, lo=92.0), _bar(DAYS[2], 89.0, lo=88.0),
            _bar(DAYS[3], 112.0, hi=113.0, lo=100.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[1]), _rec(), bars)
    lh = r["later_high"]
    assert lh["structural_breached_before"] is True
    assert lh["above_t1"] is True


def test_later_high_own_session_flagged_apart():
    bars = [_bar(DAYS[1], 94.0, lo=92.0), _bar(DAYS[2], 105.0, hi=109.0, lo=89.5)]
    lh = cpr.analyse_lot(_lot(exit_=DAYS[1]), _rec(), bars)["later_high"]
    assert lh["structural_breached_before"] is False
    assert lh["structural_breached_on_high_day"] is True


def test_later_high_none_without_bars_after_exit():
    r = cpr.analyse_lot(_lot(exit_=DAYS[1]), _rec(), [_bar(DAYS[1], 94.0, lo=92.0)])
    assert r["later_high"] is None


def test_later_high_only_for_stopped_lots():
    bars = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 120.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[1], source="model_exit"), _rec(), bars)
    assert "later_high" not in r


def test_add_trigger_fired_and_after():
    bars = [_bar(DAYS[1], 105.0, hi=111.5),          # high over it, close under: not fired
            _bar(DAYS[2], 112.0),                    # first close at or over 111
            _bar(DAYS[3], 115.0, hi=117.6), _bar(DAYS[4], 100.0, lo=95.2)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[4], exit_price=96.0), _rec(), bars)
    a = r["add"]
    assert a["fired"] is True
    assert a["session"] == 1
    assert a["date"] == DAYS[2]
    assert a["exit_source"] == "broker_exit"
    assert a["exit_vs_close_pct"] == round((96.0 / 112.0 - 1) * 100, 2)
    assert a["max_high_pct"] == 5.0
    assert a["min_low_pct"] == -15.0
    assert a["structural_after"] is False


def test_add_trigger_after_exit_is_not_counted():
    bars = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 120.0)]
    r = cpr.analyse_lot(_lot(exit_=DAYS[1]), _rec(), bars)
    assert r["add"] == {"fired": False}


def test_time_stop_passed():
    bars = [_bar(d, 100.0) for d in DAYS[:4]]
    assert cpr.analyse_lot(_lot(exit_=DAYS[3]), _rec(time_stop=DAYS[2]), bars)["time_stop_passed"] is True
    assert cpr.analyse_lot(_lot(exit_=DAYS[3]), _rec(time_stop=DAYS[3]), bars)["time_stop_passed"] is False


def test_lot_without_record_has_no_level_fields():
    r = cpr.analyse_lot(_lot(), None, [_bar(DAYS[1], 100.0)])
    assert r["rec"] is None
    assert "pct" not in r
    assert "add" not in r


# --- describe / count --------------------------------------------------------

def test_describe_insufficient_under_five():
    assert cpr.describe([1.0, 2.0, 3.0, 4.0]) == "insufficient (n=4)"
    assert cpr.describe([]) == "insufficient (n=0)"


def test_describe_five_or_more():
    s = cpr.describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.startswith("n=5, min +1.00")
    assert "median +3.00" in s
    assert "max +5.00" in s


def test_count_marks_insufficient():
    assert cpr.count(1, 4) == "1 of 4 (insufficient)"
    assert cpr.count(2, 5) == "2 of 5 (40%)"


# --- readout -----------------------------------------------------------------

def _fixture(n_lots):
    lots, approvals, bars = [], {}, {}
    for i in range(n_lots):
        t = f"T{i}"
        lot = _lot(i + 1, ticker=t, exit_=DAYS[3], exit_price=93.37)
        lots.append(lot)
        approvals[lot["approval_id"]] = _approval(lot["approval_id"], price=100.0,
                                                  stop_gap_pct=float(i) - 1.0)
        bars[t] = [_bar(DAYS[1], 101.0, hi=110.25), _bar(DAYS[2], 97.0, lo=91.0),
                   _bar(DAYS[3], 94.0, lo=93.0), _bar(DAYS[4], 104.0, hi=106.0, lo=89.0)]
    lots.append(_lot(99, ticker="OLD", exit_=DAYS[3]))   # no record
    return lots, approvals, bars


def test_readout_reports_counts_and_marks_insufficient():
    lots, approvals, bars = _fixture(3)
    text = cpr.build(lots, [], approvals, [], bars, date(2026, 11, 1), "live", "abc1234",
                     datetime(2026, 11, 25, 12, tzinfo=timezone.utc))
    assert "Closed lots: 4. With a compile_plan record: 3 on 3 names" in text
    assert "Without one: 1" in text
    assert "Distribution: insufficient (n=3)" in text
    assert "| t1 first | 3 of 3 (insufficient) |" in text
    assert "| 99 | OLD | none |" in text


def test_readout_distribution_with_five():
    lots, approvals, bars = _fixture(5)
    text = cpr.build(lots, [], approvals, [], bars, date(2026, 11, 1), "live", "abc1234",
                     datetime(2026, 11, 25, 12, tzinfo=timezone.utc))
    assert "structural below placed 3 of 5 (60%), above 1 of 5 (20%), equal 1 of 5 (20%)" in text
    assert "Distribution: n=5, min -1.00" in text
    # Every stopped lot: structural (90) breached on the later high's own session only.
    assert "| not traded through before it | 5 of 5 (100%) |" in text
    assert "| low on the later high's own session at or below it (order unknown) | 5 of 5 (100%) |" in text


def test_readout_has_no_prices_or_dollars():
    lots, approvals, bars = _fixture(5)
    text = cpr.build(lots, [], approvals, [], bars, date(2026, 11, 1), "live", "abc1234",
                     datetime(2026, 11, 25, 12, tzinfo=timezone.utc))
    assert "$" not in text
    for raw in ("93.37", "110.25", "106.0", "94.0", "90.0"):
        assert raw not in text
    assert not re.search(r"shares|account", text, re.I)


def test_open_lots_only_in_the_stop_gap_line():
    lots, approvals, bars = _fixture(1)
    op = _lot(50, ticker="OPN", exit_=None)
    approvals[op["approval_id"]] = _approval(op["approval_id"], stop_gap_pct=-2.0)
    text = cpr.build(lots, [op], approvals, [], bars, date(2026, 11, 1), "live", "abc1234",
                     datetime(2026, 11, 25, 12, tzinfo=timezone.utc))
    assert "- open lots (not in the sections below): structural below placed 0 of 1" in text
    assert "| 50 |" not in text


def test_fetch_is_read_only():
    src = inspect.getsource(cpr._fetch)
    assert "SET TRANSACTION READ ONLY" in src
    assert "rollback()" in src
    for word in ("INSERT", "UPDATE", "DELETE", "commit("):
        assert word not in src
