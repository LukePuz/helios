"""scripts/held_side_readout.py on fixture rows: one row per held position
per ET day from the exit frame, held_udr and earnings check records; keys
not live yet read "not recorded yet", a missing source "no row"; the day's
latest row wins; the book row; the markdown (no prices); the port check and
the read-only fetch; and that weekly_readouts doesn't load it. No DB."""
import importlib.util
import os
import re
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "scripts", "held_side_readout.py")
_spec = importlib.util.spec_from_file_location("held_side_readout", _PATH)
hsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hsr)

NR, NO = hsr.NOT_RECORDED, hsr.NO_ROW


def _ts(d, h, m=0):
    """UTC timestamp; 22:30 UTC is 17:30 EST, 13:00 UTC is 08:00 EST."""
    return datetime(2026, 11, d, h, m, tzinfo=timezone.utc)


def _frame_live(**kw):
    """A frame row as live writes it today: no plan or open-risk keys."""
    return {"gain_pct": 4.1, "above_stop_pct": 7.5, "close_is_today": True, "broker_held": True,
            "env": "live", **kw}


def _frame_1134(**kw):
    book = {"book_open_risk_to_stop_frac": 0.0123, "book_uncapped_weight_frac": 0.0,
            "book_open_risk_gap_p5_frac": 0.015, "book_open_risk_gap_p1_frac": 0.02,
            "n_held_without_stop_row": 0, "snapshot_read": "ok", "n_by_read": {"ok": 2}}
    return _frame_live(plan_read="ok", plan_id=41, plan_revision=3, plan_source="analyst",
                       invalidation_vs_close_pct=-8.2, target_vs_close_pct=12.5, t1_vs_close_pct=6.0,
                       horizon_days_left=9, catalyst_date_past=False, next_print_days=84,
                       next_print_read="ok", open_risk_to_stop_frac=0.0061,
                       open_risk_to_stop={"read": "ok", "stop_source": "position_stops",
                                          "price_source": "price_history_close",
                                          "stop_at_or_above_price": False},
                       open_risk_gap_adjusted={"read": "ok", "gap_p5_frac": 0.0072,
                                               "gap_p1_frac": 0.0101},
                       open_risk_book=book, **kw)


def _udr(**kw):
    rec = {"status": "ok", "udr_now": 1.8, "open_lots": 1,
           "first": {"approval_id": 7, "udr": 2.4, "udr_at_sizing": 2.1},
           "last": None, "change_since_first": -0.6, "change_since_last": None}
    rec.update(kw)
    return rec


def _udr_1134(**kw):
    return _udr(components_now={"expected_return_pct": 9.0, "expected_downside_pct": 5.0},
                code_udr_now={"read": "ok", "udr": 1.6, "level_source": "t1"},
                plan_id=41, plan_revision=3, **kw)


def _earn(day, tickers):
    return {"ts": _ts(int(day[-2:]), 13), "extra_json": {"day": day, "env": "live", "tickers": tickers}}


def _earn_1134(held=True, **kw):
    r = {"result": "future_date", "next_date": "2027-02-18", "is_estimate": True,
         "has_future_row": True, "days_since_last_print": 6, "held": held,
         "catalyst_write": "ok", "est_next_print_from_cadence": "2027-02-17",
         "cadence_median_days": 91, "cadence_none_reason": None}
    r.update(kw)
    return r


def _row(rows, day, t):
    hit = [r for r in rows if r["day"] == day and r["ticker"] == t]
    assert len(hit) == 1, hit
    return hit[0]


def test_live_frame_rows_read_not_recorded_yet_for_the_1133_1134_keys():
    frames = [{"ts": _ts(24, 22, 30), "ticker": "NVDA", "extra_json": _frame_live()}]
    rows, book = hsr.build_rows(frames, [], [])
    r = _row(rows, "2026-11-24", "NVDA")
    assert r["gain_pct"] == 4.1 and r["sources"] == "frame"
    for c in ("plan_id", "plan_revision", "invalidation_vs_close_pct", "t1_vs_close_pct",
              "open_risk_to_stop_frac", "open_risk_read", "gap_p5_frac", "uncapped_weight_frac"):
        assert r[c] == NR, c
    for c in ("udr_status", "code_udr_now", "earn_result", "est_next_print_from_cadence",
              "cadence_est_days"):
        assert r[c] == NO, c
    assert book == [{"day": "2026-11-24", **{c: NR for c, _ in hsr.BOOK_FIELDS}}]


def test_a_1133_held_udr_without_the_1134_components_is_not_recorded_yet_there():
    frames = [{"ts": _ts(24, 22, 30), "ticker": "NVDA", "extra_json": _frame_live()}]
    udrs = [{"symbol": "NVDA", "timestamp": _ts(24, 15), "signal": "HOLD", "held_udr": _udr()}]
    r = _row(hsr.build_rows(frames, udrs, [])[0], "2026-11-24", "NVDA")
    assert (r["udr_status"], r["udr_now"], r["first_udr"], r["change_since_first"]) == ("ok", 1.8, 2.4, -0.6)
    assert r["change_since_last"] is None                       # there, no value: n/a
    assert r["expected_return_pct_now"] == NR and r["code_udr_now"] == NR and r["udr_plan_id"] == NR


def test_all_keys_once_1133_and_1134_are_live():
    frames = [{"ts": _ts(25, 22, 30), "ticker": "NVDA", "extra_json": _frame_1134()}]
    udrs = [{"symbol": "NVDA", "timestamp": _ts(25, 15), "signal": "HOLD", "held_udr": _udr_1134()}]
    earn = [_earn("2026-11-25", {"NVDA": _earn_1134(), "TSLA": _earn_1134(held=False)})]
    rows, book = hsr.build_rows(frames, udrs, earn)
    assert [(r["day"], r["ticker"]) for r in rows] == [("2026-11-25", "NVDA")]   # TSLA not held
    r = rows[0]
    assert r["sources"] == "frame/held_udr/earnings"
    assert (r["plan_id"], r["plan_revision"], r["udr_plan_id"], r["udr_plan_revision"]) == (41, 3, 41, 3)
    assert (r["invalidation_vs_close_pct"], r["target_vs_close_pct"], r["t1_vs_close_pct"]) == (-8.2, 12.5, 6.0)
    assert (r["open_risk_to_stop_frac"], r["open_risk_read"], r["gap_p5_frac"], r["gap_p1_frac"]) == \
        (0.0061, "ok", 0.0072, 0.0101)
    assert r["uncapped_weight_frac"] == NR                     # only on a row without a stop price
    assert (r["expected_return_pct_now"], r["expected_downside_pct_now"]) == (9.0, 5.0)
    assert (r["code_udr_read"], r["code_udr_now"], r["code_udr_level_source"]) == ("ok", 1.6, "t1")
    assert (r["earn_result"], r["catalyst_write"], r["cadence_median_days"]) == ("future_date", "ok", 91)
    assert r["est_next_print_from_cadence"] == "2027-02-17" and r["cadence_est_days"] == 84
    assert book[0]["book_open_risk_to_stop_frac"] == 0.0123 and book[0]["n_by_read"] == {"ok": 2}
    cov = {c["group"]: c for c in hsr.coverage(rows)}
    assert cov["held_udr"]["recorded"] == 1
    assert cov["open_risk"]["recorded"] == 1                  # uncapped_weight_frac is conditional
    assert cov["frame"]["recorded"] == 1 and cov["earnings"]["recorded"] == 1


def test_the_days_latest_rows_win_and_verdicts_are_counted():
    frames = [{"ts": _ts(25, 22, 30), "ticker": "NVDA", "extra_json": _frame_live(gain_pct=1.0)},
              {"ts": _ts(25, 23, 5), "ticker": "NVDA", "extra_json": _frame_live(gain_pct=2.0)}]
    udrs = [{"symbol": "NVDA", "timestamp": _ts(25, 14), "signal": "HOLD", "held_udr": _udr(udr_now=1.0)},
            {"symbol": "NVDA", "timestamp": _ts(25, 19), "signal": "TRIM", "held_udr": _udr(udr_now=0.7)}]
    r = _row(hsr.build_rows(frames, udrs, [])[0], "2026-11-25", "NVDA")
    assert (r["gain_pct"], r["udr_now"], r["udr_verdicts"], r["udr_signal"]) == (2.0, 0.7, 2, "TRIM")


def test_days_are_et_days():
    # 03:00 UTC on 11-26 is 22:00 EST on 11-25.
    udrs = [{"symbol": "ANET", "timestamp": _ts(26, 3), "signal": "HOLD", "held_udr": _udr()}]
    assert [(r["day"], r["ticker"]) for r in hsr.build_rows([], udrs, [])[0]] == [("2026-11-25", "ANET")]


def test_held_without_a_frame_row_comes_from_held_udr_or_the_earnings_check():
    udrs = [{"symbol": "IONQ", "timestamp": _ts(25, 15), "signal": "HOLD", "held_udr": _udr()},
            {"symbol": "AMD", "timestamp": _ts(25, 15), "signal": "WATCH",
             "held_udr": {"status": "no_open_lot", "open_lots": 0}}]
    earn = [_earn("2026-11-25", {"DRAM": {"result": "no_future_date", "held": True}})]
    rows, book = hsr.build_rows([], udrs, earn)
    assert [r["ticker"] for r in rows] == ["DRAM", "IONQ"]       # AMD: no open lot, not held
    d, i = rows
    assert d["sources"] == "earnings" and d["gain_pct"] == NO and d["earn_result"] == "no_future_date"
    assert d["est_next_print_from_cadence"] == NR and d["cadence_est_days"] == NR
    assert i["earn_result"] == NO                              # that day's row has no IONQ entry
    assert book == []


def test_the_markdown_shows_percent_of_nav_and_no_prices():
    frames = [{"ts": _ts(25, 22, 30), "ticker": "NVDA", "extra_json": _frame_1134()},
              {"ts": _ts(24, 22, 30), "ticker": "NVDA", "extra_json": _frame_live()}]
    rows, book = hsr.build_rows(frames, [], [_earn("2026-11-25", {"NVDA": _earn_1134()})])
    md = hsr.render(rows, book, "live", "2026-11-24", "2026-11-25")
    assert "0.61% NAV" in md and "1.23% NAV" in md and "-8.20%" in md
    assert NR in md and NO in md
    assert "$" not in md and "price" not in md.replace("next_print", "").replace("stop_at_or_above_price", "")
    assert md.count("## ") == 5


def test_a_wrong_port_reads_nothing(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert hsr.main([]) == 2
    assert "6543" in capsys.readouterr().err


class _Cur:
    def __init__(self, log):
        self.log, self.description = log, [("x",)]

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self):
        self.log, self.rolled_back = [], False

    def cursor(self):
        return _Cur(self.log)

    def rollback(self):
        self.rolled_back = True


def test_fetch_is_read_only_and_rolled_back():
    c = _Conn()
    assert hsr.fetch(c, "live", "2026-11-12", "2026-11-26") == ([], [], [])
    assert c.log[0] == "SET TRANSACTION READ ONLY" and c.log[1].startswith("SET LOCAL statement_timeout")
    assert c.rolled_back
    assert not [s for s in c.log if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", s)]
    lo, hi = hsr._bounds("2026-11-25", "2026-11-25")
    assert (lo.hour, hi.hour, (hi - lo).days) == (5, 5, 1)       # ET midnight to ET midnight


def test_weekly_readouts_does_not_load_it():
    with open(os.path.join(_ROOT, "weekly_readouts.py")) as f:
        assert "held_side_readout" not in f.read()


def test_coverage_of_a_group_recorded_nowhere_is_not_loaded():
    frames = [{"ts": _ts(24, 22, 30), "ticker": "NVDA", "extra_json": _frame_live()}]
    rows, book = hsr.build_rows(frames, [], [])
    cov = {c["group"]: c for c in hsr.coverage(rows)}
    assert cov["frame"]["recorded"] == 1 and cov["frame"]["no_row"] == 0
    for g in ("plan", "open_risk", "held_udr", "earnings"):
        assert (cov[g]["recorded"], cov[g]["not_recorded_yet"], cov[g]["no_row"]) == (hsr.NOT_LOADED,) * 3
    md = hsr.render(rows, book, "live", "2026-11-24", "2026-11-24")
    assert "| plan | 1 | not loaded | not loaded | not loaded |" in md
