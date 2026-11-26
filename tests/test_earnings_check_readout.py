"""scripts/earnings_check_readout.py on fixture rows: run fields, per-ticker
tallies, "not recorded yet" for row 5's keys on a row-2-only row, missing
weekdays and duplicate days, tickers dropped from the latest row, "not
loaded" before the stage loads, the port check, the read-only fetch, and that
weekly_readouts doesn't load it. No DB."""
import importlib.util
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "earnings_check_readout", os.path.join(_ROOT, "scripts", "earnings_check_readout.py"))
ecr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ecr)

NR = ecr.NOT_RECORDED


def _row(day, tickers, row5=True, **kw):
    ex = {"day": day, "env": "live", "checked": len(tickers),
          "counts": {"future_date": sum(1 for c in tickers.values() if c["result"] == "future_date"),
                     "no_future_date": sum(1 for c in tickers.values() if c["result"] == "no_future_date"),
                     "error": sum(1 for c in tickers.values() if c["result"] == "error")},
          "tickers": tickers, "held_read": "ok", "held_no_future": [], "warned": []}
    if row5:
        ex.update(catalyst_write_counts={"ok": 1, "no_event": 0, "error": 0}, cadence_estimated=1)
    ex.update(kw)
    return {"ts": None, "extra_json": ex}


def _c(result="future_date", row5=True, **kw):
    c = {"result": result, "next_date": None, "error_at": None, "held": False}
    if row5:
        c.update(catalyst_write="ok" if result == "future_date" else None, cadence_none_reason=None)
    c.update(kw)
    return c


def test_tallies_and_missing_rows():
    rows = [_row("2026-12-01", {"NVDA": _c(), "DRAM": _c("no_future_date", cadence_none_reason="no_prints"),
                                "OLD": _c()}),
            _row("2026-12-02", {"NVDA": _c(catalyst_write="error"), "DRAM": _c("no_future_date"),
                                "AMD": _c("error", error_at="read", catalyst_write=None)}),
            _row("2026-12-02", {"NVDA": _c(), "DRAM": _c("no_future_date"), "AMD": _c()})]
    res = ecr.build(rows, rows_ever=3, since="2026-11-30", until="2026-12-04")
    t = res["totals"]
    assert res["loaded"] and (t["rows"], t["ticker_checks"]) == (3, 9)
    assert t["result"] == {"error": 1, "future_date": 5, "no_future_date": 3}
    assert t["error_at"] == {"read": 1}
    assert t["catalyst_write"] == {"error": 1, "n/a": 4, "ok": 4}
    assert t["cadence_none_reason"] == {"n/a": 8, "no_prints": 1}
    assert t["latest_no_future_date"] == ["DRAM"] and t["catalyst_write_error_tickers"] == ["NVDA"]
    assert t["weekdays_without_row"] == ["2026-11-30", "2026-12-03", "2026-12-04"]
    assert t["days_with_several_rows"] == ["2026-12-02"] and t["dropped_from_latest"] == ["OLD"]
    assert [r["day"] for r in res["runs"]] == ["2026-12-01", "2026-12-02", "2026-12-02"]


def test_row5_keys_on_a_row2_only_row_read_not_recorded_yet():
    rows = [_row("2026-12-01", {"NVDA": _c(row5=False)}, row5=False)]
    res = ecr.build(rows, rows_ever=1, since="2026-12-01", until="2026-12-01")
    assert res["totals"]["catalyst_write"] == NR and res["totals"]["cadence_none_reason"] == NR
    assert res["runs"][0]["catalyst_write_counts"] == NR and res["runs"][0]["cadence_estimated"] == NR
    assert res["totals"]["result"] == {"future_date": 1}
    md = ecr.render(res, "live", "2026-12-01", "2026-12-01")
    assert "| catalyst_write | not recorded yet |" in md


def test_not_loaded_prints_not_loaded_not_zero():
    res = ecr.build([], rows_ever=0, since="2026-11-12", until="2026-11-26")
    assert res["loaded"] is False and set(res["totals"].values()) == {ecr.NOT_LOADED}
    md = ecr.render(res, "live", "2026-11-12", "2026-11-26")
    assert "earnings_calendar_check: not loaded" in md and "| rows | not loaded |" in md
    assert "| rows | 0 |" not in md


def test_loaded_with_no_row_in_the_window_is_a_real_zero():
    res = ecr.build([], rows_ever=4, since="2026-11-30", until="2026-11-30")
    assert res["totals"]["rows"] == 0 and res["totals"]["weekdays_without_row"] == ["2026-11-30"]


def test_a_wrong_port_reads_nothing(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert ecr.main([]) == 2
    assert "6543" in capsys.readouterr().err


class _Cur:
    def __init__(self, log):
        self.log, self.description = log, [("x",)]

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchall(self):
        return []

    def fetchone(self):
        return (0,)

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
    assert ecr.fetch(c, "live", "2026-11-12", "2026-11-26") == ([], 0)
    assert c.log[0] == "SET TRANSACTION READ ONLY" and c.log[1].startswith("SET LOCAL statement_timeout")
    assert c.rolled_back
    assert not [s for s in c.log if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", s)]


def test_weekly_readouts_does_not_load_it():
    with open(os.path.join(_ROOT, "weekly_readouts.py")) as f:
        assert "earnings_check_readout" not in f.read()
