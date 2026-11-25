"""scripts/entry_gate_readout.py: blocked vs passed by gate, the market read,
and the blocked names' forward moves minus SPY at 5 and 20 sessions, with n
on every figure and 'insufficient' below 5. Read-only, port 6543 only."""
import importlib.util
import os
import sys
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "scripts", "entry_gate_readout.py")
_spec = importlib.util.spec_from_file_location("entry_gate_readout_t", _P)
egr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(egr)

UTC = timezone.utc


def _ts(d: date, hour_utc: int = 15) -> datetime:
    return datetime(d.year, d.month, d.day, hour_utc, 0, tzinfo=UTC)


# Row shapes: the live tree (fcad43c) writes stopout/trend only; batch-1124
# adds path, approval_id and add_loser.
def _live(stopout=False, trend=False, market=None, below=(True, True)):
    return {"route_exit": "ENTRY_GATE_BLOCKED", "verdict_action": "OPEN", "entry_gate": {
        "stopout": {"days": 10, "read": "ok", "hours_since": 30.0, "blocked": stopout},
        "trend": {"read": "ok", "bar_age_days": 1, "max_bar_age_days": 5,
                  "below_sma50": below[0], "below_sma200": below[1], "pct_vs_sma50": -3.0,
                  "pct_vs_sma200": -6.0, "market": market, "blocked": trend}}}


def _mkt(down=True, read="ok", below=True):
    return {"ticker": "SPY", "read": read, "bar_age_days": 1, "below_sma200": below,
            "pct_vs_sma200": -2.0 if below else 3.0, "downtrend": down}


def _batch(path=None, add_loser=None, **k):
    d = _live(**k)
    if path:
        d["entry_gate"]["path"] = path
        d["entry_gate"]["approval_id"] = 7
    if add_loser is not None:
        d["entry_gate"]["add_loser"] = {"blocked": add_loser}
    return d


# ---- rows ----

@pytest.mark.parametrize("dec, want", [
    (_live(stopout=True), "stopout"),
    (_live(trend=True, market=_mkt()), "trend"),
    (_live(stopout=True, trend=True, market=_mkt()), "stopout+trend"),
    (_batch(add_loser=True), "add_loser"),
    (_batch(stopout=True, add_loser=True), "stopout+add_loser"),
    (_live(), "unknown"),
    ({"entry_gate": "garbage"}, "unknown"),
    (None, "unknown"),
])
def test_reason_of(dec, want):
    assert egr.reason_of(dec) == want


def test_path_of():
    assert egr.path_of(_live(stopout=True)) == "routed buy"
    assert egr.path_of(_batch(path="deferred_retry", stopout=True)) == "deferred_retry"
    assert egr.path_of(None) == "routed buy"


@pytest.mark.parametrize("dec, want", [
    (_live(trend=True, market=_mkt()), "trend blocked: market below its 200-day"),
    (_live(trend=True, market=None), "trend blocked, no market check"),
    (_live(stopout=True, market=_mkt(down=False, below=False)),
     "stood down: market above its 200-day"),
    (_live(stopout=True, market=_mkt(down=False, read="error: RuntimeError", below=None)),
     "stood down: market read failed"),
    (_live(stopout=True, market=_mkt(down=False, below=True)),       # stale bar
     "stood down: market read stale or incomplete"),
    (_live(stopout=True, market=None), "stock below both, no market check"),
    (_live(stopout=True, below=(True, False)), None),                 # not below both
    ({"entry_gate": {"stopout": {"blocked": True}}}, None),           # no trend half
])
def test_market_read_of(dec, want):
    assert egr.market_read_of(dec) == want


def test_session_range_skips_weekends_and_holidays():
    s = egr.session_range(date(2026, 11, 23), date(2026, 11, 30))
    assert s == [date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25),
                 date(2026, 11, 27), date(2026, 11, 30)]         # 11-26 Thanksgiving
    assert egr.session_range(date(2026, 11, 28), date(2026, 11, 29)) == []


# ---- tally ----

def test_tally_counts_by_session_reason_path_and_names():
    s = egr.session_range(date(2026, 11, 23), date(2026, 11, 24))
    rows = [("AMD", "ENTRY_GATE_BLOCKED", _ts(date(2026, 11, 23)), _live(stopout=True)),
            ("AMD", "ENTRY_GATE_BLOCKED", _ts(date(2026, 11, 23), 18), _live(stopout=True)),
            ("MU", "ENTRY_GATE_BLOCKED", _ts(date(2026, 11, 23)),
             _live(trend=True, market=_mkt())),
            ("KLAC", "ENTRY_GATE_BLOCKED", _ts(date(2026, 11, 24)),
             _batch(path="deferred_retry", stopout=True, trend=True, market=_mkt())),
            ("OLD", "ENTRY_GATE_BLOCKED", _ts(date(2026, 11, 20)), _live(stopout=True))]
    funnel = {date(2026, 11, 23): {"passed": 2}, date(2026, 11, 24): {"passed": 0}}
    t = egr.tally(s, rows, funnel)
    d23, d24 = t["sessions"][date(2026, 11, 23)], t["sessions"][date(2026, 11, 24)]
    assert d23["blocked"] == 3 and d23["reasons"] == {"stopout": 2, "trend": 1}
    assert d23["names"] == {"AMD", "MU"} and d23["passed"] == 2
    assert d24["reasons"] == {"stopout+trend": 1} and d24["paths"] == {"deferred_retry": 1}
    assert t["all"]["blocked"] == 4 and t["all"]["passed"] == 2       # OLD is outside
    assert t["all"]["paths"] == {"routed buy": 3, "deferred_retry": 1}
    assert t["market"] == {"trend blocked: market below its 200-day": 2,
                           "stock below both, no market check": 2}


def test_a_weekend_row_counts_toward_monday():
    s = egr.session_range(date(2026, 11, 23), date(2026, 11, 23))
    rows = [("AMD", "ENTRY_GATE_BLOCKED", _ts(date(2026, 11, 22)), _live(stopout=True))]
    t = egr.tally(s, rows, {})
    assert t["sessions"][date(2026, 11, 23)]["blocked"] == 1


# ---- formatting ----

def test_every_figure_carries_n_and_is_insufficient_below_five():
    assert egr.share(1, 4) == "insufficient (n=4)"
    assert egr.share(2, 5) == "40% (n=5)"
    assert egr.share(0, 0) == "n/a (n=0)"
    assert egr.move([0.01] * 4) == "insufficient (n=4)"
    assert egr.move([0.01, 0.02, 0.03, 0.04, 0.10]) == "mean +4.0%, median +3.0% (n=5)"
    assert egr.MIN_N == 5 and egr.READOUT_HORIZONS == (5, 20)


# ---- forward moves ----

def _sessions(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(days, spy_step=0.0, names=None):
    """SPY rising spy_step per session; each name: (d0 index, move at +5, move at +20)."""
    bars = {"SPY": [(d, 100.0 * (1 + spy_step) ** i) for i, d in enumerate(days)]}
    for name, (i0, m5, m20) in (names or {}).items():
        series = []
        for i, d in enumerate(days):
            c = 50.0
            if i >= i0 + 5:
                c = 50.0 * (1 + m5)
            if i >= i0 + 20:
                c = 50.0 * (1 + m20)
            series.append((d, c))
        bars[name] = series
    return bars


def test_forward_moves_are_minus_spy_over_the_same_dates_and_deduped():
    days = _sessions(date(2026, 11, 2), 30)
    d0 = days[2]
    bars = _bars(days, spy_step=0.0, names={"AMD": (2, 0.10, -0.20)})
    bars["SPY"] = [(d, 100.0 if i < 7 else 101.0) for i, d in enumerate(days)]  # +1% by d0+5
    rows = [("AMD", "ENTRY_GATE_BLOCKED", _ts(d0), _live(stopout=True)),
            ("AMD", "ENTRY_GATE_BLOCKED", _ts(d0, 17), _live(stopout=True))]   # same session 0
    g = egr.forward_groups(rows, bars)
    assert set(g) == {"stopout", "all blocked"}
    (rec,) = g["stopout"]
    assert rec["d0"] == d0
    assert rec["x5"] == pytest.approx(0.10 - 0.01)
    assert rec["x20"] == pytest.approx(-0.20 - 0.01)


def test_a_row_after_the_close_starts_at_the_next_session():
    days = _sessions(date(2026, 11, 2), 30)
    rows = [("AMD", "ENTRY_GATE_BLOCKED", _ts(days[2], 22), _live(stopout=True))]  # 17:00 ET
    g = egr.forward_groups(rows, _bars(days, names={"AMD": (3, 0.1, 0.1)}))
    assert g["stopout"][0]["d0"] == days[3]


def test_readout_insufficient_pending_and_only_the_given_horizons():
    days = _sessions(date(2026, 10, 1), 40)
    names = {f"N{i}": (i, 0.05 * (i + 1), 0.01) for i in range(6)}
    bars = _bars(days, names=names)
    rows = [(n, "ENTRY_GATE_BLOCKED", _ts(days[i0]), _live(stopout=True))
            for n, (i0, _a, _b) in names.items()]
    rows.append(("MU", "ENTRY_GATE_BLOCKED", _ts(days[0]),
                 _live(trend=True, market=_mkt())))                   # no bars: pending
    s = egr.session_range(days[0], days[5])
    t = egr.tally(s, rows, {d: {"passed": 1} for d in s})
    text = egr.readout(s, t, egr.forward_groups(rows, bars), "abc1234",
                       datetime(2026, 11, 30, 21, tzinfo=UTC), days[-1])
    stop = next(ln for ln in text.splitlines() if ln.startswith("| stopout |"))
    assert "mean +17.5%, median +17.5% (n=6)" in stop                  # 5..30% at +5
    trend = next(ln for ln in text.splitlines() if ln.startswith("| trend |"))
    assert "insufficient (n=0) | 1 | insufficient (n=0) | 1 |" in trend
    assert "| 5 sessions | pending at 5 | 20 sessions | pending at 20 |" in text
    assert "10 sessions" not in text
    assert "$" not in text and "abc1234" in text
    assert "| all | 7 | 6 | 1 | 7 | 6 | 54% (n=13) |" in text           # blocked vs passed
    assert f"Price cache through {days[-1]}" in text


def test_readout_with_no_rows():
    s = egr.session_range(date(2026, 11, 23), date(2026, 11, 24))
    text = egr.readout(s, egr.tally(s, [], {}), {}, "h", datetime(2026, 11, 24, tzinfo=UTC), None)
    assert "| all | 0 | 0 | 0 | n/a (n=0) |" in text
    assert "No row had a stock below both averages." in text
    assert "(no blocked rows with a cached session 0)" in text


# ---- I/O ----

class _Cur:
    def __init__(self, results, fail_at=None):
        self.sql, self.params, self._results, self._fail_at = [], [], list(results), fail_at

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if self._fail_at is not None and len(self.sql) == self._fail_at:
            raise RuntimeError("db")
        self.sql.append(sql)
        self.params.append(params)

    def fetchall(self):
        return self._results.pop(0)


def test_fetch_is_read_only_live_only_and_rolls_back():
    d = date(2026, 11, 23)
    cur = _Cur([[("AMD", "ENTRY_GATE_BLOCKED", _ts(d), _live(stopout=True))],
                [("AMD", d, 50.0), ("SPY", d, 100.0)]])
    conn = MagicMock()
    conn.cursor.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        rows, bars = egr._fetch([d])
    assert cur.sql[0] == "SET TRANSACTION READ ONLY" and "statement_timeout" in cur.sql[1]
    assert cur.params[2][0] == "live" and "ENTRY_GATE_BLOCKED" in cur.sql[2]
    assert cur.params[3][0] == ["AMD", "SPY"]
    assert not any(w in s.upper() for s in cur.sql for w in ("INSERT", "UPDATE", "DELETE"))
    conn.rollback.assert_called_once()
    put.assert_called_once_with(conn)
    assert len(rows) == 1 and set(bars) == {"AMD", "SPY"}


def test_fetch_rolls_back_on_a_failed_read():
    conn = MagicMock()
    conn.cursor.return_value = _Cur([], fail_at=2)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put, \
         pytest.raises(RuntimeError):
        egr._fetch([date(2026, 11, 23)])
    conn.rollback.assert_called_once()
    put.assert_called_once_with(conn)


def test_main_refuses_without_the_pooler_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    with patch.object(egr, "_fetch") as f:
        assert egr.main(["--until", "2026-11-24"]) == 2
    f.assert_not_called()
    assert "SUPABASE_PORT=6543" in capsys.readouterr().err


def test_main_rejects_a_window_without_sessions(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    with patch.object(egr, "_fetch") as f:
        assert egr.main(["--since", "2026-11-28", "--until", "2026-11-29"]) == 2
    f.assert_not_called()


def test_main_runs_the_readout(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    d = date(2026, 11, 23)
    rows = [("AMD", "ENTRY_GATE_BLOCKED", _ts(d), _live(stopout=True))]
    ef = egr._ef()
    out = tmp_path / "r.md"
    with patch.object(egr, "_fetch", return_value=(rows, {"SPY": [(d, 100.0)]})), \
         patch.object(egr, "_fetch_reclaims", return_value=[]), \
         patch.object(ef, "run", return_value={d: {"passed": 3}}) as run, \
         patch.object(ef, "_git_head", return_value="abc1234"):
        assert egr.main(["--since", "2026-11-23", "--until", "2026-11-23",
                         "--out", str(out)]) == 0
    run.assert_called_once_with([d], "live")
    text = capsys.readouterr().out
    assert "| all | 1 | 1 | 1 | 3 | insufficient (n=4) |" in text
    assert out.read_text() == text


def test_importing_the_script_touches_no_db():
    assert "db" not in egr.__dict__
    assert egr.ENV == "live" and egr.READ_PORT == 6543


# ---- the reclaim shadow section (record only) ----

def _rc(ticker, blocked_at, event, gates=("trend",), sessions=None, env="live"):
    return {"shadow_key": f"{env}:{ticker}:{blocked_at}:{event}", "event": event,
            "gates": list(gates), **({"sessions": sessions} if sessions is not None else {})}


def test_reclaim_tally_groups_records_by_gates_and_skips_other_envs():
    rows = [_rc("NVDA", "t1", "forward_5"), _rc("NVDA", "t1", "reclaim", sessions=3),
            _rc("NVDA", "t1", "expired"),
            _rc("AVGO", "t2", "forward_5", gates=("stopout", "trend")),
            _rc("AVGO", "t2", "expired", gates=("stopout", "trend")),
            _rc("AMD", "t3", "reclaim", sessions=2),
            _rc("MU", "t4", "reclaim", sessions=1, env="paper")]
    out = egr.reclaim_tally(rows)
    assert out["trend"] == {"records": 2, "reclaimed": 2, "sessions": [3, 2], "closed": 1,
                            "closed_unreclaimed": 0, "superseded": 0}
    assert out["stopout+trend"] == {"records": 1, "reclaimed": 0, "sessions": [], "closed": 1,
                                    "closed_unreclaimed": 1, "superseded": 0}
    assert "MU" not in str(out)


def test_reclaim_tally_counts_superseded_records_and_dedupes_by_shadow_key():
    # A spill replay or a skipped book update can repeat a row: one shadow_key counts once
    # (the first row); a replaced record counts as superseded beside reclaimed and closed.
    t1 = "2026-11-23T15:00:00+00:00"
    rows = [_rc("NVDA", t1, "reclaim", sessions=3), _rc("NVDA", t1, "reclaim", sessions=9),
            _rc("NVDA", t1, "superseded"), _rc("NVDA", t1, "superseded"),
            _rc("NVDA", "2026-11-24T15:00:00+00:00", "forward_5"),
            _rc("AMD", "t3", "expired"), _rc("AMD", "t3", "expired")]
    out = egr.reclaim_tally(rows)
    assert out["trend"] == {"records": 3, "reclaimed": 1, "sessions": [3], "closed": 1,
                            "closed_unreclaimed": 1, "superseded": 1}


def test_reclaim_section_prints_counts_and_sessions_only():
    s = [date(2026, 11, 23)]
    rc = egr.reclaim_tally([_rc(f"T{i}", "t", "reclaim", sessions=i + 1) for i in range(5)])
    text = egr.readout(s, egr.tally(s, [], {}), {}, "h", datetime(2026, 11, 24, tzinfo=UTC), None, rc)
    assert "## Reclaim shadow (record only)" in text
    assert "| trend | 5 | 5 | median 3 (n=5) | 0 | 0 | 0 |" in text
    assert "which horizon an evaluation uses is a research choice (Luke)" in text
    assert "Which blocks an evaluation counts (first, latest, all) is a research choice (Luke)" in text
    doc = " ".join(egr.__doc__.split())
    assert "which horizon an evaluation uses is a research choice (Luke)" in doc
    assert "superseded rows keep every block countable" in doc
    assert "$" not in text.split("## Reclaim shadow")[1]
    plain = egr.readout(s, egr.tally(s, [], {}), {}, "h", datetime(2026, 11, 24, tzinfo=UTC), None)
    assert "Reclaim shadow" not in plain           # the existing output is unchanged without it


def test_main_reports_an_unreadable_reclaim_read_and_still_prints(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    d = date(2026, 11, 23)
    ef = egr._ef()
    with patch.object(egr, "_fetch", return_value=([], {"SPY": [(d, 100.0)]})), \
         patch.object(egr, "_fetch_reclaims", side_effect=RuntimeError("db down")), \
         patch.object(ef, "run", return_value={d: {"passed": 0}}), \
         patch.object(ef, "_git_head", return_value="abc1234"):
        assert egr.main(["--since", "2026-11-23", "--until", "2026-11-23"]) == 0
    text = capsys.readouterr().out
    assert "## Blocked vs passed" in text and "Reclaim rows not read (RuntimeError)." in text


def test_main_prints_the_reclaim_rows(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    d = date(2026, 11, 23)
    ef = egr._ef()
    with patch.object(egr, "_fetch", return_value=([], {"SPY": [(d, 100.0)]})), \
         patch.object(egr, "_fetch_reclaims", return_value=[_rc("NVDA", "t1", "reclaim", sessions=4)]) as fr, \
         patch.object(ef, "run", return_value={d: {"passed": 0}}), \
         patch.object(ef, "_git_head", return_value="abc1234"):
        assert egr.main(["--since", "2026-11-23", "--until", "2026-11-23"]) == 0
    fr.assert_called_once_with(d)
    assert "| trend | 1 | 1 | insufficient (n=1) | 0 | 0 | 0 |" in capsys.readouterr().out
