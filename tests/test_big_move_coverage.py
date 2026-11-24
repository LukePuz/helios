"""scripts/big_move_coverage.py: big-move days and what the screener and
analyst did in each session's close-to-close window. Pure parts plus the
read's shape; no DB."""
import importlib.util
import os
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NY = ZoneInfo("America/New_York")


def _mod():
    spec = importlib.util.spec_from_file_location("big_move_coverage_t",
                                                  os.path.join(ROOT, "scripts", "big_move_coverage.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bmc = _mod()


def _bars(closes, start=date(2026, 10, 1), spread=1.0):
    """Weekday bars, high/low = close +/- spread."""
    out, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append({"date": d, "high": c + spread, "low": c - spread, "close": c})
        d += timedelta(days=1)
    return out


def test_atr_matches_dip_review():
    import tasks
    bars = _bars([100 + (i % 3) for i in range(20)])
    assert bmc._atr14(bars) == pytest.approx(tasks._atr14(bars))
    assert bmc._atr14(bars[:14]) is None and tasks._atr14(bars[:14]) is None
    m = tasks._dip_metrics(bars[-1]["close"], bars[-2]["close"], bars[:-1])
    row = [r for r in bmc.day_moves({"X": bars}, bars[-1]["date"], bars[-1]["date"])][0]
    assert row["atr_mult"] == m["atr_mult"] and row["move_pct"] == m["move_pct"]


def test_day_moves_uses_only_earlier_sessions_and_skips_missing():
    bars = _bars([100.0] * 16 + [110.0])
    rows = bmc.day_moves({"X": bars}, bars[0]["date"], bars[-1]["date"])
    assert rows[0]["atr_mult"] is None                    # under 15 earlier sessions
    last = rows[-1]
    assert last["move_pct"] == 10.0 and last["atr_pct"] == 2.0 and last["atr_mult"] == 5.0
    assert last["prev_date"] == bars[-2]["date"]
    bad = bars[:-1] + [{**bars[-1], "high": None}]
    assert bmc.day_moves({"X": bad}, bars[-1]["date"], bars[-1]["date"]) == []


def _ts(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=NY)


def test_signals_count_in_the_close_to_close_window():
    bars = _bars([100.0] * 16 + [110.0, 100.0])
    d1, d2 = bars[-2]["date"], bars[-1]["date"]
    prev = bars[-3]["date"]
    moves = bmc.day_moves({"X": bars}, d1, d2)
    sigs = [{"ticker": "X", "ts": _ts(prev, 16, 30), "kind": "out", "own_ticker_rejected": True},   # d1: after prev close
            {"ticker": "X", "ts": _ts(d1, 16, 0), "kind": "out", "own_ticker_rejected": False},     # d1: at the close
            {"ticker": "X", "ts": _ts(d1, 16, 1), "kind": "in", "own_ticker_rejected": "absent"},   # d2
            {"ticker": "X", "ts": _ts(d2, 9, 45), "kind": "verdict", "own_ticker_rejected": "absent"},
            {"ticker": "X", "ts": _ts(d2, 17, 0), "kind": "out", "own_ticker_rejected": True},      # after the last close
            {"ticker": "X", "ts": _ts(prev, 15, 0), "kind": "out", "own_ticker_rejected": True},    # before the first window
            {"ticker": "Y", "ts": _ts(d1, 12, 0), "kind": "out", "own_ticker_rejected": True}]
    res = bmc.coverage(moves, sigs, k=2)
    r1, r2 = res["rows"]
    assert (r1["date"], r1["n_items"], r1["n_screened_out"], r1["n_own_ticker_rejected"],
            r1["n_own_ticker_recorded"], r1["verdict"], r1["all_screened_out"]) == (
        d1.isoformat(), 2, 2, 1, 2, False, True)
    assert (r2["n_items"], r2["n_screened_in"], r2["verdict"], r2["all_screened_out"]) == (1, 1, True, False)


def test_k_is_compared_unrounded_and_stored():
    moves = [{"ticker": "A", "date": date(2026, 11, 5), "prev_date": date(2026, 11, 4), "move_pct": -10.8,
              "atr_pct": 5.4, "atr_mult": -2.0, "atr_mult_raw": -1.9968},
             {"ticker": "B", "date": date(2026, 11, 5), "prev_date": date(2026, 11, 4), "move_pct": 3.0,
              "atr_pct": 1.0, "atr_mult": 3.0, "atr_mult_raw": 3.0}]
    res = bmc.coverage(moves, [], k=2)
    assert res["k"] == 2 and [r["ticker"] for r in res["rows"]] == ["B"] and res["n_ordinary"] == 1
    assert bmc.coverage(moves, [], k=3.5)["n_big"] == 0


def test_partition_and_summary():
    d, p = date(2026, 11, 5), date(2026, 11, 4)
    mk = lambda t, mult: {"ticker": t, "date": d, "prev_date": p, "move_pct": mult, "atr_pct": 1.0,
                          "atr_mult": mult, "atr_mult_raw": mult}
    moves = [mk("V", 3), mk("O", -3), mk("N", 2.5), mk("I", 2.2), mk("Q", 0.5), mk("R", 1.0)]
    s = lambda t, kind, h=11: {"ticker": t, "ts": _ts(d, h), "kind": kind, "own_ticker_rejected": "absent"}
    sigs = [s("V", "out"), s("V", "verdict"), s("O", "out"), s("O", "out"), s("I", "in"), s("R", "verdict")]
    res = bmc.coverage(moves, sigs, k=2)
    assert res["n_big"] == 4 and res["n_big_verdict"] == 1
    assert (res["n_big_no_verdict_all_screened_out"], res["n_big_no_verdict_no_items"],
            res["n_big_no_verdict_screened_in"]) == (1, 1, 1)
    assert res["n_big_verdict"] + 1 + 1 + 1 == res["n_big"]
    assert res["n_big_all_screened_out"] == 2 and res["n_big_all_screened_out_with_verdict"] == 1
    assert (res["n_ordinary"], res["n_ordinary_verdict"], res["ordinary_verdict_pct"]) == (2, 1, 50.0)


def test_render_has_no_prices():
    d, p = date(2026, 11, 5), date(2026, 11, 4)
    res = bmc.coverage([{"ticker": "APP", "date": d, "prev_date": p, "move_pct": 31.04, "atr_pct": 4.7,
                         "atr_mult": 6.58, "atr_mult_raw": 6.58}], [], k=2)
    text = bmc.render(res, d, d, 75)
    assert "| 2026-11-05 | APP | +31.0 | +6.6 |" in text and "$" not in text
    assert "k = 2 ATR" in text and "window:" in text


def test_fetch_is_read_only_and_rolls_back():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    ts = _ts(date(2026, 11, 5), 10)
    cur.fetchall.side_effect = [[("APP", date(2026, 11, 5), 10, 9, 9.5)],
                                [("APP", ts, "SCREENED_OUT", "true", False),
                                 ("APP", ts, "SCREENED_OUT", "absent", False),
                                 ("APP", ts, "HOLD", "absent", True)]]
    bars, sigs = bmc.fetch(conn, "live", ["APP"], date(2026, 11, 5), date(2026, 11, 5))
    sql = [c.args[0] for c in cur.execute.call_args_list]
    assert sql[0] == "SET TRANSACTION READ ONLY" and sql[1].startswith("SET LOCAL statement_timeout")
    assert "FROM price_history" in sql[2] and "FROM signals" in sql[3]
    assert "reasoning" not in sql[3] and "title" not in sql[3]
    conn.rollback.assert_called_once()
    assert bars["APP"][0]["close"] == 9.5
    assert [(s["kind"], s["own_ticker_rejected"]) for s in sigs] == [("out", True), ("out", "absent"),
                                                                       ("verdict", "absent")]


def test_main_refuses_the_session_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert bmc.main(["--start", "2026-11-05", "--end", "2026-11-05"]) == 2
    assert "6543" in capsys.readouterr().err
