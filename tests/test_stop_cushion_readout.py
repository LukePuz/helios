"""scripts/stop_cushion_readout.py (audit 2026-11-26 22:53 #3): row 21's
stop_drop_pct and ATR definitions, the close-window counts, the non-finite
handling, the percent-only report and export, and the read-only guards. No
DB; synthetic prices only."""
import importlib.util
import json
import math
import os
import types
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "stop_cushion_readout.py")
_spec = importlib.util.spec_from_file_location("stop_cushion_readout", _PATH)
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 26, 23, 0, tzinfo=NY)


def _bars(closes, start=date(2025, 6, 2), spread=1.0):
    out, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append({"date": d, "high": c + spread, "low": c - spread, "close": c})
        d += timedelta(days=1)
    return out


# ── definitions ─────────────────────────────────────────────────────────────

def test_atr_matches_risk_manager_compute_atr():
    from risk_manager import _compute_atr
    bars = _bars([100 + (i * 7) % 5 - 2 + i * 0.3 for i in range(40)], spread=1.5)
    a, read = src.atr(bars)
    hlc = [{"h": b["high"], "l": b["low"], "c": b["close"]} for b in bars[-src.ATR_ROWS:]]
    assert read == "ok" and a == _compute_atr(hlc)
    a14, _ = src.atr(bars, src.ATR14_ROWS)
    assert a14 == _compute_atr([{"h": b["high"], "l": b["low"], "c": b["close"]}
                                for b in bars[-src.ATR14_ROWS:]])
    assert src.ATR_ROWS == 19 and src.ATR14_ROWS == 15


def test_atr_reads():
    assert src.atr([])[1] == "too_few_bars" and src.atr(_bars([100]))[1] == "too_few_bars"
    bars = _bars([100.0] * 30)
    bars[-3]["low"] = float("nan")
    assert src.atr(bars) == (None, "nonfinite")
    assert src.atr(_bars([100.0] * 30, spread=0.0)) == (None, "zero")
    old_nan = _bars([100.0 + i for i in range(30)])
    old_nan[0]["close"] = Decimal("NaN")                     # outside the 19-row window
    assert src.atr(old_nan)[1] == "ok"


def test_name_record_row21_definitions():
    bars = _bars([100.0] * 30, spread=2.0)                   # every true range 4.0
    bars[-1]["close"] = 100.0
    r = src.name_record("AAA", 94.0, bars)
    assert r["stop_drop_pct"] == 6.0                         # (100 - 94) / 100 x 100
    assert r["atr_pct"] == 4.0 and r["stop_dist_atr"] == 1.5  # 6 / 4
    assert r["atr14_pct"] == 4.0 and r["stop_dist_atr14"] == 1.5
    assert not {"price", "stop_price", "atr", "close"} & set(r)
    # the newest finite close is the price
    bars[-1]["close"] = float("nan")
    bars[-2]["close"] = 100.0
    assert src.name_record("AAA", 94.0, bars)["last_close_date"] == str(bars[-2]["date"])


def test_name_record_notes():
    assert src.name_record("AAA", 101.0, _bars([100.0] * 30))["note"] == "price at or below the stop"
    assert src.name_record("AAA", 101.0, _bars([100.0] * 30))["windows"] is None
    assert src.name_record("AAA", None, _bars([100.0] * 30))["note"] == "no price or stop"
    assert src.name_record("AAA", 90.0, [])["note"] == "no price or stop"


# ── windows ─────────────────────────────────────────────────────────────────

def test_windows_counts():
    closes = [100.0, 95.0, 100.0, 100.0, 89.0, 100.0]
    w1 = src.windows(closes, 5.0, 1)
    # starts 0..4; next close at or below start x 0.95: 0->1 (95 <= 95) and 3->4
    assert w1 == {"k": 1, "n": 5, "fell": 2, "pct": 40.0, "skipped": 0}
    w5 = src.windows(closes, 5.0, 5)
    assert w5 == {"k": 5, "n": 1, "fell": 1, "pct": 100.0, "skipped": 0}
    assert src.windows(closes, 5.0, 10) == {"k": 10, "n": 0, "fell": 0, "pct": None, "skipped": 0}


def test_windows_skip_nonfinite_and_lookback():
    closes = [100.0, float("nan"), 100.0, 90.0, 100.0]
    w = src.windows(closes, 5.0, 1)
    assert w["skipped"] == 2 and w["n"] == 2 and w["fell"] == 1
    long = [100.0] * 400
    assert src.windows(long, 5.0, 1)["n"] == src.LOOKBACK - 1
    assert src.windows(long, 5.0, 10)["n"] == src.LOOKBACK - 10


def test_record_windows_use_todays_distance():
    closes = [100.0] * 260
    closes[100] = 93.0                                       # one 7% one-day drop
    bars = _bars(closes, spread=2.0)
    r = src.name_record("AAA", 94.0, bars)                   # 6% cushion
    w1, w5, w10 = r["windows"]
    assert (w1["fell"], w5["fell"], w10["fell"]) == (1, 5, 10)
    assert w1["n"] == src.LOOKBACK - 1


# ── report and export ───────────────────────────────────────────────────────

def _recs():
    closes = [100.0] * 260
    closes[100] = 93.0
    return src.build([("AAA", Decimal("94.0")), ("BBB", Decimal("120.0"))],
                     {"AAA": _bars(closes, spread=2.0), "BBB": _bars([100.0] * 30)})


def test_render_percent_only():
    text = src.render(_recs(), "live", "abc1234", NOW, "non-finite bars: 0")
    assert src.CAVEAT in text and src.NO_RULE in text and "non-finite bars: 0" in text
    assert "| AAA |" in text and "| 6.00 | 4.00 | 1.50 | 4.00 | 1.50 | ok |" in text
    assert "1 / 251 (0.4%)" in text and "price at or below the stop" in text
    assert "$" not in text and "94" not in text and "120" not in text
    for word in ("at risk", "should"):              # NO_RULE itself says "no ... recommendation"
        assert word not in text.lower()
    lines = text.splitlines()
    header = next(l for l in lines if l.startswith("| name |"))
    assert header.count("|") == lines[lines.index(header) + 1].count("|")
    assert header.count("|") == next(l for l in lines if l.startswith("| AAA")).count("|")
    empty = src.render([], "live", "x", NOW, "non-finite bars: 0")
    assert empty.splitlines()[-1].count("|") == header.count("|")


# ── main ────────────────────────────────────────────────────────────────────

def test_quiet_window_and_db_root(tmp_path):
    assert src.in_quiet_window(datetime(2026, 11, 27, 4, 50, tzinfo=NY))
    assert not src.in_quiet_window(datetime(2026, 11, 27, 5, 12, tzinfo=NY))
    assert src.db_root(str(tmp_path)) == src.ROOT
    (tmp_path / "db.py").write_text("")
    assert src.db_root(str(tmp_path)) == str(tmp_path)


def test_main_refuses_other_ports_and_the_quiet_window(capsys):
    with patch.dict(os.environ, {"SUPABASE_PORT": "5432"}):
        assert src.main([]) == 2
    fake_db = types.SimpleNamespace(_get_conn=MagicMock())
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543", "SUPABASE_HOST": "h"}), \
         patch.dict("sys.modules", {"db": fake_db}), \
         patch.object(src, "in_quiet_window", return_value=True):
        assert src.main([]) == 2
    fake_db._get_conn.assert_not_called()
    err = capsys.readouterr().err
    assert "6543" in err and "04:50-05:12" in err


def test_main_stops_without_a_host(capsys):
    fake_db = types.SimpleNamespace(__file__="/x/db.py", _get_conn=MagicMock())
    env = {k: v for k, v in os.environ.items() if k != "SUPABASE_HOST"}
    env["SUPABASE_PORT"] = "6543"
    with patch.dict(os.environ, env, clear=True), patch.dict("sys.modules", {"db": fake_db}), \
         patch.object(src, "in_quiet_window", return_value=False):
        assert src.main([]) == 2
    fake_db._get_conn.assert_not_called()
    assert "/x/db.py" in capsys.readouterr().err


def test_main_reads_once_read_only_and_rolls_back(tmp_path):
    closes = [100.0] * 30
    rows = [("AAA", b["date"], Decimal(str(b["high"])), Decimal(str(b["low"])),
             Decimal("NaN") if i == 3 else Decimal(str(b["close"])))
            for i, b in enumerate(_bars(closes, spread=2.0))]
    cur = MagicMock()
    cur.fetchall.side_effect = [[("AAA", Decimal("94.0"))], rows]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    fake_db = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=MagicMock())
    out, ex = tmp_path / "r.md", tmp_path / "r.json"
    # A fixed head: the real one is a hex hash that can hold the fixture's digits.
    head = types.SimpleNamespace(stdout="headfix\n")
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543", "SUPABASE_HOST": "h"}), \
         patch.dict("sys.modules", {"db": fake_db}), \
         patch.object(src, "in_quiet_window", return_value=False), \
         patch.object(src.subprocess, "run", return_value=head):
        assert src.main(["--out", str(out), "--export", str(ex)]) == 0
    sql = [c.args[0] for c in cur.execute.call_args_list]
    assert sql[0] == "SET TRANSACTION READ ONLY" and sql[2:] == [src._STOPS_SQL, src._BARS_SQL]
    assert cur.execute.call_args_list[3].args[1] == {"tickers": ["AAA"], "rows": src.BARS_READ}
    for q in sql[2:]:
        assert not {"insert", "update", "delete"} & set(q.lower().split())
    conn.rollback.assert_called_once()
    fake_db._put_conn.assert_called_once_with(conn)
    text = out.read_text()
    assert "code headfix." in text
    assert f"non-finite bars: 1 (AAA {rows[3][1]})" in text
    # No stop digits in the rows or the flag line (the header carries the head and the clock).
    body = [l for l in text.splitlines() if l.startswith("|") or l.startswith("non-finite")]
    assert body and not any("94" in l for l in body)
    data = json.loads(ex.read_text())
    assert data["nonfinite"]["n"] == 1 and data["names"][0]["stop_drop_pct"] == 6.0
    assert data["names"][0]["windows"][0]["skipped"] == 2       # the NaN close's two windows
    assert "94" not in ex.read_text().replace("2026", "")
