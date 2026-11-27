"""scripts/plan_target_hit_readout.py (audit 2026-11-26 20:53 #1): the plan
life window, the first close hit, the forward path and giveback, the verdict
periods, the held filter, the percent-only report and the read-only guard.
No DB. Prices here are synthetic (entry 100, target 110)."""
import importlib.util
import json
import os
import types
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "plan_target_hit_readout.py")
_spec = importlib.util.spec_from_file_location("plan_target_hit_readout", _PATH)
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

NY = ZoneInfo("America/New_York")
D0 = date(2026, 11, 2)                                  # a Monday


def _days(n):
    out, d = [], D0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(closes):
    return [{"date": d, "close": float(c)} for d, c in zip(_days(len(closes)), closes)]


def _plan(pid=10, created=datetime(2026, 11, 2, 11, 0, tzinfo=NY), closed=None, target=110.0,
          ticker="AAA", source="held_seed"):
    return {"id": pid, "ticker": ticker, "status": "active" if closed is None else "closed",
            "source": source, "revision": 0, "created_at": created, "closed_at": closed,
            "close_reason": None, "entry_price": 100.0, "target_price": target}


# ── window and hit ──────────────────────────────────────────────────────────

def test_life_skips_the_creation_session_and_stops_at_close():
    bars = _bars([100, 101, 102, 103, 104])
    life = src.life_sessions(bars, datetime(2026, 11, 2, 11, 0, tzinfo=NY), None)
    assert [b["date"] for b in life] == _days(5)[1:]
    life = src.life_sessions(bars, datetime(2026, 11, 2, 9, 30, tzinfo=NY), None)
    assert len(life) == 5                                # open at created_at counts
    closed = datetime(2026, 11, 5, 15, 0, tzinfo=NY)     # before that day's 16:00 close
    life = src.life_sessions(bars, datetime(2026, 11, 2, 11, 0, tzinfo=NY), closed)
    assert [b["date"] for b in life] == _days(5)[1:3]


def test_hit_forward_path_and_giveback():
    closes = [100, 105, 111, 115, 112] + [108] * 4 + [104] + [100] * 10 + [96]
    r = src.analyse(_plan(), _bars(closes))
    assert r["hit_date"] == _days(3)[2].isoformat() and r["sessions_to_hit"] == 2
    assert r["target_vs_entry_pct"] == 10.0
    assert r["hit_close_vs_target_pct"] == round((111 / 110 - 1) * 100, 2)
    assert r["sessions_since_hit"] == 18
    assert r["fwd5_vs_target_pct"] == round((108 / 110 - 1) * 100, 2)
    assert r["fwd10_vs_target_pct"] == round((100 / 110 - 1) * 100, 2)
    assert r["fwd20_vs_target_pct"] is None and r["low_through_20_vs_target_pct"] is None
    assert r["peak_vs_entry_pct"] == 15.0 and r["peak_date"] == _days(4)[3].isoformat()
    assert r["latest_vs_entry_pct"] == -4.0
    assert r["latest_vs_target_pct"] == round((96 / 110 - 1) * 100, 2)
    assert r["giveback_from_peak_pct"] == round((96 / 115 - 1) * 100, 2)
    assert r["max_drawdown_from_peak_pct"] == r["giveback_from_peak_pct"]


def test_low_through_20_and_a_recovered_drawdown():
    closes = [100, 111, 120, 90] + [115] * 20
    r = src.analyse(_plan(), _bars(closes))
    assert r["sessions_to_hit"] == 1 and r["sessions_since_hit"] == 22
    assert r["fwd20_vs_target_pct"] == round((115 / 110 - 1) * 100, 2)
    assert r["low_through_20_vs_target_pct"] == round((90 / 110 - 1) * 100, 2)
    assert r["max_drawdown_from_peak_pct"] == -25.0       # 90 below the 120 peak
    assert r["giveback_from_peak_pct"] == round((115 / 120 - 1) * 100, 2)


def test_no_hit_inside_the_life_or_no_target():
    closes = [100, 105, 108, 111, 115]
    closed = datetime.combine(_days(3)[2], datetime.min.time(), tzinfo=NY).replace(hour=17)
    r = src.analyse(_plan(closed=closed), _bars(closes))  # hits only after it closed
    assert r["life_sessions"] == 2 and r["hit_date"] is None and r["latest_vs_entry_pct"] is None
    r = src.analyse(_plan(target=None), _bars(closes))
    assert r["hit_date"] is None and r["target_vs_entry_pct"] is None
    assert src.analyse(_plan(), [])["life_sessions"] == 0


def test_records_hold_no_prices():
    r = src.analyse(_plan(), _bars([100, 105, 111, 115]))
    assert not {"entry_price", "target_price", "close", "price"} & set(r)


# ── verdicts ────────────────────────────────────────────────────────────────

def test_period():
    hit = _days(3)[2].isoformat()
    at = lambda d, h: datetime.combine(d, datetime.min.time(), tzinfo=NY).replace(hour=h)
    assert src.period(at(_days(3)[1], 15), hit) == "before hit"
    assert src.period(at(_days(3)[2], 10), hit) == "hit session"
    # 21:00 ET is the next UTC day but the same ET date
    assert src.period(at(_days(3)[2], 21).astimezone(timezone.utc), hit) == "hit session"
    assert src.period(at(_days(4)[3], 10), hit) == "after hit"
    assert src.period(at(_days(4)[3], 10), None) is None


def _v(vid, d, h, pid="10", rel="FOLLOW", why="within plan", held="true", final="HOLD"):
    ts = datetime.combine(d, datetime.min.time(), tzinfo=NY).replace(hour=h)
    return (vid, ts, "AAA", pid, rel, why, held, final)


def _fixture():
    days = _days(6)
    plans = [_plan(10), _plan(11, ticker="BBB", source="analyst"), _plan(12, ticker="CCC")]
    bars = {"AAA": _bars([100, 105, 111, 115, 112, 108]),
            "BBB": _bars([100, 105, 111, 115, 112, 108]),
            "CCC": _bars([100, 101, 102, 103, 104, 105])}
    verdicts = [_v(1, days[1], 10), _v(2, days[2], 10, rel="DIVERGE", why="target reached"),
                _v(3, days[2], 14, rel="DIVERGE", why="target reached"),
                _v(4, days[4], 10, rel="DIVERGE", why="target reached", final="ADD"),
                _v(5, days[2], 10, pid="11", held="false"),          # never held
                _v(6, days[3], 10, pid="12", rel="DIVERGE", why="target reached")]
    return plans, verdicts, bars


def test_build_held_filter_and_counts():
    plans, verdicts, bars = _fixture()
    out = src.build(plans, verdicts, bars, all_plans=False)
    assert [r["plan_id"] for r in out["records"]] == [10, 12]
    assert out["counts"] == {"plans": 3, "held": 2, "read": 2, "with_target": 2,
                             "with_session": 2, "hit": 1}
    pp = out["per_period"]
    assert pp[(10, "before hit", "FOLLOW", "within plan", "HOLD")] == 1
    assert pp[(10, "hit session", "DIVERGE", "target reached", "HOLD")] == 2
    assert pp[(10, "after hit", "DIVERGE", "target reached", "ADD")] == 1
    assert sum(pp.values()) == 4                                       # plan 12 has no hit
    assert out["reached"] == {("close hit", "HOLD"): 2, ("close hit", "ADD"): 1,
                              ("no close hit", "HOLD"): 1}
    every = src.build(plans, verdicts, bars, all_plans=True)
    assert every["counts"]["read"] == 3 and every["counts"]["hit"] == 2


def test_render_percent_only():
    plans, verdicts, bars = _fixture()
    text = src.render(src.build(plans, verdicts, bars, False), "live", D0, False, "abc1234",
                      "2026-11-09")
    assert "with a close hit 1." in text and "held plans" in text
    assert "| 10 | AAA | held_seed | 0 | active |" in text
    assert "| 10 | hit session | DIVERGE | target reached | HOLD | 2 |" in text
    assert "| close hit | 1 | 2 | 3 |" in text and "| no close hit | 0 | 1 | 1 |" in text
    header = next(l for l in text.splitlines() if l.startswith("| plan | name"))
    sep = text.splitlines()[text.splitlines().index(header) + 1]
    assert header.count("|") == sep.count("|")
    for word in ("$", "110", "115"):
        assert word not in text
    empty = src.render(src.build([], [], {}, False), "live", D0, False, "x", None)
    assert empty.count("None.") == 3


# ── main ────────────────────────────────────────────────────────────────────

def test_db_root(tmp_path):
    assert src.db_root(str(tmp_path)) == src.ROOT
    (tmp_path / "db.py").write_text("")
    assert src.db_root(str(tmp_path)) == str(tmp_path)


def test_main_refuses_other_ports(capsys):
    with patch.dict(os.environ, {"SUPABASE_PORT": "5432"}):
        assert src.main([]) == 2
    assert "6543" in capsys.readouterr().err


def test_main_stops_without_a_host(capsys):
    fake_db = types.SimpleNamespace(__file__="/x/db.py", _get_conn=MagicMock())
    env = {k: v for k, v in os.environ.items() if k != "SUPABASE_HOST"}
    env["SUPABASE_PORT"] = "6543"
    with patch.dict(os.environ, env, clear=True), patch.dict("sys.modules", {"db": fake_db}):
        assert src.main([]) == 2
    fake_db._get_conn.assert_not_called()
    assert "/x/db.py" in capsys.readouterr().err


def test_main_reads_once_read_only_and_rolls_back(tmp_path):
    plans, verdicts, bars = _fixture()
    cols = ("id", "ticker", "status", "source", "revision", "created_at", "closed_at",
            "close_reason", "entry_price", "target_price")
    bar_rows = [(t, b["date"], b["close"]) for t, bs in bars.items() for b in bs]
    cur = MagicMock()
    cur.fetchall.side_effect = [[tuple(p[c] for c in cols) for p in plans], verdicts, bar_rows]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    fake_db = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=MagicMock())
    out, ex = tmp_path / "r.md", tmp_path / "r.json"
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543", "SUPABASE_HOST": "h"}), \
         patch.dict("sys.modules", {"db": fake_db}):
        assert src.main(["--out", str(out), "--export", str(ex), "--since", "2026-11-01"]) == 0
    sql = [c.args[0] for c in cur.execute.call_args_list]
    assert sql[0] == "SET TRANSACTION READ ONLY"
    assert sql[2:] == [src._PLANS_SQL, src._VERDICTS_SQL, src._BARS_SQL]
    assert cur.execute.call_args_list[3].args[1]["ids"] == ["10", "11", "12"]
    for q in sql[2:]:
        assert not {"insert", "update", "delete"} & set(q.lower().split())
    conn.rollback.assert_called_once()
    fake_db._put_conn.assert_called_once_with(conn)
    assert "with a close hit 1." in out.read_text()
    data = json.loads(ex.read_text())
    assert data["counts"]["hit"] == 1 and len(data["records"]) == 2
    assert "latest cached bar 2026-11-09" in out.read_text()
