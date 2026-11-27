"""scripts/_bar_qa.py (eng-candidates 2026-11-26 #2): the non-finite bar flag,
and, per readout that loads price_history bars, that main prints the flag
line and nothing else changes. Each script test wraps the script's own,
unchanged builder (render / build / readout) and checks main's output is
exactly that builder's text plus one flag line, on a fixture with a NaN bar.
No DB; synthetic prices."""
import importlib.util
import json
import math
import os
import sys
import types
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_bq", os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qa = _load("_bar_qa")
NAN = float("nan")


def _days(n, start=date(2026, 10, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _days(30)
NAN_DAY = DAYS[20]
LINE = f"non-finite bars: 1 (AAA {NAN_DAY})"


def _dict_bars(date_fn=lambda d: d, nan_at=20):
    """{AAA, SPY}: 30 daily dict bars; AAA's close at nan_at is NaN."""
    out = {}
    for t, base in (("AAA", 100.0), ("SPY", 500.0)):
        rows = []
        for i, d in enumerate(DAYS):
            c = base * (1 + 0.004 * ((i * 7) % 5 - 2))
            rows.append({"date": date_fn(d), "high": c * 1.01, "low": c * 0.99,
                         "close": NAN if (t == "AAA" and i == nan_at) else c})
        out[t] = rows
    return out


def _tuple_bars():
    return {t: [(b["date"], b["close"]) for b in bs] for t, bs in _dict_bars().items()}


class _Spy:
    """Wraps mod.name; keeps the text it returned."""
    def __init__(self, mod, name):
        self.real, self.text = getattr(mod, name), None
        self.p = patch.object(mod, name, self)

    def __call__(self, *a, **k):
        self.text = self.real(*a, **k)
        return self.text

    def __enter__(self):
        self.p.__enter__()
        return self

    def __exit__(self, *e):
        return self.p.__exit__(*e)


def _flagged(spy_text: str) -> str:
    return spy_text + "\n" + LINE + "\n"


# ── the helper ──────────────────────────────────────────────────────────────

def test_is_finite():
    for v in (1.0, 0, -3, Decimal("1.5"), None, True, "abc", "1.2"):
        assert qa.is_finite(v), v
    for v in (NAN, float("inf"), -float("inf"), Decimal("NaN"), Decimal("Infinity"),
              Decimal("-Infinity"), "NaN", "inf"):
        assert not qa.is_finite(v), v


def test_nonfinite_bars_dicts_tuples_and_columns():
    bars = {"BBB": [{"date": date(2026, 11, 3), "open": Decimal("NaN"), "close": 1.0},
                    {"date": "2026-11-04", "high": 2.0, "low": None, "close": 2.0},
                    {"date": date(2026, 11, 5), "close": 3.0, "volume": NAN}],   # volume not OHLC
            "AAA": [(date(2026, 11, 2), float("inf")), (date(2026, 11, 3), 1.0)],
            "CCC": []}
    assert qa.nonfinite_bars(bars) == [("AAA", "2026-11-02"), ("BBB", "2026-11-03")]
    assert qa.flag_line(bars) == "non-finite bars: 2 (AAA 2026-11-02, BBB 2026-11-03)"
    assert qa.flag_record(bars) == {"n": 2, "bars": [["AAA", "2026-11-02"], ["BBB", "2026-11-03"]]}
    assert qa.flag_line({}) == "non-finite bars: 0" and qa.flag_line(None) == "non-finite bars: 0"


def test_helper_does_not_touch_the_bars():
    bars = _dict_bars()
    before = repr(bars)
    qa.flag_line(bars)
    qa.flag_record(bars)
    assert repr(bars) == before and len(bars["AAA"]) == 30
    assert math.isnan(bars["AAA"][20]["close"])


def test_imports_are_stdlib_only():
    src = open(os.path.join(_SCRIPTS, "_bar_qa.py")).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from "))}
    assert imports <= {"__future__", "math", "decimal"}


# ── one test per readout ────────────────────────────────────────────────────

def _argv(*a):
    return patch.object(sys, "argv", ["x", *a])


def test_big_move_coverage(tmp_path, monkeypatch):
    m = _load("big_move_coverage")
    monkeypatch.setenv("SUPABASE_PORT", str(m.READ_PORT))
    bars = _dict_bars()
    fake_db = types.SimpleNamespace(_get_conn=MagicMock(), _put_conn=MagicMock())
    kl = types.SimpleNamespace(load_watchlist=lambda: {"AAA": {}, "SPY": {}})
    args = ["--start", str(DAYS[18]), "--end", str(DAYS[25])]
    with patch.dict("sys.modules", {"db": fake_db, "knowledge_loader": kl}), \
         patch.object(m, "fetch", return_value=(bars, [])):
        out = tmp_path / "o.md"
        with _Spy(m, "render") as spy:
            assert m.main(args + ["--out", str(out)]) == 0
        assert out.read_text() == _flagged(spy.text)
        assert "| AAA |" in spy.text or "Days with" in spy.text
        js = tmp_path / "o.json"
        with _Spy(m, "coverage") as cov:
            assert m.main(args + ["--json", "--out", str(js)]) == 0
    data = json.loads(js.read_text())
    assert data.pop("nonfinite_bars") == {"n": 1, "bars": [["AAA", str(NAN_DAY)]]}
    res = dict(cov.text)
    res.pop("nonfinite_bars")
    assert data == json.loads(json.dumps(res, default=str))


def test_compile_plan_readout(tmp_path):
    m = _load("compile_plan_readout")
    out = tmp_path / "o.md"
    with patch.object(m, "_fetch", return_value=([], [], {}, [], _dict_bars())), \
         _argv("--out", str(out)), _Spy(m, "build") as spy:
        m.main()
    assert out.read_text() == _flagged(spy.text)


def test_entry_gate_readout(tmp_path, monkeypatch):
    m = _load("entry_gate_readout")
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    d = date(2026, 11, 23)
    bars = _tuple_bars()
    ef = m._ef()
    out = tmp_path / "o.md"
    with patch.object(m, "_fetch", return_value=([], bars)), \
         patch.object(m, "_fetch_reclaims", return_value=[]), \
         patch.object(ef, "run", return_value={d: {"passed": 0}}), \
         patch.object(ef, "_git_head", return_value="abc1234"), _Spy(m, "readout") as spy:
        assert m.main(["--since", str(d), "--until", str(d), "--out", str(out)]) == 0
    assert out.read_text() == _flagged(spy.text)


def test_forward_readout(tmp_path):
    m = _load("forward_readout")
    d = DAYS[5]
    ts = datetime(d.year, d.month, d.day, 15, 0, tzinfo=timezone.utc)
    rows = [("AAA", "ENTRY_GATE_BLOCKED", ts, {"gate": {"kind": "trend"}})]
    for by, builder in (("gate", "readout"), ("conviction", "conviction_readout")):
        out = tmp_path / f"{by}.md"
        with patch.object(m, "_fetch", return_value=(rows, _tuple_bars())), \
             _argv("--since", str(DAYS[0]), "--by", by, "--out", str(out)), \
             _Spy(m, builder) as spy:
            m.main()
        assert out.read_text() == _flagged(spy.text), by


def test_plan_forward_readout(tmp_path):
    m = _load("plan_forward_readout")
    plan = {"id": 1, "ticker": "AAA", "status": "active", "source": "analyst",
            "created_at": datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc), "closed_at": None,
            "close_reason": None, "entry_price": 100.0, "invalidation_price": 95.0,
            "target_price": 110.0, "horizon_end": DAYS[-1], "last_relation": None, "revision": 0,
            "structural_stop": None, "t1": None, "levels_source": None}
    out = tmp_path / "o.md"
    with patch.object(m, "_fetch", return_value=([plan], _dict_bars())), \
         _argv("--out", str(out)), _Spy(m, "build") as spy:
        m.main()
    assert out.read_text() == _flagged(spy.text)


def test_plan_level_readout(tmp_path):
    m = _load("plan_level_readout")
    out = tmp_path / "o.md"
    with patch.object(m, "_fetch", return_value=([], [], _dict_bars())), \
         _argv("--out", str(out)), _Spy(m, "build") as spy:
        m.main()
    assert out.read_text() == _flagged(spy.text)


def test_plan_target_hit_readout(tmp_path):
    m = _load("plan_target_hit_readout")
    plan = (10, "AAA", "active", "held_seed", 0,
            datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc), None, None, 100.0, 100.5)
    verdict = (1, datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc), "AAA", "10", "FOLLOW",
               "within plan", "true", "HOLD")
    bar_rows = [(t, b["date"], b["close"]) for t, bs in _dict_bars().items() for b in bs]
    cur = MagicMock()
    cur.fetchall.side_effect = [[plan], [verdict], bar_rows]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    fake_db = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=MagicMock())
    out = tmp_path / "o.md"
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543", "SUPABASE_HOST": "h"}), \
         patch.dict("sys.modules", {"db": fake_db}), _Spy(m, "render") as spy:
        assert m.main(["--since", "2026-10-01", "--out", str(out)]) == 0
    assert out.read_text() == _flagged(spy.text)
    assert "with a close hit 1." in spy.text


def test_rescale_stale_head_dry_run(capsys):
    m = _load("rescale_stale_head")
    rows = [(b["date"], Decimal("NaN") if math.isnan(b["close"]) else Decimal(str(b["close"])), 1000)
            for b in _dict_bars()["AAA"]]

    def conn_for():
        cur = MagicMock()
        cur.fetchall.side_effect = [[("AAA",)], rows]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        return conn

    expected = m.dry_run(conn_for(), None)
    fake_db = types.SimpleNamespace(_get_conn=conn_for, _put_conn=MagicMock())
    with patch.dict("sys.modules", {"db": fake_db}):
        assert m.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == LINE
    assert lines[-2] == f"{len(expected)} candidate step(s)."
    assert len(lines) == len(expected) + 3                  # header, steps, count, flag
    seen = {}
    assert m.dry_run(conn_for(), None, seen) == expected and len(seen["AAA"]) == 30


def test_shadow_outcomes(tmp_path):
    m = _load("shadow_outcomes")
    data = {"trim": [], "events": [], "approvals": [], "routing": [], "loser": [], "guard": [],
            "scans": [], "dips": [], "bars": _tuple_bars()}
    out = tmp_path / "o.md"
    with patch.object(m, "_fetch", return_value=data), \
         patch.object(m, "_git_head", return_value="abc1234"), _Spy(m, "build") as spy:
        assert m.main(["--since", "2026-10-01", "--out", str(out)]) == 0
    assert out.read_text() == _flagged(spy.text)
    assert m.run.__code__.co_code                          # run() kept for other callers


def test_vol_shadow_readout(tmp_path):
    m = _load("vol_shadow_readout")
    bars = _dict_bars(date_fn=str)
    fill = {"symbol": "AAA", "qty": 1.0, "price": 100.0,
            "ts": datetime(DAYS[22].year, DAYS[22].month, DAYS[22].day, 15, 0, tzinfo=timezone.utc),
            "env": "live", "approval_id": None, "buys_before_qty": 0.0, "sells_before_qty": 0.0}
    out = tmp_path / "o.md"
    with patch.object(m, "_fetch", return_value=([fill], bars, {}, [])), \
         _argv("--out", str(out)), _Spy(m, "readout") as spy:
        m.main()
    assert out.read_text() == _flagged(spy.text)
