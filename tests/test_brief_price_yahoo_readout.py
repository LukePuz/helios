"""scripts/brief_price_yahoo_readout.py on fixture rows: the ticker +
brief_built_at join, the verdict picked over a routing row, the missing-row
counts, the due rule, "not loaded" before the writer loads, the markdown (no
prices), the port check, the read-only fetch, and that weekly_readouts
doesn't load it. No DB."""
import importlib.util
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "brief_price_yahoo_readout", os.path.join(_ROOT, "scripts", "brief_price_yahoo_readout.py"))
bpy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpy)

# 14:31 UTC on Fri 11-27 is 09:31 EST; 21:00 UTC is 16:00 EST; Sat 11-28.
B1, B2, B3 = "2026-11-27T14:31:02.123456+00:00", "2026-11-27T15:00:00+00:00", "2026-11-27T16:00:00+00:00"
AFTER, SAT = "2026-11-27T21:00:00+00:00", "2026-11-28T15:00:00+00:00"


def _v(sym, built, signal="HOLD", status="queued", error=None, marker=True, has_key=True, **kw):
    r = {"symbol": sym, "timestamp": None, "signal": signal, "brief_built_at": built,
         "marker": ({"status": status, "queued_at": built, "error": error} if marker else None),
         "has_marker_key": has_key, "price_source": "latest_quote", "quote_source": "daily_fallback",
         "is_stale": "true", "brief_price_age_s": "62000.0"}
    r.update(kw)
    return r


def _y(sym, built, status="ok", gap=-1.234, error=None):
    return {"ts": None, "ticker": sym,
            "extra_json": {"brief_built_at": built, "status": status, "bar_age_s": 42.0,
                           "bar_in_rth": True, "yahoo_vs_brief_pct": gap, "queue_delay_s": 1.5,
                           "elapsed_s": 0.8, "error": error}}


def test_join_and_missing_row_counts():
    rows = [_y("NVDA", B1), _y("AMD", B2, status="error", gap=None, error="SoftTimeLimitExceeded"),
            _y("MU", B3)]
    vs = [_v("NVDA", B1), _v("AMD", B2), _v("TSLA", B3, status="not_queued", error="OperationalError"),
          _v("ANET", B2, marker=False), _v("IONQ", B1)]
    res = bpy.build(rows, vs, marker_key_rows=5, rows_ever=3)
    c = res["counts"]
    assert res["loaded"] is True
    assert (c["keys"], c["due"], c["marker_queued"], c["marker_not_queued"]) == (6, 5, 3, 1)
    assert (c["due_marker_null"], c["queued_with_row"], c["queued_no_row"], c["row_no_marker"]) == (1, 2, 1, 1)
    assert c["rows_in_window"] == 3 and c["rows_by_status"] == {"error": 1, "ok": 2}
    by = {r["ticker"]: r for r in res["keys"]}
    assert by["NVDA"]["yahoo_vs_brief_pct"] == -1.234 and by["NVDA"]["brief_price_age_s"] == 62000.0
    assert by["MU"]["marker"] == "no marker" and by["MU"]["due"] is None
    assert by["TSLA"]["marker_error"] == "OperationalError" and by["TSLA"]["yahoo_status"] == "no row"
    assert by["AMD"]["yahoo_error"] == "SoftTimeLimitExceeded"


def test_the_verdict_is_picked_over_a_routing_row():
    vs = [_v("NVDA", B1, signal="ROUTE_NO_PRICE", marker=False), _v("NVDA", B1, signal="OPEN")]
    k, = bpy.build([_y("NVDA", B1)], vs, non_analyst=("ROUTE_NO_PRICE",), marker_key_rows=2)["keys"]
    assert (k["signal"], k["signal_rows"], k["marker"]) == ("OPEN", 2, "queued")


def test_the_due_rule():
    assert bpy.due_by_rule(_v("X", B1))
    assert not bpy.due_by_rule(_v("X", AFTER))                      # 16:00 ET
    assert not bpy.due_by_rule(_v("X", SAT))
    assert not bpy.due_by_rule(_v("X", B1, is_stale="false"))
    assert not bpy.due_by_rule(_v("X", B1, quote_source="1min"))
    assert not bpy.due_by_rule(_v("X", B1, price_source="polygon_snapshot"))
    assert not bpy.due_by_rule(_v("X", None))


def test_before_the_load_every_count_is_not_loaded():
    vs = [_v("NVDA", B1, marker=False, has_key=False)]
    res = bpy.build([], vs, marker_key_rows=0, rows_ever=0)
    assert res["loaded"] is False
    assert set(res["counts"].values()) == {bpy.NOT_LOADED}
    assert res["keys"][0]["marker"] == "before load"
    md = bpy.render(res, "live", "2026-11-12", "2026-11-26")
    assert "brief_price_yahoo: not loaded" in md and "| keys | not loaded |" in md
    assert "| keys | 0 |" not in md and "| field |" not in md


def test_one_old_activity_row_means_loaded():
    assert bpy.build([], [], marker_key_rows=0, rows_ever=1)["counts"]["keys"] == 0


def test_markdown_has_no_prices():
    res = bpy.build([_y("NVDA", B1)], [_v("NVDA", B1)], marker_key_rows=1, rows_ever=1)
    md = bpy.render(res, "live", "2026-11-12", "2026-11-27")
    assert "-1.23%" in md and "| due | 1 |" in md and "$" not in md


def test_a_wrong_port_reads_nothing(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert bpy.main([]) == 2
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
    assert bpy.fetch(c, "live", "2026-11-12", "2026-11-26") == ([], [], 0, 0)
    assert c.log[0] == "SET TRANSACTION READ ONLY" and c.log[1].startswith("SET LOCAL statement_timeout")
    assert c.rolled_back and len(c.log) == 6
    assert not [s for s in c.log if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", s)]


def test_weekly_readouts_does_not_load_it():
    with open(os.path.join(_ROOT, "weekly_readouts.py")) as f:
        assert "brief_price_yahoo_readout" not in f.read()
