"""scripts/post_exit_readout.py: each closed live episode's name and SPY
after the exit-day close (+5/+10/+20/latest, the 20-bar low and high),
percent only, n/a where a window isn't complete; the non-finite bar and
split-step flags; the read is read-only on the pooler port and skips the
stack's start."""
import importlib.util
import json
import math
import os
import sys
import types
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "post_exit_readout.py")
_spec = importlib.util.spec_from_file_location("post_exit_readout", _PATH)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 27, 3, 30, tzinfo=NY)
HEAD = "pinned0"
EXIT = date(2026, 10, 1)
# Distinctive made-up prices, so a leak into the output is detectable.
BASE = 123.47


def _sessions(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(closes, start=EXIT, dates=None):
    dates = dates or _sessions(start, len(closes))
    return [{"date": d, "open": c, "high": c, "low": c, "close": c} for d, c in zip(dates, closes)]


def _path(n_after, step=0.01, base=BASE):
    """base, then n_after closes rising 1% of base per bar."""
    return [base] + [round(base * (1 + step * i), 4) for i in range(1, n_after + 1)]


def _ep(ticker="ANET", exit_day=EXIT, entry=date(2026, 8, 26), lots=1, src="broker_exit", differs=0):
    return (ticker, exit_day, entry, lots, src, differs)


# --- figures ----------------------------------------------------------------

def test_returns_are_from_the_exit_day_close_by_bar_count():
    f = pe.path_figures(_bars(_path(25)), EXIT)
    assert f["base"] and f["bars_after"] == 25
    assert (f["r5"], f["r10"], f["r20"]) == (5.0, 10.0, 20.0)
    assert f["latest"] == 25.0
    assert f["d5"] == _sessions(EXIT, 6)[5].isoformat()


def test_low_and_high_are_over_the_20_closes_after_the_base():
    closes = [100.0] + [100.0 + i for i in range(1, 21)] + [50.0, 200.0]
    closes[7] = 90.0
    f = pe.path_figures(_bars(closes), EXIT)
    assert f["range_complete"] and f["range_bars"] == 20
    assert f["low"] == -10.0 and f["high"] == 20.0      # bars 21-22 are outside


def test_the_base_bar_is_not_in_the_low_and_high():
    f = pe.path_figures(_bars([100.0] + [110.0] * 20), EXIT)
    assert f["low"] == 10.0 and f["high"] == 10.0


def test_incomplete_windows_are_none_and_the_range_is_partial():
    f = pe.path_figures(_bars(_path(7)), EXIT)
    assert f["r5"] == 5.0 and f["r10"] is None and f["r20"] is None and f["d10"] is None
    assert not f["range_complete"] and f["range_bars"] == 7
    assert f["low"] == 1.0 and f["high"] == 7.0


def test_exactly_k_bars_complete_the_k_window():
    f = pe.path_figures(_bars(_path(20)), EXIT)
    assert f["r20"] == 20.0 and f["range_complete"]
    f = pe.path_figures(_bars(_path(19)), EXIT)
    assert f["r20"] is None and not f["range_complete"]


def test_an_exit_on_the_last_bar_has_latest_zero_and_no_range():
    f = pe.path_figures(_bars([BASE]), EXIT)
    assert f["bars_after"] == 0 and f["latest"] == 0.0
    assert f["low"] is None and f["high"] is None and f["r5"] is None


def test_no_bar_on_the_exit_day_is_no_base():
    later = _sessions(EXIT + timedelta(days=1), 10)
    assert pe.path_figures(_bars([BASE] * 10, dates=later), EXIT) == {"base": False}
    assert pe.path_figures([], EXIT) == {"base": False}


def test_bars_before_the_exit_are_ignored():
    dates = _sessions(EXIT - timedelta(days=14), 30)
    i = dates.index(EXIT)
    closes = [999.0] * i + _path(30 - i - 1)
    f = pe.path_figures(_bars(closes, dates=dates), EXIT)
    assert f["r5"] == 5.0 and f["bars_after"] == 30 - i - 1


@pytest.mark.parametrize("bad, label", [(None, "missing"), (float("nan"), "non-finite"),
                                        (Decimal("NaN"), "non-finite"), (float("inf"), "non-finite"),
                                        ("NaN", "non-finite"), (0, "non-positive"),
                                        (-1.0, "non-positive")])
def test_a_bad_close_prints_its_label_and_poisons_the_range(bad, label):
    closes = _path(22)
    closes[5] = bad
    f = pe.path_figures(_bars(closes), EXIT)
    assert f["r5"] == label and f["r10"] == 10.0
    assert f["low"] == label and f["high"] == label
    closes = _path(22)
    closes[0] = bad
    f = pe.path_figures(_bars(closes), EXIT)
    assert f["r5"] == f["r20"] == f["latest"] == f["low"] == label


def test_decimal_and_datetime_inputs():
    bars = [{"date": datetime.combine(d, datetime.min.time()), "close": Decimal(str(c))}
            for d, c in zip(_sessions(EXIT, 6), _path(5))]
    f = pe.path_figures(bars, EXIT)
    assert f["r5"] == 5.0 and isinstance(f["r5"], float)


# --- episodes ---------------------------------------------------------------

def test_episode_record_has_the_name_and_spy_on_the_same_base_date():
    bars = {"ANET": _bars(_path(25)), "SPY": _bars(_path(25, step=-0.002, base=600.13))}
    r = pe.episode_record(_ep(), bars)
    assert r["ticker"] == "ANET" and r["exit_day"] == "2026-10-01" and r["entry_day"] == "2026-08-26"
    assert r["lots"] == 1 and r["exit_sources"] == "broker_exit"
    assert r["name"]["r20"] == 20.0 and r["spy"]["r20"] == -4.0
    assert r["session_mismatch"] == []


def test_a_missing_bar_in_one_series_is_a_session_mismatch_not_realigned():
    days = _sessions(EXIT, 26)
    gap = days[:3] + days[4:]
    bars = {"ANET": _bars(_path(24), dates=gap), "SPY": _bars(_path(25))}
    r = pe.episode_record(_ep(), bars)
    assert r["session_mismatch"] == ["+5", "+10", "+20"]
    assert r["name"]["r5"] == 5.0      # the ticker's own 5th bar


def test_spy_without_the_exit_day_bar():
    bars = {"ANET": _bars(_path(5)), "SPY": _bars(_path(5), start=EXIT + timedelta(days=1))}
    r = pe.episode_record(_ep(), bars)
    assert r["name"]["base"] and r["spy"] == {"base": False}


def test_no_name_bar_means_no_spy_figures_either():
    r = pe.episode_record(_ep("ZZZZ"), {"SPY": _bars(_path(5))})
    assert r["name"] == {"base": False} and r["spy"] == {"base": False}


def test_window_bars_start_at_each_tickers_first_base_day():
    days = _sessions(EXIT - timedelta(days=7), 20)
    bars = {"ANET": _bars([BASE] * 20, dates=days), "SPY": _bars([500.0] * 20, dates=days),
            "QBTS": _bars([BASE] * 20, dates=days)}
    later = days[10]
    recs = [pe.episode_record(_ep("ANET", later), bars), pe.episode_record(_ep("QBTS", EXIT), bars),
            pe.episode_record(_ep("NONE", EXIT), bars)]
    w = pe.window_bars(recs, bars)
    assert set(w) == {"ANET", "QBTS", "SPY"}
    assert pe._day(w["ANET"][0]["date"]) == later and pe._day(w["QBTS"][0]["date"]) == EXIT
    assert pe._day(w["SPY"][0]["date"]) == EXIT


def test_the_nonfinite_flag_reads_only_the_window_bars():
    qa = pe._load("_bar_qa")
    days = _sessions(EXIT - timedelta(days=7), 12)
    closes = [float("nan")] + [BASE] * 11
    bars = {"ANET": _bars(closes, dates=days), "SPY": _bars([500.0] * 12, dates=days)}
    w = pe.window_bars([pe.episode_record(_ep(), bars)], bars)
    assert qa.flag_line(w) == "non-finite bars: 0"
    bars["ANET"][-1]["close"] = float("nan")
    w = pe.window_bars([pe.episode_record(_ep(), bars)], bars)
    assert qa.flag_line(w) == f"non-finite bars: 1 (ANET {pe._day(days[-1]).isoformat()})"


def test_split_steps_flag_a_one_day_split_ratio_in_the_window():
    cand = pe._load("rescale_stale_head").candidates
    closes = _path(10, step=0.001)
    closes[6:] = [c / 8 for c in closes[6:]]
    w = {"SNDK": _bars(closes), "SPY": _bars(_path(10, step=0.001))}
    steps = pe.split_steps(w, cand)
    assert len(steps) == 1 and steps[0][0] == "SNDK" and steps[0][3] == "1/8"
    assert steps[0][1] == _sessions(EXIT, 7)[6].isoformat()
    w["SNDK"][3]["close"] = None     # a missing close is skipped, not a step
    assert len(pe.split_steps(w, cand)) == 1


def test_group_bars_sorts_by_date():
    raw = [("ANET", date(2026, 10, 2), 1, 1, 1, 2), ("ANET", date(2026, 10, 1), 1, 1, 1, 1),
           ("SPY", date(2026, 10, 1), 1, 1, 1, 3)]
    g = pe.group_bars(raw)
    assert [b["close"] for b in g["ANET"]] == [1, 2] and set(g) == {"ANET", "SPY"}


# --- render -----------------------------------------------------------------

def _records():
    bars = {"ANET": _bars(_path(25)), "NVDA": _bars(_path(4), start=date(2026, 11, 19)),
            "SPY": _bars(_path(60, step=-0.001, base=600.13))}
    return [pe.episode_record(_ep(), bars),
            pe.episode_record(_ep("NVDA", date(2026, 11, 19), lots=3), bars),
            pe.episode_record(_ep("MSFT", date(2026, 11, 20), src="unknown"), bars)], bars


def _flags(bars):
    w = pe.window_bars(_records()[0], bars)
    return {"nonfinite": pe._load("_bar_qa").flag_line(w),
            "splits": pe.split_steps(w, pe._load("rescale_stale_head").candidates)}


def test_render_has_every_section_na_and_no_price():
    recs, bars = _records()
    text = pe.render(recs, _flags(bars), "live", None, HEAD, NOW)
    assert text.startswith("# Post-exit readout (live, closed episodes, all exit days)")
    assert "Generated 2026-11-27 03:30 ET. Code version: pinned0." in text
    assert "- episodes: 3; with an exit-day bar: 2; complete +5: 1, +10: 1, +20: 1" in text
    assert "non-finite bars: 0" in text and "split-like steps in the windows read: 0" in text
    assert "split caveat (audit-method-notes, price cache basis)" in text
    assert "| ANET | 2026-10-01 | 2026-08-26 | 1 | broker_exit | +5.0% | +10.0% | +20.0% | +25.0% (" in text
    assert "| NVDA | 2026-11-19 | 2026-08-26 | 3 | broker_exit | n/a | n/a | n/a | +4.0% (2026-11-25, 4 bars) " \
           "| n/a (so far +1.0%, 4 bars) | n/a (so far +4.0%, 4 bars) |" in text
    assert "| MSFT | 2026-11-20 | 2026-08-26 | 1 | unknown | no exit-day bar | n/a" in text
    assert "## SPY over the same sessions" in text
    assert "| MSFT | 2026-11-20 | n/a (no exit-day bar for the name) |" in text
    for price in ("123.47", "600.13", "123.4", "600.1"):
        assert price not in text


def test_render_names_the_scope_and_lists_flags():
    recs, bars = _records()
    flags = {"nonfinite": "non-finite bars: 1 (ANET 2026-10-05)",
             "splits": [("SNDK", "2026-10-09", 0.125, "1/8")]}
    recs[0]["session_mismatch"] = ["+5", "latest"]
    text = pe.render(recs, flags, "live", date(2026, 8, 17), HEAD, NOW)
    assert "(live, closed episodes, exit day 2026-08-17 on)" in text
    assert "- non-finite bars: 1 (ANET 2026-10-05)" in text
    assert "split-like steps in the windows read: 1 (SNDK 2026-10-09 x0.125 (1/8))" in text
    assert "(the name's and SPY's k-th bar on different dates): ANET 2026-10-01 (+5/latest)" in text


def test_every_table_row_has_its_header_width():
    recs, bars = _records()
    for r in (recs, []):
        text = pe.render(r, _flags(bars), "live", None, HEAD, NOW)
        width = None
        for line in text.splitlines():
            if line.startswith("| Ticker"):
                width = line.count("|")
            elif line.startswith("|") and width:
                assert line.count("|") == width, line
            elif not line.startswith("|"):
                width = None


def test_render_has_no_label_or_verdict_words():
    recs, bars = _records()
    text = pe.render(recs, _flags(bars), "live", None, HEAD, NOW).lower()
    for word in ("whipsaw", "recommend", "should", "good", "bad stop", "kept falling"):
        assert word not in text


# --- main -------------------------------------------------------------------

class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append((sql, params))

    def fetchall(self):
        sql = self.conn.log[-1][0]
        if sql == pe._EPISODES_SQL:
            return self.conn.eps
        if sql == pe._BARS_SQL:
            return self.conn.bars
        return []


class _Conn:
    def __init__(self, eps=(), bars=()):
        self.log, self.eps, self.bars, self.rolled = [], list(eps), list(bars), 0

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled += 1


def _fake_db(monkeypatch, conn):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append, __file__="fake")
    monkeypatch.setitem(sys.modules, "db", mod)
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("SUPABASE_HOST", "fake-host")
    monkeypatch.setattr(pe, "code_version", lambda: HEAD)
    return put


def _raw(ticker, closes, start=EXIT):
    return [(ticker, b["date"], b["open"], b["high"], b["low"], b["close"])
            for b in _bars(closes, start)]


def test_main_reads_once_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    eps = [_ep(), _ep("RGTI", date(2026, 9, 14), src="unknown")]
    bars = _raw("ANET", _path(25)) + _raw("SPY", _path(70, base=600.13), date(2026, 9, 14)) \
        + _raw("RGTI", _path(30), date(2026, 9, 14))
    conn = _Conn(eps, bars)
    put = _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert pe.main(["--out", str(out), "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2:] == [pe._EPISODES_SQL, pe._BARS_SQL]
    assert conn.log[2][1] == {"env": "live"}
    assert conn.log[3][1] == {"tickers": ["ANET", "RGTI", "SPY"], "start": date(2026, 9, 14)}
    assert conn.rolled == 1 and put == [conn]
    text, data = out.read_text(), js.read_text()
    assert "Code version: pinned0." in text and "- episodes: 2; with an exit-day bar: 2" in text
    d = json.loads(data)
    assert d["code_version"] == HEAD and d["nonfinite_bars"] == {"n": 0, "bars": []}
    assert [e["ticker"] for e in d["episodes"]] == ["ANET", "RGTI"]
    assert d["episodes"][0]["name"]["r20"] == 20.0
    for price in ("123.47", "600.13"):
        assert price not in text and price not in data


def test_since_narrows_the_episodes_and_the_bar_start(monkeypatch):
    eps = [_ep("RGTI", date(2026, 9, 14)), _ep()]
    conn = _Conn(eps, _raw("ANET", _path(5)) + _raw("SPY", _path(5)))
    _fake_db(monkeypatch, conn)
    assert pe.main(["--since", "2026-09-15", "--out", os.devnull], now=NOW) == 0
    assert conn.log[3][1] == {"tickers": ["ANET", "SPY"], "start": EXIT}


def test_no_episodes_reads_no_bars(monkeypatch, tmp_path):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    out = tmp_path / "r.md"
    assert pe.main(["--out", str(out)], now=NOW) == 0
    assert [s for s, _ in conn.log][2:] == [pe._EPISODES_SQL]
    assert "- episodes: 0" in out.read_text()


def test_the_sql_reads_closed_outcomes_and_bars_only():
    sql = pe._EPISODES_SQL
    assert "FROM trade_outcomes" in sql and "trading_env = %(env)s" in sql
    assert "exit_date IS NOT NULL" in sql
    assert "(COALESCE(exit_fill_at, exit_date) AT TIME ZONE 'America/New_York')::date" in sql
    for col in ("exit_price", "entry_price", "qty", "quantity", "outcome_pct", "notes"):
        assert col not in sql and col not in pe._BARS_SQL
    assert "FROM price_history" in pe._BARS_SQL and "volume" not in pe._BARS_SQL


def test_the_exit_time_convention_is_dbs():
    src = open(os.path.join(pe.ROOT, "db.py")).read()
    assert '"SELECT id, exit_date, COALESCE(exit_fill_at, exit_date), exit_source "' in src
    assert "trading_env     VARCHAR(10) NOT NULL DEFAULT 'paper'," in src


def test_the_split_caveat_matches_the_rescale_script():
    src = open(os.path.join(pe.ROOT, "scripts", "rescale_stale_head.py")).read()
    assert "def candidates(rows: list[dict]) -> list[dict]:" in src
    assert "import db" not in src.split("def main")[0]


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert pe.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = pe.main(["--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert pe.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


def test_main_rejects_a_bad_since(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert pe.main(["--since", "11-05"], now=NOW) == 2 and conn.log == []


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        pe.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert pe.db_root(str(tmp_path)) == pe.ROOT
    (tmp_path / "db.py").write_text("")
    assert pe.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "importlib", "json", "math", "os", "subprocess", "sys",
                       "datetime", "zoneinfo"}
    assert "    import db\n" in src


def test_code_version_is_the_short_head(monkeypatch):
    calls = []
    monkeypatch.setattr(pe.subprocess, "run",
                        lambda *a, **k: calls.append(a) or types.SimpleNamespace(stdout="abc1234\n"))
    assert pe.code_version() == "abc1234" and calls[0][0][-1] == "HEAD"
    monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert pe.code_version() == "unknown"
