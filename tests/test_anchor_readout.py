"""scripts/anchor_readout.py: groups by the stored anchor values as they are
(nothing merged or labelled), the per-group conviction / UDR / action /
ticker counts, the read-only fetch on a fake connection, and the port, time
and db-origin guards. No DB."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import anchor_readout as ar  # noqa: E402

TS = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)


def _r(i=1, sym="NVDA", sig="HOLD", ta=True, street=True, fv=False, da=False, pair=None,
       conv=7, udr=3.5):
    return (i, sym, sig, TS, ta, street, fv, da, pair, conv, udr)


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.conn.executed.append((sql, args))

    def fetchall(self):
        return list(self.conn.rows)


class _Conn:
    def __init__(self, rows=()):
        self.rows, self.executed, self.rolled_back = rows, [], 0

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled_back += 1


# ---- grouping keys ----------------------------------------------------------

@pytest.mark.parametrize("street, fv, want", [
    (True, False, "street=true fv=false"), (False, False, "street=false fv=false"),
    (None, None, "street=null fv=null"), (True, True, "street=true fv=true")])
def test_target_key_is_the_raw_pair(street, fv, want):
    assert ar.verdict(_r(street=street, fv=fv))["target_anchor"] == want


def test_no_target_anchor_is_not_recorded():
    assert ar.verdict(_r(ta=False, street=None, fv=None))["target_anchor"] == "not recorded"


@pytest.mark.parametrize("da, pair, want", [
    (True, "street+support", "street+support"), (True, "none+stop", "none+stop"),
    (True, "fv+support+stop", "fv+support+stop"), (True, None, "null"), (False, None, "not recorded")])
def test_anchor_pair_is_the_stored_string(da, pair, want):
    assert ar.verdict(_r(da=da, pair=pair))["anchor_pair"] == want


def test_a_bool_or_text_udr_is_missing():
    assert ar.verdict(_r(udr=True))["udr"] is None and ar.verdict(_r(udr="3.5"))["udr"] is None
    assert ar.verdict(_r(udr=float("nan")))["udr"] is None and ar.verdict(_r(udr=2))["udr"] == 2.0


# ---- summaries --------------------------------------------------------------

def _rows():
    raw = [_r(1, conv=7, udr=3.5, da=True, pair="street+support"),
           _r(2, conv=7, udr=3.6, da=True, pair="street+support"),
           _r(3, sym="ANET", conv=6, udr=1.9, street=False, da=True, pair="none+support"),
           _r(4, sym="ANET", sig="WATCH", conv=5, udr=None, street=False, da=True, pair="none+stop"),
           _r(5, sym="MSFT", conv=None, udr=2.0, ta=False, street=None, fv=None)]
    return [ar.verdict(x) for x in raw]


def test_each_value_is_its_own_group_and_nothing_is_merged():
    res = ar.readout(_rows())
    assert res["verdicts"] == 5
    assert set(res["by"]["target_anchor"]) == {"street=true fv=false", "street=false fv=false",
                                               "not recorded"}
    assert set(res["by"]["anchor_pair"]) == {"street+support", "none+support", "none+stop",
                                             "not recorded"}
    for k in ar.GROUPINGS:
        assert sum(g["n"] for g in res["by"][k].values()) == 5
        assert "other" not in res["by"][k] and "echo" not in res["by"][k]


def test_group_stats():
    g = ar.readout(_rows())["by"]["target_anchor"]["street=true fv=false"]
    assert g["n"] == 2 and g["pct"] == 40.0
    assert g["conviction"] == {"7": {"n": 2, "pct": 100.0}}
    assert g["udr"] == {"n": 2, "n_missing": 0, "min": 3.5, "p25": 3.52, "median": 3.55,
                        "p75": 3.58, "max": 3.6}
    assert g["action"] == {"HOLD": 2} and g["tickers"] == {"NVDA": 2}


def test_missing_values_are_counted_not_dropped():
    by = ar.readout(_rows())["by"]
    nr = by["target_anchor"]["not recorded"]
    assert nr["conviction"] == {"null": {"n": 1, "pct": 100.0}}
    ff = by["target_anchor"]["street=false fv=false"]
    assert ff["udr"]["n"] == 1 and ff["udr"]["n_missing"] == 1
    assert ff["action"] == {"HOLD": 1, "WATCH": 1} and ff["tickers"] == {"ANET": 2}


def test_conviction_keys_run_high_to_low_with_text_last():
    rows = [ar.verdict(_r(i, conv=c)) for i, c in enumerate([5, 7, 6.5, None, 7], 1)]
    g = ar.readout(rows)["by"]["target_anchor"]["street=true fv=false"]
    assert list(g["conviction"]) == ["7", "6.5", "5", "null"]


def test_groups_are_ordered_by_size_then_value():
    res = ar.readout(_rows())
    assert list(res["by"]["target_anchor"]) == ["street=false fv=false", "street=true fv=false",
                                                "not recorded"]
    assert res["tickers"] == {"ANET": 2, "NVDA": 2, "MSFT": 1}


def test_percentiles_are_linear():
    assert ar._pctile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5 and ar._pctile([], 0.5) is None


def test_render_prints_counts_and_percent_only():
    res = {"env": "live", "since": "2026-11-24", "until": "2026-11-26", **ar.readout(_rows())}
    text = ar.render(res)
    assert "## By decision.target_anchor" in text and "## By decision.downside_anchor.anchor_pair" in text
    assert "| street=true fv=false | 2 | 40.0 | c7 2 (100.0%) | 2 / 3.5 / 3.52 / 3.55 / 3.58 / 3.6 (0) |" in text
    assert "| not recorded |" in text and "$" not in text


# ---- fetch, guards ----------------------------------------------------------

def test_fetch_is_read_only_bounded_and_rolled_back():
    conn = _Conn([_r()])
    rows = ar.fetch(conn, "live", "2026-11-24", "2026-11-26", ("SCREENED_OUT",))
    assert len(rows) == 1 and conn.rolled_back == 1
    sqls = [s for s, _ in conn.executed]
    assert sqls[0] == "SET TRANSACTION READ ONLY" and sqls[1].startswith("SET LOCAL statement_timeout")
    sql, args = conn.executed[2]
    assert "trading_env = %(env)s" in sql and "decision ? 'model_output'" in sql
    assert "NOT (signal = ANY(%(non)s))" in sql and args["non"] == ["SCREENED_OUT"]
    assert args["lo"].isoformat() == "2026-11-24T00:00:00-05:00"
    assert args["hi"].isoformat() == "2026-11-27T00:00:00-05:00"
    assert not any(w in sql.upper() for w in ("INSERT", "UPDATE", "DELETE"))


def test_fetch_rolls_back_when_the_read_fails():
    class Bad(_Conn):
        def cursor(self):
            raise RuntimeError("down")
    conn = Bad()
    with pytest.raises(RuntimeError):
        ar.fetch(conn, "live", "2026-11-24", "2026-11-26", ())
    assert conn.rolled_back == 1


def test_ticker_filter():
    conn = _Conn([])
    ar.fetch(conn, "live", "2026-11-24", "2026-11-26", (), ticker="NVDA")
    sql, args = conn.executed[2]
    assert sql.endswith("AND symbol = %(ticker)s") and args["ticker"] == "NVDA"


def test_run_returns_the_window_and_both_groupings():
    res = ar.run(_Conn([_r()]), "live", "2026-11-24", "2026-11-26", ())
    assert res["env"] == "live" and res["since"] == "2026-11-24" and set(res["by"]) == set(ar.GROUPINGS)


def test_main_refuses_without_the_pooler_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert ar.main([]) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hhmm, quiet", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_the_restart_window(hhmm, quiet):
    now = datetime(2026, 11, 27, *hhmm, tzinfo=ar.NY)
    assert ar.in_quiet_window(now) is quiet


def test_main_refuses_in_the_restart_window(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    called = []
    monkeypatch.setattr(ar, "_import_db", lambda: called.append(1))
    assert ar.main([], now=datetime(2026, 11, 27, 5, 0, tzinfo=ar.NY)) == 2
    assert called == [] and "restart window" in capsys.readouterr().err


def test_main_rejects_bad_dates(monkeypatch):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    assert ar.main(["--since", "bad"]) == 2


def test_db_comes_from_the_working_directory(monkeypatch, tmp_path):
    (tmp_path / "db.py").write_text("X = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "db", raising=False)
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + [p for p in sys.path if p != str(tmp_path)])
    mod = ar._import_db()
    assert mod.__file__ == str(tmp_path / "db.py")
    monkeypatch.delitem(sys.modules, "db", raising=False)


def test_the_working_directory_wins_over_an_earlier_path(monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "db.py").write_text("X = 2\n")
    (tmp_path / "db.py").write_text("X = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "db", raising=False)
    monkeypatch.setattr(sys, "path", [str(other), str(tmp_path)] + sys.path)
    assert ar._import_db().X == 1
    monkeypatch.delitem(sys.modules, "db", raising=False)


def test_a_db_not_in_the_working_directory_is_refused(monkeypatch, tmp_path):
    """No db.py in the working directory: the one found elsewhere (here a
    subdirectory on the path) is refused, not used."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "db.py").write_text("X = 2\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "db", raising=False)
    monkeypatch.setattr(sys, "path", [str(other)] + sys.path)
    with pytest.raises(RuntimeError):
        ar._import_db()
    monkeypatch.delitem(sys.modules, "db", raising=False)


def test_main_writes_json(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_PORT", "6543")

    class FakeDb:
        _NON_ANALYST_SIGNALS = ("SCREENED_OUT",)

        @staticmethod
        def _get_conn():
            return _Conn([_r()])

        @staticmethod
        def _put_conn(c):
            pass
    monkeypatch.setattr(ar, "_import_db", lambda: FakeDb)
    out = tmp_path / "o.json"
    assert ar.main(["--env", "live", "--json", "--out", str(out)],
                   now=datetime(2026, 11, 26, 22, 0, tzinfo=ar.NY)) == 0
    import json
    assert json.loads(out.read_text())["verdicts"] == 1
