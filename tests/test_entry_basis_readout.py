"""scripts/entry_basis_readout.py: the two entry figures a held name's brief
prints (CURRENT PORTFOLIO's broker average cost, POSITION CONTEXT's stop
record) as a % gap and the P&L on each, per ticker and day, percentages
only; the read is read-only on the pooler port, returns only the two
sections and skips the stack's start."""
import importlib.util
import json
import os
import sys
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "entry_basis_readout.py")
_spec = importlib.util.spec_from_file_location("entry_basis_readout", _PATH)
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 26, 23, 50, tzinfo=NY)
# Distinctive made-up prices, so a leak into the output is detectable.
B_ENTRY, CUR, R_ENTRY = "123.45", "100.24", "107.53"


def _pf(*lines):
    return "\n".join([eb.PORTFOLIO_HEADER, *lines]) + "\n\n=== ACCOUNT STATE ===\nx"


def _pf_line(sym="ANET", entry=B_ENTRY, cur=CUR, pl="-18.8%", weight="~10% of portfolio"):
    return f"  {sym:<6} {weight} | Entry: ${entry} | Current: ${cur} | P&L: {pl}"


def _pc(sym="ANET", entry=R_ENTRY, stop="95.17"):
    return (f"=== POSITION CONTEXT — {sym} (held) ===\nEntry ${entry}; protective stop ${stop} "
            f"(-5.1% from current price) [2.0× ATR (14) below entry = 12.36]\nHeld 22 days; bucket")


PF = _pf(_pf_line("NVDA", "176.10", "122.57", "-30.4%"), _pf_line())


def _row(sid, ts, ticker="ANET", path="scheduled_review", held="true", prompt_null=False,
         pf=PF, pc=None):
    return (sid, ts, ticker, path, held, prompt_null, pf, _pc() if pc is None else pc)


TS = datetime(2026, 11, 26, 15, 17, tzinfo=UTC)


# --- parsing ----------------------------------------------------------------

def test_portfolio_line_reads_the_tickers_line_only():
    assert eb.portfolio_line(PF, "ANET") == (123.45, 100.24, -18.8)
    assert eb.portfolio_line(PF, "NVDA") == (176.10, 122.57, -30.4)


def test_a_ticker_prefix_is_not_the_ticker():
    assert eb.portfolio_line(_pf(_pf_line("ANETX")), "ANET") is None


def test_portfolio_line_stops_at_the_blocks_end():
    block = eb.PORTFOLIO_HEADER + "\n" + _pf_line("NVDA") + "\n\n" + _pf_line("ANET")
    assert eb.portfolio_line(block, "ANET") is None
    block = eb.PORTFOLIO_HEADER + "\n" + _pf_line("NVDA") + "\n=== X ===\n" + _pf_line("ANET")
    assert eb.portfolio_line(block, "ANET") is None


@pytest.mark.parametrize("block, ticker", [(None, "ANET"), ("", "ANET"), (PF, ""),
                                           (eb.PORTFOLIO_HEADER + "\nNo open positions.", "ANET"),
                                           ("=== OTHER ===\n" + _pf_line(), "ANET")])
def test_no_block_or_no_line_is_none(block, ticker):
    assert eb.portfolio_line(block, ticker) is None


def test_a_line_that_does_not_parse_is_unparsed():
    assert eb.portfolio_line(_pf("  ANET   ~10% | Entry: n/a | Current: $1.00 | P&L: +1.0%"),
                             "ANET") == "unparsed"


def test_weight_forms_and_positive_pl_parse():
    block = _pf(_pf_line(weight="<5% of portfolio", pl="+2.9%"), _pf_line("IONQ", weight="weight unknown"))
    assert eb.portfolio_line(block, "ANET") == (123.45, 100.24, 2.9)
    assert eb.portfolio_line(block, "IONQ")[0] == 123.45


def test_position_entry_reads_the_header_ticker_and_entry():
    assert eb.position_entry(_pc()) == ("ANET", 107.53)
    assert eb.position_entry(_pc("BRK.B")) == ("BRK.B", 107.53)


@pytest.mark.parametrize("block", [None, "", "=== POSITION CONTEXT — ANET ===\nEntry $1.00; protective stop $1"])
def test_position_entry_without_a_header_is_none(block):
    assert eb.position_entry(block) is None


def test_position_entry_with_another_first_line_is_unparsed():
    assert eb.position_entry("=== POSITION CONTEXT — ANET (held) ===\nHeld 3 days") == ("ANET", "unparsed")


# --- per verdict ------------------------------------------------------------

def test_classify_gives_percentages_only():
    r = eb.classify(_row(7, TS))
    assert r["broker_seen"] and r["record_seen"] and r["reason"] is None
    assert r["gap_pct"] == round((123.45 / 107.53 - 1) * 100, 2) == 14.81
    assert r["broker_pl_pct"] == -18.8
    assert r["record_pl_pct"] == round((100.24 / 107.53 - 1) * 100, 2) == -6.78
    assert r["pl_gap_pts"] == round((100.24 / 107.53 - 1) * 100 + 18.8, 2)
    assert r["at"] == "2026-11-26 10:17" and r["day"] == "2026-11-26"
    dumped = json.dumps(r)
    for price in (B_ENTRY, CUR, R_ENTRY, "95.17"):
        assert price not in dumped


def test_the_gap_sign_is_broker_over_record():
    r = eb.classify(_row(1, TS, pf=_pf(_pf_line(entry="100.00", cur="110.00", pl="+10.0%")),
                         pc=_pc(entry="120.00")))
    assert r["gap_pct"] == -16.67 and r["record_pl_pct"] == -8.33 and r["pl_gap_pts"] == -18.33


def test_equal_entries_have_a_zero_gap():
    r = eb.classify(_row(1, TS, pf=_pf(_pf_line(entry="107.53")), pc=_pc(entry="107.53")))
    assert r["gap_pct"] == 0.0


@pytest.mark.parametrize("kw, reason, broker, record", [
    ({"prompt_null": True, "pf": None, "pc": ""}, "prompt NULL", False, False),
    ({"pf": _pf(_pf_line("NVDA"))}, "no portfolio line", False, True),
    ({"pc": ""}, "no position context", True, False),
    ({"pc": _pc("NVDA")}, "context for another ticker", True, False),
    ({"pf": _pf("  ANET   x | Entry: n/a")}, "unparsed", False, True),
    ({"pc": "=== POSITION CONTEXT — ANET (held) ===\nHeld 3 days"}, "unparsed", True, False),
    ({"pf": _pf(_pf_line(entry="0.00"))}, "unparsed", False, False),
])
def test_a_verdict_without_both_gets_a_reason(kw, reason, broker, record):
    r = eb.classify(_row(1, TS, **kw))
    assert r["reason"] == reason and r["gap_pct"] is None
    assert r["broker_seen"] is broker and r["record_seen"] is record


@pytest.mark.parametrize("v, label", [("true", "held"), ("false", "not held"), (None, "unknown")])
def test_held_labels(v, label):
    assert eb.held_label(v) == label


def test_a_naive_timestamp_is_utc():
    assert eb.classify(_row(1, datetime(2026, 11, 27, 3, 0)))["day"] == "2026-11-26"


# --- per cell -------------------------------------------------------------

def _rows():
    pf_small = _pf(_pf_line(entry="108.00"))          # gap +0.44%
    pf_mid = _pf(_pf_line(entry="108.20"))            # gap +0.62%
    return [eb.classify(_row(1, datetime(2026, 11, 4, 15, 0, tzinfo=UTC), pf=pf_small)),
            eb.classify(_row(2, datetime(2026, 11, 5, 15, 0, tzinfo=UTC), pf=pf_mid)),
            eb.classify(_row(3, datetime(2026, 11, 6, 15, 0, tzinfo=UTC))),
            eb.classify(_row(4, datetime(2026, 11, 6, 16, 0, tzinfo=UTC))),
            eb.classify(_row(5, datetime(2026, 11, 6, 17, 0, tzinfo=UTC), ticker="ZZ", pc="")),
            eb.classify(_row(6, datetime(2026, 11, 6, 18, 0, tzinfo=UTC), pc=""))]


def test_tally_counts_and_marks_since_and_steps():
    t = eb.tally(_rows(), 0.5)
    assert t["verdicts"] == 6 and t["both"] == 4 and t["over_cut"] == 3
    assert t["broker_seen"] == 5 and t["record_seen"] == 4
    assert t["reasons"]["no portfolio line"] == 1 and t["reasons"]["no position context"] == 1
    assert t["first_both"] == {"id": 1, "at": "2026-11-04 10:00", "gap_pct": 0.44}
    assert t["first_over_cut"] == {"id": 2, "at": "2026-11-05 10:00", "gap_pct": 0.62}
    assert t["since_is_first_both"] is False
    assert [s["id"] for s in t["steps"]] == [1, 2, 3]
    assert t["gap"]["n"] == 4 and t["gap"]["max"] == 14.81
    assert t["last"]["id"] == 4 and t["last"]["broker_pl_pct"] == -18.8


def test_the_cut_is_on_the_absolute_gap_and_strict():
    rows = [eb.classify(_row(1, TS, pf=_pf(_pf_line(entry="100.00")), pc=_pc(entry="120.00"))),
            eb.classify(_row(2, TS, pf=_pf(_pf_line(entry="100.50")), pc=_pc(entry="100.00")))]
    assert rows[1]["gap_pct"] == 0.5
    t = eb.tally(rows, 0.5)
    assert t["over_cut"] == 1 and t["first_over_cut"]["id"] == 1 and t["since_is_first_both"] is True
    assert eb.tally(rows, 0.0)["over_cut"] == 2


def test_an_empty_cell_has_no_marks():
    t = eb.tally([eb.classify(_row(1, TS, pc=""))], 0.5)
    assert t["both"] == 0 and t["first_both"] is None and t["first_over_cut"] is None
    assert t["last"] is None and t["steps"] == [] and t["gap"] == {"n": 0}


def test_build_keeps_tickers_with_an_entry_and_days_with_both():
    out = eb.build(_rows(), 0.5)
    assert list(out["by_ticker"]) == ["ANET"]            # ZZ printed neither figure
    assert list(out["by_ticker_day"]) == ["ANET / 2026-11-04", "ANET / 2026-11-05", "ANET / 2026-11-06"]
    assert out["by_ticker_day"]["ANET / 2026-11-06"]["both"] == 2
    assert out["by_ticker"]["ANET"]["verdicts"] == 5


def _render(rows, cut=0.5):
    return eb.render(eb.build(rows, cut), "live", date(2026, 10, 28), date(2026, 11, 27), cut, "abc", NOW)


def test_render_has_every_section_and_no_price():
    text = _render(_rows())
    for title in ("By ticker", "Gap steps", "Ticker x day"):
        assert f"## {title}" in text
    for price in (B_ENTRY, CUR, R_ENTRY, "95.17", "108.00", "108.20", "12.36"):
        assert price not in text
    assert "$" not in text
    assert "Code version: abc" in text and "2026-11-26 23:50 ET" in text
    assert "- ANET (3): " in text


def test_every_table_row_has_its_header_width():
    text = _render(_rows())
    tables, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    assert len(tables) == 2
    for t in tables:
        assert len({l.count("|") for l in t}) == 1, t[0]


def test_render_names_the_cut():
    assert "abs(gap) > 1.5%" in _render(_rows(), 1.5)


# --- the read ---------------------------------------------------------------

class _Cur:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows=()):
        self.log, self.rows, self.rolled = [], list(rows), 0

    def cursor(self):
        return _Cur(self.log, self.rows)

    def rollback(self):
        self.rolled += 1


def _fake_db(monkeypatch, conn):
    put = []
    mod = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=put.append, __file__="fake")
    monkeypatch.setitem(sys.modules, "db", mod)
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setenv("SUPABASE_HOST", "fake-host")
    return put


def test_main_reads_once_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    conn = _Conn([_row(1, TS)])
    put = _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert eb.main(["--since", "2026-11-01", "--until", "2026-11-27", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2] == eb._VERDICTS_SQL and len(sqls) == 3
    p = conn.log[2][1]
    assert p["env"] == "live" and p["pf_hdr"] == eb.PORTFOLIO_HEADER and p["pf_slice"] == 4000
    assert p["pc_hdr"] == eb.POSITION_HEADER and p["pc_slice"] == 400
    assert p["start"] == datetime(2026, 11, 1, tzinfo=NY) and p["end"] == datetime(2026, 11, 27, tzinfo=NY)
    assert conn.rolled == 1 and put == [conn]
    text, data = out.read_text(), js.read_text()
    assert "with both figures: 1" in text
    assert json.loads(data)["verdicts"][0]["gap_pct"] == 14.81
    for price in (B_ENTRY, CUR, R_ENTRY):
        assert price not in text and price not in data


def test_the_sql_returns_only_the_two_sections_of_verdicts():
    sql = eb._VERDICTS_SQL
    assert "s.decision ? 'model_output'" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    assert "s.timestamp >= %(start)s AND s.timestamp < %(end)s" in sql
    assert "substr(s.prompt, strpos(s.prompt, %(pf_hdr)s), %(pf_slice)s)" in sql
    assert "substr(s.prompt, strpos(s.prompt, %(pc_hdr)s), %(pc_slice)s)" in sql
    rest = sql
    for use in ("s.prompt IS NULL", "strpos(s.prompt, %(pf_hdr)s)", "strpos(s.prompt, %(pc_hdr)s)",
                "substr(s.prompt,"):
        rest = rest.replace(use, "")
    assert "prompt" not in rest


def test_the_section_formats_are_context_engines():
    src = open(os.path.join(eb.ROOT, "context_engine.py")).read()
    assert src.count('pos_lines = ["=== CURRENT PORTFOLIO ==="]') == 2
    assert src.count('f"Entry: ${pos.get(\'entry_price\', 0):.2f} | "') == 2
    assert src.count('f"Current: ${pos.get(\'current_price\', 0):.2f} | "') == 2
    assert src.count('f"P&L: {pl_str}"') == 2
    assert 'lines = [f"=== POSITION CONTEXT — {ticker} (held) ==="]' in src
    assert 'line = f"Entry ${entry:.2f}; protective stop ${sp:.2f}"' in src
    crew = open(os.path.join(eb.ROOT, "agents", "crew.py")).read()
    assert 'entry   = float(p.avg_entry_price)' in crew
    assert '"pl_pct":       (current - entry) / entry,' in crew


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert eb.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = eb.main(["--until", "2026-11-27", "--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert eb.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--since", "2026-11-26", "--until", "2026-11-26"],
                                  ["--gap-cut", "-1"], ["--gap-cut", "nan"]])
def test_main_rejects_bad_arguments(monkeypatch, argv):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert eb.main(argv, now=NOW) == 2 and conn.log == []


def test_main_defaults_the_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert eb.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["start"] == datetime(2026, 10, 28, tzinfo=NY)
    assert conn.log[2][1]["end"] == datetime(2026, 11, 27, tzinfo=NY)


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        eb.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert eb.db_root(str(tmp_path)) == eb.ROOT
    (tmp_path / "db.py").write_text("")
    assert eb.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "json", "os", "re", "statistics", "subprocess", "sys",
                       "collections", "datetime", "zoneinfo"}
    assert "    import db\n" in src
