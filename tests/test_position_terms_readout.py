"""scripts/position_terms_readout.py: per ticker, day and held flag, how
many analyst verdicts mention each position-management term group, and
whether a held verdict's prompt carried POSITION CONTEXT; counts only, no
text out; the read is read-only on the pooler port and skips the stack's
start."""
import importlib.util
import json
import os
import sys
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "position_terms_readout.py")
_spec = importlib.util.spec_from_file_location("position_terms_readout", _PATH)
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 27, 1, 30, tzinfo=NY)
TS = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)
# A distinctive phrase, so any text leak into the output is detectable.
SECRET = "Zyxwv-quokka"


def _row(sid=1, ts=TS, sym="ANET", held="true", reasoning=None, thesis=None, risks=None,
         gap=None, prompt_null=False, pc_this=True, pc_any=True, mo_type="object"):
    return (sid, ts, sym, held, mo_type, reasoning, thesis, risks, gap, prompt_null,
            pc_this, pc_any)


@pytest.mark.parametrize("text, groups", [
    ("Hold; the protective stop sits below support.", ["stop"]),
    ("A stop-loss is in place.", ["stop"]),
    ("Consider trimming into strength.", ["trim_exit"]),
    ("Take profits above the target.", ["trim_exit"]),
    ("Exit if guidance is cut.", ["trim_exit"]),
    ("Up since entry; the cost basis is well below.", ["entry_pnl"]),
    ("Unrealized P&L is positive.", ["entry_pnl"]),
    ("The plan target is intact.", ["plan"]),
    ("This would invalidate the thesis.", ["invalidation"]),
    ("The thesis is broken if margins fall.", ["invalidation"]),
    ("Within the holding period; horizon unchanged.", ["horizon"]),
    ("Revenue grew 20% per Q2 fundamentals.", []),
    ("Stopped out? No: trim the position before the horizon.", ["stop", "trim_exit", "horizon"]),
])
def test_term_groups_v1(text, groups):
    assert pt.groups_hit(text) == groups


def test_words_inside_others_do_not_hit():
    assert pt.groups_hit("unstoppable demand, nonstop orders, a planner, an exitless maze") == []
    assert pt.groups_hit("re-entry of a competitor") == ["entry_pnl"]   # \b at the hyphen


def test_the_sha_is_of_the_groups_and_printed():
    import hashlib
    want = hashlib.sha256(json.dumps([list(t) for t in pt.TERM_GROUPS],
                                     separators=(",", ":")).encode()).hexdigest()[:12]
    assert pt.TERM_GROUPS_SHA == want and pt.TERM_GROUPS_VERSION == 1
    text = pt.render([], "live", date(2026, 11, 12), date(2026, 11, 28), "headfix")
    assert f"v1 (sha {want})" in text and pt.AID_NOTE in text
    assert "reporting aid, not a study definition" in text and "Luke's" in text


def test_field_text_reads_lists_objects_and_json_strings():
    assert pt.field_text(["a stop", {"x": "trim"}]) == "a stop\ntrim"
    assert pt.field_text({"primary_risk": "thesis broken", "n": 3}) == "thesis broken"
    assert pt.field_text('["horizon"]') == "horizon"
    assert pt.field_text(None) == "" and pt.field_text(4.5) == ""


@pytest.mark.parametrize("flags, label", [((True, False, False), "prompt NULL"),
                                          ((False, True, True), "this ticker"),
                                          ((False, False, True), "another ticker"),
                                          ((False, False, False), "none"),
                                          ((False, None, None), "none")])
def test_context_label(flags, label):
    assert pt.context_label(*flags) == label


def test_classify_keeps_flags_not_text():
    r = pt.classify(_row(reasoning=f"{SECRET} stop", risks=["exit risk"],
                         gap={"primary_risk": "invalidate"}, held=None, pc_this=False))
    assert r["fields"] == {"reasoning": ["stop"], "thesis_summary": [],
                           "key_risks": ["trim_exit"], "expectation_gap": ["invalidation"]}
    assert r["groups"] == ["stop", "trim_exit", "invalidation", "any"]
    assert r["held"] == "unknown" and r["position_context"] == "another ticker"
    assert r["day"] == "2026-11-20" and SECRET not in json.dumps(r)
    assert pt.classify(_row())["groups"] == []


def _rows():
    return [pt.classify(r) for r in (
        _row(1, reasoning="stop below support", thesis="horizon intact"),
        _row(2, reasoning="nothing here"),
        _row(3, sym="NVDA", reasoning="trim", pc_this=False, pc_any=False),
        _row(4, sym="NVDA", held="false", reasoning="stop", pc_this=False, pc_any=False),
        _row(5, ts=datetime(2026, 11, 21, 15, tzinfo=timezone.utc), prompt_null=True,
             reasoning=SECRET, mo_type="string"),
    )]


def test_render_counts_and_percent():
    text = pt.render(_rows(), "live", date(2026, 11, 12), date(2026, 11, 28), "headfix")
    assert "Verdicts: 5 (held 4, not held 1, unknown 0). model_output not an object: 1." in text
    cols = " | ".join(pt.COLUMNS)
    assert f"| held | n | {cols} |" in text
    assert "| held | 4 | 1 (25.0%) | 1 (25.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (25.0%) | 2 (50.0%) |" in text
    assert "| ANET | held | 3 | 1 (33.3%) |" in text
    assert "| NVDA | not held | 1 | 1 (100.0%) |" in text
    assert "| 2026-11-20 | held | 3 |" in text and "| 2026-11-21 | held | 1 |" in text
    assert "| reasoning | 4 | 1 (25.0%) | 1 (25.0%) |" in text
    assert "| thesis_summary | 4 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (25.0%) | 1 (25.0%) |" in text
    assert "| this ticker | 2 | 1 (50.0%) |" in text and "| prompt NULL | 1 | 0 (0.0%) |" in text
    assert "| ANET | 2 | 0 | 0 | 1 | 3 |" in text and "| NVDA | 0 | 0 | 1 | 0 | 1 |" in text
    assert SECRET not in text and "code headfix." in text


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
    monkeypatch.setattr(pt.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="headfix\n"))
    return put


def test_main_reads_once_read_only_bounded_and_rolls_back(monkeypatch, tmp_path):
    conn = _Conn([_row(1, reasoning=f"{SECRET} stop"), _row(2, thesis=SECRET)])
    put = _fake_db(monkeypatch, conn)
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert pt.main(["--since", "2026-11-12", "--until", "2026-11-28", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2] == pt._VERDICTS_SQL and len(sqls) == 3
    p = conn.log[2][1]
    assert p["env"] == "live" and p["pc_hdr"] == pt.POSITION_HEADER
    assert p["start"] == datetime(2026, 11, 12, tzinfo=NY)
    assert p["end"] == datetime(2026, 11, 28, tzinfo=NY)
    assert conn.rolled == 1 and put == [conn]
    text, data = out.read_text(), js.read_text()
    assert "code headfix." in text and "Verdicts: 2 (held 2," in text
    d = json.loads(data)
    assert d["code_version"] == "headfix" and d["term_groups_sha"] == pt.TERM_GROUPS_SHA
    assert d["verdicts"][0]["groups"] == ["stop", "any"]
    assert SECRET not in text and SECRET not in data


def test_the_sql_returns_only_flags_from_the_prompt():
    sql = pt._VERDICTS_SQL
    assert "s.decision ? 'model_output'" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    assert "s.timestamp >= %(start)s AND s.timestamp < %(end)s" in sql
    rest = sql
    for use in ("s.prompt IS NULL,", "strpos(s.prompt, %(pc_hdr)s || s.symbol || ' (held) ===') > 0,",
                "strpos(s.prompt, %(pc_hdr)s) > 0"):
        assert use in rest
        rest = rest.replace(use, "")
    assert "prompt" not in rest
    for f in pt.FIELDS:
        assert f"s.decision->'model_output'->'{f}'" in sql


def test_the_header_is_context_engines():
    src = open(os.path.join(pt.ROOT, "context_engine.py")).read()
    assert 'lines = [f"=== POSITION CONTEXT — {ticker} (held) ==="]' in src
    assert pt.POSITION_HEADER + "{ticker} (held) ===" in src


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert pt.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = pt.main(["--until", "2026-11-28", "--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert pt.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


def test_main_rejects_an_empty_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert pt.main(["--since", "2026-11-26", "--until", "2026-11-26"], now=NOW) == 2
    assert conn.log == []


def test_main_defaults_the_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert pt.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["start"] == datetime(2026, 11, 12, tzinfo=NY)
    assert conn.log[2][1]["end"] == datetime(2026, 11, 28, tzinfo=NY)


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        pt.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert pt.db_root(str(tmp_path)) == pt.ROOT
    (tmp_path / "db.py").write_text("")
    assert pt.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "hashlib", "json", "os", "re", "subprocess", "sys",
                       "collections", "datetime", "zoneinfo"}
    assert "    import db\n" in src
    assert "position_terms_readout" not in open(os.path.join(pt.ROOT,
                                                             "weekly_readouts.py")).read()
