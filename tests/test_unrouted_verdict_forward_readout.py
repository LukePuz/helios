"""scripts/unrouted_verdict_forward_readout.py on fixture rows: the book
counts and overdue records, the row counts by event and action, expired
records' missing forward events, "not loaded" before the stage loads, no
move in the output, the port check, the read-only fetch, and that
weekly_readouts doesn't load it. No DB."""
import importlib.util
import json
import os
import re
from datetime import date

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "unrouted_verdict_forward_readout",
    os.path.join(_ROOT, "scripts", "unrouted_verdict_forward_readout.py"))
uvf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uvf)

TODAY = date(2026, 12, 30)


def _rec(t, et, action="HOLD", base=True, logged=()):
    return {"ticker": t, "action": action, "et_date": et, "verdict_at": f"{et}T15:00:00+00:00",
            "base_session": et, "base_bar_date": et if base else None,
            "logged": {k: "2026-12-01" for k in logged}}


def _row(t, et, event, action="HOLD", **kw):
    ex = {"event": event, "action": action, "verdict_et_date": et, "base_bar_matches": True,
          "env": "live", "move_pct": 3.21}
    ex.update(kw)
    return {"ts": None, "ticker": t, "extra_json": ex}


def test_book_counts_and_overdue():
    book = json.dumps({"a": _rec("NVDA", "2026-12-20", logged=("forward_5",)),
                       "b": _rec("AMD", "2026-12-29", action="WATCH", base=False),
                       "c": _rec("MU", "2026-11-01", logged=("forward_5", "forward_10"))})
    res = uvf.build([], book, rows_ever=0, today=TODAY)
    b = res["book"]
    assert res["loaded"] and res["book_read"] == "ok"
    assert (b["records"], b["with_base_bar"], b["waiting_for_base_bar"], b["overdue"]) == (3, 2, 1, 1)
    assert b["by_action"] == {"HOLD": 2, "WATCH": 1}
    assert b["logged_events"] == {"forward_10": 1, "forward_5": 2}
    o, = res["overdue"]
    assert (o["ticker"], o["days"], o["logged"]) == ("MU", 59, ["forward_10", "forward_5"])
    assert res["rows"]["rows"] == 0                        # loaded: a real 0


def test_row_counts_and_missing_forward_events():
    rows = [_row("NVDA", "2026-11-20", "forward_5"), _row("NVDA", "2026-11-20", "forward_10"),
            _row("AMD", "2026-11-20", "forward_5", action="PASS", base_bar_matches=False),
            _row("NVDA", "2026-11-20", "expired", sessions=20, window_complete=True,
                 forward_logged=["forward_10", "forward_5"]),
            _row("QTUM", "2026-10-01", "expired", action="WATCH", sessions=0, window_complete=False,
                 forward_logged=[])]
    res = uvf.build(rows, None, rows_ever=5, today=TODAY)
    r = res["rows"]
    assert res["loaded"] and res["book_read"] == "no key" and res["book"]["records"] == "no key"
    assert r["rows"] == 5 and r["records_seen"] == 3
    assert r["by_event"] == {"expired": 2, "forward_10": 1, "forward_5": 2}
    assert r["by_action_event"]["PASS forward_5"] == 1 and r["by_action_event"]["WATCH expired"] == 1
    assert (r["base_bar_mismatch"], r["expired_window_incomplete"]) == (1, 1)
    assert r["expired_missing_forward"] == {"forward_5": 1, "forward_10": 1, "forward_20": 2}
    assert [(m["ticker"], m["missing"]) for m in res["missing"]] == [
        ("NVDA", ["forward_20"]), ("QTUM", ["forward_5", "forward_10", "forward_20"])]


def test_not_loaded_prints_not_loaded_not_zero():
    res = uvf.build([], None, rows_ever=0, today=TODAY)
    assert res["loaded"] is False
    assert set(res["book"].values()) == {uvf.NOT_LOADED} and set(res["rows"].values()) == {uvf.NOT_LOADED}
    md = uvf.render(res, "live", "2026-11-26", "2026-12-30")
    assert "unrouted_verdict_forward_shadow: not loaded" in md
    assert "| rows | not loaded |" in md and "| 0 |" not in md


def test_an_unreadable_book_says_so():
    res = uvf.build([], "not json", rows_ever=0, today=TODAY)
    assert res["loaded"] and res["book_read"] == "unreadable" and res["book"]["records"] == "unreadable"


def test_the_output_carries_no_move():
    rows = [_row("NVDA", "2026-11-20", "forward_5"),
            _row("NVDA", "2026-11-20", "expired", sessions=3, window_complete=False,
                 forward_logged=["forward_5"])]
    book = json.dumps({"c": _rec("MU", "2026-11-01")})
    res = uvf.build(rows, book, rows_ever=2, today=TODAY)
    md = uvf.render(res, "live", "2026-11-26", "2026-12-30")
    js = json.dumps(res, default=str)
    assert "3.21" not in md and "3.21" not in js and "move" not in js and "$" not in md
    assert "## Overdue" in md and "## Expired records" in md


def test_a_wrong_port_reads_nothing(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert uvf.main([]) == 2
    assert "6543" in capsys.readouterr().err


class _Cur:
    def __init__(self, log):
        self.log, self.description = log, [("x",)]

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def fetchall(self):
        return []

    def fetchone(self):
        return None

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
    assert uvf.fetch(c, "live", "2026-11-26", "2026-12-30") == ([], None, 0)
    sqls = [s for s, _ in c.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY" and sqls[1].startswith("SET LOCAL statement_timeout")
    assert c.rolled_back
    assert c.log[-1][1]["key"] == "unrouted_verdict_forward_shadow_live"
    assert not [s for s in sqls if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", s)]


def test_the_constants_match_the_writer_when_it_is_there():
    """crew's own constants on a tree that has the writer (batch 1128 on);
    skipped on one that doesn't."""
    with open(os.path.join(_ROOT, "agents", "crew.py")) as f:
        src = f.read()
    if "_UVF_FORWARD_SESSIONS" not in src:
        return
    m = re.search(r"^_UVF_FORWARD_SESSIONS = \(([0-9, ]+)\)", src, re.M)
    assert tuple(int(x) for x in m.group(1).split(",") if x.strip()) == uvf.HORIZONS
    assert f'_UVF_STAGE = "{uvf.STAGE}"' in src
    assert 'unrouted_verdict_forward_shadow_{env}' in src


def test_weekly_readouts_does_not_load_it():
    with open(os.path.join(_ROOT, "weekly_readouts.py")) as f:
        assert "unrouted_verdict_forward_readout" not in f.read()
