"""scripts/market_data_check.py on fake rows and temp log files, and the two
batch_verify additions (order_quote tally, watchlist_parse WARN). No DB."""
import importlib.util
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


md = _load("market_data_check")
bv = _load("batch_verify")
EF = md._entry_funnel()
NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)      # Wed 11-25 15:00 ET
SESSIONS, START = md.session_window(NOW, 3, EF)                # 11-23, 11-24, 11-25


def _et(m, d, h, mi=0):
    return datetime(2026, m, d, h, mi, tzinfo=NY)


W10167 = ("Warning 10167, reqId 54: Requested market data is not subscribed. Displaying delayed "
          "market data., contract: Stock(conId=4815747, symbol='{t}', exchange='SMART')")
P403 = ("Polygon plan doesn't include /v2/snapshot/locale/us/markets/stocks/tickers data "
        "(403 NOT_AUTHORIZED) — suppressing repeat warnings")


def test_window():
    assert SESSIONS == [date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25)]
    assert START == datetime(2026, 11, 21, 0, 0, tzinfo=NY).astimezone(timezone.utc)


def test_docstring_states_the_entitlement_signal():
    assert ("10167" in md.__doc__ and "reqMarketDataType(3)" in md.__doc__
            and "entitlement signal from our own" in md.__doc__)


# ── logs ────────────────────────────────────────────────────────────────────

def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_log_paths_are_root_and_logs_star_log_only(tmp_path):
    (tmp_path / "logs").mkdir()
    for p in ("helios.log", "logs/celery_llm_worker.log", "logs/celery_worker.log.1",
              "logs/db_write_spill.jsonl", "logs/telegram_outbox.jsonl", "notes.txt",
              "logs/archive.log.gz"):
        (tmp_path / p).write_text("x\n")
    (tmp_path / "logs" / "archive").mkdir()
    (tmp_path / "logs" / "archive" / "old.log").write_text("x\n")
    got = [os.path.relpath(p, tmp_path) for p in md.log_paths(str(tmp_path))]
    assert got == ["helios.log", "logs/celery_llm_worker.log"]


def test_scan_counts_both_formats_from_the_window_start(tmp_path):
    p = tmp_path / "a.log"
    _write(p, [
        "[2026-11-20 10:28:50,449: INFO/ForkPoolWorker-1] " + W10167.format(t="NVDA"),   # before
        "[2026-11-24 10:28:50,449: INFO/ForkPoolWorker-1] " + W10167.format(t="NVDA"),
        "2026-11-24 11:43:45,033 INFO     " + W10167.format(t="IONQ"),
        "[2026-11-24 05:18:45,257: WARNING/ForkPoolWorker-1] " + P403,
        "[2026-11-24 12:00:00,000: INFO/x] nothing here 10167 in a timestamp .110167",
        "no timestamp " + W10167.format(t="AMD")])
    sc = md.scan_log(str(p), START, tz=NY)
    assert [(t.day, tk) for t, tk in sc["w10167"]] == [(24, "NVDA"), (24, "IONQ")]
    assert [t.hour for t in sc["p403"]] == [5] and sc["cut"] is False and sc["error"] is None


def test_scan_reads_only_the_tail_of_a_big_log(tmp_path):
    p = tmp_path / "big.log"
    head = ["[2026-11-24 09:00:00,000: INFO/x] " + W10167.format(t="OLD")]
    filler = ["[2026-11-24 09:30:00,000: INFO/x] " + "y" * 200] * 200
    tail = ["[2026-11-24 15:00:00,000: INFO/x] " + W10167.format(t="NEW")]
    _write(p, head + filler + tail)
    sc = md.scan_log(str(p), START, max_bytes=4096, tz=NY)
    assert sc["cut"] is True and [tk for _, tk in sc["w10167"]] == ["NEW"]


def test_scan_never_raises_on_a_missing_file(tmp_path):
    sc = md.scan_log(str(tmp_path / "gone.log"), START, tz=NY)
    assert sc["error"] == "FileNotFoundError" and sc["w10167"] == []


# ── counting and output ─────────────────────────────────────────────────────

def _res():
    rows = [{"stage": "order_quote", "ts": _et(11, 24, 10, 28), "md_type_label": "delayed", "side": "BUY"},
            {"stage": "order_quote", "ts": _et(11, 24, 14, 0), "md_type_label": "delayed", "side": "SELL"},
            {"stage": "order_quote", "ts": _et(11, 25, 10, 0), "md_type_label": None, "side": "BUY"},
            {"stage": "entry_ref_age", "ts": _et(11, 24, 10, 28), "md_type_label": "delayed", "side": None},
            {"stage": "order_quote", "ts": _et(11, 22, 12, 0), "md_type_label": "live", "side": "BUY"},  # Sun -> Mon
            {"stage": "other", "ts": _et(11, 24, 10, 0), "md_type_label": "live", "side": None}]
    scans = [{"path": "/x/logs/a.log", "cut": False, "error": None,
              "w10167": [(_et(11, 24, 10, 28), "NVDA"), (_et(11, 25, 10, 0), None)],
              "p403": [_et(11, 24, 5, 18)]}]
    newest = {"order_quote": (_et(11, 25, 10, 0), None), "entry_ref_age": (None, None)}
    return md.build(rows, newest, scans, SESSIONS, EF)


def test_build_buckets_by_session():
    res = _res()
    by = {s["session"]: s for s in res["sessions"]}
    assert by["2026-11-23"]["order_quote"] == {"live": 1}          # the Sunday row
    assert by["2026-11-24"]["order_quote"] == {"delayed": 2}
    assert by["2026-11-24"]["order_quote_side"] == {"BUY": 1, "SELL": 1}
    assert by["2026-11-24"]["entry_ref_age"] == {"delayed": 1}
    assert by["2026-11-24"]["w10167"] == 1 and by["2026-11-24"]["w10167_tickers"] == {"NVDA": 1}
    assert by["2026-11-24"]["p403"] == 1
    assert by["2026-11-25"]["order_quote"] == {"none": 1} and by["2026-11-25"]["w10167_tickers"] == {"?": 1}
    assert res["totals"] == {"order_quote": {"live": 1, "delayed": 2, "none": 1},
                             "entry_ref_age": {"delayed": 1}, "w10167": 2, "p403": 1}
    assert res["newest"]["w10167"].startswith("2026-11-25T10:00")
    assert res["newest"]["entry_ref_age"]["at"] is None


def test_text_and_json():
    meta = {"env": "live", "clone": "abc", "generated": "x", "window_start": "y", "max_log_bytes": 1}
    text = md.render_text(_res(), meta)
    assert "entitlement signal from our own reqMarketDataType(3)" in text
    assert "order_quote: delayed 2 (side: BUY 1, SELL 1)" in text
    assert "Warning 10167 lines: 1 (NVDA 1)" in text and "$" not in text
    json.dumps({"meta": meta, **_res()}, default=str)


# ── the read ────────────────────────────────────────────────────────────────

class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.conn.sql.append((sql, args))

    def fetchall(self):
        return [("order_quote", _et(11, 24, 10), "delayed", "BUY")]

    def fetchone(self):
        return (_et(11, 24, 10), "delayed")


class _Conn:
    def __init__(self):
        self.sql, self.rolled_back = [], False

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled_back = True


def test_fetch_is_read_only_env_filtered_and_rolled_back():
    conn = _Conn()
    rows, newest = md.fetch(conn, "live", START)
    sqls = [s for s, _ in conn.sql]
    assert sqls[0] == "SET TRANSACTION READ ONLY" and "statement_timeout" in sqls[1]
    assert all(s.lstrip().upper().startswith(("SELECT", "SET")) for s in sqls)
    assert all("extra_json->>'env'" in s for s in sqls[2:])
    assert conn.rolled_back and rows[0]["md_type_label"] == "delayed"
    assert set(newest) == {"order_quote", "entry_ref_age"}


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert md.main([]) == 2 and "6543" in capsys.readouterr().err


# ── batch_verify: order_quote tally ─────────────────────────────────────────

def test_batch_verify_tallies_order_quote_md_type():
    rows = [{"id": 1, "stage": "order_quote", "ticker": "NVDA",
             "extra": {"md_type_label": "delayed", "quote_age_s": 1.0, "order_id": 5, "side": "BUY",
                       "env": "live"}},
            {"id": 2, "stage": "order_quote", "ticker": "NVDA",
             "extra": {"md_type_label": "delayed", "quote_age_s": 1.0, "order_id": 6, "side": "SELL",
                       "env": "paper"}},
            {"id": 3, "stage": "order_quote", "ticker": "AMD", "extra": {"side": "BUY", "env": "live"}}]
    chk = next(c for c in bv.check_activity(rows, "live") if c["name"] == "order_quote")
    assert chk["rows"] == 2 and chk["missing"] == [3] and chk["status"] == "FAIL"
    assert chk["reads"]["md_type_label"] == {"delayed": 1, "none": 1}
    assert any("reqMarketDataType(3)" in n for n in chk["notes"])
    ok = next(c for c in bv.check_activity(rows[:1], "live") if c["name"] == "order_quote")
    assert ok["status"] == "OK"
    assert "order_quote" in bv.ACTIVITY_STAGES


# ── batch_verify: watchlist_parse ───────────────────────────────────────────

def test_watchlist_parse_is_warn_only():
    assert bv.check_watchlist_parse([])["status"] == "OK"
    w = bv.check_watchlist_parse(["INTC", "NLST"])
    assert w["status"] == "WARN" and "INTC, NLST" in w["why"] and w["rows"] == 2
    e = bv.check_watchlist_parse(RuntimeError("x"))
    assert e["status"] == "WARN" and "RuntimeError" in e["why"]


def test_watchlist_unparsed_reads_the_loaders_own_warnings():
    import knowledge_loader
    text = ("### AI Infrastructure — Compute\n| NVDA | Nvidia | Compounder | Core |\n"
            "| INTC | Intel | Watch | Watch |\n| NLST | Netlist | Speculative | Speculative |\n")
    with patch.object(knowledge_loader, "_read", return_value=text):
        got = bv.watchlist_unparsed()
    knowledge_loader.load_watchlist.cache_clear()
    assert got == ["INTC", "NLST"]
    handlers = logging.getLogger(knowledge_loader.__name__).handlers
    assert not any(type(h).__name__ == "_Catch" for h in handlers)


def test_watchlist_unparsed_returns_the_error():
    import knowledge_loader
    with patch.object(knowledge_loader, "_read", side_effect=OSError("gone")):
        got = bv.watchlist_unparsed()
    knowledge_loader.load_watchlist.cache_clear()
    assert isinstance(got, OSError)


def test_run_checks_adds_the_watchlist_line_only_when_given():
    now = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)
    since = now - timedelta(hours=10)
    args = ([], [], [], [], [], [], {}, since, now, "live")
    names = [c["name"] for c in bv.run_checks(*args)[0]]
    assert "watchlist_parse" not in names
    names = [c["name"] for c in bv.run_checks(*args, watchlist=["INTC"])[0]]
    assert "watchlist_parse" in names


def test_knowledge_loader_is_untouched():
    """The prompt sha covers knowledge_loader.py; this branch reads it only."""
    import subprocess
    out = subprocess.run(["git", "-C", _HERE, "diff", "--name-only", "f4e7bfa", "--", "knowledge_loader.py"],
                         capture_output=True, text=True).stdout.strip()
    assert out == ""
