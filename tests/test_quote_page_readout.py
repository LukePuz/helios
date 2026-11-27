"""scripts/quote_page_readout.py: option-chain quote pages among the screen
worker's screens, per day and ticker, from the log only; counts, percent
and seconds, no title text."""
import importlib.util
import json
import os
import types
from datetime import date

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "quote_page_readout.py")
_spec = importlib.util.spec_from_file_location("quote_page_readout", _PATH)
qp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qp)

# Made-up strike digits and a distinctive title word, so a leak is detectable.
QUOTE = "ANET Nov 2026 777.000 call (ANET261127C00777000) Stock Price"
CUT = "GOOGL Dec 2026 777.500 put (GOOGL2612"
NEWS = "Arista lands Quokkaword deal with a cloud customer"


def _l(day, t, worker, msg):
    return f"[{day} {t},123: INFO/ForkPoolWorker-{worker}] {msg}\n"


def _start(day, t, w, tk, title):
    return _l(day, t, w, f"[{tk}] Screening event: news — {title}")


def _screened(day, t, w, tk, rel):
    return _l(day, t, w, f"Screened [news] {tk}: relevant={rel} urgency=eod tone=neutral "
                         "new_fact=False tickers=[] model=m")


def _done(day, t, w, s):
    return _l(day, t, w, f"Task tasks.screen_event[abc-1] succeeded in {s}s: None")


def _log():
    d1, d2 = "2026-11-19", "2026-11-26"
    return [
        "watchlist.md row not parsed: | INTC | Intel |\n",
        _start(d1, "09:00:00", 1, "ANET", QUOTE),
        _start(d1, "09:00:01", 2, "ANET", NEWS),
        _screened(d1, "09:00:05", 1, "ANET", "False"),
        _done(d1, "09:00:06", 1, "6.0"),
        _screened(d1, "09:00:07", 2, "ANET", "True"),
        _done(d1, "09:00:09", 2, "8.0"),
        _start(d2, "10:00:00", 1, "GOOGL", CUT),
        _l(d2, "10:00:01", 1, "ChromaDB initialized at /x/.chromadb"),
        _screened(d2, "10:00:04", 1, "GOOGL", "False"),
        _done(d2, "10:00:05", 1, "4.5"),
        _start(d2, "10:01:00", 3, "NVDA", "NVDA Jan 2027 777.000 call (ANET270115C00777000) x"),
        _start(d2, "10:01:30", 3, "NVDA", NEWS),   # the first NVDA screen is left unpaired
        _done(d2, "10:01:40", 3, "2.0"),
    ]


@pytest.mark.parametrize("title, want", [
    (QUOTE, ("occ", "ANET")),
    (CUT, ("prefix", None)),
    ("TER Dec 2026 777.000 call (TER261218C00777000) Interactive S", ("occ", "TER")),
    ("CRWD Nov 2026 777.000 put (CRWD261127P00777000) stock price,", ("occ", "CRWD")),
    (NEWS, (None, None)),
    ("Calls and puts on ANET jump in November", (None, None)),
    ("", (None, None)),
])
def test_quote_rule_v1(title, want):
    assert qp.quote_rule(title) == want


def test_the_sha_is_of_the_rules():
    import hashlib
    want = hashlib.sha256(json.dumps([list(t) for t in qp.OCC_RULES],
                                     separators=(",", ":")).encode()).hexdigest()[:12]
    assert qp.OCC_RULES_SHA == want and qp.OCC_RULES_VERSION == 1


def test_parse_pairs_by_worker_and_keeps_no_title():
    recs, n = qp.parse(_log())
    assert n == 14 and len(recs) == 5
    assert [r["rule"] for r in recs] == ["occ", None, "prefix", "occ", None]
    assert [r["seconds"] for r in recs] == [6.0, 8.0, 4.5, None, 2.0]
    assert [r["relevant"] for r in recs] == [False, True, False, None, None]
    assert [r["root_mismatch"] for r in recs] == [False, False, False, True, False]
    assert recs[2]["day"] == "2026-11-26" and recs[2]["ticker"] == "GOOGL"
    blob = json.dumps(recs)
    assert "777" not in blob and "Stock" not in blob and "Quokkaword" not in blob


def test_parse_bounds_the_days():
    recs, _ = qp.parse(_log(), since=date(2026, 11, 20))
    assert {r["day"] for r in recs} == {"2026-11-26"} and len(recs) == 3
    recs, _ = qp.parse(_log(), until=date(2026, 11, 20))
    assert len(recs) == 2 and recs[0]["seconds"] == 6.0


def test_a_failed_task_closes_the_screen():
    lines = [_start("2026-11-26", "10:00:00", 1, "ANET", QUOTE),
             _l("2026-11-26", "10:00:02", 1, "Task tasks.screen_event[a] raised unexpected: X"),
             _done("2026-11-26", "10:00:03", 1, "9.0")]
    recs, _ = qp.parse(lines)
    assert recs[0]["seconds"] is None


def test_summarise_counts_and_seconds():
    recs, _ = qp.parse(_log())
    s = qp.summarise(recs)
    assert s == {"screens": 5, "quote_pages": 3, "occ": 2, "prefix": 1, "root_mismatch": 1,
                 "relevant_true": 0, "relevant_false": 2, "relevant_unknown": 1, "unpaired": 1,
                 "seconds_all": 20.5, "seconds_quote": 10.5}


def test_render_tables_without_title_text():
    recs, n = qp.parse(_log())
    text = qp.render(recs, ["/x/logs/celery_screen_worker.log"], n, "headfix")
    assert "Log celery_screen_worker.log (14 lines); code headfix." in text
    assert f"v1 (sha {qp.OCC_RULES_SHA})" in text and qp.RULE_NOTE in text
    assert "is Luke's" in text
    assert ("Screens: 5; quote pages 3 (60.0%); screen seconds 20.5, on quote pages 10.5 "
            "(51.2%); unpaired 1.") in text
    assert "| 2026-11-19 | 2 | 1 | 50.0% | 1 | 0 | 0 | 0/1/0 | 14.0 | 6.0 | 42.9% | 0 |" in text
    assert "| 2026-11-26 | 3 | 2 | 66.7% | 1 | 1 | 1 | 0/1/1 | 6.5 | 4.5 | 69.2% | 1 |" in text
    assert "| NVDA | 2 | 1 | 50.0% | 1 | 0 | 1 | 0/0/1 | 2.0 | 0.0 | 0.0% | 1 |" in text
    assert "| 2026-11-26 | GOOGL | 1 | 1 | 100.0% |" in text
    assert "| 2026-11-19 | ANET | 2 | 1 |" in text
    body = text.replace(qp.OCC_RULES_SHA, "")   # a hex sha could hold the digits
    for leak in ("777", "Stock", "Quokkaword", "Arista", "call (", "put ("):
        assert leak not in body


def test_main_reads_the_logs_and_writes_json(monkeypatch, tmp_path):
    log, old = tmp_path / "w.log", tmp_path / "w.log.1"
    log.write_bytes("".join(_log()[7:]).encode() + b"\xff\xfe broken bytes\n")
    old.write_text("".join(_log()[:7]))
    monkeypatch.setattr(qp.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="headfix\n"))
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert qp.main(["--log", str(old), "--log", str(log), "--out", str(out),
                    "--json", str(js)]) == 0
    text, data = out.read_text(), js.read_text()
    assert "Log w.log.1, w.log (15 lines); code headfix." in text
    d = json.loads(data)
    assert d["total"]["quote_pages"] == 3 and d["per_day"]["2026-11-26"]["screens"] == 3
    assert d["per_ticker"]["ANET"]["quote_pages"] == 1 and d["occ_rules_sha"] == qp.OCC_RULES_SHA
    sha = qp.OCC_RULES_SHA
    assert "777" not in text.replace(sha, "") and "777" not in data.replace(sha, "")


def test_main_default_log_is_under_the_cwd(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert qp.main([]) == 2
    assert "logs/celery_screen_worker.log" in capsys.readouterr().err
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "celery_screen_worker.log").write_text("".join(_log()))
    monkeypatch.setattr(qp.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="headfix\n"))
    assert qp.main(["--out", os.devnull]) == 0


@pytest.mark.parametrize("argv", [["--since", "2026-11-26", "--until", "2026-11-26"],
                                  ["--since", "11/26/2026"]])
def test_main_rejects_bad_days(argv, tmp_path):
    assert qp.main(argv + ["--log", str(tmp_path / "none")]) == 2


def test_imports_are_the_standard_library_only():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "hashlib", "json", "os", "re", "subprocess", "sys",
                       "collections", "datetime"}
    assert "import db" not in src and "chromadb" not in src.replace(".chromadb", "")
    assert "quote_page_readout" not in open(os.path.join(qp.ROOT, "weekly_readouts.py")).read()


def test_the_log_line_is_crews():
    src = open(os.path.join(qp.ROOT, "agents", "crew.py")).read()
    assert ('log.info(f"[{ticker}] Screening event: {event.get(\'source\',\'?\')} — '
            '{event.get(\'title\',\'\')[:60]}")') in src
