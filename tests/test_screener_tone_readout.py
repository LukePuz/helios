"""scripts/screener_tone_readout.py: news screener pass rates by headline
category and prompt version; DB rows where they carry titles, the screen
worker log for the other days (kept apart); read-only."""
import importlib.util
import os
import sys
from datetime import date, datetime, timezone
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "screener_tone_readout",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "screener_tone_readout.py"))
str_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(str_)

D10, D11 = date(2026, 11, 10), date(2026, 11, 11)


def _ts(d, hour=15):
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)


def test_the_categories_are_the_audits_four():
    assert [n for n, _ in str_.CATEGORIES] == ["upgrade_raise", "downgrade_cut", "exec_departure",
                                               "offering_dilution"]
    assert str_.categories("Broadcom (AVGO) Downgraded As Wall Street Questions AI") == ["downgrade_cut"]
    assert str_.categories("Morgan Stanley raises price target on NVDA") == ["upgrade_raise"]
    assert str_.categories("IonQ Stock Falls 6.4% as CFO Inder Singh Steps Down") == ["exec_departure"]
    assert str_.categories("AAOI Stock Tumbles On $550M At-The-Market Equity Offering") == [
        "offering_dilution"]
    assert str_.categories(None) == [] and str_.categories("Nvidia beats estimates") == []
    assert len(str_.LEXICON_SHA) == 12


def test_only_fully_titled_days_are_measured_from_the_db():
    days = str_.day_table([(date(2026, 11, 9), ["ee4bf82515e1"], 668, 144, 47),
                           (D10, ["ee4bf82515e1", None], 567, 567, 35)])
    assert days[D10]["shas"] == ["ee4bf82515e1"]
    assert str_.db_days(days) == {D10}


def _db(d, sym, signal, title, names="true", override=None, sha="ee4bf82515e1", item=None):
    return (_ts(d), sym, signal, sha, override, names, title, item)


def test_db_rows_pass_override_and_names_ticker():
    rows = str_.from_db([
        _db(D10, "NVDA", "SCREENED_IN", "Nvidia CEO steps down", override="held_negative_or_rating"),
        _db(D10, "AMD", "SCREENED_OUT", "AMD downgraded", names="false"),
        _db(D11, "MU", "SCREENED_IN", "Micron upgraded"),           # day not measured from the DB
    ], {D10})
    assert len(rows) == 2
    assert rows[0]["passed"] and rows[0]["override"] and rows[0]["names_ticker"] is True
    assert not rows[1]["passed"] and rows[1]["names_ticker"] is False and rows[1]["origin"] == "db"


_LOG = """\
[2026-11-05 10:00:00,001: INFO/ForkPoolWorker-1] [IONQ] Screening event: news — IonQ CFO Inder Singh Steps Down
[2026-11-05 10:00:00,002: INFO/ForkPoolWorker-2] [AMD] Screening event: news — AMD downgraded at Foo
[2026-11-05 10:00:01,000: INFO/ForkPoolWorker-1] [IONQ] Screener override: held position, news, tone=neg
[2026-11-05 10:00:02,000: INFO/ForkPoolWorker-1] Screened [news] IONQ: relevant=True urgency=eod tone=neg
[2026-11-05 10:00:03,000: INFO/ForkPoolWorker-2] Screened [news] AMD: relevant=False urgency=eod tone=neg
[2026-11-05 10:01:00,000: INFO/ForkPoolWorker-3] [MU] Screening event: options — MU call sweep
[2026-11-05 10:01:01,000: INFO/ForkPoolWorker-3] Screened [options] MU: relevant=True urgency=eod
[2026-11-05 10:02:00,000: INFO/ForkPoolWorker-3] [MU] Screening event: news — Micron upgraded
[2026-11-05 10:02:01,000: INFO/ForkPoolWorker-3] Screened [news] AVGO: relevant=True urgency=eod
[2026-11-10 10:00:00,000: INFO/ForkPoolWorker-1] [NVDA] Screening event: news — Nvidia upgraded
[2026-11-10 10:00:01,000: INFO/ForkPoolWorker-1] Screened [news] NVDA: relevant=True urgency=eod
"""


def test_the_log_pairs_each_result_with_its_workers_event():
    names = []
    rows = str_.parse_log(_LOG.splitlines(), date(2026, 11, 1), {date(2026, 11, 5)},
                          names_ticker=lambda t, title: names.append((t, title)) or t != "AMD")
    assert [(r["ticker"], r["passed"], r["override"]) for r in rows] == [
        ("IONQ", True, True), ("AMD", False, False)]   # options, mismatched ticker, 11-10 left out
    assert rows[0]["title"] == "IonQ CFO Inder Singh Steps Down" and rows[0]["origin"] == "log"
    assert rows[1]["names_ticker"] is False and ("AMD", "AMD downgraded at Foo") in names
    assert str_.parse_log(_LOG.splitlines(), date(2026, 11, 6), {date(2026, 11, 5)},
                          names_ticker=lambda *a: True) == []


def test_log_rows_take_the_days_db_sha():
    rows = [{"day": date(2026, 11, 2)}, {"day": date(2026, 11, 5)}, {"day": date(2026, 11, 7)}]
    str_.assign_log_shas(rows, {date(2026, 11, 2): {"shas": ["344564c47658"]},
                                date(2026, 11, 5): {"shas": ["a", "b"]}})
    assert [r["sha"] for r in rows] == ["344564c47658", "mixed", "unknown"]


def _row(sym, title, passed, names=True, override=False, origin="db", sha="s1", d=D10, item=None):
    return {"day": d, "sha": sha, "ticker": sym, "title": title, "item": item or title,
            "passed": passed, "override": override, "names_ticker": names, "origin": origin}


def test_tally_and_render_rates_kept_apart_by_origin():
    rows = [_row("NVDA", "Nvidia CEO steps down", True, override=True),
            _row("NVDA", "Nvidia CEO steps down", True),                 # a repeat screen
            _row("IONQ", "IonQ CFO resigns", False),
            _row("AMD", "AMD CEO exits, says Foo", False, names=False),
            _row("MU", "Micron upgraded", True, names=None),
            _row("MU", "Micron upgraded", False, origin="log", sha="s0", d=date(2026, 11, 5))]
    groups = str_.tally(rows)
    g = groups[("db", "s1")]
    dep = g["cats"]["exec_departure"]
    assert (dep["named"]["n"], dep["named"]["passed"], dep["named"]["passed_7b"]) == (3, 2, 1)
    assert dep["all"]["n"] == 4 and dep["tickers"]["NVDA"] == 2
    assert g["unknown_named"] == 1 and g["dupes"] == 1
    assert groups[("log", "s0")]["cats"]["upgrade_raise"]["all"]["n"] == 1
    days = {D10: {"shas": ["s1"], "rows": 5, "titled": 5, "passed": 3},
            date(2026, 11, 5): {"shas": ["s0"], "rows": 1, "titled": 0, "passed": 0}}
    text = str_.render(groups, days, {D10}, {date(2026, 11, 5)}, "live", date(2026, 11, 1), "abc",
                       ["x.log"])
    assert "## Prompt s1 (DB rows, 2026-11-10 to 2026-11-10, 1 days)" in text
    assert "## Prompt s0 (LOG rows" in text
    assert "| exec_departure | 3 | 2 (67%) | 1 (33%) | 4 | 2 (50%) | 1 (25%) | NVDA 2, IONQ 1 |" in text
    assert "| offering_dilution | 0 | 0 (-) |" in text
    assert "| 2026-11-05 | s0 | 1 | 0 | 0 | log |" in text
    assert str_.LEXICON_SHA in text and "`exec_departure`" in text


class _Cur:
    def __init__(self, log):
        self.log, self.n = log, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchall(self):
        self.n += 1
        if self.n == 1:
            return [(D10, ["s1"], 2, 2, 1)]
        return [_db(D10, "NVDA", "SCREENED_IN", "Nvidia CEO steps down"),
                _db(D10, "AMD", "SCREENED_OUT", "AMD downgraded")]


class _Conn:
    def __init__(self):
        self.log, self.rolled = [], 0

    def cursor(self):
        return _Cur(self.log)

    def rollback(self):
        self.rolled += 1


def test_main_reads_in_one_read_only_transaction_and_rolls_back(tmp_path):
    conn = _Conn()
    out = tmp_path / "r.md"
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put, \
         patch.object(sys, "argv", ["x", "--out", str(out)]):
        str_.main()
    assert conn.log[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1].startswith("SET LOCAL statement_timeout")
    assert all(s.lstrip().upper().startswith(("SET", "SELECT")) for s in conn.log)
    assert conn.rolled == 1 and put.called
    text = out.read_text()
    assert "| exec_departure | 1 | 1 (100%) |" in text and "| downgrade_cut | 1 | 0 (0%) |" in text
