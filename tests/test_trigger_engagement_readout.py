"""scripts/trigger_engagement_readout.py (audit 2026-11-26 14:53 #1): the
word rules, both bounds, the row build (scheduled reviews out of the rows but
in the word frequencies), the tallies and markdown, the port check and the
read-only fetch, the imports, that weekly_readouts doesn't load it, and the
AI-labelled fixture (tests/fixtures/trigger_engagement_fixture.json) with
each bound's agreement with its labels. No DB.

Fixture agreement (AI labels, not Luke's; description only, and which bound,
if either, is the measure is Luke's call). 40 event verdicts: 23 engaged,
8 not engaged, 9 no new fact.

| bound | engaged: hit / miss | not engaged: hit / miss | no new fact: hit / miss | agreement on the 31 engaged or not engaged |
|---|---|---|---|---|
| loose | 23 / 0 | 4 / 4 | 6 / 3 | 27 of 31 |
| strict | 15 / 8 | 1 / 7 | 4 / 5 | 22 of 31 |

The loose bound's 4 false hits (93504, 94193, 102170, 103593) share a
general word with the text but not the item's claim. The strict bound's
8 misses on engaged rows (93404, 94239, 96807, 100209, 100784, 100867,
102251, 102883) engage by a word that is common in that ticker's verdicts
(huang, jensen, return, ceo, scrutiny, earnings, valuation, leader: each in
at least 25% of them) or by paraphrase ("raised guidance" for "raises full
year outlook")."""
import ast
import importlib.util
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "scripts", "trigger_engagement_readout.py")
_spec = importlib.util.spec_from_file_location("trigger_engagement_readout", _PATH)
ter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ter)

_FIXTURE = os.path.join(_ROOT, "tests", "fixtures", "trigger_engagement_fixture.json")
with open(_FIXTURE) as _f:
    FIX = json.load(_f)


def _tw(title, ticker="NVDA", company=frozenset(), text="", df=None):
    tw = ter.title_words(title, ticker, company)
    for x in tw:
        x.update(ter.match_flags(x["w"], text))
        x["df"] = (df or {}).get(x["w"])
    return tw


# --- word rules -------------------------------------------------------------

def test_words_are_lowercase_and_start_with_a_letter():
    assert ter.words("Nvidia's H200 sales: 3 Big Wins, Q3") == ["nvidia", "h200", "sales", "big", "wins"]
    assert ter.words(None) == []


def test_title_words_drop_stopwords_and_the_ticker_and_flag_company_words():
    tw = ter.title_words("Arm Holdings (ARM) Stock Soars On Arm Royalty Deal", "ARM",
                         frozenset({"arm", "holdings"}))
    assert [(x["w"], x["company"]) for x in tw] == [
        ("holdings", True), ("soars", False), ("royalty", False), ("deal", False)]
    # "arm" is the ticker as well as a company word: dropped as the ticker.


def test_the_stopword_list_is_the_audits():
    assert {"nvidia", "ai", "stock", "the", "barron"} <= ter.STOPWORDS
    assert len(ter.STOPWORDS) == 133 and ter.STOPWORDS_SHA == "66d1610b0a6f"


def test_match_flags_substring_and_prefix():
    assert ter.match_flags("probe", "the sec probes") == {"sub": True, "prefix": False}
    assert ter.match_flags("reinstate", "the reinstatement of") == {"sub": True, "prefix": True}
    assert ter.match_flags("reinstated", "it will reinstate him") == {"sub": False, "prefix": True}
    # Five characters or fewer: no prefix rule.
    assert ter.match_flags("rally", "a rall") == {"sub": False, "prefix": False}


def test_verdict_text_reads_string_values_not_keys():
    mo = {"reasoning": "Alpha", "thesis_summary": "Beta",
          "key_risks": ["Gamma", {"risk": "Delta"}],
          "expectation_gap": {"market_implied": "Epsilon", "n": 3},
          "catalyst": {"event": "Zeta"}}
    text = ter.verdict_text(mo)
    assert text == "alpha beta gamma delta epsilon"
    assert "market_implied" not in text and "zeta" not in text
    assert ter.verdict_text(None) == ""


# --- bounds -----------------------------------------------------------------

def test_loose_hits_on_substring_or_prefix_and_ignores_company_words():
    assert ter.loose_hit(_tw("Nvidia probe widens", text="the probe")) is True
    assert ter.loose_hit(_tw("Lumentum outage", "LITE", frozenset({"lumentum"}),
                             text="lumentum leads")) is False
    assert ter.loose_hit(_tw("Lumentum", "LITE", frozenset({"lumentum"}))) is None
    assert ter.loose_hit(_tw("The Nvidia")) is None


def test_strict_counts_only_rare_words_and_none_when_there_is_none():
    df = {"return": 0.6, "probe": 0.1}
    assert ter.strict_hit(_tw("Return and probe", text="return", df=df)) is False
    assert ter.strict_hit(_tw("Return and probe", text="probe", df=df)) is True
    assert ter.strict_hit(_tw("Return", text="return", df=df)) is None
    assert ter.strict_hit(_tw("Return", text="return", df=df), rare_df=0.7) is True
    # A prefix match alone never counts for strict.
    assert ter.strict_hit(_tw("Reinstated", text="reinstate", df={"reinstated": 0.0})) is False


# --- rows, tallies, markdown ------------------------------------------------

def _raw(sid, sym, path, title, text, sig="HOLD", src="news", nf="true"):
    return (sid, datetime(2026, 11, 26, 15, sid % 60, tzinfo=timezone.utc), sym, sig, path, src,
            nf, title, {"reasoning": text})


def test_build_rows_counts_reviews_in_df_but_leaves_them_out():
    raw = [_raw(1, "NVDA", "event", "Nvidia probe widens", "probe risk"),
           _raw(2, "NVDA", "scheduled_review", "Periodic position review", "probe again"),
           _raw(3, "NVDA", "event", "Nvidia outage", "nothing here"),
           _raw(4, "NVDA", "event", "Nvidia widens lead", "probe")]
    rows = ter.build_rows(raw, {"NVDA": frozenset({"nvidia"})})
    assert [r["id"] for r in rows] == [1, 3, 4]
    r1 = rows[0]
    probe = next(x for x in r1["title_words"] if x["w"] == "probe")
    assert probe["df"] == 0.75 and r1["n_ticker_verdicts"] == 4      # 3 of 4 texts, review included
    # probe is in the text but common (0.75); widens is rare (0) and absent.
    assert (r1["loose"], r1["strict"]) == (True, False)
    assert rows[1]["loose"] is False and rows[1]["strict"] is False
    assert rows[1]["company_known"] is True


def test_build_rows_accepts_a_json_string_model_output():
    raw = [(1, "2026-11-26", "NVDA", "HOLD", "event", "news", "true", "Probe", '{"reasoning": "probe"}')]
    assert ter.build_rows(raw, {})[0]["loose"] is True


def test_tally_and_render_print_counts_and_no_titles():
    raw = [_raw(1, "NVDA", "event", "Nvidia probe widens $5 billion", "probe", nf="true"),
           _raw(2, "NVDA", "event", "Nvidia outage", "x", nf="false", sig="ADD"),
           _raw(3, "AMD", "event", "AMD", "x", nf=None),
           _raw(4, "NVDA", "earnings_review", "Earnings review", "earnings")]
    rows = ter.build_rows(raw, {"NVDA": frozenset({"nvidia"})})
    t = ter.tally(rows, lambda r: r["path"])
    assert t["event"]["n"] == 3 and t["event"]["no_words"] == 1 and t["event"]["loose_miss"] == 1
    md = ter.render(rows, "live", 14, datetime(2026, 11, 12, tzinfo=timezone.utc), "abc1234", 2)
    assert "probe" not in md and "outage" not in md and "$" not in md
    assert "2 scheduled reviews left out" in md and "research choice (Luke)" in md
    assert "Tickers with no watchlist name (company words not dropped): AMD" in md
    assert md.count("\n## ") == 4


def test_company_names_reads_every_ticker_row(tmp_path):
    p = tmp_path / "watchlist.md"
    p.write_text("| Ticker | Company | Role | Tier |\n|---|---|---|---|\n"
                 "| NVDA | Nvidia | Compounder | Core |\n| INTC | Intel Corp | Watch | Watch |\n")
    assert ter.company_names(str(p)) == {"NVDA": frozenset({"nvidia"}),
                                         "INTC": frozenset({"intel", "corp"})}
    assert ter.company_names(str(tmp_path / "missing.md")) == {}


# --- main: port, read-only ---------------------------------------------------

def test_a_wrong_port_reads_nothing(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert ter.main([]) == 2
    assert "6543" in capsys.readouterr().err


class _Cur:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchall(self):
        return [_raw(1, "NVDA", "event", "Nvidia probe", "probe")]

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


def test_main_reads_in_one_read_only_transaction(monkeypatch, tmp_path):
    import db
    conn, put = _Conn(), []
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    monkeypatch.setattr(db, "_get_conn", lambda: conn)
    monkeypatch.setattr(db, "_put_conn", lambda c: put.append(c))
    out, js = tmp_path / "o.md", tmp_path / "o.json"
    assert ter.main(["--out", str(out), "--json", str(js), "--watchlist", str(tmp_path / "none")]) == 0
    assert conn.log[0] == "SET TRANSACTION READ ONLY" and conn.rolled_back and put == [conn]
    assert not any(re.match(r"\s*(INSERT|UPDATE|DELETE)", s, re.I) for s in conn.log)
    rec = json.loads(js.read_text())
    assert rec["rare_df"] == 0.25 and rec["rows"][0]["loose"] is True
    assert "# Trigger engagement readout" in out.read_text()


def test_imports_are_db_and_stdlib_only():
    tree = ast.parse(open(_PATH).read())
    mods = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert mods <= {"__future__", "argparse", "hashlib", "json", "os", "re", "subprocess", "sys",
                    "collections", "datetime", "db", "config"}


def test_weekly_readouts_does_not_load_it():
    with open(os.path.join(_ROOT, "weekly_readouts.py")) as f:
        assert "trigger_engagement" not in f.read()


# --- the AI-labelled fixture --------------------------------------------------

def test_the_fixture_is_ai_labelled_and_masked():
    meta, rows = FIX["meta"], FIX["rows"]
    assert meta["labels"].startswith("AI-labelled") and "not Luke's" in meta["labels"]
    assert len(rows) == len({r["id"] for r in rows}) == 40
    assert Counter(r["label"] for r in rows) == {"engaged": 23, "not_engaged": 8, "no_new_fact": 9}
    assert all(r["label_note"] for r in rows)
    for r in rows:
        txt = r["title"] + " " + " ".join(s for v in r["fields"].values() for s in v)
        assert not re.search(r"(?<![A-Za-z0-9_\-])\d|\$\s?\d", txt), r["id"]


def _fixture_bounds():
    out = {}
    for r in FIX["rows"]:
        tw = ter.title_words(r["title"], r["ticker"], frozenset(r["company"]))
        text = ter.verdict_text(r["fields"])
        for x in tw:
            x.update(ter.match_flags(x["w"], text))
            x["df"] = r["df"].get(x["w"])
        out[r["id"]] = (r["label"], ter.loose_hit(tw), ter.strict_hit(tw))
    return out


def test_the_fixture_reproduces_the_bounds_read_from_the_db():
    b = _fixture_bounds()
    for r in FIX["rows"]:
        assert b[r["id"]][1:] == (r["loose_at_read"], r["strict_at_read"]), r["id"]


def test_each_bounds_agreement_with_the_labels():
    b = _fixture_bounds()
    word = {True: "hit", False: "miss", None: "none"}
    loose = Counter((lab, word[lo]) for lab, lo, _ in b.values())
    strict = Counter((lab, word[st]) for lab, _, st in b.values())
    assert loose == {("engaged", "hit"): 23, ("not_engaged", "hit"): 4, ("not_engaged", "miss"): 4,
                     ("no_new_fact", "hit"): 6, ("no_new_fact", "miss"): 3}
    assert strict == {("engaged", "hit"): 15, ("engaged", "miss"): 8, ("not_engaged", "hit"): 1,
                      ("not_engaged", "miss"): 7, ("no_new_fact", "hit"): 4, ("no_new_fact", "miss"): 5}
    agree = lambda c: c[("engaged", "hit")] + c[("not_engaged", "miss")]
    assert (agree(loose), agree(strict)) == (27, 22)
    assert {s for s, (lab, lo, _) in b.items() if lab == "not_engaged" and lo} == \
        {93504, 94193, 102170, 103593}
    assert {s for s, (lab, _, st) in b.items() if lab == "engaged" and st is False} == \
        {93404, 94239, 96807, 100209, 100784, 100867, 102251, 102883}
