"""scripts/memory_relevance_readout.py: the stored prompts' memory sections
parsed into per-verdict counts (items, scores, below 0, on company,
collections), tallied per held, day, path and ticker; the read is read-only
on the pooler port, returns only the memory block and skips the stack's
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
                     "scripts", "memory_relevance_readout.py")
_spec = importlib.util.spec_from_file_location("memory_relevance_readout", _PATH)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
NOW = datetime(2026, 11, 26, 23, 0, tzinfo=NY)
SECRET = "ZZQX-SENTINEL-TEXT"   # must never reach the output
C = mr.news_constants()
NAMES = {"ANET": "Arista Networks", "NOW": "ServiceNow", "NVDA": "NVIDIA"}


def _item(coll, score, text, day="2026-11-20"):
    return f"[{coll} | score={score} | {day}]\n{text}\n"


def _block(*items, after="=== RELATED SYMBOLS IN WATCHLIST ===\nANET may have"):
    return "\n".join([mr.MEMORY_HEADER, *items]) + ("\n\n" + after if after else "")


ANET_BLOCK = _block(
    _item("NEWS", "0.41", f"Arista Networks beats estimates {SECRET}"),
    _item("FILINGS", "0.12", f"8-K filed by Other Corp: Item 2.02 {SECRET}"),
    _item("ARXIV", "-0.08", f"Perovskite stability under strain {SECRET}"),
    _item("OPTIONS", "-0.00", f"$ANET call sweep {SECRET}"),
)


def _row(sid, ts, ticker="ANET", path="event", held="true", prompt_null=False,
         block=ANET_BLOCK, empty=False):
    return (sid, ts, ticker, path, held, prompt_null, block, empty)


TS = datetime(2026, 11, 26, 15, 17, tzinfo=UTC)


# --- parsing the memory block -------------------------------------------------

def test_parse_reads_each_item_header_and_stops_at_the_next_section():
    items = mr.parse_memory(ANET_BLOCK)
    assert [i["collection"] for i in items] == ["NEWS", "FILINGS", "ARXIV", "OPTIONS"]
    assert [i["score"] for i in items] == [0.41, 0.12, -0.08, -0.0]
    assert [i["date"] for i in items] == ["2026-11-20"] * 4
    assert "RELATED SYMBOLS" not in items[-1]["text"] and "may have" not in items[-1]["text"]
    assert items[0]["text"].startswith("Arista Networks beats")


def test_a_printed_minus_zero_counts_as_below_zero():
    items = mr.parse_memory(ANET_BLOCK)
    assert [i["below0"] for i in items] == [False, False, True, True]


def test_item_text_with_blank_lines_stays_with_its_item():
    block = _block(_item("NEWS", "0.30", "line one\n\nline two"), _item("MACRO", "0.10", "cpi"))
    items = mr.parse_memory(block)
    assert len(items) == 2 and "line two" in items[0]["text"]


def test_a_block_at_the_end_of_the_slice_keeps_its_last_item():
    items = mr.parse_memory(_block(_item("NEWS", "0.30", "a"), _item("NEWS", "0.20", "b"), after=None))
    assert [i["text"] for i in items] == ["a", "b"]


@pytest.mark.parametrize("block", [None, "", mr.MEMORY_HEADER, "=== OTHER ===\n[NEWS | score=0.1 | x]\n"])
def test_no_block_or_a_header_alone_has_no_items(block):
    assert mr.parse_memory(block) == []


def test_a_header_line_after_the_next_section_is_not_an_item():
    block = _block(_item("NEWS", "0.30", "a"),
                   after="=== YOUR TASK ===\n[NEWS | score=0.99 | 2026-11-01]\nnot memory")
    assert [i["score"] for i in mr.parse_memory(block)] == [0.3]


# --- on company: mentions_company's rule ------------------------------------

def test_the_constants_match_ingestion_news():
    import ingestion.news as news
    for k in mr._NEWS_CONSTANTS:
        assert C[k] == getattr(news, k), k
    assert C["_NAME_SUFFIXES"].pattern == news._NAME_SUFFIXES.pattern
    assert C["_NAME_SUFFIXES"].flags == news._NAME_SUFFIXES.flags


_CASES = [
    ("ANET", "Arista Networks", "Arista Networks posts record quarter"),
    ("ANET", "Arista Networks", "arista networks guidance"),
    ("ANET", "Arista Networks", "Arista beats"),                       # alias
    ("ANET", "Arista Networks", "ANET shares rise"),
    ("ANET", "Arista Networks", "Cisco and Juniper update"),
    ("ANET", "", "8-K filed by ANET"),
    ("ANET", "", "8-K filed by Juniper"),
    ("NOW", "ServiceNow", "Now is the time to buy"),
    ("NOW", "ServiceNow", "NOW Stock jumps"),
    ("NOW", "ServiceNow", "BUY NOW STOCK TODAY"),
    ("NOW", "ServiceNow", "($NOW) earnings"),
    ("NOW", "ServiceNow", "ServiceNow, Inc. 10-Q"),
    ("NOW", "ServiceNow", "NOW"),
    ("NVDA", "NVIDIA Corporation", "Nvidia ships Blackwell"),
    ("NVDA", "NVIDIA Corporation", "NASDAQ: NVDA call sweep"),
    ("NVDA", "NVIDIA Corporation", "NVDAX is a fund"),
    ("META", "Meta Platforms", "meta-analysis of returns"),
    ("META", "Meta Platforms", "Meta raises capex"),
    ("QUBT", "Quantum Computing Inc", "quantum computing paper on arXiv"),
    ("QUBT", "Quantum Computing Inc", "QCi wins contract"),
    ("GOOG", "Alphabet Inc", "Google cloud deal"),
    ("A", "Agilent Technologies", "A new 8-K"),
    ("", "", "anything"),
]


@pytest.mark.parametrize("ticker, company, text", _CASES)
def test_on_company_agrees_with_mentions_company(ticker, company, text):
    import ingestion.news as news
    assert mr.on_company(ticker, company, text, C) == news.mentions_company(ticker, company, text)


def test_on_company_examples():
    assert mr.on_company("ANET", "Arista Networks", "Arista Networks beats", C)
    assert not mr.on_company("ANET", "Arista Networks", "8-K filed by Other Corp", C)
    assert not mr.on_company("NOW", "ServiceNow", "Now is the time", C)


def test_news_constants_names_what_is_missing(tmp_path):
    p = tmp_path / "news.py"
    p.write_text("_NAME_ALIASES = {}\n_NAME_OVERRIDES = {}\n")
    with pytest.raises(ValueError, match="_WORDLIKE_NAMES"):
        mr.news_constants(str(p))


def test_company_names_read_the_first_row_of_each_ticker(tmp_path):
    p = tmp_path / "w.md"
    p.write_text("| TICKER | Company | Role |\n|---|---|---|\n| ANET | Arista Networks | Core |\n"
                 "| ANET | Something Else | Watch |\nprose | not a row\n| NVDA | NVIDIA | Core |\n")
    assert mr.company_names(str(p)) == {"ANET": "Arista Networks", "NVDA": "NVIDIA"}
    assert mr.company_names(str(tmp_path / "missing.md")) == {}


def test_the_repo_watchlist_names_the_held_tickers():
    names = mr.company_names()
    assert names.get("ANET") == "Arista Networks" and "NVDA" in names


# --- per verdict and per cell -----------------------------------------------

def test_classify_counts_items_scores_company_and_forms_with_no_text():
    r = mr.classify(_row(7, TS), NAMES, C)
    assert r["id"] == 7 and r["day"] == "2026-11-26" and r["ticker"] == "ANET"
    assert r["held"] == "held" and r["path"] == "event" and r["missing"] is None
    assert r["n"] == 4 and r["scores"] == [0.41, 0.12, -0.08, -0.0]
    assert r["below0"] == [False, False, True, True]
    assert r["on_company"] == [True, False, False, True]
    assert r["collections"] == ["NEWS", "FILINGS", "ARXIV", "OPTIONS"]
    assert r["filing_forms"] == {"8-K": 1}
    assert SECRET not in json.dumps(r)


def test_classify_uses_the_new_york_day():
    r = mr.classify(_row(1, datetime(2026, 11, 27, 3, 0, tzinfo=UTC)), NAMES, C)
    assert r["day"] == "2026-11-26"
    naive = mr.classify(_row(1, datetime(2026, 11, 27, 3, 0)), NAMES, C)   # naive = UTC
    assert naive["day"] == "2026-11-26"


@pytest.mark.parametrize("kw, missing", [
    ({"prompt_null": True, "block": None}, "prompt NULL"),
    ({"block": None, "empty": True}, "empty memory"),
    ({"block": None}, "no memory section"),
    ({"block": mr.MEMORY_HEADER + "\n\n=== YOUR TASK ===\n"}, "no memory section"),
])
def test_a_verdict_without_items_gets_a_reason(kw, missing):
    r = mr.classify(_row(1, TS, **kw), NAMES, C)
    assert r["n"] == 0 and r["missing"] == missing


@pytest.mark.parametrize("v, label", [("true", "held"), ("false", "not held"), (None, "unknown"),
                                      ("True", "held"), ("maybe", "unknown")])
def test_held_labels(v, label):
    assert mr.held_label(v) == label


def test_a_ticker_with_no_watchlist_name_is_flagged():
    r = mr.classify(_row(1, TS, ticker="ZZZZ"), NAMES, C)
    assert r["has_company_name"] is False
    assert mr.classify(_row(1, TS), NAMES, C)["has_company_name"] is True


def test_tally_counts_items_below_zero_and_on_company():
    rows = [mr.classify(_row(1, TS), NAMES, C),
            mr.classify(_row(2, TS, block=_block(_item("NEWS", "0.50", "Arista Networks"),
                                                 _item("MACRO", "0.20", "CPI print"))), NAMES, C),
            mr.classify(_row(3, TS, block=None), NAMES, C)]
    t = mr.tally(rows)
    assert t["verdicts"] == 3 and t["with_items"] == 2 and t["items"] == 6
    assert t["below0"] == 2 and t["below0_pct"] == 33.3
    assert t["on_company"] == 3 and t["on_company_pct"] == 50.0
    assert t["missing"] == {"prompt NULL": 0, "empty memory": 0, "no memory section": 1}
    # per verdict: below-0 fractions 0.5 and 0.0; on-company 0.5 and 0.5; min scores -0.08, 0.2
    assert t["verdict_below0_frac_median"] == 0.25
    assert t["verdict_on_company_frac_median"] == 0.5
    assert t["verdict_min_score_median"] == 0.06
    assert t["score"] == {"n": 6, "min": -0.08, "p25": 0.03, "median": 0.16, "p75": 0.36, "max": 0.5}
    assert t["collections"] == {"NEWS": 2, "FILINGS": 1, "ARXIV": 1, "OPTIONS": 1, "MACRO": 1}
    assert t["filing_forms"] == {"8-K": 1}


def test_a_score_of_exactly_zero_is_not_below_zero():
    r = mr.classify(_row(1, TS, block=_block(_item("NEWS", "0.00", "x"))), NAMES, C)
    assert r["below0"] == [False]


def test_an_empty_cell_has_no_percent_and_no_distribution():
    t = mr.tally([mr.classify(_row(1, TS, block=None), NAMES, C)])
    assert t["items"] == 0 and t["below0_pct"] is None and t["on_company_pct"] is None
    assert t["score"] == {"n": 0} and t["verdict_min_score_median"] is None


def test_a_filing_with_no_form_in_its_first_60_characters_is_other():
    block = _block(_item("FILINGS", "0.10", "x" * 61 + " 8-K"), _item("FILINGS", "0.10", "10-Q for Q3"))
    assert mr.classify(_row(1, TS, block=block), NAMES, C)["filing_forms"] == {"other": 1, "10-Q": 1}


def test_collection_tally_counts_each_collection():
    rows = [mr.classify(_row(1, TS), NAMES, C), mr.classify(_row(2, TS), NAMES, C)]
    ct = mr.collection_tally(rows)
    assert list(ct) == ["ARXIV", "FILINGS", "NEWS", "OPTIONS"]   # ties by name
    assert ct["OPTIONS"] == {"items": 2, "below0": 2, "below0_pct": 100.0, "on_company": 2,
                             "on_company_pct": 100.0, "off_company": 0,
                             "score": {"n": 2, "min": -0.0, "p25": 0.0, "median": 0.0, "p75": 0.0,
                                       "max": -0.0}}
    assert ct["FILINGS"]["off_company"] == 2 and ct["FILINGS"]["on_company_pct"] == 0.0


def test_build_groups_by_held_day_path_ticker_and_collections():
    rows = [mr.classify(_row(1, TS), NAMES, C),
            mr.classify(_row(2, datetime(2026, 11, 25, 15, 0, tzinfo=UTC), ticker="NVDA",
                             path="scheduled_review", held="false"), NAMES, C),
            mr.classify(_row(3, TS, held=None), NAMES, C)]
    out = mr.build(rows)
    assert list(out["by_held"]) == ["held", "not held", "unknown"]
    assert list(out["by_day"]) == ["2026-11-25", "2026-11-26"]
    assert set(out["by_held_day"]) == {"held / 2026-11-26", "not held / 2026-11-25",
                                       "unknown / 2026-11-26"}
    assert list(out["by_path"]) == ["event", "scheduled_review"]
    assert list(out["by_ticker"]) == ["ANET", "NVDA"]
    assert set(out["collections_by_held"]) == {"held", "not held", "unknown"}
    assert out["total"]["verdicts"] == 3


def _render(rows):
    return mr.render(mr.build(rows), "live", date(2026, 11, 20), date(2026, 11, 27), "abc", NOW)


def test_render_has_every_section_no_text_and_no_relevance_words():
    text = _render([mr.classify(_row(1, TS), NAMES, C)])
    for title in ("By held", "By day", "Held x day", "Collections, all verdicts",
                  "Collections, held verdicts", "By trigger path", "By ticker"):
        assert f"## {title}" in text
    assert SECRET not in text and "Perovskite" not in text and "Arista Networks beats" not in text
    low = text.lower()
    for w in ("irrelevant", "noise", "should"):
        assert w not in low
    assert "Code version: abc" in text and "2026-11-26 23:00 ET" in text


def test_every_table_row_has_its_header_width():
    text = _render([mr.classify(_row(1, TS), NAMES, C), mr.classify(_row(2, TS, block=None), NAMES, C)])
    tables, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    assert len(tables) == 7
    for t in tables:
        assert len({l.count("|") for l in t}) == 1, t[0]


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
    assert mr.main(["--since", "2026-11-20", "--until", "2026-11-27", "--out", str(out),
                    "--json", str(js)], now=NOW) == 0
    sqls = [s for s, _ in conn.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY"
    assert conn.log[1] == ("SET LOCAL statement_timeout = %s", (60000,))
    assert sqls[2] == mr._VERDICTS_SQL and len(sqls) == 3
    params = conn.log[2][1]
    assert params["env"] == "live" and params["slice"] == 12000
    assert params["hdr"] == mr.MEMORY_HEADER and params["empty"] == mr.EMPTY_HEADER
    assert params["mac"] == mr.MACRO_HEADER and params["mac_slice"] == 2000
    assert params["start"] == datetime(2026, 11, 20, tzinfo=NY)
    assert params["end"] == datetime(2026, 11, 27, tzinfo=NY)
    assert conn.rolled == 1 and put == [conn]
    text = out.read_text()
    assert "items: 4" in text and SECRET not in text
    data = json.loads(js.read_text())
    assert data["verdicts"][0]["scores"] == [0.41, 0.12, -0.08, -0.0]
    assert SECRET not in js.read_text()


def test_the_sql_returns_only_the_memory_block_of_verdicts():
    sql = mr._VERDICTS_SQL
    assert "s.decision ? 'model_output'" in sql
    assert "(s.trading_env = %(env)s OR s.trading_env IS NULL)" in sql
    assert "s.timestamp >= %(start)s AND s.timestamp < %(end)s" in sql
    assert "substr(s.prompt, strpos(s.prompt, %(hdr)s), %(slice)s)" in sql
    assert "substr(s.prompt, strpos(s.prompt, %(mac)s), %(mac_slice)s)" in sql
    assert "s.provenance->'plan'->>'held'" in sql and "s.provenance->'trigger'->>'path'" in sql
    # the prompt column is never selected whole: each use is a test or the slice
    rest = sql
    for use in ("s.prompt IS NULL", "strpos(s.prompt, %(hdr)s)", "substr(s.prompt,",
                "strpos(s.prompt, %(empty)s)", "strpos(s.prompt, %(mac)s)"):
        rest = rest.replace(use, "")
    assert "prompt" not in rest
    assert "price" not in sql


def test_the_empty_header_is_context_engines():
    src = open(os.path.join(mr.ROOT, "context_engine.py")).read()
    assert '"=== RELEVANT HISTORICAL CONTEXT ===\\nNo prior history found' in src
    assert src.count(f'["{mr.MEMORY_HEADER}"]') == 2
    assert src.count('f"[{source.upper()} | score={score:.2f} | {date[:10]}]\\n"') == 2


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert mr.main([], now=NOW) == 2
    assert "6543" in capsys.readouterr().err


@pytest.mark.parametrize("hm, refused", [((4, 49), False), ((4, 50), True), ((5, 11), True),
                                         ((5, 12), False)])
def test_main_does_not_read_at_the_stacks_start(monkeypatch, capsys, hm, refused):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    now = datetime(2026, 11, 27, *hm, tzinfo=NY)
    rc = mr.main(["--since", "2026-11-20", "--until", "2026-11-27", "--out", os.devnull], now=now)
    assert (rc == 2) is refused and (conn.log == []) is refused
    if refused:
        assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(monkeypatch, capsys):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    monkeypatch.delenv("SUPABASE_HOST")
    assert mr.main([], now=NOW) == 2 and conn.log == []
    assert "Not connecting" in capsys.readouterr().err


def test_main_rejects_an_empty_window(monkeypatch):
    _fake_db(monkeypatch, _Conn())
    assert mr.main(["--since", "2026-11-26", "--until", "2026-11-26"], now=NOW) == 2


def test_main_defaults_the_window(monkeypatch):
    conn = _Conn()
    _fake_db(monkeypatch, conn)
    assert mr.main(["--out", os.devnull], now=NOW) == 0
    assert conn.log[2][1]["start"] == datetime(2026, 11, 20, tzinfo=NY)
    assert conn.log[2][1]["end"] == datetime(2026, 11, 27, tzinfo=NY)


def test_a_failed_read_still_rolls_back_and_returns_the_conn(monkeypatch):
    class Boom(_Conn):
        def cursor(self):
            raise RuntimeError("canceling statement due to statement timeout")
    conn = Boom()
    put = _fake_db(monkeypatch, conn)
    with pytest.raises(RuntimeError):
        mr.main(["--out", os.devnull], now=NOW)
    assert conn.rolled == 1 and put == [conn]


def test_db_root_prefers_a_working_directory_with_db_py(tmp_path):
    assert mr.db_root(str(tmp_path)) == mr.ROOT
    (tmp_path / "db.py").write_text("")
    assert mr.db_root(str(tmp_path)) == str(tmp_path)


def test_imports_are_db_and_the_standard_library():
    src = open(_PATH).read()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines()
               if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imports <= {"argparse", "ast", "json", "os", "re", "statistics", "subprocess", "sys",
                       "collections", "datetime", "zoneinfo"}
    assert "    import db\n" in src
    lines = src.splitlines()
    assert not any(l.strip().startswith(("import ingestion", "from ingestion")) for l in lines)



# --- stale macro blocks (audit 2026-11-27 00:53 #3) --------------------------

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
GOLDEN = os.path.join(_FIXTURES, "memory_relevance_golden_4244e839.md")
LIVE_VIX, OLD_VIX = "17.12", "23.47"     # distinctive levels; never printed


def _macro_text(vix=OLD_VIX, as_of="2026-09-24"):
    return (f"{mr.MACRO_HEADER}\nFed Funds Rate: data unavailable\n"
            f"VIX Volatility Index: {vix} (as of {as_of})\nUnemployment Rate: 4.4 (as of 2026-08-01)")


LIVE_MACRO = _macro_text(LIVE_VIX, "2026-11-25") + "\n\n=== YOUR TRACK RECORD ===\nVIX Volatility Index: 99.0"


def _golden_rows():
    def item(coll, score, text, day="2026-11-20"):
        return f"[{coll} | score={score} | {day}]\n{text}\n"

    def block(*items):
        return "\n".join([mr.MEMORY_HEADER, *items]) + "\n\n=== RELATED SYMBOLS IN WATCHLIST ===\nx"

    a = block(item("NEWS", "0.41", "Arista Networks beats"), item("FILINGS", "0.12", "8-K filed by Other Corp"),
              item("ARXIV", "-0.08", "Perovskite stability"), item("OPTIONS", "-0.00", "$ANET call sweep"))
    b = block(item("NEWS", "0.50", "Nvidia ships"), item("MACRO", "-0.14", "CPI print"),
              item("FILINGS", "0.20", "10-Q for NVIDIA Corporation"), item("INSIDER", "-0.21", "Form 4 by director"))
    return [
        (1, datetime(2026, 11, 25, 15, 0, tzinfo=UTC), "ANET", "event", "true", False, a, False),
        (2, datetime(2026, 11, 26, 15, 0, tzinfo=UTC), "NVDA", "scheduled_review", "false", False, b, False),
        (3, datetime(2026, 11, 26, 16, 0, tzinfo=UTC), "NVDA", "scan", None, False, None, True),
        (4, datetime(2026, 11, 26, 17, 0, tzinfo=UTC), "ANET", "event", "true", True, None, False),
        (5, datetime(2026, 11, 27, 3, 0, tzinfo=UTC), "ANET", "event", "true", False, None, False),
    ]


_GNAMES = {"ANET": "Arista Networks", "NVDA": "NVIDIA Corporation"}
_GNOW = datetime(2026, 11, 27, 1, 0, tzinfo=NY)


def _golden_out(rows):
    return mr.build([mr.classify(r, _GNAMES, C) for r in rows])


def test_the_original_output_is_byte_identical_to_4244e839():
    # tests/fixtures/memory_relevance_golden_4244e839.md is render()'s output
    # from 4244e839 (before the stale macro columns) on _golden_rows(), with
    # the code version pinned to "pinned0".
    with open(GOLDEN) as f:
        golden = f.read()
    text = mr.render(_golden_out(_golden_rows()), "live", date(2026, 11, 20), date(2026, 11, 27),
                     "pinned0", _GNOW)
    assert text == golden
    # A macro slice on the row (the new SQL column) changes nothing above the new section.
    with_macro = [r + (LIVE_MACRO,) for r in _golden_rows()]
    assert mr.render(_golden_out(with_macro), "live", date(2026, 11, 20), date(2026, 11, 27),
                     "pinned0", _GNOW) == golden


def test_main_writes_the_golden_output_then_the_new_section(monkeypatch, tmp_path):
    rows = [r + (LIVE_MACRO,) for r in _golden_rows()]
    conn = _Conn(rows)
    _fake_db(monkeypatch, conn)
    monkeypatch.setattr(mr.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="pinned0\n"))
    monkeypatch.setattr(mr, "company_names", lambda path=None: _GNAMES)
    out = tmp_path / "r.md"
    assert mr.main(["--since", "2026-11-20", "--until", "2026-11-27", "--out", str(out)], now=_GNOW) == 0
    with open(GOLDEN) as f:
        golden = f.read()
    text = out.read_text()
    assert text == golden + mr.render_stale(_golden_out(rows))
    assert text[len(golden):].startswith("\n## Stale macro blocks\n")


def _stale_block():
    return _block(_item("NEWS", "0.41", "Arista Networks beats"),
                  _item("MACRO", "-0.12", _macro_text(), day="2026-09-24"),
                  _item("FILINGS", "0.10", "8-K filed by Other Corp"),
                  _item("MACRO", "-0.20", _macro_text("data unavailable"), day="2026-11-01"),
                  _item("MACRO", "-0.05", "CPI print"))


def test_an_item_whose_text_starts_with_a_header_does_not_end_the_block():
    items = mr.parse_memory(_stale_block())
    assert [i["collection"] for i in items] == ["NEWS", "MACRO", "FILINGS", "MACRO", "MACRO"]
    assert items[1]["text"].startswith(mr.MACRO_HEADER) and "VIX Volatility Index" in items[1]["text"]
    assert "RELATED SYMBOLS" not in items[-1]["text"]


def test_a_stale_block_as_the_last_item_ends_at_the_real_section():
    block = _block(_item("NEWS", "0.4", "a"), _item("MACRO", "-0.1", _macro_text()),
                   after="=== YOUR TASK ===\nAnswer these questions")
    items = mr.parse_memory(block)
    assert len(items) == 2 and "YOUR TASK" not in items[1]["text"]


def test_section_vix_reads_only_its_own_section():
    assert mr.section_vix(LIVE_MACRO) == 17.12
    assert mr.section_vix(_macro_text("data unavailable")) is None
    assert mr.section_vix(mr.MACRO_HEADER + "\nCPI: 3\n\n=== X ===\nVIX Volatility Index: 30") is None
    assert mr.section_vix(None) is None and mr.section_vix("VIX Volatility Index: 20") is None


def test_classify_counts_stale_blocks_ages_and_vix_diffs():
    r = mr.classify(_row(1, TS, block=_stale_block()) + (LIVE_MACRO,), NAMES, C)
    assert r["n"] == 5 and r["stale_macro_blocks"] == 2
    assert r["stale_macro_ages_days"] == [63, 25] and r["stale_macro_max_age_days"] == 63
    assert r["live_vix_seen"] is True
    assert r["stale_vix_abs_diff_pts"] == [round(23.47 - 17.12, 2)] == [6.35]
    dumped = json.dumps(r)
    assert OLD_VIX not in dumped and LIVE_VIX not in dumped


def test_no_live_vix_means_no_diff():
    r = mr.classify(_row(1, TS, block=_stale_block()), NAMES, C)
    assert r["stale_macro_blocks"] == 2 and r["live_vix_seen"] is False
    assert r["stale_vix_abs_diff_pts"] == []


def test_a_macro_item_without_the_header_is_not_stale():
    r = mr.classify(_row(1, TS, block=_block(_item("MACRO", "-0.1", "CPI print"),
                                             _item("NEWS", "0.2", mr.MACRO_HEADER))) + (LIVE_MACRO,), NAMES, C)
    assert r["stale_macro_blocks"] == 0 and r["stale_macro_max_age_days"] is None


def test_an_unparsed_printed_date_has_no_age():
    r = mr.classify(_row(1, TS, block=_block(_item("MACRO", "-0.1", _macro_text(), day=""))), NAMES, C)
    assert r["stale_macro_ages_days"] == [None] and r["stale_macro_max_age_days"] is None


def _stale_rows():
    return [mr.classify(_row(1, TS, block=_stale_block()) + (LIVE_MACRO,), NAMES, C),
            mr.classify(_row(2, datetime(2026, 11, 25, 15, 0, tzinfo=UTC), path="scan", held="false")
                        + (LIVE_MACRO,), NAMES, C),
            mr.classify(_row(3, TS, block=None) + (None,), NAMES, C)]


def test_stale_tally_counts_blocks_ages_and_diffs():
    t = mr.stale_tally(_stale_rows())
    assert t["verdicts"] == 3 and t["with_blocks"] == 1 and t["with_blocks_pct"] == 33.3
    assert t["blocks"] == 2 and t["blocks_per_verdict_median"] == 2
    assert t["max_age"] == {"n": 1, "min": 63, "p25": 63, "median": 63, "p75": 63, "max": 63}
    assert t["age"]["n"] == 2 and t["age"]["median"] == 44
    assert t["vix_diff"]["n"] == 1 and t["vix_diff"]["max"] == 6.35
    assert t["blocks_without_vix_diff"] == 1


def test_an_empty_stale_cell():
    t = mr.stale_tally([])
    assert t["verdicts"] == 0 and t["with_blocks_pct"] is None and t["max_age"] == {"n": 0}


def test_build_groups_stale_blocks_by_held_day_and_path():
    out = mr.build(_stale_rows())
    assert list(out["stale_by_held"]) == ["held", "not held"]
    assert list(out["stale_by_day"]) == ["2026-11-25", "2026-11-26"]
    assert list(out["stale_by_path"]) == ["event", "scan"]
    assert out["stale_by_held"]["held"]["blocks"] == 2 and out["stale_total"]["blocks"] == 2


def test_render_stale_has_its_tables_no_vix_level_and_no_label():
    text = mr.render_stale(mr.build(_stale_rows()))
    assert text.startswith("\n## Stale macro blocks\n")
    for title in ("By held", "By day", "By trigger path"):
        assert f"### {title}" in text
    assert OLD_VIX not in text and LIVE_VIX not in text and "6.35" in text
    low = text.lower()
    for w in ("irrelevant", "noise", "should", "wrong"):
        assert w not in low
    tables, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    assert len(tables) == 3
    for t in tables:
        assert len({l.count("|") for l in t}) == 1, t[0]


def test_the_macro_header_and_vix_line_are_ingestion_macros_and_poll_macro_stores_them():
    macro = open(os.path.join(mr.ROOT, "ingestion", "macro.py")).read()
    assert f'lines = ["{mr.MACRO_HEADER}"]' in macro
    assert '"VIXCLS":    "VIX Volatility Index",' in macro
    assert "lines.append(f\"{data['label']}: {data['value']} (as of {data['date']})\")" in macro
    tasks = open(os.path.join(mr.ROOT, "tasks.py")).read()
    i = tasks.index("def poll_macro(")
    body = tasks[i:i + 1200]
    assert 'collection="macro",' in body and "text=summary," in body
    vs = open(os.path.join(mr.ROOT, "vector_store.py")).read()
    assert '"timestamp": datetime.now(timezone.utc).isoformat(),' in vs
