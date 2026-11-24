"""Record-only source mix (agents/source_mix.py, analyst._source_mix_record):
the analyst's cited sources split into external, brief-section and other on
the corroboration gate's own dedupe, the publishers named, and whether the
gate's bar would be met by external sources alone. The gate, the prompt and
the verdict are unchanged: the analysis is compared with the record on and
broken, and the template hashes do not read the new code. Synthetic data."""
import copy
import importlib.util
import inspect
import json
import pathlib
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import db
from agents import analyst
from agents import source_mix as sm
from agents.source_mix import publisher, source_mix

FUND = {"type": "fundamentals_data",
        "detail": "Revenue growth (YoY): +50.1%, Gross margin: +55.7%, Forward P/E: 30.04x"}
PRICE = {"type": "price_action",
         "detail": "Price structure is a genuine uptrend with higher high and higher low"}
NEWS = {"type": "news", "detail": "HSBC raises AMD target to $540 but flags a risk - Yahoo Finance UK"}


def _mix(sources):
    return source_mix(sources, analyst._MIN_DISTINCT_SOURCES)


class TestSplit:
    def test_the_passed_buy_shape_passes_the_gate_on_one_external_source(self):
        """FUNDAMENTALS block + one article: the gate counts 2 and passes;
        external sources alone are 1."""
        rec = _mix([FUND, NEWS])
        assert rec["source_count"] == 2 and rec["gate_would_pass"] is True
        assert rec["external_source_count"] == 1 and rec["brief_section_sources"] == 1
        assert rec["gate_would_pass_external_only"] is False and rec["min_sources"] == 2

    def test_two_external_sources_pass_external_only(self):
        rating = {"type": "analyst_rating", "detail": "UBS raised ANET price target from $259 to $291"}
        rec = _mix([NEWS, rating, PRICE])
        assert rec["external_source_count"] == 2 and rec["gate_would_pass_external_only"] is True
        assert rec["brief_section_sources"] == 1

    @pytest.mark.parametrize("t", sm.EXTERNAL_TYPES)
    def test_each_external_type(self, t):
        assert _mix([{"type": t, "detail": "x"}])["external_source_count"] == 1

    @pytest.mark.parametrize("t", ["filing", "upcoming_catalyst", "catalyst", "funds_data",
                                   "congressional_trade", "arxiv_paper", "other"])
    def test_other_types_are_neither_external_nor_brief(self, t):
        rec = _mix([{"type": t, "detail": "x"}])
        assert rec["other_sources"] == 1 and rec["external_source_count"] == 0
        assert rec["type_counts"] == {t: 1}

    def test_the_split_uses_the_gates_dedupe_and_sums_to_its_count(self):
        long = dict(NEWS, detail="x" * 80 + " - Reuters")
        dup = dict(NEWS, detail="x" * 80 + " - a rewording past the gate's 80 characters")
        srcs = [FUND, FUND, NEWS, long, dup, {"type": "News", "detail": "case"}, PRICE,
                {"type": "filing", "detail": "8-K"}, {"detail": "no type"}, "junk", None,
                {"type": "", "detail": "empty type"}]
        rec = _mix(srcs)
        assert rec["source_count"] == analyst.count_distinct_sources(srcs)
        assert (rec["external_source_count"] + rec["brief_section_sources"]
                + rec["other_sources"]) == rec["source_count"]
        assert rec["type_counts"] == {"News": 1, "filing": 1, "fundamentals_data": 1,
                                      "news": 2, "price_action": 1}
        assert rec["external_source_count"] == 3          # "News" reads as news

    def test_no_sources(self):
        for srcs in ([], None, "text", {"type": "news"}):
            rec = _mix(srcs)
            assert rec["source_count"] == 0 and rec["gate_would_pass"] is False
            assert rec["gate_would_pass_external_only"] is False

    def test_never_raises(self):
        rec = _mix([{"type": {"unhashable": 1}, "detail": "x"}])
        assert rec == {"error": "TypeError"}
        with patch("agents.source_mix.publisher", side_effect=RuntimeError("x")):
            assert _mix([NEWS]) == {"error": "RuntimeError"}

    def test_record_is_plain_json(self):
        rec = _mix([FUND, NEWS, PRICE])
        assert json.loads(json.dumps(rec)) == rec


class TestPublisher:
    @pytest.mark.parametrize("src, expected", [
        (NEWS, ("yahoo finance uk", "title_suffix")),
        ({"type": "news", "detail": "COHR Q1 Earnings Call Highlights - Yahoo Finance "
                                    "(published Thu, 05 Nov 2026 16:18:00 GMT)"},
         ("yahoo finance", "title_suffix")),
        ({"type": "news", "detail": "ANET jumped 14% after hours (per MarketBeat article)"},
         ("marketbeat", "per")),
        ({"type": "news", "detail": "Oracle's database use is growing, per Yahoo Finance article"},
         ("yahoo finance", "per")),
        ({"type": "news", "detail": "Huang ousted as CEO, national security concerns (WSJ)."},
         ("wsj", "paren")),
        ({"type": "news", "detail": "Stock falls - CNBC per event content"},
         ("cnbc", "title_suffix")),
        ({"type": "news", "detail": "see https://www.reuters.com/x/y for the story"},
         ("reuters.com", "url")),
        ({"type": "edgar_filing", "detail": "x", "url": "https://www.sec.gov/Archives/1"},
         ("sec.gov", "url")),
    ])
    def test_named(self, src, expected):
        assert publisher(src) == expected

    @pytest.mark.parametrize("detail", [
        "MSFT rises 5% on Azure growth (per news article)",
        "Anthropic IPO triggers AI selloff — washout or deeper unwind? (published 2026-11-20)",
        "Revenue guide raised per UPCOMING CATALYST CALENDAR",
        "ANET guides higher for Q4 per EVENT CONTENT",
        "Shares rise (per RELEVANT NEWS section)",
        "Nvidia falls below Alphabet (2026-11-20)",
        "CoreWeave raised its forecast per 2026-11-11 news",
        "Teradyne results - revenue up 70% and earnings up over 150%",
        "Jensen Huang's forced exit from Nvidia",
    ])
    def test_none_for_fragments_and_generic_references(self, detail):
        assert publisher({"type": "news", "detail": detail}) == (None, None)

    def test_only_news_is_parsed_from_text(self):
        """A filing's title suffix is not a publisher."""
        src = {"type": "edgar_filing", "detail": "Teradyne Reports Results - Strong Quarter"}
        assert publisher(src) == (None, None)

    def test_distinct_publishers_count_external_items_only(self):
        a = {"type": "news", "detail": "One - Reuters"}
        b = {"type": "news", "detail": "Two - Reuters"}
        c = {"type": "news", "detail": "Three - CNBC"}
        brief = {"type": "price_action", "detail": "Up - Seeking Alpha"}
        rec = _mix([a, b, c, brief, {"type": "insider_trade", "detail": "Director bought"}])
        assert rec["distinct_publishers"] == 2 and rec["publishers"] == ["cnbc", "reuters"]
        assert rec["n_external_no_publisher"] == 1
        assert rec["publisher_basis"] == {"title_suffix": 3}

    def test_publishers_are_capped(self):
        rec = _mix([{"type": "news", "detail": f"Story - Paper {chr(65 + i)}x"} for i in range(14)])
        assert rec["distinct_publishers"] == 14 and len(rec["publishers"]) == 10


# ---- on the verdict, and nothing else changes ----

_FUND_BRIEF = ("FUNDAMENTALS:\n  Revenue growth (YoY): +20.0%\n  Forward P/E: 8.79x\n")


def _analyze(record=None, sources=None, action="OPEN", udr=2.4):
    parsed = {
        "action": action, "conviction": 7, "thesis_strength": 4, "catalyst_quality": 4,
        "valuation_risk_reward": 4, "data_completeness": 4, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-12-20"},
        "valuation": {"current_multiple": "8.79x", "peer_range": "6x to 12x",
                      "bull_case_target": "$130", "downside_scenario": "$90"},
        "expected_return_pct": 30.0, "expected_downside_pct": 10.0, "key_risks": [],
        "pass_to_risk": True, "upside_downside_ratio": udr, "reasoning": "r",
        "sources": [FUND, NEWS] if sources is None else sources,
    }
    log_signal = MagicMock(return_value=7)
    ollama = MagicMock(return_value=json.dumps(parsed))
    patches = [
        patch("agents.analyst._call_ollama", ollama),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.get_trade_ladder", return_value=None),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
    ]
    if record is not None:
        patches.append(patch("agents.source_mix.source_mix", record))
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "NVDA", "role": "Compounder", "tier": "Core",
                                  "sector": "Semis", "held": False},
                                 situation_brief="Current Price: $100.00\n" + _FUND_BRIEF,
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"], ollama.call_args


def _without_clock(d):
    if isinstance(d, dict):
        return {k: _without_clock(v) for k, v in d.items() if not k.endswith("_at")}
    if isinstance(d, list):
        return [_without_clock(v) for v in d]
    return d


def _same(a, b):
    """The verdict and its decision, compared without the record and the
    wall-clock values."""
    assert a[2] == b[2]                                   # the prompt sent is the same
    (ra, da), (rb, db_) = copy.deepcopy(a[:2]), copy.deepcopy(b[:2])
    for r in (ra, rb):
        r.pop("timestamp", None)
    assert _without_clock(ra) == _without_clock(rb)
    skip = {"source_mix", "catalyst_check"}
    assert ({k: v for k, v in da.items() if k not in skip}
            == {k: v for k, v in db_.items() if k not in skip})


@pytest.mark.parametrize("sources", [None, [FUND], [NEWS, {"type": "analyst_rating", "detail": "x"}],
                                     []])
def test_on_the_verdicts_decision_and_nothing_else_changes(sources):
    on = _analyze(sources=sources)
    broken = _analyze(record=MagicMock(side_effect=RuntimeError("x")), sources=sources)
    assert on[1]["source_mix"]["source_count"] == analyst.count_distinct_sources(
        [FUND, NEWS] if sources is None else sources)
    assert broken[1]["source_mix"] == {"error": "RuntimeError"}
    _same(on, broken)
    assert "source_mix" not in on[0]                      # the result routing reads


def test_the_gate_still_counts_brief_sections():
    """The record's external-only flag is False, and the buy still passes."""
    result, decision, _ = _analyze()
    assert decision["source_mix"]["gate_would_pass_external_only"] is False
    assert decision["source_mix"]["gate_would_pass"] is True
    assert not any("sources" in g for g in decision["gates"])
    _, decision, _ = _analyze(sources=[NEWS])
    assert any(g.startswith("sources 1 < 2") for g in decision["gates"])
    assert decision["final_action"] == "WATCH"


def test_the_bar_is_the_gates_constant():
    with patch.object(analyst, "_MIN_DISTINCT_SOURCES", 3):
        rec = analyst._source_mix_record([FUND, NEWS, PRICE])
    assert rec["min_sources"] == 3 and rec["gate_would_pass"] is True


def test_pure_and_not_in_the_prompt_path():
    src = inspect.getsource(sm)
    for word in ("import db", "requests", "open(", "context_engine", "knowledge_loader"):
        assert word not in src, word
    assert "source_mix" not in analyst._prompt_template_full_text()
    assert "source_mix" not in analyst._prompt_db_queries_text()
    for m in analyst._BRIEF_MODULES:
        assert "source_mix" not in (analyst._module_source(m) or "")
    assert "agents.source_mix" not in analyst._BRIEF_MODULES


# ---- batch_verify and the registry ----

def _script(name):
    path = pathlib.Path(analyst.__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_smx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_verify_stage():
    bv = _script("batch_verify")
    rec = _mix([FUND, NEWS])
    assert set(bv.SOURCE_MIX_REQUIRED) == set(rec)
    rows = [{"id": 1, "has_model_output": True, "source_mix": rec},
            {"id": 2, "has_model_output": False, "source_mix": None},          # a stub
            {"id": 3, "source_mix": None}]                                     # older caller
    c = bv.check_source_mix(rows)
    assert c["status"] == "OK" and c["rows"] == 1
    assert c["reads"]["gate_would_pass_external_only"] == {"false": 1}
    assert bv.check_source_mix([])["status"] == "WARN"
    c = bv.check_source_mix([{"id": 4, "has_model_output": True, "source_mix": None}])
    assert c["status"] == "FAIL" and c["missing"] == [4]
    bad = dict(rec, other_sources=5)
    c = bv.check_source_mix([{"id": 5, "has_model_output": True, "source_mix": bad}])
    assert c["status"] == "FAIL" and "does not sum" in c["why"]
    err = [{"id": i, "has_model_output": True, "source_mix": {"error": "TypeError"}}
           for i in range(3)] + [{"id": 9, "has_model_output": True, "source_mix": rec}]
    c = bv.check_source_mix(err)
    assert c["status"] == "FAIL" and c["reads"]["record"] == {"error: TypeError": 3, "ok": 1}


def test_registry_stage():
    rc = _script("shadow_registry_check")
    st = {s["stage"]: s for s in rc.STAGES}["source_mix"]
    assert st["switch"] is None and st["expect_rows"] == "conditional"
    assert st["trigger"] == ("signals", "decision ? 'model_output'")
    assert st["sources"] == (("signals", "decision ? 'source_mix'"),)
    root = pathlib.Path(analyst.__file__).resolve().parent.parent
    assert rc.writer_present(str(root), st["writer"])


def test_db_is_not_read():
    with patch.object(db, "_get_conn", side_effect=AssertionError("no DB")):
        assert _mix([FUND, NEWS])["source_count"] == 2
