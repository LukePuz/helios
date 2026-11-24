"""Record-only valuation consistency check (agents/valuation_check.py): the
reply's current_multiple and peer_range against the prompt's FUNDAMENTALS
block and against its own below/above/within-range wording, on the verdict's
decision. Nothing in the prompt, gates or routing reads it."""
import inspect
import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents import analyst
from agents.valuation_check import (VALUATION_MISMATCH_TOL_PCT, brief_multiples,
                                    parse_valuation, valuation_check)

_FUND = ("=== FUNDAMENTALS — NVDA ===\nRevenue growth (YoY): +60.0%\nGross margin: +70.0%\n"
         "FCF margin: +40.0%\nPrice/Sales: 9.74x\nEV/EBITDA: 14.71x\nForward P/E: 8.79x\n"
         "Market cap: $1\n\n=== PRICE STRUCTURE ===\nForward P/E: 99.00x\n")
_PROMPT = "header\n" + _FUND
_REPLY = {"valuation": {"current_multiple": "8.79x NTM earnings",
                        "peer_range": "6x to 12x NTM earnings"},
          "reasoning": "Trading at 8.79x NTM earnings, below the peer range of 6x to 12x "
                       "(per FUNDAMENTALS)."}


def test_the_audit_example_reads_as_a_range_contradiction():
    out = valuation_check(_REPLY, _PROMPT, "NVDA")
    p = out["valuation_parse"]
    assert (p["stated"], p["stated_basis"], p["range_lo"], p["range_hi"]) == (8.79, "forward_pe", 6.0, 12.0)
    vb = out["valuation_vs_brief"]
    assert vb["brief"] == {"forward_pe": 8.79, "price_to_sales": 9.74, "ev_to_ebitda": 14.71}
    assert vb["closest"] == "forward_pe" and vb["closest_gap_pct"] == 0.0
    assert vb["basis_gap_pct"] == 0.0 and vb["reason"] is None
    assert out["valuation_mismatch"] is False
    assert out["range_contradiction"] is True
    rc = out["range_check"]
    assert rc["actual"] == "inside" and rc["basis_differs"] is False
    assert rc["claims"] == [{"in": "reasoning", "phrase": "below the peer range",
                             "claimed": "below", "contradicts": True}]


def test_only_the_tickers_own_block_is_read_and_it_stops_at_the_next_header():
    assert brief_multiples(_PROMPT, "NVDA")["forward_pe"] == 8.79   # not the 99.00x below
    assert brief_multiples(_PROMPT, "AMD") is None
    assert brief_multiples("=== FUNDAMENTALS — NVDA ===\nNot yet cached (nightly refresh "
                           "pending)", "NVDA") is None
    na = brief_multiples("=== FUNDAMENTALS — X ===\nPrice/Sales: n/a\nEV/EBITDA: -3.10x\n"
                         "Forward P/E: 0.00x\n", "X")
    assert na == {"forward_pe": None, "price_to_sales": None, "ev_to_ebitda": None}
    assert brief_multiples(None, "X") is None and brief_multiples("x", "") is None


def test_mismatch_when_no_brief_multiple_is_within_the_tolerance():
    assert VALUATION_MISMATCH_TOL_PCT == 10.0
    far = {**_REPLY, "valuation": {**_REPLY["valuation"], "current_multiple": "20x NTM earnings"}}
    out = valuation_check(far, _PROMPT, "NVDA")
    assert out["valuation_mismatch"] is True
    assert out["valuation_vs_brief"]["closest"] == "ev_to_ebitda"
    assert out["valuation_vs_brief"]["closest_gap_pct"] == round((20 / 14.71 - 1) * 100, 2)
    # The closest of any of the three counts, not only the named basis: the
    # basis gap is recorded beside it.
    ps = {**_REPLY, "valuation": {**_REPLY["valuation"], "current_multiple": "10.2x NTM earnings"}}
    out = valuation_check(ps, _PROMPT, "NVDA")
    vb = out["valuation_vs_brief"]
    assert out["valuation_mismatch"] is False and vb["closest"] == "price_to_sales"
    assert vb["basis_gap_pct"] == round((10.2 / 8.79 - 1) * 100, 2)
    # Exactly at the tolerance is not a mismatch.
    pe_only = "=== FUNDAMENTALS — NVDA ===\nPrice/Sales: n/a\nForward P/E: 8.79x\n"
    edge = {**_REPLY, "valuation": {**_REPLY["valuation"], "current_multiple": "9.669x P/E"}}
    out = valuation_check(edge, pe_only, "NVDA")
    assert out["valuation_vs_brief"]["closest_gap_pct"] == 10.0
    assert out["valuation_mismatch"] is False
    edge["valuation"]["current_multiple"] = "9.68x P/E"
    assert valuation_check(edge, pe_only, "NVDA")["valuation_mismatch"] is True


def test_claimed_position_against_the_actual_one():
    def rc(reasoning, cm="8.79x NTM earnings", pr="6x to 12x NTM earnings"):
        return valuation_check({"valuation": {"current_multiple": cm, "peer_range": pr},
                                "reasoning": reasoning}, _PROMPT, "NVDA")
    assert rc("It sits within the 6x to 12x range.")["range_contradiction"] is False
    assert rc("It sits within the peer range.", cm="14x P/E")["range_contradiction"] is True
    assert rc("Above the peer range.", cm="14x P/E")["range_contradiction"] is False
    assert rc("Below the peer range.", cm="14x P/E")["range_contradiction"] is True
    assert rc("At the low end of its peer range.")["range_check"]["claims"][0]["claimed"] == "inside"
    assert rc("At a discount to the peer range.", cm="5x P/E")["range_contradiction"] is False
    # Ends count as inside.
    assert rc("Below the peer range.", cm="6x P/E")["range_contradiction"] is True
    # A price or trading range is not the peer multiple range.
    out = rc("Below its 52-week trading range.")
    assert out["range_contradiction"] is None and out["range_check"]["claims"] == []
    assert out["range_check"]["reason"] == "no below/above/within-range phrase"
    # The phrase can be in the valuation strings too.
    out = rc("r", pr="6x to 12x, so 8.79x is below the range")
    assert out["range_contradiction"] is True and out["range_check"]["claims"][0]["in"] == "peer_range"


def test_parse_shapes_seen_in_live_replies():
    p = parse_valuation("9.86x NTM earnings", "Forward P/E: 10-15x, Price/Sales: 10-15x")
    assert (p["range_lo"], p["range_hi"], p["range_basis"], p["n_ranges"]) == (10.0, 15.0, "forward_pe", 2)
    p = parse_valuation("24.24x NTM revenue, 69.66x forward P/E", "P/S at the 81% percentile")
    assert p["stated"] == 24.24 and p["stated_basis"] == "price_to_sales" and p["n_stated"] == 2
    assert p["range_lo"] is None and "peer_range" in p["reason"]
    p = parse_valuation("15.26x NTM revenue", "8% to 60% (P/S at the 60th percentile)")
    assert p["range_lo"] is None                         # a percent range is not a multiple range
    p = parse_valuation("32.13x NTM earnings", "20.61x to 41.70x P/S and EV/EBITDA multiples")
    assert p["range_basis"] == "mixed"
    p = parse_valuation("22.16x NTM earnings", "18x - 24x NTM earnings")
    assert (p["range_lo"], p["range_hi"]) == (18.0, 24.0)
    p = parse_valuation("8-20x NTM revenue", "6x to 12x")
    assert p["stated"] is None and "is a range" in p["reason"]
    p = parse_valuation("Unknown", None)
    assert p["stated"] is None and p["range_lo"] is None
    assert p["reason"] == "no '<n>x' in current_multiple; peer_range missing"


def test_unparseable_inputs_give_none_with_a_reason_and_never_raise():
    out = valuation_check({"valuation": "8x"}, _PROMPT, "NVDA")
    assert out["valuation_mismatch"] is None and out["range_contradiction"] is None
    assert out["valuation_parse"]["reason"] == "no valuation object in the reply"
    assert out["range_check"]["reason"] == "no valuation object in the reply"
    out = valuation_check(_REPLY, "no fundamentals here", "NVDA")
    assert out["valuation_mismatch"] is None
    assert out["valuation_vs_brief"]["reason"] == "no FUNDAMENTALS multiples in the prompt"
    assert out["range_contradiction"] is True            # needs only the reply's own numbers
    out = valuation_check({"valuation": {"current_multiple": "n/a", "peer_range": "6x to 12x"},
                           "reasoning": "below the peer range"}, _PROMPT, "NVDA")
    assert out["valuation_mismatch"] is None and out["range_contradiction"] is None
    assert out["valuation_vs_brief"]["reason"] == "no stated multiple"
    assert out["range_check"]["reason"] == "no stated multiple"
    for bad in (None, object(), {"valuation": {"current_multiple": 5, "peer_range": ["x"]}}):
        r = valuation_check(bad, object(), None)
        assert set(r) >= {"valuation_parse", "valuation_vs_brief", "valuation_mismatch",
                          "range_contradiction"}
    with patch("agents.valuation_check.parse_valuation", side_effect=RuntimeError("boom")):
        r = valuation_check(_REPLY, _PROMPT, "NVDA")
    assert r["valuation_mismatch"] is None and "boom" in r["valuation_check_error"]


def _analyze(check=None):
    parsed = {
        "action": "HOLD", "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {"current_multiple": "8.79x NTM earnings", "peer_range": "6x to 12x NTM earnings",
                      "bull_case_target": "$130", "downside_scenario": "$90"},
        "expected_return_pct": 8.0, "key_risks": [], "pass_to_risk": False,
        "upside_downside_ratio": 2.4, "reasoning": "8.79x is below the peer range of 6x to 12x.",
        "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}],
    }
    log_signal = MagicMock()
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
    if check is not None:
        patches.append(patch("agents.valuation_check.valuation_check", check))
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "NVDA", "role": "Compounder", "tier": "Core",
                                  "sector": "Semis", "held": True},
                                 situation_brief="Current Price: $100.00\n" + _FUND,
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"], ollama.call_args


def _without_clock(d):
    if isinstance(d, dict):
        return {k: _without_clock(v) for k, v in d.items() if not k.endswith("_at")}
    return d


_FIELDS = ("valuation_parse", "valuation_vs_brief", "valuation_mismatch", "range_contradiction",
           "range_check")


def test_on_the_verdicts_decision_and_nothing_else_changes():
    result, decision, sent = _analyze()
    assert decision["range_contradiction"] is True and decision["valuation_mismatch"] is False
    assert decision["valuation_vs_brief"]["brief"]["forward_pe"] == 8.79
    off, decision_off, sent_off = _analyze(check=MagicMock(side_effect=RuntimeError("x")))
    assert decision_off["valuation_check_error"] == "RuntimeError"
    assert not any(k in decision_off for k in _FIELDS)
    assert sent == sent_off                              # the prompt sent is the same
    for r in (result, off):
        r.pop("timestamp")
    assert _without_clock(result) == _without_clock(off)
    assert not any(k in result for k in _FIELDS)
    skip = set(_FIELDS) | {"catalyst_check", "valuation_check_error"}
    assert {k: v for k, v in decision.items() if k not in skip} == \
           {k: v for k, v in decision_off.items() if k not in skip}


def test_not_in_the_prompt_path():
    for f in (analyst._build_prompt, analyst._prompt_template_full_text):
        assert "valuation_check" not in inspect.getsource(f)
    import context_engine
    import knowledge_loader
    for m in (context_engine, knowledge_loader):
        assert "valuation_check" not in inspect.getsource(m)


def test_compares_against_the_prompt_that_was_sent_not_a_fresh_read():
    import agents.valuation_check as vc
    spy = MagicMock(wraps=vc.valuation_check)
    _, decision, sent = _analyze(check=spy)
    assert spy.call_count == 1
    assert spy.call_args.args[1] == sent.args[0]       # the exact string _call_ollama got
    # db.get_fundamentals is patched to have no multiples, so 8.79 can only
    # come from the brief text in the prompt.
    assert decision["valuation_vs_brief"]["brief"]["forward_pe"] == 8.79
    src = inspect.getsource(vc)
    for word in ("import db", "get_fundamentals", "context_engine", "requests", "open("):
        assert word not in src, word
