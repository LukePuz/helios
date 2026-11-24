"""Record-only DCF growth-branch audit (agents/valuation_audit.py): which DCF
branch a name takes, the same DCF on the other branch, the step between them
and how many intrinsic methods the blend rests on. One activity row per
trade ladder refresh; valuation.py's numbers and everything that reads them
are unchanged."""
import inspect
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from agents import trade_ladder
from agents import valuation as v
from agents import valuation_audit as va

NY = ZoneInfo("America/New_York")
# Ratios only: price 100, so a value of 100 reads 1.0.
FUND = {"free_cash_flow": 5e9, "shares_outstanding": 1e9, "net_cash": 2e9, "beta": 1.3,
        "revenue_growth_yoy": 0.40, "updated_at": "2026-11-20T22:00:00+00:00"}


def _ladder(fund, price=100.0, comps=None, ri=None, jm=None):
    dcf = v.compute_dcf_fair_value("X", fund)
    methods = {"dcf": dcf, "comps": comps, "residual_income": ri, "justified_multiple": jm}
    used = sorted(k for k, x in methods.items() if x)
    intr = [x for k, x in methods.items() if x and k != "comps"]
    blended = None
    if intr and comps:
        s = sorted(intr)
        med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
        blended = round((med + comps) / 2, 2)
    return {"current_price": price, "dcf_fair_value": round(dcf, 2) if dcf else None,
            "comps_fair_value": comps, "residual_income_fair_value": ri,
            "justified_multiple_fair_value": jm, "fair_value_blended": blended,
            "valuation_methods_used": used}


@pytest.mark.parametrize("growth", [-0.5, -0.1, 0.0, 0.2, 0.59, 0.61, 0.749, 0.75, 0.751, 0.79, 0.81, 1.5, 4.0])
def test_the_used_branch_reproduces_valuations_dcf(growth):
    f = {**FUND, "revenue_growth_yoy": growth}
    branch = va.dcf_branch_of(f)
    assert branch == ("hypergrowth" if growth >= v.DCF_HYPERGROWTH_THRESHOLD else "standard")
    assert va.dcf_value_on_branch(f, branch) == pytest.approx(v.compute_dcf_fair_value("X", f), rel=1e-12)


def test_the_branch_is_none_where_the_dcf_abstains():
    for f in ({**FUND, "free_cash_flow": -1.0}, {**FUND, "free_cash_flow": None},
              {**FUND, "shares_outstanding": 0}, {**FUND, "revenue_growth_yoy": None}):
        assert va.dcf_branch_of(f) == "none"
        assert va.dcf_value_on_branch(f, "standard") is None
        assert v.compute_dcf_fair_value("X", f) is None
    with patch.object(v, "_discount_rate", return_value=v.DCF_TERMINAL_GROWTH + 0.01):
        assert va.dcf_branch_of(FUND) == "none"
    e = va.dcf_branch_entry("X", {**FUND, "free_cash_flow": -1.0}, {"current_price": 100.0})
    assert e["dcf_branch"] == "none"
    assert e["reason"] == "the DCF abstains"
    assert e["dcf_value_rel"] is None
    assert va.dcf_branch_entry("X", None, {"current_price": 100.0})["reason"] == "no fundamentals row"


def test_the_step_at_the_line():
    below, above = {**FUND, "revenue_growth_yoy": 0.749}, {**FUND, "revenue_growth_yoy": 0.751}
    lo, hi = va.dcf_branch_entry("X", below, _ladder(below)), va.dcf_branch_entry("X", above, _ladder(above))
    assert lo["dcf_branch"] == "standard"
    assert hi["dcf_branch"] == "hypergrowth"
    # Each side's other branch is the other side's used branch, near enough.
    assert lo["dcf_value_rel_other_branch"] == pytest.approx(hi["dcf_value_rel"], rel=0.01)
    assert hi["dcf_value_rel_other_branch"] == pytest.approx(lo["dcf_value_rel"], rel=0.01)
    # step_ratio is hypergrowth / standard whichever branch is used.
    assert lo["step_ratio"] > 1
    assert hi["step_ratio"] > 1
    assert lo["step_ratio"] == pytest.approx(hi["dcf_value_rel"] / lo["dcf_value_rel"], rel=0.01)
    # At the threshold itself the ratio doesn't depend on the name's own growth.
    slow = {**FUND, "revenue_growth_yoy": 0.10}
    assert va.dcf_branch_entry("X", slow, _ladder(slow))["step_ratio_at_threshold"] == \
        lo["step_ratio_at_threshold"]
    h = va.dcf_value_on_branch(FUND, "hypergrowth", v.DCF_HYPERGROWTH_THRESHOLD)
    s = va.dcf_value_on_branch(FUND, "standard", v.DCF_HYPERGROWTH_THRESHOLD)
    assert lo["step_ratio_at_threshold"] == round(h / s, 4)


def test_ratios_to_the_ladder_price_and_the_match_check():
    f = {**FUND, "revenue_growth_yoy": 0.40}
    lad = _ladder(f, price=50.0)
    e = va.dcf_branch_entry("X", f, lad)
    assert e["dcf_value_rel"] == round(v.compute_dcf_fair_value("X", f) / 50.0, 4)
    assert e["dcf_matches_ladder"] is True
    assert e["growth_input"] == 0.40
    assert e["fundamentals_updated_at"] == "2026-11-20T22:00:00+00:00"
    moved = va.dcf_branch_entry("X", f, {**lad, "dcf_fair_value": lad["dcf_fair_value"] * 1.1})
    assert moved["dcf_matches_ladder"] is False
    assert moved["blend_rel_other_branch"] is None          # not recomputed on different inputs
    no_px = va.dcf_branch_entry("X", f, {**lad, "current_price": None})
    assert no_px["dcf_value_rel"] is None
    assert no_px["reason"] == "no ladder price"
    assert no_px["step_ratio"] is not None


def test_a_one_method_intrinsic_family_and_the_blend_on_the_other_branch():
    fast = {**FUND, "revenue_growth_yoy": 0.80}
    lad = _ladder(fast, comps=60.0)
    e = va.dcf_branch_entry("X", fast, lad)
    assert e["intrinsic_methods_used"] == ["dcf"]
    assert e["n_intrinsic_methods"] == 1
    assert e["blend_rel"] == round(lad["fair_value_blended"] / 100.0, 4)
    other = va.dcf_value_on_branch(fast, "standard")
    assert e["blend_rel_other_branch"] == round((other + 60.0) / 2 / 100.0, 4)
    # Three intrinsic methods: the median damps the DCF's swing.
    slow = {**FUND, "revenue_growth_yoy": 0.20}
    lad3 = _ladder(slow, comps=60.0, ri=40.0, jm=45.0)
    e3 = va.dcf_branch_entry("X", slow, lad3)
    assert e3["intrinsic_methods_used"] == ["dcf", "justified_multiple", "residual_income"]
    assert e3["n_intrinsic_methods"] == 3
    assert e3["blend_rel_other_branch"] == e3["blend_rel"]
    # No market family: no blend either way.
    assert va.dcf_branch_entry("X", fast, _ladder(fast))["blend_rel_other_branch"] is None


def test_the_entry_never_raises():
    with patch.object(va, "dcf_branch_of", side_effect=RuntimeError("boom")):
        e = va.dcf_branch_entry("X", FUND, _ladder(FUND))
    assert e["reason"].startswith("error: RuntimeError")
    assert e["dcf_branch"] is None


def test_the_summary_and_the_one_row_per_refresh():
    fast, slow = {**FUND, "revenue_growth_yoy": 0.9}, {**FUND, "revenue_growth_yoy": 0.2}
    entries = [va.dcf_branch_entry("A", fast, _ladder(fast, comps=60.0)),
               va.dcf_branch_entry("B", slow, _ladder(slow, comps=60.0, ri=40.0)),
               va.dcf_branch_entry("C", {**FUND, "free_cash_flow": -1}, {"current_price": 10.0})]
    s = va.dcf_branch_summary(entries, "paper")
    assert (s["n_names"], s["n_dcf"], s["n_hypergrowth"], s["n_standard"], s["n_none"]) == (3, 2, 1, 1, 1)
    assert s["dcf_only_intrinsic"] == ["A"]
    assert s["n_dcf_mismatch"] == 0
    assert s["params"]["threshold"] == v.DCF_HYPERGROWTH_THRESHOLD
    assert s["params"]["hypergrowth"] == [v.DCF_HYPERGROWTH_PROJECTION_YEARS,
                                          v.DCF_HYPERGROWTH_MAX_NEAR_TERM_GROWTH]
    with patch("db.log_activity", return_value=True) as row:
        assert va.log_dcf_branch(entries) is True
    args = row.call_args.args
    assert args[0] == "dcf_branch"
    assert "$" not in args[2]
    with patch("db.log_activity", side_effect=RuntimeError("x")):
        assert va.log_dcf_branch(entries) is False


def test_refresh_writes_one_row_and_the_ladder_is_unchanged():
    lad = _ladder(FUND, comps=60.0)
    upsert, row = MagicMock(), MagicMock(return_value=True)
    with patch("knowledge_loader.get_all_tickers", return_value=["A", "B"]), \
         patch.object(trade_ladder, "build_trade_ladder", side_effect=lambda t: dict(lad)), \
         patch("db.upsert_trade_ladder", upsert), patch("db.get_fundamentals", return_value=FUND), \
         patch("db.log_activity", row):
        assert trade_ladder.refresh_all_ladders() == 2
    assert row.call_count == 1
    extra = row.call_args.args[4]
    assert [n["ticker"] for n in extra["names"]] == ["A", "B"]
    assert [c.args[1] for c in upsert.call_args_list] == [lad, lad]   # nothing added to the ladder
    # The record failing changes nothing about the refresh.
    upsert.reset_mock()
    with patch("knowledge_loader.get_all_tickers", return_value=["A"]), \
         patch.object(trade_ladder, "build_trade_ladder", side_effect=lambda t: dict(lad)), \
         patch("db.upsert_trade_ladder", upsert), \
         patch("db.get_fundamentals", side_effect=RuntimeError("down")), \
         patch("db.log_activity", side_effect=RuntimeError("down")):
        assert trade_ladder.refresh_all_ladders() == 1
    assert upsert.call_count == 1


def test_valuation_and_the_prompt_path_do_not_name_it():
    from agents import analyst
    assert "valuation_audit" not in inspect.getsource(v)
    assert "agents.valuation_audit" not in analyst._BRIEF_MODULES
    assert "agents.valuation" not in analyst._BRIEF_MODULES
    import context_engine
    import knowledge_loader
    for m in (context_engine, knowledge_loader, analyst):
        assert "valuation_audit" not in inspect.getsource(m)
    src = inspect.getsource(va)
    for word in ("INSERT", "UPDATE ", "DELETE", "commit(", "requests", "yfinance"):
        assert word not in src, word


def test_batch_verify_check():
    from scripts import batch_verify as bv
    fast = {**FUND, "revenue_growth_yoy": 0.9}
    entries = [va.dcf_branch_entry("A", fast, _ladder(fast, comps=60.0))]
    with patch("db.log_activity", return_value=True) as row:
        va.log_dcf_branch(entries)
    extra = row.call_args.args[4]
    env = extra["env"]
    since = datetime(2026, 11, 24, 5, 0, tzinfo=NY)
    noon, eve = datetime(2026, 11, 24, 12, 0, tzinfo=NY), datetime(2026, 11, 24, 18, 30, tzinfo=NY)
    act = lambda i, **x: {"id": i, "stage": "dcf_branch", "ticker": "", "extra": x}
    assert bv.check_dcf_branch([], env, since, noon)["status"] == "OK"
    assert bv.check_dcf_branch([], env, since, eve)["status"] == "WARN"
    c = bv.check_dcf_branch([act(1, **extra)], env, since, eve)
    assert c["status"] == "OK"
    assert c["reads"]["dcf_branch"] == {"hypergrowth": 1}
    for k in bv.DCF_BRANCH_REQUIRED:
        bad = {kk: x for kk, x in extra.items() if kk != k}
        assert bv.check_dcf_branch([act(2, **bad)], env, since, eve)["status"] == "FAIL", k
    for k in bv.DCF_BRANCH_NAME_REQUIRED:
        names = [{kk: x for kk, x in extra["names"][0].items() if kk != k}]
        assert bv.check_dcf_branch([act(3, **{**extra, "names": names})], env, since, eve)["status"] == \
            "FAIL", k
    errs = {**extra, "names": [{**extra["names"][0], "reason": "error: RuntimeError"}] * 3
            + extra["names"]}
    c = bv.check_dcf_branch([act(4, **errs)], env, since, eve)
    assert c["status"] == "FAIL"
    assert "name_read" in c["why"]
    mism = {**extra, "names": [{**extra["names"][0], "dcf_matches_ladder": False}]}
    c = bv.check_dcf_branch([act(5, **mism)], env, since, eve)
    assert c["status"] == "OK"
    assert "differs from the ladder" in " ".join(c["notes"])
    assert bv.check_dcf_branch([act(6, **{**extra, "env": "elsewhere"})], env, since, eve)["status"] == "WARN"


def test_registry_stage():
    from scripts import shadow_registry_check as src
    st = next(s for s in src.STAGES if s["stage"] == "dcf_branch")
    assert st["expect_rows"] == "always"
    assert st["switch"] is None
    assert src.writer_present(src.ROOT, st["writer"]) is True
    assert "dcf_branch" in st["sources"][0][1]
