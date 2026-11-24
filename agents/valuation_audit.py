"""Record-only audit of the DCF growth branch (audit 2026-11-22 06:56 #1).

agents/valuation.compute_dcf_fair_value projects 10 years from an 80% cap
when revenue_growth_yoy >= DCF_HYPERGROWTH_THRESHOLD, and 5 years from a 60%
cap below it. Above SINGLE_STAGE_MAX_APPLICABLE_GROWTH, residual income and
the justified multiple abstain, so for fast growers the intrinsic family of
the blend can be that one DCF. This module recomputes the DCF on both
branches from the same fundamentals row and records how far apart they are,
as ratios to the ladder's price.

It runs after each ladder is built, from agents/trade_ladder.refresh_all_ladders,
and writes one activity row per refresh (stage DCF_BRANCH_STAGE). Nothing in
valuation.py changes, and nothing in the ladder, the brief, the prompt,
the plans or the gates reads it. The math is valuation's own: _discount_rate
and _dcf_equity_value_for_growth with the branch's years and cap. Prices
appear only as ratios. Never raises.

Per name (names[]):
  dcf_branch                  hypergrowth / standard (the branch valuation
                              used) / none (the DCF abstains)
  growth_input                revenue_growth_yoy, the input that picks it
  dcf_value_rel               DCF value / ladder price, on the used branch
  dcf_value_rel_other_branch  the same DCF with the other branch's years and
                              cap
  step_ratio                  DCF on the hypergrowth branch / DCF on the
                              standard branch at the name's own growth (the
                              same direction whichever branch is used)
  step_ratio_at_threshold     the jump at the line: the DCF with growth at
                              DCF_HYPERGROWTH_THRESHOLD on the hypergrowth
                              branch / the same growth on the standard branch
  intrinsic_methods_used      the intrinsic-family methods in the ladder's
                              valuation_methods_used; n_intrinsic_methods
  blend_rel                   the ladder's fair_value_blended / price
  blend_rel_other_branch      the blend recomputed with the other branch's
                              DCF (valuation's family rule: the mean of the
                              intrinsic and market family medians)
  dcf_matches_ladder          the recomputed used-branch DCF equals the
                              ladder's dcf_fair_value, both None counting
                              as equal (the inputs are the same)
plus the book-level counts in the row's extra.
"""
from __future__ import annotations

import math

DCF_BRANCH_STAGE = "dcf_branch"
INTRINSIC_METHODS = ("dcf", "residual_income", "justified_multiple")
MARKET_METHODS = ("comps",)


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _branch_params(branch: str) -> tuple[int, float]:
    from agents import valuation as v
    if branch == "hypergrowth":
        return v.DCF_HYPERGROWTH_PROJECTION_YEARS, v.DCF_HYPERGROWTH_MAX_NEAR_TERM_GROWTH
    return v.DCF_PROJECTION_YEARS, v.DCF_MAX_NEAR_TERM_GROWTH


def dcf_branch_of(fund: dict) -> str:
    """The branch compute_dcf_fair_value takes for fund, or "none" when it
    abstains (no positive FCF or shares, no growth, or the discount rate
    too close to terminal growth)."""
    from agents import valuation as v
    fcf, shares, growth = (fund.get("free_cash_flow"), fund.get("shares_outstanding"),
                           fund.get("revenue_growth_yoy"))
    if not fcf or fcf <= 0 or not shares or shares <= 0 or growth is None:
        return "none"
    if v._discount_rate(fund) - v.DCF_TERMINAL_GROWTH < v.DCF_MIN_DISCOUNT_SPREAD:
        return "none"
    return "hypergrowth" if growth >= v.DCF_HYPERGROWTH_THRESHOLD else "standard"


def dcf_value_on_branch(fund: dict, branch: str, growth: float | None = None) -> float | None:
    """compute_dcf_fair_value's per-share value with branch's years and
    near-term cap, whatever growth says (growth: in place of
    revenue_growth_yoy); None where it abstains."""
    from agents import valuation as v
    if dcf_branch_of(fund) == "none":
        return None
    years, cap = _branch_params(branch)
    g = fund["revenue_growth_yoy"] if growth is None else growth
    g0 = max(min(g, cap), v.DCF_MIN_NEAR_TERM_GROWTH)
    equity = v._dcf_equity_value_for_growth(
        fund["free_cash_flow"], fund["shares_outstanding"], fund.get("net_cash") or 0.0,
        v._discount_rate(fund), g0, years=years)
    return equity / fund["shares_outstanding"] if equity > 0 else None


def _median(vals):
    s = sorted(vals)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _blend_with_dcf(ladder: dict, dcf: float | None) -> float | None:
    """valuation's family blend from the ladder's method values, the DCF
    replaced by dcf (None: without it)."""
    vals = {"dcf": dcf, "residual_income": ladder.get("residual_income_fair_value"),
            "justified_multiple": ladder.get("justified_multiple_fair_value"),
            "comps": ladder.get("comps_fair_value")}
    vals = {k: x for k, x in ((k, _finite(x)) for k, x in vals.items()) if x and x > 0}
    fams = [m for m in (_median([vals[k] for k in INTRINSIC_METHODS if k in vals]),
                        _median([vals[k] for k in MARKET_METHODS if k in vals])) if m is not None]
    return sum(fams) / 2 if len(fams) == 2 else None


def _rel(x, price):
    x = _finite(x)
    return round(x / price, 4) if x is not None and price else None


def dcf_branch_entry(ticker: str, fund: dict | None, ladder: dict) -> dict:
    """The per-name record from the fundamentals row and the built ladder.
    Never raises."""
    out = {"ticker": ticker, "dcf_branch": None, "growth_input": None, "dcf_value_rel": None,
           "dcf_value_rel_other_branch": None, "step_ratio": None,
           "step_ratio_at_threshold": None,
           "intrinsic_methods_used": [], "n_intrinsic_methods": 0, "blend_rel": None,
           "blend_rel_other_branch": None, "dcf_matches_ladder": None,
           "fundamentals_updated_at": None, "reason": None}
    try:
        used = [m for m in (ladder.get("valuation_methods_used") or []) if m in INTRINSIC_METHODS]
        out["intrinsic_methods_used"], out["n_intrinsic_methods"] = sorted(used), len(used)
        price = _finite(ladder.get("current_price"))
        price = price if price and price > 0 else None
        out["blend_rel"] = _rel(ladder.get("fair_value_blended"), price)
        if not fund:
            out["reason"] = "no fundamentals row"
            return out
        out["fundamentals_updated_at"] = str(fund.get("updated_at") or "") or None
        out["growth_input"] = _finite(fund.get("revenue_growth_yoy"))
        branch = dcf_branch_of(fund)
        out["dcf_branch"] = branch
        if branch == "none":
            out["reason"] = "the DCF abstains"
            return out
        other = "standard" if branch == "hypergrowth" else "hypergrowth"
        v_used, v_other = dcf_value_on_branch(fund, branch), dcf_value_on_branch(fund, other)
        lad = _finite(ladder.get("dcf_fair_value"))
        out["dcf_matches_ladder"] = (v_used is None and lad is None) or (
            v_used is not None and lad is not None and round(v_used, 2) == round(lad, 2))
        if price is None:
            out["reason"] = "no ladder price"
        elif v_used is None:
            out["reason"] = "the DCF value is <= 0 on the used branch"
        out["dcf_value_rel"], out["dcf_value_rel_other_branch"] = _rel(v_used, price), _rel(v_other, price)
        hyper, std = (v_used, v_other) if branch == "hypergrowth" else (v_other, v_used)
        out["step_ratio"] = round(hyper / std, 4) if hyper and std else None
        from agents.valuation import DCF_HYPERGROWTH_THRESHOLD as th
        h_t, s_t = dcf_value_on_branch(fund, "hypergrowth", th), dcf_value_on_branch(fund, "standard", th)
        out["step_ratio_at_threshold"] = round(h_t / s_t, 4) if h_t and s_t else None
        if out["dcf_matches_ladder"]:
            out["blend_rel_other_branch"] = _rel(_blend_with_dcf(ladder, v_other), price)
    except Exception as e:
        out["reason"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    return out


def dcf_branch_summary(entries: list[dict], env: str) -> dict:
    """The activity row's extra: the names plus book-level counts."""
    from agents import valuation as v
    by = lambda b: [e for e in entries if e.get("dcf_branch") == b]
    steps = sorted(e["step_ratio"] for e in entries if e.get("step_ratio"))
    at = sorted(e["step_ratio_at_threshold"] for e in entries if e.get("step_ratio_at_threshold"))
    one = [e["ticker"] for e in entries if e.get("dcf_branch") not in (None, "none")
           and e.get("intrinsic_methods_used") == ["dcf"]]
    return {"env": env, "n_names": len(entries), "n_dcf": len(by("hypergrowth")) + len(by("standard")),
            "n_hypergrowth": len(by("hypergrowth")), "n_standard": len(by("standard")),
            "n_none": len(by("none")), "dcf_only_intrinsic": one,
            "n_dcf_mismatch": sum(1 for e in entries if e.get("dcf_matches_ladder") is False),
            "n_errors": sum(1 for e in entries if str(e.get("reason") or "").startswith("error")),
            "step_ratio_median": round(_median(steps), 4) if steps else None,
            "step_ratio_at_threshold_median": round(_median(at), 4) if at else None,
            "params": {"threshold": v.DCF_HYPERGROWTH_THRESHOLD,
                       "hypergrowth": list(_branch_params("hypergrowth")),
                       "standard": list(_branch_params("standard")),
                       "single_stage_max_growth": v.SINGLE_STAGE_MAX_APPLICABLE_GROWTH},
            "names": entries}


def log_dcf_branch(entries: list[dict]) -> bool:
    """Writes the refresh's one activity row. Never raises."""
    try:
        import db
        from config import TRADING_ENV
        extra = dcf_branch_summary(entries, TRADING_ENV)
        return bool(db.log_activity(
            DCF_BRANCH_STAGE, "", f"DCF branch record: {extra['n_dcf']} of {extra['n_names']} names "
            f"with a DCF, {extra['n_hypergrowth']} on the hypergrowth branch", "neutral", extra))
    except Exception:
        return False
