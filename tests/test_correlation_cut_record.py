"""The 25% correlation cut is recorded on the sizing (professor m507).
order_pct follows the cut shares (66c); position_size_pct, the target, stays."""
import inspect

from agents import crew


def test_cut_is_recorded_and_order_pct_follows_the_shares():
    src = inspect.getsource(crew._route_to_approval)
    i = src.index('corr = rm.check_correlation(ticker)')
    block = src[i:src.index("_apply_micro_name_cap", i)]
    assert 'sizing["correlation_cut"] = {' in block
    assert '"shares_before": shares_before, "shares_after": sizing["shares_initial"]' in block
    assert 'sizing["correlation_cut"] = {"applied": False' in block
    assert "if shares_before and op:" in block
    assert 'sizing["order_pct"] = round(op * sizing["shares_initial"] / shares_before, 4)' in block
    assert 'sizing["position_size_pct"] =' not in block
    assert '"errored": str(corr.get("note") or "").startswith("RISK CHECK FAILED")' in block


def test_fail_closed_note_matches_the_errored_prefix():
    import risk_manager
    src = inspect.getsource(risk_manager.RiskManager.check_correlation)
    assert 'f"RISK CHECK FAILED — correlation unverifiable' in src
