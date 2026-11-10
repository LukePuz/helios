"""Log-only: the model's own target/floor against the sizing price."""
from agents import crew

RESULT = {"upside_downside_ratio": 2.0,
          "valuation": {"bull_case_target": "$120 (12 months)", "downside_scenario": "$90 if guidance slips"}}


def test_ratio_at_the_brief_price_clears_the_gate():
    r = crew._udr_at_sizing(RESULT, 100.0)
    assert r["udr"] == 2.0 and r["gate_at_sizing"] is True and r["model_udr"] == 2.0


def test_a_higher_sizing_price_can_fail_the_gate():
    r = crew._udr_at_sizing(RESULT, 108.0)          # (120-108)/(108-90) = 0.67
    assert r["udr"] == 0.67 and r["gate_at_sizing"] is False


def test_price_outside_target_and_floor_is_recorded_not_computed():
    r = crew._udr_at_sizing(RESULT, 125.0)
    assert r["udr"] is None and r["reason"] == "price_outside_target_floor"
    assert crew._udr_at_sizing({"valuation": {}}, 100.0)["reason"] == "no_target_or_floor"
    assert crew._udr_at_sizing(RESULT, None)["reason"] == "no_price"


def test_never_raises():
    assert "error" in crew._udr_at_sizing(None, 100.0) or crew._udr_at_sizing(None, 100.0)["udr"] is None


def test_gate_uses_the_unrounded_ratio():
    # (120-p)/(p-90) = 1.995 rounds to 2.0 but must not clear a 2.0 floor
    p = (120 + 2 * 90 - 0.005 * 90) / 2.995
    r = crew._udr_at_sizing(RESULT, p)
    assert r["udr"] == 2.0 and r["gate_at_sizing"] is False
