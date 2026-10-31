"""Per-share risk as sized and at the fill, log-only (audit A1031-1053-1)."""
from unittest.mock import patch

import execution


def _approval(stop=95.0, sized_at=100.0):
    return {"id": 1, "stop_loss_price": stop, "risk_json": {"price_at_sizing": sized_at}}


def test_a_higher_fill_records_the_larger_risk():
    assert execution._risk_per_share(_approval(), 107.0) == (5.0, 12.0)


def test_a_lower_fill_uses_the_anchored_stop():
    # filled at 95.5: the 95 stop is within 1% of the fill, so it moves to 95.5*0.95
    planned, at_fill = execution._risk_per_share(_approval(), 95.5)
    assert planned == 5.0 and at_fill == round(95.5 - round(95.5 * 95 / 100, 2), 4)


def test_missing_inputs_give_none():
    assert execution._risk_per_share({"risk_json": {}}, 100.0) == (None, None)
    assert execution._risk_per_share(_approval(sized_at=None), 100.0) == (None, 5.0)
    assert execution._risk_per_share(_approval(), None) == (5.0, None)


def test_the_outcome_row_carries_both():
    with patch("db.log_trade_outcome", return_value=True) as log_outcome:
        execution.log_buy_outcome(_approval(), "NVDA", "OPEN", 107.0)
    kw = log_outcome.call_args.kwargs
    assert kw["planned_risk_per_share"] == 5.0 and kw["fill_risk_per_share"] == 12.0
