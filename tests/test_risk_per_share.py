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


def test_a_failed_risk_computation_is_logged_at_warning(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger=execution.log.name)
    with patch("execution._risk_per_share", side_effect=ValueError("bad")), \
         patch("db.log_trade_outcome", return_value=True) as log_outcome:
        execution.log_buy_outcome(_approval(), "NVDA", "OPEN", 107.0)
    assert any("risk-per-share record failed" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)
    kw = log_outcome.call_args.kwargs
    assert kw["planned_risk_per_share"] is None and kw["fill_risk_per_share"] is None


def test_the_outcome_row_carries_imputed_fields():
    """imputed_fields reaches trade_outcomes, so a stored default sub-score
    can be told from one the model gave (professor 65f). Record only."""
    approval = {**_approval(), "analyst_json": {"imputed_fields": ["catalyst_quality"]}}
    with patch("db.log_trade_outcome", return_value=True) as log_outcome:
        execution.log_buy_outcome(approval, "NVDA", "OPEN", 107.0)
    assert log_outcome.call_args.kwargs["imputed_fields"] == ["catalyst_quality"]


def test_imputed_fields_column_migration():
    import db
    sql = dict(db._MIGRATIONS)[86]
    assert "trade_outcomes" in sql and "imputed_fields JSONB" in sql
    src = open(db.__file__).read()
    assert "fill_risk_per_share, imputed_fields) " in src
