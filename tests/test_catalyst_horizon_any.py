"""Thesis snapshots record a catalyst horizon for any bucket (audit 11-06
11:34 #4): the gate's weeks_to_catalyst is None for every Compounder."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import db


def test_compounder_with_a_dated_company_catalyst():
    d = (datetime.now(timezone.utc) + timedelta(days=21)).date().isoformat()
    out = db.catalyst_horizon_any({"event": "Q4 earnings", "date": d})
    assert 2.5 < out["weeks_to_catalyst_any"] <= 3.0
    assert out["no_near_term_catalyst_any"] is False
    # the unchanged field still reads the gate's None as "no catalyst"
    assert db.compute_no_near_term_catalyst({"event": "Q4 earnings", "date": d}, None) is True


def test_undated_or_missing_catalyst():
    assert db.catalyst_horizon_any({"event": "ongoing AI demand", "date": "ongoing"}) == \
        {"weeks_to_catalyst_any": None, "no_near_term_catalyst_any": True, "days_to_catalyst_raw_any": None}
    assert db.catalyst_horizon_any(None)["no_near_term_catalyst_any"] is True


def test_never_raises():
    with patch("agents.analyst.compute_weeks_to_catalyst", side_effect=RuntimeError("x")):
        out = db.catalyst_horizon_any({"date": "2026-12-01"})
    assert out["weeks_to_catalyst_any"] is None and out["no_near_term_catalyst_any"] is None


def test_raw_days_match_catalyst_check():
    """SEF 139b: same reading as decision.catalyst_check, no join needed."""
    from agents import analyst
    past = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    with patch("db.get_next_earnings", return_value=None):
        for raw in (past, "2026-12"):
            cat = {"event": "Q4 earnings", "date": raw}
            assert db.catalyst_horizon_any(cat)["days_to_catalyst_raw_any"] == \
                analyst._catalyst_check("NVDA", {"catalyst": cat})["days_to_catalyst_raw"]
