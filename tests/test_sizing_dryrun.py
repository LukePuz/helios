"""scripts/sizing_dryrun.py mirrors crew's sizing steps (professor M29)."""
import importlib.util
import os
from unittest.mock import MagicMock

_spec = importlib.util.spec_from_file_location(
    "sizing_dryrun", os.path.join(os.path.dirname(__file__), "..", "scripts", "sizing_dryrun.py"))
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)


def _rm(sizing, conc):
    rm = MagicMock()
    rm.size_position.return_value = sizing
    rm.check_concentration.return_value = conc
    return rm


def test_zero_from_size_position_reports_its_flag():
    rm = _rm({"shares_initial": 0, "flags": ["Allocation buys less than 1 share"]}, {"passed": True})
    shares, why = dr.size_like_crew(rm, "MSFT", 7, "Compounder", "Neutral", 500.0)
    assert shares == 0 and dr._reason_category(shares, why) == "under_one_share"


def test_concentration_cap_int_rounding_to_zero_matches_crew():
    rm = _rm({"shares_initial": 1, "position_size_pct": 0.15, "flags": []},
             {"passed": False, "flags": ["theme"], "theme_size_cap_pct": 0.10})
    shares, why = dr.size_like_crew(rm, "NVDA", 7, "Accelerator", "Neutral", 100.0)
    assert shares == 0 and dr._reason_category(shares, why) == "concentration_zero"


def test_buyable():
    rm = _rm({"shares_initial": 2, "position_size_pct": 0.15, "flags": []}, {"passed": True})
    shares, why = dr.size_like_crew(rm, "SOFI", 7, "Accelerator", "Neutral", 16.0)
    assert shares == 2 and dr._reason_category(shares, why) == "ok"


def test_refuses_csv_when_setup_gates_are_on(monkeypatch, tmp_path):
    import sys
    import pytest
    import config
    monkeypatch.setattr(config, "SETUP_GATES_ENABLED", True)
    monkeypatch.setattr(sys, "argv", ["sizing_dryrun.py", "--csv", str(tmp_path / "x.csv")])
    with pytest.raises(SystemExit) as e:
        dr.main()
    assert "SETUP_GATES_ENABLED" in str(e.value)
    assert not (tmp_path / "x.csv").exists()
