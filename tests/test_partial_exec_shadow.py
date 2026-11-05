from unittest.mock import patch

import pytest

from agents import crew


@pytest.mark.parametrize("qty, profit, executable, trim, full_exit", [
    (1, {"Trending/Expanding": 0, "Choppy/Distribution": 0}, False, 1, True),
    (3, {"Trending/Expanding": 0, "Choppy/Distribution": 1}, False, 1, False),
    (4, {"Trending/Expanding": 1, "Choppy/Distribution": 1}, True, 1, False),
    (10, {"Trending/Expanding": 2, "Choppy/Distribution": 4}, True, 3, False),
])
def test_partial_exec_shadow(qty, profit, executable, trim, full_exit):
    got = crew._partial_exec_shadow({"shares_initial": qty}, "Accelerator")
    assert got == {"qty": qty, "bucket": "Accelerator", "profit_target_applies": True,
                   "profit_partial_shares": profit,
                   "profit_partial_executable": executable,
                   "trim_shares": trim, "trim_is_full_exit": full_exit}


def test_compounder_has_no_profit_partial():
    got = crew._partial_exec_shadow({"shares_initial": 2}, "Compounder")
    assert got["profit_target_applies"] is False
    assert got["profit_partial_shares"] is None and got["profit_partial_executable"] is None
    assert got["trim_shares"] == 1 and got["trim_is_full_exit"] is False


def test_partial_exec_shadow_na_without_shares():
    assert crew._partial_exec_shadow({"shares_initial": 0}) is None
    assert crew._partial_exec_shadow({}) is None


def test_attach_shadows_records_partial_exec_and_survives_an_error():
    with patch("agents.crew._compiled_plan", return_value=None):
        sizing = {"shares_initial": 1, "stop_loss_price": 90.0}
        crew._attach_shadows(sizing, "ZZZ", {"thesis_bucket": "Moonshot"}, 100.0)
        assert sizing["partial_exec"]["trim_is_full_exit"] is True
        assert sizing["partial_exec"]["bucket"] == "Moonshot"
        assert sizing["shadow_status"]["partial_exec"] == "ok"
        with patch("agents.crew._partial_exec_shadow", side_effect=RuntimeError("boom")):
            sizing = {"shares_initial": 1}
            crew._attach_shadows(sizing, "ZZZ", {}, 100.0)
        assert sizing["shadow_status"]["partial_exec"] == "error"
