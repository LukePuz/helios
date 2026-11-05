import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents import crew


def _result(action="ADD", path="scheduled_review", new_fact=None):
    trig = {"path": path}
    if new_fact is not None:
        trig["new_fact"] = new_fact
    return {"action": action, "provenance": {"trigger": trig}}


def _client(avg):
    c = MagicMock()
    c.get_all_positions.return_value = [SimpleNamespace(symbol="ZZZ", avg_entry_price=avg)]
    return c


def _run(result, entry=100.0, price=90.0, client=None):
    with patch("db.get_position_stop", return_value={"entry_price": entry} if entry else None):
        return crew._add_to_loser_shadow("ZZZ", result, price, client)


def test_not_an_add():
    assert _run(_result(action="OPEN")) is None


@pytest.mark.parametrize("path, new_fact, price, breaks", [
    ("scheduled_review", None, 90.0, True),    # review path, below entry: unchecked today
    ("event", False, 90.0, True),
    ("event", True, 90.0, False),              # new catalyst: allowed by the rule
    ("scheduled_review", None, 110.0, False),  # above entry: not a loser
])
def test_breaks_rule_on_last_buy_without_a_client(path, new_fact, price, breaks, caplog):
    with caplog.at_level(logging.WARNING):
        rec = _run(_result(path=path, new_fact=new_fact), price=price)
    assert rec["basis"] == "last_buy" and rec["pct_vs_avg_cost"] is None
    assert rec["avg_cost_read"] == "no_client"
    assert rec["breaks_rule"] is breaks and rec["path"] == path
    assert rec["pct_vs_last_buy"] == pytest.approx(price - 100)
    assert any("never add to a loser" in r.getMessage() for r in caplog.records) is breaks


def test_avg_cost_is_the_basis_when_the_broker_reads():
    """An earlier ADD at 100 after a first buy at 80: avg 90. At 95 the
    position is in profit although it is below the last buy (m538)."""
    rec = _run(_result(), entry=100.0, price=95.0, client=_client(90.0))
    assert rec["basis"] == "avg_cost" and rec["below_entry"] is False and rec["breaks_rule"] is False
    assert rec["avg_cost_read"] == "ok"
    assert rec["pct_vs_last_buy"] == pytest.approx(-5.0)
    assert rec["pct_vs_avg_cost"] == pytest.approx(5.56)


def test_broker_error_falls_back_to_last_buy():
    c = MagicMock()
    c.get_all_positions.side_effect = RuntimeError("down")
    rec = _run(_result(), client=c)
    assert rec["basis"] == "last_buy" and rec["breaks_rule"] is True
    assert rec["avg_cost_read"] == "error: RuntimeError"


def test_not_held_at_broker():
    c = MagicMock()
    c.get_all_positions.return_value = []
    assert _run(_result(), client=c)["avg_cost_read"] == "not_held"


def test_new_fact_ignored_off_the_news_path():
    rec = _run(_result(path="scheduled_review", new_fact=True))
    assert rec["new_fact"] is None and rec["breaks_rule"] is True


def test_unknown_entry_or_price():
    assert _run(_result(), entry=None) == {"entry_known": False, "price_known": True, "below_entry": None,
                                            "avg_cost_read": "no_client"}
    assert _run(_result(), price=0) == {"entry_known": True, "price_known": False, "below_entry": None,
                                        "avg_cost_read": "no_client"}


def test_attached_and_guarded():
    with patch("agents.crew._compiled_plan", return_value=None), \
         patch("db.get_position_stop", return_value={"entry_price": 100.0}):
        sizing = {"shares_initial": 1}
        crew._attach_shadows(sizing, "ZZZ", _result(), 90.0, _client(95.0))
    assert sizing["add_to_loser"]["basis"] == "avg_cost" and sizing["add_to_loser"]["breaks_rule"] is True
    assert sizing["shadow_status"]["add_to_loser"] == "ok"
    with patch("agents.crew._compiled_plan", return_value=None), \
         patch("db.get_position_stop", side_effect=RuntimeError("db down")):
        sizing = {"shares_initial": 1}
        crew._attach_shadows(sizing, "ZZZ", _result(), 90.0)
    assert sizing["shadow_status"]["add_to_loser"] == "error"
    with patch("agents.crew._compiled_plan", return_value=None):
        sizing = {"shares_initial": 1}
        crew._attach_shadows(sizing, "ZZZ", _result(action="OPEN"), 90.0)
    assert sizing["shadow_status"]["add_to_loser"] == "n/a"
