"""Log-only add_context on ADD routes (audit 2026-11-10 16:53 #2)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents import crew


def _client(avg=None, err=None):
    c = MagicMock()
    if err:
        c.get_open_position.side_effect = err
    else:
        c.get_open_position.return_value = SimpleNamespace(avg_entry_price=avg)
    return c


def test_nvda_11_10_add():
    sizing = {"stop_loss_price": 146.24, "atr_stop": 8.44}
    with patch("db.get_position_stop", return_value={"stop_price": 163.68}):
        rec = crew._add_context("NVDA", 166.56, sizing, _client(avg=183.39))
    assert rec["kept_stop"] == 163.68 and rec["stop_distance_atr"] == 0.34
    assert rec["below_avg_cost"] is True and rec["pct_vs_avg_cost"] == -9.18
    assert rec["error"] is None


def test_new_stop_above_the_record_is_kept():
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}):
        rec = crew._add_context("X", 110.0, {"stop_loss_price": 100.0, "atr_stop": 5.0}, _client(avg=100.0))
    assert rec["kept_stop"] == 100.0 and rec["stop_distance_atr"] == 2.0 and rec["below_avg_cost"] is False


def test_unreadable_record_and_position_are_recorded_not_raised():
    with patch("db.get_position_stop", side_effect=RuntimeError("db")):
        rec = crew._add_context("X", 110.0, {"stop_loss_price": 100.0, "atr_stop": 5.0},
                                _client(err=RuntimeError("ib")))
    assert rec["kept_stop"] == 100.0 and rec["avg_cost"] is None
    assert "stop record: RuntimeError" in rec["error"] and "position: RuntimeError" in rec["error"]


def test_missing_atr_leaves_distance_null():
    with patch("db.get_position_stop", return_value=None):
        rec = crew._add_context("X", 110.0, {"stop_loss_price": 100.0}, _client(avg=0))
    assert rec["stop_distance_atr"] is None and rec["avg_cost"] is None and rec["error"] is None


def test_never_raises_on_garbage():
    rec = crew._add_context("X", "not-a-price", {}, None)
    assert rec["error"] == "ValueError"
