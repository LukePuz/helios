"""The stop's ATR and the brief's ATR recorded side by side (audit
2026-11-07 11:07 #7). Log-only."""
import inspect
from decimal import Decimal
from unittest.mock import patch

import risk_manager
from agents import crew


def test_atr_brief_reads_the_ladder():
    with patch("db.get_trade_ladder", return_value={"atr": Decimal("4.1234567"),
                                                     "computed_at": "2026-11-06 17:15"}):
        assert crew._atr_brief("NVDA") == {"atr": 4.1235, "ladder_row": True,
                                           "computed_at": "2026-11-06 17:15"}
    with patch("db.get_trade_ladder", return_value=None):
        assert crew._atr_brief("NVDA") == {"atr": None, "ladder_row": False, "computed_at": None}
    with patch("db.get_trade_ladder", side_effect=RuntimeError("db")):
        assert crew._atr_brief("NVDA") is None


def test_calculate_stop_keeps_the_atr_it_used():
    rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
    rm.simulation = True
    with patch.object(rm, "get_atr", return_value=3.5), \
         patch.object(risk_manager, "SETUP_GATES_ENABLED", False, create=True):
        rm.calculate_stop("NVDA", 100.0, "Accelerator", log_fallback=False)
    assert rm.last_stop_atr == 3.5


def test_size_position_records_atr_stop_and_resets_it():
    src = inspect.getsource(risk_manager.RiskManager.size_position)
    assert "self.last_stop_atr = None\n        stop_price, stop_method = self.calculate_stop(" in src
    assert '"atr_stop":' in src


def test_crew_records_atr_brief():
    assert 'sizing["atr_brief"] = _atr_brief(ticker)' in inspect.getsource(crew)


def test_a_zero_atr_records_none():
    src = inspect.getsource(risk_manager.RiskManager.size_position)
    assert "and self.last_stop_atr > 0 else None" in src
