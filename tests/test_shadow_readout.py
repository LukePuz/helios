"""Shadow readout counts key paths and true booleans, never values (A1031-1453-4)."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "shadow_readout", os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_readout.py"))
shadow_readout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shadow_readout)


def test_counts_paths_and_true_booleans_without_values():
    rows = [
        ("HOLD", {"gate_flags": {"rubric_gate_conflict": True, "verdicts_today": 3}}, None),
        ("HOLD", {"gate_flags": {"rubric_gate_conflict": False}}, {"price": 123.45}),
        ("SELL_SKIPPED_CLOSED", None, None),
    ]
    text = shadow_readout.readout(rows, 7, "live", "abc123")
    assert "| HOLD | 2 |" in text and "| SELL_SKIPPED_CLOSED | 1 |" in text
    assert "| provenance.gate_flags.rubric_gate_conflict | 2 | 1 |" in text
    assert "| provenance.gate_flags.verdicts_today | 1 |  |" in text
    assert "123.45" not in text and "Rows: 3" in text


def test_depth_is_capped():
    deep = {"a": {"b": {"c": {"d": True}}}}
    text = shadow_readout.readout([("X", deep, None)], 7, "live", "h")
    assert "provenance.a.b" in text and "provenance.a.b.c.d" not in text
