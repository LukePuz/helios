"""Regime/scan measurement (audit 2026-10-28 05:00, finding 4)."""
import tasks


def test_scan_summary_line():
    regime = {"regime": "Neutral", "score": 53, "signals": {"missing_inputs": ["credit"]}}
    cands = [{"priority": "MEDIUM", "composite_score": 66}, {"priority": "LOW", "composite_score": 46},
             {"priority": "PASS", "composite_score": 20}]
    line = tasks._scan_summary(regime, cands)
    assert "regime Neutral score 53 -> threshold 65" in line
    assert "HIGH 0 MEDIUM 1 LOW 1 PASS 1" in line and "top scores [66, 46, 20]" in line
    assert "regime inputs unavailable: credit" in line
