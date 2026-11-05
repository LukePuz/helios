import logging
from unittest.mock import patch

import pytest

from agents import crew


@pytest.mark.parametrize("entry, want", [
    (None, "watchlist entry missing"),
    ({"tier": "Core"}, "watchlist entry has no role"),
    ({"role": "Compounder"}, None),
    ({"role": None}, "empty"),
    ({"role": ""}, "empty"),
])
def test_held_review_logs_a_defaulted_role(entry, want, caplog):
    seen = {}

    def fake_analyze(candidate, **kw):
        seen["role"] = candidate["role"]
        return {"action": "HOLD"}

    with patch("context_engine.assemble_brief", return_value="brief"), \
         patch("knowledge_loader.get_watchlist_entry", return_value=entry), \
         patch("agents.analyst.analyze", side_effect=fake_analyze), \
         caplog.at_level(logging.WARNING):
        crew._analyze_held_position("SNOW", None, {"SNOW": {"current_price": 100.0}}, {"regime": "Neutral"})
    msgs = [r.getMessage() for r in caplog.records if "role defaulted" in r.getMessage()]
    if want == "empty":
        role = entry["role"]
        assert [r.getMessage() for r in caplog.records if "role is empty" in r.getMessage()] == [
            f"[SNOW] held review role is empty ({role!r}), passed as is"]
        assert msgs == [] and seen["role"] == role
    elif want:
        assert msgs == [f"[SNOW] held review role defaulted to Accelerator ({want})"]
        assert seen["role"] == "Accelerator"
    else:
        assert msgs == [] and seen["role"] == "Compounder"
