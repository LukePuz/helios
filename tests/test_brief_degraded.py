"""An event verdict built on the fallback brief (event text only: no book,
no price) says so in provenance.trigger: brief_degraded, brief_built_at and
the error class. Log-only."""
from datetime import datetime
from unittest.mock import MagicMock, patch

from agents import crew

_EVENT = {"ticker": "NVDA", "source": "news", "title": "NVIDIA signs supply deal", "text": "x"}


def _run(ibkr_patch, brief_patch):
    seen = {}

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        seen["trigger"], seen["brief"] = trigger, situation_brief
        return {"action": "HOLD", "ticker": "NVDA", "conviction": 5, "pass_to_risk": False}
    with ibkr_patch, brief_patch, \
         patch("agents.crew._get_positions", return_value={}), \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("agents.analyst.analyze", fake_analyze), \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("agents.crew._c6_bar_unrouted"):
        crew.deep_analyze(dict(_EVENT), {"relevant": True})
    return seen


def test_fallback_brief_is_flagged():
    seen = _run(patch("ibkr.IBKRClient", side_effect=RuntimeError("no broker")),
                patch("context_engine.assemble_brief"))
    t = seen["trigger"]
    assert seen["brief"].startswith("Event: ")
    assert t["brief_degraded"] is True and t["brief_error"] == "RuntimeError"
    datetime.fromisoformat(t["brief_built_at"])


def test_normal_brief_is_not_flagged():
    seen = _run(patch("ibkr.IBKRClient", return_value=MagicMock()),
                patch("context_engine.assemble_brief", return_value="full brief"))
    t = seen["trigger"]
    assert seen["brief"] == "full brief"
    assert t["brief_degraded"] is False and "brief_error" not in t
    datetime.fromisoformat(t["brief_built_at"])
