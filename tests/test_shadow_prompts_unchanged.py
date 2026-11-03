"""The log-only shadows (roadmap 01(c)) must leave what the models read
byte-identical. These hashes are the analyst and screener prompts at
c93ebe5, before the shadows. A deliberate prompt change (Luke's call)
updates them in the same commit."""
import hashlib
from unittest.mock import patch

from agents import analyst, screener

_CAND = {"ticker": "NVDA", "role": "Accelerator", "tier": "Core", "sector": "Semis", "held": True}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_analyst_prompt_template_unchanged():
    assert _sha(analyst._prompt_template_full_text())[:12] == "c553694b23b4"
    assert analyst._template_shas()["prompt_template_sha"] == "6b93fe2037b7"


def test_analyst_rendered_prompt_unchanged():
    with patch.object(analyst, "_build_track_record_text", return_value="TRACK RECORD"):
        p = analyst._build_prompt(_CAND, "Current Price: $100.00\nBRIEF", "THESIS",
                                  {"regime": "Neutral", "score": 50}, "PHILO")
    assert _sha(p) == "fe15dc437b317cf83064a1b5cc89ea7bd375812430d206800a2b8fb6a3215f34"


def test_screener_prompt_template_unchanged():
    assert _sha(screener._SCREENER_PROMPT) == (
        "344564c4765893c4f8cec555d4ed68a30787747b8d6986f65e7782d1bb4447c9")
