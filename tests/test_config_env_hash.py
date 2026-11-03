"""Every env-overridable config.py setting is hashed into provenance (M182)."""
import re
from pathlib import Path
from unittest.mock import patch

import config
from agents import analyst, screener


def test_every_getenv_name_in_config_is_found():
    names = analyst._config_env_names()
    src = (Path(config.__file__)).read_text()
    # Each `NAME = ...getenv("KEY"...` line at any indentation outside functions.
    expected = set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)\s*=.*getenv\(', src, re.M))
    assert expected and expected <= set(names), expected - set(names)
    assert names["START_VALUE"] == "LIVE_START_VALUE"      # found inside the if/else
    assert names["USE_BRACKET_ORDERS"] == "USE_BRACKET_ORDERS"   # via `_os`


def test_hash_moves_with_a_value_and_skips_excluded_names(monkeypatch):
    monkeypatch.setattr(analyst, "_CONFIG_ENV_CACHE", {})
    base = dict(analyst._config_env_record())
    assert len(base["env_sha"]) == 12 and base["n"] > 20
    assert "IB_CLIENT_ID" in base["excluded"] and "START_VALUE" in base["excluded"]

    monkeypatch.setattr(analyst, "_CONFIG_ENV_CACHE", {})
    monkeypatch.setattr(config, "IB_CLIENT_ID", 9999)          # per-process: not hashed
    assert analyst._config_env_record()["env_sha"] == base["env_sha"]

    monkeypatch.setattr(analyst, "_CONFIG_ENV_CACHE", {})
    monkeypatch.setattr(config, "QUIVER_ENABLED", not config.QUIVER_ENABLED)
    assert analyst._config_env_record()["env_sha"] != base["env_sha"]


def test_env_set_lists_names_only(monkeypatch):
    monkeypatch.setattr(analyst, "_CONFIG_ENV_CACHE", {})
    monkeypatch.setenv("QUIVER_ENABLED", "secret-looking-value")
    rec = analyst._config_env_record()
    assert "QUIVER_ENABLED" in rec["env_set"]
    assert "secret-looking-value" not in repr(rec)


def test_verdict_and_screener_provenance_carry_the_config_hash():
    with patch.object(analyst, "_code_mixed_check", return_value={}):
        assert analyst._provenance("p")["config_env"]["env_sha"]
    with patch("agents.screener._call_ollama",
               return_value='{"relevant": false, "urgency": "low", "reason": "noise"}'), \
         patch("feed.emit"):
        prov = screener.screen({"ticker": "NVDA", "source": "news", "title": "t", "text": "x"})["provenance"]
    assert prov["config"]["SCREENER_MODEL"] == screener.SCREENER_MODEL
    assert prov["config"]["env"]["env_sha"] == analyst._config_env_record()["env_sha"]
