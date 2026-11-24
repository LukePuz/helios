"""Risk controls that used to turn off without an error:
- risk_manager's config import fallback now logs an ERROR and alerts once;
- a raising theme / moonshot-total concentration sub-check is recorded and
  alerted (RISK_SUBCHECK_FAIL_MODE shadow) or capped at 0 (enforce);
- a missing watchlist.md logs an ERROR and the empty result isn't cached."""
import importlib.util
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import knowledge_loader
import risk_manager
from risk_manager import RiskManager


def _text(send):
    """The alert text of the last alerts._send call; every send here is one attempt."""
    assert send.call_args.kwargs.get("_max_attempts") == 1
    return send.call_args.kwargs.get("text") or send.call_args.args[0]

# ---------------------------------------------------------------- d1: config fallback


def test_normal_import_has_no_fallback():
    assert risk_manager._CONFIG_FALLBACK_ERROR is None


def test_config_fallback_logs_an_error_and_alerts_once(caplog):
    stub = types.ModuleType("config")          # none of the names risk_manager imports
    spec = importlib.util.spec_from_file_location("risk_manager_fallback_copy", risk_manager.__file__)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"config": stub}), patch("alerts._send") as send, \
            caplog.at_level(logging.ERROR):
        spec.loader.exec_module(mod)
    assert mod._CONFIG_FALLBACK_ERROR.startswith("ImportError: cannot import name")
    assert mod.TRADING_ENV == "paper" and mod.DRAWDOWN_HALT_PCT == 0.10     # fallbacks unchanged
    assert send.call_count == 1 and "RISK CONFIG FALLBACK" in _text(send)
    assert any("config import failed" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)


def test_config_fallback_with_a_raising_alert_still_imports():
    stub = types.ModuleType("config")
    spec = importlib.util.spec_from_file_location("risk_manager_fallback_copy2", risk_manager.__file__)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"config": stub}), patch("alerts._send", side_effect=RuntimeError("down")):
        spec.loader.exec_module(mod)
    assert mod._CONFIG_FALLBACK_ERROR and hasattr(mod, "RiskManager")


# ---------------------------------------------------------------- d2/d3: sub-checks

WL = {"NVDA": {"sector": "AI Infrastructure — Compute / Accelerators", "role": "Compounder"},
      "IONQ": {"sector": "Quantum Computing", "role": "Moonshot"}}


def _rm(simulation=False):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1000.0)
    client.get_all_positions.return_value = [MagicMock(symbol="NVDA", market_value=100.0),
                                             MagicMock(symbol="IONQ", market_value=20.0)]
    return RiskManager(client, simulation=simulation)


@pytest.fixture(autouse=True)
def _reset_throttle():
    risk_manager._SUBCHECK_ALERT_AT.clear()
    yield
    risk_manager._SUBCHECK_ALERT_AT.clear()


def _conc(ticker="NVDA", bucket="Compounder", mode="shadow", wl_effect=RuntimeError("watchlist broke"),
          simulation=False, rm=None):
    rm = rm or _rm(simulation)
    with patch("config.RISK_SUBCHECK_FAIL_MODE", mode), \
         patch("knowledge_loader.load_watchlist", side_effect=wl_effect), \
         patch("db.log_activity_queued") as row, patch("alerts._send") as send:
        out = rm.check_concentration(ticker, 0.05, bucket=bucket)
    return out, row, send


def test_shadow_theme_failure_sizes_as_before_and_records_it():
    out, row, send = _conc(mode="shadow")
    assert out["theme_size_cap_pct"] is None and out["passed"] is True and "errored" not in out
    assert out["subcheck_failures"] == [{"check": "theme", "error": "RuntimeError", "mode": "shadow",
                                         "would_fail_closed": True, "applied": False}]
    assert row.call_count == 1
    stage, ticker, _msg, sentiment, extra = row.call_args.args
    assert (stage, ticker, sentiment) == ("risk_subcheck_fail", "NVDA", "alert")
    assert extra["would_fail_closed"] is True and extra["applied"] is False
    assert send.call_count == 1 and "Enforce would have blocked it" in _text(send)


def test_enforce_theme_failure_caps_at_zero():
    out, row, send = _conc(mode="enforce")
    assert out["theme_size_cap_pct"] == 0.0 and out["passed"] is False and out["errored"] is True
    assert any(f.startswith("RISK CHECK FAILED — theme") for f in out["flags"])
    assert out["subcheck_failures"][0]["applied"] is True
    assert "blocked (enforce)" in _text(send)


def test_enforce_blocks_the_shares_and_shadow_does_not():
    sizing = {"shares_initial": 4, "position_size_pct": 0.05, "order_pct": 0.05}
    for mode, want in (("shadow", 4), ("enforce", 0)):
        rm = _rm()
        with patch("config.RISK_SUBCHECK_FAIL_MODE", mode), \
             patch("knowledge_loader.load_watchlist", side_effect=RuntimeError("x")), \
             patch("db.log_activity_queued"), patch("alerts._send"):
            assert rm.shares_after_concentration("NVDA", sizing, "Compounder") == want, mode


def test_moonshot_total_failure_alone():
    # theme read works, the moonshot read raises
    out, row, _ = _conc(ticker="IONQ", bucket="Moonshot", mode="enforce",
                        wl_effect=[WL, RuntimeError("second read")])
    assert out["theme_size_cap_pct"] is not None and out["theme_size_cap_pct"] > 0
    assert out["moonshot_size_cap_pct"] == 0.0 and out["errored"] is True
    assert [r["check"] for r in out["subcheck_failures"]] == ["moonshot_total"]


def test_no_failure_adds_nothing():
    out, row, send = _conc(mode="enforce", wl_effect=lambda: WL)
    assert out["subcheck_failures"] == [] and "errored" not in out
    assert not row.called and not send.called


def test_alert_throttled_hourly_per_check_rows_every_time():
    rm = _rm()
    _, row1, send1 = _conc(rm=rm)
    _, row2, send2 = _conc(rm=rm)
    assert row1.call_count == 1 and row2.call_count == 1
    assert send1.call_count == 1 and send2.call_count == 0


def test_simulation_writes_and_sends_nothing_but_enforce_still_caps():
    out, row, send = _conc(mode="enforce", simulation=True)
    assert out["theme_size_cap_pct"] == 0.0 and not row.called and not send.called


def test_unknown_mode_reads_as_shadow():
    with patch("config.RISK_SUBCHECK_FAIL_MODE", "ENFORCE-ish"):
        assert risk_manager._subcheck_fail_mode() == "shadow"
    with patch("config.RISK_SUBCHECK_FAIL_MODE", " Enforce "):
        assert risk_manager._subcheck_fail_mode() == "enforce"


def test_raising_row_and_alert_never_break_the_check():
    rm = _rm()
    with patch("config.RISK_SUBCHECK_FAIL_MODE", "enforce"), \
         patch("knowledge_loader.load_watchlist", side_effect=RuntimeError("x")), \
         patch("db.log_activity_queued", side_effect=RuntimeError("db")), \
         patch("alerts._send", side_effect=RuntimeError("tg")):
        out = rm.check_concentration("NVDA", 0.05, bucket="Compounder")
    assert out["theme_size_cap_pct"] == 0.0


def test_shipped_default_is_shadow():
    import config
    assert config.RISK_SUBCHECK_FAIL_MODE == "shadow"


# ---------------------------------------------------------------- d4: empty / missing watchlist


@pytest.mark.parametrize("mode,cleared", [("enforce", True), ("shadow", False)])
def test_empty_watchlist_is_a_subcheck_failure_cache_cleared_only_in_enforce(mode, cleared, caplog):
    loader = MagicMock(return_value={})
    rm = _rm()
    with patch("config.RISK_SUBCHECK_FAIL_MODE", mode), \
         patch("knowledge_loader.load_watchlist", loader), \
         patch("db.log_activity_queued") as row, patch("alerts._send"), \
         caplog.at_level(logging.ERROR):
        out = rm.check_concentration("NVDA", 0.05, bucket="Compounder")
    assert out["theme_size_cap_pct"] == (0.0 if mode == "enforce" else None)
    assert out["subcheck_failures"][0]["check"] == "theme"
    assert loader.cache_clear.called is cleared
    assert any("watchlist read empty" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)
    assert row.call_count == 1


def test_empty_watchlist_shadow_keeps_todays_values():
    """Shadow: the theme record and the moonshot cap are what they were before
    this change (parity with helios3d's limits.concentration); enforce: 0."""
    for mode in ("shadow", "enforce"):
        rm = _rm()
        with patch("config.RISK_SUBCHECK_FAIL_MODE", mode), \
             patch("knowledge_loader.load_watchlist", MagicMock(return_value={})), \
             patch("db.log_activity_queued"), patch("alerts._send"):
            out = rm.check_concentration("IONQ", 0.05, bucket="Moonshot")
        assert [r["check"] for r in out["subcheck_failures"]] == ["theme", "moonshot_total"]
        assert out["theme_shadow"]["reason"] == "no_watchlist_sector"
        if mode == "shadow":
            assert out["moonshot_size_cap_pct"] == round(risk_manager.MAX_MOONSHOT_TOTAL_PCT + 0.05, 4)
            assert out["theme_size_cap_pct"] is None and out["passed"] is True
        else:
            assert out["moonshot_size_cap_pct"] == 0.0 and out["theme_size_cap_pct"] == 0.0


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_missing_watchlist_file_end_to_end(mode, tmp_path, monkeypatch):
    """A missing watchlist.md is recorded in both modes. Once the file is back,
    enforce re-reads it on the next check; shadow leaves the loader's cache
    alone (any caller's later read is unchanged), so it stays empty."""
    knowledge_loader.load_watchlist.cache_clear()
    monkeypatch.setattr(knowledge_loader, "_KNOWLEDGE_DIR", str(tmp_path))
    def check():                       # the real (cached) loader, not a mock
        rm = _rm()
        with patch("config.RISK_SUBCHECK_FAIL_MODE", mode), \
             patch("db.log_activity_queued") as row, patch("alerts._send"):
            return rm.check_concentration("NVDA", 0.05, bucket="Compounder"), row
    try:
        out, row = check()
        assert out["subcheck_failures"][0]["check"] == "theme" and row.call_count == 1
        (tmp_path / "watchlist.md").write_text(
            "### AI Infrastructure — Compute / Accelerators\n| NVDA | Nvidia | Compounder | Core |\n")
        out2, row2 = check()
        if mode == "enforce":
            assert out2["subcheck_failures"] == [] and out2["theme_size_cap_pct"] is not None
            assert not row2.called
        else:
            assert out2["subcheck_failures"][0]["check"] == "theme" and row2.call_count == 1
            assert knowledge_loader.load_watchlist() == {}
    finally:
        knowledge_loader.load_watchlist.cache_clear()


def test_quiet_what_ifs_write_and_send_nothing():
    rm = _rm()
    rm.quiet_fallback_alert = True                 # the swap pre-check's sim, sizing_dryrun
    out, row, send = _conc(mode="shadow", rm=rm)
    assert out["subcheck_failures"][0]["check"] == "theme"
    assert not row.called and not send.called
    import os
    src = open(os.path.join(os.path.dirname(risk_manager.__file__), "scripts", "sizing_dryrun.py")).read()
    assert "rm.quiet_fallback_alert = True" in src
    assert "sim.quiet_fallback_alert = True" in open(risk_manager.__file__).read()


def test_knowledge_loader_source_untouched():
    """knowledge_loader's source feeds the analyst's prompt_template_full_sha."""
    from agents import analyst
    import hashlib
    assert hashlib.sha256(analyst._prompt_template_full_text().encode()).hexdigest()[:12] == "c553694b23b4"
