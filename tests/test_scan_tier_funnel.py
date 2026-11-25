"""The daily scan's tier funnel row (agents/scan_funnel.py): one record-only
activity row per run_daily_scan pass, after its enqueue, with per-tier
counts from the watchlist to the analyst. Also pins the shared selection
(crew._select_for_analysis) that process_candidates uses, and run_scan's
additive skip key and watchlist_tiers."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

import config
import tasks
from agents import crew, scan_funnel
from agents import scanner

pytestmark = pytest.mark.switches(SCAN_TIER_FUNNEL_RECORD_ENABLED=True)


def _c(ticker, priority, score=50, tier="Watch"):
    return {"ticker": ticker, "priority": priority, "composite_score": score, "tier": tier}


# ── the shared selection: process_candidates is unchanged ────────────────────

MIXED = [_c("M1", "MEDIUM", 90), _c("H1", "HIGH", 85), _c("M2", "MEDIUM", 80),
         _c("H2", "HIGH", 70), _c("H3", "HIGH", 60), _c("M3", "MEDIUM", 55), _c("L1", "LOW", 40)]


@pytest.mark.parametrize("cap, expected", [
    (5, ["H1", "H2", "H3", "M1", "M2"]),
    (4, ["H1", "H2", "H3", "M1"]),
    (3, ["H1", "H2", "H3"]),
    (2, ["H1", "H2"]),
    (10, ["H1", "H2", "H3", "M1", "M2", "M3"]),
])
def test_selection_takes_every_high_then_medium_at_the_cap(cap, expected):
    assert [c["ticker"] for c in crew._select_for_analysis(MIXED, cap)] == expected


def _old_selection(candidates, max_per_run):
    """process_candidates' selection before the extraction (d7ac25b)."""
    high = [c for c in candidates if c.get("priority") == "HIGH"]
    medium = [c for c in candidates if c.get("priority") == "MEDIUM"]
    return (high + medium)[:max_per_run]


@pytest.mark.parametrize("cap", [0, 1, 2, 3, 4, 5, 6, 7])
def test_selection_matches_the_old_inline_code(cap):
    assert crew._select_for_analysis(MIXED, cap) == _old_selection(MIXED, cap)


def test_process_candidates_hands_the_selection_to_the_analyst():
    batch = MagicMock(return_value=[])
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst.analyze_batch", batch), \
         patch("ibkr.IBKRClient", return_value=MagicMock()), \
         patch.object(crew, "_get_positions", MagicMock(return_value={})), \
         patch("agents.scan_guard.log_lost_candidates"):
        crew.process_candidates(list(MIXED), max_per_run=5)
    assert [c["ticker"] for c in batch.call_args.args[0]] == ["H1", "H2", "H3", "M1", "M2"]


# ── run_scan: the skip key and watchlist_tiers are additive ──────────────────

WATCHLIST = {"NOBAR": {"tier": "Core"}, "BLACK": {"tier": "Watch"}, "FLOOR": {"tier": "Watch"},
             "LOWW": {"tier": "Speculative"}, "HIGHH": {"tier": "Core"}, "NOTIER": {}}
SCORES = {"FLOOR": 10, "LOWW": 40, "HIGHH": 95, "NOTIER": 95}


def _run_scan(regime="Neutral"):
    def score(bars, ticker=""):
        return SCORES[ticker], "n"
    with patch("knowledge_loader.load_watchlist", return_value=dict(WATCHLIST)), \
         patch.object(scanner, "_get_recent_filings", return_value=set()), \
         patch.object(scanner, "_fetch_bars", return_value=[{"c": 1}]), \
         patch("agents.market_regime.completed_bars",
               side_effect=lambda bars: bars), \
         patch.object(scanner, "_fetch_bars",
               side_effect=lambda t, days=220: [] if t == "NOBAR" else [{"c": 1}]), \
         patch("db.is_near_earnings", side_effect=lambda t, days: t == "BLACK"), \
         patch.object(config, "EARNINGS_ENTRY_BLACKOUT_DAYS", 5), \
         patch.object(scanner, "_score_technical", side_effect=score), \
         patch.object(scanner, "_score_momentum_with_flow", return_value=(0, "")), \
         patch.object(scanner, "_score_catalyst", return_value=(0, "", "routine")), \
         patch.object(scanner, "_score_thesis_alignment", return_value=(0, "")), \
         patch("agents.setup_gates.classify_setup", return_value=(None, {})), \
         patch("db.log_scanner_candidate"):
        return scanner.run_scan({"regime": regime, "score": 50})


def test_run_scan_disqualified_entries_gain_only_the_skip_key():
    out = _run_scan()
    by = {d["ticker"]: d for d in out["disqualified"]}
    assert by["NOBAR"] == {"ticker": "NOBAR", "reason": "No price data available",
                           "composite_score": 0, "skip": "no_price_data"}
    assert by["BLACK"] == {"ticker": "BLACK", "reason": "Earnings within 5 days (blackout)",
                           "composite_score": 0, "skip": "earnings_blackout"}
    assert by["FLOOR"] == {"ticker": "FLOOR",
                           "reason": "Score 10/100 below threshold (65) in Neutral regime",
                           "composite_score": 10, "priority": "PASS", "skip": "below_floor"}
    assert out["watchlist_tiers"] == {"NOBAR": "Core", "BLACK": "Watch", "FLOOR": "Watch",
                                      "LOWW": "Speculative", "HIGHH": "Core", "NOTIER": None}
    assert {c["ticker"]: c["priority"] for c in out["candidates"]} == {
        "LOWW": "LOW", "HIGHH": "HIGH", "NOTIER": "HIGH"}
    assert "skip" not in out["candidates"][0]


def test_run_scan_summary_is_unchanged_by_the_skip_key():
    out = _run_scan()
    line = tasks._scan_summary({"regime": "Neutral", "score": 50}, out["candidates"],
                               out["disqualified"])
    assert "HIGH 2 MEDIUM 0 LOW 1 PASS 1 skipped 2" in line


# ── tier_funnel (pure) ───────────────────────────────────────────────────────

def test_funnel_counts_every_stage_by_tier_and_the_identity_holds():
    out = _run_scan()
    high_medium = [c for c in out["candidates"] if c["priority"] in ("HIGH", "MEDIUM")]
    sent = crew._select_for_analysis(high_medium, 1)              # cap 1: one over the cap
    f = scan_funnel.tier_funnel(out["watchlist_tiers"], out, sent)
    assert list(f["by_tier"]) == ["Core", "Watch", "Speculative", "unknown"]
    core, watch, spec, unk = (f["by_tier"][k] for k in ("Core", "Watch", "Speculative", "unknown"))
    assert (core["watchlist_in"], core["no_price_data"], core["sent_to_analyst"],
            core["sent_high"]) == (2, 1, 1, 1)
    assert (watch["watchlist_in"], watch["earnings_blackout"], watch["below_floor"]) == (2, 1, 1)
    assert (spec["watchlist_in"], spec["below_regime_threshold"]) == (1, 1)
    assert (unk["watchlist_in"], unk["over_run_cap"]) == (1, 1)
    assert f["total"]["watchlist_in"] == 6 and f["total"]["unclassified"] == 0
    assert all(t["identity_ok"] for t in (*f["by_tier"].values(), f["total"]))
    assert f["not_in_watchlist"] == 0


def test_strong_risk_off_counts_every_name_as_suppressed():
    f = scan_funnel.tier_funnel({"A": "Core", "B": "Watch"}, {"candidates": [], "disqualified": []},
                                [], suppressed=True)
    assert f["by_tier"]["Core"]["regime_suppressed"] == 1
    assert f["total"]["regime_suppressed"] == 2 and f["total"]["identity_ok"]


def test_an_unplaced_name_is_unclassified_not_hidden():
    scan = {"candidates": [_c("X", "WEIRD")], "disqualified": [{"ticker": "Y", "reason": "?"}]}
    f = scan_funnel.tier_funnel({"X": "Core", "Y": "Core", "Z": "Core"}, scan, [])
    assert f["by_tier"]["Core"]["unclassified"] == 3 and f["total"]["identity_ok"]


def test_counts_only_no_tickers_in_the_row():
    out = _run_scan()
    with patch("db.TRADING_ENV", "live"):
        row = scan_funnel.build_row(out, [c for c in out["candidates"] if c["priority"] == "HIGH"],
                                    {"regime": "Neutral", "score": 50}, 5, True, task_id="t1",
                                    now=datetime(2026, 11, 23, 14, 40, tzinfo=timezone.utc))
    text = repr(row)
    assert not any(t in text for t in WATCHLIST)
    assert row["day"] == "2026-11-23" and row["shadow_key"] == "live:2026-11-23:t1:0"
    assert row["threshold"] == 65 and row["max_per_run"] == 5 and row["enqueue"] == "queued"
    assert row["stages"] == list(scan_funnel.FUNNEL_KEYS)


# ── build_row's edge passes ──────────────────────────────────────────────────

def test_strong_risk_off_pass_reads_the_watchlist_for_its_tiers():
    scan = {"agent": "scanner", "regime_context": "Strong Risk-Off", "candidates": [],
            "disqualified": [], "scan_notes": "Scanner suppressed"}
    with patch("knowledge_loader.load_watchlist", return_value={"A": {"tier": "Core"}}):
        row = scan_funnel.build_row(scan, [], {"regime": "Strong Risk-Off"}, 5, False)
    assert row["total"]["regime_suppressed"] == 1 and row["enqueue"] == "none"
    assert row["threshold"] == 999 and row["watchlist_error"] is None


def test_watchlist_load_failure_is_an_empty_funnel_with_the_error():
    scan = {"agent": "scanner", "candidates": [], "disqualified": [],
            "scan_notes": "Watchlist load failed: boom"}
    row = scan_funnel.build_row(scan, [], {"regime": "Neutral"}, 5, False)
    assert row["total"]["watchlist_in"] == 0 and row["watchlist_error"].startswith("Watchlist load")
    assert row["by_tier"] == {}


# ── record and run_daily_scan ────────────────────────────────────────────────

@pytest.mark.switches(SCAN_TIER_FUNNEL_RECORD_ENABLED=False)
def test_off_writes_nothing_and_counts_nothing():
    with patch("agents.scan_funnel.build_row") as build, patch("db.log_activity") as act:
        assert scan_funnel.record({}, [], {}, 5, False) is False
    build.assert_not_called()
    act.assert_not_called()


def test_record_never_raises():
    with patch("agents.scan_funnel.build_row", side_effect=RuntimeError("x")):
        assert scan_funnel.record({}, [], {}, 5, False) is False
    with patch("db.log_activity", side_effect=RuntimeError("db")):
        assert scan_funnel.record({"candidates": [], "disqualified": [], "watchlist_tiers": {}},
                                  [], {}, 5, False) is False


def _daily(record_effect=None):
    out = _run_scan()
    events = []
    queue = MagicMock(side_effect=lambda *a, **k: events.append("enqueue"))
    act = MagicMock(side_effect=lambda *a, **k: events.append("row") or True)
    rec = ({"side_effect": record_effect} if record_effect is not None
           else {"wraps": scan_funnel.record})
    with patch("agents.market_regime.get_cached_regime",
               return_value={"regime": "Neutral", "score": 50}), \
         patch("agents.scanner.run_scan", return_value=out), \
         patch.object(tasks.analyze_candidates, "apply_async", new=queue), \
         patch("agents.scan_funnel.record", **rec) as record, \
         patch("db.log_activity", new=act):
        tasks.run_daily_scan.run()
    return queue, act, record, events


def test_daily_scan_enqueues_as_before_then_writes_one_row():
    queue, act, record, events = _daily()
    assert events == ["enqueue", "row"]
    assert queue.call_args.kwargs["kwargs"] == {"max_per_run": 5}
    assert [c["ticker"] for c in queue.call_args.kwargs["args"][0]] == ["HIGHH", "NOTIER"]
    stage, ticker, _, _, extra = act.call_args.args
    assert (stage, ticker) == ("scan_tier_funnel", "")
    assert extra["total"]["sent_to_analyst"] == 2 and extra["total"]["watchlist_in"] == 6
    assert extra["total"]["identity_ok"]


def test_a_failing_record_never_blocks_the_scan():
    queue, act, record, events = _daily(record_effect=RuntimeError("boom"))
    queue.assert_called_once()
    act.assert_not_called()


@pytest.mark.switches(SCAN_TIER_FUNNEL_RECORD_ENABLED=False)
def test_daily_scan_with_the_switch_off_writes_no_row():
    queue, act, _, events = _daily()
    assert events == ["enqueue"]
    act.assert_not_called()


# ── batch_verify and the registry ────────────────────────────────────────────

import importlib.util   # noqa: E402
import os   # noqa: E402


def _script(name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_scan_funnel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bv = _script("batch_verify")
src = _script("shadow_registry_check")

NOW = datetime(2026, 11, 23, 16, 0, tzinfo=timezone.utc)      # 11:00 ET Monday
SINCE = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)    # 05:00 ET


def _row(extra, rid=1):
    return {"id": rid, "stage": "scan_tier_funnel", "ticker": "", "extra": extra}


def _good_extra():
    out = _run_scan()
    hm = [c for c in out["candidates"] if c["priority"] in ("HIGH", "MEDIUM")]
    with patch("db.TRADING_ENV", "live"):
        return scan_funnel.build_row(out, hm, {"regime": "Neutral"}, 5, True, task_id="t",
                                     now=datetime(2026, 11, 23, 14, 40, tzinfo=timezone.utc))


def test_check_ok_on_a_good_row():
    chk = bv.check_scan_tier_funnel([_row(_good_extra())], "live", SINCE, NOW)
    assert chk["status"] == "OK", chk


def test_check_fails_on_a_broken_identity():
    extra = _good_extra()
    extra["by_tier"]["Core"]["sent_to_analyst"] += 1
    chk = bv.check_scan_tier_funnel([_row(extra)], "live", SINCE, NOW)
    assert chk["status"] == "FAIL" and "sum identity broken" in chk["why"] and "Core" in chk["why"]


def test_check_fails_on_a_missing_key_or_stage_count():
    extra = _good_extra()
    del extra["enqueue"]
    assert bv.check_scan_tier_funnel([_row(extra)], "live", SINCE, NOW)["status"] == "FAIL"
    extra = _good_extra()
    del extra["total"]["over_run_cap"]
    chk = bv.check_scan_tier_funnel([_row(extra)], "live", SINCE, NOW)
    assert chk["status"] == "FAIL" and "a stage count missing" in chk["why"]


def test_check_warns_on_unclassified_and_a_watchlist_error():
    extra = _good_extra()
    for t in (extra["by_tier"]["Core"], extra["total"]):
        t["unclassified"] += 1
        t["watchlist_in"] += 1
    extra["watchlist_error"] = "Watchlist load failed: x"
    chk = bv.check_scan_tier_funnel([_row(extra)], "live", SINCE, NOW)
    assert chk["status"] == "WARN" and "unclassified" in chk["why"] and "watchlist error" in chk["why"]


def test_check_due_scan_without_a_row_warns_and_none_due_is_ok():
    assert bv.check_scan_tier_funnel([], "live", SINCE, NOW)["status"] == "WARN"
    early = datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc)   # 09:00 ET
    assert bv.check_scan_tier_funnel([], "live", SINCE, early)["status"] == "OK"


def test_check_off_rows_fail_and_no_rows_ok():
    assert bv.check_scan_tier_funnel([], "live", SINCE, NOW, enabled=False)["status"] == "OK"
    chk = bv.check_scan_tier_funnel([_row(_good_extra())], "live", SINCE, NOW, enabled=False)
    assert chk["status"] == "FAIL"


def test_check_ignores_other_envs():
    extra = {**_good_extra(), "env": "paper"}
    chk = bv.check_scan_tier_funnel([_row(extra)], "live", SINCE, NOW)
    assert chk["rows"] == 0


def test_registry_stage():
    (st,) = [s for s in src.STAGES if s["stage"] == "scan_tier_funnel"]
    assert st["switch"] == "SCAN_TIER_FUNNEL_RECORD_ENABLED" and st["expect_rows"] == "always"
    assert st["sources"] == (("activity_feed", "stage = 'scan_tier_funnel'"),)
    assert getattr(config, st["switch"]) in (True, False)
