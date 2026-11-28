"""scripts/shadow_outcomes.py on fixture rows: each section's counts, the
forward tables from forward_readout's windows, and that it only reads."""
import importlib.util
import inspect
import os
from datetime import date, datetime, timedelta, timezone

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "shadow_outcomes.py")
_spec = importlib.util.spec_from_file_location("shadow_outcomes", _PATH)
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)

SINCE = date(2026, 10, 1)
DATES = [date(2026, 10, 1) + timedelta(days=i) for i in range(60)
         if (date(2026, 10, 1) + timedelta(days=i)).weekday() < 5]


def _bars():
    """SPY flat; AMD up 1% a session; NVDA down 1% a session."""
    return {"SPY": [(d, 100.0) for d in DATES],
            "AMD": [(d, 100.0 * 1.01 ** i) for i, d in enumerate(DATES)],
            "NVDA": [(d, 100.0 * 0.99 ** i) for i, d in enumerate(DATES)]}


def _ts(d, hour=15):
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)   # 10/11 ET, before the close


def _cand(**fires):
    from agents.held_exit import CANDIDATES
    return {"candidates": {c: {"would_exit": fires.get(c)} for c in CANDIDATES}}


def _data(**over):
    d = {"trim": [], "events": [], "approvals": [], "routing": [], "loser": [], "guard": [],
         "scans": [], "dips": [], "bars": _bars()}
    d.update(over)
    return d


def _build(**over):
    return so.build(_data(**over), SINCE, "live", "abc1234",
                    now=datetime(2026, 12, 1, 22, tzinfo=timezone.utc))


# ---------------------------------------------------------------- held exit

def test_held_exit_fires_per_rule_and_forward_split():
    trim = [("AMD", _ts(DATES[2]), _cand(below_both_sma=False, time_stop=True)),
            ("NVDA", _ts(DATES[3]), _cand(below_both_sma=True, time_stop=False)),
            ("NVDA", _ts(DATES[4]), None)]                                  # no record
    text = so.section_held_exit(trim, _bars(), SINCE, "live", "x")
    assert "TRIM_EXIT_RULE rows: 3; without the record: 1." in text
    assert "| below_both_sma | 1 | 1 | 0 | 50% |" in text
    assert "| time_stop | 1 | 1 | 0 | 50% |" in text
    assert "| atr_drawdown | 0 | 0 | 2 | - |" in text                      # None = unknown, not a fire
    assert "| below_both_sma: fire | 5 | 1 | 0 | 0 | 1 |" in text              # NVDA, falling
    assert "| time_stop: fire | 5 | 1 | 0 | 0 | 1 |" in text                   # AMD, rising
    assert so.DEFINITION_NOTE in text


# ---------------------------------------------------------------- story dedup

def test_story_dedup_split_per_trigger_source():
    def ev(src, rec):
        return ("AMD", _ts(DATES[1]), {"path": "event", "source": src, **({"story_dedup": rec} if rec else {})})
    rows = [ev("finnhub", {"read": "ok", "would_skip_duplicate": True, "would_skip_stale": False}),
            ev("finnhub", {"read": "ok", "would_skip_duplicate": False, "would_skip_stale": True}),
            ev("finnhub", {"read": "ok", "would_skip_duplicate": False, "would_skip_stale": False}),
            ev("edgar", {"read": "error: OperationalError"}),
            ev("edgar", None)]
    text = so.section_story_dedup(rows)
    assert "| finnhub | 3 | 0 | 0 | 1 | 1 | 2 (67%) | 1 |" in text
    assert "| edgar | 2 | 1 | 1 | 0 | 0 | 0 (0%) | 0 |" in text
    assert "| all | 5 | 1 | 1 | 1 | 1 | 2 (40%) | 1 |" in text


# ---------------------------------------------------------------- book heat

def _appr(i, action="OPEN", ticker="AMD", d=None, is_test=False, **risk):
    return {"id": i, "ticker": ticker, "action": action, "ts": _ts(d or DATES[2]), "is_test": is_test,
            "risk_json": risk}


def test_book_heat_spread_as_percent_of_portfolio():
    rows = [_appr(1, book_heat={"read": "ok", "sum_open_risk_pct": 0.01, "sum_open_risk_after_pct": 0.02,
                                "max_single_risk_pct": 0.005, "order_risk_pct": 0.01, "n_positions": 3,
                                "n_no_stop": 0}),
            _appr(2, action="ADD", book_heat={"read": "ok", "sum_open_risk_pct": 0.03,
                                              "sum_open_risk_after_pct": 0.05, "max_single_risk_pct": 0.02,
                                              "order_risk_pct": 0.02, "n_positions": 5, "n_no_stop": 1}),
            _appr(3, book_heat={"read": "error: OperationalError"}),
            _appr(4),                                                    # no record
            _appr(5, action="SELL", book_heat={"read": "ok", "sum_open_risk_pct": 0.9}),
            _appr(6, is_test=True, book_heat={"read": "ok", "sum_open_risk_pct": 0.9})]
    text = so.section_book_heat(rows)
    assert "Read states: ok 2, error: OperationalError 1, missing 1." in text
    assert "| open risk before the order, % of portfolio | 2 | 1.00 | 2.00 | 3.00 |" in text
    assert "| open risk after the order, % of portfolio | 2 | 2.00 | 3.50 | 5.00 |" in text
    assert "| positions held | 2 | 3 | 4 | 5 |" in text
    assert "90.00" not in text                                           # SELL and test rows ignored


# ---------------------------------------------------------------- buy blocks

def test_recheck_and_add_loser_counts_and_forward_by_would_block():
    approvals = [_appr(1, ticker="AMD", recheck_contradiction={"would_block": True}),
                 _appr(2, ticker="NVDA", recheck_contradiction={"would_block": False}),
                 _appr(3, action="ADD", ticker="NVDA", d=DATES[3],
                       recheck_contradiction={"would_block": None},
                       add_loser_gate={"would_block": True}),
                 _appr(4, action="ADD", ticker="AMD", d=DATES[4])]           # no records
    routing = [("AMD", _ts(DATES[5]), {"would_block": True})]
    loser = [("AMD", _ts(DATES[6]), {"add_loser_gate": {"would_block": False}})]
    text = so.section_buy_blocks(approvals, routing, loser, _bars(), SINCE, "live", "x")
    assert "| recheck_contradiction, OPEN/ADD approvals | 4 | 1 | 1 | 1 | 1 |" in text
    assert "| recheck_contradiction, routing rows that ended before sizing | 1 | 1 | 0 | 0 | 0 |" in text
    assert "| add_loser_gate, ADD approvals (shadow) | 2 | 1 | 0 | 0 | 1 |" in text
    assert "| add_loser_gate_shadow, deferred retries | 1 | 0 | 1 | 0 | 0 |" in text
    assert "| recheck would_block true | 5 | 1 | 0 | 0 | 1 |" in text
    assert "| add_loser would_block true | 5 | 1 | 0 | 0 | 1 |" in text
    assert so.DEFINITION_NOTE in text


# ---------------------------------------------------------------- stop guard

def test_stop_guard_outcome_by_path():
    rows = [("AMD", _ts(DATES[1]), {"mode": "shadow", "path": "add_merge",
                                    "outcome": "would_placed_below_market"}),
            ("AMD", _ts(DATES[1]), {"mode": "shadow", "path": "add_merge",
                                    "outcome": "would_no_level_below_market"}),
            ("NVDA", _ts(DATES[2]), {"mode": "shadow", "path": "trim_replace",
                                     "outcome": "would_placed_below_market"})]
    text = so.section_stop_guard(rows)
    assert "Rows: 3; mode: shadow 3." in text
    assert "| path | rows | would_no_level_below_market | would_placed_below_market |" in text
    assert "| add_merge | 2 | 1 | 1 |" in text and "| trim_replace | 1 | 0 | 1 |" in text
    assert "No rows." in so.section_stop_guard([])


# ---------------------------------------------------------------- setup gates

def test_setup_gate_qualifiers_and_forward():
    def scan(t, d, **q):
        return (t, _ts(d), {g: {"qualifies": q.get(g, False)} for g in so.GATES})
    rows = [scan("AMD", DATES[1], Breakout=True), scan("AMD", DATES[2], Breakout=True, Catalyst=True),
            scan("NVDA", DATES[1]), scan("NVDA", DATES[2]), ("NVDA", _ts(DATES[3]), None)]
    text = so.section_setup_gates(rows, _bars(), SINCE, "live", "x")
    assert "Scanned rows with the record: 4." in text
    assert "| Breakout | 2 | 50% |" in text and "| Catalyst | 1 | 25% |" in text
    assert "| any | 2 | 50% |" in text
    assert "| qualifies (any gate) | 5 | 2 |" in text and "| no gate qualifies | 5 | 2 |" in text


# ---------------------------------------------------------------- drop watch

def test_dip_watch_split_by_would_review():
    rows = [("AMD", _ts(DATES[1]), True), ("NVDA", _ts(DATES[1]), False),
            ("NVDA", _ts(DATES[2]), False), ("AMD", _ts(DATES[3]), None)]
    text = so.section_dip_watch(rows, _bars(), SINCE, "live", "x")
    assert "Rows: 4; would_review true 1, false 2, unknown 1." in text
    assert "| DIP_WATCH_SHADOW would_review false | 5 | 2 |" in text
    assert "| DIP_WATCH_SHADOW would_review true | 5 | 1 |" in text


# ---------------------------------------------------------------- build

def test_build_has_every_section_and_isolates_a_failure():
    text = _build(dips=[("AMD", _ts(DATES[1]), True)])
    for title in ("Held-exit candidates per rule", "Story de-duplication per trigger source",
                  "Book heat at each buy", "Recheck contradiction and add-to-loser: would_block",
                  "Above-market stop guard by path", "Setup-gate qualifiers", "Drop watch by would_review"):
        assert f"\n## {title}\n" in text
    assert "section failed" not in text and "$" not in text
    bad = _build(events=[("AMD", _ts(DATES[1]), {"path": "event"})], guard=None)   # guard unreadable
    assert "section failed: TypeError" in bad
    assert "## Drop watch by would_review" in bad and "| all | 1 |" in bad


def test_empty_window_renders():
    text = _build()
    assert "TRIM_EXIT_RULE rows: 0" in text and "| (no rows) |" in text and "No rows." in text


def test_fetch_only_reads():
    src = inspect.getsource(so._fetch)
    assert "SET TRANSACTION READ ONLY" in src and "statement_timeout" in src and "rollback()" in src
    whole = open(_PATH).read()
    for word in ("INSERT", "UPDATE ", "DELETE", "TRUNCATE", "ALTER", "DROP", ".commit(", "set_state(",
                 "log_activity", "_send(", "send_task(", "apply_async(", ".delay("):
        assert word not in whole, word


def test_activity_rows_of_another_env_are_dropped():
    assert so._row_env_ok({"env": "live"}, "live") and so._row_env_ok({}, "live")
    assert not so._row_env_ok({"env": "paper"}, "live")
