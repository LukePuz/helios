"""Post-deploy check of the new records (engineering diagnostic, not a study
analysis). For each record added in this batch it reports, from rows written
since --since (default: today 05:00 ET): rows written, the read-state
distribution, the error reasons, and one OK / WARN / FAIL line.

Rules (the same for every record unless its line says more):
  FAIL  a record whose trigger happened but the field is missing (a signals
        or approval row of the kind that must carry it, without it), or more
        than 20% of a read's states are errors ("error..."), or a stated
        extra condition below.
  WARN  nothing triggered yet (no rows of that kind since --since).
  OK    otherwise.

Records:
  entry_gate          ENTRY_GATE_BLOCKED signals rows, decision.entry_gate.
                      Each row needs the stopout key; rows whose path is not
                      deferred_retry also need trend and trend.market (the
                      deferred retry applies the stop-out half only).
                      Extra FAIL: a trend-blocked row whose market is not a
                      dict (the market is read whenever the stock would block).
  story_dedup         event verdicts (provenance.trigger.path 'event', analyst
                      kinds only): provenance.trigger.story_dedup.
  rr_vs_stop, vol_target_shadow, stop_vol, book_heat, recheck_contradiction
                      risk_json on new OPEN/ADD approvals (is_test excluded).
                      Extra FAIL: vol_target_shadow.weights_at not
                      'final_order'.
  source_mix          event verdicts with a model reply (decision ? model_output):
                      decision.source_mix with every SOURCE_MIX_REQUIRED key.
                      Extra FAIL: external + brief_section + other sources not
                      summing to source_count. A record that came back
                      {error} is an error read. Presence and shape only.
  news_timeline       activity_feed rows (stage news_timeline, written by a side
                      thread after the verdict) for every analyst verdict on
                      config.NEWS_TIMELINE_PATHS older than 10 min, matched on
                      extra.signal_id; moded on NEWS_TIMELINE_RECORD_ENABLED.
                      Extra FAIL: a row missing a NEWS_TIMELINE_REQUIRED key
                      (NEWS_TIMELINE_BASE on a read-error row).
  plan_ledger_record  the plan observation (provenance.plan) of full analyst
                      verdicts: every NEW_PLAN, SEEDED or REVISE carries
                      horizon_basis; with PLAN_RESEED_RECORD_ENABLED on (as
                      this clone loads it), every NEW_PLAN or SEEDED also
                      carries reseeded_after_invalidate and a reseed dict
                      (reseed.status "error:..." is an error read; off: not
                      checked, the rows still carrying it are noted).
                      reseeded_after_invalidate and horizon_basis are tallied,
                      as are the stopped_out_since_plan records (a note).
  own_ticker_rejected SCREENED_OUT signals rows with a provenance
                      (agents/own_ticker.py, written in crew.py):
                      provenance.own_ticker_rejected and own_ticker_patterns_sha.
                      own_ticker_error is an error read. Extra FAIL: a
                      patterns sha other than the one this clone loads (the
                      deploy wrote under other patterns). The flag is tallied.
  same_day_consensus  risk_json.same_day_consensus on new OPEN/ADD approvals
                      (is_test excluded), presence and shape only, moded on
                      config.SAME_DAY_CONSENSUS_RECORD_ENABLED as this clone
                      loads it (off: not expected, OK). FAIL: an approval
                      without it, a record missing a SAME_DAY_REQUIRED key, or
                      more than 20% "error" reads. "skipped_budget" reads are
                      tallied, not errors. ENTRY_GATE_BLOCKED rows carrying it
                      are tallied (deferred retries never do).
  add_loser_gate      risk_json.add_loser_gate on new ADD approvals, only
                      while ADD_LOSER_GATE_MODE is shadow (enforce records it
                      on ENTRY_GATE_BLOCKED rows instead; off records nothing:
                      the line then says so and is WARN).
  held_trend, held_exit_candidates
                      TRIM_EXIT_RULE decision keys; reads.held_trend and
                      reads.held_exit.
  add_lot_r, broker_sell_seen, entry_gate_split_fail_open,
  price_cache_rescale, price_cache_scale_gap, price_cache_rescale_failed,
  add_stop_qty, add_stale_record, stop_above_market_guard,
  add_loser_gate_shadow, plan_compiled_hit, pt_stop_qty, late_fill_expect,
  late_fill_held_before
                      activity rows; no independent trigger is in the DB, so
                      FAIL means a row missing a required field or >20% error
                      reads. A price_cache_rescale_failed row counts as an
                      error read of price_cache_rescale. The price-cache rows
                      are nightly (after 16:45 ET on a weekday) and appear
                      only when a cached ticker's price scale changed.
                      broker_sell_seen also lists broker exits since --since
                      with no seen row within 30 min of the fill (a note, not
                      a FAIL: the watcher skips a sell the reconcile already
                      booked under its order id).
                      Rows of a moded record (stop_above_market_guard,
                      late_fill_expect, late_fill_held_before) carry mode:
                      extra FAIL when a row's mode is not the config mode
                      this clone loads (the deploy didn't load the intended
                      mode). add_loser_gate_shadow is written only in shadow.
                      pt_stop_qty and late_fill_expect land with the
                      profit-target / late-BUY stop-count change, and
                      late_fill_held_before with the late-fill held-count
                      change; before those are deployed they read WARN.
  entry_basis         activity rows (execution._record_entry_basis), one per ADD
                      that writes the broker's average cost as the stop
                      record's entry. Conditional: no row is WARN, never
                      FAIL; the line says how many ADD approvals the window
                      had (an ADD writes a row only when it merges into an
                      existing stop record). FAIL: a row missing a required
                      key, or more than 20% error reads.
  entry_ref_age       activity rows (execution._record_entry_ref_age), one per
                      OPEN/ADD BUY that reached the submit point: the sizing
                      and pre-submit bar times and ages, the quote taken just before
                      the order went. Activity rule (WARN with no rows; FAIL on a
                      missing key or more than 20% ref errors).
  order_quote         activity rows (ibkr._write_order_quote), one per market
                      order (BUY and SELL): the quote's market-data type and
                      age. Activity rule (WARN with no rows; FAIL on a missing
                      key). Its md_type_label tally is the entitlement signal:
                      the order path asks reqMarketDataType(3), live where the
                      account is entitled and delayed otherwise (IBKR Warning
                      10167). scripts/market_data_check.py has the per-session
                      readout.
  stop_slippage_partial activity rows (tasks._book_partial_stop_fills), one
                      per booked partial exit. Conditional: no row is WARN;
                      FAIL only on a missing key.
  book_gap_risk       activity rows (tasks.book_gap_risk_record), one per
                      session after the close. Presence and shape only, no
                      threshold: WARN with no rows; FAIL on a missing key or
                      more than 20% error reads.
  book_deployment     activity rows (tasks.book_deployment_record), one per
                      session after the close, beside book_gap_risk.
                      Presence and shape only, no threshold: WARN with no
                      rows; FAIL on a missing key, more than 20% error reads,
                      or a dollar-amount key (cash, total_value, ...) in a
                      row, since the row is percentages only.
  verdict_vote_shadow activity rows (agents/verdict_vote.py), moded on
                      config.VERDICT_VOTE_SHADOW_ENABLED as this clone loads
                      it. Off: OK with no rows, FAIL on any row (the switch
                      the deploy loaded is not the one rows were written
                      under). On: the activity rule (WARN with no rows;
                      FAIL on a missing key or more than 20% "error" status).
  risk_subcheck_fail  activity rows written when a theme or moonshot-total
                      concentration sub-check raised (or the watchlist read
                      empty) in live sizing. Inverted: OK with no rows, WARN
                      on any row (a sub-check is failing); never FAIL. The
                      line shows check, error, mode and applied.
  hwm_guard           activity rows (RiskManager._hwm_guard): the per-buy
                      drawdown check found its baseline unusable. Inverted:
                      OK with no rows, WARN on any row (the line lists the
                      reasons); FAIL on a row missing reason, env, mode,
                      action or stored.
  fundamentals_changes
                      activity rows (tasks._fundamentals_change_record), one
                      per fundamentals refresh (beat 17:00 ET, Mon-Fri).
                      FAIL: a row missing a required key. WARN: no row when
                      a session day's refresh fell in the window (started at
                      or after --since, 18:00 ET by now); no row and none due
                      is OK. The line tallies flagged (a source freeze).
  dcf_branch          activity rows (agents/valuation_audit.log_dcf_branch), one
                      per trade ladder refresh (beat 17:15 ET, Mon-Fri). FAIL:
                      a row missing a required key, a per-name entry missing
                      one, or more than 20% of names with an error. WARN: no
                      row when a session day's refresh fell in the window
                      (started at or after --since, 18:15 ET by now); none
                      due is OK. The line tallies dcf_branch per name and
                      notes names whose recomputed DCF differs from the
                      ladder's (the record then compares different inputs).
  source_yield        activity rows (tasks.source_yield, 18:20 ET Mon-Fri):
                      every row computed since --since carries feeds and
                      feed_drops (None allowed: the breakdown was unread).
                      FAIL: a row without either key. WARN: no row, or a
                      row whose feed_drops is non-empty (a feed below half
                      its baseline; the line names them). A replayed row
                      computed before --since is left out and noted.
  feed_health         activity rows (tasks.feed_health, 18:40 ET Mon-Fri), one
                      per run on a session day. FAIL: a row missing a required
                      key. WARN: no row when a session day's run fell in the
                      window (started at or after --since, 18:50 ET by now);
                      no row and none due is OK. A due day without its own row
                      is noted. The line tallies flagged sources. With
                      FEED_HEALTH_ENABLED off: OK with no rows, FAIL on any row
                      (the switch the deploy loaded is not the one rows were
                      written under).
  earnings_event_record
                      activity rows (ingestion/earnings_events.py), a pre row
                      when an earnings date is first seen and a post row when
                      its actual appears, after the 17:30 ET fetch. Conditional:
                      no row is WARN, never FAIL (most days no event is new).
                      FAIL: a row missing a required key of its kind, or a
                      None value without its reason in missing. The line
                      tallies kind, revisions, estimated dates and matched_by.
  read_through        activity rows (agents/crew.py:_read_through_record), one
                      when an analysed event's ticker has a related symbol
                      (the brief's mapping) the book holds, once per source
                      item and held ticker. Conditional and WARN-only: no row,
                      a row missing a required key, a dedupe_read error or a
                      related section past the brief's character cut is WARN;
                      never FAIL. The line tallies event_ticker and dedupe_read.
  earnings_claim      analyst verdicts' decision.catalyst_check
                      (earnings_claim.py), on every full verdict row
                      (one with decision.model_output). FAIL: a row missing
                      earnings_claim_unmatched, matches_sector_cluster_date,
                      calendar_source_used or earnings_claim, or more than 20%
                      error reads of its dates or sector store; no verdict is
                      WARN. No threshold on the flags: the line tallies them.
  catalyst_kind       the same verdict rows' decision.catalyst_check
                      (earnings_claim.py). FAIL: a row missing catalyst_kind,
                      catalyst_days_ahead, catalyst_matched_source,
                      catalyst_matched_event_type or catalyst_kind_detail, a
                      catalyst_kind outside the six kinds and None, or more
                      than 20% error reads of the catalyst_events store; no
                      verdict is WARN. The line tallies the kinds and the
                      matched sources.
  target_anchor       full analyst verdicts (as earnings_claim reads them):
                      decision.target_anchor (agents/target_anchor.py), moded
                      on config.TARGET_ANCHOR_RECORD_ENABLED as this clone
                      loads it (off: not checked, OK). On: required keys on
                      every verdict; FAIL on a verdict without the record or
                      more than 20% timing_read errors; WARN with no verdict.
                      The line tallies timing_read, a street figure in the
                      brief, fundamentals_predate_last_print, fv_confidence,
                      target_equals_fv and target_equals_street.
  watchlist_parse     watchlist.md ticker rows that knowledge_loader.load_watchlist
                      does not parse (its own "row not parsed" warning, caught
                      from a fresh load in this process): such a ticker is out
                      of the universe. WARN-only: OK with none, WARN listing
                      the tickers (or the read error); never FAIL. The rows
                      themselves are Luke's (universe / data QA).
  entry_funnel        scripts/entry_funnel.py's --check over the last
                      CHECK_SESSIONS closed sessions: FAIL when it fires (0
                      passed buys on at least CHECK_MIN_VERDICTS OPEN/ADD
                      verdicts) or its read raised; WARN under that many
                      verdicts; OK with passed buys. Its own READ ONLY
                      transaction.
  watch_broker_sells  Celery log lines (logs/*.log) for the task, per hour.
                      FAIL: more than 20% of finished runs raised, or a fully
                      elapsed weekday market hour (09:30-16:00 ET) with no
                      succeeded run. WARN: no finished run yet.

activity_feed has no env column: rows are filtered on extra_json.env only
where the row carries one (broker_sell_seen). Counts and percentages only:
no prices, quantities or account values.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/batch_verify.py [--since "2026-11-24 05:00"] [--out FILE]

Read-only: one DB transaction opened READ ONLY, then rolled back (and the
entry funnel's own, the same way); the log files are only read. Exit status 1 when any line is FAIL.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
ERROR_FAIL_SHARE = 0.20                    # FAIL above this share of error reads
BUY_ACTIONS = ("OPEN", "ADD")
SEEN_MATCH_TOLERANCE = timedelta(minutes=30)
ACTIVITY_STAGES = ("add_lot_r", "broker_sell_seen", "entry_gate_split_fail_open",
                   "price_cache_rescale", "price_cache_scale_gap", "price_cache_rescale_failed",
                   "add_stop_qty", "add_stale_record", "stop_above_market_guard",
                   "add_loser_gate_shadow", "plan_compiled_hit", "pt_stop_qty",
                   "late_fill_expect", "late_fill_held_before", "entry_basis",
                   "entry_ref_age", "stop_slippage_partial", "book_gap_risk", "book_deployment",
                   "order_quote")
# book_deployment rows are percentages only: any of these keys is a FAIL.
BOOK_DEPLOYMENT_DOLLAR_KEYS = ("cash", "total_value", "portfolio_value", "net_liquidation",
                               "nav", "market_value", "equity", "buying_power")
# The verdict-vote shadow's stage, checked on its own (moded on a bool).
VOTE_STAGE = "verdict_vote_shadow"
VOTE_REQUIRED = ("signal_id", "status", "draws", "actions", "majority_action",
                 "model_pass_majority", "prompt_sha", "options_match", "busy_tries")
# Activity stages that should not appear at all: any row is a WARN
# (check_risk_subcheck_fail), and none is OK.
ALARM_STAGES = ("risk_subcheck_fail", "hwm_guard")
HWM_GUARD_REQUIRED = ("reason", "env", "mode", "action", "stored")
FUNDAMENTALS_STAGE = "fundamentals_changes"
FUNDAMENTALS_REQUIRED = ("tickers", "changed", "no_prior", "judged", "unchanged_streak", "flagged",
                         "alerted", "env", "since", "params")
FUNDAMENTALS_START, FUNDAMENTALS_DUE = time(17, 0), time(18, 0)   # beat 17:00 ET, Mon-Fri
DCF_BRANCH_STAGE = "dcf_branch"
DCF_BRANCH_REQUIRED = ("env", "n_names", "n_dcf", "n_hypergrowth", "n_standard", "n_none",
                       "dcf_only_intrinsic", "n_dcf_mismatch", "n_errors", "step_ratio_median",
                       "step_ratio_at_threshold_median", "params", "names")
DCF_BRANCH_NAME_REQUIRED = ("ticker", "dcf_branch", "growth_input", "dcf_value_rel",
                            "dcf_value_rel_other_branch", "step_ratio", "step_ratio_at_threshold",
                            "intrinsic_methods_used", "n_intrinsic_methods", "blend_rel",
                            "blend_rel_other_branch", "dcf_matches_ladder")
DCF_BRANCH_START, DCF_BRANCH_DUE = time(17, 15), time(18, 15)   # beat 17:15 ET, Mon-Fri
FEED_HEALTH_STAGE = "feed_health"
FEED_HEALTH_REQUIRED = ("day", "env", "computed_at", "flagged", "alerted", "excluded", "sources",
                        "params")
FEED_HEALTH_START, FEED_HEALTH_DUE = time(18, 40), time(18, 50)   # beat 18:40 ET, Mon-Fri
FEED_STAGE = "source_yield"
FEED_REQUIRED = ("feeds", "feed_drops")
EARNINGS_STAGE = "earnings_event_record"
EARNINGS_REQUIRED = {
    "pre": ("kind", "env", "ticker", "source", "shadow_key", "announce_date",
            "announce_date_is_estimate", "window_dates", "time_of_day", "consensus_eps",
            "consensus_revenue", "estimate_quarter_end", "last_reported_quarter", "revision",
            "as_of", "missing"),
    "post": ("kind", "env", "ticker", "source", "shadow_key", "announce_date", "pre_shadow_key",
             "quarter", "matched_by", "eps_actual", "eps_estimate_source",
             "surprise_percent_source", "revenue_actual", "as_of", "missing")}
# Value fields whose None must carry a reason in the row's missing.
EARNINGS_REASONED = {
    "pre": ("time_of_day", "consensus_eps", "consensus_revenue", "window_dates",
            "estimate_quarter_end", "last_reported_quarter"),
    "post": ("eps_estimate_source", "surprise_percent_source", "revenue_actual")}
TARGET_ANCHOR_REQUIRED = ("bull_target", "street_target", "fair_value", "target_vs_street_pct",
                          "target_vs_fair_value_pct", "fundamentals_updated_at",
                          "fundamentals_as_of", "last_print_date", "print_today",
                          "fundamentals_predate_last_print", "timing_read", "missing",
                          "fv_confidence", "target_equals_fv", "target_equals_street",
                          "target_equals_tol")
READ_THROUGH_STAGE = "read_through"
NEWS_TIMELINE_STAGE = "news_timeline"
NEWS_TIMELINE_BASE = ("shadow_key", "signal_id", "env", "path", "verdict_at", "prompt_at",
                      "window_days", "cap", "read", "tone_lexicon", "tone_sha", "read_ms", "lag_s")
NEWS_TIMELINE_REQUIRED = NEWS_TIMELINE_BASE + (
    "n_items_7d", "n_passed_7d", "n_rejected_7d", "n_no_screen_row_7d", "n_negative_7d",
    "n_negative_rejected_7d", "n_in_prompt_7d", "n_negative_in_prompt_7d",
    "n_items_before_prompt", "n_negative_before_prompt", "n_negative_rejected_before_prompt",
    "items")
PLAN_LEDGER_RESEED_RELATIONS = ("NEW_PLAN", "SEEDED")
PLAN_LEDGER_HORIZON_RELATIONS = ("NEW_PLAN", "SEEDED", "REVISE")
NEWS_TIMELINE_LAG = timedelta(minutes=10)   # a verdict younger than this may not have its row yet
EARNINGS_CLAIM_REQUIRED = ("earnings_claim_unmatched", "matches_sector_cluster_date",
                           "calendar_source_used", "earnings_claim")
CATALYST_KIND_REQUIRED = ("catalyst_kind", "catalyst_days_ahead", "catalyst_matched_source",
                          "catalyst_matched_event_type", "catalyst_kind_detail")
CATALYST_KIND_VALUES = ("already_released", "scheduled_today", "calendar_matched",
                        "unmatched_dated", "ongoing", "unparsed", None)
READ_THROUGH_REQUIRED = ("env", "event_ticker", "signal_id", "item", "related", "held_related",
                         "held_related_deduped", "held_as_of", "section_in_brief",
                         "section_within_char_cap", "dedupe_read")
SOURCE_MIX_REQUIRED = ("source_count", "external_source_count", "brief_section_sources",
                       "other_sources", "type_counts", "distinct_publishers", "publishers",
                       "n_external_no_publisher", "publisher_basis", "min_sources",
                       "gate_would_pass", "gate_would_pass_external_only", "definition")
SAME_DAY_REQUIRED = ("window", "read", "limit_hit", "current_signal_id", "n_earlier",
                     "n_buy_grade", "buy_grade_share", "n_buy_grade_unknown", "n_model_buy",
                     "median_conviction", "n_conviction", "median_udr", "n_udr",
                     "n_story_keys", "n_distinct_stories", "window_24h", "limit_hit_24h",
                     "n_earlier_24h", "n_buy_grade_24h", "buy_grade_share_24h",
                     "buy_grade_definition")
# Activity stages whose rows carry the mode they ran in, and the config name.
MODED_STAGES = {"stop_above_market_guard": "STOP_GUARD_MODE",
                "late_fill_expect": "LATE_FILL_EXPECT_MODE",
                "late_fill_held_before": "LATE_FILL_HELD_BEFORE_MODE"}
_PENDING_NOTE = {"pt_stop_qty": "lands with the profit-target / late-BUY stop-count change",
                 "late_fill_expect": "lands with the profit-target / late-BUY stop-count change",
                 "late_fill_held_before": "lands with the late-fill held-count change"}
MARKET_OPEN, MARKET_CLOSE = time(9, 30), time(16, 0)
WATCH_TASK = "tasks.watch_broker_sells"
_LOG_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[,.]\d+: [^\]]*\] "
                     r"(?:Task " + re.escape(WATCH_TASK) + r"\[[^\]]+\] (succeeded|raised|retry)"
                     r"|Discarding revoked task: " + re.escape(WATCH_TASK) + r"\[)")


# ---------------------------------------------------------------- helpers

def _is_error(read) -> bool:
    return isinstance(read, str) and read.strip().lower().startswith("error")


def _label(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _as_dict(v) -> dict | None:
    return v if isinstance(v, dict) else None


def _new(name: str, source: str, rule: str) -> dict:
    return {"name": name, "source": source, "rule": rule, "rows": 0, "triggered": 0,
            "missing": [], "reads": {}, "extra_fail": [], "notes": [], "status": None}


def _read(chk: dict, label: str, value) -> None:
    chk["reads"].setdefault(label, Counter())[_label(value)] += 1


def _finish(chk: dict) -> dict:
    """Status from the rule in the module docstring."""
    fail = list(chk["extra_fail"])
    if chk["missing"]:
        fail.append(f"{len(chk['missing'])} of {chk['triggered']} triggered rows lack the field")
    for label, c in chk["reads"].items():
        n, err = sum(c.values()), sum(v for k, v in c.items() if _is_error(k))
        if n and err / n > ERROR_FAIL_SHARE:
            fail.append(f"{label}: {err} of {n} reads are errors ({err / n:.0%})")
    if chk["triggered"] == 0:
        chk["status"], chk["why"] = "WARN", "none triggered yet"
    elif fail:
        chk["status"], chk["why"] = "FAIL", "; ".join(fail)
    else:
        chk["status"], chk["why"] = "OK", "all triggered rows carry it, error reads within 20%"
    return chk


# ---------------------------------------------------------------- checks

def check_entry_gate(rows) -> dict:
    """rows: dicts {id, ticker, decision} of ENTRY_GATE_BLOCKED signals."""
    chk = _new("entry_gate", "signals ENTRY_GATE_BLOCKED decision.entry_gate",
               "stopout on every row; trend and trend.market unless path deferred_retry")
    for r in rows:
        chk["rows"] += 1
        chk["triggered"] += 1
        gate = _as_dict((_as_dict(r.get("decision")) or {}).get("entry_gate"))
        if gate is None:
            chk["missing"].append(r.get("id"))
            continue
        path = gate.get("path")
        _read(chk, "path", path)
        need = ["stopout"] if path == "deferred_retry" else ["stopout", "trend"]
        trend = _as_dict(gate.get("trend"))
        if any(k not in gate for k in need) or (path != "deferred_retry"
                                                and (trend is None or "market" not in trend)):
            chk["missing"].append(r.get("id"))
        stop = _as_dict(gate.get("stopout"))
        if stop is not None:
            _read(chk, "stopout.read", stop.get("read"))
            _read(chk, "stopout.blocked", stop.get("blocked"))
        if trend is not None:
            _read(chk, "trend.read", trend.get("read"))
            _read(chk, "trend.blocked", trend.get("blocked"))
            market = trend.get("market")
            if isinstance(market, dict):
                _read(chk, "market.read", market.get("read"))
                _read(chk, "market.downtrend", market.get("downtrend"))
            elif trend.get("blocked") is True:
                chk["extra_fail"].append(f"row {r.get('id')}: trend-blocked with no market dict")
    return _finish(chk)


def check_story_dedup(rows) -> dict:
    """rows: dicts {id, ticker, kind, trigger} of event verdicts."""
    chk = _new("story_dedup", "event verdicts provenance.trigger.story_dedup",
               "on every event verdict")
    for r in rows:
        chk["rows"] += 1
        chk["triggered"] += 1
        trig = _as_dict(r.get("trigger")) or {}
        rec = _as_dict(trig.get("story_dedup"))
        if rec is None:
            chk["missing"].append(r.get("id"))
            _read(chk, "missing by kind", r.get("kind"))
            continue
        _read(chk, "read", rec.get("read"))
        _read(chk, "would_skip_duplicate", rec.get("would_skip_duplicate"))
        _read(chk, "would_skip_stale", rec.get("would_skip_stale"))
    return _finish(chk)


def check_source_mix(rows) -> dict:
    """rows: event verdict dicts as check_story_dedup takes them, with
    has_model_output and source_mix (decision.source_mix). Rows without
    has_model_output True (a stub verdict, or a caller that predates the
    column) are not triggers."""
    chk = _new("source_mix", "event verdicts decision.source_mix",
               "on every event verdict with a model reply, with keys " + ", ".join(SOURCE_MIX_REQUIRED))
    for r in rows or ():
        if r.get("has_model_output") is not True:
            continue
        chk["rows"] += 1
        chk["triggered"] += 1
        rec = _as_dict(r.get("source_mix"))
        if rec is not None and "error" in rec:
            _read(chk, "record", f"error: {rec.get('error')}")
            continue
        if rec is None or any(k not in rec for k in SOURCE_MIX_REQUIRED):
            chk["missing"].append(r.get("id"))
            continue
        _read(chk, "record", "ok")
        _read(chk, "gate_would_pass_external_only", rec.get("gate_would_pass_external_only"))
        parts = [rec.get(k) for k in ("external_source_count", "brief_section_sources",
                                      "other_sources", "source_count")]
        if not all(isinstance(v, int) for v in parts) or sum(parts[:3]) != parts[3]:
            chk["extra_fail"].append(f"verdict {r.get('id')}: the source split does not sum "
                                     "to source_count")
    return _finish(chk)


def check_add_loser(rows, mode: str = "shadow") -> dict:
    """rows: as check_approvals. risk_json.add_loser_gate on every new ADD
    approval, which crew writes only in shadow mode."""
    chk = _new("add_loser_gate", "pending_approvals risk_json.add_loser_gate (ADD)",
               f"on every ADD approval while ADD_LOSER_GATE_MODE is shadow (now {mode})")
    if mode != "shadow":
        chk["notes"].append(f"mode {mode}: approvals carry no add_loser_gate; nothing checked")
        return _finish(chk)
    for r in rows:
        if r.get("action") != "ADD" or r.get("is_test"):
            continue
        chk["rows"] += 1
        chk["triggered"] += 1
        rec = _as_dict((_as_dict(r.get("risk_json")) or {}).get("add_loser_gate"))
        if rec is None:
            chk["missing"].append(r.get("id"))
            continue
        _read(chk, "avg_cost_read", rec.get("avg_cost_read"))
        _read(chk, "trigger_new_fact_source", rec.get("trigger_new_fact_source"))
        _read(chk, "would_block", rec.get("would_block"))
        if rec.get("blocked") is True:
            chk["extra_fail"].append(f"approval {r.get('id')}: blocked in shadow")
    return _finish(chk)


def check_approvals(rows) -> list[dict]:
    """rows: dicts {id, ticker, action, is_test, risk_json} of new approvals.
    Returns the rr_vs_stop, vol_target_shadow, stop_vol, book_heat and
    recheck_contradiction checks."""
    specs = (("rr_vs_stop", "reason"), ("vol_target_shadow", "read"),
             ("stop_vol", "reason"), ("book_heat", "read"), ("recheck_contradiction", "read"))
    out = {k: _new(k, f"pending_approvals risk_json.{k} (OPEN/ADD)", "on every OPEN/ADD approval")
           for k, _ in specs}
    for r in rows:
        if r.get("action") not in BUY_ACTIONS or r.get("is_test"):
            continue
        risk = _as_dict(r.get("risk_json")) or {}
        for key, field in specs:
            chk = out[key]
            chk["rows"] += 1
            chk["triggered"] += 1
            rec = _as_dict(risk.get(key))
            if rec is None:
                chk["missing"].append(r.get("id"))
                continue
            v = rec.get(field)
            _read(chk, field, "ok" if v is None and field == "reason" else v)
            if key == "vol_target_shadow":
                _read(chk, "weights_at", rec.get("weights_at"))
                if rec.get("weights_at") != "final_order":
                    chk["extra_fail"].append(f"approval {r.get('id')}: weights_at "
                                             f"{_label(rec.get('weights_at'))}")
    out["vol_target_shadow"]["rule"] += "; weights_at final_order"
    return [_finish(out[k]) for k, _ in specs]


def check_same_day_consensus(approvals, gate_rows, enabled: bool) -> dict:
    """approvals: as check_approvals takes them; gate_rows: ENTRY_GATE_BLOCKED
    rows ({id, ticker, decision}). enabled: SAME_DAY_CONSENSUS_RECORD_ENABLED
    as this clone loads it. Presence and shape only: nothing here judges the
    values."""
    chk = _new("same_day_consensus", "pending_approvals risk_json.same_day_consensus (OPEN/ADD)",
               ("on every OPEN/ADD approval, with keys " + ", ".join(SAME_DAY_REQUIRED))
               if enabled else "SAME_DAY_CONSENSUS_RECORD_ENABLED False: not expected")
    for r in approvals or ():
        if r.get("action") not in BUY_ACTIONS or r.get("is_test"):
            continue
        rec = _as_dict((_as_dict(r.get("risk_json")) or {}).get("same_day_consensus"))
        if not enabled:
            continue
        chk["rows"] += 1
        chk["triggered"] += 1
        if rec is None or any(k not in rec for k in SAME_DAY_REQUIRED):
            chk["missing"].append(r.get("id"))
            continue
        _read(chk, "read", rec.get("read"))
        _read(chk, "limit_hit", rec.get("limit_hit"))
        _read(chk, "limit_hit_24h", rec.get("limit_hit_24h"))
        for w in ("window", "window_24h"):
            if not isinstance(rec.get(w), dict):
                chk["extra_fail"].append(f"approval {r.get('id')}: {w} is not a dict")
    carried = sum(1 for g in gate_rows or ()
                  if isinstance((_as_dict(g.get("decision")) or {}).get("same_day_consensus"), dict))
    chk["notes"].append(f"ENTRY_GATE_BLOCKED rows carrying it: {carried}")
    if not enabled:
        chk["status"], chk["why"] = "OK", "record off in this clone; not expected"
        return chk
    return _finish(chk)


def same_day_consensus_enabled() -> bool:
    import config
    return bool(getattr(config, "SAME_DAY_CONSENSUS_RECORD_ENABLED", False))


def check_trim_exit(rows) -> list[dict]:
    """rows: dicts {id, ticker, decision} of TRIM_EXIT_RULE signals."""
    out = []
    for key, read_key, sub in (("held_trend", "held_trend", "read"),
                               ("held_exit_candidates", "held_exit", "read")):
        chk = _new(key, f"signals TRIM_EXIT_RULE decision.{key}",
                   f"key on every row; read from decision.reads.{read_key}")
        for r in rows:
            chk["rows"] += 1
            chk["triggered"] += 1
            dec = _as_dict(r.get("decision")) or {}
            if key not in dec:
                chk["missing"].append(r.get("id"))
                continue
            reads = _as_dict(dec.get("reads")) or {}
            _read(chk, f"reads.{read_key}", reads.get(read_key))
        out.append(_finish(chk))
    return out


_ACTIVITY_SPECS = {
    # stage: (required extra keys, [(label, getter)])
    "add_lot_r": (("read", "path", "max_lot_r", "lots"),
                  [("read", lambda e: e.get("read")), ("path", lambda e: e.get("path"))]),
    "broker_sell_seen": (("first_exec_at", "seen_lag_s", "held_read", "exec_time_source"),
                         [("held_read", lambda e: e.get("held_read")),
                          ("exec_time_source", lambda e: e.get("exec_time_source"))]),
    "entry_gate_split_fail_open": (("entry_gate",),
                                   [("trend.read", lambda e: ((e.get("entry_gate") or {})
                                                              .get("trend") or {}).get("read")),
                                    ("action", lambda e: e.get("action"))]),
    "price_cache_rescale": (("applied", "k", "factor"),
                            [("applied", lambda e: e.get("applied"))]),
    "price_cache_scale_gap": (("k", "overlap"), [("action", lambda e: e.get("action"))]),
    "price_cache_rescale_failed": (("k", "factor"), [("action", lambda e: e.get("action"))]),
    "add_stop_qty": (("source", "broker_read", "used_qty", "undersize_flags"),
                     [("broker_read", lambda e: e.get("broker_read")),
                      ("source", lambda e: e.get("source")),
                      ("undersized", lambda e: bool(e.get("undersize_flags")))]),
    "add_stale_record": (("record", "new_stop_vs_prior_stop_pct", "reclassified_open"),
                         [("reclassified_open", lambda e: e.get("reclassified_open"))]),
    "stop_above_market_guard": (("mode", "path", "outcome"),
                                [("path", lambda e: e.get("path")),
                                 ("outcome", lambda e: e.get("outcome")),
                                 ("resting_read", lambda e: e.get("resting_read"))]),
    "add_loser_gate_shadow": (("add_loser_gate", "path", "approval_id"),
                              [("avg_cost_read", lambda e: (e.get("add_loser_gate") or {})
                                .get("avg_cost_read")),
                               ("would_block", lambda e: (e.get("add_loser_gate") or {})
                                .get("would_block"))]),
    "plan_compiled_hit": (("plan_id", "newly_crossed", "levels"),
                          [("held", lambda e: e.get("held")),
                           ("levels_source", lambda e: e.get("levels_source"))]),
    "pt_stop_qty": (("source", "broker_read", "used_qty", "capped"),
                    [("broker_read", lambda e: e.get("broker_read")),
                     ("source", lambda e: e.get("source")),
                     ("capped", lambda e: e.get("capped")),
                     ("below_expected", lambda e: e.get("below_expected"))]),
    "late_fill_expect": (("mode", "path", "source", "fresh_read", "expect_at_least"),
                         [("fresh_read", lambda e: e.get("fresh_read")),
                          ("broker_read", lambda e: e.get("broker_read")),
                          ("source", lambda e: e.get("source")),
                          ("would_defer", lambda e: e.get("would_defer"))]),
    "entry_basis": (("read", "broker_avg", "fill_avg", "basis_gap_frac", "qty_coverage",
                     "trail_arm_vs_fill", "breakeven_vs_fill", "record_written", "stop_outcome",
                     "truncated", "approval_id"),
                    [("read", lambda e: e.get("read")),
                     ("record_written", lambda e: e.get("record_written")),
                     ("truncated", lambda e: e.get("truncated")),
                     ("legacy_tables_apply", lambda e: e.get("legacy_tables_apply"))]),
    "entry_ref_age": (("submitted_at", "sizing_bar_time", "submit_bar_time", "ref_age_s",
                       "submit_ref_age_s", "bid", "ask", "md_type_label", "approval_id",
                       "action", "order_type", "submit_attempted", "outcome_reason",
                       "ref_error"),
                      [("ref", lambda e: f"error: {e['ref_error']}" if e.get("ref_error")
                        else "ok"),
                       ("sizing_price_source", lambda e: e.get("sizing_price_source")),
                       ("md_type_label", lambda e: e.get("md_type_label")),
                       ("submit_attempted", lambda e: e.get("submit_attempted"))]),
    "order_quote": (("md_type_label", "quote_age_s", "order_id", "side"),
                    [("md_type_label", lambda e: e.get("md_type_label")),
                     ("side", lambda e: e.get("side"))]),
    "stop_slippage_partial": (("stop_price", "fill_price", "slippage_pct", "shares", "fill_time",
                               "fill_match", "perm_id", "booked_qty_for_perm",
                               "fill_price_basis", "shadow_key"),
                              [("fill_match", lambda e: e.get("fill_match")),
                               ("fill_price_basis", lambda e: e.get("fill_price_basis"))]),
    "book_gap_risk": (("read", "session_date", "window_gaps", "definitions", "positions",
                       "invested_pct", "loss_at_stops_pct_nav", "n_common_nights",
                       "book_gap_p1_pct_nav", "book_gap_worst_pct_nav", "close_dates",
                       "close_is_today", "shadow_key", "n_suspect_gaps", "suspect_names",
                       "suspect_nights", "n_common_nights_ex_suspect",
                       "book_gap_p1_pct_nav_ex_suspect", "book_gap_worst_pct_nav_ex_suspect"),
                      [("read", lambda e: e.get("read")),
                       ("close_is_today", lambda e: e.get("close_is_today"))]),
    "book_deployment": (("read", "session_date", "source", "source_as_of", "invested_pct",
                         "cash_pct", "other_pct", "cash_target_pct", "gap_pct", "regime",
                         "n_positions", "sector_weights_pct", "largest_name_weight_pct",
                         "definitions", "shadow_key"),
                        [("read", lambda e: e.get("read")),
                         ("cash_target", lambda e: "ok" if e.get("cash_target_pct") is not None
                          else "none"),
                         ("watchlist_read", lambda e: e.get("watchlist_read"))]),
    "late_fill_held_before": (("order_id", "stored_held_before", "held_read", "action_today",
                               "action_would", "mode", "applied"),
                              [("held_read", lambda e: e.get("held_read")),
                               ("action_would", lambda e: e.get("action_would")),
                               ("applied", lambda e: e.get("applied"))]),
}


def check_activity(rows, env: str, modes: dict | None = None) -> list[dict]:
    """rows: dicts {id, stage, ticker, extra} of activity_feed rows. A row
    whose extra carries env is kept only for this env. modes: {stage: the
    config mode this clone loads} for MODED_STAGES; a row in another mode is
    an extra FAIL (absent or None: not checked)."""
    modes = modes or {}
    by_stage: dict = {s: [] for s in ACTIVITY_STAGES}
    for r in rows:
        extra = _as_dict(r.get("extra")) or {}
        if extra.get("env") not in (None, env):
            continue
        if r.get("stage") in by_stage:
            by_stage[r["stage"]].append((r, extra))
    out = []
    for stage in ACTIVITY_STAGES:
        required, getters = _ACTIVITY_SPECS[stage]
        chk = _new(stage, f"activity_feed stage {stage}",
                   "required keys " + ", ".join(required) + " on every row")
        for r, extra in by_stage[stage]:
            chk["rows"] += 1
            chk["triggered"] += 1
            if any(k not in extra for k in required):
                chk["missing"].append(r.get("id"))
            for label, get in getters:
                try:
                    _read(chk, label, get(extra))
                except Exception:
                    _read(chk, label, "error: unreadable")
        if stage == "price_cache_rescale":
            failed = len(by_stage["price_cache_rescale_failed"])
            if failed:
                chk["reads"].setdefault("applied", Counter())["error: rescale write failed"] += failed
                chk["triggered"] += failed
                chk["rule"] += "; a price_cache_rescale_failed row is an error read"
        if stage in MODED_STAGES:
            want = modes.get(stage)
            for r, extra in by_stage[stage]:
                _read(chk, "mode", extra.get("mode"))
                if want is not None and extra.get("mode") != want:
                    chk["extra_fail"].append(f"row {r.get('id')}: mode {_label(extra.get('mode'))}, "
                                             f"config {MODED_STAGES[stage]} is {want}")
            if want is not None:
                chk["rule"] += f"; mode {want} ({MODED_STAGES[stage]})"
        if stage == "add_loser_gate_shadow":
            for r, extra in by_stage[stage]:
                if ((extra.get("add_loser_gate") or {}).get("blocked")) is True:
                    chk["extra_fail"].append(f"row {r.get('id')}: blocked in shadow")
        if stage in _PENDING_NOTE:
            chk["notes"].append(_PENDING_NOTE[stage] + "; WARN until that is deployed")
        if stage == "book_deployment":
            for r, extra in by_stage[stage]:
                bad = sorted(k for k in extra if k in BOOK_DEPLOYMENT_DOLLAR_KEYS)
                if bad:
                    chk["extra_fail"].append(f"row {r.get('id')}: dollar key(s) {', '.join(bad)}")
            chk["rule"] += "; no dollar-amount key (" + ", ".join(BOOK_DEPLOYMENT_DOLLAR_KEYS) + ")"
        if stage == "order_quote":
            chk["notes"].append("md_type_label is the account's data entitlement as the order "
                                "saw it: the order path asks reqMarketDataType(3), so 'delayed' "
                                "(with IBKR Warning 10167) means no live subscription")
        if stage.startswith("price_cache"):
            chk["notes"].append("nightly, after 16:45 ET on a weekday, only when a cached "
                                "ticker's price scale changed; none is normal on most nights")
        out.append(_finish(chk))
    return out


def check_verdict_vote(rows, env: str, enabled: bool) -> dict:
    """rows: activity rows ({id, stage, ticker, extra}); only VOTE_STAGE is
    read, and a row whose extra carries env only for this env. enabled: the
    VERDICT_VOTE_SHADOW_ENABLED this clone loads. Off: OK with none, FAIL on
    any row. On: required keys on every row, status reads (an "error" status
    is an error read), WARN with none."""
    chk = _new(VOTE_STAGE, f"activity_feed stage {VOTE_STAGE}",
               ("required keys " + ", ".join(VOTE_REQUIRED) + " on every row; "
                "VERDICT_VOTE_SHADOW_ENABLED True") if enabled else
               "VERDICT_VOTE_SHADOW_ENABLED False: no rows expected")
    vote = []
    for r in rows:
        if r.get("stage") != VOTE_STAGE:
            continue
        extra = _as_dict(r.get("extra")) or {}
        if extra.get("env") not in (None, env):
            continue
        vote.append((r, extra))
    if not enabled:
        chk["rows"] = chk["triggered"] = len(vote)
        if vote:
            chk["status"] = "FAIL"
            chk["why"] = (f"{len(vote)} row(s) while VERDICT_VOTE_SHADOW_ENABLED is False in this "
                          "clone: the deploy did not load the switch the rows were written under")
        else:
            chk["status"], chk["why"] = "OK", "switch off, no rows"
        return chk
    for r, extra in vote:
        chk["rows"] += 1
        chk["triggered"] += 1
        if any(k not in extra for k in VOTE_REQUIRED):
            chk["missing"].append(r.get("id"))
        _read(chk, "status", extra.get("status"))
        _read(chk, "options_match", extra.get("options_match"))
        _read(chk, "majority_action", extra.get("majority_action"))
    return _finish(chk)


def vote_enabled() -> bool:
    """VERDICT_VOTE_SHADOW_ENABLED as this clone's config.py loads it
    (False when config lacks it)."""
    import config
    return bool(getattr(config, "VERDICT_VOTE_SHADOW_ENABLED", False))


def _note_entry_basis(checks: list[dict], approvals) -> None:
    """The entry_basis line names the window's ADD approvals (is_test left
    out), so "no ADD" and "ADDs but no row" read differently. Both WARN."""
    adds = sum(1 for a in approvals or () if a.get("action") == "ADD" and not a.get("is_test"))
    for chk in checks:
        if chk["name"] != "entry_basis":
            continue
        chk["notes"].append(f"{adds} ADD approval(s) in the window; a row is written only when "
                            f"an ADD merges into an existing stop record")
        if chk["rows"] == 0:
            chk["why"] = ("no ADD in the window" if adds == 0 else
                          f"{adds} ADD approval(s) but no row: none merged into a stop record, "
                          f"or the row was not written (check the execution log)")


def check_risk_subcheck_fail(rows, env: str) -> dict:
    """rows: activity rows ({id, stage, ticker, extra}); only stage
    risk_subcheck_fail is read, and a row whose extra carries env only for
    this env. Inverted rule: OK when there are none, WARN on any row."""
    chk = _new("risk_subcheck_fail", "activity_feed stage risk_subcheck_fail",
               "none expected; any row is WARN (a concentration sub-check is failing in live)")
    for r in rows:
        if r.get("stage") != "risk_subcheck_fail":
            continue
        extra = _as_dict(r.get("extra")) or {}
        if extra.get("env") not in (None, env):
            continue
        chk["rows"] += 1
        chk["triggered"] += 1
        for key in ("check", "error", "mode", "applied"):
            _read(chk, key, extra.get(key))
    if chk["rows"]:
        chk["status"] = "WARN"
        chk["why"] = (f"{chk['rows']} row(s): a theme or moonshot-total sub-check raised "
                      "in live sizing; see the reads and the risk_manager log")
    else:
        chk["status"], chk["why"] = "OK", "no sub-check failed"
    return chk


def _env_rows(rows, stage: str, env: str) -> list:
    """(row, extra) for the activity rows of one stage, a row whose extra
    carries env only for this env."""
    out = []
    for r in rows:
        if r.get("stage") != stage:
            continue
        extra = _as_dict(r.get("extra")) or {}
        if extra.get("env") not in (None, env):
            continue
        out.append((r, extra))
    return out


def check_hwm_guard(rows, env: str) -> dict:
    """rows: activity rows ({id, stage, ticker, extra}); only stage hwm_guard
    is read. Inverted rule: OK with none, WARN on any row (listing the
    reasons), FAIL on a row missing a HWM_GUARD_REQUIRED key."""
    chk = _new("hwm_guard", "activity_feed stage hwm_guard",
               "none expected; any row is WARN (the per-buy drawdown check's high-water mark "
               "is unusable); required keys " + ", ".join(HWM_GUARD_REQUIRED))
    for r, extra in _env_rows(rows, "hwm_guard", env):
        chk["rows"] += 1
        chk["triggered"] += 1
        if any(k not in extra for k in HWM_GUARD_REQUIRED):
            chk["missing"].append(r.get("id"))
        for key in ("reason", "mode", "action", "stored"):
            _read(chk, key, extra.get(key))
    if chk["missing"]:
        chk["status"] = "FAIL"
        chk["why"] = f"{len(chk['missing'])} of {chk['rows']} rows lack a required key"
    elif chk["rows"]:
        reasons = ", ".join(sorted(chk["reads"].get("reason", {})))
        chk["status"] = "WARN"
        chk["why"] = (f"{chk['rows']} row(s), reason(s) {reasons}: the stored high-water mark "
                      "is unusable; check system_state high_water_mark_" + env)
    else:
        chk["status"], chk["why"] = "OK", "no unusable high-water mark"
    return chk


def _refresh_days_due(since: datetime, now: datetime, start: time = FUNDAMENTALS_START,
                      due_at: time = FUNDAMENTALS_DUE) -> list:
    """Session days (Mon-Fri, not market_calendar.NYSE_HOLIDAYS) whose run at
    start ET (default the 17:00 fundamentals refresh) started at or after
    since and is due_at ET (default 18:00) by now."""
    from market_calendar import NYSE_HOLIDAYS
    d, last, due = since.astimezone(NY).date(), now.astimezone(NY).date(), []
    while d <= last:
        if (d.weekday() < 5 and d not in NYSE_HOLIDAYS
                and datetime.combine(d, start, tzinfo=NY) >= since
                and datetime.combine(d, due_at, tzinfo=NY) <= now):
            due.append(d)
        d += timedelta(days=1)
    return due


def check_fundamentals_changes(rows, env: str, since: datetime, now: datetime) -> dict:
    """rows: activity rows; only FUNDAMENTALS_STAGE is read. The activity
    rule (FAIL on a missing key), with no row WARN only when a session day's
    refresh was due in the window. flagged is tallied."""
    chk = _new(FUNDAMENTALS_STAGE, f"activity_feed stage {FUNDAMENTALS_STAGE}",
               "required keys " + ", ".join(FUNDAMENTALS_REQUIRED) + " on every row; one row "
               "per session-day refresh (17:00 ET)")
    for r, extra in _env_rows(rows, FUNDAMENTALS_STAGE, env):
        chk["rows"] += 1
        chk["triggered"] += 1
        if any(k not in extra for k in FUNDAMENTALS_REQUIRED):
            chk["missing"].append(r.get("id"))
        _read(chk, "flagged", extra.get("flagged"))
        _read(chk, "alerted", extra.get("alerted"))
    due = _refresh_days_due(since, now)
    chk["notes"].append(f"session-day refreshes due in the window: {len(due)}"
                        + (f" ({', '.join(d.isoformat() for d in due)})" if due else ""))
    flagged = chk["reads"].get("flagged", {}).get("true", 0)
    if flagged:
        chk["notes"].append(f"{flagged} flagged row(s): targets and forward EPS unchanged "
                            "for the configured run of refreshes (a possible source freeze)")
    if chk["rows"] == 0 and not due:
        chk["status"], chk["why"] = "OK", "no session-day refresh due in the window yet"
        return chk
    _finish(chk)
    if chk["rows"] == 0:
        chk["why"] = f"no row, with {len(due)} session-day refresh(es) due in the window"
    return chk


def check_dcf_branch(rows, env: str, since: datetime, now: datetime) -> dict:
    """rows: activity rows; only DCF_BRANCH_STAGE is read. The activity rule
    (FAIL on a missing key, here per name too), with no row WARN only when a
    session day's ladder refresh was due in the window."""
    chk = _new(DCF_BRANCH_STAGE, f"activity_feed stage {DCF_BRANCH_STAGE}",
               "required keys " + ", ".join(DCF_BRANCH_REQUIRED) + " on every row and "
               + ", ".join(DCF_BRANCH_NAME_REQUIRED) + " on every name; one row per "
               "session-day ladder refresh (17:15 ET)")
    mismatch = []
    for r, extra in _env_rows(rows, DCF_BRANCH_STAGE, env):
        chk["rows"] += 1
        chk["triggered"] += 1
        names = extra.get("names") if isinstance(extra.get("names"), list) else []
        if any(k not in extra for k in DCF_BRANCH_REQUIRED) or any(
                not isinstance(n, dict) or any(k not in n for k in DCF_BRANCH_NAME_REQUIRED)
                for n in names):
            chk["missing"].append(r.get("id"))
            continue
        for n in names:
            reason = str(n.get("reason") or "")
            _read(chk, "dcf_branch", n.get("dcf_branch"))
            _read(chk, "name_read", reason if reason.startswith("error") else "ok")
            if n.get("dcf_matches_ladder") is False:
                mismatch.append(n.get("ticker"))
    if mismatch:
        chk["notes"].append(f"{len(mismatch)} name(s) whose recomputed DCF differs from the "
                            "ladder's: " + ", ".join(str(t) for t in mismatch[:10]))
    due = _refresh_days_due(since, now, DCF_BRANCH_START, DCF_BRANCH_DUE)
    chk["notes"].append(f"session-day ladder refreshes due in the window: {len(due)}")
    chk["notes"].append("record only: dcf_branch is tallied, nothing is judged on it")
    if chk["rows"] == 0 and not due:
        chk["status"], chk["why"] = "OK", "no session-day ladder refresh due in the window yet"
        return chk
    _finish(chk)
    if chk["rows"] == 0:
        chk["why"] = f"no row, with {len(due)} session-day ladder refresh(es) due in the window"
    return chk


def check_feed_health(rows, env: str, since: datetime, now: datetime,
                      enabled: bool = True) -> dict:
    """rows: activity rows; only FEED_HEALTH_STAGE is read. On: the activity
    rule (FAIL on a missing key), with no row WARN only when a session day's
    18:40 run was due in the window; a due day with no row of its own
    ("day") is noted. Off: OK with no rows, FAIL on any. flagged is tallied
    per source."""
    chk = _new(FEED_HEALTH_STAGE, f"activity_feed stage {FEED_HEALTH_STAGE}",
               "required keys " + ", ".join(FEED_HEALTH_REQUIRED) + " on every row; one row "
               "per session-day run (18:40 ET)")
    days, flagged = set(), Counter()
    for r, extra in _env_rows(rows, FEED_HEALTH_STAGE, env):
        chk["rows"] += 1
        chk["triggered"] += 1
        if any(k not in extra for k in FEED_HEALTH_REQUIRED):
            chk["missing"].append(r.get("id"))
        if extra.get("day"):
            days.add(str(extra["day"]))
        fl = extra.get("flagged")
        _read(chk, "any_flagged", bool(fl) if isinstance(fl, list) else fl)
        for src in fl if isinstance(fl, list) else ():
            flagged[str(src)] += 1
    if not enabled:
        if chk["rows"]:
            chk["status"] = "FAIL"
            chk["why"] = (f"FEED_HEALTH_ENABLED is off in this clone but {chk['rows']} row(s) "
                          "exist: the deploy did not load the switch the rows were written under")
        else:
            chk["status"], chk["why"] = "OK", "FEED_HEALTH_ENABLED off: no rows expected"
        return chk
    due = _refresh_days_due(since, now, FEED_HEALTH_START, FEED_HEALTH_DUE)
    chk["notes"].append(f"session-day runs due in the window: {len(due)}"
                        + (f" ({', '.join(d.isoformat() for d in due)})" if due else ""))
    no_row = [d.isoformat() for d in due if d.isoformat() not in days]
    if chk["rows"] and no_row:
        chk["notes"].append(f"due day(s) with no row: {', '.join(no_row)}")
    if flagged:
        chk["notes"].append("flagged: " + ", ".join(f"{k} x{v}" for k, v in sorted(flagged.items())))
    if chk["rows"] == 0 and not due:
        chk["status"], chk["why"] = "OK", "no session-day run due in the window yet"
        return chk
    _finish(chk)
    if chk["rows"] == 0:
        chk["why"] = f"no row, with {len(due)} session-day run(s) due in the window"
    return chk


def feed_health_enabled() -> bool:
    """FEED_HEALTH_ENABLED as this clone's config.py loads it (False when
    config lacks it)."""
    import config
    return bool(getattr(config, "FEED_HEALTH_ENABLED", False))


def _computed_at(extra: dict):
    try:
        t = datetime.fromisoformat(str(extra.get("computed_at")))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def check_source_yield_feeds(rows, env: str, since: datetime) -> dict:
    """rows: activity rows; only FEED_STAGE is read. Every row computed at or
    after since carries FEED_REQUIRED (None allowed); WARN with none, or on
    a non-empty feed_drops. A row computed before since (a spilled row
    replayed) is left out and noted."""
    chk = _new("source_yield", f"activity_feed stage {FEED_STAGE} (feeds, feed_drops)",
               "keys " + ", ".join(FEED_REQUIRED) + " on every row computed since --since "
               "(None allowed); a non-empty feed_drops is WARN")
    drops, older = Counter(), 0
    for r, extra in _env_rows(rows, FEED_STAGE, env):
        at = _computed_at(extra)
        if at is not None and at < since:
            older += 1
            continue
        chk["rows"] += 1
        chk["triggered"] += 1
        if any(k not in extra for k in FEED_REQUIRED):
            chk["missing"].append(r.get("id"))
        _read(chk, "feeds", "present" if isinstance(extra.get("feeds"), dict)
              else extra.get("feeds", "missing"))
        fd = extra.get("feed_drops")
        _read(chk, "feed_drops", "none" if fd is None else ("empty" if not fd else "non-empty"))
        for f in fd or ():
            drops[str(f)] += 1
    if older:
        chk["notes"].append(f"{older} row(s) computed before --since (replayed) left out")
    chk["notes"].append("written at 18:20 ET Mon-Fri for each finished day not yet written")
    _finish(chk)
    if chk["status"] == "OK" and drops:
        chk["status"] = "WARN"
        chk["why"] = ("feed(s) below half their baseline: "
                      + ", ".join(f"{f} ({n})" for f, n in sorted(drops.items())))
    return chk


def check_earnings_events(rows, env: str) -> dict:
    """rows: activity rows; only EARNINGS_STAGE is read. Conditional: WARN
    with none, never FAIL for absence. FAIL on a row missing a required key
    of its kind (an unknown kind counts as missing), or a None value in
    EARNINGS_REASONED without its reason in missing."""
    chk = _new(EARNINGS_STAGE, f"activity_feed stage {EARNINGS_STAGE}",
               "required keys per kind (pre / post) on every row; a None value carries its "
               "reason in missing; conditional (no row is WARN)")
    for r, extra in _env_rows(rows, EARNINGS_STAGE, env):
        chk["rows"] += 1
        chk["triggered"] += 1
        kind = extra.get("kind")
        _read(chk, "kind", kind)
        required = EARNINGS_REQUIRED.get(kind)
        if required is None or any(k not in extra for k in required):
            chk["missing"].append(r.get("id"))
            continue
        why = _as_dict(extra.get("missing")) or {}
        unexplained = [f for f in EARNINGS_REASONED[kind] if extra.get(f) is None and f not in why]
        if unexplained:
            chk["extra_fail"].append(f"row {r.get('id')}: None without a reason: "
                                     + ", ".join(unexplained))
        if kind == "pre":
            _read(chk, "revision", "revised" if extra.get("revision") else "first_seen")
            _read(chk, "announce_date_is_estimate", extra.get("announce_date_is_estimate"))
        else:
            _read(chk, "matched_by", extra.get("matched_by"))
    chk["notes"].append("conditional: a row is written only when an event's date is first "
                        "seen or revised, or its actual appears; no row is normal on most days")
    _finish(chk)
    if chk["rows"] == 0:
        chk["why"] = "no event row in the window (conditional)"
    return chk


def check_read_through(rows, env: str) -> dict:
    """rows: activity rows; only READ_THROUGH_STAGE is read. Conditional and
    WARN-only: no row is WARN (an event whose ticker has a held related
    symbol may not come), and so is a row missing a required key, a
    dedupe_read error or a related section cut from the brief; never FAIL."""
    chk = _new(READ_THROUGH_STAGE, f"activity_feed stage {READ_THROUGH_STAGE}",
               "required keys " + ", ".join(READ_THROUGH_REQUIRED) + " on every row; "
               "conditional and WARN-only (never FAIL)")
    warn = []
    for r, extra in _env_rows(rows, READ_THROUGH_STAGE, env):
        chk["rows"] += 1
        chk["triggered"] += 1
        if any(k not in extra for k in READ_THROUGH_REQUIRED):
            chk["missing"].append(r.get("id"))
        _read(chk, "dedupe_read", extra.get("dedupe_read"))
        _read(chk, "section_within_char_cap", extra.get("section_within_char_cap"))
        _read(chk, "event_ticker", extra.get("event_ticker"))
    if chk["missing"]:
        warn.append(f"{len(chk['missing'])} of {chk['rows']} rows lack a required key")
    errs = sum(v for k, v in chk["reads"].get("dedupe_read", Counter()).items() if _is_error(k))
    if errs:
        warn.append(f"dedupe_read error on {errs} row(s) (Redis; rows written undeduped)")
    cut = chk["reads"].get("section_within_char_cap", Counter()).get("false", 0)
    if cut:
        warn.append(f"{cut} row(s) whose related section was past the brief's character cut")
    chk["notes"].append("conditional: a row only when an analysed event's ticker has a related "
                        "symbol the book holds, once per source item and held ticker")
    if chk["rows"] == 0:
        chk["status"], chk["why"] = "WARN", "no row in the window (conditional)"
    elif warn:
        chk["status"], chk["why"] = "WARN", "; ".join(warn)
    else:
        chk["status"], chk["why"] = "OK", "all rows carry the required keys"
    return chk


def check_news_timeline(verdict_rows, activity, env: str, now: datetime, enabled: bool = True,
                        paths=("event", "daily_scan")) -> dict:
    """verdict_rows: _fetch_verdicts rows ({id, ts, path}); activity: rows of
    NEWS_TIMELINE_STAGE. enabled and paths: NEWS_TIMELINE_RECORD_ENABLED and
    NEWS_TIMELINE_PATHS as this clone loads them. Every verdict on those
    paths older than NEWS_TIMELINE_LAG needs a row with its signal_id (the
    row comes from a side thread, so a younger one is not yet due). FAIL: a
    due verdict without a row, an ok row missing a NEWS_TIMELINE_REQUIRED
    key (a read-error row needs only NEWS_TIMELINE_BASE), or more than 20%
    read errors. Presence and shape only."""
    chk = _new(NEWS_TIMELINE_STAGE, f"activity_feed stage {NEWS_TIMELINE_STAGE} per analyst verdict",
               (f"a row for every verdict on {', '.join(paths)} older than "
                f"{int(NEWS_TIMELINE_LAG.total_seconds() // 60)} min")
               if enabled else "NEWS_TIMELINE_RECORD_ENABLED False: not expected")
    rows = {}
    for r, extra in _env_rows(activity or (), NEWS_TIMELINE_STAGE, env):
        rows.setdefault(extra.get("signal_id"), []).append((r, extra))
        base = NEWS_TIMELINE_BASE if extra.get("read") != "ok" else NEWS_TIMELINE_REQUIRED
        if any(k not in extra for k in base):
            chk["extra_fail"].append(f"activity {r.get('id')}: a required key is missing")
        _read(chk, "read", extra.get("read"))
        _read(chk, "path", extra.get("path"))
    if not enabled:
        chk["status"], chk["why"] = "OK", "record off in this clone; not expected"
        chk["notes"].append(f"rows seen: {sum(len(v) for v in rows.values())}")
        return chk
    young = 0
    for v in verdict_rows or ():
        if v.get("path") not in paths:
            continue
        ts = v.get("ts")
        if isinstance(ts, datetime):
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if now - ts < NEWS_TIMELINE_LAG:
                young += 1
                continue
        chk["rows"] += 1
        chk["triggered"] += 1
        if v.get("id") not in rows:
            chk["missing"].append(v.get("id"))
    dup = sum(1 for k, v in rows.items() if len(v) > 1)
    chk["notes"].append(f"verdicts younger than the lag, not yet due: {young}; "
                        f"verdicts with more than one row: {dup}")
    return _finish(chk)


def news_timeline_config() -> tuple[bool, tuple]:
    """(NEWS_TIMELINE_RECORD_ENABLED, NEWS_TIMELINE_PATHS) as this clone's
    config.py loads them; a clone without the switch reads (False, ())."""
    import config
    return (bool(getattr(config, "NEWS_TIMELINE_RECORD_ENABLED", False)),
            tuple(getattr(config, "NEWS_TIMELINE_PATHS", ())))


def check_plan_ledger_record(rows, reseed_on: bool = True) -> dict:
    """rows: full analyst verdict dicts as _fetch_verdicts returns them, with
    plan (provenance.plan, None when absent). reseed_on: the
    PLAN_RESEED_RECORD_ENABLED this clone loads. Triggers: plan observations
    whose relation wrote a plan (NEW_PLAN, SEEDED, REVISE)."""
    chk = _new("plan_ledger_record", "analyst verdicts provenance.plan",
               "horizon_basis on every NEW_PLAN/SEEDED/REVISE"
               + ("; reseeded_after_invalidate and reseed on every NEW_PLAN/SEEDED"
                  if reseed_on else "; PLAN_RESEED_RECORD_ENABLED False: reseed not checked"))
    stopped, reseed_while_off = 0, 0
    for r in rows or ():
        plan = _as_dict(r.get("plan"))
        if plan is None:
            continue
        chk["rows"] += 1
        if plan.get("stopped_out_since_plan") is not None:
            stopped += 1
        rel = plan.get("relation")
        if rel not in PLAN_LEDGER_HORIZON_RELATIONS:
            continue
        chk["triggered"] += 1
        missing = "horizon_basis" not in plan
        if not missing:
            _read(chk, "horizon_basis", plan.get("horizon_basis") if "horizon_record_error" not in plan
                  else f"error: {plan.get('horizon_record_error')}")
        if rel in PLAN_LEDGER_RESEED_RELATIONS:
            if reseed_on:
                rec = _as_dict(plan.get("reseed"))
                if rec is None or "reseeded_after_invalidate" not in plan:
                    missing = True
                else:
                    _read(chk, "reseed.status", rec.get("status"))
                    _read(chk, "reseeded_after_invalidate", plan.get("reseeded_after_invalidate"))
            elif "reseed" in plan:
                reseed_while_off += 1
        if missing:
            chk["missing"].append(r.get("id"))
    chk["notes"].append(f"stopped_out_since_plan recorded on {stopped} plan observations "
                        "(plans kept active after their trade closed)")
    if reseed_while_off:
        chk["notes"].append(f"{reseed_while_off} NEW_PLAN/SEEDED rows carry reseed with the "
                            "switch off here (written under another config)")
    chk["notes"].append("record only: the flags are tallied, nothing is judged on them")
    return _finish(chk)


def plan_reseed_enabled() -> bool:
    import config
    return bool(getattr(config, "PLAN_RESEED_RECORD_ENABLED", False))


def check_own_ticker_rejected(rows, patterns_sha: str | None = None) -> dict:
    """rows: SCREENED_OUT dicts {id, has_provenance, has_key, value, sha,
    error} (_fetch_screened_out). patterns_sha: agents.own_ticker.PATTERNS_SHA
    as this clone loads it (None: not compared). Rows without a provenance
    (its build failed) are not triggers."""
    chk = _new("own_ticker_rejected", "SCREENED_OUT signals provenance.own_ticker_rejected",
               "on every SCREENED_OUT row with a provenance, with own_ticker_patterns_sha")
    other_sha = Counter()
    for r in rows or ():
        if not r.get("has_provenance"):
            continue
        chk["rows"] += 1
        chk["triggered"] += 1
        if not r.get("has_key") or not r.get("sha"):
            chk["missing"].append(r.get("id"))
            continue
        if r.get("error"):
            _read(chk, "record", f"error: {r.get('error')}")
            continue
        _read(chk, "record", "ok")
        _read(chk, "own_ticker_rejected", r.get("value"))
        if patterns_sha and r.get("sha") != patterns_sha:
            other_sha[r.get("sha")] += 1
    for sha, n in sorted(other_sha.items()):
        chk["extra_fail"].append(f"{n} rows written under patterns sha {sha}, not {patterns_sha}")
    chk["notes"].append("record only: own_ticker_rejected is tallied, nothing is judged on it")
    return _finish(chk)


def own_ticker_patterns_sha() -> str | None:
    try:
        from agents.own_ticker import PATTERNS_SHA
        return PATTERNS_SHA
    except Exception:
        return None


def _fetch_screened_out(since: datetime, env: str) -> list[dict]:
    """SCREENED_OUT rows since `since`: the own_ticker keys only (no text).
    Read only (rolled back)."""
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT id, provenance IS NOT NULL, COALESCE(provenance ? 'own_ticker_rejected', false), "
                        "provenance->>'own_ticker_rejected', provenance->>'own_ticker_patterns_sha', "
                        "provenance->>'own_ticker_error' FROM signals "
                        "WHERE timestamp >= %s AND trading_env = %s AND signal = 'SCREENED_OUT'",
                        (since, env))
            return [{"id": i, "has_provenance": hp, "has_key": hk,
                     "value": {"true": True, "false": False}.get(v), "sha": sha, "error": err}
                    for i, hp, hk, v, sha, err in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)


def check_target_anchor(rows, enabled: bool = True) -> dict:
    """rows: dicts {id, ticker, target_anchor} of full analyst verdicts
    (_fetch_verdicts: decision.target_anchor, None when absent). enabled: the
    TARGET_ANCHOR_RECORD_ENABLED this clone loads."""
    chk = _new("target_anchor", "analyst verdicts decision.target_anchor",
               ("required keys " + ", ".join(TARGET_ANCHOR_REQUIRED) + " on every verdict; "
                "TARGET_ANCHOR_RECORD_ENABLED True") if enabled else
               "TARGET_ANCHOR_RECORD_ENABLED False: not checked")
    if not enabled:
        chk["rows"] = len(rows or [])
        chk["status"], chk["why"] = "OK", "switch off, not checked"
        return chk
    for r in rows or []:
        chk["rows"] += 1
        chk["triggered"] += 1
        rec = _as_dict(r.get("target_anchor"))
        if rec is None or any(k not in rec for k in TARGET_ANCHOR_REQUIRED):
            chk["missing"].append(r.get("id"))
            continue
        _read(chk, "timing_read", rec.get("timing_read"))
        _read(chk, "street_target in brief", rec.get("street_target") is not None)
        _read(chk, "fundamentals_predate_last_print", rec.get("fundamentals_predate_last_print"))
        _read(chk, "fv_confidence", rec.get("fv_confidence"))
        _read(chk, "target_equals_fv", rec.get("target_equals_fv"))
        _read(chk, "target_equals_street", rec.get("target_equals_street"))
    return _finish(chk)


def target_anchor_enabled() -> bool:
    import config
    return bool(getattr(config, "TARGET_ANCHOR_RECORD_ENABLED", False))



def check_earnings_claim(rows) -> dict:
    """rows: dicts {id, ticker, catalyst_check} of full analyst verdicts (rows
    with decision.model_output). The record is on every one."""
    chk = _new("earnings_claim", "analyst verdicts decision.catalyst_check",
               "required keys " + ", ".join(EARNINGS_CLAIM_REQUIRED) + " on every full verdict")
    for r in rows:
        chk["rows"] += 1
        chk["triggered"] += 1
        cc = _as_dict(r.get("catalyst_check")) or {}
        if any(k not in cc for k in EARNINGS_CLAIM_REQUIRED):
            chk["missing"].append(r.get("id"))
            continue
        detail = _as_dict(cc.get("earnings_claim")) or {}
        _read(chk, "earnings_claim_unmatched", cc.get("earnings_claim_unmatched"))
        _read(chk, "matches_sector_cluster_date", cc.get("matches_sector_cluster_date"))
        _read(chk, "calendar_source_used", cc.get("calendar_source_used"))
        _read(chk, "read_dates", detail.get("read_dates") or detail.get("error"))
        _read(chk, "read_sector", detail.get("read_sector") or detail.get("error"))
    chk["notes"].append("record only: earnings_claim_unmatched and matches_sector_cluster_date "
                        "are tallied, nothing is judged on them")
    return _finish(chk)


def check_catalyst_kind(rows) -> dict:
    """rows: as check_earnings_claim. catalyst_kind is on every full verdict."""
    chk = _new("catalyst_kind", "analyst verdicts decision.catalyst_check",
               "required keys " + ", ".join(CATALYST_KIND_REQUIRED) + " on every full verdict")
    for r in rows:
        chk["rows"] += 1
        chk["triggered"] += 1
        cc = _as_dict(r.get("catalyst_check")) or {}
        if any(k not in cc for k in CATALYST_KIND_REQUIRED):
            chk["missing"].append(r.get("id"))
            continue
        kind = cc.get("catalyst_kind")
        if kind not in CATALYST_KIND_VALUES:
            chk["extra_fail"].append(f"row {r.get('id')}: catalyst_kind {str(kind)[:40]!r} is not a kind")
            continue
        detail = _as_dict(cc.get("catalyst_kind_detail")) or {}
        _read(chk, "catalyst_kind", kind)
        _read(chk, "catalyst_matched_source", cc.get("catalyst_matched_source"))
        _read(chk, "read_events", detail.get("read_events") or detail.get("error"))
    chk["notes"].append("record only: catalyst_kind is tallied, nothing is judged on it")
    return _finish(chk)


def _fetch_verdicts(since: datetime, env: str) -> list[dict]:
    """Full analyst verdicts since `since` (decision.model_output present):
    their id, ticker, decision.catalyst_check and target_anchor, time, path
    and provenance.plan. Read only (rolled back)."""
    import db
    non_analyst = list(getattr(db, "_NON_ANALYST_SIGNALS", ()))
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT id, symbol, decision->'catalyst_check', "
                        "decision->'target_anchor', timestamp, provenance->'trigger'->>'path', "
                        "provenance->'plan' FROM signals "
                        "WHERE timestamp >= %s AND trading_env = %s AND decision ? 'model_output' "
                        "AND NOT (signal = ANY(%s))", (since, env, non_analyst))
            return [{"id": i, "ticker": t, "catalyst_check": c, "target_anchor": ta, "ts": ts,
                     "path": p, "plan": pl}
                    for i, t, c, ta, ts, p, pl in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)


_WATCHLIST_ROW_RE = re.compile(r"\|\s*([A-Z]{1,6})\s*\|")


def watchlist_unparsed():
    """The tickers of watchlist.md rows knowledge_loader.load_watchlist
    doesn't parse, from its own "row not parsed" warnings on a fresh load
    (cache_clear in this process only; knowledge_loader is not changed).
    Returns a sorted list, or the exception when the load raised."""
    import logging
    try:
        import knowledge_loader
    except Exception as e:
        return e
    caught: list = []

    class _Catch(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return
            if "watchlist.md row not parsed" in msg:
                caught.append(msg)

    logger = logging.getLogger(knowledge_loader.__name__)
    handler = _Catch(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        knowledge_loader.load_watchlist.cache_clear()
        knowledge_loader.load_watchlist()
    except Exception as e:
        return e
    finally:
        logger.removeHandler(handler)
    out = set()
    for msg in caught:
        m = _WATCHLIST_ROW_RE.search(msg.split(":", 1)[-1])
        out.add(m.group(1) if m else msg[-60:])
    return sorted(out)


def check_watchlist_parse(result) -> dict:
    """result: watchlist_unparsed()'s list, or an exception. WARN-only."""
    chk = _new("watchlist_parse", "knowledge_loader.load_watchlist (its 'row not parsed' warning)",
               "none expected; an unparsed ticker row is WARN (out of the universe); never FAIL")
    if isinstance(result, Exception):
        chk["status"], chk["why"] = "WARN", f"watchlist read failed ({type(result).__name__})"
        return chk
    chk["rows"] = chk["triggered"] = len(result)
    if result:
        chk["status"] = "WARN"
        chk["why"] = (f"{len(result)} watchlist.md row(s) not parsed, so out of the universe: "
                      + ", ".join(result) + " (the rows are Luke's to fix)")
    else:
        chk["status"], chk["why"] = "OK", "every ticker row parses"
    return chk


def check_entry_funnel(result) -> dict:
    """result: {"line", "verdicts", "passed", "sessions"} from the entry
    funnel's check over its last CHECK_SESSIONS sessions, {"error": text}
    when its read raised, or None when it was not run."""
    chk = _new("entry_funnel", "scripts/entry_funnel.py --check",
               "passed buys > 0 over the last check sessions when they have enough verdicts")
    if result is None:
        chk["notes"].append("not run")
        return _finish(chk)
    if result.get("error"):
        chk["triggered"] = 1
        chk["extra_fail"].append(f"the funnel read raised {result['error']}")
        return _finish(chk)
    sessions = result.get("sessions") or []
    chk["rows"] = int(result.get("verdicts") or 0)
    chk["reads"]["passed"] = Counter({"passed": int(result.get("passed") or 0)})
    if sessions:
        chk["notes"].append(f"sessions {sessions[0]}..{sessions[-1]}: {result.get('verdicts')} "
                            f"OPEN/ADD verdicts, {result.get('passed')} passed")
    if result.get("line"):
        chk["triggered"] = 1
        chk["extra_fail"].append(result["line"])
    elif int(result.get("passed") or 0) > 0:
        chk["triggered"] = 1
    else:
        chk["notes"].append("too few verdicts for the check")
    return _finish(chk)


def unmatched_broker_exits(exit_rows, seen_rows) -> list[dict]:
    """exit_rows: (ticker, exit_fill_at). seen_rows: activity dicts of stage
    broker_sell_seen. Exits with no seen row whose first_exec_at is within
    SEEN_MATCH_TOLERANCE of the fill."""
    seen = []
    for r in seen_rows:
        extra = _as_dict(r.get("extra")) or {}
        try:
            seen.append((r.get("ticker"), _aware(extra.get("first_exec_at"))))
        except (TypeError, ValueError):
            continue
    out, done = [], set()
    for ticker, fill in exit_rows:
        fill = _aware(fill)
        if fill is None or (ticker, fill) in done:
            continue
        done.add((ticker, fill))
        if not any(t == ticker and f is not None and abs(f - fill) <= SEEN_MATCH_TOLERANCE
                   for t, f in seen):
            out.append({"ticker": ticker, "fill": fill})
    return out


def _aware(t):
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def watcher_runs(lines, since: datetime) -> dict:
    """Counts per ET hour of tasks.watch_broker_sells outcomes in Celery log
    lines (the box logs in ET): {hour datetime: Counter(succeeded, raised,
    retry, discarded)}."""
    hours: dict = {}
    for line in lines:
        if WATCH_TASK not in line:
            continue
        m = _LOG_RE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY)
        if ts < since:
            continue
        hour = ts.replace(minute=0, second=0)
        hours.setdefault(hour, Counter())[m.group(2) or "discarded"] += 1
    return hours


def check_watcher(hours: dict, since: datetime, now: datetime) -> dict:
    chk = _new("watch_broker_sells", "Celery logs logs/*.log, per ET hour",
               ">=1 succeeded run in each elapsed weekday market hour; raised within 20%")
    total = Counter()
    for c in hours.values():
        total.update(c)
    finished = total["succeeded"] + total["raised"]
    # "raised" is this record's error read (the 20% rule in _finish)
    chk["reads"]["outcome"] = Counter({("error: raised" if k == "raised" else k): v
                                       for k, v in total.items() if v})
    elapsed = 0
    d = since.astimezone(NY).date()
    while d <= now.astimezone(NY).date():
        if d.weekday() < 5:
            for h in range(MARKET_OPEN.hour, MARKET_CLOSE.hour):
                start = max(datetime.combine(d, time(h), tzinfo=NY),
                            datetime.combine(d, MARKET_OPEN, tzinfo=NY))
                end = datetime.combine(d, time(h + 1), tzinfo=NY)
                if start >= since and end <= now:
                    elapsed += 1
                    if not hours.get(datetime.combine(d, time(h), tzinfo=NY), Counter())["succeeded"]:
                        chk["extra_fail"].append(f"no succeeded run {start:%m-%d %H:%M}-{end:%H:%M} ET")
        d += timedelta(days=1)
    chk["rows"] = finished
    # An elapsed market hour is a trigger too: no runs through one is a FAIL, not a WARN.
    chk["triggered"] = finished or elapsed
    chk["hours"] = hours
    return _finish(chk)


# ---------------------------------------------------------------- output

def _dist_text(c: Counter) -> str:
    n = sum(c.values())
    return ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in c.most_common()) if n else "none"


def readout(checks: list[dict], since: datetime, env: str, head: str, now: datetime,
            unmatched: list[dict] | None = None) -> str:
    lines = [f"# Batch verify ({env}, rows since {since.astimezone(NY):%Y-%m-%d %H:%M} ET)", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version (git HEAD of this "
             f"clone): {head}. Engineering diagnostic; counts and percentages only.", "",
             "| status | record | rows | missing | why |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['status']} | {c['name']} | {c['rows']} | {len(c['missing'])} | {c['why']} |")
    for c in checks:
        lines += ["", f"## {c['name']}: {c['status']}", "",
                  f"- source: {c['source']}", f"- rule: {c['rule']}",
                  f"- rows written: {c['rows']}"]
        if c["missing"]:
            ids = ", ".join(str(i) for i in c["missing"][:10])
            more = f" (+{len(c['missing']) - 10} more)" if len(c["missing"]) > 10 else ""
            lines.append(f"- missing the field: {len(c['missing'])} of {c['triggered']} "
                         f"({len(c['missing']) / c['triggered']:.0%}); row ids {ids}{more}")
        for label, cnt in c["reads"].items():
            lines.append(f"- {label}: {_dist_text(cnt)}")
            errs = Counter({k: v for k, v in cnt.items() if _is_error(k)})
            if errs:
                lines.append(f"  - error reasons: {_dist_text(errs)}")
        for x in c["extra_fail"][:10]:
            lines.append(f"- FAIL condition: {x}")
        for note in c["notes"]:
            lines.append(f"- note: {note}")
        if c["name"] == "broker_sell_seen" and unmatched is not None:
            lines.append(f"- broker exits since --since with no seen row within 30 min of the "
                         f"fill: {len(unmatched)}"
                         + (" (" + ", ".join(f"{u['ticker']} {u['fill'].astimezone(NY):%H:%M} ET"
                                            for u in unmatched[:10]) + ")" if unmatched else "")
                         + "; a note, not a FAIL")
        if c["name"] == "watch_broker_sells":
            hrs = c.get("hours") or {}
            if hrs:
                lines += ["", "| ET hour | succeeded | raised | retry | discarded |",
                          "|---|---|---|---|---|"]
                for h in sorted(hrs):
                    k = hrs[h]
                    lines.append(f"| {h:%m-%d %H}:00 | {k['succeeded']} | {k['raised']} | "
                                 f"{k['retry']} | {k['discarded']} |")
                lines.append("")
                lines.append("Beat runs it every 120 s (about 30 an hour); off market hours "
                             "the run returns at once and still counts as succeeded.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def _fetch(since: datetime, env: str):
    import db
    non_analyst = list(getattr(db, "_NON_ANALYST_SIGNALS", ()))
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT id, symbol, signal, decision FROM signals WHERE timestamp >= %s "
                        "AND trading_env = %s AND signal IN ('ENTRY_GATE_BLOCKED', 'TRIM_EXIT_RULE')",
                        (since, env))
            gate_rows, trim_rows = [], []
            for sid, sym, kind, dec in cur.fetchall():
                (gate_rows if kind == "ENTRY_GATE_BLOCKED" else trim_rows).append(
                    {"id": sid, "ticker": sym, "decision": dec})
            cur.execute("SELECT id, symbol, signal, provenance->'trigger', "
                        "decision ? 'model_output', decision->'source_mix' FROM signals "
                        "WHERE timestamp >= %s AND trading_env = %s "
                        "AND provenance->'trigger'->>'path' = 'event' AND NOT (signal = ANY(%s))",
                        (since, env, non_analyst))
            event_rows = [{"id": i, "ticker": s, "kind": k, "trigger": t,
                           "has_model_output": mo is True, "source_mix": sm}
                          for i, s, k, t, mo, sm in cur.fetchall()]
            cur.execute("SELECT id, ticker, action, is_test, risk_json FROM pending_approvals "
                        "WHERE trading_env = %s AND timestamp >= %s", (env, since))
            approvals = [{"id": i, "ticker": t, "action": a, "is_test": x, "risk_json": rj}
                         for i, t, a, x, rj in cur.fetchall()]
            cur.execute("SELECT id, stage, ticker, extra_json FROM activity_feed "
                        "WHERE ts >= %s AND stage = ANY(%s)",
                        (since, list(ACTIVITY_STAGES) + list(ALARM_STAGES)
                         + [VOTE_STAGE, FUNDAMENTALS_STAGE, DCF_BRANCH_STAGE, FEED_HEALTH_STAGE, FEED_STAGE,
                            EARNINGS_STAGE, READ_THROUGH_STAGE, NEWS_TIMELINE_STAGE]))
            activity = [{"id": i, "stage": s, "ticker": t, "extra": e}
                        for i, s, t, e in cur.fetchall()]
            cur.execute("SELECT ticker, exit_fill_at FROM trade_outcomes WHERE trading_env = %s "
                        "AND exit_source = 'broker_exit' AND exit_fill_at >= %s", (env, since))
            exits = cur.fetchall()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return gate_rows, trim_rows, event_rows, approvals, activity, exits


def _log_lines(log_dir: str):
    """Lines of logs/*.log that name the watcher. Only *.log files are read."""
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    if WATCH_TASK in line:
                        yield line
        except OSError:
            continue


def _git_head(root: str) -> str:
    import subprocess
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        subj = subprocess.run(["git", "-C", root, "log", "-1", "--format=%s"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha} on {branch} ({subj[:80]})" if sha else "unknown"
    except Exception:
        return "unknown"


def parse_since(text: str | None, now: datetime) -> datetime:
    """'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' in ET; default today 05:00 ET."""
    if not text:
        return datetime.combine(now.astimezone(NY).date(), time(5), tzinfo=NY)
    if len(text) <= 10:
        return datetime.combine(date.fromisoformat(text), time(0), tzinfo=NY)
    return datetime.fromisoformat(text).replace(tzinfo=NY)


def _entry_funnel_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_funnel.py")
    spec = importlib.util.spec_from_file_location("entry_funnel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def entry_funnel_result(now: datetime, env: str) -> dict:
    """The entry funnel's --check over its last CHECK_SESSIONS closed
    sessions, as check_entry_funnel takes it. Never raises."""
    try:
        ef = _entry_funnel_module()
        sessions = ef.sessions_until(ef.default_until(now), ef.CHECK_SESSIONS)
        funnel = ef.run(sessions, env)
        return {"line": ef.funnel_check(funnel, sessions), "sessions": sessions,
                "verdicts": sum(sum(funnel[s]["verdicts"].values()) for s in sessions),
                "passed": sum(funnel[s]["passed"] for s in sessions)}
    except Exception as e:
        return {"error": type(e).__name__}


def config_modes() -> tuple[dict, str]:
    """({stage: mode} for MODED_STAGES, the add-to-loser mode) as this clone's
    config.py loads them; a name config lacks is None (not checked)."""
    import config
    modes = {}
    for stage, name in MODED_STAGES.items():
        v = getattr(config, name, None)
        modes[stage] = str(v).lower() if v is not None else None
    loser = str(getattr(config, "ADD_LOSER_GATE_MODE", "shadow")).lower()
    return modes, (loser if loser in ("enforce", "off") else "shadow")


def run_checks(gate_rows, trim_rows, event_rows, approvals, activity, exits, hours,
               since: datetime, now: datetime, env: str, modes: dict | None = None,
               add_loser_mode: str = "shadow", funnel: dict | None = None,
               vote_on: bool = False, watchlist=None, consensus_on: bool = True,
               feed_health_on: bool = True, verdict_rows=(), target_anchor_on: bool = True,
               news_timeline=(True, ("event", "daily_scan")), plan_reseed_on: bool = True,
               screened_out_rows=(), own_ticker_sha: str | None = None):
    checks = [check_entry_gate(gate_rows), check_story_dedup(event_rows),
              check_source_mix(event_rows),
              *check_approvals(approvals),
              check_same_day_consensus(approvals, gate_rows, consensus_on),
              check_add_loser(approvals, add_loser_mode),
              *check_trim_exit(trim_rows), *check_activity(activity, env, modes),
              check_verdict_vote(activity, env, vote_on),
              check_risk_subcheck_fail(activity, env), check_hwm_guard(activity, env),
              check_fundamentals_changes(activity, env, since, now),
              check_dcf_branch(activity, env, since, now),
              check_feed_health(activity, env, since, now, feed_health_on),
              check_source_yield_feeds(activity, env, since), check_earnings_events(activity, env),
              check_read_through(activity, env), check_earnings_claim(verdict_rows),
              check_catalyst_kind(verdict_rows),
              check_target_anchor(verdict_rows, target_anchor_on),
              check_news_timeline(verdict_rows, activity, env, now, *news_timeline),
              check_plan_ledger_record(verdict_rows, plan_reseed_on),
              check_own_ticker_rejected(screened_out_rows, own_ticker_sha),
              check_entry_funnel(funnel), check_watcher(hours, since, now)]
    if watchlist is not None:
        checks.append(check_watchlist_parse(watchlist))
    _note_entry_basis(checks, approvals)
    seen = [r for r in activity if r.get("stage") == "broker_sell_seen"
            and (_as_dict(r.get("extra")) or {}).get("env") in (None, env)]
    return checks, unmatched_broker_exits(exits, seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="ET 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'; default today 05:00 ET")
    ap.add_argument("--logs-dir", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    since = parse_since(args.since, now)
    from config import TRADING_ENV
    rows = _fetch(since, TRADING_ENV)
    hours = watcher_runs(_log_lines(args.logs_dir), since)
    modes, loser = config_modes()
    checks, unmatched = run_checks(*rows, hours, since, now, TRADING_ENV, modes=modes,
                                   add_loser_mode=loser,
                                   funnel=entry_funnel_result(now, TRADING_ENV),
                                   vote_on=vote_enabled(), watchlist=watchlist_unparsed(),
                                   consensus_on=same_day_consensus_enabled(),
                                   feed_health_on=feed_health_enabled(),
                                   verdict_rows=_fetch_verdicts(since, TRADING_ENV),
                                   target_anchor_on=target_anchor_enabled(),
                                   news_timeline=news_timeline_config(),
                                   plan_reseed_on=plan_reseed_enabled(),
                                   screened_out_rows=_fetch_screened_out(since, TRADING_ENV),
                                   own_ticker_sha=own_ticker_patterns_sha())
    text = readout(checks, since, TRADING_ENV, _git_head(ROOT), now, unmatched)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    sys.exit(1 if any(c["status"] == "FAIL" for c in checks) else 0)


if __name__ == "__main__":
    main()
