"""Row readout for the shadow registry (engineering inventory, not a study
analysis): for each batch shadow card and record card in helios-research
"08 Helios Engineering/shadow-registry.md" (STAGES) it reports

  - rows since the load date (--load-date; default BATCH_LOAD_DEFAULT, the
    batch-1124 load day; pass the deploy's restart time once known),
  - rows since the start of the last SESSIONS NYSE sessions (ET; the
    session in progress counts when today is one),
  - the newest row's time (all time, ET),
  - for a conditional stage that names one, its trigger's rows since
    max(load, the first of those sessions),
  - the switch as this clone's config.py loads it, and whether it records.

Each stage says how its rows should come (expect_rows):
  - always: rows every session (a daily task, or event verdicts). Silent
    when the switch records and no row is in the last SESSIONS sessions.
  - conditional: rows only when something happens (an ADD, a partial stop
    fill, a new earnings date). Silent only when its trigger (a named source,
    where one exists) had rows since max(load, the first of those sessions)
    and the stage had none in the same span. With no trigger rows, or no
    trigger named, 0 rows is expected: "no trigger yet", never silent.
  - alarm: rows only when something went wrong (a sub-check raised, an
    unusable high-water mark). Never silent; any row in the last SESSIONS
    sessions is flagged "alarm".
These are presence flags only: no readout or switch criterion is applied
here (those are Luke's), and no count is judged. A stage with no switch (it
always records, as shipped) shows "no switch".

Known ways a conditional stage reads "silent" when nothing is wrong:
  - a buy-triggered stage whose only OPEN/ADD approval in the span came from
    a path outside crew routing (a deferred retry, a manual approval);
  - add_loser_gate with ADD_LOSER_GATE_MODE "enforce": an approved ADD then
    carries no risk_json key (it ships as "shadow");
  - a trigger row seconds old when this runs, before its record is written;
  - same_day_consensus counts approvals only, though it is also written on
    the routing row: approvals are a lower bound for its trigger.

"Records" is mechanical: a True bool, or a mode other than "off". A switch
config lacks is "absent" (this clone predates the card), not recording; so
is a no-switch stage whose writer token is not in this clone's file.
The registry's default-off (env-controlled) features are not listed: their
rows are written while their switch is off, so "switch on" doesn't mean the
rows should be coming in.

Sources: signals, pending_approvals, orders and trades rows are this
TRADING_ENV only (approvals is_test excluded); activity_feed has no env
column, so a row is counted when its extra_json.env is this env or missing.
A card with two sources (recheck_contradiction, add-to-loser gate) sums them.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/shadow_registry_check.py [--load-date "2026-11-23 19:00"] [--json]

Read-only: SUPABASE_PORT must be 6543 (the transaction pooler; the script
exits otherwise). One transaction, SET TRANSACTION READ ONLY with a statement
timeout, then rolled back. Counts and times only: no prices, quantities or
account values.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
SESSIONS = 5
BATCH_LOAD_DEFAULT = "2026-11-23"          # batch-1124 load day (Mon evening restart)
STATEMENT_TIMEOUT_MS = 15000

# Row sources. Each is (table, condition); the table fixes the time column
# and the env filter (_SQL_BASE).
_SIG = "signals"
_APR = "pending_approvals"
_ACT = "activity_feed"
_ORD = "orders"
_TRD = "trades"
_SQL_BASE = {
    _SIG: ("SELECT timestamp FROM signals WHERE trading_env = %(env)s AND ({cond}) "
           "AND timestamp >= %(since)s",
           "SELECT max(timestamp) FROM signals WHERE trading_env = %(env)s AND ({cond})"),
    _APR: ("SELECT timestamp FROM pending_approvals WHERE trading_env = %(env)s "
           "AND COALESCE(is_test, false) = false AND ({cond}) AND timestamp >= %(since)s",
           "SELECT max(timestamp) FROM pending_approvals WHERE trading_env = %(env)s "
           "AND COALESCE(is_test, false) = false AND ({cond})"),
    _ACT: ("SELECT ts FROM activity_feed WHERE COALESCE(extra_json->>'env', %(env)s) = %(env)s "
           "AND ({cond}) AND ts >= %(since)s",
           "SELECT max(ts) FROM activity_feed WHERE COALESCE(extra_json->>'env', %(env)s) = %(env)s "
           "AND ({cond})"),
    _ORD: ("SELECT created_at FROM orders WHERE trading_env = %(env)s AND ({cond}) "
           "AND created_at >= %(since)s",
           "SELECT max(created_at) FROM orders WHERE trading_env = %(env)s AND ({cond})"),
    _TRD: ("SELECT timestamp FROM trades WHERE trading_env = %(env)s AND ({cond}) "
           "AND timestamp >= %(since)s",
           "SELECT max(timestamp) FROM trades WHERE trading_env = %(env)s AND ({cond})"),
}
EXPECT = ("always", "conditional", "alarm")
TRIGGERS_KEY = "__triggers__"   # fetch's {stage: [trigger row times]}

# Triggers: rows that make a conditional stage due (a superset is avoided:
# each is a row the stage's writer runs on).
_BUY_APPROVALS = (_APR, "action IN ('OPEN', 'ADD')")
_ADD_APPROVALS = (_APR, "action = 'ADD'")
_ADD_FILLS = (_ORD, "side = 'BUY' AND intent = 'ADD' AND status IN ('filled', 'partial')")


def _stage(cond: str) -> tuple[str, str]:
    return (_ACT, f"stage = '{cond}'")


def _risk(key: str) -> tuple[str, str]:
    return (_APR, f"risk_json ? '{key}'")


# One entry per record named on a registry card, in the registry's order.
# switch None: no switch, the record is always written as shipped; "writer"
# (file, token) is how a clone that predates it is told apart (writer_present).
# source_yield_feeds counts only rows whose per-feed breakdown was read
# (feeds is null when it was not).
STAGES = (
    {"stage": "held_exit_candidates", "card": "held-exit candidates",
     "switch": "HELD_EXIT_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": (_SIG, "signal = 'TRIM_EXIT_RULE'"),
     "sources": ((_SIG, "signal = 'TRIM_EXIT_RULE' AND decision ? 'held_exit_candidates'"),)},
    {"stage": "plan_compiled_hit", "card": "plan compiled-level hits",
     "switch": "PLAN_COMPILED_HIT_SHADOW_ENABLED", "expect_rows": "conditional",
     "sources": (_stage("plan_compiled_hit"),)},
    {"stage": "rr_vs_stop", "card": "reward-to-risk vs the placed stop",
     "switch": "RR_VS_STOP_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS, "sources": (_risk("rr_vs_stop"),)},
    {"stage": "add_loser_gate", "card": "add-to-loser gate",
     "switch": "ADD_LOSER_GATE_MODE", "expect_rows": "conditional", "trigger": _ADD_APPROVALS,
     "sources": (_risk("add_loser_gate"), _stage("add_loser_gate_shadow"))},
    {"stage": "recheck_contradiction", "card": "recheck contradiction",
     "switch": "RECHECK_CONTRADICTION_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS,
     "sources": (_risk("recheck_contradiction"),
                 (_SIG, "decision ? 'recheck_contradiction'"))},
    {"stage": "story_dedup", "card": "story de-duplication and staleness",
     "switch": "STORY_DEDUP_SHADOW_ENABLED", "expect_rows": "always",
     "sources": ((_SIG, "provenance->'trigger' ? 'story_dedup'"),)},
    {"stage": "vol_target_shadow", "card": "volatility target and stop-in-sigmas",
     "switch": "VOL_TARGET_SIZING_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS, "sources": (_risk("vol_target_shadow"),)},
    {"stage": "stop_vol", "card": "volatility target and stop-in-sigmas",
     "switch": "VOL_TARGET_SIZING_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS, "sources": (_risk("stop_vol"),)},
    {"stage": "book_heat", "card": "book heat",
     "switch": "BOOK_HEAT_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS, "sources": (_risk("book_heat"),)},
    {"stage": "add_lot_r", "card": "per-lot R at an ADD's stop",
     "switch": "ADD_LOT_R_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _ADD_FILLS, "sources": (_stage("add_lot_r"),)},
    {"stage": "stop_above_market_guard", "card": "above-market stop guard",
     "switch": "STOP_GUARD_MODE", "expect_rows": "conditional",
     "sources": (_stage("stop_above_market_guard"),)},
    {"stage": "late_fill_expect", "card": "late BUY fill: wait for the cache",
     "switch": "LATE_FILL_EXPECT_MODE", "expect_rows": "conditional",
     "sources": (_stage("late_fill_expect"),)},
    {"stage": "late_fill_held_before", "card": "late fill of an OPEN on a held ticker",
     "switch": "LATE_FILL_HELD_BEFORE_MODE", "expect_rows": "conditional",
     "sources": (_stage("late_fill_held_before"),)},
    {"stage": "entry_basis", "card": "ADD entry basis vs the bot's fills",
     "switch": "ENTRY_BASIS_RECORD_ENABLED", "expect_rows": "conditional",
     "trigger": _ADD_FILLS, "sources": (_stage("entry_basis"),)},
    {"stage": "verdict_vote_shadow", "card": "self-consistency vote on routed entries",
     "switch": "VERDICT_VOTE_SHADOW_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS, "sources": (_stage("verdict_vote_shadow"),)},
    {"stage": "book_gap_risk", "card": "book overnight gap risk",
     "switch": "GAP_RISK_RECORD_ENABLED", "expect_rows": "always",
     "sources": (_stage("book_gap_risk"),)},
    {"stage": "book_deployment", "card": "book deployment vs the cash target",
     "switch": "BOOK_DEPLOYMENT_RECORD_ENABLED", "expect_rows": "conditional",
     "trigger": _stage("position_exit_frame"), "sources": (_stage("book_deployment"),)},
    {"stage": "feed_health", "card": "ingestion feed health",
     "switch": "FEED_HEALTH_ENABLED", "expect_rows": "always",
     "sources": (_stage("feed_health"),)},
    {"stage": "broker_sell_seen", "card": "broker sells the bot did not place",
     "switch": "BROKER_SELL_WATCH_ENABLED", "expect_rows": "conditional",
     "sources": (_stage("broker_sell_seen"),)},
    {"stage": "hwm_guard", "card": "unusable high-water mark",
     "switch": "HWM_GUARD_MODE", "expect_rows": "alarm", "sources": (_stage("hwm_guard"),)},
    {"stage": "risk_subcheck_fail", "card": "concentration sub-check failures",
     "switch": "RISK_SUBCHECK_FAIL_MODE", "expect_rows": "alarm",
     "sources": (_stage("risk_subcheck_fail"),)},
    {"stage": "entry_ref_age", "card": "entry price-reference age",
     "switch": None, "writer": ("execution.py", '"entry_ref_age"'),
     "expect_rows": "conditional",
     "trigger": (_ORD, "side = 'BUY' AND intent IN ('OPEN', 'ADD')"),
     "sources": (_stage("entry_ref_age"),)},
    {"stage": "stop_slippage_partial", "card": "partial stop fill slippage",
     "switch": None, "writer": ("tasks.py", '"stop_slippage_partial"'),
     "expect_rows": "conditional",
     "trigger": (_TRD, "side = 'SELL' AND reason LIKE 'Partial exit%%'"),
     "sources": (_stage("stop_slippage_partial"),)},
    {"stage": "fundamentals_changes", "card": "fundamentals refresh changes",
     "switch": None, "writer": ("tasks.py", '"fundamentals_changes"'),
     "expect_rows": "always",
     "sources": (_stage("fundamentals_changes"),)},
    # One row per trade ladder refresh (17:15 ET), record only.
    {"stage": "dcf_branch", "card": "DCF growth branch and step",
     "switch": None, "writer": ("agents/valuation_audit.py", 'DCF_BRANCH_STAGE = "dcf_branch"'),
     "expect_rows": "always",
     "sources": (_stage("dcf_branch"),)},
    {"stage": "source_yield_feeds", "card": "per-feed news yield",
     "switch": None, "writer": ("tasks.py", '"feed_drops"'),
     "expect_rows": "always",
     "sources": ((_ACT, "stage = 'source_yield' AND jsonb_typeof(extra_json->'feeds') = 'object'"),)},
    {"stage": "earnings_event_record", "card": "per-event earnings records",
     "switch": None, "writer": ("ingestion/earnings_events.py", '"earnings_event_record"'),
     "expect_rows": "conditional",
     "sources": (_stage("earnings_event_record"),)},
    {"stage": "order_quote", "card": "quote a market order was sent on",
     "switch": None, "writer": ("ibkr.py", '"order_quote"'),
     "expect_rows": "conditional",
     "trigger": (_ORD, "order_type = 'MARKET'"), "sources": (_stage("order_quote"),)},
    # Also written on the routing row (signals decision), so approvals are a
    # lower bound for its trigger: silent means approvals without the key.
    {"stage": "same_day_consensus", "card": "same-day consensus on OPEN/ADD routing",
     "switch": "SAME_DAY_CONSENSUS_RECORD_ENABLED", "expect_rows": "conditional",
     "trigger": _BUY_APPROVALS, "sources": (_risk("same_day_consensus"),)},
    # On every analyst verdict with a model reply (signals decision); the
    # trigger is those verdicts, so silent means replies without the key.
    {"stage": "source_mix", "card": "external sources vs the corroboration gate",
     "switch": None, "writer": ("agents/analyst.py", '"source_mix"'),
     "expect_rows": "conditional", "trigger": (_SIG, "decision ? 'model_output'"),
     "sources": ((_SIG, "decision ? 'source_mix'"),)},
    # Written by a side thread after each event/scan verdict, so a verdict
    # seconds old reads silent until its row lands.
    {"stage": "news_timeline", "card": "ticker news timeline (last 7 days)",
     "switch": "NEWS_TIMELINE_RECORD_ENABLED", "expect_rows": "conditional",
     "trigger": (_SIG, "decision ? 'model_output' "
                       "AND provenance->'trigger'->>'path' IN ('event', 'daily_scan')"),
     "sources": (_stage("news_timeline"),)},
    # On the plan observation (signals provenance.plan) of a verdict that
    # writes a NEW_PLAN or SEEDED plan; the trigger is those verdicts.
    {"stage": "plan_ledger_record", "card": "plan re-seed after an invalidation breach",
     "switch": "PLAN_RESEED_RECORD_ENABLED", "expect_rows": "conditional",
     "trigger": (_SIG, "provenance->'plan'->>'relation' IN ('NEW_PLAN', 'SEEDED')"),
     "sources": ((_SIG, "provenance->'plan' ? 'reseed'"),)},
    # On the provenance of every SCREENED_OUT row that has one (crew.py, from
    # agents/own_ticker.py); the trigger is those rows.
    {"stage": "own_ticker_rejected", "card": "own-ticker new fact screened out on other names",
     "switch": None, "writer": ("agents/own_ticker.py", '"own_ticker_rejected"'),
     "expect_rows": "conditional",
     "trigger": (_SIG, "signal = 'SCREENED_OUT' AND provenance IS NOT NULL"),
     "sources": ((_SIG, "provenance ? 'own_ticker_rejected'"),)},
    # Only events whose ticker has a held related symbol write a row, so no
    # trigger is named: 0 rows reads "no trigger yet".
    {"stage": "target_anchor", "card": "bull target vs the brief's targets",
     "switch": "TARGET_ANCHOR_RECORD_ENABLED", "expect_rows": "always",
     "sources": ((_SIG, "decision ? 'target_anchor'"),)},
    {"stage": "read_through", "card": "read-through to held names",
     "switch": "READ_THROUGH_RECORD_ENABLED", "expect_rows": "conditional",
     "sources": (_stage("read_through"),)},
    # On decision.catalyst_check of every full verdict; non-null only when the
    # catalyst names earnings with a day date and the dates read worked. The
    # trigger's "earnings" match is a subset of the writer's wording rule.
    {"stage": "earnings_claim", "card": "earnings claim vs the earnings calendars",
     "switch": None, "writer": ("earnings_claim.py", '"earnings_claim_unmatched"'),
     "expect_rows": "conditional",
     "trigger": (_SIG, "decision->'model_output'->'catalyst'->>'event' ~* 'earnings' "
                       "AND decision->'model_output'->'catalyst'->>'date' "
                       "~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'"),
     "sources": ((_SIG, "decision->'catalyst_check'->>'earnings_claim_unmatched' IS NOT NULL"),)},
    # On decision.catalyst_check of every full verdict, from the same reads.
    {"stage": "catalyst_kind", "card": "catalyst date vs the verdict day and the calendars",
     "switch": None, "writer": ("earnings_claim.py", '"catalyst_kind"'),
     "expect_rows": "always",
     "sources": ((_SIG, "decision->'catalyst_check' ? 'catalyst_kind'"),)},
)


# ---------------------------------------------------------------- pure parts

def _entry_funnel():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_funnel.py")
    spec = importlib.util.spec_from_file_location("entry_funnel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def session_window(now: datetime, n: int = SESSIONS) -> tuple[list[date], datetime]:
    """The last n NYSE sessions (today counts when it is one, so the session
    in progress is included) and the UTC start of the first one's rows
    (entry_funnel.window: from the day after the session before it)."""
    ef = _entry_funnel()
    d = now.astimezone(NY).date()
    while not ef._is_session(d):
        d -= timedelta(days=1)
    sessions = ef.sessions_until(d, n)
    start, _ = ef.window(sessions)
    return sessions, start


def parse_load(text: str | None) -> datetime:
    """'YYYY-MM-DD' (00:00) or 'YYYY-MM-DD HH:MM', ET; default BATCH_LOAD_DEFAULT."""
    text = (text or BATCH_LOAD_DEFAULT).strip()
    if len(text) <= 10:
        return datetime.combine(date.fromisoformat(text), time(0), tzinfo=NY)
    return datetime.fromisoformat(text).replace(tzinfo=NY)


def switch_state(cfg, name: str | None, writer_present: bool = True) -> dict:
    """{"value": the value as loaded (str), "records": bool}. A True bool or a
    mode other than "off" records; a name config lacks is "absent"; no
    switch (None) is "no switch" and records, or "absent" when the clone
    lacks its writer (writer_present False)."""
    if name is None:
        if not writer_present:
            return {"value": "absent", "records": False}
        return {"value": "no switch", "records": True}
    if not hasattr(cfg, name):
        return {"value": "absent", "records": False}
    v = getattr(cfg, name)
    if isinstance(v, bool):
        return {"value": str(v), "records": v}
    if isinstance(v, str):
        return {"value": v, "records": v.strip().lower() != "off"}
    return {"value": repr(v), "records": bool(v)}


def _aware(t):
    if t is None:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def summarize(stage: dict, per_source: list[dict], load: datetime, window_start: datetime,
              switch: dict, trigger_times: list | None = None) -> dict:
    """One stage's line from its sources' rows. per_source: [{"times": [row
    times since min(load, window_start)], "newest": max time or None}].
    trigger_times: the stage's trigger rows over the same read (None when it
    names no trigger). The silent and alarm flags follow expect_rows (module
    docstring)."""
    load_u, win_u = load.astimezone(timezone.utc), window_start.astimezone(timezone.utc)
    span_u = max(load_u, win_u)
    times = [_aware(t) for s in per_source for t in s["times"] if t is not None]
    newest = [_aware(s["newest"]) for s in per_source if s.get("newest") is not None]
    n_window = sum(1 for t in times if t >= win_u)
    expect = stage.get("expect_rows", "always")
    n_trig = (sum(1 for t in trigger_times if t is not None and _aware(t) >= span_u)
              if trigger_times is not None else None)
    if expect == "always":
        silent = bool(switch["records"] and n_window == 0)
    elif expect == "conditional":
        silent = bool(switch["records"] and n_trig
                      and not any(t >= span_u for t in times))
    else:
        silent = False
    out = {"stage": stage["stage"], "card": stage["card"], "switch": stage["switch"],
           "switch_value": switch["value"], "records": switch["records"],
           "expect_rows": expect,
           "rows_since_load": sum(1 for t in times if t >= load_u),
           "rows_last_sessions": n_window,
           "trigger_rows": n_trig,
           "newest": max(newest).astimezone(NY).isoformat(timespec="minutes") if newest else None,
           "silent": silent,
           "alarm": bool(expect == "alarm" and n_window > 0),
           "no_trigger_yet": bool(expect == "conditional" and not silent and n_window == 0
                                  and not n_trig)}
    return out


def writer_present(root: str, writer) -> bool:
    """True when the clone at root has the stage's writer: writer is (file
    relative to root, token) and the file contains the token. No writer
    named: True. An unreadable file: False."""
    if not writer:
        return True
    try:
        with open(os.path.join(root, writer[0]), encoding="utf-8") as fh:
            return writer[1] in fh.read()
    except OSError:
        return False


def build(results: dict, cfg, load: datetime, window_start: datetime,
          root: str | None = None) -> list[dict]:
    """results: {stage: per_source list} (fetch). A stage missing from
    results (its read failed) gets read_error and counts of None. root:
    the clone whose writers a no-switch stage is checked against (None:
    not checked, every writer taken as present)."""
    lines = []
    triggers = results.get(TRIGGERS_KEY) or {}
    for st in STAGES:
        present = True if root is None else writer_present(root, st.get("writer"))
        sw = switch_state(cfg, st["switch"], present)
        got = results.get(st["stage"])
        if isinstance(got, Exception) or got is None:
            lines.append({"stage": st["stage"], "card": st["card"], "switch": st["switch"],
                          "switch_value": sw["value"], "records": sw["records"],
                          "expect_rows": st.get("expect_rows", "always"),
                          "rows_since_load": None, "rows_last_sessions": None,
                          "trigger_rows": None, "newest": None,
                          "silent": None, "alarm": None, "no_trigger_yet": None,
                          "read_error": type(got).__name__ if got is not None else "not read"})
            continue
        trig = triggers.get(st["stage"]) if st.get("trigger") else None
        lines.append(summarize(st, got, load, window_start, sw,
                               trig if st.get("trigger") else None))
    return lines


def render_text(lines: list[dict], meta: dict) -> str:
    out = [f"Shadow registry check — {meta['env']}, clone {meta['clone']}",
           f"Generated {meta['generated']} ET. Load date {meta['load']} ET. "
           f"Last {len(meta['sessions'])} sessions: {meta['sessions'][0]} .. {meta['sessions'][-1]} "
           f"(rows since {meta['window_start']} ET).",
           "silent: an always stage with no row in those sessions, or a conditional stage "
           "whose trigger had rows since the later of the load and the first session, with "
           "none of its own. alarm: an alarm stage with rows. Presence flags only.",
           ""]
    head = ("stage", "expect", "switch", "since load", f"last {len(meta['sessions'])}",
            "trigger", "newest (ET)", "")
    rows = []
    for ln in lines:
        if ln.get("read_error"):
            flag = f"read error: {ln['read_error']}"
        elif ln.get("silent"):
            flag = "silent"
        elif ln.get("alarm"):
            flag = "alarm"
        elif ln.get("no_trigger_yet"):
            flag = "no trigger yet"
        else:
            flag = ""
        sw = (f"{ln['switch']}={ln['switch_value']}" if ln["switch"]
              else ("no switch" if ln["switch_value"] == "no switch"
                    else "no switch, writer absent"))
        rows.append((ln["stage"], ln.get("expect_rows") or "always",
                     sw + ("" if ln["records"] else " (not recording)"),
                     "-" if ln["rows_since_load"] is None else str(ln["rows_since_load"]),
                     "-" if ln["rows_last_sessions"] is None else str(ln["rows_last_sessions"]),
                     "-" if ln.get("trigger_rows") is None else str(ln["trigger_rows"]),
                     ln["newest"] or "none", flag))
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
    n_silent = sum(1 for ln in lines if ln.get("silent"))
    n_alarm = sum(1 for ln in lines if ln.get("alarm"))
    out += ["", f"{n_silent} silent, {n_alarm} alarm of {len(lines)} stages."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- I/O

def fetch(conn, env: str, since: datetime) -> dict:
    """{stage: [{"times": [...], "newest": t}, ...]} for every STAGES entry,
    or {stage: exception} when its read failed, plus out[TRIGGERS_KEY] =
    {stage: [trigger row times since `since`]} for stages that name a
    trigger. One READ ONLY transaction, rolled back. Never writes."""
    out: dict = {TRIGGERS_KEY: {}}
    trig_rows: dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            for st in STAGES:
                per = []
                try:
                    for table, cond in st["sources"]:
                        rows_sql, max_sql = _SQL_BASE[table]
                        args = {"env": env, "since": since}
                        cur.execute(rows_sql.format(cond=cond), args)
                        times = [r[0] for r in cur.fetchall()]
                        cur.execute(max_sql.format(cond=cond), args)
                        row = cur.fetchone()
                        per.append({"times": times, "newest": row[0] if row else None})
                    trig = st.get("trigger")
                    if trig:
                        if trig not in trig_rows:      # one read per trigger, shared
                            table, cond = trig
                            cur.execute(_SQL_BASE[table][0].format(cond=cond),
                                        {"env": env, "since": since})
                            trig_rows[trig] = [r[0] for r in cur.fetchall()]
                        out[TRIGGERS_KEY][st["stage"]] = list(trig_rows[trig])
                    out[st["stage"]] = per
                except Exception as e:
                    out[st["stage"]] = e
                    break           # a failed statement aborts the transaction
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
    return out


def _git_head(root: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rows per record in the shadow registry")
    ap.add_argument("--load-date", help="ET 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'; "
                                        f"default {BATCH_LOAD_DEFAULT}")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    import config
    import db
    now = datetime.now(timezone.utc)
    load = parse_load(args.load_date)
    sessions, win_start = session_window(now)
    conn = db._get_conn()
    try:
        results = fetch(conn, config.TRADING_ENV, min(load.astimezone(timezone.utc), win_start))
    finally:
        db._put_conn(conn)
    lines = build(results, config, load, win_start, ROOT)
    meta = {"env": config.TRADING_ENV, "clone": _git_head(ROOT),
            "generated": now.astimezone(NY).isoformat(timespec="minutes"),
            "load": load.isoformat(timespec="minutes"),
            "sessions": [d.isoformat() for d in sessions],
            "window_start": win_start.astimezone(NY).isoformat(timespec="minutes")}
    if args.json:
        print(json.dumps({"meta": meta, "stages": lines}, indent=2))
    else:
        print(render_text(lines, meta), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
