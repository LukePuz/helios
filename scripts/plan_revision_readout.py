"""Plan revision readout: how the shadow trade plans are created and
revised, per ticker, per day and per day and ticker, from trade_plans and
the plan record on each analyst verdict (signals.provenance.plan) (audit
2026-11-27 08:53 #3). Record-only engineering readout; counts, percent,
ATR multiples, hours, days and ids only: no price, dollar figure or share
count is printed.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/plan_revision_readout.py [--since 2026-11-09] [--until 2026-11-28] \
        [--env live] [--out FILE] [--json FILE]

Window: [--since, --until) as New York dates. Default since 2026-11-09, the
first live tree whose plan record carries revise_triggers (a readout
default, not a sample choice). Plans are the trade_plans rows created in the
window in --env or with a NULL env; each one's replaced plan (replaces_id) is
read whenever it was created. Verdicts are the signals rows in the window
with a provenance.plan record, in --env or with a NULL env. A plan's day is
the New York date of created_at; a verdict's is that of signals.timestamp.

Plans (per key):
  - created, and revisions (plans with a replaces_id);
  - lifetime, hours from created_at to closed_at, for the plans that are
    superseded (insert_plan closes the replaced plan when its revision is
    written): n / median / min / max;
  - horizon span, days from the created New York date to horizon_end:
    n / median / min / max;
  - catalyst_date against the plan's created New York date: "past" (before
    it), "same week" (that date or later, in the same ISO week, Monday to
    Sunday), "later" (a later week), "none". A month-only catalyst is
    stored as the month's last day (plans._catalyst_date). Whether a date
    is the ticker's own catalyst is not classified here (item 37, Luke's).
Verdicts (per key): n; relation counts (provenance.plan.relation);
revise_triggers by kind (target / invalidation / catalyst_date, a REVISE
can have several; "missing" when a REVISE has no revise_triggers key);
catalyst_month_only_pair true / false / missing on REVISE verdicts.
Revision moves: for each plan in the window with a replaces_id, the move of
t1, add_trigger and structural_stop from the replaced plan to it, in percent
of the replaced plan's level ((new - old) / old x 100) and in ATRs
((new - old) / ladder_atr, the ATR the REVISE verdict whose new_plan_id is
this plan recorded); a level missing on either side, or a non-positive one,
gives none, and the ATR column is none without a recorded ATR. Listed per
revision by id, and summarised per ticker (n / median / min / max).

Code-level flips (plan-identity-design.md, the R list): compile_plan picks
t1 and add_trigger from the first ladder rung at least 1 ATR above the
verdict's brief price, so a plan revised on an unchanged ladder can land on
another rung. Each revision pair is labelled by ladder build
(ladder_computed_at): "same build" (both set and equal: the rung set was
fixed, so a level change is a re-pick), "new build" (both set, different:
a change can't be split into a re-pick and a rebuild, as past rung sets are
not stored) or "build unknown". A level counts as changed when it moved by
FLIP_TOL_PCT percent or more of the prior level (a rounding tolerance for
cent-rounded levels, not a threshold on the plan). The ATR per plan is the
REVISE verdict's ladder_atr where recorded, else derived as (add_trigger -
t1) / RUNG_BAND_ATR_FRACTION: trade_ladder._merge_rungs sets band_high =
round(anchor + atr x 0.25, 2) and the rung price round(anchor, 2) from the
ATR it stores on the ladder row, and compile_plan takes t1 and add_trigger
both from the same rung; derived and recorded are compared where both exist.
t1_edge_margin_atr = (t1 - entry_price) / ATR - 1, how far beyond the 1-ATR
cutoff the chosen rung sat. It is computed only where entry_price is the
price compile_plan used: source "analyst" with levels_source "ladder"
(plans._observe passes one brief_price to plan_from_verdict, whose
entry_price it is, and to _with_levels). Other plans are "basis unknown"
(a held_seed plan's entry is the position's stop-record entry, and it has
no compiled levels). On a same-build t1 flip, the replaced t1's distance
above the new entry in ATRs is given too (below 1: that rung fell inside the
cutoff). Flips are a description, not a revise rule.

Nothing here is a rule or threshold; what any of it means for the plan
logic is Luke's. SUPABASE_PORT must be 6543 (the transaction pooler); one
read-only transaction with a statement timeout, rolled back. It does not
read between 04:50 and 05:12 ET (the stack's start). db is imported from
the working directory when it holds a db.py (the live clone, with its .env),
else from beside this script; with no SUPABASE_HOST after the import the
readout stops before connecting. Imports: db and the standard library. Not
loaded by weekly_readouts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
QUIET_START, QUIET_END = time(4, 50), time(5, 12)   # no DB reads at the stack's start
DEFAULT_SINCE = "2026-11-09"
LEVELS = ("t1", "add_trigger", "structural_stop")
RELATIONS = ("NEW_PLAN", "SEEDED", "FOLLOW", "DIVERGE", "REVISE", "INVALIDATE", "NO_PLAN", "ERROR")
TRIGGERS = ("target", "invalidation", "catalyst_date")
CATALYST_BINS = ("past", "same week", "later", "none")
FLIP_TOL_PCT = 0.01             # rounding tolerance for "changed", in percent of the level
RUNG_BAND_ATR_FRACTION = 0.25   # trade_ladder.RUNG_BAND_ATR_FRACTION (a test pins it)
FLIP_KINDS = ("same-build flip", "same-build unchanged", "same-build no levels",
              "new build changed", "new build unchanged", "build unknown")
FLIP_NOTE = ("Flips describe code-level re-picks; they are not a revise rule. Which revise rule, "
             "plan identity or add level applies (options A/B/C) and any X, Y or k are Luke's. "
             f"Changed means a move of {FLIP_TOL_PCT}% or more of the prior level, the readout's "
             "rounding tolerance. Edge margins use entry_price only where it is the price "
             "compile_plan used (analyst plans with ladder levels); the rest are basis unknown.")
NOTE = ("Record-only: no rule or threshold is applied; whether a catalyst date is the ticker's "
        "own is not classified (item 37, Luke's).")

_PLANS_SQL = """
WITH w AS (
    SELECT tp.* FROM trade_plans tp
    WHERE tp.created_at >= %(start)s AND tp.created_at < %(end)s
      AND (tp.trading_env = %(env)s OR tp.trading_env IS NULL))
SELECT p.id, p.ticker, p.status, p.source, p.created_at, p.closed_at, p.revision,
       p.replaces_id, p.catalyst_date, p.horizon_end, p.t1, p.add_trigger, p.structural_stop,
       p.entry_price, p.levels_source, p.ladder_computed_at, true
FROM w p
UNION ALL
SELECT r.id, r.ticker, r.status, r.source, r.created_at, r.closed_at, r.revision,
       r.replaces_id, r.catalyst_date, r.horizon_end, r.t1, r.add_trigger, r.structural_stop,
       r.entry_price, r.levels_source, r.ladder_computed_at, false
FROM trade_plans r
WHERE r.id IN (SELECT replaces_id FROM w WHERE replaces_id IS NOT NULL)
  AND r.id NOT IN (SELECT id FROM w)
"""

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol,
       s.provenance->'plan'->>'relation',
       s.provenance->'plan'->'revise_triggers',
       s.provenance->'plan'->'catalyst_month_only_pair',
       s.provenance->'plan'->>'new_plan_id',
       s.provenance->'plan'->>'ladder_atr'
FROM signals s
WHERE s.provenance ? 'plan' AND s.timestamp >= %(start)s AND s.timestamp < %(end)s
  AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
ORDER BY s.timestamp, s.id
"""


def db_root(cwd: str | None = None) -> str:
    """The directory db is imported from: cwd when it holds a db.py, else
    this script's checkout."""
    cwd = os.getcwd() if cwd is None else cwd
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    """True between 04:50 and 05:12 ET, when the readout doesn't read."""
    t = now.astimezone(NY).time()
    return QUIET_START <= t < QUIET_END


def _num(v) -> float | None:
    """A finite float, or None."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _day(ts) -> str | None:
    return ts.astimezone(NY).date().isoformat() if isinstance(ts, datetime) else None


def catalyst_bin(catalyst, created_day: date | None) -> str:
    """past / same week / later / none for a catalyst date against the
    plan's created New York date (module docstring). Pure."""
    if not isinstance(catalyst, date) or created_day is None:
        return "none"
    if isinstance(catalyst, datetime):
        catalyst = catalyst.date()
    if catalyst < created_day:
        return "past"
    return "same week" if catalyst.isocalendar()[:2] == created_day.isocalendar()[:2] else "later"


def move(old, new, atr) -> tuple[float | None, float | None]:
    """(percent of old, ATR multiple) for a level's move; None when either
    level is missing or not positive, and None ATR without a positive ATR.
    Pure."""
    o, n, a = _num(old), _num(new), _num(atr)
    if o is None or n is None or o <= 0 or n <= 0:
        return None, None
    return (round((n - o) / o * 100, 2),
            round((n - o) / a, 3) if a is not None and a > 0 else None)


def basis(lv: dict) -> str:
    """"match" when the plan's entry_price is the price compile_plan used
    (an analyst plan with ladder levels), else "basis unknown". Pure."""
    return ("match" if lv.get("source") == "analyst" and lv.get("levels_source") == "ladder"
            else "basis unknown")


def plan_atr(lv: dict, recorded) -> tuple[float | None, str]:
    """(ATR, "recorded" / "derived" / "none"): the recorded ladder_atr when
    positive, else (add_trigger - t1) / RUNG_BAND_ATR_FRACTION. Pure."""
    a = _num(recorded)
    if a is not None and a > 0:
        return a, "recorded"
    d = derived_atr(lv)
    return (d, "derived") if d is not None else (None, "none")


def derived_atr(lv: dict) -> float | None:
    t1, add = _num(lv.get("t1")), _num(lv.get("add_trigger"))
    if t1 is None or add is None or t1 <= 0 or add <= t1:
        return None
    return (add - t1) / RUNG_BAND_ATR_FRACTION


def build_plans(raw) -> tuple[list[dict], dict]:
    """(window plan records, {id: raw level dict} for every row read). The
    levels stay here: records carry percent, hours, days and ids only."""
    levels, recs = {}, []
    for (pid, tk, status, source, created, closed, revision, replaces, cat, horizon,
         t1, add, stop, entry, lsrc, built, in_window) in raw:
        levels[pid] = {"t1": t1, "add_trigger": add, "structural_stop": stop, "ticker": tk,
                       "entry": entry, "source": source, "levels_source": lsrc,
                       "build": built.astimezone(NY).strftime("%Y-%m-%d %H:%M")
                       if isinstance(built, datetime) else None}
        if not in_window:
            continue
        cday = created.astimezone(NY).date() if isinstance(created, datetime) else None
        life = ((closed - created).total_seconds() / 3600
                if status == "superseded" and isinstance(closed, datetime)
                and isinstance(created, datetime) else None)
        hz = horizon.date() if isinstance(horizon, datetime) else horizon
        recs.append({"plan_id": pid, "ticker": tk or "none", "day": _day(created) or "none",
                     "status": status, "source": source, "revision": _int(revision),
                     "replaces_id": replaces,
                     "lifetime_h": round(life, 2) if life is not None else None,
                     "horizon_days": (hz - cday).days if isinstance(hz, date) and cday else None,
                     "catalyst": catalyst_bin(cat, cday),
                     "build": levels[pid]["build"], "levels_source": lsrc,
                     "basis": basis(levels[pid])})
    return recs, levels


def build_verdicts(raw) -> list[dict]:
    """One record per verdict with a plan record. Pure."""
    out = []
    for sid, ts, sym, rel, trig, pair, new_id, atr in raw:
        trig, pair = _json(trig), _json(pair)
        out.append({"signal_id": sid, "day": _day(ts) or "none", "ticker": sym or "none",
                    "relation": rel or "none",
                    "triggers": ([t for t in trig if isinstance(t, str)]
                                 if isinstance(trig, list) else None),
                    "pair": pair if isinstance(pair, bool) else None,
                    "new_plan_id": _int(new_id), "ladder_atr": _num(atr)})
    return out


def build_moves(plans: list[dict], levels: dict, verdicts: list[dict]) -> list[dict]:
    """One record per window plan with a replaces_id: the level moves in
    percent and ATRs, ids only. Pure."""
    atr = {}
    for v in verdicts:
        if v["relation"] == "REVISE" and v["new_plan_id"] is not None:
            atr.setdefault(v["new_plan_id"], v["ladder_atr"])
    out = []
    for p in plans:
        if p["replaces_id"] is None:
            continue
        old, new = levels.get(p["replaces_id"]), levels.get(p["plan_id"], {})
        a = atr.get(p["plan_id"])
        rec = {"plan_id": p["plan_id"], "replaces_id": p["replaces_id"], "ticker": p["ticker"],
               "day": p["day"], "revision": p["revision"], "prior_found": old is not None,
               "atr_recorded": a is not None}
        for lv in LEVELS:
            pct, x = move((old or {}).get(lv), new.get(lv), a) if old else (None, None)
            rec[f"{lv}_pct"], rec[f"{lv}_atr"] = pct, x
        out.append(rec)
    return out


def _changed(old, new) -> str:
    """changed / unchanged / missing: the move, unrounded, against
    FLIP_TOL_PCT percent of the old level. Pure."""
    o, n = _num(old), _num(new)
    if o is None or n is None or o <= 0 or n <= 0:
        return "missing"
    return "changed" if abs(n - o) / o * 100 >= FLIP_TOL_PCT else "unchanged"


def flip_kind(build_old, build_new, changes: dict) -> str:
    """The FLIP_KINDS label of a revision pair. Pure."""
    if not build_old or not build_new:
        return "build unknown"
    comparable = [c for c in changes.values() if c != "missing"]
    if build_old == build_new:
        if not comparable:
            return "same-build no levels"
        return "same-build flip" if "changed" in comparable else "same-build unchanged"
    return "new build changed" if "changed" in comparable else "new build unchanged"


def _trigger_label(v) -> str:
    if v is None:
        return "no verdict"
    if v["triggers"] is None:
        return "missing"
    t = [x for x in TRIGGERS if x in v["triggers"]]
    return "+".join(t) if t else "none listed"


def build_flips(plans: list[dict], levels: dict, verdicts: list[dict]) -> dict:
    """Per-plan ATR, basis and edge margin, and per-revision flip labels;
    percent, ATR multiples and ids only. Pure."""
    rev = {}
    for v in verdicts:
        if v["relation"] == "REVISE" and v["new_plan_id"] is not None:
            rev.setdefault(v["new_plan_id"], v)
    per_plan, pairs = [], []
    for p in plans:
        lv = levels.get(p["plan_id"], {})
        v = rev.get(p["plan_id"])
        atr, src = plan_atr(lv, v["ladder_atr"] if v else None)
        d = derived_atr(lv)
        agree = (round((d / atr - 1) * 100, 2) if src == "recorded" and d is not None else None)
        t1, entry = _num(lv.get("t1")), _num(lv.get("entry"))
        margin = (round((t1 - entry) / atr - 1, 3)
                  if p["basis"] == "match" and t1 and entry and entry > 0 and atr else None)
        per_plan.append({"plan_id": p["plan_id"], "ticker": p["ticker"], "day": p["day"],
                         "build": p["build"], "basis": p["basis"], "atr_source": src,
                         "derived_vs_recorded_pct": agree, "t1_edge_margin_atr": margin})
        if p["replaces_id"] is None:
            continue
        old = levels.get(p["replaces_id"])
        changes = {l: (_changed(old.get(l), lv.get(l)) if old else "missing") for l in LEVELS}
        kind = flip_kind(old.get("build") if old else None, lv.get("build"), changes)
        old_t1 = _num(old.get("t1")) if old else None
        old_from_new = (round((old_t1 - entry) / atr, 3)
                        if kind == "same-build flip" and changes["t1"] == "changed"
                        and p["basis"] == "match" and old_t1 and entry and atr else None)
        pairs.append({"plan_id": p["plan_id"], "replaces_id": p["replaces_id"],
                      "ticker": p["ticker"], "day": p["day"], "build": lv.get("build"),
                      "kind": kind, "changes": changes, "triggers": _trigger_label(v),
                      "atr_source": src, "t1_edge_margin_atr": margin,
                      "old_t1_from_new_entry_atr": old_from_new})
    return {"plans": per_plan, "pairs": pairs}


def _stats(vals) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "0 | - | - | -"
    return f"{len(v)} | {statistics.median(v):.2f} | {min(v):.2f} | {max(v):.2f}"


def _keys(recs, key) -> dict:
    g: dict = {}
    for r in recs:
        g.setdefault(key(r), []).append(r)
    return g


_KEYINGS = (("Per day", ["day"], lambda r: r["day"]),
            ("Per ticker", ["ticker"], lambda r: r["ticker"]),
            ("Per day and ticker", ["day", "ticker"], lambda r: (r["day"], r["ticker"])))


def _row(k) -> str:
    return " | ".join(k if isinstance(k, tuple) else (k,))


def _plan_table(title, cols, groups) -> list[str]:
    head = ("created | revisions | " + " | ".join(CATALYST_BINS) +
            " | life n | life med h | life min h | life max h"
            " | hz n | hz med d | hz min d | hz max d")
    out = [f"### Plans: {title}", "", "| " + " | ".join(cols) + " | " + head + " |",
           "|" + "---|" * len(cols) + "---:|" * (2 + len(CATALYST_BINS) + 8)]
    for k in sorted(groups):
        g = groups[k]
        cb = Counter(r["catalyst"] for r in g)
        out.append(f"| {_row(k)} | {len(g)} | {sum(r['replaces_id'] is not None for r in g)} | "
                   + " | ".join(str(cb.get(b, 0)) for b in CATALYST_BINS) + " | "
                   + _stats(r["lifetime_h"] for r in g) + " | "
                   + _stats(r["horizon_days"] for r in g) + " |")
    return out + [""]


def _verdict_table(title, cols, groups, other: list[str]) -> list[str]:
    rels = list(RELATIONS) + other
    head = (" | ".join(rels) + " | trig " + " | trig ".join(TRIGGERS) +
            " | trig missing | pair T | pair F | pair missing")
    out = [f"### Verdicts: {title}", "", "| " + " | ".join(cols) + " | n | " + head + " |",
           "|" + "---|" * len(cols) + "---:|" * (1 + len(rels) + len(TRIGGERS) + 4)]
    for k in sorted(groups):
        g = groups[k]
        rc = Counter(r["relation"] for r in g)
        rev = [r for r in g if r["relation"] == "REVISE"]
        tc = Counter(t for r in rev for t in (r["triggers"] or []))
        out.append(f"| {_row(k)} | {len(g)} | " + " | ".join(str(rc.get(x, 0)) for x in rels)
                   + " | " + " | ".join(str(tc.get(t, 0)) for t in TRIGGERS)
                   + f" | {sum(r['triggers'] is None for r in rev)}"
                   + f" | {sum(r['pair'] is True for r in rev)}"
                   + f" | {sum(r['pair'] is False for r in rev)}"
                   + f" | {sum(r['pair'] is None for r in rev)} |")
    return out + [""]


def _f(v, nd) -> str:
    return "none" if v is None else f"{v:.{nd}f}"


def render_flips(flips: dict) -> list[str]:
    pp, pairs = flips["plans"], flips["pairs"]
    out = ["## Code-level flips", "", FLIP_NOTE, "",
           f"Revision pairs: {len(pairs)} (" + ", ".join(
               f"{k} {sum(x['kind'] == k for x in pairs)}" for k in FLIP_KINDS) + "). "
           "Plans by basis: " + ", ".join(
               f"{b} {sum(x['basis'] == b for x in pp)}" for b in ("match", "basis unknown"))
           + "; ATR source: " + ", ".join(
               f"{s} {sum(x['atr_source'] == s for x in pp)}"
               for s in ("recorded", "derived", "none")) + ".", "",
           "Derived vs recorded ATR, percent (plans with both): n | median | min | max: "
           + _stats(x["derived_vs_recorded_pct"] for x in pp) + ".", ""]
    head = (" | ".join(FLIP_KINDS) + " | " + " | ".join(f"same-build {l} changed" for l in LEVELS))
    for title, cols, key in _KEYINGS:
        g = _keys(pairs, key)
        out += [f"### Flips: {title}", "", "| " + " | ".join(cols) + " | revisions | " + head + " |",
                "|" + "---|" * len(cols) + "---:|" * (1 + len(FLIP_KINDS) + len(LEVELS))]
        for k in sorted(g):
            x = g[k]
            same = [r for r in x if r["kind"].startswith("same-build")]
            out.append(f"| {_row(k)} | {len(x)} | "
                       + " | ".join(str(sum(r["kind"] == f for r in x)) for f in FLIP_KINDS)
                       + " | " + " | ".join(str(sum(r["changes"][l] == "changed" for r in same))
                                            for l in LEVELS) + " |")
        out.append("")
    builds = _keys(pp, lambda r: (r["ticker"], r["build"] or "none"))
    out += ["### Per ticker and ladder build (ladder_computed_at, ET)", "",
            "| ticker | build | plans | distinct t1 | same-build flips |",
            "|---|---|---:|---:|---:|"]
    pf = Counter((r["ticker"], r["build"] or "none") for r in pairs if r["kind"] == "same-build flip")
    for k in sorted(builds):
        out.append(f"| {k[0]} | {k[1]} | {len(builds[k])} | "
                   f"{flips.get('distinct_t1', {}).get(k, 0)} | {pf.get(k, 0)} |")
    out.append("")
    tc = Counter((r["triggers"], r["kind"]) for r in pairs)
    labels = sorted({t for t, _ in tc})
    out += ["### revise_triggers by flip kind (the REVISE verdict that wrote the plan)", "",
            "| triggers | " + " | ".join(FLIP_KINDS) + " | all |",
            "|---|" + "---:|" * (len(FLIP_KINDS) + 1)]
    for t in labels:
        cells = [tc.get((t, k), 0) for k in FLIP_KINDS]
        out.append(f"| {t} | " + " | ".join(str(c) for c in cells) + f" | {sum(cells)} |")
    out.append("")
    out += ["### t1 edge margin, ATRs beyond the 1-ATR cutoff (basis match only): "
            "n | median | min | max", "",
            "| ticker | margin n | margin med | margin min | margin max | basis unknown | "
            "replaced t1 above new entry, ATR (same-build t1 flips) n | med | min | max | below 1 |",
            "|---|" + "---:|" * 10]
    pk, rk = _keys(pp, lambda r: r["ticker"]), _keys(pairs, lambda r: r["ticker"])
    for tk in sorted(pk):
        o = [r["old_t1_from_new_entry_atr"] for r in rk.get(tk, [])]
        out.append(f"| {tk} | {_stats(r['t1_edge_margin_atr'] for r in pk[tk])} | "
                   f"{sum(r['basis'] == 'basis unknown' for r in pk[tk])} | {_stats(o)} | "
                   f"{sum(x is not None and x < 1 for x in o)} |")
    out += ["", "### Flips, per revision", "",
            "| day | ticker | plan id | replaces id | build | kind | t1 | add_trigger | "
            "structural_stop | triggers | ATR source | t1 edge margin | replaced t1 vs new entry |",
            "|---|---|---:|---:|---|---|---|---|---|---|---|---:|---:|"]
    for r in sorted(pairs, key=lambda r: (r["day"], r["plan_id"])):
        out.append(f"| {r['day']} | {r['ticker']} | {r['plan_id']} | {r['replaces_id']} | "
                   f"{r['build'] or 'none'} | {r['kind']} | "
                   + " | ".join(r["changes"][l] for l in LEVELS)
                   + f" | {r['triggers']} | {r['atr_source']} | {_f(r['t1_edge_margin_atr'], 3)} | "
                   f"{_f(r['old_t1_from_new_entry_atr'], 3)} |")
    return out + [""]


def distinct_t1(plans: list[dict], levels: dict) -> dict:
    """(ticker, build) -> the number of distinct t1 values (cents) among the
    window plans. Pure; only the count leaves."""
    seen: dict = {}
    for p in plans:
        t1 = _num(levels.get(p["plan_id"], {}).get("t1"))
        s = seen.setdefault((p["ticker"], p["build"] or "none"), set())
        if t1 is not None:
            s.add(round(t1, 2))
    return {k: len(v) for k, v in seen.items()}


def render(plans, verdicts, moves, env: str, since: date, until: date, head: str,
           flips: dict | None = None) -> str:
    other = sorted({r["relation"] for r in verdicts} - set(RELATIONS))
    lines = ["# Plan revision readout", "",
             f"Env {env}; {since.isoformat()} to {until.isoformat()} (New York dates, end "
             f"exclusive); code {head}. Counts, percent, ATR multiples, hours, days and ids "
             "only; no price or level is printed.", "", NOTE, "",
             f"Plans created: {len(plans)} (revisions {sum(p['replaces_id'] is not None for p in plans)}, "
             f"superseded {sum(p['status'] == 'superseded' for p in plans)}). Verdicts with a plan "
             f"record: {len(verdicts)} (REVISE {sum(v['relation'] == 'REVISE' for v in verdicts)}). "
             f"Revisions with the replaced plan read: {sum(m['prior_found'] for m in moves)} of "
             f"{len(moves)}; with a recorded ATR: {sum(m['atr_recorded'] for m in moves)}.", ""]
    for title, cols, key in _KEYINGS:
        lines += _plan_table(title, cols, _keys(plans, key))
    for title, cols, key in _KEYINGS:
        lines += _verdict_table(title, cols, _keys(verdicts, key), other)
    lines += ["### Revision moves per ticker (percent of the prior level; ATRs): "
              "n | median | min | max", "",
              "| ticker | level | % n | % med | % min | % max | ATR n | ATR med | ATR min | ATR max |",
              "|---|---|" + "---:|" * 8]
    for tk, g in sorted(_keys(moves, lambda m: m["ticker"]).items()):
        for lv in LEVELS:
            lines.append(f"| {tk} | {lv} | {_stats(m[f'{lv}_pct'] for m in g)} | "
                         f"{_stats(m[f'{lv}_atr'] for m in g)} |")
    lines += ["", "### Revision moves, per revision", "",
              "| day | ticker | plan id | replaces id | revision | "
              + " | ".join(f"{lv} % | {lv} ATR" for lv in LEVELS) + " |",
              "|---|---|---:|---:|---:|" + "---:|" * (2 * len(LEVELS))]
    for m in sorted(moves, key=lambda m: (m["day"], m["plan_id"])):
        lines.append(f"| {m['day']} | {m['ticker']} | {m['plan_id']} | {m['replaces_id']} | "
                     f"{m['revision'] if m['revision'] is not None else 'none'} | "
                     + " | ".join(f"{_f(m[f'{lv}_pct'], 2)} | {_f(m[f'{lv}_atr'], 3)}"
                                  for lv in LEVELS) + " |")
    lines.append("")
    if flips is not None:
        lines += render_flips(flips)
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the records (no prices or levels) here")
    args = ap.parse_args(argv)
    now = now or datetime.now(timezone.utc)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    if in_quiet_window(now):
        print("04:50-05:12 ET is the stack's start; not reading.", file=sys.stderr)
        return 2
    try:
        since = date.fromisoformat(args.since)
        until = (date.fromisoformat(args.until) if args.until
                 else now.astimezone(NY).date() + timedelta(days=1))
    except ValueError:
        print("--since / --until must be YYYY-MM-DD.", file=sys.stderr)
        return 2
    if until <= since:
        print("--until must be after --since.", file=sys.stderr)
        return 2
    params = {"env": args.env, "start": datetime.combine(since, time(0), tzinfo=NY),
              "end": datetime.combine(until, time(0), tzinfo=NY)}

    sys.path.insert(0, db_root())
    import db
    if not os.getenv("SUPABASE_HOST"):
        print(f"No SUPABASE_HOST after importing {getattr(db, '__file__', 'db')} (no .env beside "
              "it?); run with env -C ~/Desktop/trading_bot. Not connecting.", file=sys.stderr)
        return 2
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
            cur.execute(_PLANS_SQL, params)
            raw_plans = cur.fetchall()
            cur.execute(_VERDICTS_SQL, params)
            raw_verdicts = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    plans, levels = build_plans(raw_plans)
    verdicts = build_verdicts(raw_verdicts)
    moves = build_moves(plans, levels, verdicts)
    flips = build_flips(plans, levels, verdicts)
    flips["distinct_t1"] = distinct_t1(plans, levels)
    del levels, raw_plans
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(plans, verdicts, moves, args.env, since, until, head or "unknown", flips)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head or "unknown",
                       "env": args.env, "since": since.isoformat(), "until": until.isoformat(),
                       "note": NOTE, "plans": plans,
                       "verdicts": [{k: v for k, v in r.items() if k != "ladder_atr"}
                                    for r in verdicts],
                       "moves": moves, "flip_plans": flips["plans"],
                       "flip_pairs": flips["pairs"]}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
