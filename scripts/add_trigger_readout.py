"""Add trigger readout: how each OPEN/ADD verdict's price stood against its
plan's add_trigger, from the log-only keys _add_plan_record (agents/plans.py)
writes into signals.provenance.plan (audit 2026-11-27 08:53 #1). Read-only
engineering readout; percentages and counts only, no price, share count or
dollar figure.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/add_trigger_readout.py [--since 2026-11-23] [--until 2026-11-28] \
        [--env live] [--action ADD] [--share-basis signal] [--out FILE] [--json FILE]

Verdicts: signals rows with decision.model_output (the analyst's verdicts)
whose signal (the final action) is OPEN or ADD, timestamp in [--since,
--until) as New York dates, in --env or with a NULL env. The day is the New
York date of signals.timestamp. The keys exist from about 2026-11-23 (the
default --since); verdicts without them are counted as "no add record".

Keys read from provenance.plan (classify's record on an active plan):
  - add_under_hold_plan: an ADD while the plan says hold;
  - add_trigger: only whether it is a number or null (its value is a price
    and is not read);
  - price_vs_add_trigger_pct: the verdict's price over add_trigger - 1, in
    percent;
  - relation (FOLLOW, REVISE, DIVERGE, INVALIDATE, or the no-plan relations
    NEW_PLAN / SEEDED / NO_PLAN, which carry no add record), plan_says, held.

Per verdict, its trigger side: "below" (pct < 0), "at or above" (pct >= 0),
"null trigger" (add_trigger null: held_seed plans and plans without a
ladder), "no pct" (a trigger but no pct: the price was unknown) or "no add
record". Per cell (all, day, ticker, day x ticker): verdicts, with
add_under_hold_plan true, the sides, the pct distribution (n, median, min,
max) and the relation counts. A totals table splits OPEN and ADD; the day,
ticker and day x ticker tables are for --action (default ADD).

classify counts an ADD on a held name whose plan says hold as FOLLOW
("adding within plan"), whatever the price against add_trigger. The readout
says so in one line, with the line in agents/plans.py (found in the source
text, not imported) and the count of FOLLOW ADDs that were held and below
the trigger. No label or verdict is applied: a gate on these keys is Luke's
(live trading).

ADD share cells (audit 2026-11-27 10:53 #1): held-name verdicts (decision
.model_output) whose signal (or, with --share-basis model, whose
decision.model_action) is ADD or HOLD, in the same window and env. Held =
provenance.brief_exposure.held_at_gate true (the broker flag the gates used)
or decision.held_review true; held_at_gate false is "not held", neither is
"held unknown", and both are counted and left out. Each cell is ADD n /
ADD+HOLD n with the share, for:
  - conviction (decision.final_conviction) x UDR band (decision.udr_check
    .model: <2, 2-3, 3-4, >=4 with each lower edge included, or missing);
  - ticker;
  - session, from the New York time of signals.timestamp by market_calendar
    (loaded from this repo by path; standard library only): pre-market
    before 09:30, RTH 09:30 up to 16:00 (13:00 on its early-close days,
    2026-11-27 among them), after-hours from the close; weekends and NYSE
    holidays are "non-trading day".
Held verdicts whose model action and signal differ are counted by pair
(model -> signal), which is what separates the two bases. Counts and shares
only; what an ADD should be is Luke's.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db, market_calendar (by path) and the standard
library.
"""
from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_SINCE = "2026-11-23"
ACTIONS = ("ADD", "OPEN")
SIDES = ("below", "at or above", "null trigger", "no pct", "no add record")
MAIN_RELATIONS = ("FOLLOW", "REVISE", "DIVERGE")
FOLLOW_MARK = "# adding within plan"
SHARE_ACTIONS = ("ADD", "HOLD")
SHARE_BASES = ("signal", "model")
UDR_BANDS = ("<2", "2-3", "3-4", ">=4", "missing")
SESSIONS = ("pre-market", "RTH", "after-hours", "non-trading day")
RTH_OPEN = time(9, 30)

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol, s.signal,
       s.provenance->'plan'->>'relation',
       s.provenance->'plan'->>'plan_says',
       s.provenance->'plan'->>'held',
       COALESCE(s.provenance->'plan' ? 'add_under_hold_plan', false),
       s.provenance->'plan'->>'add_under_hold_plan',
       jsonb_typeof(s.provenance->'plan'->'add_trigger'),
       s.provenance->'plan'->>'price_vs_add_trigger_pct'
FROM signals s
WHERE s.decision ? 'model_output' AND s.signal IN ('OPEN', 'ADD')
  AND s.timestamp >= %(start)s AND s.timestamp < %(end)s
  AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
ORDER BY s.timestamp, s.id
"""

_SHARE_SQL = """
SELECT s.id, s.timestamp, s.symbol, s.signal, s.decision->>'model_action',
       s.decision->>'final_conviction',
       s.decision->'udr_check'->>'model',
       s.provenance->'brief_exposure'->>'held_at_gate',
       s.decision->>'held_review'
FROM signals s
WHERE s.decision ? 'model_output'
  AND (s.signal IN ('ADD', 'HOLD') OR s.decision->>'model_action' IN ('ADD', 'HOLD'))
  AND s.timestamp >= %(start)s AND s.timestamp < %(end)s
  AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
ORDER BY s.timestamp, s.id
"""


def db_root(cwd: str | None = None) -> str:
    """The directory to import db from: the working directory when it holds
    db.py (the live clone, whose .env db loads), else this script's repo."""
    cwd = cwd or os.getcwd()
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET_START <= t < QUIET_END


def _bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return {"true": True, "false": False}.get(str(v).lower())


def _pct(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def classify(row: tuple) -> dict:
    """One verdict row (the _VERDICTS_SQL columns) as a record. Pure."""
    sid, ts, sym, action, rel, says, held, has_rec, under, trig_type, pct = row
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    pct = _pct(pct)
    if not has_rec:
        side = "no add record"
    elif trig_type != "number":
        side = "null trigger"
    elif pct is None:
        side = "no pct"
    else:
        side = "below" if pct < 0 else "at or above"
    return {"id": sid, "day": ts.astimezone(NY).date().isoformat(), "ticker": sym,
            "action": action, "relation": rel or "none", "plan_says": says,
            "held": _bool(held), "add_record": bool(has_rec),
            "add_under_hold_plan": _bool(under) if has_rec else None,
            "side": side, "pct": pct if side in ("below", "at or above") else None}


def follow_below(r: dict) -> bool:
    """A FOLLOW ADD on a held name below its plan's add_trigger."""
    return (r["relation"] == "FOLLOW" and r["action"] == "ADD" and r["held"] is True
            and r["side"] == "below")


def tally(rows: list[dict]) -> dict:
    pcts = [r["pct"] for r in rows if r["pct"] is not None]
    return {"verdicts": len(rows),
            "under_hold": sum(1 for r in rows if r["add_under_hold_plan"] is True),
            "sides": {s: sum(1 for r in rows if r["side"] == s) for s in SIDES},
            "pct": {"n": len(pcts),
                    "median": round(statistics.median(pcts), 2) if pcts else None,
                    "min": min(pcts) if pcts else None, "max": max(pcts) if pcts else None},
            "relations": dict(Counter(r["relation"] for r in rows)),
            "follow_held_below": sum(1 for r in rows if follow_below(r))}


def build(rows: list[dict], action: str) -> dict:
    sel = [r for r in rows if r["action"] == action]
    by = lambda key: {k: tally([r for r in sel if key(r) == k])
                      for k in sorted({key(r) for r in sel})}
    return {"by_action": {a: tally([r for r in rows if r["action"] == a]) for a in ACTIONS},
            "action": action, "all": tally(sel),
            "by_day": by(lambda r: r["day"]), "by_ticker": by(lambda r: r["ticker"]),
            "by_day_ticker": by(lambda r: (r["day"], r["ticker"])),
            "relations": sorted({r["relation"] for r in rows},
                                key=lambda x: (MAIN_RELATIONS.index(x) if x in MAIN_RELATIONS
                                               else len(MAIN_RELATIONS), x))}


def follow_line_no(path: str | None = None) -> int | None:
    """The line of classify's held-ADD FOLLOW rule in agents/plans.py (read
    as text; agents.* is not imported), or None when it isn't found."""
    path = path or os.path.join(ROOT, "agents", "plans.py")
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if FOLLOW_MARK in line and "held" in line:
                    return i
    except OSError:
        return None
    return None


def calendar(path: str | None = None):
    """market_calendar.py beside this repo's root as a module (loaded by
    path; it imports only datetime)."""
    path = path or os.path.join(ROOT, "market_calendar.py")
    spec = importlib.util.spec_from_file_location("market_calendar", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def session_of(ts: datetime, holidays, early_closes) -> str:
    """The NYSE session of an aware time, by New York clock and calendar."""
    et = ts.astimezone(NY)
    d = et.date()
    if et.weekday() >= 5 or d in holidays:
        return "non-trading day"
    close = time(13, 0) if d in early_closes else time(16, 0)
    if et.time() < RTH_OPEN:
        return "pre-market"
    return "RTH" if et.time() < close else "after-hours"


def udr_band(v) -> str:
    u = _pct(v)
    if u is None:
        return "missing"
    return "<2" if u < 2 else "2-3" if u < 3 else "3-4" if u < 4 else ">=4"


def conviction_of(v):
    """decision.final_conviction as an int when it is a whole number, else
    "missing" (or the text as stored when it is not a number)."""
    if v is None or str(v).strip() == "":
        return "missing"
    f = _pct(v)
    if f is None:
        return str(v)
    return int(f) if f == int(f) else f


def held_state(at_gate, held_review) -> str:
    if _bool(at_gate) is True or _bool(held_review) is True:
        return "held"
    return "not held" if _bool(at_gate) is False else "held unknown"


def share_record(row: tuple, holidays, early_closes) -> dict:
    """One _SHARE_SQL row as a record. Pure."""
    sid, ts, sym, sig, model, conv, udr, at_gate, review = row
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {"id": sid, "day": ts.astimezone(NY).date().isoformat(), "ticker": sym,
            "signal": sig, "model_action": model, "conviction": conviction_of(conv),
            "udr_band": udr_band(udr), "session": session_of(ts, holidays, early_closes),
            "held": held_state(at_gate, review)}


def _share(rows: list[dict], basis: str) -> dict:
    a = sum(1 for r in rows if r[basis] == "ADD")
    return {"add": a, "n": len(rows)}


def _conv_key(c):
    return (1, 0, str(c)) if c == "missing" else (0, c, "") if isinstance(c, (int, float)) \
        else (0, float("inf"), str(c))


def share_build(recs: list[dict], basis: str) -> dict:
    """ADD n / ADD+HOLD n per cell for held verdicts whose basis field
    (signal or model_action) is ADD or HOLD."""
    field = "signal" if basis == "signal" else "model_action"
    inb = [r for r in recs if r[field] in SHARE_ACTIONS]
    pop = [r for r in inb if r["held"] == "held"]
    convs = sorted({r["conviction"] for r in pop}, key=_conv_key)
    grid = {c: {b: _share([r for r in pop if r["conviction"] == c and r["udr_band"] == b], field)
                for b in UDR_BANDS} for c in convs}
    return {"basis": basis, "field": field, "all": _share(pop, field),
            "excluded": dict(Counter(r["held"] for r in inb if r["held"] != "held")),
            "grid": grid,
            "by_udr": {b: _share([r for r in pop if r["udr_band"] == b], field) for b in UDR_BANDS},
            "by_conviction": {c: _share([r for r in pop if r["conviction"] == c], field)
                              for c in convs},
            "by_ticker": {t: _share([r for r in pop if r["ticker"] == t], field)
                          for t in sorted({r["ticker"] for r in pop})},
            "by_session": {s: _share([r for r in pop if r["session"] == s], field)
                           for s in SESSIONS},
            "pairs": dict(Counter(f"{r['model_action']} -> {r['signal']}" for r in recs
                                  if r["held"] == "held" and r["model_action"] != r["signal"]))}


def _cell(c: dict) -> str:
    return f"{c['add']}/{c['n']}" + (f" ({100 * c['add'] / c['n']:.1f}%)" if c["n"] else "")


def share_render(sh: dict) -> list[str]:
    f, t = sh["field"], sh["all"]
    ex = sh["excluded"]
    lines = ["", f"## ADD share cells (held names, {f} ADD or HOLD)", "",
             f"Held = provenance.brief_exposure.held_at_gate true or decision.held_review true. "
             f"Cells are ADD n / ADD+HOLD n by {f}. Conviction = decision.final_conviction; UDR "
             "= decision.udr_check.model, bands <2, 2-3, 3-4, >=4 (lower edge included). "
             "Session by market_calendar on the New York time of signals.timestamp: pre-market "
             "before 09:30, RTH up to 16:00 (13:00 on early-close days), after-hours from the "
             "close; weekends and holidays are non-trading day. Counts and shares only.", "",
             f"- held verdicts: {_cell(t)}; left out (not held / held unknown): "
             f"{ex.get('not held', 0)} / {ex.get('held unknown', 0)}",
             "- held verdicts whose model action and signal differ (model -> signal): "
             + (", ".join(f"{k} {v}" for k, v in sorted(sh["pairs"].items())) or "none")]
    head = ["conviction"] + list(UDR_BANDS) + ["all"]
    lines += ["", "### Conviction x UDR band", "", "| " + " | ".join(head) + " |",
              "|---|" + "---:|" * (len(head) - 1)]
    for c, row in sh["grid"].items():
        lines.append("| " + " | ".join([str(c)] + [_cell(row[b]) for b in UDR_BANDS]
                                       + [_cell(sh["by_conviction"][c])]) + " |")
    lines.append("| all | " + " | ".join([_cell(sh["by_udr"][b]) for b in UDR_BANDS]
                                         + [_cell(t)]) + " |")
    for title, key, cells in (("By ticker", "ticker", sh["by_ticker"]),
                              ("By session", "session", sh["by_session"])):
        lines += ["", f"### {title}", "", f"| {key} | ADD n | ADD+HOLD n | share |",
                  "|---|---:|---:|---:|"]
        for k, c in cells.items():
            lines.append(f"| {k} | {c['add']} | {c['n']} | "
                         + (f"{100 * c['add'] / c['n']:.1f}%" if c["n"] else "n/a") + " |")
        if not cells:
            lines.append("| none | | | |")
    return lines


def _p(v) -> str:
    return "n/a" if v is None else f"{v:+.1f}%"


def _rels(t: dict, rels: list[str]) -> list[str]:
    return [str(t["relations"].get(x, 0)) for x in rels]


def _table(title: str, keycols: list[str], cells: dict, rels: list[str]) -> list[str]:
    head = keycols + ["verdicts", "under hold plan"] + list(SIDES) + \
        ["pct n", "pct median", "pct min", "pct max"] + rels
    lines = ["", f"## {title}", "", "| " + " | ".join(head) + " |",
             "|" + "---|" * len(keycols) + "---:|" * (len(head) - len(keycols))]
    for k, t in cells.items():
        keys = list(k) if isinstance(k, tuple) else [k]
        lines.append("| " + " | ".join(
            keys + [str(t["verdicts"]), str(t["under_hold"])]
            + [str(t["sides"][s]) for s in SIDES]
            + [str(t["pct"]["n"]), _p(t["pct"]["median"]), _p(t["pct"]["min"]), _p(t["pct"]["max"])]
            + _rels(t, rels)) + " |")
    if not cells:
        lines.append("| none |" + " |" * (len(head) - 1))
    return lines


def render(out: dict, env: str, since: date, until: date, head: str, now: datetime,
           line_no: int | None, share: dict | None = None) -> str:
    a, t = out["action"], out["all"]
    rels = out["relations"] or list(MAIN_RELATIONS)
    where = f"agents/plans.py:{line_no}" if line_no else "agents/plans.py classify (line not found)"
    lines = [f"# Add trigger readout ({env}, OPEN/ADD verdicts {since.isoformat()} to before "
             f"{until.isoformat()} ET)", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             "Engineering readout; percentages and counts only. Keys from signals.provenance.plan "
             "(_add_plan_record, agents/plans.py): add_under_hold_plan, add_trigger (number or "
             "null only) and price_vs_add_trigger_pct (the verdict's price over add_trigger - 1). "
             "Below = pct < 0; at or above = pct >= 0. No label: a gate on these keys is Luke's.", "",
             f"- FOLLOW includes held ADDs below the trigger: classify ({where}, \"adding within "
             "plan\") counts an ADD on a held name whose plan says hold as FOLLOW, whatever its "
             f"price against add_trigger. Here {t['follow_held_below']} of "
             f"{t['relations'].get('FOLLOW', 0)} FOLLOW {a} verdicts were held and below the "
             "trigger.",
             f"- {a} verdicts: {t['verdicts']}; add_under_hold_plan true: {t['under_hold']}; "
             + "; ".join(f"{s}: {t['sides'][s]}" for s in SIDES)
             + f"; pct median {_p(t['pct']['median'])}, min {_p(t['pct']['min'])}, "
               f"max {_p(t['pct']['max'])} (n {t['pct']['n']})",
             f"- relations ({a}): " + (", ".join(f"{x} {t['relations'][x]}" for x in rels
                                                 if x in t["relations"]) or "none")]
    lines += _table("Totals by action", ["action"], out["by_action"], rels)
    lines += _table(f"{a} verdicts by day", ["day"], out["by_day"], rels)
    lines += _table(f"{a} verdicts by ticker", ["ticker"], out["by_ticker"], rels)
    lines += _table(f"{a} verdicts by day and ticker", ["day", "ticker"], out["by_day_ticker"], rels)
    if share is not None:
        lines += share_render(share)
    return "\n".join(lines) + "\n"


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--action", default="ADD", choices=ACTIONS,
                    help="the action the day/ticker tables are for")
    ap.add_argument("--share-basis", default="signal", choices=SHARE_BASES,
                    help="the ADD share cells by the final signal or decision.model_action")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-verdict records (percentages only) here")
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
        print("--since and --until must be YYYY-MM-DD.", file=sys.stderr)
        return 2
    if until <= since:
        print("--until must be after --since.", file=sys.stderr)
        return 2
    start = datetime.combine(since, time(0), tzinfo=NY)
    end = datetime.combine(until, time(0), tzinfo=NY)

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
            cur.execute(_VERDICTS_SQL, {"env": args.env, "start": start, "end": end})
            raw = cur.fetchall()
            cur.execute(_SHARE_SQL, {"env": args.env, "start": start, "end": end})
            share_raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = [classify(r) for r in raw]
    out = build(rows, args.action)
    mc = calendar()
    recs = [share_record(r, mc.NYSE_HOLIDAYS, mc.NYSE_EARLY_CLOSES) for r in share_raw]
    share = share_build(recs, args.share_basis)
    head = code_version()
    text = render(out, args.env, since, until, head, now, follow_line_no(), share)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head, "env": args.env,
                       "since": since.isoformat(), "until": until.isoformat(),
                       "verdicts": rows, "share_basis": args.share_basis,
                       "share_verdicts": recs}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
