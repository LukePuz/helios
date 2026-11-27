"""Plan target-hit readout: for each held plan, the first close at or above
its target_price, the verdicts recorded on the plan around it, and what the
price did afterwards (audit 2026-11-26 20:53 #1). Read-only engineering
readout; counts and percentages only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/plan_target_hit_readout.py [--since 2026-10-28] [--env live] \
        [--all-plans] [--out FILE] [--export FILE.json]

Plans: trade_plans rows created since --since (New York date) in --env or
with a NULL env. A plan is held when at least one analyst verdict recorded
on it (provenance.plan.plan_id) has provenance.plan.held true, the flag
plans.observe records; --all-plans drops that filter. Each revision is its
own row with its own target, so plans are not independent.

Hit: the first full session of the plan's life whose close is at or above
target_price. A session is in the life when its 09:30 ET open is at or
after created_at (the creation day's partial session is left out, as in
plan_forward_readout) and, for a closed plan, its 16:00 ET close is at or
before closed_at. Bars: the cached daily closes in price_history (no vendor
or broker call). Close is the only hit rule tried.

After a hit, every later session up to the latest cached bar counts,
whether or not the plan was closed by then (closed_at is shown beside it):
  - +5 / +10 / +20: the close that many sessions after the hit, as % vs the
    target, and the lowest close over sessions +1..+20 vs the target ("-"
    while fewer sessions exist);
  - peak: the highest close from the hit session on, as % vs entry;
  - latest close, % vs entry and vs target;
  - giveback from peak: the latest close vs that peak, %; max drawdown: the
    deepest close below the running peak since the hit, %.

Verdicts: every analyst verdict whose plan record names the plan, by period
(before the hit session, the hit session's ET date, after), relation,
plan_why and final_action. A separate table counts every verdict on the
plans read whose plan_why is "target reached" (the live price at the
verdict, not a close) by whether the plan has a close hit.

Percent only: no price, level or account figure is printed or exported.
No rule, threshold or rate is chosen; the export (--export, JSON, the same
percentages) lets another be applied.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. db is imported from the
working directory when it holds a db.py (the live clone, with its .env),
else from beside this script; with no SUPABASE_HOST after the import the
readout stops before connecting. Imports: db and the standard library only.
Not loaded by weekly_readouts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
HORIZONS = (5, 10, 20)
TARGET_REACHED = "target reached"
PERIODS = ("before hit", "hit session", "after hit")

_PLANS_SQL = """
SELECT id, ticker, status, source, revision, created_at, closed_at, close_reason,
       entry_price, target_price
FROM trade_plans
WHERE (trading_env = %(env)s OR trading_env IS NULL) AND created_at >= %(start)s
ORDER BY id
"""

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol, s.provenance->'plan'->>'plan_id',
       s.provenance->'plan'->>'relation', s.provenance->'plan'->>'plan_why',
       s.provenance->'plan'->>'held', s.decision->>'final_action'
FROM signals s
WHERE s.decision ? 'model_output' AND s.timestamp >= %(start)s
  AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
  AND s.provenance->'plan'->>'plan_id' = ANY(%(ids)s)
ORDER BY s.timestamp, s.id
"""

_BARS_SQL = """
SELECT ticker, date, close FROM price_history
WHERE ticker = ANY(%(tickers)s) AND date >= %(first)s
ORDER BY ticker, date
"""


def db_root(cwd: str | None = None) -> str:
    """The directory db is imported from: cwd when it holds a db.py, else
    this script's checkout."""
    cwd = os.getcwd() if cwd is None else cwd
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _aware(t):
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _pct(a, b):
    """a vs b in %, rounded to 2 places; None when either is missing or b is 0."""
    return round((a / b - 1) * 100, 2) if a is not None and b else None


def _et_day(t) -> date | None:
    t = _aware(t)
    return t.astimezone(NY).date() if t else None


def life_sessions(bars: list[dict], created_at, closed_at) -> list[dict]:
    """The plan's full sessions: 09:30 ET open at or after created_at and, when
    closed, 16:00 ET close at or before closed_at. bars sorted by date."""
    c, end = _aware(created_at), _aware(closed_at)
    out = []
    for b in bars:
        if c is not None and datetime.combine(b["date"], SESSION_OPEN, tzinfo=NY) < c:
            continue
        if end is not None and datetime.combine(b["date"], SESSION_CLOSE, tzinfo=NY) > end:
            break
        out.append(b)
    return out


def analyse(plan: dict, bars: list[dict]) -> dict:
    """The percent-only record of one plan against its sorted daily closes.
    Pure; no price is kept in the record."""
    entry, target = _num(plan.get("entry_price")), _num(plan.get("target_price"))
    life = life_sessions(bars, plan.get("created_at"), plan.get("closed_at"))
    rec = {"plan_id": plan.get("id"), "ticker": plan.get("ticker"),
           "source": plan.get("source"), "revision": plan.get("revision"),
           "status": plan.get("status"), "close_reason": plan.get("close_reason"),
           "created_et": _fmt(plan.get("created_at")), "closed_et": _fmt(plan.get("closed_at")),
           "target_vs_entry_pct": _pct(target, entry), "life_sessions": len(life),
           "hit_date": None, "sessions_to_hit": None, "hit_close_vs_target_pct": None,
           "hit_close_vs_entry_pct": None, "sessions_since_hit": None,
           **{f"fwd{n}_vs_target_pct": None for n in HORIZONS},
           f"low_through_{HORIZONS[-1]}_vs_target_pct": None,
           "peak_date": None, "peak_vs_entry_pct": None, "latest_date": None,
           "latest_vs_entry_pct": None, "latest_vs_target_pct": None,
           "giveback_from_peak_pct": None, "max_drawdown_from_peak_pct": None}
    if target is None:
        return rec
    hit_i = next((i for i, b in enumerate(life) if b["close"] >= target), None)
    if hit_i is None:
        return rec
    hit = life[hit_i]
    after = [b for b in bars if b["date"] > hit["date"]]
    rec.update(hit_date=hit["date"].isoformat(), sessions_to_hit=hit_i + 1,
               hit_close_vs_target_pct=_pct(hit["close"], target),
               hit_close_vs_entry_pct=_pct(hit["close"], entry),
               sessions_since_hit=len(after))
    for n in HORIZONS:
        if len(after) >= n:
            rec[f"fwd{n}_vs_target_pct"] = _pct(after[n - 1]["close"], target)
    if len(after) >= HORIZONS[-1]:
        rec[f"low_through_{HORIZONS[-1]}_vs_target_pct"] = _pct(
            min(b["close"] for b in after[:HORIZONS[-1]]), target)
    path = [hit] + after
    peak, run, deepest = path[0], path[0]["close"], 0.0
    for b in path:
        if b["close"] > peak["close"]:
            peak = b
        run = max(run, b["close"])
        deepest = min(deepest, b["close"] / run - 1)
    last = path[-1]
    rec.update(peak_date=peak["date"].isoformat(), peak_vs_entry_pct=_pct(peak["close"], entry),
               latest_date=last["date"].isoformat(),
               latest_vs_entry_pct=_pct(last["close"], entry),
               latest_vs_target_pct=_pct(last["close"], target),
               giveback_from_peak_pct=_pct(last["close"], peak["close"]),
               max_drawdown_from_peak_pct=round(deepest * 100, 2))
    return rec


def _fmt(t) -> str | None:
    t = _aware(t)
    return t.astimezone(NY).strftime("%Y-%m-%d %H:%M") if t else None


def period(verdict_ts, hit_date: str | None) -> str | None:
    """before hit / hit session / after hit by the verdict's ET date; None
    without a hit."""
    if not hit_date:
        return None
    d, h = _et_day(verdict_ts), date.fromisoformat(hit_date)
    if d is None:
        return None
    return PERIODS[0] if d < h else PERIODS[1] if d == h else PERIODS[2]


def build(plans: list[dict], verdicts: list[tuple], bars: dict, all_plans: bool) -> dict:
    """Records, verdict counts and plan counts from the fetched rows. Pure.
    verdicts: the _VERDICTS_SQL column order."""
    by_plan: dict = {}
    for v in verdicts:
        by_plan.setdefault(str(v[3]), []).append(v)
    held = {pid for pid, vs in by_plan.items() if any(str(v[6]).lower() == "true" for v in vs)}
    chosen = [p for p in plans if all_plans or str(p["id"]) in held]
    recs = [analyse(p, bars.get(p["ticker"], [])) for p in chosen]
    hits = {str(r["plan_id"]): r for r in recs if r["hit_date"]}
    per_period = Counter()
    reached = Counter()
    for r in recs:
        for v in by_plan.get(str(r["plan_id"]), []):
            _, ts, _, pid, rel, why, _, final = v
            if pid in hits:
                per_period[(r["plan_id"], period(ts, hits[pid]["hit_date"]),
                            rel or "none", why or "none", final or "none")] += 1
            if why == TARGET_REACHED:
                reached[("close hit" if pid in hits else "no close hit", final or "none")] += 1
    return {"records": recs, "per_period": per_period, "reached": reached,
            "counts": {"plans": len(plans), "held": sum(str(p["id"]) in held for p in plans),
                       "read": len(chosen),
                       "with_target": sum(_num(p.get("target_price")) is not None for p in chosen),
                       "with_session": sum(r["life_sessions"] > 0 for r in recs),
                       "hit": len(hits)}}


def _c(v) -> str:
    return "-" if v is None else f"{v:+.2f}" if isinstance(v, float) else str(v)


def render(out: dict, env: str, since: date, all_plans: bool, head: str,
           latest_bar: str | None) -> str:
    k = out["counts"]
    lines = ["# Plan target-hit readout", "",
             f"Env {env}; plans created since {since} (New York); "
             f"{'all plans' if all_plans else 'held plans (a verdict recorded plan.held true)'}; "
             f"code {head}; latest cached bar {latest_bar or '-'}. Percent only, no prices; "
             "counts only; no rule or threshold is chosen. Hit = first full session of the "
             "plan's life with a close at or above target_price.", "",
             f"Plans: {k['plans']} (held {k['held']}); read {k['read']}; with a target "
             f"{k['with_target']}; with a full session in their life {k['with_session']}; "
             f"with a close hit {k['hit']}.", "",
             "## Plans whose close reached the target", ""]
    hits = [r for r in out["records"] if r["hit_date"]]
    if hits:
        lo = f"low_through_{HORIZONS[-1]}_vs_target_pct"
        lines += ["| plan | name | source | rev | status | created (ET) | closed (ET) | hit session "
                  "| sessions to hit | target vs entry % | hit close vs target % | "
                  + " | ".join(f"+{n} vs target %" for n in HORIZONS)
                  + f" | low +1..+{HORIZONS[-1]} vs target % | sessions since hit | peak (date) "
                  "vs entry % | latest (date) vs entry % | latest vs target % | giveback from "
                  "peak % | max drawdown from peak % |",
                  "|---" * (18 + len(HORIZONS)) + "|"]
        for r in hits:
            lines.append(
                f"| {r['plan_id']} | {r['ticker']} | {r['source'] or '-'} | {_c(r['revision'])} "
                f"| {r['status']} | {r['created_et']} | {r['closed_et'] or '-'} | {r['hit_date']} "
                f"| {r['sessions_to_hit']} | {_c(r['target_vs_entry_pct'])} "
                f"| {_c(r['hit_close_vs_target_pct'])} | "
                + " | ".join(_c(r[f"fwd{n}_vs_target_pct"]) for n in HORIZONS)
                + f" | {_c(r[lo])} | {r['sessions_since_hit']} "
                f"| {_c(r['peak_vs_entry_pct'])} ({r['peak_date']}) "
                f"| {_c(r['latest_vs_entry_pct'])} ({r['latest_date']}) "
                f"| {_c(r['latest_vs_target_pct'])} | {_c(r['giveback_from_peak_pct'])} "
                f"| {_c(r['max_drawdown_from_peak_pct'])} |")
    else:
        lines.append("None.")
    lines += ["", "## Verdicts on those plans, by period, relation, plan_why and final action", ""]
    if out["per_period"]:
        lines += ["| plan | period | relation | plan_why | final action | n |",
                  "|---|---|---|---|---|---:|"]
        order = {p: i for i, p in enumerate(PERIODS)}
        for (pid, per, rel, why, fin), n in sorted(
                out["per_period"].items(),
                key=lambda kv: (kv[0][0], order.get(kv[0][1], 9), kv[0][2:])):
            lines.append(f"| {pid} | {per} | {rel} | {why} | {fin} | {n} |")
    else:
        lines.append("None.")
    lines += ["", f'## Verdicts with plan_why "{TARGET_REACHED}" on the plans read, by close-hit '
              "status and final action", ""]
    rc = out["reached"]
    if rc:
        cols = sorted({c for _, c in rc})
        lines += ["| | " + " | ".join(cols) + " | all |", "|---|" + "---:|" * (len(cols) + 1)]
        for row in ("close hit", "no close hit"):
            cells = [rc.get((row, c), 0) for c in cols]
            lines.append(f"| {row} | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    else:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"


def _bar_qa():
    """scripts/_bar_qa.py (the non-finite bar flag) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bar_qa.py")
    spec = importlib.util.spec_from_file_location("_bar_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-28", help="YYYY-MM-DD, New York date")
    ap.add_argument("--env", default="live")
    ap.add_argument("--all-plans", action="store_true",
                    help="every plan, not only those a verdict recorded as held")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--export", help="write the per-plan records (JSON, percent only) here")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    since = date.fromisoformat(args.since)
    start = datetime.combine(since, time(0), tzinfo=NY)

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
            cur.execute(_PLANS_SQL, {"env": args.env, "start": start})
            cols = ("id", "ticker", "status", "source", "revision", "created_at", "closed_at",
                    "close_reason", "entry_price", "target_price")
            plans = [dict(zip(cols, row)) for row in cur.fetchall()]
            verdicts, bars = [], {}
            if plans:
                cur.execute(_VERDICTS_SQL, {"env": args.env, "start": start,
                                            "ids": [str(p["id"]) for p in plans]})
                verdicts = cur.fetchall()
                cur.execute(_BARS_SQL, {"tickers": sorted({p["ticker"] for p in plans}),
                                        "first": since - timedelta(days=7)})
                for t, d, c in cur.fetchall():
                    if _num(c) is not None:
                        bars.setdefault(t, []).append({"date": d, "close": float(c)})
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    out = build(plans, verdicts, bars, args.all_plans)
    latest = max((b[-1]["date"] for b in bars.values() if b), default=None)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(out, args.env, since, args.all_plans, head or "unknown",
                  latest.isoformat() if latest else None)
    text += "\n" + _bar_qa().flag_line(bars) + "\n"   # flag only (scripts/_bar_qa.py)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    if args.export:
        with open(args.export, "w") as f:
            json.dump({"since": since.isoformat(), "env": args.env, "all_plans": args.all_plans,
                       "horizons": list(HORIZONS), "counts": out["counts"],
                       "records": out["records"],
                       "verdicts": [{"plan_id": k[0], "period": k[1], "relation": k[2],
                                     "plan_why": k[3], "final_action": k[4], "n": n}
                                    for k, n in sorted(out["per_period"].items(), key=str)]},
                      f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
