"""Plan forward readout: for each trade_plans row, which of its levels the
price reached first within the plan's horizon.

Engineering diagnostic (pilot phase), not a study analysis. Scores plans,
not closed trades: every trade_plans row since --since (default 2026-10-28)
counts, whether or not a position was ever opened or closed.

For each plan, two pairs of levels, each scored the same way:
  - analyst: invalidation_price (down) vs target_price (up), the levels the
    analyst stated;
  - compiled: structural_stop (down) vs t1 (up), compile_plan's levels
    (levels_source set; NULL for held_seed plans and plans before 11-04).

Window: the cached daily bars (price_history: date, high, low, close; no
vendor or broker call) of the full sessions inside the plan's horizon: a
session counts when its 09:30 ET open is at or after created_at and its
date is on or before horizon_end; the creation day's partial session is
left out. The window does not stop when the plan is superseded or closed:
the question is what the price did against the levels as stated, over the
horizon as stated. Whether the first hit came while the plan was still in
force (before closed_at) is reported beside it.

Two hit rules, the only thresholds tried, each applied to every level:
  - intraday: low at or below a down level, high at or above an up level;
  - close: the close at or beyond the level.
Outcome per pair: "down first", "up first", "both/same bar" (both reached
inside the same daily bar: daily bars can't order them, so neither is
picked), "neither, horizon over", "neither yet, horizon open", or
"no sessions yet" ("no full session in horizon" when the horizon ended
before one). A plan whose level was already on the wrong side of the
entry price at creation is reported as "level past entry at creation" and
not scored for that pair.

A level-distance section groups the scored outcomes by fixed display bins
of the up/down distance ratio and of each level's own distance from entry
(RATIO_BINS, DOWN_BINS, UP_BINS): description only, no expected-rate model
and no skill score.

Plans of one ticker overlap (revisions every few hours share the same
bars), so they are not independent; the per-ticker table shows how
concentrated the sample is. Distributions are described only: no
threshold or promotion rule is chosen here, and any count with n < 5 is
marked insufficient. Levels are shown as % from the plan's entry price,
never as prices.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/plan_forward_readout.py [--since 2026-10-28] [--out FILE]

Read-only: one transaction opened READ ONLY, then rolled back.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

NY = ZoneInfo("America/New_York")
MIN_N = 5
SESSION_OPEN = time(9, 30)
PAIRS = (("analyst", "invalidation_price", "target_price"),
         ("compiled", "structural_stop", "t1"))
RULES = ("intraday", "close")
OUTCOMES = ("down first", "up first", "both/same bar", "neither, horizon over",
            "neither yet, horizon open", "no sessions yet", "no full session in horizon",
            "level past entry at creation",
            "level missing")
SCORED = ("down first", "up first", "both/same bar", "neither, horizon over")
# Fixed display bins for the level-distance section: (low, high, label),
# low inclusive, high exclusive, None = open. For grouping the table only;
# no bin edge is a threshold or a promotion rule.
RATIO_BINS = ((None, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, 3.0, "2-3"), (3.0, None, ">=3"))
DOWN_BINS = ((None, 5.0, "<5%"), (5.0, 10.0, "5-10%"), (10.0, 20.0, "10-20%"), (20.0, None, ">=20%"))
UP_BINS = ((None, 10.0, "<10%"), (10.0, 25.0, "10-25%"), (25.0, 50.0, "25-50%"), (50.0, None, ">=50%"))


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


def _day(v) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10]) if v else None
    except ValueError:
        return None


def horizon_sessions(bars: list[dict], created_at, horizon_end) -> list[dict]:
    """Full sessions from created_at's next 09:30 ET open through horizon_end."""
    c = _aware(created_at)
    end = _day(horizon_end)
    out = []
    for b in sorted(bars, key=lambda b: b["date"]):
        if datetime.combine(b["date"], SESSION_OPEN, tzinfo=NY) < c:
            continue
        if end is not None and b["date"] > end:
            break
        out.append(b)
    return out


def _hit(b: dict, level: float, down: bool, rule: str) -> bool:
    if rule == "close":
        return b["close"] <= level if down else b["close"] >= level
    return b["low"] <= level if down else b["high"] >= level


def score_pair(sessions: list[dict], entry, down_level, up_level, rule: str,
               horizon_end, last_bar: date | None) -> dict:
    """{"outcome", "index", "date"} for one level pair under one hit rule.
    index is the 0-based session of the first hit (None when nothing hit)."""
    if down_level is None or up_level is None or not entry:
        return {"outcome": "level missing", "index": None, "date": None}
    if down_level >= entry or up_level <= entry:
        return {"outcome": "level past entry at creation", "index": None, "date": None}
    for i, b in enumerate(sessions):
        d, u = _hit(b, down_level, True, rule), _hit(b, up_level, False, rule)
        if d and u:
            return {"outcome": "both/same bar", "index": i, "date": b["date"]}
        if d:
            return {"outcome": "down first", "index": i, "date": b["date"]}
        if u:
            return {"outcome": "up first", "index": i, "date": b["date"]}
    end = _day(horizon_end)
    if not sessions:
        over = end is not None and last_bar is not None and last_bar >= end
        return {"outcome": "no full session in horizon" if over else "no sessions yet",
                "index": None, "date": None}
    over = end is not None and last_bar is not None and last_bar >= end
    return {"outcome": "neither, horizon over" if over else "neither yet, horizon open",
            "index": None, "date": None}


def analyse_plan(plan: dict, bars: list[dict]) -> dict:
    """One plan's levels (% from entry), sessions, and each pair's outcome
    under each rule, with whether the first hit came while the plan was in
    force (session date on or before closed_at's ET date)."""
    bars = sorted(bars, key=lambda b: b["date"])
    last_bar = bars[-1]["date"] if bars else None
    sess = horizon_sessions(bars, plan["created_at"], plan.get("horizon_end"))
    entry = _num(plan.get("entry_price"))
    closed = _aware(plan.get("closed_at"))
    closed_d = closed.astimezone(NY).date() if closed else None
    out = {"plan": plan, "sessions": len(sess), "pairs": {},
           "pct": {k: (round((_num(plan.get(k)) / entry - 1) * 100, 2)
                       if _num(plan.get(k)) and entry else None)
                   for k in ("invalidation_price", "target_price", "structural_stop", "t1")}}
    for name, down_k, up_k in PAIRS:
        res = {}
        for rule in RULES:
            r = score_pair(sess, entry, _num(plan.get(down_k)), _num(plan.get(up_k)), rule,
                           plan.get("horizon_end"), last_bar)
            r["in_force"] = (None if r["date"] is None else
                             closed_d is None or r["date"] <= closed_d)
            res[rule] = r
        out["pairs"][name] = res
    return out


def describe(values: list[float]) -> str:
    n = len(values)
    if n < MIN_N:
        return f"insufficient (n={n})"
    q = statistics.quantiles(values, n=4, method="inclusive")
    return (f"n={n}, min {min(values):+.2f}, p25 {q[0]:+.2f}, median {statistics.median(values):+.2f}, "
            f"p75 {q[2]:+.2f}, max {max(values):+.2f}")


def count(k: int, n: int) -> str:
    if n < MIN_N:
        return f"{k} of {n} (insufficient)"
    return f"{k} of {n} ({k / n:.0%})"


def _pair_table(recs: list[dict], pair: str, rule: str) -> list[str]:
    res = [r["pairs"][pair][rule] for r in recs]
    scored = [x for x in res if x["outcome"] in SCORED]
    n = len(scored)
    lines = [f"| outcome | plans (of {n} scored) |", "|---|---|"]
    for o in SCORED:
        lines.append(f"| {o} | {count(sum(x['outcome'] == o for x in scored), n)} |")
    lines.append("")
    rest = [o for o in OUTCOMES if o not in SCORED]
    lines.append("Not scored: " + ", ".join(f"{o} {sum(x['outcome'] == o for x in res)}" for o in rest) + ".")
    for o in ("down first", "up first"):
        idx = [float(x["index"]) for x in scored if x["outcome"] == o]
        lines.append(f"Sessions to {o.split()[0]} hit (0 = first full session): {describe(idx)}.")
    hits = [x for x in scored if x["date"] is not None]
    lines.append(f"First hit while the plan was still in force: "
                 f"{count(sum(bool(x['in_force']) for x in hits), len(hits))}.")
    return lines


def _bin_label(v, bins) -> str | None:
    """The label of the bin v falls in (low inclusive, high exclusive), or
    None when v is None."""
    if v is None:
        return None
    for lo, hi, label in bins:
        if (lo is None or v >= lo) and (hi is None or v < hi):
            return label
    return None


def _distances(r: dict, down_k: str, up_k: str) -> dict:
    """A plan's level distances from entry, in % points (both positive), and
    their ratio up / down; None where a level is missing or not on its side."""
    d, u = r["pct"].get(down_k), r["pct"].get(up_k)
    down = -d if d is not None and d < 0 else None
    up = u if u is not None and u > 0 else None
    return {"down": down, "up": up,
            "ratio": (up / down) if up is not None and down else None}


def _bin_table(recs: list[dict], pair: str, down_k: str, up_k: str, rule: str,
               measure: str, bins) -> list[str]:
    """Scored outcomes per fixed display bin of one distance measure (ratio,
    down or up); n < MIN_N per bin is insufficient."""
    rows = [(_bin_label(_distances(r, down_k, up_k)[measure], bins), r["pairs"][pair][rule])
            for r in recs]
    rows = [(lab, x) for lab, x in rows if x["outcome"] in SCORED]
    lines = ["| bin | scored | down first | up first | both/same bar | neither, horizon over |",
             "|---|---|---|---|---|---|"]
    for _, _, label in bins:
        xs = [x for lab, x in rows if lab == label]
        n = len(xs)
        lines.append(f"| {label} | {n} | " + " | ".join(
            count(sum(x["outcome"] == o for x in xs), n) for o in SCORED) + " |")
    return lines


def readout(recs: list[dict], since: date, env: str, head: str, now: datetime,
            last_bar: date | None) -> str:
    tickers: dict = {}
    for r in recs:
        tickers.setdefault(r["plan"]["ticker"], []).append(r)
    compiled = [r for r in recs if r["plan"].get("levels_source")]
    lines = [f"# Plan forward readout ({env}, plans since {since})", "",
             f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} at code {head}. Pilot-phase "
             "engineering diagnostic, not a study result. Bars: price_history daily bars "
             f"(latest {last_bar or '-'}); window = full sessions from the first 09:30 ET open "
             "after creation through horizon_end. Tie rule: both levels inside one daily bar = "
             "\"both/same bar\", never ordered. Two hit rules (intraday high/low, close). "
             f"Levels as % from the plan's entry price; no prices. n < {MIN_N} = insufficient. "
             "No thresholds chosen.", "",
             f"Plans: {len(recs)} on {len(tickers)} names; {len(compiled)} with compiled levels. "
             "Revisions of one name overlap and share the same bars, so plans are not "
             "independent: see the per-name table.", ""]
    for name, down_k, up_k in PAIRS:
        sub = recs if name == "analyst" else compiled
        lines += [f"## {name.capitalize()} levels: {down_k} vs {up_k}", "",
                  f"Down level vs entry (%): {describe([r['pct'][down_k] for r in sub if r['pct'][down_k] is not None])}. "
                  f"Up level vs entry (%): {describe([r['pct'][up_k] for r in sub if r['pct'][up_k] is not None])}.", ""]
        for rule in RULES:
            lines += [f"### {rule} rule", ""] + _pair_table(sub, name, rule) + [""]
    lines += ["## Outcomes by level distance (fixed display bins)", "",
              "The levels' distances from entry shape which one is reached first: a near "
              "down level and a far up level favour \"down first\" whatever the plan's "
              "merit. These tables only group the scored outcomes above by distance; the "
              "bins are for display, not thresholds, and nothing here models an expected "
              "rate or scores skill. ratio = up distance / down distance (both as % of "
              "entry).", ""]
    for name, down_k, up_k in PAIRS:
        sub = recs if name == "analyst" else compiled
        for rule in RULES:
            for measure, bins, title in (("ratio", RATIO_BINS, "up / down distance ratio"),
                                         ("down", DOWN_BINS, f"{down_k} distance below entry"),
                                         ("up", UP_BINS, f"{up_k} distance above entry")):
                lines += [f"### {name}, {rule} rule: by {title}", ""] + \
                    _bin_table(sub, name, down_k, up_k, rule, measure, bins) + [""]
    lines += ["## By status and source (analyst levels, intraday rule)", "",
              "| status / source | plans | down first | up first | both/same bar | neither, over | not scored |",
              "|---|---|---|---|---|---|---|"]
    groups: dict = {}
    for r in recs:
        p = r["plan"]
        groups.setdefault(f"{p.get('status')} / {p.get('source')}", []).append(r)
    for g in sorted(groups):
        res = [r["pairs"]["analyst"]["intraday"]["outcome"] for r in groups[g]]
        lines.append(f"| {g} | {len(res)} | {res.count('down first')} | {res.count('up first')} | "
                     f"{res.count('both/same bar')} | {res.count('neither, horizon over')} | "
                     f"{sum(o not in SCORED for o in res)} |")
    lines += ["", "## Per name (concentration)", "",
              "| name | plans | analyst down first | up first | both/same bar | neither, over | not scored |",
              "|---|---|---|---|---|---|---|"]
    for t in sorted(tickers, key=lambda t: (-len(tickers[t]), t)):
        res = [r["pairs"]["analyst"]["intraday"]["outcome"] for r in tickers[t]]
        lines.append(f"| {t} | {len(res)} | {res.count('down first')} | {res.count('up first')} | "
                     f"{res.count('both/same bar')} | {res.count('neither, horizon over')} | "
                     f"{sum(o not in SCORED for o in res)} |")
    lines += ["", "## Per plan", "",
              "| plan | name | status | created (ET) | horizon end | sessions | inval % | target % | "
              "analyst (intraday) | analyst (close) | stop % | t1 % | compiled (intraday) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:+.1f}"

    def o(x):
        s = x["outcome"]
        return s + (f" @{x['index']}" if x["index"] is not None else "")
    for r in sorted(recs, key=lambda r: (r["plan"]["ticker"], _aware(r["plan"]["created_at"]))):
        p, pc, pr = r["plan"], r["pct"], r["pairs"]
        lines.append(
            f"| {p['id']} | {p['ticker']} | {p.get('status')} | "
            f"{_aware(p['created_at']).astimezone(NY).strftime('%m-%d %H:%M')} | "
            f"{_day(p.get('horizon_end')) or '-'} | {r['sessions']} | {f(pc['invalidation_price'])} | "
            f"{f(pc['target_price'])} | {o(pr['analyst']['intraday'])} | {o(pr['analyst']['close'])} | "
            f"{f(pc['structural_stop'])} | {f(pc['t1'])} | "
            f"{o(pr['compiled']['intraday']) if p.get('levels_source') else '-'} |")
    return "\n".join(lines) + "\n"


def build(plans: list[dict], bars: dict, since: date, env: str, head: str, now: datetime) -> str:
    recs = [analyse_plan(p, bars.get(p["ticker"], [])) for p in plans]
    last_bar = max((b["date"] for bs in bars.values() for b in bs), default=None)
    return readout(recs, since, env, head, now, last_bar)


def _fetch(since: date, env: str):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cols = ("id", "ticker", "status", "source", "created_at", "closed_at", "close_reason",
                    "entry_price", "invalidation_price", "target_price", "horizon_end",
                    "last_relation", "revision", "structural_stop", "t1", "levels_source")
            cur.execute(f"SELECT {', '.join(cols)} FROM trade_plans "
                        "WHERE trading_env = %s AND created_at >= %s", (env, start))
            plans = []
            for row in cur.fetchall():
                p = dict(zip(cols, row))
                for k in ("entry_price", "invalidation_price", "target_price", "structural_stop", "t1"):
                    p[k] = _num(p[k])
                plans.append(p)
            tickers = sorted({p["ticker"] for p in plans})
            cur.execute("SELECT ticker, date, high, low, close FROM price_history "
                        "WHERE ticker = ANY(%s) AND date >= %s ORDER BY ticker, date",
                        (tickers, since - timedelta(days=7)))
            bars: dict = {}
            for t, d, h, l, c in cur.fetchall():
                if None in (h, l, c):
                    continue
                bars.setdefault(t, []).append({"date": d, "high": float(h), "low": float(l),
                                               "close": float(c)})
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return plans, bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-28", help="YYYY-MM-DD, New York date")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since = date.fromisoformat(args.since)
    from config import TRADING_ENV
    plans, bars = _fetch(since, TRADING_ENV)
    try:
        head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    text = build(plans, bars, since, TRADING_ENV, head, datetime.now(timezone.utc))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
