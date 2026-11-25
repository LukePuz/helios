"""Compiled plan vs placed stop readout, per closed lot.

Engineering diagnostic (pilot phase), not a study analysis. For each lot in
trade_outcomes (closed, entry on or after --since, default 2026-08-17) it
takes the compile_plan record nearest the lot's entry:
  1. the approval's risk_json.plan_compiled (compile_plan run at sizing,
     agents/crew.py:_compiled_plan), which carries the placed stop
     (actual_stop) and stop_gap_pct;
  2. else the newest trade_plans row of the name with compiled levels
     (levels_source set) created at or before the entry; the placed stop
     is then the approval's risk_json.stop_loss_price and stop_gap_pct is
     computed the same way, against the plan's entry_price;
  3. else none (the lot is counted as having no record).
Neither record is recomputed: compile_plan is never re-run on old data.

Prices come from the cached daily bars (price_history: date, high, low,
close; no vendor or broker call). Daily bars can't order events inside one
session, so a lot's life is the sessions after its entry day up to and
including its exit day, and two events on the same session are reported
as "same session" rather than ordered.

Sections, all counts and percentages, levels as % from the record's price:
  - stop_gap_pct, (placed - structural) / price * 100: + means the placed
    stop is tighter (structural below it), - means structural above it.
  - stopped-out lots (exit_source broker_exit: a resting stop fill, or a
    sale made at the broker; the table doesn't separate the two): the
    lot's later high is the highest high after the exit session up to the
    latest cached bar; whether any session low from the first life session
    up to the session before that high traded at or below the structural
    stop (the high's own session is flagged apart: its order is unknown).
  - t1: first session high at or above t1, against the stop-out session
    (lot life) and against the structural stop's first crossing (low at
    or below it, entry day + 1 up to the latest bar).
  - add_trigger: first session close at or above it inside the lot's life
    (the tranche2 shadow's definition); after it, the lot's exit source,
    the exit vs that close and the highest high / lowest low between.
  - time_stop: whether the date passed before the exit day.
No thresholds are chosen here: distributions are described and any count
with n < 5 is marked insufficient.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/compile_plan_readout.py [--since 2026-08-17] [--out FILE]

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
STOPPED = "broker_exit"
LEVELS = ("structural_stop", "t1", "t2", "add_trigger")


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
    if isinstance(t, date) and not isinstance(t, datetime):
        return datetime.combine(t, time(16), tzinfo=NY)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _et_day(t) -> date | None:
    t = _aware(t)
    return t.astimezone(NY).date() if t else None


def _day(v) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10]) if v else None
    except ValueError:
        return None


def pick_record(lot: dict, approval: dict | None, plans: list[dict]) -> dict | None:
    """The compile_plan record nearest the lot's entry, normalised:
    {source, recorded_at, price, structural_stop, t1, t2, add_trigger,
    time_stop, actual_stop, stop_gap_pct}, or None."""
    risk = (approval or {}).get("risk_json")
    risk = risk if isinstance(risk, dict) else {}
    pc = risk.get("plan_compiled")
    entry_at = _aware(lot["entry_date"])
    if isinstance(pc, dict) and not pc.get("error"):
        stop, actual = _num(pc.get("structural_stop")), _num(pc.get("actual_stop"))
        # The price compile_plan ran at; else back out of structural_stop_pct.
        price = _num(risk.get("price_at_sizing"))
        pct = _num(pc.get("structural_stop_pct"))
        if not price and pct is not None and stop and pct < 100:
            price = stop / (1 - pct / 100)
        price = price or _num(lot.get("entry_price"))
        gap = _num(pc.get("stop_gap_pct"))
        if gap is None and actual and stop and price:
            gap = round((actual - stop) / price * 100, 2)
        return {"source": "sizing", "recorded_at": _aware((approval or {}).get("timestamp")),
                "price": price, "structural_stop": stop, "t1": _num(pc.get("t1")),
                "t2": _num(pc.get("t2")), "add_trigger": _num(pc.get("add_trigger")),
                "time_stop": _day(pc.get("time_stop")), "actual_stop": actual, "stop_gap_pct": gap}
    mine = [p for p in plans if p["ticker"] == lot["ticker"] and p.get("levels_source")
            and entry_at and _aware(p["created_at"]) <= entry_at]
    if not mine:
        return None
    p = max(mine, key=lambda p: _aware(p["created_at"]))
    price, stop = _num(p.get("entry_price")), _num(p.get("structural_stop"))
    actual = _num(risk.get("stop_loss_price"))
    gap = round((actual - stop) / price * 100, 2) if actual and stop and price else None
    return {"source": "trade_plans", "recorded_at": _aware(p["created_at"]), "price": price,
            "structural_stop": stop, "t1": _num(p.get("t1")), "t2": _num(p.get("t2")),
            "add_trigger": _num(p.get("add_trigger")), "time_stop": _day(p.get("horizon_end")),
            "actual_stop": actual, "stop_gap_pct": gap}


def _first(sessions: list[dict], hit) -> int | None:
    return next((i for i, b in enumerate(sessions) if hit(b)), None)


def analyse_lot(lot: dict, rec: dict | None, bars: list[dict]) -> dict:
    """One lot's facts from its record and the name's daily bars."""
    bars = sorted(bars, key=lambda b: b["date"])
    entry_d = _et_day(lot["entry_date"])
    exit_d = _et_day(lot.get("exit_at") or lot.get("exit_date"))
    life = [b for b in bars if entry_d < b["date"] <= exit_d] if exit_d else \
        [b for b in bars if b["date"] > entry_d]
    after_entry = [b for b in bars if b["date"] > entry_d]
    out = {"lot": lot, "rec": rec, "life_sessions": len(life),
           "stopped": lot.get("exit_source") == STOPPED, "bars_cover_exit":
           bool(exit_d and bars and bars[-1]["date"] >= exit_d)}
    if not rec:
        return out
    price = rec.get("price")
    out["pct"] = {k: (round((rec[k] / price - 1) * 100, 2) if rec.get(k) and price else None)
                  for k in LEVELS + ("actual_stop",)}
    stop, t1, add = rec.get("structural_stop"), rec.get("t1"), rec.get("add_trigger")

    # t1 against the stop-out session (lot life) and the structural stop.
    if t1:
        i_t1 = _first(life, lambda b: b["high"] >= t1)
        if out["stopped"] and life and life[-1]["date"] == exit_d:
            stop_i = len(life) - 1
            out["t1_vs_stopout"] = ("no t1 touch" if i_t1 is None else
                                    "same session" if i_t1 == stop_i else "t1 first")
        elif out["stopped"]:
            out["t1_vs_stopout"] = "no bar for the exit day"
        else:
            out["t1_vs_stopout"] = "t1 touched, not stopped" if i_t1 is not None \
                else "no t1 touch, not stopped"
        if stop:
            j_t1 = _first(after_entry, lambda b: b["high"] >= t1)
            j_st = _first(after_entry, lambda b: b["low"] <= stop)
            out["t1_vs_structural"] = (
                "neither" if j_t1 is None and j_st is None else
                "structural first" if j_t1 is None or (j_st is not None and j_st < j_t1) else
                "t1 first" if j_st is None or j_t1 < j_st else "same session")

    # Stopped-out lots: did price trade through the structural stop before
    # the later high?
    if out["stopped"] and stop and exit_d:
        post = [b for b in bars if b["date"] > exit_d]
        if post:
            hi = max(post, key=lambda b: (b["high"], -b["date"].toordinal()))
            before = [b for b in after_entry if b["date"] < hi["date"]]
            out["later_high"] = {
                "date": hi["date"], "sessions_after_exit": post.index(hi) + 1,
                "pct_vs_price": round((hi["high"] / price - 1) * 100, 2) if price else None,
                "above_t1": bool(t1 and hi["high"] >= t1),
                "structural_breached_before": any(b["low"] <= stop for b in before),
                "structural_breached_on_high_day": hi["low"] <= stop}
        else:
            out["later_high"] = None

    # add_trigger: first close at or over it inside the lot's life.
    if add:
        k = _first(life, lambda b: b["close"] >= add)
        if k is None:
            out["add"] = {"fired": False}
        else:
            fire = life[k]
            rest = life[k + 1:]
            ex = _num(lot.get("exit_price"))
            c = fire["close"]
            out["add"] = {
                "fired": True, "session": k, "date": fire["date"],
                "exit_source": lot.get("exit_source") or "unrecorded",
                "exit_vs_close_pct": round((ex / c - 1) * 100, 2) if ex else None,
                "max_high_pct": round((max(b["high"] for b in rest) / c - 1) * 100, 2) if rest else None,
                "min_low_pct": round((min(b["low"] for b in rest) / c - 1) * 100, 2) if rest else None,
                "structural_after": bool(stop and any(b["low"] <= stop for b in rest))}

    ts = rec.get("time_stop")
    if ts and exit_d:
        out["time_stop_passed"] = ts < exit_d
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


def _tally(labels: list[str], order: tuple) -> list[str]:
    n = len(labels)
    return [f"| {lab} | {count(labels.count(lab), n)} |" for lab in order]


def readout(recs: list[dict], open_recs: list[dict], since: date, env: str, head: str,
            now: datetime) -> str:
    have = [r for r in recs if r["rec"]]
    names = sorted({r["lot"]["ticker"] for r in have})
    src = [r["rec"]["source"] for r in have]
    last_bar = max((r.get("last_bar") for r in recs if r.get("last_bar")), default=None)
    lines = [f"# Compiled plan vs placed stop ({env}, closed lots with entry since {since})", "",
             f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} at code {head}. Pilot-phase "
             "engineering diagnostic, not a study result. Prices: price_history daily bars "
             f"(latest bar {last_bar or '-'}). Levels as % from the record's price; no prices. "
             f"Counts with n < {MIN_N} are marked insufficient. No thresholds chosen.", "",
             f"Closed lots: {len(recs)}. With a compile_plan record: {len(have)} on "
             f"{len(names)} names ({src.count('sizing')} from the sizing record, "
             f"{src.count('trade_plans')} from trade_plans). Without one: {len(recs) - len(have)} "
             "(entered before the records existed). Lots of one name that exit together share "
             "the same bars, so they aren't independent.", "",
             "## 1. Structural stop vs placed stop at entry", "",
             "stop_gap_pct = (placed - structural) / price * 100; + = placed tighter "
             "(structural below it), - = structural above it.", ""]
    for label, rows in (("closed lots", have), ("open lots (not in the sections below)",
                                                 [r for r in open_recs if r["rec"]])):
        gaps = [r["rec"]["stop_gap_pct"] for r in rows if r["rec"].get("stop_gap_pct") is not None]
        n = len(gaps)
        lines += [f"- {label}: structural below placed {count(sum(g > 0 for g in gaps), n)}, "
                  f"above {count(sum(g < 0 for g in gaps), n)}, equal "
                  f"{count(sum(g == 0 for g in gaps), n)}; no comparison "
                  f"{len(rows) - n}. Distribution: {describe(gaps)}."]

    stopped = [r for r in have if r["stopped"]]
    lh = [r for r in stopped if r.get("later_high")]
    lines += ["", "## 2. Stopped-out lots: structural stop before the later high", "",
              f"Stopped-out (broker_exit) lots with a record: {len(stopped)}; with bars after the "
              f"exit: {len(lh)}. Later high = highest high after the exit session up to the "
              "latest bar.", "",
              "| | lots |", "|---|---|"]
    n = len(lh)
    lines += [f"| structural stop traded through before the later high | "
              f"{count(sum(r['later_high']['structural_breached_before'] for r in lh), n)} |",
              f"| not traded through before it | "
              f"{count(sum(not r['later_high']['structural_breached_before'] for r in lh), n)} |",
              f"| low on the later high's own session at or below it (order unknown) | "
              f"{count(sum(r['later_high']['structural_breached_on_high_day'] for r in lh), n)} |",
              f"| later high at or above t1 | {count(sum(r['later_high']['above_t1'] for r in lh), n)} |",
              "", f"Later high vs the record's price (%): "
              f"{describe([r['later_high']['pct_vs_price'] for r in lh if r['later_high']['pct_vs_price'] is not None])}. "
              f"Sessions from exit to the later high: "
              f"{describe([float(r['later_high']['sessions_after_exit']) for r in lh])}."]

    t1s = [r["t1_vs_stopout"] for r in have if "t1_vs_stopout" in r]
    t1x = [r["t1_vs_structural"] for r in have if "t1_vs_structural" in r]
    lines += ["", "## 3. t1 touched before the stop", "",
              f"Against the placed stop (lot life, entry day excluded; {len(t1s)} lots with t1):", "",
              "| | lots |", "|---|---|"]
    lines += _tally(t1s, ("t1 first", "same session", "no t1 touch", "t1 touched, not stopped",
                          "no t1 touch, not stopped", "no bar for the exit day"))
    lines += ["", f"Against the structural stop's first crossing (entry day + 1 to the latest bar; "
              f"{len(t1x)} lots with both):", "", "| | lots |", "|---|---|"]
    lines += _tally(t1x, ("t1 first", "same session", "structural first", "neither"))

    adds = [r["add"] for r in have if "add" in r]
    fired = [a for a in adds if a["fired"]]
    lines += ["", "## 4. add_trigger (first close at or over it during the lot)", "",
              f"Lots with an add_trigger: {len(adds)}; fired: {count(len(fired), len(adds))}.", ""]
    if fired:
        srcs = sorted({a["exit_source"] for a in fired})
        lines += [f"- exit after firing: " + ", ".join(
                      f"{s} {count(sum(a['exit_source'] == s for a in fired), len(fired))}" for s in srcs),
                  f"- exit vs the firing close (%): "
                  f"{describe([a['exit_vs_close_pct'] for a in fired if a['exit_vs_close_pct'] is not None])}",
                  f"- highest high after firing (%): "
                  f"{describe([a['max_high_pct'] for a in fired if a['max_high_pct'] is not None])}",
                  f"- lowest low after firing (%): "
                  f"{describe([a['min_low_pct'] for a in fired if a['min_low_pct'] is not None])}",
                  f"- structural stop traded through after firing: "
                  f"{count(sum(a['structural_after'] for a in fired), len(fired))}"]

    tsp = [r["time_stop_passed"] for r in have if "time_stop_passed" in r]
    lines += ["", "## 5. time_stop", "",
              f"time_stop date passed before the exit day: {count(sum(tsp), len(tsp))}.", "",
              "## Per lot", "",
              "| lot | name | record | stop % | placed % | gap | t1 % | add % | sessions | exit | "
              "t1 vs stop-out | t1 vs structural | later high: structural through before | add fired |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:+.1f}"
    for r in sorted(recs, key=lambda r: (_aware(r["lot"]["entry_date"]), r["lot"]["id"])):
        lot, rec = r["lot"], r["rec"]
        ex = lot.get("exit_source") or "unrecorded"
        if not rec:
            lines.append(f"| {lot['id']} | {lot['ticker']} | none | - | - | - | - | - | "
                         f"{r['life_sessions']} | {ex} | - | - | - | - |")
            continue
        pc, lhr, a = r["pct"], r.get("later_high"), r.get("add")
        lines.append(
            f"| {lot['id']} | {lot['ticker']} | {rec['source']} | {f(pc['structural_stop'])} | "
            f"{f(pc['actual_stop'])} | {f(rec.get('stop_gap_pct'))} | {f(pc['t1'])} | "
            f"{f(pc['add_trigger'])} | {r['life_sessions']} | {ex} | {r.get('t1_vs_stopout', '-')} | "
            f"{r.get('t1_vs_structural', '-')} | "
            f"{'-' if not lhr else 'yes' if lhr['structural_breached_before'] else 'no'} | "
            f"{'-' if not a else 'yes' if a['fired'] else 'no'} |")
    return "\n".join(lines) + "\n"


def build(lots, open_lots, approvals, plans, bars, since, env, head, now) -> str:
    def run(rows):
        out = []
        for lot in rows:
            rec = pick_record(lot, approvals.get(lot.get("approval_id")), plans)
            b = bars.get(lot["ticker"], [])
            r = analyse_lot(lot, rec, b)
            r["last_bar"] = max((x["date"] for x in b), default=None)
            out.append(r)
        return out
    return readout(run(lots), run(open_lots), since, env, head, now)


def _fetch(since: date, env: str):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cols = ("id", "ticker", "action", "approval_id", "entry_date", "entry_price",
                    "exit_date", "exit_at", "exit_price", "exit_source")
            cur.execute("SELECT id, ticker, action, approval_id, entry_date, entry_price, exit_date, "
                        "COALESCE(exit_fill_at, exit_date), exit_price, exit_source "
                        "FROM trade_outcomes WHERE trading_env = %s AND entry_date >= %s",
                        (env, start))
            all_lots = [dict(zip(cols, r)) for r in cur.fetchall()]
            for lot in all_lots:
                for k in ("entry_price", "exit_price"):
                    lot[k] = _num(lot[k])
            lots = [x for x in all_lots if x["exit_date"] is not None]
            open_lots = [x for x in all_lots if x["exit_date"] is None]
            ids = sorted({int(x["approval_id"]) for x in all_lots if x["approval_id"]})
            cur.execute("SELECT id, timestamp, risk_json FROM pending_approvals WHERE id = ANY(%s)",
                        (ids,))
            approvals = {i: {"id": i, "timestamp": ts, "risk_json": rj} for i, ts, rj in cur.fetchall()}
            tickers = sorted({x["ticker"] for x in all_lots})
            pcols = ("id", "ticker", "created_at", "entry_price", "structural_stop", "t1", "t2",
                     "add_trigger", "horizon_end", "levels_source")
            cur.execute("SELECT id, ticker, created_at, entry_price, structural_stop, t1, t2, "
                        "add_trigger, horizon_end, levels_source FROM trade_plans "
                        "WHERE trading_env = %s AND ticker = ANY(%s) AND levels_source IS NOT NULL",
                        (env, tickers))
            plans = [dict(zip(pcols, r)) for r in cur.fetchall()]
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
    return lots, open_lots, approvals, plans, bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-17", help="YYYY-MM-DD, New York date")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since = date.fromisoformat(args.since)
    from config import TRADING_ENV
    lots, open_lots, approvals, plans, bars = _fetch(since, TRADING_ENV)
    try:
        head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    text = build(lots, open_lots, approvals, plans, bars, since, TRADING_ENV, head,
                 datetime.now(timezone.utc))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
