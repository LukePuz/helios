"""Compiled plan levels readout: which of compile_plan's levels the price
crossed during each plan's life, before the position's exit, and the forward
excess return after a crossing.

Engineering diagnostic (pilot phase), not a study analysis. For each trade
plan since --since (default 2026-10-28) with compiled levels (levels_source
set: NULL for held_seed plans and before 11-04) it reads the cached daily
bars (price_history; no vendor or broker call) and finds, per level, the
first full session inside the plan's life where the level was crossed:
  - structural_stop: low at or below it (close: close at or below it);
  - t1, t2, add_trigger: high at or above it (close: close at or above it);
  - time_stop: the first session after horizon_end.
Two crossing rules, intraday (high/low) and close, are the only thresholds
tried. A plan's life is the full sessions after created_at (a session counts
when its 09:30 ET open is at or after created_at) up to closed_at or now (a
session counts when its 16:00 ET close is at or before the end). The exit is
the first trade_outcomes exit of the name after created_at; a crossing is
"before the exit" when its session is before the exit's ET date, "exit day"
on that date.

Plans are often revised within a day, so a second view takes, for each
session, the newest compiled plan in force at that day's 09:30 ET open and
checks its levels on that session's bar (intraday rule).

The forward table reuses scripts/forward_readout.py (collect, readout): each
name's first in-force crossing of t1 / structural_stop / add_trigger
(session 0 = the crossing session, so the return runs from that close),
5/10/20 sessions, raw and minus SPY over the same dates; the baseline is
every compiled plan at its creation session.
Percentages only: levels are shown as % from the plan's entry price, never
as prices.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/plan_level_readout.py [--since 2026-10-28] [--out FILE]

Read-only: one transaction opened READ ONLY, then rolled back.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

NY = ZoneInfo("America/New_York")
PRICE_LEVELS = (("structural_stop", "down"), ("t1", "up"), ("t2", "up"), ("add_trigger", "up"))
LEVELS = tuple(n for n, _ in PRICE_LEVELS) + ("time_stop",)
FORWARD_LEVELS = ("t1", "structural_stop", "add_trigger")
SESSION_OPEN, SESSION_CLOSE = time(9, 30), time(16, 0)


def _forward_readout():
    spec = importlib.util.spec_from_file_location("plan_level_forward_readout",
                                                  os.path.join(HERE, "forward_readout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _aware(t):
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    if isinstance(t, date) and not isinstance(t, datetime):
        return datetime.combine(t, SESSION_CLOSE, tzinfo=NY)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def window(bars: list[dict], created_at, end) -> list[dict]:
    """The full sessions inside [created_at, end]: bars dicts date/high/low/close."""
    c, e = _aware(created_at), _aware(end)
    out = []
    for b in bars:
        d = b["date"]
        if datetime.combine(d, SESSION_OPEN, tzinfo=NY) >= c and \
                datetime.combine(d, SESSION_CLOSE, tzinfo=NY) <= e:
            out.append(b)
    return out


def first_crossings(plan: dict, sessions: list[dict]) -> dict:
    """{level: {"intraday": (index, date) | None, "close": (index, date) | None}}
    for each level the plan has; index 0 is the plan's first full session."""
    out = {}
    for name, direction in PRICE_LEVELS:
        level = plan.get(name)
        if level is None or level <= 0:
            continue
        rec = {"intraday": None, "close": None}
        for i, b in enumerate(sessions):
            hit_i = b["low"] <= level if direction == "down" else b["high"] >= level
            hit_c = b["close"] <= level if direction == "down" else b["close"] >= level
            if hit_i and rec["intraday"] is None:
                rec["intraday"] = (i, b["date"])
            if hit_c and rec["close"] is None:
                rec["close"] = (i, b["date"])
        out[name] = rec
    end = plan.get("horizon_end")
    if isinstance(end, date):
        hit = next(((i, b["date"]) for i, b in enumerate(sessions) if b["date"] > end), None)
        out["time_stop"] = {"intraday": hit, "close": hit}
    return out


def exit_after(outcomes, ticker: str, created_at):
    """The first exit time of the name at or after created_at, else None.
    outcomes: (ticker, exit_at)."""
    c = _aware(created_at)
    times = [_aware(x) for t, x in outcomes if t == ticker and x is not None and _aware(x) >= c]
    return min(times) if times else None


def analyse(plans: list[dict], bars_by_ticker: dict, outcomes, now: datetime) -> list[dict]:
    """One record per plan: sessions, crossings, exit, each level's % from entry."""
    out = []
    for p in plans:
        bars = sorted(bars_by_ticker.get(p["ticker"], []), key=lambda b: b["date"])
        end = p.get("closed_at") or now
        sess = window(bars, p["created_at"], end)
        ex = exit_after(outcomes, p["ticker"], p["created_at"])
        entry = p.get("entry_price")
        pct = {n: (round((p[n] / entry - 1) * 100, 2) if p.get(n) and entry else None)
               for n, _ in PRICE_LEVELS}
        out.append({"plan": p, "sessions": sess, "n_sessions": len(sess),
                    "crossings": first_crossings(p, sess), "exit_at": ex, "pct_from_entry": pct})
    return out


def _vs_exit(hit, exit_at) -> str:
    if hit is None:
        return "not crossed"
    if exit_at is None:
        return "no exit"
    xd = exit_at.astimezone(NY).date()
    return "before exit" if hit[1] < xd else "exit day" if hit[1] == xd else "after exit"


def in_force(plans: list[dict], bars_by_ticker: dict, outcomes, now: datetime) -> dict:
    """Per name, the sessions where a compiled plan was in force at the open
    (created before that day's 09:30 ET, not closed before it; the newest
    such plan), with that plan's crossings that session:
    {ticker: [{"date", "plan_id", "crossed": {level: True/False}}]}. Plans
    revised within a day are mostly never in force at an open."""
    out = {}
    for t in sorted({p["ticker"] for p in plans}):
        mine = sorted((p for p in plans if p["ticker"] == t), key=lambda p: _aware(p["created_at"]))
        days = []
        for b in sorted(bars_by_ticker.get(t, []), key=lambda b: b["date"]):
            op = datetime.combine(b["date"], SESSION_OPEN, tzinfo=NY)
            if datetime.combine(b["date"], SESSION_CLOSE, tzinfo=NY) > now:
                continue
            live = [p for p in mine if _aware(p["created_at"]) < op
                    and (p.get("closed_at") is None or _aware(p["closed_at"]) > op)]
            if not live:
                continue
            p = live[-1]
            crossed = {}
            for name, direction in PRICE_LEVELS:
                lv = p.get(name)
                if lv:
                    crossed[name] = b["low"] <= lv if direction == "down" else b["high"] >= lv
            if isinstance(p.get("horizon_end"), date):
                crossed["time_stop"] = b["date"] > p["horizon_end"]
            days.append({"date": b["date"], "plan_id": p["id"], "crossed": crossed})
        out[t] = days
    return out


def in_force_summary(view: dict, outcomes) -> list[dict]:
    """Per level: sessions in force, sessions crossed, names with a crossing,
    and each name's first crossing vs its first exit after that plan."""
    rows = []
    for name in LEVELS:
        n_sess = n_hit = 0
        firsts = {}
        for t, days in view.items():
            for d in days:
                if name not in d["crossed"]:
                    continue
                n_sess += 1
                if d["crossed"][name]:
                    n_hit += 1
                    firsts.setdefault(t, d["date"])
        vs = []
        for t, d0 in firsts.items():
            ex = [x for tk, x in outcomes if tk == t and x is not None
                  and _aware(x).astimezone(NY).date() >= d0]
            xd = min(_aware(x) for x in ex).astimezone(NY).date() if ex else None
            vs.append("no exit" if xd is None else "before exit" if d0 < xd else "exit day")
        rows.append({"level": name, "sessions": n_sess, "crossed": n_hit, "names": len(firsts),
                     "firsts": firsts, "vs_exit": vs})
    return rows


def forward_rows(recs: list[dict], summary: list[dict]) -> list[tuple]:
    """(symbol, label, ts, decision) rows for forward_readout.collect: each
    name's first in-force intraday crossing of t1 / structural_stop /
    add_trigger at that session's 15:00 ET (so session 0 is the crossing
    session), and every compiled plan at its created_at (baseline)."""
    rows = [(r["plan"]["ticker"], "plan created (baseline)", _aware(r["plan"]["created_at"]), None)
            for r in recs]
    for s in summary:
        if s["level"] in FORWARD_LEVELS:
            for t, d0 in s["firsts"].items():
                rows.append((t, f"{s['level']} crossed (in force, intraday)",
                             datetime.combine(d0, time(15), tzinfo=NY), None))
    return rows


def readout(recs: list[dict], summary: list[dict], fwd_md: str, since: date, env: str,
            head: str, excluded: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tickers = {r["plan"]["ticker"] for r in recs}
    lines = [f"# Compiled plan levels readout ({env}, plans since {since})", "",
             f"Generated {now} at code {head}. Pilot-phase engineering diagnostic, not a study "
             f"result. Plans with compiled levels: {len(recs)} on {len(tickers)} names "
             f"({excluded} plans since {since} have none: held_seed or before 11-04). Two "
             "crossing rules tried (intraday high/low, close). Levels as % from the plan's entry "
             "price; no prices.", "",
             "## Per level", "",
             "| level | plans with it | crossed intraday | crossed on a close | median sessions to "
             "first intraday crossing | before exit | exit day | after exit | no exit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in LEVELS:
        have = [r for r in recs if name in r["crossings"]]
        hi = [r for r in have if r["crossings"][name]["intraday"]]
        hc = [r for r in have if r["crossings"][name]["close"]]
        idx = [r["crossings"][name]["intraday"][0] for r in hi]
        vs = [_vs_exit(r["crossings"][name]["intraday"], r["exit_at"]) for r in hi]
        n = len(have)
        pc = lambda k: f"{k} ({k / n:.0%})" if n else "0"
        lines.append(f"| {name} | {n} | {pc(len(hi))} | {pc(len(hc))} | "
                     f"{statistics.median(idx) if idx else '-'} | {vs.count('before exit')} | "
                     f"{vs.count('exit day')} | {vs.count('after exit')} | {vs.count('no exit')} |")
    lines += ["", "## Levels in force at the open", "",
              "Each session's levels from the newest compiled plan in force at 09:30 ET (intraday "
              "rule). First crossing per name, against the name's first exit on or after it.", "",
              "| level | sessions in force | sessions crossed | names crossed | first crossing "
              "before exit | on exit day | no exit |", "|---|---|---|---|---|---|---|"]
    for s_ in summary:
        n = s_["sessions"]
        lines.append(f"| {s_['level']} | {n} | {s_['crossed']}"
                     + (f" ({s_['crossed'] / n:.0%})" if n else "")
                     + f" | {s_['names']} | {s_['vs_exit'].count('before exit')} | "
                     f"{s_['vs_exit'].count('exit day')} | {s_['vs_exit'].count('no exit')} |")
    lines += ["", "## Per plan", "",
              "| plan | name | status | sessions | stop % | t1 % | t2 % | add % | first intraday "
              "crossing (session index) | exit |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: (r["plan"]["ticker"], r["plan"]["id"])):
        p, c = r["plan"], r["crossings"]
        f = lambda v: "-" if v is None else f"{v:+.1f}"
        hits = ", ".join(f"{k} {c[k]['intraday'][0]}" for k in LEVELS
                         if k in c and c[k]["intraday"]) or "none"
        ex = r["exit_at"].astimezone(NY).strftime("%m-%d") if r["exit_at"] else "-"
        pf = r["pct_from_entry"]
        lines.append(f"| {p['id']} | {p['ticker']} | {p.get('status')} | {r['n_sessions']} | "
                     f"{f(pf['structural_stop'])} | {f(pf['t1'])} | {f(pf['t2'])} | "
                     f"{f(pf['add_trigger'])} | {hits} | {ex} |")
    lines += ["", "## Forward return after a crossing", "",
              "First in-force crossing per name and level (session 0 = the crossing session); "
              "baseline: every compiled plan at its creation session, one per name per session 0.", "",
              "\n".join("##" + l if l.startswith("#") else l for l in fwd_md.splitlines())]
    return "\n".join(lines) + "\n"


def _fetch(since: date, env: str):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT id, ticker, status, source, created_at, closed_at, close_reason, "
                        "entry_price, structural_stop, t1, t2, add_trigger, horizon_end, revision, "
                        "levels_source FROM trade_plans WHERE trading_env = %s AND created_at >= %s",
                        (env, start))
            cols = ("id", "ticker", "status", "source", "created_at", "closed_at", "close_reason",
                    "entry_price", "structural_stop", "t1", "t2", "add_trigger", "horizon_end",
                    "revision", "levels_source")
            plans = []
            for row in cur.fetchall():
                p = dict(zip(cols, row))
                for k in ("entry_price", "structural_stop", "t1", "t2", "add_trigger"):
                    p[k] = float(p[k]) if p[k] is not None else None
                plans.append(p)
            tickers = sorted({p["ticker"] for p in plans} | {"SPY"})
            cur.execute("SELECT ticker, COALESCE(exit_fill_at, exit_date) FROM trade_outcomes "
                        "WHERE trading_env = %s AND ticker = ANY(%s) AND exit_date IS NOT NULL "
                        "AND COALESCE(exit_fill_at, exit_date) >= %s", (env, tickers, start))
            outcomes = cur.fetchall()
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
    return plans, outcomes, bars


def build(plans, outcomes, bars, since: date, env: str, head: str, now: datetime) -> str:
    compiled = [p for p in plans if p.get("levels_source")]
    recs = analyse(compiled, bars, outcomes, now)
    summary = in_force_summary(in_force(compiled, bars, outcomes, now), outcomes)
    fr = _forward_readout()
    closes = {t: [(b["date"], b["close"]) for b in bs] for t, bs in bars.items()}
    fwd = fr.readout(fr.collect(forward_rows(recs, summary), closes,
                                label_fn=lambda kind, dec: kind), since, env, head)
    return readout(recs, summary, fwd, since, env, head, len(plans) - len(compiled))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-28", help="YYYY-MM-DD, New York date")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since = date.fromisoformat(args.since)
    from config import TRADING_ENV
    plans, outcomes, bars = _fetch(since, TRADING_ENV)
    try:
        head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    text = build(plans, outcomes, bars, since, TRADING_ENV, head, datetime.now(timezone.utc))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
