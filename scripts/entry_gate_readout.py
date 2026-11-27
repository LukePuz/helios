"""Live entry gate readout (roadmap 04aw): what the entry gates blocked, next
to the buys that got through, and how the blocked names moved afterwards.
Engineering readout, not a study analysis; live trading_env only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/entry_gate_readout.py [--since 2026-11-23] [--until 2026-11-30] [--out FILE]

--since / --until are NYSE sessions (ET dates, inclusive). --since defaults to
2026-11-23, the gates' first live session; --until to the last closed session.

Per session, from rows the bot already writes:
  blocked   ENTRY_GATE_BLOCKED signals rows, split by the gate(s) whose
            decision.entry_gate.<stopout|trend|add_loser>.blocked is True
            ("stopout+trend" when both fired, "unknown" when none says so),
            and by entry_gate.path (the routed buy, or the deferred retry).
            A row counts toward its session as in entry_funnel.py.
  passed    buys past every gate, exactly as entry_funnel.py counts them:
            OPEN/ADD approvals minus those the deferred retry's gate expired.
  share     blocked / (blocked + passed).
The market read: every row's trend half, tallied by what the market check
(entry_gate.trend.market, the trend gate's market ticker vs its 200-day)
said. A trend block needs the market below its 200-day; a row written by
another gate can show a stock below both averages that the market check
stood down.

Forward moves of the blocked names, as the 11-21 gap hunt and
forward_readout.py define them (their functions, imported): one row per name
per session 0, where session 0 is the row's New York date if written before
16:00 ET on a cached session, else the next one; the move at k sessions is
close(0+k) / close(0) - 1 on the name's own price_history bars, minus SPY over
the same two dates. Shown at 5 and 20 sessions only (the gap hunt's
horizons; this script adds none), as mean and median, by gate and for all
blocked rows. A row whose k-th bar is not cached yet is pending; a row whose
session 0 is past the last cached bar is left out until the nightly cache
has it.

Reclaim shadow (record only; agents/crew.py:_check_entry_gate_reclaims), from
its activity_feed rows (stage entry_gate_reclaim_shadow, this env): per blocking
gate group, the block records seen, how many logged a reclaim (every blocking
gate would pass again on completed bars) and the sessions it took, and how many
closed (expired) without one, and how many a newer block of the same name
superseded (the shadow keeps one live record per name; each replaced record
gets a superseded row). Rows are deduped by shadow_key (a spill replay or a
skipped book update can repeat one). Counts and sessions only: the shadow's
own forward-move rows are not summarised here, since this script's horizons
are the gap hunt's. The forward rows are records; which horizon an evaluation
uses is a research choice (Luke). Which blocks an evaluation counts (first,
latest, all) is a research choice (Luke); superseded rows keep every block
countable.

Every figure carries its n, and a figure on fewer than 5 values prints
"insufficient (n=...)". Percentages and counts only: no prices, sizes or
account values. Which number decides anything is not this script's choice.

Read-only: one READ ONLY transaction for the gate rows and bars, one for the
reclaim shadow rows and one in entry_funnel.py for the funnel, each with a
statement timeout and rolled back; a failed reclaim read leaves only that
section out. Refuses to run unless SUPABASE_PORT is 6543 (the transaction pooler).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENV = "live"
GATES_LIVE_FROM = date(2026, 11, 23)
READ_PORT = 6543
# The 11-21 gap hunt's horizons and the n floor, both given with the task
# (11-23). Adding a horizon or changing the floor is a research choice.
READOUT_HORIZONS = (5, 20)
MIN_N = 5
GATES = ("stopout", "trend", "add_loser")
STATEMENT_TIMEOUT_MS = 10000
_MODULES: dict = {}


def _sibling(name: str):
    """scripts/<name>.py as a module (scripts/ is not a package)."""
    if name not in _MODULES:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_egr_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def _ef():
    return _sibling("entry_funnel")


def _fr():
    return _sibling("forward_readout")


# ---------------------------------------------------------------- rows

def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _gate(decision) -> dict:
    return _as_dict(_as_dict(decision).get("entry_gate"))


def reason_of(decision) -> str:
    """The gate(s) that blocked: 'stopout', 'trend', 'stopout+trend', ...;
    'unknown' when no half says blocked."""
    gate = _gate(decision)
    fired = [g for g in GATES if _as_dict(gate.get(g)).get("blocked") is True]
    return "+".join(fired) if fired else "unknown"


def path_of(decision) -> str:
    """entry_gate.path: 'routed buy' when absent (the routing path, and every
    row written before the deferred retry got the gate), else as written."""
    p = _gate(decision).get("path")
    return "routed buy" if p in (None, "", "route") else str(p)


def market_read_of(decision) -> str | None:
    """What the trend half's market check said on this row; None when the
    row has no trend half or the stock was not below both averages (the
    market is read only for a stock that would be blocked)."""
    trend = _as_dict(_gate(decision).get("trend"))
    if not trend:
        return None
    below_both = trend.get("below_sma50") is True and trend.get("below_sma200") is True
    if not below_both and trend.get("blocked") is not True:
        return None
    market = trend.get("market")
    if not isinstance(market, dict):
        return ("trend blocked, no market check" if trend.get("blocked") is True
                else "stock below both, no market check")
    if trend.get("blocked") is True:
        return "trend blocked: market below its 200-day"
    if str(market.get("read")) != "ok":
        return "stood down: market read failed"
    if market.get("below_sma200") is False:
        return "stood down: market above its 200-day"
    return "stood down: market read stale or incomplete"


def session_range(since: date, until: date) -> list[date]:
    """The NYSE sessions from since to until, inclusive, oldest first."""
    ef, out, d = _ef(), [], since
    while d <= until:
        if ef._is_session(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def tally(sessions: list[date], gate_rows, funnel: dict) -> dict:
    """gate_rows: (symbol, kind, ts, decision) of ENTRY_GATE_BLOCKED rows.
    funnel: entry_funnel.build_funnel's {session: counts}. Returns
    {session: {"blocked", "reasons", "paths", "names", "passed"}} plus "all",
    and "market": the market-read Counter over every row."""
    ef = _ef()
    out = {s: {"blocked": 0, "reasons": Counter(), "paths": Counter(), "names": set(),
               "passed": int(_as_dict(funnel.get(s)).get("passed") or 0)} for s in sessions}
    market = Counter()
    for symbol, _kind, ts, decision in gate_rows:
        s = ef.session_of(ts)
        if s not in out:
            continue
        f = out[s]
        f["blocked"] += 1
        f["reasons"][reason_of(decision)] += 1
        f["paths"][path_of(decision)] += 1
        f["names"].add(symbol)
        m = market_read_of(decision)
        if m is not None:
            market[m] += 1
    tot = {"blocked": 0, "reasons": Counter(), "paths": Counter(), "names": set(), "passed": 0}
    for s in sessions:
        for k in ("blocked", "passed"):
            tot[k] += out[s][k]
        for k in ("reasons", "paths"):
            tot[k].update(out[s][k])
        tot["names"] |= out[s]["names"]
    return {"sessions": out, "all": tot, "market": market}


def forward_groups(gate_rows, bars) -> dict[str, list[dict]]:
    """forward_readout.collect by gate reason, plus 'all blocked' (one row per
    name per session 0 across reasons)."""
    fr = _fr()
    groups = fr.collect(gate_rows, bars, label_fn=lambda _k, d: reason_of(d))
    allg = fr.collect(gate_rows, bars, label_fn=lambda _k, _d: "all blocked")
    groups.update(allg)
    return groups


RECLAIM_STAGE = "entry_gate_reclaim_shadow"


def reclaim_tally(rows) -> dict:
    """rows: extra_json dicts of the reclaim shadow's activity rows. One block
    record per shadow_key prefix (env:ticker:blocked_at), grouped by its gates
    ('trend', 'stopout+trend', ...): {group: {"records", "reclaimed",
    "sessions" (to reclaim, one per reclaimed record), "closed",
    "closed_unreclaimed", "superseded"}}. A repeated shadow_key counts once
    (the first row). Rows of another env are left out."""
    recs: dict[str, dict] = {}
    seen: set[str] = set()
    for x in rows:
        x = _as_dict(x)
        sk = str(x.get("shadow_key") or "")
        if not sk.startswith(f"{ENV}:") or ":" not in sk or sk in seen:
            continue
        seen.add(sk)
        ident = sk.rsplit(":", 1)[0]
        r = recs.setdefault(ident, {"gates": x.get("gates"), "reclaim": None, "expired": None,
                                    "superseded": None})
        if x.get("event") in ("reclaim", "expired", "superseded"):
            r[x["event"]] = x
    out: dict[str, dict] = {}
    for r in recs.values():
        g = "+".join(str(v) for v in (r["gates"] or [])) or "unknown"
        o = out.setdefault(g, {"records": 0, "reclaimed": 0, "sessions": [], "closed": 0,
                               "closed_unreclaimed": 0, "superseded": 0})
        o["records"] += 1
        if r["reclaim"] is not None:
            o["reclaimed"] += 1
            s = r["reclaim"].get("sessions")
            if isinstance(s, int) and not isinstance(s, bool):
                o["sessions"].append(s)
        if r["expired"] is not None:
            o["closed"] += 1
            if r["reclaim"] is None:
                o["closed_unreclaimed"] += 1
        if r["superseded"] is not None:
            o["superseded"] += 1
    return out


# ---------------------------------------------------------------- output

def figure(n: int, text: str) -> str:
    """text with its n; 'insufficient' below MIN_N."""
    return f"insufficient (n={n})" if n < MIN_N else f"{text} (n={n})"


def share(a: int, b: int) -> str:
    return figure(b, f"{a / b:.0%}") if b else "n/a (n=0)"


def move(vals: list[float]) -> str:
    if len(vals) < MIN_N:
        return figure(len(vals), "")
    return figure(len(vals), f"mean {statistics.mean(vals) * 100:+.1f}%, "
                             f"median {statistics.median(vals) * 100:+.1f}%")


def _reason_key(r: str):
    order = ("stopout", "trend", "stopout+trend", "add_loser")
    return (order.index(r) if r in order else len(order), r)


def readout(sessions: list[date], t: dict, groups: dict, head: str, now: datetime,
            cached_through: date | None, reclaims: dict | None = None,
            reclaim_read: str | None = None) -> str:
    sess, tot = t["sessions"], t["all"]
    reasons = sorted(tot["reasons"], key=_reason_key)
    lines = [f"# Live entry gates ({ENV}), sessions {sessions[0]}..{sessions[-1]}", "",
             f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} at code {head}. Engineering "
             "readout, not a study result. Counts per session (ET). passed = buys past every "
             f"gate, as entry_funnel.py counts them; share = blocked / (blocked + passed). "
             f"Figures on fewer than {MIN_N} values print insufficient.", "",
             "## Blocked vs passed", "",
             "| session | blocked | " + "".join(f"{r} | " for r in reasons)
             + "names | passed | blocked share |",
             "|---|---|" + "---|" * len(reasons) + "---|---|---|"]
    for s in sessions + ["all"]:
        f = tot if s == "all" else sess[s]
        lines.append(f"| {s} | {f['blocked']} | "
                     + "".join(f"{f['reasons'][r]} | " for r in reasons)
                     + f"{len(f['names'])} | {f['passed']} | "
                     f"{share(f['blocked'], f['blocked'] + f['passed'])} |")
    lines += ["", "By path: " + (", ".join(f"{k} {v}" for k, v in sorted(tot["paths"].items()))
                                 or "no rows") + "."]

    lines += ["", "## Market read on the trend half", "",
              "Rows where the stock was below both averages (the market is read only then).", ""]
    if t["market"]:
        n = sum(t["market"].values())
        lines += ["| market read | rows | share of these rows |", "|---|---|---|"]
        for k, v in sorted(t["market"].items()):
            lines.append(f"| {k} | {v} | {share(v, n)} |")
    else:
        lines.append("No row had a stock below both averages.")

    lines += ["", "## Forward moves of blocked names, minus SPY over the same dates", "",
              "One row per name per session 0 (gap hunt / forward_readout.py). "
              f"Price cache through {cached_through or 'unknown'}; rows past it are left out "
              "until it has them.", "",
              "| group | rows | " + "".join(f"{k} sessions | pending at {k} | "
                                           for k in READOUT_HORIZONS),
              "|---|---|" + "---|---|" * len(READOUT_HORIZONS)]
    labels = sorted((g for g in groups if g != "all blocked"), key=_reason_key)
    for g in labels + (["all blocked"] if "all blocked" in groups else []):
        recs = groups[g]
        cells = ""
        for k in READOUT_HORIZONS:
            xs = [r[f"x{k}"] for r in recs if r.get(f"x{k}") is not None]
            cells += f"{move(xs)} | {len(recs) - len(xs)} | "
        lines.append(f"| {g} | {len(recs)} | {cells}")
    if not groups:
        lines.append("| (no blocked rows with a cached session 0) | 0 | "
                     + "- | - | " * len(READOUT_HORIZONS))
    lines += ["", "Rows of one day move together, so a few sessions can look like a pattern; "
              "n counts names x sessions, not independent draws."]
    if reclaims is not None or reclaim_read:
        lines += ["", "## Reclaim shadow (record only)", "",
                  "Block records the nightly reclaim shadow has logged rows for: reclaimed = "
                  "every blocking gate would pass again on completed bars; closed = the record "
                  "expired; superseded = a newer block of the same name replaced the record. "
                  "Counts and sessions only. The forward rows are records; which horizon an "
                  "evaluation uses is a research choice (Luke). Which blocks an evaluation "
                  "counts (first, latest, all) is a research choice (Luke); superseded rows "
                  "keep every block countable.", ""]
        if reclaim_read:
            lines.append(f"Reclaim rows not read ({reclaim_read}).")
        elif not reclaims:
            lines.append("No reclaim shadow rows yet.")
        else:
            lines += ["| gates | records | reclaimed | sessions to reclaim | closed | "
                      "closed without a reclaim | superseded |", "|---|---|---|---|---|---|---|"]
            for g in sorted(reclaims, key=_reason_key):
                o = reclaims[g]
                ss = o["sessions"]
                sess_txt = (figure(len(ss), f"median {statistics.median(ss):g}")
                            if ss else "n/a (n=0)")
                lines.append(f"| {g} | {o['records']} | {o['reclaimed']} | {sess_txt} | "
                             f"{o['closed']} | {o['closed_unreclaimed']} | {o['superseded']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def _fetch(sessions: list[date]):
    """ENTRY_GATE_BLOCKED rows in the sessions' window and the price_history
    closes of their names and SPY. One READ ONLY transaction, rolled back."""
    import db
    start, end = _ef().window(sessions)
    bench = _fr().BENCHMARK
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT symbol, signal, timestamp, decision FROM signals "
                        "WHERE trading_env = %s AND signal = 'ENTRY_GATE_BLOCKED' "
                        "AND timestamp >= %s AND timestamp < %s", (ENV, start, end))
            rows = cur.fetchall()
            tickers = sorted({r[0] for r in rows} | {bench})
            cur.execute("SELECT ticker, date, close FROM price_history "
                        "WHERE ticker = ANY(%s) AND date >= %s",
                        (tickers, sessions[0] - timedelta(days=7)))
            bars: dict[str, list] = {}
            for tk, d, c in cur.fetchall():
                bars.setdefault(tk, []).append((d, c))
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return rows, bars


def _fetch_reclaims(since: date) -> list[dict]:
    """The reclaim shadow's activity rows written since the first session
    (rows come nightly after their block). One READ ONLY transaction, rolled
    back."""
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT extra_json FROM activity_feed WHERE stage = %s AND ts >= %s",
                        (RECLAIM_STAGE, since))
            return [r[0] for r in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)


def _bar_qa():
    """scripts/_bar_qa.py (the non-finite bar flag) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bar_qa.py")
    spec = importlib.util.spec_from_file_location("_bar_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=GATES_LIVE_FROM.isoformat(),
                    help="first session, YYYY-MM-DD (default: the gates' first live session)")
    ap.add_argument("--until", help="last session, YYYY-MM-DD (default: the last closed one)")
    ap.add_argument("--out", help="also write the markdown here")
    a = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT") != str(READ_PORT):
        print(f"entry_gate_readout: run with SUPABASE_PORT={READ_PORT} (the transaction pooler)",
              file=sys.stderr)
        return 2
    ef = _ef()
    since = date.fromisoformat(a.since)
    until = (date.fromisoformat(a.until) if a.until
             else ef.default_until(datetime.now(timezone.utc)))
    sessions = session_range(since, until)
    if not sessions:
        print(f"entry_gate_readout: no NYSE session in {since}..{until}", file=sys.stderr)
        return 2
    rows, bars = _fetch(sessions)
    funnel = ef.run(sessions, ENV)
    bench = bars.get(_fr().BENCHMARK) or []
    cached = max((d for d, c in bench if c is not None), default=None)
    reclaims, reclaim_read = None, None
    try:
        reclaims = reclaim_tally(_fetch_reclaims(sessions[0]))
    except Exception as e:
        reclaim_read = type(e).__name__
    text = readout(sessions, tally(sessions, rows, funnel), forward_groups(rows, bars),
                   ef._git_head(), datetime.now(timezone.utc), cached, reclaims, reclaim_read)
    text += "\n" + _bar_qa().flag_line(bars) + "\n"   # flag only (scripts/_bar_qa.py)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
