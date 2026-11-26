"""Position weight readout: each held name's weight in the live book per
NYSE session, and the sessions a name sat above 15% or above its bucket's cap.
Engineering readout, record only; live trading_env only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/position_weight_readout.py [--since 2026-08-17] [--until 2026-11-23] [--out FILE]

--since / --until are NYSE sessions (ET dates, inclusive). --since defaults to
the first session with a live portfolio snapshot, --until to the last closed
session.

Why: config.py's Compounder cap carries the comment "12% at entry; allow up
to 15% via gains" (Accelerator: "7% at entry; allow up to 8% via gains"), and
no code enforces either ceiling: sizing caps a buy at entry, and nothing
trims a name that grows past it. This readout only shows how often that has
happened. Whether to enforce, trim or drop the comment is Luke's call; the
script changes nothing.

From the stored portfolio_snapshots rows (the bot writes one about every 5
minutes), per session:
  in-session snapshots  those stamped from 09:30 ET to the session's close
                        (16:00, 13:00 on an early close; market_calendar.py).
                        Snapshots outside those hours are not read.
  close                 the last in-session snapshot of the session.
  weight                a name's market_value / the snapshot's total_value,
                        long positions with a price only, the rule
                        tasks._book_deployment uses (shorts and unpriced
                        names are counted, not weighed).
  > 15%                 weight above 15% (the config comment's Compounder
                        figure, applied to every name).
  > bucket cap          weight above the at-entry cap sizing uses for the
                        name's bucket at that snapshot's account tier
                        (config.get_active_limits(total_value), as
                        RiskManager.size_position reads it). The bucket is
                        the name's role in today's watchlist.md
                        (knowledge_loader.load_watchlist); role changes are
                        not dated, so a name whose role changed is read with
                        its current one. A name off the watchlist has no
                        bucket and is listed as such, never counted above a
                        cap.
Each is shown at the close and at any in-session snapshot. A session with no
in-session snapshot is listed as "no snapshot". Which reading (close or any
snapshot) or threshold counts for anything is a research choice (Luke).

Percentages and counts only: no dollar values, cash or share counts are
printed; the tier name is printed so the cap can be read.

Read-only: one READ ONLY transaction with a statement timeout, rolled back.
Refuses to run unless SUPABASE_PORT is 6543 (the transaction pooler).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from datetime import date, datetime, time, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENV = "live"
READ_PORT = 6543
STATEMENT_TIMEOUT_MS = 15000
# The figure in config.py's Compounder cap comment ("allow up to 15% via
# gains"), applied to every name as the task (11-23 16:56 audit) asked. Not a
# config constant: no code reads it.
VIA_GAINS_PCT = 15.0
BUCKETS = ("Compounder", "Accelerator", "Moonshot")
NO_BUCKET = "off watchlist"
_MODULES: dict = {}


def _ef():
    """scripts/entry_funnel.py as a module (scripts/ is not a package)."""
    if "entry_funnel" not in _MODULES:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_funnel.py")
        spec = importlib.util.spec_from_file_location("_pwr_entry_funnel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES["entry_funnel"] = mod
    return _MODULES["entry_funnel"]


# ---------------------------------------------------------------- pure

def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def weights(positions, pv) -> tuple[dict, dict]:
    """({TICKER: weight as a fraction}, {"n_short", "n_unpriced"}) from a
    snapshot's positions_json and total_value: long, priced names only, as
    tasks._book_deployment weighs them. ({}, counts) when pv is missing or
    not positive."""
    counts = {"n_short": 0, "n_unpriced": 0}
    pv = _num(pv)
    out: dict = {}
    if not pv or pv <= 0:
        return out, counts
    for t, p in sorted((positions if isinstance(positions, dict) else {}).items()):
        if not isinstance(p, dict):
            continue
        qty, mv = _num(p.get("qty")), _num(p.get("market_value"))
        if qty is not None and qty < 0:
            counts["n_short"] += 1
            continue
        if not qty:
            continue
        if mv is None or mv <= 0:
            counts["n_unpriced"] += 1
            continue
        out[str(t).upper()] = mv / pv
    return out, counts


def caps(pv) -> dict:
    """{"tier", "Compounder", "Accelerator", "Moonshot"}: the account tier
    and each bucket's at-entry cap (fractions) at this total_value, from
    config.get_active_limits. Tier None and caps None when pv is unusable."""
    pv = _num(pv)
    if not pv or pv <= 0:
        return {"tier": None, **{b: None for b in BUCKETS}}
    from config import get_active_limits
    lim = get_active_limits(pv)
    return {"tier": lim["tier"],
            "Compounder": lim["max_position_pct_compounder"],
            "Accelerator": lim["max_position_pct_accelerator"],
            "Moonshot": lim["max_position_pct_moonshot"]}


def session_close(d: date) -> time:
    from market_calendar import EARLY_CLOSE, NYSE_EARLY_CLOSES, RTH_CLOSE
    return EARLY_CLOSE if d in NYSE_EARLY_CLOSES else RTH_CLOSE


def in_session(snaps, sessions: list[date]) -> dict:
    """snaps: (timestamp, total_value, positions_json) rows. {session: [rows
    stamped from 09:30 ET to that session's close, oldest first]} for each
    session in sessions (an empty list when it has none)."""
    from market_calendar import RTH_OPEN
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    want = set(sessions)
    out: dict = {d: [] for d in sessions}
    for row in snaps:
        ts = row[0]
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        et = ts.astimezone(ny)
        d = et.date()
        if d in want and RTH_OPEN <= et.time() <= session_close(d):
            out[d].append((ts, row[1], row[2]))
    for d in out:
        out[d].sort(key=lambda r: r[0])
    return out


def _read(row, roles: dict) -> dict:
    """One snapshot: its weights, tier, caps, and the names above 15% and
    above their bucket cap ({name: weight fraction})."""
    _, pv, positions = row
    w, counts = weights(positions, pv)
    c = caps(pv)
    over15, overcap = {}, {}
    for t, x in w.items():
        if x > VIA_GAINS_PCT / 100:    # on fractions: 0.15 * 100 > 15.0 in floats
            over15[t] = x
        cap = c.get(roles.get(t)) if roles.get(t) in BUCKETS else None
        if cap is not None and x > cap:
            overcap[t] = x
    return {"weights": w, "counts": counts, "caps": c, "over15": over15, "overcap": overcap}


def tally(sessions: list[date], snaps, roles: dict) -> dict:
    """{"sessions": {session: per-session fields}, "names": {name: per-name
    fields}}. roles: {TICKER: watchlist role}. Pure."""
    by = in_session(snaps, sessions)
    per_s: dict = {}
    names: dict = {}

    def name(t):
        return names.setdefault(t, {
            "bucket": roles.get(t) if roles.get(t) in BUCKETS else NO_BUCKET,
            "sessions_held": 0, "max_close_pct": None,
            "s_over15_close": 0, "s_over15_any": 0,
            "s_overcap_close": 0, "s_overcap_any": 0})
    for d in sessions:
        rows = by.get(d) or []
        if not rows:
            per_s[d] = {"n_snapshots": 0}
            continue
        reads = [_read(r, roles) for r in rows]
        close = reads[-1]
        any15: dict = {}
        anycap: dict = {}
        for r in reads:
            for t, x in r["over15"].items():
                any15[t] = max(x, any15.get(t, 0.0))
            for t, x in r["overcap"].items():
                anycap[t] = max(x, anycap.get(t, 0.0))
        cw = close["weights"]
        top = max(sorted(cw), key=lambda t: cw[t]) if cw else None
        per_s[d] = {"n_snapshots": len(rows), "tier": close["caps"]["tier"],
                    "caps": {b: close["caps"][b] for b in BUCKETS},
                    "n_names": len(cw), "counts": close["counts"],
                    "largest": (top, cw[top]) if top else None,
                    "over15_close": close["over15"], "overcap_close": close["overcap"],
                    "over15_any": any15, "overcap_any": anycap}
        for t, x in cw.items():
            n = name(t)
            n["sessions_held"] += 1
            n["max_close_pct"] = max(x * 100, n["max_close_pct"] or 0.0)
        for key, field in (("over15_close", "s_over15_close"), ("over15_any", "s_over15_any"),
                           ("overcap_close", "s_overcap_close"), ("overcap_any", "s_overcap_any")):
            for t in per_s[d][key]:
                name(t)[field] += 1
    return {"sessions": per_s, "names": names}


# ---------------------------------------------------------------- output

def _pct(x, nd=1) -> str:
    return f"{x * 100:.{nd}f}%"


def _list(d: dict) -> str:
    return ", ".join(f"{t} {_pct(x)}" for t, x in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"


def readout(sessions: list[date], t: dict, head: str, now: datetime) -> str:
    ps, names = t["sessions"], t["names"]
    read = [d for d in sessions if ps[d]["n_snapshots"]]
    lines = [f"# Position weight readout ({ENV})", "",
             f"Sessions {sessions[0]} to {sessions[-1]} ({len(sessions)} NYSE sessions, "
             f"{len(read)} with an in-session snapshot); git {head}; generated "
             f"{now.strftime('%Y-%m-%d %H:%M UTC')}.",
             "",
             f"Record only. The config.py comment \"allow up to 15% via gains\" is enforced "
             f"nowhere; this shows how often a name sat above {VIA_GAINS_PCT:g}% of the book or "
             "above its bucket's at-entry cap (tier-resolved, as sizing reads it). Bucket = the "
             "name's role in today's watchlist.md. Which reading (close or any snapshot) or "
             "threshold counts for anything is a research choice (Luke).", "",
             "## Summary", "",
             "| reading | sessions with a name > 15% | sessions with a name > bucket cap |",
             "|---|---|---|"]
    for label, a, b in (("close (last in-session snapshot)", "over15_close", "overcap_close"),
                        ("any in-session snapshot", "over15_any", "overcap_any")):
        n15 = sum(1 for d in read if ps[d][a])
        ncap = sum(1 for d in read if ps[d][b])
        lines.append(f"| {label} | {n15} of {len(read)} | {ncap} of {len(read)} |")
    lines += ["", "## Per session", "",
              "| session | snapshots | tier | caps C/A/M | names | largest | > 15% at close | "
              "> bucket cap at close | > 15% at any snapshot | > bucket cap at any snapshot |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for d in sessions:
        s = ps[d]
        if not s["n_snapshots"]:
            lines.append(f"| {d} | 0 | no snapshot | | | | | | | |")
            continue
        cap_txt = "/".join(f"{s['caps'][b] * 100:g}" if s["caps"][b] is not None else "n/a"
                           for b in BUCKETS)
        big = f"{s['largest'][0]} {_pct(s['largest'][1])}" if s["largest"] else "none"
        extra = []
        if s["counts"]["n_short"]:
            extra.append(f"{s['counts']['n_short']} short")
        if s["counts"]["n_unpriced"]:
            extra.append(f"{s['counts']['n_unpriced']} unpriced")
        names_txt = f"{s['n_names']}" + (f" (+{', '.join(extra)})" if extra else "")
        lines.append(f"| {d} | {s['n_snapshots']} | {s['tier']} | {cap_txt}% | {names_txt} | {big} | "
                     f"{_list(s['over15_close'])} | {_list(s['overcap_close'])} | "
                     f"{_list(s['over15_any'])} | {_list(s['overcap_any'])} |")
    lines += ["", "## Per name", "",
              "| name | bucket | sessions held (close) | max close weight | sessions > 15% "
              "(close / any) | sessions > bucket cap (close / any) |",
              "|---|---|---|---|---|---|"]
    for n in sorted(names, key=lambda k: (-(names[k]["max_close_pct"] or 0), k)):
        r = names[n]
        mx = f"{r['max_close_pct']:.1f}%" if r["max_close_pct"] is not None else "n/a"
        cap_txt = ("n/a (no bucket)" if r["bucket"] == NO_BUCKET
                   else f"{r['s_overcap_close']} / {r['s_overcap_any']}")
        lines.append(f"| {n} | {r['bucket']} | {r['sessions_held']} | {mx} | "
                     f"{r['s_over15_close']} / {r['s_over15_any']} | {cap_txt} |")
    if not names:
        lines.append("| none | | | | | |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def _fetch(since: date | None, until: date):
    """(first live snapshot's ET date or None, rows (timestamp, total_value,
    positions_json) from since (or the first snapshot) through until, ET).
    One READ ONLY transaction, rolled back."""
    import db
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT min(timestamp) FROM portfolio_snapshots WHERE trading_env = %s", (ENV,))
            first = cur.fetchone()[0]
            if first is None:
                return None, []
            if first.tzinfo is None:
                first = first.replace(tzinfo=timezone.utc)
            first_d = first.astimezone(ny).date()
            start_d = since or first_d
            start = datetime.combine(start_d, time(0), tzinfo=ny).astimezone(timezone.utc)
            end = datetime.combine(until + timedelta(days=1), time(0), tzinfo=ny).astimezone(timezone.utc)
            cur.execute("SELECT timestamp, total_value, positions_json FROM portfolio_snapshots "
                        "WHERE trading_env = %s AND timestamp >= %s AND timestamp < %s "
                        "ORDER BY timestamp", (ENV, start, end))
            return first_d, cur.fetchall()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)


def _roles() -> dict:
    from knowledge_loader import load_watchlist
    return {str(t).upper(): (e or {}).get("role") for t, e in load_watchlist().items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", help="first session, YYYY-MM-DD (default: the first live snapshot's)")
    ap.add_argument("--until", help="last session, YYYY-MM-DD (default: the last closed one)")
    ap.add_argument("--out", help="also write the markdown here")
    a = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT") != str(READ_PORT):
        print(f"position_weight_readout: run with SUPABASE_PORT={READ_PORT} (the transaction pooler)",
              file=sys.stderr)
        return 2
    ef = _ef()
    since = date.fromisoformat(a.since) if a.since else None
    until = (date.fromisoformat(a.until) if a.until
             else ef.default_until(datetime.now(timezone.utc)))
    first, snaps = _fetch(since, until)
    if first is None:
        print("position_weight_readout: no live portfolio snapshot", file=sys.stderr)
        return 2
    start = since or first
    sessions = []
    d = start
    while d <= until:
        if ef._is_session(d):
            sessions.append(d)
        d += timedelta(days=1)
    if not sessions:
        print(f"position_weight_readout: no NYSE session in {start}..{until}", file=sys.stderr)
        return 2
    text = readout(sessions, tally(sessions, snaps, _roles()), ef._git_head(),
                   datetime.now(timezone.utc))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
