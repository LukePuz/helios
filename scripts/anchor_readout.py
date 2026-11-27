"""Conviction and UDR by anchor readout (audit 2026-11-26-1253 #1;
eng-candidates-2026-11-26 sweep #1). Engineering readout, record only:
counts and percents. Nothing in the bot reads it.

For full analyst verdicts (decision ? 'model_output', not a
db._NON_ANALYST_SIGNALS row) of one trading env in ET days --since ..
--until (inclusive), grouped by the stored anchor records' raw values:

  target_anchor   decision.target_anchor's target_equals_street and
                  target_equals_fv as stored (true / false / null), shown as
                  "street=<v> fv=<v>"; "not recorded" when the verdict has no
                  target_anchor (it is live from the 2026-11-24 load).
  anchor_pair     decision.downside_anchor.anchor_pair as stored (e.g.
                  "street+support", "none+stop"; "null" when stored null);
                  "not recorded" when the verdict has no downside_anchor (it
                  loads with batch 1135).

No group is merged or labelled: an "other" or "echo" bucket, or any band, is
Luke's. Each group shows n and its share of the verdicts, the final
conviction (decision.final_conviction) counts per stored value, the model
UDR (decision.udr_check.model) as n / min / p25 / median / p75 / max
(linear interpolation between order statistics; no bins), the action
(signals.signal) counts, and the count per ticker. Percent and counts only:
no price, target or level is printed.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        <path>/scripts/anchor_readout.py [--env live] [--since D] [--until D] \\
        [--ticker T] [--json] [--out FILE]

db is imported from the working directory's clone (run it with env -C on the
clone whose DB settings it should use). Read-only: SUPABASE_PORT must be 6543
(the transaction pooler); one transaction, SET TRANSACTION READ ONLY with a
statement timeout, rolled back. It refuses to read between 04:50 and 05:12
ET (the morning restart window).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 30000
DEFAULT_SINCE = "2026-11-24"           # target_anchor's first load day
QUIET = (time(4, 50), time(5, 12))     # ET: no DB reads in this window
NOT_RECORDED = "not recorded"
GROUPINGS = ("target_anchor", "anchor_pair")

VERDICTS_SQL = (
    "SELECT id, symbol, signal, timestamp, "
    "decision ? 'target_anchor', "
    "decision->'target_anchor'->'target_equals_street', "
    "decision->'target_anchor'->'target_equals_fv', "
    "decision ? 'downside_anchor', "
    "decision->'downside_anchor'->'anchor_pair', "
    "decision->'final_conviction', "
    "decision->'udr_check'->'model' "
    "FROM signals "
    "WHERE trading_env = %(env)s AND decision ? 'model_output' "
    "AND NOT (signal = ANY(%(non)s)) "
    "AND timestamp >= %(lo)s AND timestamp < %(hi)s")


def _bounds(since: str, until: str) -> tuple[datetime, datetime]:
    """ET day bounds, since inclusive to the day after until."""
    lo = datetime.combine(date.fromisoformat(since), time(0), NY)
    hi = datetime.combine(date.fromisoformat(until) + timedelta(days=1), time(0), NY)
    return lo, hi


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET[0] <= t < QUIET[1]


def _raw(v) -> str:
    """A stored JSON scalar as text: true / false / null, numbers and strings
    as they are."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def _num(v):
    """A finite number, else None (a bool is not a number here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def verdict(row) -> dict:
    """One fetched row to {id, ticker, signal, target_anchor, anchor_pair,
    conviction, udr}: the two grouping keys as raw values."""
    (i, sym, sig, ts, has_ta, street, fv, has_da, pair, conv, udr) = row
    return {"id": i, "ticker": str(sym or "?").upper(), "signal": sig,
            "target_anchor": (f"street={_raw(street)} fv={_raw(fv)}" if has_ta else NOT_RECORDED),
            "anchor_pair": (_raw(pair) if has_da else NOT_RECORDED),
            "conviction": conv, "udr": _num(udr)}


def _pctile(vals: list[float], q: float):
    n = len(vals)
    if not n:
        return None
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _conv_key(v) -> str:
    """A stored conviction as a count key: integral numbers without '.0',
    anything else as stored ("null" when missing)."""
    n = _num(v)
    if n is None:
        return _raw(v)
    return str(int(n)) if n == int(n) else str(n)


def _ordered(c: Counter) -> dict:
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def summarise(rows: list[dict], key: str) -> dict:
    """{group value: stats} for one grouping, groups by size (then value)."""
    total = len(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        udr = sorted(r["udr"] for r in rs if r["udr"] is not None)
        conv = Counter(_conv_key(r["conviction"]) for r in rs)

        def conv_order(k):
            n = _num(_try_float(k))
            return (0, -n) if n is not None else (1, k)
        out[g] = {
            "n": len(rs),
            "pct": round(100.0 * len(rs) / total, 1) if total else None,
            "conviction": {k: {"n": conv[k], "pct": round(100.0 * conv[k] / len(rs), 1)}
                           for k in sorted(conv, key=conv_order)},
            "udr": {"n": len(udr), "n_missing": len(rs) - len(udr),
                    "min": _r(udr[0] if udr else None), "p25": _r(_pctile(udr, 0.25)),
                    "median": _r(_pctile(udr, 0.5)), "p75": _r(_pctile(udr, 0.75)),
                    "max": _r(udr[-1] if udr else None)},
            "action": _ordered(Counter(str(r["signal"]) for r in rs)),
            "tickers": _ordered(Counter(r["ticker"] for r in rs)),
        }
    return out


def _try_float(k):
    try:
        return float(k)
    except (TypeError, ValueError):
        return None


def _r(x, nd=2):
    return round(x, nd) if x is not None else None


def readout(rows: list[dict]) -> dict:
    return {"verdicts": len(rows),
            "tickers": _ordered(Counter(r["ticker"] for r in rows)),
            "by": {k: summarise(rows, k) for k in GROUPINGS}}


def _cell(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "none"


def render(res: dict) -> str:
    L = [f"# Conviction and UDR by anchor ({res['env']}, {res['since']} to {res['until']} ET)", "",
         f"Verdicts: {res['verdicts']} (full analyst verdicts). Tickers: {_cell(res['tickers'])}.",
         "", "Groups are the stored values as they are; none is merged or labelled. "
         "Conviction = decision.final_conviction; UDR = decision.udr_check.model.", ""]
    titles = {"target_anchor": "decision.target_anchor (target_equals_street, target_equals_fv)",
              "anchor_pair": "decision.downside_anchor.anchor_pair"}
    for k in GROUPINGS:
        L += [f"## By {titles[k]}", "",
              "| value | n | % of verdicts | conviction n (% of group) | UDR n / min / p25 / median / "
              "p75 / max (missing) | action | tickers |", "|---|---|---|---|---|---|---|"]
        for g, s in res["by"][k].items():
            conv = ", ".join(f"c{c} {v['n']} ({v['pct']}%)" for c, v in s["conviction"].items())
            u = s["udr"]
            ustr = (f"{u['n']} / {u['min']} / {u['p25']} / {u['median']} / {u['p75']} / {u['max']} "
                    f"({u['n_missing']})")
            L.append(f"| {g} | {s['n']} | {s['pct']} | {conv} | {ustr} | {_cell(s['action'])} | "
                     f"{_cell(s['tickers'])} |")
        L.append("")
    return "\n".join(L)


def fetch(conn, env: str, since: str, until: str, non_analyst, ticker: str | None = None) -> list[dict]:
    """The window's verdicts in one read-only transaction, rolled back."""
    lo, hi = _bounds(since, until)
    sql, args = VERDICTS_SQL, {"lo": lo, "hi": hi, "env": env, "non": list(non_analyst)}
    if ticker:
        sql += " AND symbol = %(ticker)s"
        args["ticker"] = ticker
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(sql, args)
            return [verdict(r) for r in cur.fetchall()]
    finally:
        conn.rollback()


def run(conn, env: str, since: str, until: str, non_analyst, ticker: str | None = None) -> dict:
    rows = fetch(conn, env, since, until, non_analyst, ticker)
    return {"env": env, "since": since, "until": until, "ticker": ticker, **readout(rows)}


def _import_db():
    """db from the working directory's clone, checked."""
    cwd = os.path.abspath(os.getcwd())
    sys.path[:] = [cwd] + [p for p in sys.path if os.path.abspath(p or ".") != cwd]
    import db
    if os.path.dirname(os.path.abspath(db.__file__)) != cwd:
        raise RuntimeError(f"db imported from {db.__file__}, not the working directory {cwd}")
    return db


def main(argv=None, now: datetime | None = None) -> int:
    today = datetime.now(NY).date().isoformat()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", help="trading env (default config.TRADING_ENV)")
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, ET, inclusive")
    ap.add_argument("--until", default=today, help="YYYY-MM-DD, ET, inclusive")
    ap.add_argument("--ticker", help="one ticker only")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--out", help="write the output here instead of stdout")
    a = ap.parse_args(argv)
    try:
        date.fromisoformat(a.since), date.fromisoformat(a.until)
    except ValueError:
        print("bad --since or --until (YYYY-MM-DD)", file=sys.stderr)
        return 2
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    if in_quiet_window(now or datetime.now(timezone.utc)):
        print("04:50-05:12 ET is the restart window; not reading.", file=sys.stderr)
        return 2
    db = _import_db()
    env = a.env
    if not env:
        import config
        env = config.TRADING_ENV
    ticker = a.ticker.upper() if a.ticker else None
    conn = db._get_conn()
    try:
        res = run(conn, env, a.since, a.until, getattr(db, "_NON_ANALYST_SIGNALS", ()), ticker)
    finally:
        db._put_conn(conn)
    text = json.dumps(res, indent=2, default=str) + "\n" if a.json else render(res)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
