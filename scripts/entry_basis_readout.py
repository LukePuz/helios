"""Entry basis readout: for each analyst verdict on a held name, the two
entry figures its brief printed, as percentages only (audit 2026-11-26 22:53
#9; the KNOWN class from audit 2026-11-26 06:53 #3). Read-only engineering
readout; no price, share count or dollar figure is printed.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/entry_basis_readout.py [--since 2026-10-28] [--until 2026-11-27] \
        [--env live] [--gap-cut 0.5] [--out FILE] [--json FILE]

Verdicts: signals rows with decision.model_output (the analyst's verdicts),
timestamp in [--since, --until) as New York dates, in --env or with a NULL
env. Each verdict is counted once; its day is the New York date of
signals.timestamp.

The two brief sections that print an entry for the position (no other
section prints one):
  - CURRENT PORTFOLIO (context_engine.assemble_brief / build_scan_brief):
    "Entry: $X | Current: $Y | P&L: Z%" per held symbol. Entry is the
    broker's average cost (crew._get_positions: avg_entry_price) and P&L is
    Current / Entry - 1 on that basis.
  - POSITION CONTEXT — TICKER (held) (context_engine._position_context_
    section): "Entry $E; protective stop ..." from the stop record
    (db.get_position_stop: entry_price).
The SQL returns only those two slices of the stored prompt (CURRENT
PORTFOLIO from its header, PORTFOLIO_SLICE characters; POSITION CONTEXT from
its header, POSITION_SLICE characters), never the rest. The readout parses
the verdict ticker's CURRENT PORTFOLIO line and the POSITION CONTEXT Entry
line; the prices stay inside classify() and only percentages leave it.

Per verdict (both figures seen):
  - gap %: broker entry / stop-record entry - 1, in percent (the sign and
    definition of portfolio_snapshots.positions_json basis_gap_frac, x100);
  - broker P&L %: the printed CURRENT PORTFOLIO P&L;
  - record P&L %: CURRENT PORTFOLIO's Current / the stop-record entry - 1;
  - P&L gap, points: record P&L minus broker P&L.
A verdict that did not see both is counted by why: "prompt NULL", "no
portfolio line" (no CURRENT PORTFOLIO line for the ticker), "no position
context", "context for another ticker" or "unparsed".

Per ticker: verdicts, verdicts with the broker line, with the record entry
and with both; the gap and P&L-gap distributions (n, min, p25, median, p75,
max; statistics.quantiles, inclusive); verdicts whose |gap| is above
--gap-cut; the first verdict with both figures and its gap; the first verdict
whose |gap| is above --gap-cut ("since"); the last verdict's figures; and the
gap steps: each verdict whose gap differs from the ticker's previous verdict
with both figures (the two entries are fixed figures, so the gap changes only
when one of them does). The
first-gap verdict is bounded by the window and by the first verdict with
both figures (POSITION CONTEXT appears from 2026-10-30): when they are the
same verdict, the gap may be older. --gap-cut (default 0.5, from the
request) only marks "since"; nothing is excluded by it. A ticker x day
table gives the same counts per day.

No basis is chosen and no label is applied: which basis the brief shows, or
whether to label both, is Luke's (analyst input; audit 06:53 #3).

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db and the standard library.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_SINCE = "2026-10-28"
DEFAULT_GAP_CUT = 0.5
PORTFOLIO_HEADER = "=== CURRENT PORTFOLIO ==="
POSITION_HEADER = "=== POSITION CONTEXT — "
PORTFOLIO_SLICE = 4000
POSITION_SLICE = 400
REASONS = ("prompt NULL", "no portfolio line", "no position context",
           "context for another ticker", "unparsed")
_NUM = r"(-?\d+(?:\.\d+)?)"
_POS_HEAD = re.compile(r"^=== POSITION CONTEXT — (\S+) \(held\) ===$", re.M)
_POS_ENTRY = re.compile(r"^Entry \$" + _NUM + r"; protective stop", re.M)

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol,
       s.provenance->'trigger'->>'path',
       s.provenance->'plan'->>'held',
       s.prompt IS NULL,
       CASE WHEN strpos(s.prompt, %(pf_hdr)s) > 0
            THEN substr(s.prompt, strpos(s.prompt, %(pf_hdr)s), %(pf_slice)s) END,
       CASE WHEN strpos(s.prompt, %(pc_hdr)s) > 0
            THEN substr(s.prompt, strpos(s.prompt, %(pc_hdr)s), %(pc_slice)s) END
FROM signals s
WHERE s.decision ? 'model_output' AND s.timestamp >= %(start)s AND s.timestamp < %(end)s
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


def portfolio_line(block: str | None, ticker: str):
    """(entry, current, printed P&L %) from the ticker's CURRENT PORTFOLIO
    line, None when the block has no line for it, "unparsed" when the line
    is there but doesn't parse. The block ends at its first blank line or
    the next "===" header. Pure."""
    if not block or not block.startswith(PORTFOLIO_HEADER) or not ticker:
        return None
    lines = block[len(PORTFOLIO_HEADER):].lstrip("\n").split("\n")
    for line in lines:
        if not line.strip() or line.startswith("==="):
            break
        parts = line.strip().split(None, 1)
        if not parts or parts[0] != ticker:
            continue
        m = re.search(r"\| Entry: \$" + _NUM + r" \| Current: \$" + _NUM
                      + r" \| P&L: ([+-]?\d+(?:\.\d+)?)%", line)
        if not m:
            return "unparsed"
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def position_entry(block: str | None):
    """(header ticker, stop-record entry) from a POSITION CONTEXT slice;
    (ticker, "unparsed") when its Entry line doesn't parse; None without a
    header. Pure."""
    if not block:
        return None
    h = _POS_HEAD.search(block)
    if not h:
        return None
    rest = block[h.end():].lstrip("\n").split("\n", 1)[0]
    m = _POS_ENTRY.match(rest)
    return h.group(1), (float(m.group(1)) if m else "unparsed")


def held_label(v) -> str:
    return {"true": "held", "false": "not held"}.get(str(v).lower() if v is not None else "", "unknown")


def _r(v: float) -> float:
    return round(v, 2)


def classify(row) -> dict:
    """One verdict row (as _VERDICTS_SQL returns it) as a dict of
    percentages and labels; no price leaves this function."""
    sid, ts, ticker, path, held, prompt_null, pf_block, pc_block = row
    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    ticker = ticker or ""
    out = {"id": sid, "at": ts.astimezone(NY).strftime("%Y-%m-%d %H:%M"),
           "day": ts.astimezone(NY).date().isoformat(), "ticker": ticker,
           "path": path or "none", "held": held_label(held),
           "broker_seen": False, "record_seen": False, "reason": None,
           "gap_pct": None, "broker_pl_pct": None, "record_pl_pct": None, "pl_gap_pts": None}
    if prompt_null:
        out["reason"] = "prompt NULL"
        return out
    pf = portfolio_line(pf_block, ticker)
    pc = position_entry(pc_block)
    out["broker_seen"] = isinstance(pf, tuple)
    out["record_seen"] = bool(pc) and pc[0] == ticker and pc[1] != "unparsed"
    if pf == "unparsed" or (pc and pc[0] == ticker and pc[1] == "unparsed"):
        out["reason"] = "unparsed"
    elif pc and pc[0] != ticker:
        out["reason"] = "context for another ticker"
    elif pf is None:
        out["reason"] = "no portfolio line"
    elif pc is None:
        out["reason"] = "no position context"
    if out["reason"]:
        return out
    b_entry, current, b_pl = pf
    r_entry = pc[1]
    if b_entry <= 0 or r_entry <= 0:
        out["reason"] = "unparsed"
        out["broker_seen"] = out["record_seen"] = False
        return out
    r_pl = (current / r_entry - 1) * 100
    out.update(gap_pct=_r((b_entry / r_entry - 1) * 100), broker_pl_pct=_r(b_pl),
               record_pl_pct=_r(r_pl), pl_gap_pts=_r(r_pl - b_pl))
    return out


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) >= 2 else [v[0], v[0], v[0]]
    return {"n": len(v), "min": v[0], "p25": round(q[0], 2), "median": round(statistics.median(v), 2),
            "p75": round(q[2], 2), "max": v[-1]}


def _mark(r: dict | None) -> dict | None:
    return None if r is None else {"id": r["id"], "at": r["at"], "gap_pct": r["gap_pct"]}


def tally(rows: list[dict], cut: float) -> dict:
    """The column set for one cell (rows in time order)."""
    both = [r for r in rows if r["gap_pct"] is not None]
    over = [r for r in both if abs(r["gap_pct"]) > cut]
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    last = both[-1] if both else None
    steps = [_mark(r) for i, r in enumerate(both) if i == 0 or r["gap_pct"] != both[i - 1]["gap_pct"]]
    return {"verdicts": len(rows), "broker_seen": sum(r["broker_seen"] for r in rows),
            "record_seen": sum(r["record_seen"] for r in rows), "both": len(both),
            "over_cut": len(over), "reasons": {k: reasons.get(k, 0) for k in REASONS},
            "gap": _dist([r["gap_pct"] for r in both]),
            "pl_gap": _dist([r["pl_gap_pts"] for r in both]),
            "first_both": _mark(both[0] if both else None),
            "first_over_cut": _mark(over[0] if over else None),
            "since_is_first_both": bool(over and both and over[0]["id"] == both[0]["id"]),
            "steps": steps,
            "last": None if last is None else {
                "id": last["id"], "at": last["at"], "gap_pct": last["gap_pct"],
                "broker_pl_pct": last["broker_pl_pct"], "record_pl_pct": last["record_pl_pct"],
                "pl_gap_pts": last["pl_gap_pts"]}}


def build(rows: list[dict], cut: float) -> dict:
    def group(key, keep=lambda r: True):
        out = {}
        for r in rows:
            if keep(r):
                out.setdefault(key(r), []).append(r)
        return {k: tally(v, cut) for k, v in sorted(out.items())}
    return {"total": tally(rows, cut),
            "by_ticker": group(lambda r: r["ticker"],
                               lambda r: r["broker_seen"] or r["record_seen"]),
            "by_ticker_day": group(lambda r: f"{r['ticker']} / {r['day']}",
                                   lambda r: r["gap_pct"] is not None)}


def _c(v) -> str:
    return "-" if v is None else str(v)


def _d(d: dict) -> str:
    return f"{d['median']} [{d['p25']}, {d['p75']}], {d['min']}..{d['max']}" if d["n"] else "-"


def _m(m: dict | None) -> str:
    return "-" if m is None else f"{m['at']} (#{m['id']}, {m['gap_pct']:+.2f}%)"


def _ticker_table(cells: dict, cut: float) -> list[str]:
    lines = ["", "## By ticker (verdicts whose brief printed at least one entry for the ticker)", "",
             "| ticker | verdicts | broker line | record entry | both | "
             f"abs(gap) > {cut}% | gap %: median [p25, p75], min..max | "
             "P&L gap, points: median [p25, p75], min..max | first with both | "
             f"first abs(gap) > {cut}% (since) | since = first with both | last: gap % | "
             "last: broker P&L % | last: record P&L % |",
             "|---|" + "---|" * 13]
    for k, t in cells.items():
        last = t["last"] or {}
        lines.append(f"| {k} | {t['verdicts']} | {t['broker_seen']} | {t['record_seen']} | {t['both']} | "
                     f"{t['over_cut']} | {_d(t['gap'])} | {_d(t['pl_gap'])} | {_m(t['first_both'])} | "
                     f"{_m(t['first_over_cut'])} | {'yes' if t['since_is_first_both'] else 'no'} | "
                     f"{_c(last.get('gap_pct'))} | {_c(last.get('broker_pl_pct'))} | "
                     f"{_c(last.get('record_pl_pct'))} |")
    return lines


def _steps(cells: dict) -> list[str]:
    lines = ["", "## Gap steps (each verdict where a ticker's gap % differs from its previous "
             "verdict with both figures; the first is the first verdict with both)", ""]
    for k, t in cells.items():
        if t["steps"]:
            lines.append(f"- {k} ({len(t['steps'])}): " + "; ".join(_m(m) for m in t["steps"]))
    return lines


def _day_table(cells: dict, cut: float) -> list[str]:
    lines = ["", "## Ticker x day (verdicts that saw both figures)", "",
             f"| ticker / day | both | abs(gap) > {cut}% | gap %: median [p25, p75], min..max | "
             "P&L gap, points: median [p25, p75], min..max |",
             "|---|" + "---|" * 4]
    for k, t in cells.items():
        lines.append(f"| {k} | {t['both']} | {t['over_cut']} | {_d(t['gap'])} | {_d(t['pl_gap'])} |")
    return lines


def render(out: dict, env: str, since: date, until: date, cut: float, head: str,
           now: datetime) -> str:
    t = out["total"]
    lines = [f"# Entry basis readout ({env}, verdicts {since.isoformat()} to before "
             f"{until.isoformat()} ET)", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             "Engineering readout; percentages only. CURRENT PORTFOLIO prints the broker's average "
             "cost (crew._get_positions avg_entry_price) and P&L on it; POSITION CONTEXT prints the "
             "stop record's entry (db.get_position_stop). Gap % = broker entry / stop-record entry "
             "- 1. Record P&L % = CURRENT PORTFOLIO's current price / stop-record entry - 1. P&L gap "
             f"= record P&L minus broker P&L, in points. The {cut}% cut only marks 'since'. Which "
             "basis to show, or a label on both, is Luke's.", "",
             f"- verdicts: {t['verdicts']}; with the broker line for their ticker: {t['broker_seen']}; "
             f"with the stop-record entry: {t['record_seen']}; with both figures: {t['both']}; "
             f"abs(gap) > {cut}%: {t['over_cut']}",
             "- without both: " + ", ".join(f"{r} {t['reasons'][r]}" for r in REASONS),
             f"- first verdict with both: {_m(t['first_both'])}; first with abs(gap) > {cut}%: "
             f"{_m(t['first_over_cut'])}"]
    lines += _ticker_table(out["by_ticker"], cut)
    lines += _steps(out["by_ticker"])
    lines += _day_table(out["by_ticker_day"], cut)
    return "\n".join(lines) + "\n"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--gap-cut", type=float, default=DEFAULT_GAP_CUT,
                    help="abs(gap %%) above which a verdict marks 'since' (reporting only)")
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
    since = date.fromisoformat(args.since)
    until = (date.fromisoformat(args.until) if args.until
             else now.astimezone(NY).date() + timedelta(days=1))
    if until <= since:
        print("--until must be after --since.", file=sys.stderr)
        return 2
    if not args.gap_cut >= 0:
        print("--gap-cut must be 0 or more.", file=sys.stderr)
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
            cur.execute(_VERDICTS_SQL, {"env": args.env, "start": start, "end": end,
                                        "pf_hdr": PORTFOLIO_HEADER, "pf_slice": PORTFOLIO_SLICE,
                                        "pc_hdr": POSITION_HEADER, "pc_slice": POSITION_SLICE})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = [classify(r) for r in raw]
    out = build(rows, args.gap_cut)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(out, args.env, since, until, args.gap_cut, head or "unknown", now)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head or "unknown",
                       "env": args.env, "since": since.isoformat(), "until": until.isoformat(),
                       "gap_cut": args.gap_cut, "verdicts": rows}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
