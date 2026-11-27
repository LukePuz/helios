"""Post-exit readout: what each closed live episode's name did after the
exit, and SPY over the same sessions, as percentages only (audit
2026-11-27 02:53 #4). Read-only engineering readout; no price, share count
or dollar figure is printed.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/post_exit_readout.py [--env live] [--since YYYY-MM-DD] \
        [--out FILE] [--json FILE]

Episodes: trade_outcomes rows in --env with an exit, grouped by ticker and
the New York date of the exit (the exit fill time where known, else
exit_date: db.outcome_closed_since's COALESCE(exit_fill_at, exit_date)).
Lots that closed on different days are separate episodes. --since (New York
date, on the exit day) narrows the list; by default every closed episode is
read. Rows whose exit_date falls on another New York day than the fill time
are counted.

Base: the ticker's price_history close on the exit day. With no stored bar
on that day the episode has no base ("no exit-day bar") and no figures.
Sessions are the ticker's own stored bars after the base bar:
  - +5, +10, +20: close at the 5th, 10th, 20th bar after the base / base - 1;
  - latest: the last stored close / base - 1, with its date and the number
    of bars after the base;
  - 20-session low and high: the lowest and highest close over the 20 bars
    after the base, over base - 1 (closes only, not intraday lows/highs).
SPY: the same figures from SPY's stored bars, based on SPY's close on the
same date as the ticker's base bar. Where the ticker's and SPY's k-th bars
fall on different dates (a missing bar in one series) the episode is listed
under "session date mismatches"; nothing is realigned.

"n/a" where the window isn't complete yet: fewer than k bars after the base
for +k; fewer than 20 for the low and high, which then print the partial
figure and its bar count beside the n/a. A figure that would read a missing
(NULL), non-finite or non-positive close prints that word instead of a
number; nothing is dropped.

Data checks, flag only (scripts/_bar_qa.py and scripts/rescale_stale_head.py
candidates(), loaded with importlib):
  - non-finite bars: OHLC NaN/inf in the bars read, from each ticker's first
    exit day;
  - split-like steps: one-day close ratios near a split factor (or at or
    beyond 3x) inside the windows read. The price cache keeps one
    split-adjusted year and never rewrites older rows, so a window that
    crosses a stored step mixes two scales (audit-method-notes: price cache
    basis; SNDK, MNST, APH).

No label, no summary statistic and no verdict on the episodes: whether the
readout sits beside the stop-out re-entry gate evidence, and what a figure
means for the stops, is Luke's.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db and the standard library.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
QUIET_START, QUIET_END = time(4, 50), time(5, 12)   # no DB reads at the stack's start
BENCH = "SPY"
HORIZONS = (5, 10, 20)
RANGE_BARS = 20

_EPISODES_SQL = """
SELECT ticker,
       (COALESCE(exit_fill_at, exit_date) AT TIME ZONE 'America/New_York')::date AS exit_day,
       min((entry_date AT TIME ZONE 'America/New_York')::date),
       count(*),
       string_agg(DISTINCT COALESCE(exit_source, 'unknown'), ','
                  ORDER BY COALESCE(exit_source, 'unknown')),
       count(*) FILTER (WHERE (exit_date AT TIME ZONE 'America/New_York')::date
                          <> (COALESCE(exit_fill_at, exit_date) AT TIME ZONE 'America/New_York')::date)
FROM trade_outcomes
WHERE trading_env = %(env)s AND exit_date IS NOT NULL AND ticker IS NOT NULL
GROUP BY 1, 2
ORDER BY 2, 1
"""

_BARS_SQL = """
SELECT ticker, date, open, high, low, close
FROM price_history
WHERE ticker = ANY(%(tickers)s) AND date >= %(start)s
ORDER BY ticker, date
"""


def db_root(cwd: str | None = None) -> str:
    """The directory to import db from: the working directory when it holds
    db.py (the live clone, whose .env db loads), else this script's repo."""
    cwd = cwd or os.getcwd()
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET_START <= t < QUIET_END


def _load(name: str):
    """A sibling script (scripts/ is not a package) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _day(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def _close(bar):
    """The bar's close as a float, or "missing" (NULL), "non-finite" (NaN or
    infinite) or "non-positive" (zero or below, no return on it)."""
    c = bar["close"]
    if c is None:
        return "missing"
    try:
        f = float(c)
    except (TypeError, ValueError, OverflowError):
        return "non-finite"
    if not math.isfinite(f):
        return "non-finite"
    return f if f > 0 else "non-positive"


def _ret(c, base):
    """c / base - 1 in percent (2 dp), or the bad value's label."""
    for v in (base, c):
        if isinstance(v, str):
            return v
    return round((c / base - 1) * 100, 2)


def path_figures(bars: list[dict], base_day: date) -> dict:
    """The post-exit figures of one series from its bar on base_day. Pure;
    bars sorted by date, each a dict with "date" and "close". Returns
    {"base": False} when there is no bar on base_day."""
    idx = next((i for i, b in enumerate(bars) if _day(b["date"]) == base_day), None)
    if idx is None:
        return {"base": False}
    base = _close(bars[idx])
    after = bars[idx + 1:]
    out = {"base": True, "bars_after": len(after),
           "latest_date": _day(bars[-1]["date"]).isoformat(),
           "latest": _ret(_close(bars[-1]), base)}
    for k in HORIZONS:
        if len(after) >= k:
            out[f"r{k}"] = _ret(_close(after[k - 1]), base)
            out[f"d{k}"] = _day(after[k - 1]["date"]).isoformat()
        else:
            out[f"r{k}"], out[f"d{k}"] = None, None
    window = after[:RANGE_BARS]
    out["range_bars"] = len(window)
    out["range_complete"] = len(window) == RANGE_BARS
    closes = [_close(b) for b in window]
    bad = next((c for c in [base, *closes] if isinstance(c, str)), None)
    if not window:
        out["low"] = out["high"] = None
    elif bad:
        out["low"] = out["high"] = bad
    else:
        out["low"] = round((min(closes) / base - 1) * 100, 2)
        out["high"] = round((max(closes) / base - 1) * 100, 2)
    return out


def episode_record(ep: tuple, bars: dict) -> dict:
    """One episode's record: its keys and the name's and SPY's figures.
    ep is an _EPISODES_SQL row; bars maps ticker -> sorted bar dicts."""
    ticker, exit_day, entry_day, lots, sources, day_differs = ep
    exit_day = _day(exit_day)
    rec = {"ticker": ticker, "exit_day": exit_day.isoformat(),
           "entry_day": _day(entry_day).isoformat() if entry_day is not None else None,
           "lots": int(lots), "exit_sources": sources, "exit_date_day_differs": int(day_differs or 0)}
    name = path_figures(bars.get(ticker, []), exit_day)
    rec["name"] = name
    rec["spy"] = path_figures(bars.get(BENCH, []), exit_day) if name["base"] else {"base": False}
    mism = []
    if name["base"] and rec["spy"]["base"]:
        for k in HORIZONS:
            if name[f"d{k}"] and rec["spy"][f"d{k}"] and name[f"d{k}"] != rec["spy"][f"d{k}"]:
                mism.append(f"+{k}")
        if name["latest_date"] != rec["spy"]["latest_date"]:
            mism.append("latest")
    rec["session_mismatch"] = mism
    return rec


def window_bars(records: list[dict], bars: dict) -> dict:
    """Per ticker (SPY included), the bars from its first base day on: the
    bars the figures read. Pure."""
    first: dict = {}
    for r in records:
        if r["name"]["base"]:
            d = date.fromisoformat(r["exit_day"])
            for t in (r["ticker"], BENCH):
                first[t] = min(first.get(t, d), d)
    return {t: [b for b in bars.get(t, []) if _day(b["date"]) >= d] for t, d in first.items()}


def split_steps(wbars: dict, candidates) -> list[tuple[str, str, float, str]]:
    """(ticker, step date, ratio, nearest) of every split-like one-day close
    step inside the windows read, using rescale_stale_head.candidates."""
    out = []
    for t in sorted(wbars):
        rows = [{"date": b["date"], "close": b["close"], "volume": None} for b in wbars[t]
                if isinstance(_close(b), float)]
        for c in candidates(rows):
            out.append((t, c["step_date"], c["ratio"], c["nearest"]))
    return out


def group_bars(raw: list[tuple]) -> dict:
    out: dict = {}
    for ticker, d, o, h, lo, c in raw:
        out.setdefault(ticker, []).append({"date": d, "open": o, "high": h, "low": lo, "close": c})
    for t in out:
        out[t].sort(key=lambda b: _day(b["date"]))
    return out


def _p(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:+.1f}%"


def _range(f: dict, key: str) -> str:
    if not f.get("base"):
        return "n/a"
    if f["range_complete"]:
        return _p(f[key])
    if f[key] is None:
        return "n/a"
    return f"n/a (so far {_p(f[key])}, {f['range_bars']} bars)"


def _cells(f: dict) -> list[str]:
    if not f.get("base"):
        return ["no exit-day bar"] + ["n/a"] * 5
    return ([_p(f[f"r{k}"]) for k in HORIZONS]
            + [f"{_p(f['latest'])} ({f['latest_date']}, {f['bars_after']} bars)",
               _range(f, "low"), _range(f, "high")])


def render(records: list[dict], flags: dict, env: str, since: date | None, head: str,
           now: datetime) -> str:
    n_base = sum(1 for r in records if r["name"]["base"])
    complete = {k: sum(1 for r in records if r["name"].get(f"r{k}") is not None) for k in HORIZONS}
    rng = sum(1 for r in records if r["name"].get("range_complete"))
    differs = sum(r["exit_date_day_differs"] for r in records)
    scope = f"exit day {since.isoformat()} on" if since else "all exit days"
    lines = [f"# Post-exit readout ({env}, closed episodes, {scope})", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             "Engineering readout; percentages only. Episode = trade_outcomes rows of one ticker "
             "exiting on one New York day (the exit fill time where known, else exit_date). Base = "
             "the ticker's price_history close on the exit day. +k = the close k stored bars after "
             "the base / base - 1; latest = the last stored close; low and high = the lowest and "
             f"highest close over the {RANGE_BARS} bars after the base (closes, not intraday). SPY: "
             "the same figures from SPY's own bars, based on its close on the same date. n/a = the "
             "window isn't complete yet. No label and no verdict: what these mean for the stops or "
             "the re-entry gate is Luke's.", "",
             f"- episodes: {len(records)}; with an exit-day bar: {n_base}; complete +5: "
             f"{complete[5]}, +10: {complete[10]}, +20: {complete[20]}; complete "
             f"{RANGE_BARS}-bar low/high: {rng}",
             f"- outcome rows whose exit_date is on another New York day than the fill time: {differs}",
             f"- {flags['nonfinite']}",
             "- split-like steps in the windows read: " + (
                 str(len(flags["splits"])) + " (" + ", ".join(
                     f"{t} {d} x{r} ({lab})" for t, d, r, lab in flags["splits"]) + ")"
                 if flags["splits"] else "0"),
             "- split caveat (audit-method-notes, price cache basis): the nightly refresh writes one "
             "split-adjusted year and never rewrites older rows, so a window that crosses a stored "
             "step (SNDK before 2026-11-20 about 8x, MNST and APH before about 2026-09-11 about 2x) "
             "mixes scales; check scripts/rescale_stale_head.py's dry run before trusting a window "
             "that starts more than a year back.",
             "- session date mismatches (the name's and SPY's k-th bar on different dates): " + (
                 ", ".join(f"{r['ticker']} {r['exit_day']} ({'/'.join(r['session_mismatch'])})"
                           for r in records if r["session_mismatch"]) or "none"),
             "", "## The name after the exit", "",
             "| Ticker | Exit day | Entry day | Lots | Exit source | +5 | +10 | +20 | Latest "
             "(date, bars) | 20-bar low | 20-bar high |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['ticker']} | {r['exit_day']} | {r['entry_day'] or 'n/a'} | {r['lots']} "
                     f"| {r['exit_sources']} | " + " | ".join(_cells(r["name"])) + " |")
    if not records:
        lines.append("| none | | | | | | | | | | |")
    lines += ["", f"## {BENCH} over the same sessions", "",
              f"| Ticker | Exit day | {BENCH} +5 | {BENCH} +10 | {BENCH} +20 | {BENCH} latest "
              f"(date, bars) | {BENCH} 20-bar low | {BENCH} 20-bar high |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        cells = _cells(r["spy"]) if r["name"]["base"] else ["n/a (no exit-day bar for the name)"] + ["n/a"] * 5
        if r["name"]["base"] and not r["spy"]["base"]:
            cells = [f"no {BENCH} bar on the exit day"] + ["n/a"] * 5
        lines.append(f"| {r['ticker']} | {r['exit_day']} | " + " | ".join(cells[:6]) + " |")
    if not records:
        lines.append("| none | | | | | | | |")
    return "\n".join(lines) + "\n"


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="live")
    ap.add_argument("--since", help="YYYY-MM-DD, New York exit day; default every closed episode")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-episode records (percentages only) here")
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
        since = date.fromisoformat(args.since) if args.since else None
    except ValueError:
        print("--since must be YYYY-MM-DD.", file=sys.stderr)
        return 2

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
            cur.execute(_EPISODES_SQL, {"env": args.env})
            eps = [e for e in cur.fetchall() if since is None or _day(e[1]) >= since]
            raw = []
            if eps:
                cur.execute(_BARS_SQL, {"tickers": sorted({e[0] for e in eps} | {BENCH}),
                                        "start": min(_day(e[1]) for e in eps)})
                raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    bars = group_bars(raw)
    records = [episode_record(e, bars) for e in eps]
    wbars = window_bars(records, bars)
    flags = {"nonfinite": _load("_bar_qa").flag_line(wbars),
             "splits": split_steps(wbars, _load("rescale_stale_head").candidates)}
    head = code_version()
    text = render(records, flags, args.env, since, head, now)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head, "env": args.env,
                       "since": since.isoformat() if since else None,
                       "nonfinite_bars": _load("_bar_qa").flag_record(wbars),
                       "split_steps": [list(s) for s in flags["splits"]],
                       "episodes": records}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
