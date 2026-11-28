"""Forward-return readout: how names moved after rows of chosen signal kinds.

Engineering diagnostic, not a study analysis. Built to measure the live entry
gates from their first session: compare the buys the gate let through (OPEN,
ADD) with the ones it stopped (ENTRY_GATE_BLOCKED, split by which gate
fired), and the held-trend shadow rows (TRIM_EXIT_RULE) the same way.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/forward_readout.py [--since 2026-11-23] \
        [--kinds ENTRY_GATE_BLOCKED,OPEN,ADD] [--by conviction] [--out FILE]

--by conviction (default kinds OPEN,ADD) groups the buy verdicts instead by
the model's conviction, two ways, one row per name per session 0 in each:
  - the integer: decision.final_conviction, else signals.confidence x 10;
  - the token-level expectation provenance.llm_call.conviction_logprobs.expected
    (recorded since late October), in the bands fixed in LOGPROB_BANDS before
    any outcome was read: < 6.5, 6.5 to < 7.5, >= 7.5, plus "no logprobs".
It adds, per horizon, Spearman's rank correlation of the integer and of the
expectation with the excess return, and a 95% interval from a bootstrap that
resamples session-0 dates (same-day rows move together). The correlation is
printed only with at least two values held by 5+ rows (integer) or 10+ rows
(expectation) and 10+ dates; otherwise "too few".

For each kind it takes the first row per name per session 0, where session 0 is
the row's New York date if it was written before 16:00 ET on a cached session,
else the next cached session. It reads closes from the nightly price_history
cache, with no vendor or broker call. The return at k sessions is close(0+k) /
close(0) - 1 on the name's own bars. The excess return subtracts SPY over the
same two dates. A row whose k-th bar is not cached yet counts as pending.
A row whose base or k-th close (the name's, or SPY's for the excess) is NaN
or infinite counts as non-finite, with the tickers named: it is kept out of
pending and of every statistic. No bar is dropped or changed (whether to is
Luke's; scripts/_bar_qa.py flags the bars).
Percentages only: no prices, sizes or account data. Which number decides
anything (and after how many rows) is a separate choice, not this script's.
Read-only: one transaction opened READ ONLY, then rolled back.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import math
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY = ZoneInfo("America/New_York")
HORIZONS = (5, 10, 20)
BENCHMARK = "SPY"
DEFAULT_KINDS = ("ENTRY_GATE_BLOCKED", "OPEN", "ADD")
CONVICTION_KINDS = ("OPEN", "ADD")
# Fixed before any outcome was read (gap hunt 2026-11-21); do not tune on results.
LOGPROB_BANDS = ((None, 6.5, "< 6.5"), (6.5, 7.5, "6.5 to < 7.5"), (7.5, None, ">= 7.5"))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 23
NONFINITE = "non-finite"   # forward_return when the base or the k-th close is NaN or infinite


def session_zero(ts: datetime, sessions: list[date]) -> date | None:
    """The row's New York date if written before the 16:00 close on a cached
    session, else the next cached session (None past the last cached bar)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local = ts.astimezone(NY)
    d = local.date()
    i = bisect.bisect_left(sessions, d)
    if i < len(sessions) and sessions[i] == d and local.time() < time(16, 0):
        return d
    j = bisect.bisect_right(sessions, d)
    return sessions[j] if j < len(sessions) else None


def nonfinite(v) -> bool:
    """True for a NaN or infinite number; None and non-numbers are not."""
    if v is None or isinstance(v, bool):
        return False
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):
        return False


def forward_return(closes: dict[date, float], dates: list[date], d0: date,
                   k: int) -> float | str | None:
    """close k sessions after d0 over close at d0, on the name's own bars.
    None when d0 is not a cached bar of the name or the k-th bar is not in yet;
    NONFINITE when either close is NaN or infinite (`not base` is False for
    NaN, so it is tested first)."""
    if d0 not in closes:
        return None
    i = bisect.bisect_left(dates, d0)
    if i + k >= len(dates):
        return None
    base, end = closes[d0], closes[dates[i + k]]
    if nonfinite(base) or nonfinite(end):
        return NONFINITE
    if not base:
        return None
    return end / base - 1


def gate_label(kind: str, decision) -> str:
    """ENTRY_GATE_BLOCKED rows split by which gate fired; other kinds as is."""
    if kind != "ENTRY_GATE_BLOCKED":
        return kind
    gate = (decision or {}).get("entry_gate") if isinstance(decision, dict) else None
    gate = gate if isinstance(gate, dict) else {}
    fired = [g for g in ("trend", "stopout", "add_loser")
             if isinstance(gate.get(g), dict) and gate[g].get("blocked") is True]
    return f"{kind} ({'+'.join(fired) if fired else 'gate unknown'})"


def _num(x) -> float | None:
    import math
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def conviction_of(fields) -> int | None:
    """The verdict's integer conviction: decision.final_conviction, else
    signals.confidence x 10. None when neither is a number in 1..10."""
    fields = fields if isinstance(fields, dict) else {}
    v = _num(fields.get("final_conviction"))
    if v is None:
        c = _num(fields.get("confidence"))
        v = c * 10 if c is not None else None
    if v is None:
        return None
    n = int(round(v))
    return n if 1 <= n <= 10 else None


def logprob_expected_of(fields) -> float | None:
    v = _num((fields if isinstance(fields, dict) else {}).get("logprobs_expected"))
    return v if v is not None and 0 <= v <= 10 else None


def logprob_band(expected: float | None) -> str:
    """The LOGPROB_BANDS label for an expectation (lower bound inclusive)."""
    if expected is None:
        return "no logprobs"
    for lo, hi, label in LOGPROB_BANDS:
        if (lo is None or expected >= lo) and (hi is None or expected < hi):
            return label
    return "no logprobs"


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman's rho (average ranks for ties); None if either side is constant."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def day_block_rho(recs: list[dict], key: str, k: int, min_per_level: int | None,
                  min_rows: int = 10, min_dates: int = 10) -> str:
    """Spearman(recs[key], x_k) with a 95% bootstrap interval over session-0
    dates. min_per_level: for a discrete key, at least two values must each
    hold that many rows; None for a continuous key (min_rows rows instead)."""
    import random
    from collections import Counter
    pts = [(r["d0"], r[key], r[f"x{k}"]) for r in recs
           if r.get(key) is not None and r.get(f"x{k}") is not None]
    dates = {d for d, _, _ in pts}
    if min_per_level is not None:
        enough = sum(1 for c in Counter(v for _, v, _ in pts).values() if c >= min_per_level) >= 2
    else:
        enough = len(pts) >= min_rows and len({v for _, v, _ in pts}) >= 2
    if not enough or len(dates) < min_dates:
        return "too few"
    rho = spearman([v for _, v, _ in pts], [x for _, _, x in pts])
    if rho is None:
        return "too few"
    by_day: dict = {}
    for d, v, x in pts:
        by_day.setdefault(d, []).append((v, x))
    days = sorted(by_day)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [p for _ in days for p in by_day[days[rng.randrange(len(days))]]]
        r = spearman([v for v, _ in sample], [x for _, x in sample])
        if r is not None:
            draws.append(r)
    if len(draws) < BOOTSTRAP_DRAWS // 2:
        return f"{rho:+.3f} (interval unstable)"
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
    return f"{rho:+.3f} ({lo:+.3f} to {hi:+.3f})"


def conviction_extra(fields) -> dict:
    return {"conviction": conviction_of(fields), "logprob_expected": logprob_expected_of(fields)}


def collect(rows, bars: dict[str, list[tuple[date, float]]], label_fn=None,
            extra_fn=None) -> dict[str, list[dict]]:
    """rows: (symbol, kind, ts, decision). bars: ticker -> [(date, close)].
    Returns label -> one record per (symbol, session 0) with raw and excess
    returns per horizon (None while pending or non-finite) and nf<k>: the
    tickers whose non-finite close blocked horizon k (empty when none; the
    row then has f and x None). label_fn(kind, decision) names
    the group (default gate_label); extra_fn(decision) adds fields to each
    record (default none)."""
    label_fn = label_fn or gate_label
    series = {}
    for t, b in bars.items():
        clean = sorted((d, float(c)) for d, c in b if c is not None)
        series[t] = ({d: c for d, c in clean}, [d for d, _ in clean])
    bench = series.get(BENCHMARK)
    sessions = bench[1] if bench else []
    seen, out = set(), {}
    for symbol, kind, ts, decision in sorted(rows, key=lambda r: r[2]):
        d0 = session_zero(ts, sessions)
        if d0 is None:
            continue
        label = label_fn(kind, decision)
        key = (label, symbol, d0)
        if key in seen:
            continue
        seen.add(key)
        rec = {"symbol": symbol, "d0": d0}
        if extra_fn is not None:
            rec.update(extra_fn(decision))
        s = series.get(symbol)
        for k in HORIZONS:
            raw = forward_return(*s, d0, k) if s else None
            nf = []
            if raw == NONFINITE:
                raw, nf = None, [symbol]
            spy = None
            if raw is not None and bench:
                # SPY over the same two dates as the name, not SPY's own k-th bar,
                # so a bar missing from the name cannot misalign the two windows.
                dk = s[1][bisect.bisect_left(s[1], d0) + k]
                b0, bk = bench[0].get(d0), bench[0].get(dk)
                if nonfinite(b0) or nonfinite(bk):
                    raw, nf = None, [BENCHMARK]
                else:
                    spy = bk / b0 - 1 if b0 and bk is not None else None
            rec[f"f{k}"] = raw
            rec[f"x{k}"] = raw - spy if raw is not None and spy is not None else None
            rec[f"nf{k}"] = nf
        out.setdefault(label, []).append(rec)
    return out


def _cell(vals: list[float]) -> str:
    if not vals:
        return "- | - | -"
    hit = sum(v > 0 for v in vals) / len(vals)
    return (f"{statistics.mean(vals) * 100:+.1f} | {statistics.median(vals) * 100:+.1f} | "
            f"{hit * 100:.0f}%")


TABLE_HEAD = ["| group | horizon | rows | pending | non-finite | names | mean x % | median x % "
              "| x > 0 | mean f % |", "|---|---|---|---|---|---|---|---|---|---|"]


def _group_lines(label: str, recs: list[dict]) -> list[str]:
    """One table line per horizon. Non-finite rows (nf<k> non-empty) are out
    of pending and of the stats; the cell names their tickers."""
    names = len({r["symbol"] for r in recs})
    out = []
    for k in HORIZONS:
        bad = [r for r in recs if r.get(f"nf{k}")]
        xs = [r[f"x{k}"] for r in recs if r[f"x{k}"] is not None]
        fs = [r[f"f{k}"] for r in recs if r[f"f{k}"] is not None]
        f_mean = f"{statistics.mean(fs) * 100:+.1f}" if fs else "-"
        tick = sorted({t for r in bad for t in r[f"nf{k}"]})
        nf = f"{len(bad)} ({', '.join(tick)})" if bad else "0"
        out.append(f"| {label} | {k} | {len(recs)} | {len(recs) - len(xs) - len(bad)} | {nf} | "
                   f"{names} | {_cell(xs)} | {f_mean} |")
    return out


def readout(groups: dict[str, list[dict]], since: date, env: str, head: str,
            now: datetime | None = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Forward-return readout ({env}, rows since {since})", "",
             f"Generated {now} at code {head}. Engineering diagnostic, not a study result. "
             "One row per name per session 0; x = minus SPY over the same sessions, "
             "f = raw. Non-finite = the base or k-th close (the name's, or SPY's) is NaN or "
             "infinite: those rows are out of pending and the stats, and no bar is dropped or "
             "changed. Percentages only.", ""] + TABLE_HEAD
    for label in sorted(groups):
        lines += _group_lines(label, groups[label])
    if not groups:
        lines.append("| (no rows) | | | | | | | | | |")
    lines += ["", "Small groups are noise: rows of one day move together, so a few sessions "
              "can look like a pattern."]
    return "\n".join(lines) + "\n"


def conviction_groups(rows, bars) -> tuple[dict, dict]:
    """The two conviction views of the same buy verdicts: by integer, and by
    logprob band. One record per name per session 0 in each: the first row of
    the session, whatever its conviction (collect's own dedup is per label, so
    a later verdict with another conviction would otherwise count again)."""
    bench = sorted(d for d, c in bars.get(BENCHMARK, []) if c is not None)
    first, seen = [], set()
    for row in sorted(rows, key=lambda r: r[2]):
        d0 = session_zero(row[2], bench)
        if d0 is None or (row[0], d0) in seen:
            continue
        seen.add((row[0], d0))
        first.append(row)
    rows = first
    by_int = collect(rows, bars,
                     label_fn=lambda kind, f: (f"c{conviction_of(f)}" if conviction_of(f) is not None
                                               else "c? (no conviction)"),
                     extra_fn=conviction_extra)
    by_band = collect(rows, bars,
                      label_fn=lambda kind, f: f"logprob {logprob_band(logprob_expected_of(f))}",
                      extra_fn=conviction_extra)
    return by_int, by_band


def conviction_readout(rows, bars, since: date, env: str, head: str,
                       now: datetime | None = None) -> str:
    """Markdown for --by conviction: both views with per-horizon rank correlations."""
    by_int, by_band = conviction_groups(rows, bars)
    bands = "; ".join(label for _, _, label in LOGPROB_BANDS)
    text = readout(by_int, since, env, head, now).replace(
        "# Forward-return readout (", "# Forward-return readout by conviction (", 1)
    head_lines = ["Conviction = decision.final_conviction, else signals.confidence x 10. "
                  f"Logprob bands (provenance.llm_call.conviction_logprobs.expected), fixed before "
                  f"any outcome was read: {bands}; lower bound inclusive. Rank intervals: "
                  f"{BOOTSTRAP_DRAWS} bootstrap draws over session-0 dates, seed {BOOTSTRAP_SEED}.", ""]
    lines = text.rstrip("\n").split("\n")
    lines[2:2] = head_lines
    lines += ["", "## By logprob band", ""] + TABLE_HEAD
    order = [f"logprob {label}" for _, _, label in LOGPROB_BANDS] + ["logprob no logprobs"]
    for label in order:
        recs = by_band.get(label)
        if not recs:
            continue
        lines += _group_lines(label, recs)
    all_recs = [r for recs in by_int.values() for r in recs]
    lines += ["", "## Rank correlation with the excess return (95% interval, day blocks)", "",
              "| horizon | integer conviction | logprob expectation |", "|---|---|---|"]
    for k in HORIZONS:
        lines.append(f"| {k} | {day_block_rho(all_recs, 'conviction', k, 5)} | "
                     f"{day_block_rho(all_recs, 'logprob_expected', k, None)} |")
    return "\n".join(lines) + "\n"


def _fetch(kinds: list[str], since: date, env: str, by_conviction: bool = False):
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            # signals.trading_env is NULL before 2026-10-28 05:20 ET.
            fields = ("jsonb_build_object('final_conviction', decision->'final_conviction', "
                      "'confidence', confidence, 'logprobs_expected', "
                      "provenance->'llm_call'->'conviction_logprobs'->'expected')"
                      if by_conviction else "decision")
            cur.execute(f"SELECT symbol, signal, timestamp, {fields} FROM signals "
                        "WHERE signal = ANY(%s) AND timestamp >= %s "
                        "AND (trading_env = %s OR trading_env IS NULL)",
                        (kinds, datetime.combine(since, time(0), tzinfo=NY), env))
            rows = cur.fetchall()
            tickers = sorted({r[0] for r in rows} | {BENCHMARK})
            cur.execute("SELECT ticker, date, close FROM price_history "
                        "WHERE ticker = ANY(%s) AND date >= %s",
                        (tickers, since - timedelta(days=7)))
            bars: dict[str, list] = {}
            for t, d, c in cur.fetchall():
                bars.setdefault(t, []).append((d, c))
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return rows, bars


def _bar_qa():
    """scripts/_bar_qa.py (the non-finite bar flag) as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bar_qa.py")
    spec = importlib.util.spec_from_file_location("_bar_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="YYYY-MM-DD, New York date (default 30 days ago)")
    ap.add_argument("--kinds", default=None,
                    help="comma-separated signal kinds (default ENTRY_GATE_BLOCKED,OPEN,ADD; "
                         "OPEN,ADD with --by conviction)")
    ap.add_argument("--by", choices=("gate", "conviction"), default="gate",
                    help="group by signal kind and gate (default) or by the model's conviction")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since = (date.fromisoformat(args.since) if args.since
             else datetime.now(NY).date() - timedelta(days=30))
    by_conv = args.by == "conviction"
    kinds_arg = args.kinds or ",".join(CONVICTION_KINDS if by_conv else DEFAULT_KINDS)
    kinds = [k.strip() for k in kinds_arg.split(",") if k.strip()]
    from config import TRADING_ENV
    rows, bars = _fetch(kinds, since, TRADING_ENV, by_conviction=by_conv)
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = (conviction_readout(rows, bars, since, TRADING_ENV, head or "unknown") if by_conv
            else readout(collect(rows, bars), since, TRADING_ENV, head or "unknown"))
    text += "\n" + _bar_qa().flag_line(bars) + "\n"   # flag only (scripts/_bar_qa.py)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
