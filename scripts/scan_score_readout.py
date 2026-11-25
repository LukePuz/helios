"""Daily scan composite-score readout, per scan.

Engineering diagnostic (pilot phase), not a study analysis. Reads the
scanner_candidates rows (one per watchlist name scored by
agents/scanner.run_scan) and describes, per scan, the composite score and
its four components (technical, momentum, catalyst, thesis alignment):
quantiles, a histogram in fixed 10-point bins, per watchlist tier, and how
many names sit within N points below the two cuts the scan applied:

  - floor: priority PASS (composite below 25, agents/scanner._priority_label;
    a literal in the code, not stored on the row). Each row's stored
    priority says which side of it the name fell; the readout checks that
    every PASS row is below 25 and every other row is not.
  - regime threshold: read from the row's regime_detail.threshold (written
    since 2026-10-30), and from the scan_tier_funnel activity row of that
    day when one exists (written since 2026-11-25); both are shown when they
    differ. A name reaches the analyst only at or above it (HIGH/MEDIUM).
    Never computed here: a scan without either is shown as not recorded.

A scan is the rows of one New York day; rows more than 15 minutes after the
previous row start a new scan (a re-run). Tier is the name's tier in the
watchlist read now (knowledge_loader.load_watchlist); the header names the
watchlist file's last commit, and where a funnel row exists its per-tier
counts are compared with the tiers used here. scanner_candidates has no
trading_env column: every scan row is read; funnel rows are filtered on the
running env.

Quantiles: inclusive linear interpolation (statistics.quantiles). Counts
and distributions only: no threshold is proposed or chosen here; the N
values are display bands only (--within).

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/scan_score_readout.py [--since YYYY-MM-DD] [--until YYYY-MM-DD] \
        [--within 2,5,10] [--out FILE]

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
FLOOR = 25            # agents/scanner._priority_label: below it, priority PASS
SCAN_GAP_MIN = 15
BINS = [(lo, lo + 9) for lo in range(0, 90, 10)] + [(90, 100)]
COMPONENTS = ("technical_score", "momentum_score", "catalyst_score", "thesis_alignment_score")
TIER_ORDER = ("Core", "Watch", "Speculative")
FUNNEL_STAGE = "scan_tier_funnel"


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _aware(t):
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def scans(rows: list[dict]) -> list[dict]:
    """Rows -> [{label, day, rows}] in time order: one scan per New York
    day, split where the gap to the previous row exceeds SCAN_GAP_MIN."""
    rows = sorted(rows, key=lambda r: (_aware(r["timestamp"]), r["id"]))
    out: list[dict] = []
    prev = None
    for r in rows:
        at = _aware(r["timestamp"])
        day = at.astimezone(NY).date()
        if (prev is None or day != out[-1]["day"]
                or (at - prev).total_seconds() > SCAN_GAP_MIN * 60):
            out.append({"day": day, "rows": []})
        out[-1]["rows"].append(r)
        prev = at
    per_day: dict = {}
    for s in out:
        per_day[s["day"]] = per_day.get(s["day"], 0) + 1
    seen: dict = {}
    for s in out:
        seen[s["day"]] = seen.get(s["day"], 0) + 1
        s["label"] = (str(s["day"]) if per_day[s["day"]] == 1
                      else f"{s['day']} #{seen[s['day']]}")
        s["start"] = _aware(s["rows"][0]["timestamp"]).astimezone(NY)
    return out


def quantiles(xs: list[float]) -> dict | None:
    """{min, p10, p25, p50, p75, p90, max, mean, n}; None when empty."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        q = [xs[0]] * 19
    else:
        q = statistics.quantiles(xs, n=20, method="inclusive")
    return {"n": len(xs), "min": xs[0], "p10": q[1], "p25": q[4], "p50": q[9], "p75": q[14],
            "p90": q[17], "max": xs[-1], "mean": statistics.fmean(xs)}


def histogram(xs: list[float]) -> list[int]:
    """Counts per BINS: 0-9, ..., 80-89, 90-100 (below 0 into the first,
    above 100 into the last)."""
    out = [0] * len(BINS)
    for x in xs:
        out[min(max(int(x // 10), 0), len(BINS) - 1)] += 1
    return out


def thresholds(scan: dict, funnel: dict) -> dict:
    """The regime threshold in force: from the rows' regime_detail and the
    day's funnel row. {row, funnel, value, source, regimes}."""
    row_vals = set()
    regimes = set()
    for r in scan["rows"]:
        rd = r.get("regime_detail") if isinstance(r.get("regime_detail"), dict) else {}
        t = _num(rd.get("threshold"))
        if t is not None:
            row_vals.add(t)
        if r.get("regime_at_scan"):
            regimes.add(r["regime_at_scan"])
    f = funnel.get(scan["day"])
    fv = _num((f or {}).get("threshold"))
    row_v = next(iter(row_vals)) if len(row_vals) == 1 else None
    value = row_v if row_v is not None else fv
    if row_v is not None and fv is not None:
        source = "row = funnel" if row_v == fv else "row (funnel differs)"
    elif row_v is not None:
        source = "row"
    elif fv is not None:
        source = "funnel"
    else:
        source = "not recorded"
    if len(row_vals) > 1:
        source = "funnel (rows mixed)" if fv is not None else "rows mixed"
    return {"row": sorted(row_vals), "funnel": fv, "value": value, "source": source,
            "regimes": sorted(regimes)}


def floor_check(rows: list[dict]) -> str:
    """'ok' when every PASS row is below FLOOR and every other row is not;
    else how many rows disagree."""
    bad = sum(1 for r in rows if r.get("composite_score") is not None
              and ((r.get("priority") == "PASS") != (r["composite_score"] < FLOOR)))
    return "ok" if not bad else f"{bad} rows disagree"


def _tier(tiers: dict, ticker: str) -> str:
    return tiers.get(ticker) or "unknown"


def _tier_list(present) -> list[str]:
    known = [t for t in TIER_ORDER if t in present]
    rest = sorted(t for t in present if t not in TIER_ORDER and t != "unknown")
    return known + rest + (["unknown"] if "unknown" in present else [])


def _g(v) -> str:
    if v is None:
        return "-"
    return f"{v:.1f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _q_cells(q: dict | None) -> str:
    if not q:
        return "| 0 | - | - | - | - | - | - | - |"
    return (f"| {q['n']} | {_g(q['min'])} | {_g(q['p10'])} | {_g(q['p25'])} | {_g(q['p50'])} "
            f"| {_g(q['p75'])} | {_g(q['p90'])} | {_g(q['max'])} |")


def cut_counts(scores: list[float], thr: float | None) -> dict:
    return {"below_floor": sum(1 for s in scores if s < FLOOR),
            "below_thr": None if thr is None else sum(1 for s in scores if s < thr),
            "floor_to_thr": None if thr is None else sum(1 for s in scores if FLOOR <= s < thr),
            "at_or_above": None if thr is None else sum(1 for s in scores if s >= thr)}


def near(scores: list[float], cut: float | None, n: int) -> int | None:
    """Names below the cut by at most n points (cut - n <= score < cut)."""
    if cut is None:
        return None
    return sum(1 for s in scores if cut - n <= s < cut)


def readout(rows: list[dict], funnel: dict, tiers: dict, since: date, until: date, head: str,
            now: datetime, within=(2, 5, 10), watchlist_commit: str = "unknown") -> str:
    sc = scans(rows)
    lines = [f"# Daily scan composite scores ({since} to {until} New York days)", "",
             f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} at code {head}. Pilot-phase "
             "engineering diagnostic, not a study result. Source: scanner_candidates (one row "
             "per scored name), scan_tier_funnel activity rows. Floor = 25 (code literal in "
             "agents/scanner._priority_label; PASS below it). Regime threshold = the one stored "
             "on that scan's rows (regime_detail.threshold) or its funnel row; the analyst "
             "gets names at or above it. Tier = watchlist tier read now (watchlist.md last "
             f"commit {watchlist_commit}). Quantiles: inclusive interpolation. Counts and "
             "distributions only; no threshold proposed.", "",
             "## 1. Per scan", "",
             "| scan | start (NY) | regime | threshold (source) | floor check | n | min | p10 "
             "| p25 | p50 | p75 | p90 | max | below floor | floor to threshold "
             "| at/above threshold | priorities |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    info = []
    for s in sc:
        scores = [float(r["composite_score"]) for r in s["rows"]
                  if r.get("composite_score") is not None]
        th = thresholds(s, funnel)
        thr = th["value"]
        cc = cut_counts(scores, thr)
        pri: dict = {}
        for r in s["rows"]:
            pri[r.get("priority") or "?"] = pri.get(r.get("priority") or "?", 0) + 1
        info.append((s, scores, th, thr))
        q = _q_cells(quantiles(scores))
        th_txt = ("not recorded" if thr is None else
                  f"{_g(thr)} ({th['source']}" + (f"; funnel {_g(th['funnel'])}"
                                                   if th["source"] == "row (funnel differs)" else "")
                  + ")")
        lines.append(
            f"| {s['label']} | {s['start']:%H:%M} | {', '.join(th['regimes']) or '?'} "
            f"| {th_txt} | {floor_check(s['rows'])} {q} {cc['below_floor']} "
            f"| {_g(cc['floor_to_thr'])} | {_g(cc['at_or_above'])} "
            f"| {', '.join(f'{k} {v}' for k, v in sorted(pri.items()))} |")
    if not sc:
        lines.append("| - | - | - | - | - | 0 | - | - | - | - | - | - | - | 0 | - | - | - |")

    lines += ["", "## 2. Histogram (composite, 10-point bins; last bin 90-100)", "",
              "| scan | " + " | ".join(f"{lo}-{hi}" for lo, hi in BINS) + " |",
              "|---|" + "---|" * len(BINS)]
    for s, scores, _, _ in info:
        lines.append(f"| {s['label']} | " + " | ".join(str(c) for c in histogram(scores)) + " |")

    lines += ["", "## 3. Within N points below each cut", "",
              "Below the floor by at most N: 25 - N <= score < 25. Below the threshold by at "
              "most N: threshold - N <= score < threshold. N values are display bands only.", "",
              "| scan | threshold | " + " | ".join(f"floor -{n}" for n in within) + " | "
              + " | ".join(f"threshold -{n}" for n in within) + " | max score "
              "| points from max to threshold |",
              "|---|---|" + "---|" * (2 * len(within) + 2)]
    for s, scores, _, thr in info:
        mx = max(scores) if scores else None
        lines.append(
            f"| {s['label']} | {_g(thr) if thr is not None else 'not recorded'} | "
            + " | ".join(str(near(scores, FLOOR, n)) for n in within) + " | "
            + " | ".join(_g(near(scores, thr, n)) for n in within)
            + f" | {_g(mx)} | {_g(thr - mx) if thr is not None and mx is not None else '-'} |")

    lines += ["", "## 4. Per tier", "",
              "| scan | tier | funnel count | n | min | p10 | p25 | p50 | p75 | p90 | max "
              "| below floor | floor to threshold | at/above threshold | "
              + " | ".join(f"threshold -{n}" for n in within) + " |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(within)]
    for s, _, _, thr in info:
        by: dict = {}
        for r in s["rows"]:
            if r.get("composite_score") is not None:
                by.setdefault(_tier(tiers, r["ticker"]), []).append(float(r["composite_score"]))
        f = funnel.get(s["day"]) or {}
        ftiers = f.get("by_tier") if isinstance(f.get("by_tier"), dict) else {}
        for t in _tier_list(set(by) | set(ftiers)):
            xs = by.get(t, [])
            cc = cut_counts(xs, thr)
            fc = (ftiers.get(t) or {}).get("watchlist_in") if ftiers else None
            fc_txt = "-" if fc is None else (str(fc) if fc == len(xs) else f"{fc} (differs)")
            lines.append(
                f"| {s['label']} | {t} | {fc_txt} {_q_cells(quantiles(xs))} "
                f"{cc['below_floor']} | {_g(cc['floor_to_thr'])} | {_g(cc['at_or_above'])} | "
                + " | ".join(_g(near(xs, thr, n)) for n in within) + " |")

    lines += ["", "## 5. Components", "",
              "| scan | component | n | min | p10 | p25 | p50 | p75 | p90 | max | mean |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, _, _, _ in info:
        for c in COMPONENTS:
            q = quantiles([_num(r.get(c)) for r in s["rows"]])
            lines.append(f"| {s['label']} | {c.replace('_score', '')} {_q_cells(q)} "
                         f"{_g(round(q['mean'], 1)) if q else '-'} |")
    return "\n".join(lines) + "\n"


def _fetch(since: date, until: date, env: str):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    end = datetime.combine(until + timedelta(days=1), time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cols = ("id", "timestamp", "ticker", "composite_score", *COMPONENTS, "priority",
                    "regime_at_scan", "regime_detail")
            cur.execute("SELECT id, timestamp, ticker, composite_score, technical_score, "
                        "momentum_score, catalyst_score, thesis_alignment_score, priority, "
                        "regime_at_scan, regime_detail FROM scanner_candidates "
                        "WHERE timestamp >= %s AND timestamp < %s", (start, end))
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.execute("SELECT id, extra_json FROM activity_feed WHERE stage = %s "
                        "AND ts >= %s AND ts < %s ORDER BY id", (FUNNEL_STAGE, start, end))
            funnel: dict = {}
            for _, x in cur.fetchall():
                if isinstance(x, dict) and x.get("env") == env and x.get("day"):
                    funnel[date.fromisoformat(x["day"])] = x   # the day's last row
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return rows, funnel


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", HERE, *args], capture_output=True,
                              text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    today = datetime.now(NY).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=str(today - timedelta(days=8)), help="YYYY-MM-DD, New York")
    ap.add_argument("--until", default=str(today), help="YYYY-MM-DD, New York, inclusive")
    ap.add_argument("--within", default="2,5,10", help="display bands in points, comma-separated")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)
    within = tuple(int(x) for x in args.within.split(",") if x.strip())
    from config import TRADING_ENV
    from knowledge_loader import load_watchlist
    tiers = {t: e.get("tier") for t, e in load_watchlist().items()}
    rows, funnel = _fetch(since, until, TRADING_ENV)
    text = readout(rows, funnel, tiers, since, until, _git("rev-parse", "--short", "HEAD"),
                   datetime.now(timezone.utc), within=within,
                   watchlist_commit=_git("log", "-1", "--format=%h %ad", "--date=short", "--",
                                         "knowledge/watchlist.md"))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
