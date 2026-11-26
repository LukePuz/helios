"""Analyst UDR dispersion readout: the model's UDR next to the structural UDR,
per ticker per New York day.

Engineering diagnostic (pilot phase), not a study analysis. A verdict is a
signals row that carries the analyst's decision record (decision has
final_action); routing and shadow rows are not verdicts. Per verdict:
  - model UDR = decision.udr_check.model, the model's own
    upside_downside_ratio;
  - structural UDR = decision.udr_structural.udr_structural, computed in code
    by agents/analyst._udr_structural from compile_plan's levels as
    (t1 - price) / (price - structural_stop); None unless stop < price < t1,
    with the reason in udr_structural.why_null. Recorded since 2026-11-02
    (f69a515); older verdicts have no structural UDR;
  - structural UDR to t2 = decision.udr_structural.udr_structural_t2, the
    same with t2, reported in its own section;
  - difference = model UDR - structural UDR, on calls that have both.

Within one ticker and one New York day, verdicts are ordered by (timestamp,
id). A crossing is a pair of consecutive verdicts whose UDR falls on
different sides of the bar (one below it, one at or above it). The bar is
read from config.MIN_UPSIDE_DOWNSIDE_RATIO, never chosen here; the
analyst's entry gate blocks an OPEN/ADD whose UDR is below 2.0. Pairs where
either call lacks that UDR are counted as unknown, not compared. Pairs that
span two days are not compared.

Counts and distributions only: no thresholds of ours, no prices, nothing
recommended. Rows are filtered on trading_env, so rows from before
2026-10-28, whose trading_env is NULL, are not read (see
audit-method-notes.md).

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/udr_dispersion_readout.py [--since YYYY-MM-DD] [--until YYYY-MM-DD] \
        [--ticker NVDA] [--detail] [--min-calls N] [--out FILE]

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
SERIES = (("model", "model UDR"), ("structural", "structural UDR"))


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


def verdict(row: dict) -> dict:
    """One signals row (id, symbol, timestamp, signal, confidence, decision)
    normalised to {id, ticker, at, day, signal, conviction, model, structural,
    structural_t2, why_null, has_structural_record}."""
    dec = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    conv = _num(dec.get("final_conviction"))
    if conv is None:
        c = _num(row.get("confidence"))
        conv = round(c * 10, 1) if c is not None else None
    chk = dec.get("udr_check") if isinstance(dec.get("udr_check"), dict) else {}
    st = dec.get("udr_structural") if isinstance(dec.get("udr_structural"), dict) else None
    at = _aware(row["timestamp"])
    return {"id": row["id"], "ticker": row["symbol"], "at": at, "day": at.astimezone(NY).date(),
            "signal": row["signal"], "conviction": conv, "model": _num(chk.get("model")),
            "structural": _num((st or {}).get("udr_structural")),
            "structural_t2": _num((st or {}).get("udr_structural_t2")),
            "why_null": (st or {}).get("why_null") or (st or {}).get("error"),
            "has_structural_record": st is not None}


def group(verdicts: list[dict]) -> dict:
    """{(day, ticker): [verdicts ordered by (at, id)]}."""
    out: dict = {}
    for v in verdicts:
        out.setdefault((v["day"], v["ticker"]), []).append(v)
    for rows in out.values():
        rows.sort(key=lambda v: (v["at"], v["id"]))
    return out


def spread(values) -> dict | None:
    """n, min, p25, median, p75, max, IQR (inclusive quartiles), or None."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        q1 = q3 = vals[0]
    else:
        q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    return {"n": len(vals), "min": vals[0], "p25": q1, "median": statistics.median(vals),
            "p75": q3, "max": vals[-1], "iqr": q3 - q1}


def _below(v, bar) -> bool:
    return v < bar


def crossings(rows: list[dict], key: str, bar: float) -> dict:
    """Consecutive pairs of one ticker-day: pairs, known (both calls carry
    the value), crossed (the two values sit on different sides of bar)."""
    pairs = list(zip(rows, rows[1:]))
    known = [(a[key], b[key]) for a, b in pairs if a[key] is not None and b[key] is not None]
    return {"pairs": len(pairs), "known": len(known),
            "crossed": sum(1 for x, y in known if _below(x, bar) != _below(y, bar))}


def both_crossed(rows: list[dict], bar: float) -> int:
    """Pairs where the model and the structural UDR both cross the bar."""
    n = 0
    for a, b in zip(rows, rows[1:]):
        vals = (a["model"], b["model"], a["structural"], b["structural"])
        if None in vals:
            continue
        if (_below(vals[0], bar) != _below(vals[1], bar)
                and _below(vals[2], bar) != _below(vals[3], bar)):
            n += 1
    return n


def diffs(rows: list[dict]) -> list[float]:
    return [round(v["model"] - v["structural"], 4) for v in rows
            if v["model"] is not None and v["structural"] is not None]


def _f(v) -> str:
    if v is None:
        return "?"
    return f"{round(v, 2):g}"


def _spread_cells(s: dict | None) -> str:
    if not s:
        return "0 | - | - | - | - | -"
    return f"{s['n']} | {_f(s['min'])} | {_f(s['median'])} | {_f(s['max'])} | {_f(s['iqr'])}"


def _cross_cell(c: dict) -> str:
    return f"{c['crossed']} of {c['known']}"


def _sum_cross(cs: list[dict]) -> dict:
    return {k: sum(c[k] for c in cs) for k in ("pairs", "known", "crossed")}


def readout(verdicts: list[dict], since: date, until: date, env: str, head: str, now: datetime,
            bar: float, ticker: str | None = None, detail: bool = False,
            min_calls: int = 2) -> str:
    groups = group(verdicts)
    days = sorted({d for d, _ in groups})
    scope = f", ticker {ticker}" if ticker else ""
    lines = [f"# Analyst UDR dispersion ({env}, {since} to {until} New York days{scope})", "",
             f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} at code {head}. Pilot-phase "
             "engineering diagnostic, not a study result. Model UDR = decision.udr_check.model; "
             "structural UDR = decision.udr_structural.udr_structural, (t1 - price) / "
             "(price - structural_stop) from compile_plan, recorded since 2026-11-02. "
             "Difference = model - structural. A crossing is a pair of consecutive verdicts of "
             f"one ticker on one New York day on different sides of the bar {bar:g} "
             "(config.MIN_UPSIDE_DOWNSIDE_RATIO; below vs at or above). \"k of n\" = k "
             "crossings among n pairs where both calls carry that UDR. Spreads: n, min, "
             "median, max, IQR (inclusive quartiles). Counts and distributions only; no "
             "thresholds chosen.", "",
             "## 1. Per day (all tickers)", "",
             "| day | verdicts | ticker-days | model n | model min | model median | model max "
             "| model IQR | structural n | structural min | structural median | structural max "
             "| structural IQR | consecutive pairs | model crossings | structural crossings "
             "| both cross |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for day in days:
        keys = sorted(k for k in groups if k[0] == day)
        rows = [v for k in keys for v in groups[k]]
        cm = _sum_cross([crossings(groups[k], "model", bar) for k in keys])
        cs = _sum_cross([crossings(groups[k], "structural", bar) for k in keys])
        lines.append(
            f"| {day} | {len(rows)} | {len(keys)} | {_spread_cells(spread(v['model'] for v in rows))} "
            f"| {_spread_cells(spread(v['structural'] for v in rows))} | {cm['pairs']} "
            f"| {_cross_cell(cm)} | {_cross_cell(cs)} "
            f"| {sum(both_crossed(groups[k], bar) for k in keys)} |")
    if not days:
        lines.append("| - | 0 | 0 | " + " | ".join(["0", "-", "-", "-", "-"] * 2)
                     + " | 0 | 0 of 0 | 0 of 0 | 0 |")

    lines += ["", "## 2. Difference (model - structural) per day", "",
              "| day | calls with both | min | p25 | median | p75 | max | model above "
              "| model below | equal |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for day in days:
        d = diffs([v for k in groups if k[0] == day for v in groups[k]])
        s = spread(d)
        if not s:
            lines.append(f"| {day} | 0 | - | - | - | - | - | 0 | 0 | 0 |")
            continue
        lines.append(f"| {day} | {s['n']} | {_f(s['min'])} | {_f(s['p25'])} | {_f(s['median'])} "
                     f"| {_f(s['p75'])} | {_f(s['max'])} | {sum(1 for x in d if x > 0)} "
                     f"| {sum(1 for x in d if x < 0)} | {sum(1 for x in d if x == 0)} |")
    if not days:
        lines.append("| - | 0 | - | - | - | - | - | 0 | 0 | 0 |")

    lines += ["", "## 3. Calls without a structural UDR", "",
              "No record = the verdict has no udr_structural (before 2026-11-02 or a degraded "
              "row). Otherwise the reason is udr_structural.why_null (or its error).", "",
              "| day | verdicts | no record | reason | calls |", "|---|---|---|---|---|"]
    any_null = False
    for day in days:
        rows = [v for k in groups if k[0] == day for v in groups[k]]
        no_rec = sum(1 for v in rows if not v["has_structural_record"])
        reasons: dict = {}
        for v in rows:
            if v["has_structural_record"] and v["structural"] is None:
                r = str(v["why_null"] or "not recorded")
                reasons[r] = reasons.get(r, 0) + 1
        if not reasons and not no_rec:
            continue
        any_null = True
        if not reasons:
            lines.append(f"| {day} | {len(rows)} | {no_rec} | - | 0 |")
        for r, n in sorted(reasons.items()):
            lines.append(f"| {day} | {len(rows)} | {no_rec} | {r} | {n} |")
    if not any_null:
        lines.append("| - | - | 0 | - | 0 |")

    lines += ["", "## 4. Structural UDR to t2 per day", "",
              "udr_structural.udr_structural_t2, (t2 - price) / (price - structural_stop).", "",
              "| day | n | min | median | max | IQR | crossings |", "|---|---|---|---|---|---|---|"]
    for day in days:
        keys = sorted(k for k in groups if k[0] == day)
        c2 = _sum_cross([crossings(groups[k], "structural_t2", bar) for k in keys])
        lines.append(f"| {day} | "
                     + _spread_cells(spread(v["structural_t2"] for k in keys for v in groups[k]))
                     + f" | {_cross_cell(c2)} |")
    if not days:
        lines.append("| - | 0 | - | - | - | - | 0 of 0 |")

    lines += ["", f"## 5. Per ticker and day (ticker-days with at least {min_calls} verdicts)", "",
              "| day | ticker | verdicts | model n | model min | model median | model max "
              "| model IQR | structural n | structural min | structural median | structural max "
              "| structural IQR | diff median | diff min | diff max | model crossings "
              "| structural crossings | both cross |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for day in days:
        for k in sorted(k for k in groups if k[0] == day):
            rows = groups[k]
            if len(rows) < min_calls:
                continue
            any_row = True
            ds = spread(diffs(rows))
            dcells = (f"{_f(ds['median'])} | {_f(ds['min'])} | {_f(ds['max'])}" if ds
                      else "- | - | -")
            lines.append(
                f"| {day} | {k[1]} | {len(rows)} | {_spread_cells(spread(v['model'] for v in rows))} "
                f"| {_spread_cells(spread(v['structural'] for v in rows))} | {dcells} "
                f"| {_cross_cell(crossings(rows, 'model', bar))} "
                f"| {_cross_cell(crossings(rows, 'structural', bar))} "
                f"| {both_crossed(rows, bar)} |")
    if not any_row:
        lines.append("| - | - | 0 | " + " | ".join(["0", "-", "-", "-", "-"] * 2)
                     + " | - | - | - | 0 of 0 | 0 of 0 | 0 |")

    if detail:
        lines += ["", "## 6. Each call", "",
                  "Times New York. Side = below or at/above the bar; a crossing is marked x on "
                  "the later call of the pair.", "",
                  "| day | ticker | time | signal | conviction | model UDR | model side "
                  "| structural UDR | structural side | structural t2 | difference | signal id |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]

        def side(prev, cur, key):
            if cur[key] is None:
                return "?"
            s = "below" if _below(cur[key], bar) else "at/above"
            if prev is not None and prev[key] is not None and \
                    _below(prev[key], bar) != _below(cur[key], bar):
                s += " x"
            return s
        for k in sorted(groups):
            prev = None
            for v in groups[k]:
                diff = (v["model"] - v["structural"]
                        if v["model"] is not None and v["structural"] is not None else None)
                lines.append(
                    f"| {k[0]} | {k[1]} | {v['at'].astimezone(NY):%H:%M:%S} | {v['signal']} "
                    f"| {_f(v['conviction'])} | {_f(v['model'])} | {side(prev, v, 'model')} "
                    f"| {_f(v['structural'])} | {side(prev, v, 'structural')} "
                    f"| {_f(v['structural_t2'])} | {_f(diff)} | {v['id']} |")
                prev = v
        if not groups:
            lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def _fetch(since: date, until: date, env: str, ticker: str | None):
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    end = datetime.combine(until + timedelta(days=1), time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            sql = ("SELECT id, symbol, timestamp, signal, confidence, "
                   "jsonb_build_object('final_conviction', decision->'final_conviction', "
                   "'udr_check', decision->'udr_check', "
                   "'udr_structural', decision->'udr_structural') FROM signals "
                   "WHERE trading_env = %s AND decision ? 'final_action' "
                   "AND timestamp >= %s AND timestamp < %s")
            args = [env, start, end]
            if ticker:
                sql += " AND symbol = %s"
                args.append(ticker)
            cur.execute(sql, args)
            cols = ("id", "symbol", "timestamp", "signal", "confidence", "decision")
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return [verdict(r) for r in rows]


def main() -> None:
    today = datetime.now(NY).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=str(today - timedelta(days=6)), help="YYYY-MM-DD, New York")
    ap.add_argument("--until", default=str(today), help="YYYY-MM-DD, New York, inclusive")
    ap.add_argument("--ticker", help="one ticker only")
    ap.add_argument("--detail", action="store_true", help="add one row per call")
    ap.add_argument("--min-calls", type=int, default=2,
                    help="per-ticker table lists ticker-days with at least this many verdicts")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)
    ticker = args.ticker.upper() if args.ticker else None
    from config import MIN_UPSIDE_DOWNSIDE_RATIO, TRADING_ENV
    verdicts = _fetch(since, until, TRADING_ENV, ticker)
    try:
        head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    text = readout(verdicts, since, until, TRADING_ENV, head, datetime.now(timezone.utc),
                   float(MIN_UPSIDE_DOWNSIDE_RATIO), ticker=ticker, detail=args.detail,
                   min_calls=args.min_calls)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
