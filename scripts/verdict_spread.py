"""Within-day verdict spread: how far the analyst's own conviction and UDR
move for one ticker on one day (audit 2026-11-05-0525 #2).

One 72B sample per event decides each verdict, so a ticker reviewed several
times a day on much the same facts can land on both sides of the UDR entry
floor. This lists, per ticker and ET day with at least --min-verdicts
verdicts, the count, the conviction and UDR ranges (the model's values,
decision.model_conviction and decision.udr_check.model) and the final
actions, and flags a day whose UDR range straddles the live entry floor.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/verdict_spread.py [--days 21] [--min-verdicts 3] [--out FILE]

Read-only and descriptive: no prices, no account data. Whether a spread
matters, and what to do about it (G27/G30), is a study choice, not this
script's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")
ENTRY_ACTIONS = ("OPEN", "ADD")


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def spread(rows, floor: float, min_verdicts: int) -> list[dict]:
    """rows: (symbol, timestamp, decision). One dict per ticker-day with at
    least min_verdicts verdicts carrying udr_check, ordered by day, ticker."""
    days: dict[tuple, list] = defaultdict(list)
    for symbol, ts, decision in rows:
        d = decision or {}
        if not isinstance(d.get("udr_check"), dict):
            continue
        days[(ts.astimezone(ET).date(), symbol)].append(d)
    out = []
    for (day, symbol), ds in sorted(days.items()):
        if len(ds) < min_verdicts:
            continue
        conv = [c for c in (_num(d.get("model_conviction")) for d in ds) if c is not None]
        udr = [u for u in (_num(d["udr_check"].get("model")) for d in ds) if u is not None]
        out.append({
            "day": str(day), "ticker": symbol, "n": len(ds),
            "conviction": (min(conv), max(conv)) if conv else None,
            "udr": (min(udr), max(udr)) if udr else None,
            "crosses_floor": bool(udr) and min(udr) < floor <= max(udr),
            "actions": sorted({str(d.get("final_action")) for d in ds}),
        })
        acts = out[-1]["actions"]
        out[-1]["actions_differ"] = len(acts) > 1
        out[-1]["entry_action"] = any(a in ENTRY_ACTIONS for a in acts)
    return out


def render(days: list[dict], floor: float, window: int, min_verdicts: int, env: str, head: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    crossing = sum(1 for d in days if d["crosses_floor"])
    crossing_mixed = sum(1 for d in days if d["crosses_floor"] and d["actions_differ"])

    def rng(r, fmt):
        return f"{fmt.format(r[0])}-{fmt.format(r[1])}" if r else "n/a"
    lines = [f"# Within-day verdict spread ({env}, last {window} days)", "",
             f"Generated {now} at code {head}. Descriptive only.", "",
             f"Ticker-days with at least {min_verdicts} verdicts: {len(days)}; "
             f"UDR range straddles the {floor:g} entry floor on {crossing}; "
             f"of those, the final action differs within the day on {crossing_mixed}.", "",
             "Days with 3 or more verdicts are mostly held names and news-heavy days. The script does",
             "not check whether the facts were the same across a day's verdicts.", "",
             "| day | ticker | verdicts | conviction | UDR | straddles floor | actions differ | OPEN/ADD occurred | final actions |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {d['day']} | {d['ticker']} | {d['n']} | {rng(d['conviction'], '{:g}')} | "
              f"{rng(d['udr'], '{:.2f}')} | {'yes' if d['crosses_floor'] else ''} | "
              f"{'yes' if d['actions_differ'] else ''} | {'yes' if d['entry_action'] else ''} | "
              f"{', '.join(d['actions'])} |" for d in days]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--min-verdicts", type=int, default=3)
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()

    import db
    from agents.analyst import _ENTRY_UDR_FLOOR
    from config import TRADING_ENV
    from datetime import time, timedelta
    # Whole ET days: from midnight ET, --days days back, so the first day is complete.
    start = datetime.combine(datetime.now(ET).date() - timedelta(days=args.days), time(0), tzinfo=ET)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT symbol, timestamp, decision FROM signals "
                        "WHERE trading_env=%s AND timestamp >= %s "
                        "AND decision ? 'udr_check'", (TRADING_ENV, start))
            rows = cur.fetchall()
        conn.rollback()   # end the read-only transaction before the conn goes back to the pool
    finally:
        db._put_conn(conn)
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = render(spread(rows, _ENTRY_UDR_FLOOR, args.min_verdicts), _ENTRY_UDR_FLOOR,
                  args.days, args.min_verdicts, TRADING_ENV, head or "unknown")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
