"""Shadow readout: how often each log-only shadow fired (audit A1031-1453-4).

About 30 shadows (would_trim_review, udr_check.gate_disagrees, event_burst,
gate_flags.*, would_revise, plan_compiled, r_shadow, heat_shadow, would_swap,
SELL_SKIPPED_CLOSED, PLAN_LEVEL_HIT, ...) are written to signals rows and
never read back. This counts them over a window so each packet decision can
start from a standing number instead of an ad-hoc query.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/shadow_readout.py [--days 7] [--out FILE]

For the window it reports: rows per signal kind, and for every key path in
signals.provenance and signals.decision (to depth 3) the number of rows that
carry it and, for booleans, how many were true. Counts only: no values, no
prices, no account data. Which count decides anything is a study choice, not
this script's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_DEPTH = 3


def walk(obj, prefix: str, depth: int, present: Counter, true: Counter) -> None:
    """Count key paths (dict keys joined with '.') and true booleans."""
    if not isinstance(obj, dict) or depth > MAX_DEPTH:
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        present[path] += 1
        if value is True:
            true[path] += 1
        walk(value, path, depth + 1, present, true)


def readout(rows, days: int, env: str, head: str) -> str:
    kinds: Counter = Counter()
    present: Counter = Counter()
    true: Counter = Counter()
    for kind, provenance, decision in rows:
        kinds[kind or "?"] += 1
        walk(provenance or {}, "provenance", 1, present, true)
        walk(decision or {}, "decision", 1, present, true)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Shadow readout ({env}, last {days} days)", "",
             f"Generated {now} at code {head}. Counts only.", "",
             f"Rows: {sum(kinds.values())}", "", "## Signal kinds", "",
             "| kind | rows |", "|---|---|"]
    lines += [f"| {k} | {n} |" for k, n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))]
    lines += ["", "## Key paths (rows carrying the key; true = boolean true)", "",
              "| path | rows | true |", "|---|---|---|"]
    lines += [f"| {p} | {present[p]} | {true[p] if p in true else ''} |" for p in sorted(present)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()

    import db
    from config import TRADING_ENV
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT signal, provenance, decision FROM signals "
                        "WHERE trading_env=%s AND timestamp >= NOW() - make_interval(days => %s)",
                        (TRADING_ENV, args.days))
            rows = cur.fetchall()
    finally:
        db._put_conn(conn)
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = readout(rows, args.days, TRADING_ENV, head or "unknown")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
