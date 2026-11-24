"""Book gap-risk readout: the book_gap_risk rows (tasks.book_gap_risk_record,
one per session after the close) since --since, as a table, plus the newest
row's per-name table. Suspect gaps (|gap| above tasks.GAP_SUSPECT_ABS_PCT,
a possible split or stored-step artifact) are listed per name, and the book
replay is shown with and without the nights they fall on.

Engineering readout, not a study analysis. Percent only (weights and % of
NAV): no values, cash or share counts. No thresholds: what counts as too much
gap exposure is Luke's (gap-budget option 5). Definitions are the row's own
(its "definitions" key), printed once from the newest row.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/gap_risk_readout.py [--since 2026-11-24] [--env live] [--out FILE]

Read-only: one transaction opened READ ONLY with a short statement timeout,
then rolled back.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
DEFAULT_SINCE = date(2026, 11, 24)
STATEMENT_TIMEOUT_MS = 10000
BOOK_COLS = (("session_date", "session"), ("read", "read"), ("n_positions", "names"),
             ("invested_pct", "invested %"), ("loss_at_stops_pct_nav", "loss at stops % NAV"),
             ("loss_at_stops_close_basis_pct_nav", "same, close basis"),
             ("n_common_nights", "common nights"), ("book_gap_p5_pct_nav", "book gap p5 % NAV"),
             ("book_gap_p1_pct_nav", "p1"), ("book_gap_worst_pct_nav", "worst"),
             ("n_common_nights_any_past_stop", "nights any name past its stop"),
             ("suspect_nights", "suspect nights"),
             ("book_gap_p5_pct_nav_ex_suspect", "p5 without them"),
             ("book_gap_p1_pct_nav_ex_suspect", "p1 without them"),
             ("book_gap_worst_pct_nav_ex_suspect", "worst without them"),
             ("close_is_today", "close is today"))
NAME_COLS = (("ticker", "name"), ("weight_pct", "weight %"),
             ("stop_below_close_pct", "stop below close %"), ("n_gaps", "gaps"),
             ("first_gap_date", "from"), ("last_gap_date", "to"), ("gap_p5_pct", "gap p5 %"),
             ("gap_p1_pct", "p1 %"), ("gap_worst_pct", "worst %"), ("gap_worst_date", "worst on"),
             ("nights_past_stop", "nights past stop"), ("suspect_gaps", "suspect gaps %"),
             ("reason", "note"))


def _cell(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, list):    # suspect nights (dates) or suspect gaps ({date, gap_pct})
        return ", ".join(f"{x.get('date')} {_cell(x.get('gap_pct'))}" if isinstance(x, dict)
                         else str(x) for x in v) or "none"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def readout(rows: list[dict], since: date, env: str, head: str) -> str:
    """Markdown from the rows' extra_json dicts, oldest first. One row per
    session: a re-run's duplicate (same shadow_key) is shown once."""
    seen, uniq = set(), []
    for r in rows:
        k = r.get("shadow_key") or id(r)
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    lines = [f"# Book gap risk ({env}) since {since}", "",
             f"Code {head}. Percent of NAV and weights only; no thresholds. Rows: {len(uniq)}.", ""]
    if not uniq:
        return "\n".join(lines + ["No book_gap_risk rows yet."]) + "\n"
    lines += ["| " + " | ".join(t for _, t in BOOK_COLS) + " |",
              "|" + "---|" * len(BOOK_COLS)]
    lines += ["| " + " | ".join(_cell(r.get(k)) for k, _ in BOOK_COLS) + " |" for r in uniq]
    last = uniq[-1]
    lines += ["", f"## Names on {last.get('session_date')}", "",
              "| " + " | ".join(t for _, t in NAME_COLS) + " |", "|" + "---|" * len(NAME_COLS)]
    lines += ["| " + " | ".join(_cell(p.get(k)) for k, _ in NAME_COLS) + " |"
              for p in last.get("positions") or []]
    defs = last.get("definitions") or {}
    if defs:
        lines += ["", "## Definitions (from the newest row)", ""]
        lines += [f"- **{k}:** {v}" for k, v in defs.items()]
    return "\n".join(lines) + "\n"


def _fetch(since: date, env: str) -> list[dict]:
    import db
    start = datetime.combine(since, time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT extra_json FROM activity_feed WHERE stage = 'book_gap_risk' "
                        "AND ts >= %s ORDER BY ts", (start,))
            rows = [e for (e,) in cur.fetchall() if isinstance(e, dict) and e.get("env") == env]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return rows


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=DEFAULT_SINCE.isoformat())
    ap.add_argument("--env", default="live")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    since = date.fromisoformat(a.since)
    text = readout(_fetch(since, a.env), since, a.env, _git_head())
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
