"""Price-age readout: for each analyst verdict, whether the brief's price is
older than the item that triggered it, and by how much (audit 2026-11-26
16:53 #2; eng-candidates-2026-11-26 #4). Read-only engineering readout;
counts and percentages only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/price_age_readout.py [--since 2026-11-24] [--until 2026-11-27] \
        [--env live] [--out FILE]

Verdicts: signals rows with decision.model_output (the analyst's verdicts),
timestamp in [--since, --until) as New York dates, in --env or with a NULL
env. Each verdict is counted once; its day is the New York date of
signals.timestamp.

Times, both as stored on provenance.trigger (the keys exist from the 11-24
05:00 load; brief_price_as_of from 11-24 10:19):
  - brief_price_as_of: the time of the price the brief showed;
  - event_time: the triggering item's own time.
They are parsed as crew._parse_when does (ISO, or RFC 2822; a naive time is
UTC). A bare YYYY-MM-DD event_time (a filing date) is a day, not a time, so
it is counted as "bare date" and gets no gap, the same rule as 1135 row 17's
price_predates_event_h. The gap is event_time minus brief_price_as_of, in
minutes: positive means the price is older than the item.

Columns, per day, per ticker, per path and per held x session cell:
  - verdicts: all verdicts in the cell;
  - timed: both times present and parsed, event_time not a bare date;
  - older: timed with gap > 0, and its % of timed;
  - not older: timed with gap <= 0;
  - no price time / no item time / bare date / unparsed: why a verdict is
    not timed (no price time wins when both are missing);
  - gap minutes over the timed rows, and over the older rows: n, min, p25,
    median, p75, max (p25/p75: statistics.quantiles, inclusive method).

Held: provenance.plan.held as plans.observe recorded it (a stop row
existed): held, not held, or unknown when the verdict has no plan record.
Session: whether provenance.trigger.brief_built_at (else signals.timestamp)
fell in regular hours by market_calendar: a weekday that is not an NYSE
holiday, 09:30 ET up to 16:00 ET, or up to 13:00 ET on an early-close day.
Otherwise "outside RTH".

No threshold, no exclusion and no "stale" label: those are Luke's (decision
packet items 6a-c). Percent and counts only; no price is read or printed.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db, market_calendar and the standard library.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
QUIET_START, QUIET_END = time(4, 50), time(5, 12)   # no DB reads at the stack's start
DEFAULT_SINCE = "2026-11-24"
REASONS = ("no price time", "no item time", "bare date", "unparsed")
_BARE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol,
       s.provenance->'trigger'->>'path',
       s.provenance->'trigger'->>'brief_price_as_of',
       s.provenance->'trigger'->>'event_time',
       s.provenance->'trigger'->>'brief_built_at',
       s.provenance->'plan'->>'held'
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


def parse_when(value):
    """An ISO or RFC 2822 time string as an aware datetime (naive = UTC),
    else None. The same rule as agents/crew.py _parse_when."""
    if not value:
        return None
    s = str(value).strip()
    for parse in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")),
                  parsedate_to_datetime):
        try:
            ts = parse(s)
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def gap_minutes(as_of, event_time):
    """(gap in minutes, None) when both times parse, else (None, reason).
    The gap is event_time minus as_of: positive when the price is older."""
    if not as_of:
        return None, "no price time"
    if not event_time:
        return None, "no item time"
    if _BARE_DATE.fullmatch(str(event_time).strip()):
        return None, "bare date"
    a, e = parse_when(as_of), parse_when(event_time)
    if a is None or e is None:
        return None, "unparsed"
    return round((e - a).total_seconds() / 60, 1), None


def in_rth(when, holidays=None, early_closes=None) -> bool:
    """True when the aware time falls in NYSE regular hours by
    market_calendar: a weekday that is not a holiday, 09:30 ET up to 16:00
    ET (13:00 on an early-close day)."""
    if holidays is None or early_closes is None:
        import market_calendar as mc
        holidays = mc.NYSE_HOLIDAYS if holidays is None else holidays
        early_closes = mc.NYSE_EARLY_CLOSES if early_closes is None else early_closes
    et = when.astimezone(NY)
    d = et.date()
    if et.weekday() >= 5 or d in holidays:
        return False
    close = time(13, 0) if d in early_closes else time(16, 0)
    return time(9, 30) <= et.time() < close


def held_label(v) -> str:
    return {"true": "held", "false": "not held"}.get(str(v).lower() if v is not None else "", "unknown")


def classify(row, holidays=None, early_closes=None) -> dict:
    """One verdict row (as _VERDICTS_SQL returns it) as a dict with its day,
    gap and labels."""
    sid, ts, ticker, path, as_of, ev, built, held = row
    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    gap, reason = gap_minutes(as_of, ev)
    built_t = parse_when(built) or ts
    return {"id": sid, "day": ts.astimezone(NY).date().isoformat(), "ticker": ticker,
            "path": path or "none", "gap": gap, "reason": reason, "held": held_label(held),
            "session": "RTH" if in_rth(built_t, holidays, early_closes) else "outside RTH"}


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) >= 2 else [v[0], v[0], v[0]]
    return {"n": len(v), "min": v[0], "p25": round(q[0], 1), "median": round(statistics.median(v), 1),
            "p75": round(q[2], 1), "max": v[-1]}


def tally(rows: list[dict]) -> dict:
    """The column set for one cell."""
    timed = [r["gap"] for r in rows if r["gap"] is not None]
    older = [g for g in timed if g > 0]
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    return {"verdicts": len(rows), "timed": len(timed), "older": len(older),
            "not_older": len(timed) - len(older),
            "older_pct": round(100 * len(older) / len(timed), 1) if timed else None,
            "reasons": {k: reasons.get(k, 0) for k in REASONS},
            "gap_timed": _dist(timed), "gap_older": _dist(older)}


def build(rows: list[dict]) -> dict:
    def group(key):
        out = {}
        for r in rows:
            out.setdefault(key(r), []).append(r)
        return {k: tally(v) for k, v in sorted(out.items())}
    return {"total": tally(rows), "by_day": group(lambda r: r["day"]),
            "by_ticker": group(lambda r: r["ticker"]), "by_path": group(lambda r: r["path"]),
            "by_held_session": group(lambda r: f"{r['held']} / {r['session']}")}


def _c(v) -> str:
    return "-" if v is None else str(v)


def _table(title: str, label: str, cells: dict) -> list[str]:
    lines = ["", f"## {title}", "",
             f"| {label} | verdicts | timed | older | older % of timed | not older | "
             + " | ".join(REASONS) + " | gap min (timed): median [p25, p75], min..max | "
             "gap min (older): median [p25, p75], max |",
             "|---|" + "---|" * (7 + len(REASONS))]
    for k, t in cells.items():
        gt, go = t["gap_timed"], t["gap_older"]
        timed = (f"{gt['median']} [{gt['p25']}, {gt['p75']}], {gt['min']}..{gt['max']}"
                 if gt["n"] else "-")
        older = f"{go['median']} [{go['p25']}, {go['p75']}], {go['max']}" if go["n"] else "-"
        lines.append(f"| {k} | {t['verdicts']} | {t['timed']} | {t['older']} | {_c(t['older_pct'])} | "
                     f"{t['not_older']} | " + " | ".join(str(t["reasons"][r]) for r in REASONS)
                     + f" | {timed} | {older} |")
    return lines


def render(out: dict, env: str, since: date, until: date, head: str, now: datetime) -> str:
    t = out["total"]
    lines = [f"# Price-age readout ({env}, verdicts {since.isoformat()} to before {until.isoformat()} ET)", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             "Engineering readout; counts and percentages only. Gap = the item's time minus the "
             "brief price's time, in minutes (positive: the price is older than the item). "
             "No threshold, exclusion or stale label is applied (Luke's).", "",
             f"- verdicts: {t['verdicts']}; timed: {t['timed']}; older than the item: {t['older']}"
             + (f" ({t['older_pct']}% of timed)" if t["older_pct"] is not None else "")
             + f"; not older: {t['not_older']}",
             "- not timed: " + ", ".join(f"{r} {t['reasons'][r]}" for r in REASONS)]
    lines += _table("By day (New York date of the verdict)", "day", out["by_day"])
    lines += _table("Held x session (session of the brief build time, market_calendar)",
                    "held / session", out["by_held_session"])
    lines += _table("By trigger path", "path", out["by_path"])
    lines += _table("By ticker", "ticker", out["by_ticker"])
    return "\n".join(lines) + "\n"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
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
            cur.execute(_VERDICTS_SQL, {"env": args.env, "start": start, "end": end})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    out = build([classify(r) for r in raw])
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(out, args.env, since, until, head or "unknown", now)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
