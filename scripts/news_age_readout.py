"""News age readout: how old an event item was when it was screened and when
its analyst verdict came, in hours (audit 2026-11-27 08:53 #7, feed lag).
Screened-in items that reached a verdict only. Read-only engineering
readout; counts, hours and ids only, no headline or url text (the url's
host is the only part read).

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/news_age_readout.py [--since 2026-11-09] [--until 2026-11-28] \
        [--env live] [--out FILE] [--json FILE]

Where the times come from:
  - published_at: signals.provenance.trigger.published_at on the verdict
    (analyst.trigger_record: the event's "timestamp", else its
    "published_at"; decision.event_burst.published_at is the same value).
  - verdict time: the verdict row's signals.timestamp.
  - screen time: the SCREENED_IN row (crew) with the verdict's symbol and
    decision.url equal to the trigger url, the latest one at or before the
    verdict and within 7 days of it, matched in the SQL. Screen rows carry
    the url from 2026-11-09 (the default --since), so earlier verdicts
    mostly have no screen match.
Verdicts: signals rows with decision.model_output and trigger path "event",
timestamp in [--since, --until) as New York dates, in --env or with a NULL
env.

Not measurable from the database (notes in the output): no screen row
stores published_at (SCREENED_IN or SCREENED_OUT), and news_log.published_at
is the time the headline was logged (DEFAULT NOW()), not its publication.
So screened-out items, and screened-in items that never reached a verdict,
have no age here; the relevant / not-relevant split isn't possible.

Ages: screen or verdict time minus published_at, in hours. published_at is
"none" when absent and "unparseable" when it isn't an ISO 8601 or RFC 2822
time with a zone (a time without a zone, or a date alone, is unparseable:
its zone would be a guess). Per cell (all, day, feed, ticker, held, day x
ticker) and per age (screen, verdict): n, median, p90 (statistics.quantiles,
inclusive), max and min, in hours, over every parsed age; the fixed buckets
negative (< 0), < 15 min, 15-60 min, 1-4 h, 4-24 h, > 24 h (each bucket
includes its lower edge); published_at none / unparseable; and published_at
more than 5 minutes after the screen (or the verdict, for the verdict age).
The day is the verdict's New York date; the feed is the trigger source and
the url host; held is provenance.plan.held, else decision.held_review.

Screen rows: verdicts with no matched SCREENED_IN row, by whether the
verdict has a url; and SCREENED_IN rows in the window (with a url, and
without one) matched by no verdict within 7 days after them.

No age guard, cutoff or label: those are sample rules and Luke's.

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
import statistics
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
QUIET_START, QUIET_END = time(4, 50), time(5, 12)   # no DB reads at the stack's start
DEFAULT_SINCE = "2026-11-09"
FUTURE_MIN = 5
# (label, lower edge in hours, inclusive); the first bucket is everything below 0.
BUCKETS = (("negative", None), ("<15 min", 0.0), ("15-60 min", 0.25), ("1-4 h", 1.0),
           ("4-24 h", 4.0), (">24 h", 24.0))
AGES = ("screen", "verdict")

_VERDICTS_SQL = """
WITH v AS (
    SELECT s.id, s.timestamp, s.symbol,
           s.provenance->'trigger'->>'source' AS source,
           s.provenance->'trigger'->>'published_at' AS published_at,
           s.provenance->'trigger'->>'url' AS url,
           s.provenance->'plan'->>'held' AS plan_held,
           s.decision->>'held_review' AS held_review
    FROM signals s
    WHERE s.decision ? 'model_output' AND s.provenance->'trigger'->>'path' = 'event'
      AND s.timestamp >= %(start)s AND s.timestamp < %(end)s
      AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
), sc AS (
    SELECT x.id, x.timestamp, x.symbol, x.decision->>'url' AS url
    FROM signals x
    WHERE x.signal = 'SCREENED_IN' AND x.decision->>'url' IS NOT NULL
      AND x.timestamp >= %(start)s - interval '7 days' AND x.timestamp < %(end)s
      AND (x.trading_env = %(env)s OR x.trading_env IS NULL)
), m AS (
    SELECT DISTINCT ON (v.id) v.id, sc.id AS screen_id, sc.timestamp AS screened_at
    FROM v JOIN sc ON sc.symbol = v.symbol AND sc.url = v.url
                  AND sc.timestamp <= v.timestamp AND sc.timestamp > v.timestamp - interval '7 days'
    ORDER BY v.id, sc.timestamp DESC
)
SELECT v.id, v.timestamp, v.symbol, v.source,
       substring(v.url from '^[A-Za-z][A-Za-z0-9+.-]*://([^/?#:@]+)'),
       v.url IS NOT NULL, v.published_at, v.plan_held, v.held_review,
       m.screen_id, m.screened_at
FROM v LEFT JOIN m ON m.id = v.id
ORDER BY v.timestamp, v.id
"""

_SCREENS_SQL = """
WITH v AS (
    SELECT s.timestamp, s.symbol, s.provenance->'trigger'->>'url' AS url
    FROM signals s
    WHERE s.decision ? 'model_output' AND s.provenance->'trigger'->>'path' = 'event'
      AND s.timestamp >= %(start)s AND s.timestamp < %(end)s + interval '7 days'
      AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
      AND s.provenance->'trigger'->>'url' IS NOT NULL
), sc AS (
    SELECT x.timestamp, x.symbol, x.decision->>'url' AS url
    FROM signals x
    WHERE x.signal = 'SCREENED_IN'
      AND x.timestamp >= %(start)s AND x.timestamp < %(end)s
      AND (x.trading_env = %(env)s OR x.trading_env IS NULL)
)
SELECT count(*),
       count(*) FILTER (WHERE sc.url IS NULL),
       count(*) FILTER (WHERE sc.url IS NOT NULL AND NOT EXISTS (
           SELECT 1 FROM v WHERE v.symbol = sc.symbol AND v.url = sc.url
             AND v.timestamp >= sc.timestamp AND v.timestamp < sc.timestamp + interval '7 days'))
FROM sc
"""


def db_root(cwd: str | None = None) -> str:
    """The directory to import db from: the working directory when it holds
    db.py (the live clone, whose .env db loads), else this script's repo."""
    cwd = cwd or os.getcwd()
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET_START <= t < QUIET_END


def _aware(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def parse_published(v) -> datetime | str:
    """published_at as an aware datetime, or "none" / "unparseable". ISO 8601
    or RFC 2822 with a zone; a time without a zone or a date alone is
    unparseable. Pure."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return "none"
    s = str(v).strip()
    dt = None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError, IndexError):
            return "unparseable"
    if not isinstance(dt, datetime) or dt.tzinfo is None:
        return "unparseable"
    return dt


def held_label(plan_held, held_review) -> str:
    for v in (plan_held, held_review):
        if isinstance(v, bool):
            return "held" if v else "not held"
        if str(v).lower() in ("true", "false"):
            return "held" if str(v).lower() == "true" else "not held"
    return "unknown"


def _hours(later: datetime | None, pub) -> float | None:
    if later is None or not isinstance(pub, datetime):
        return None
    return round((later - pub).total_seconds() / 3600, 4)


def classify(row: tuple) -> dict:
    """One verdict row (the _VERDICTS_SQL columns) as a record of ids, labels
    and hours. Pure."""
    sid, ts, sym, source, host, has_url, pub_raw, plan_held, held_review, screen_id, screened = row
    ts, screened = _aware(ts), _aware(screened)
    pub = parse_published(pub_raw)
    return {"id": sid, "day": ts.astimezone(NY).date().isoformat(), "ticker": sym,
            "feed": f"{source or 'unknown'} {host or ('no url' if not has_url else 'no host')}",
            "has_url": bool(has_url), "held": held_label(plan_held, held_review),
            "published": pub if isinstance(pub, str) else "ok",
            "screen_id": screen_id,
            "screen_h": _hours(screened, pub), "verdict_h": _hours(ts, pub)}


def bucket(h: float) -> str:
    label = BUCKETS[0][0]
    for name, lo in BUCKETS[1:]:
        if h >= lo:
            label = name
    return label if h >= 0 else BUCKETS[0][0]


def _dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "median": None, "p90": None, "max": None, "min": None}
    p90 = statistics.quantiles(xs, n=10, method="inclusive")[8] if len(xs) > 1 else xs[0]
    return {"n": len(xs), "median": round(statistics.median(xs), 2), "p90": round(p90, 2),
            "max": round(max(xs), 2), "min": round(min(xs), 2)}


def tally(rows: list[dict]) -> dict:
    out = {"verdicts": len(rows),
           "published": {k: sum(1 for r in rows if r["published"] == k) for k in ("none", "unparseable")},
           "screen_matched": sum(1 for r in rows if r["screen_id"] is not None)}
    for age in AGES:
        xs = [r[f"{age}_h"] for r in rows if r[f"{age}_h"] is not None]
        out[age] = {**_dist(xs),
                    "buckets": {b: sum(1 for x in xs if bucket(x) == b) for b, _ in BUCKETS},
                    "future_over_5min": sum(1 for x in xs if x < -FUTURE_MIN / 60)}
    return out


def build(rows: list[dict]) -> dict:
    by = lambda key: {k: tally([r for r in rows if key(r) == k]) for k in sorted({key(r) for r in rows})}
    unmatched = [r for r in rows if r["screen_id"] is None]
    return {"all": tally(rows),
            "by_day": by(lambda r: r["day"]), "by_feed": by(lambda r: r["feed"]),
            "by_ticker": by(lambda r: r["ticker"]), "by_held": by(lambda r: r["held"]),
            "by_day_ticker": by(lambda r: (r["day"], r["ticker"])),
            "unmatched": {"with_url": sum(1 for r in unmatched if r["has_url"]),
                          "no_url": sum(1 for r in unmatched if not r["has_url"])}}


def _h(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _table(title: str, keycols: list[str], cells: dict, age: str) -> list[str]:
    head = keycols + ["verdicts", "n", "median h", "p90 h", "max h", "min h"] + \
        [b for b, _ in BUCKETS] + ["pub none", "pub unparseable", "pub > 5 min after"]
    lines = ["", f"### {title}", "", "| " + " | ".join(head) + " |",
             "|" + "---|" * len(keycols) + "---:|" * (len(head) - len(keycols))]
    for k, t in cells.items():
        a = t[age]
        keys = list(k) if isinstance(k, tuple) else [k]
        lines.append("| " + " | ".join(
            keys + [str(t["verdicts"]), str(a["n"]), _h(a["median"]), _h(a["p90"]), _h(a["max"]),
                    _h(a["min"])]
            + [str(a["buckets"][b]) for b, _ in BUCKETS]
            + [str(t["published"]["none"]), str(t["published"]["unparseable"]),
               str(a["future_over_5min"])]) + " |")
    if not cells:
        lines.append("| none |" + " |" * (len(head) - 1))
    return lines


def render(out: dict, screens: tuple, env: str, since: date, until: date, head: str,
           now: datetime) -> str:
    t, u = out["all"], out["unmatched"]
    n_sc, n_sc_nourl, n_sc_unmatched = screens
    lines = [f"# News age readout ({env}, event verdicts {since.isoformat()} to before "
             f"{until.isoformat()} ET): screened-in items that reached a verdict only", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             "Engineering readout; counts, hours and ids only. Age = screen or verdict time minus "
             "provenance.trigger.published_at, in hours. Screen time = the latest SCREENED_IN row "
             "with the verdict's symbol and url, at or before the verdict and within 7 days. "
             "Buckets include their lower edge; 'negative' is an age below 0. No age guard, cutoff "
             "or label: those are Luke's.", "",
             "Notes:",
             "- Not measurable from the database: no screen row (SCREENED_IN or SCREENED_OUT) "
             "stores published_at, and news_log.published_at is the time the headline was logged, "
             "not its publication. Screened-out items, and screened-in items with no verdict, have "
             "no age here, so there is no relevant / not-relevant split.",
             "- Screen rows carry the url from 2026-11-09; verdicts before that mostly have no "
             "screen match.",
             "- published_at is the event's 'timestamp', else its 'published_at' "
             "(analyst.trigger_record); decision.event_burst.published_at is the same value.", "",
             f"- event verdicts: {t['verdicts']}; published_at none: {t['published']['none']}, "
             f"unparseable: {t['published']['unparseable']}; with a matched screen row: "
             f"{t['screen_matched']}",
             f"- verdicts with no matched SCREENED_IN row: {u['with_url'] + u['no_url']} "
             f"({u['with_url']} with a url, {u['no_url']} without one)",
             f"- SCREENED_IN rows in the window: {n_sc}; without a url: {n_sc_nourl}; with a url and "
             f"no verdict on it within 7 days after: {n_sc_unmatched}"]
    for age, what in (("screen", "Age at the screen"), ("verdict", "Age at the verdict")):
        a = t[age]
        lines += ["", f"## {what}", "",
                  f"- n {a['n']}; median {_h(a['median'])} h, p90 {_h(a['p90'])} h, max "
                  f"{_h(a['max'])} h, min {_h(a['min'])} h; "
                  + ", ".join(f"{b} {a['buckets'][b]}" for b, _ in BUCKETS)
                  + f"; published more than {FUTURE_MIN} min after the {age}: {a['future_over_5min']}"]
        lines += _table("By day", ["day"], out["by_day"], age)
        lines += _table("By feed (trigger source and url host)", ["feed"], out["by_feed"], age)
        lines += _table("By ticker", ["ticker"], out["by_ticker"], age)
        lines += _table("By held", ["held"], out["by_held"], age)
        lines += _table("By day and ticker", ["day", "ticker"], out["by_day_ticker"], age)
    return "\n".join(lines) + "\n"


def code_version() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-verdict records (ids, labels, hours) here")
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
        since = date.fromisoformat(args.since)
        until = (date.fromisoformat(args.until) if args.until
                 else now.astimezone(NY).date() + timedelta(days=1))
    except ValueError:
        print("--since and --until must be YYYY-MM-DD.", file=sys.stderr)
        return 2
    if until <= since:
        print("--until must be after --since.", file=sys.stderr)
        return 2
    params = {"env": args.env, "start": datetime.combine(since, time(0), tzinfo=NY),
              "end": datetime.combine(until, time(0), tzinfo=NY)}

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
            cur.execute(_VERDICTS_SQL, params)
            raw = cur.fetchall()
            cur.execute(_SCREENS_SQL, params)
            screens = tuple(int(x or 0) for x in cur.fetchone())
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = [classify(r) for r in raw]
    out = build(rows)
    head = code_version()
    text = render(out, screens, args.env, since, until, head, now)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head, "env": args.env,
                       "since": since.isoformat(), "until": until.isoformat(),
                       "screen_rows": dict(zip(("n", "no_url", "url_no_verdict"), screens)),
                       "verdicts": rows}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
