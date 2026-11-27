"""Catalyst-claim window readout (audit 2026-11-26-1453 #2). Engineering
readout, record only: counts by ticker and by catalyst date. Nothing in the
bot reads it.

Two sections, each over its own window of ET verdict days (both inclusive).
Verdicts are full analyst verdicts (decision ? 'model_output', not a
db._NON_ANALYST_SIGNALS row) of one trading env:

  claims      --claim-since .. --claim-until (default 2026-11-24 ..
              2026-11-30). Per ticker, from the stored record
              (decision.catalyst_check, earnings_claim.py, as written at
              verdict time): verdicts, earnings_claim_unmatched and
              matches_sector_cluster_date (True / False / None), both True,
              and the catalyst dates named. A row without a key counts as
              "not recorded". The fields are read, never recomputed, so a run
              after the window closes gives the same counts (freeze it with
              --out).

  migration   --migration-since .. --migration-until (default 2026-12-01 ..
              2026-12-04). Each verdict's catalyst date (model_output
              .catalyst.date) sorted by what the catalyst calendar
              (catalyst_calendar.db, opened mode=ro, rows created at or
              before the verdict) holds on that date, first that applies:
                no_day_date        month-only, "ongoing", missing or not a date
                past_date          before the verdict's ET day
                own_earnings_line  a ticker earnings row of the verdict's ticker
                sector_window      a sector row of the verdict's sector ETF
                                   (the record's earnings_claim.sector_etf)
                                   whose window (window_start .. window_end,
                                   else event_date) holds the date
                macro_line         a macro row
                own_other_line     a ticker row of another kind
                invented_earnings  no row, and the catalyst text names
                                   earnings (the record's names_earnings,
                                   else earnings_claim._EARNINGS_RE)
                unmatched_date     no row, and the text names no earnings
              A row "holds the date" when it is within --tol-days (default 0,
              the same day) of it. Every row kind found on the date is listed
              (lines), and names_earnings is tallied per category, so another
              order or band can be reapplied from --json. Superseded rows
              count (the brief may have printed them when the verdict ran);
              own_earnings_rows gives how many un-superseded own earnings
              rows the ticker had then.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/catalyst_claim_readout.py [--env live] [--section claims|migration|both] \\
        [--claim-since D] [--claim-until D] [--migration-since D] \\
        [--migration-until D] [--tol-days N] [--calendar PATH] [--json] [--out FILE]

Read-only: SUPABASE_PORT must be 6543 (the transaction pooler); one
transaction, SET TRANSACTION READ ONLY with a statement timeout, rolled back.
Counts, dates and tickers only. Descriptive: which claims count as anchored,
and any rule built on them, are Luke's. Not part of weekly_readouts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 30000
CLAIM_WINDOW = ("2026-11-24", "2026-11-30")
MIGRATION_WINDOW = ("2026-12-01", "2026-12-04")
NOT_RECORDED = "not recorded"
FLAGS = ("earnings_claim_unmatched", "matches_sector_cluster_date")
CATEGORIES = ("no_day_date", "past_date", "own_earnings_line", "sector_window", "macro_line",
              "own_other_line", "invented_earnings", "unmatched_date")

VERDICTS_SQL = ("SELECT id, symbol, signal, timestamp, decision->'catalyst_check', "
                "decision->'model_output'->'catalyst' FROM signals "
                "WHERE timestamp >= %(lo)s AND timestamp < %(hi)s AND trading_env = %(env)s "
                "AND decision ? 'model_output' AND NOT (signal = ANY(%(non)s)) ORDER BY timestamp, id")
CALENDAR_SQL = ("SELECT scope, ticker, event_type, event_date, window_start, window_end, "
                "confidence, created_at, superseded_by FROM catalyst_events")


def _aware(ts) -> datetime | None:
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(ts, datetime):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _et_day(ts) -> date | None:
    t = _aware(ts)
    return t.astimezone(NY).date() if t else None


def _as_date(v) -> date | None:
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date() if v else None
    except ValueError:
        return None


def _day(raw) -> tuple[date | None, str | None]:
    """(the catalyst date when it is a day, else None; why not), as
    earnings_claim._day reads it."""
    s = str(raw or "").strip()
    if not s:
        return None, "no catalyst date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        d = _as_date(s)
        return (d, None) if d else (None, "unparseable catalyst date")
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return None, "month-only catalyst date"
    if s.lower() == "ongoing":
        return None, "ongoing catalyst"
    return None, "unparseable catalyst date"


def _bounds(since: str, until: str) -> tuple[datetime, datetime]:
    """UTC bounds for ET days since..until, both inclusive."""
    lo = datetime.combine(date.fromisoformat(since), time.min, tzinfo=NY)
    hi = datetime.combine(date.fromisoformat(until) + timedelta(days=1), time.min, tzinfo=NY)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def _in_window(r: dict, since: str, until: str) -> bool:
    d = _et_day(r.get("ts"))
    return d is not None and date.fromisoformat(since) <= d <= date.fromisoformat(until)


def _flag(cc, k):
    if not isinstance(cc, dict) or k not in cc:
        return NOT_RECORDED
    v = cc[k]
    return v if v is True or v is False or v is None else NOT_RECORDED   # 1 == True, so identity


def _tf(c: Counter) -> dict:
    return {"true": c[True], "false": c[False], "none": c[None], NOT_RECORDED: c[NOT_RECORDED]}


def claims(rows: list[dict]) -> dict:
    """Per-ticker counts of the stored flags. rows: {ticker, catalyst_check,
    catalyst}."""
    per = defaultdict(lambda: {"verdicts": 0, **{k: Counter() for k in FLAGS},
                               "both_true": 0, "dates": Counter()})
    tot = {"verdicts": 0, **{k: Counter() for k in FLAGS}, "both_true": 0, "dates": Counter()}
    for r in rows:
        cc = r.get("catalyst_check")
        vals = {k: _flag(cc, k) for k in FLAGS}
        raw = (r.get("catalyst") or {}).get("date") if isinstance(r.get("catalyst"), dict) else None
        raw = raw if raw not in (None, "") else (cc or {}).get("model_date") if isinstance(cc, dict) else raw
        for acc in (per[str(r.get("ticker") or "?").upper()], tot):
            acc["verdicts"] += 1
            for k in FLAGS:
                acc[k][vals[k]] += 1
            acc["both_true"] += all(vals[k] is True for k in FLAGS)
            acc["dates"][str(raw or "none")] += 1

    def out(a):
        return {"verdicts": a["verdicts"], **{k: _tf(a[k]) for k in FLAGS},
                "both_true": a["both_true"],
                "dates": dict(sorted(a["dates"].items(), key=lambda kv: (-kv[1], kv[0])))}
    return {"tickers": {t: out(a) for t, a in sorted(per.items())}, "total": out(tot)}


def _names_earnings(cc, text) -> bool:
    ec = (cc or {}).get("earnings_claim") if isinstance(cc, dict) else None
    if isinstance(ec, dict) and isinstance(ec.get("names_earnings"), bool):
        return ec["names_earnings"]
    from earnings_claim import _EARNINGS_RE
    return bool(_EARNINGS_RE.search(text if isinstance(text, str) else ""))


def _sector_etf(cc):
    ec = (cc or {}).get("earnings_claim") if isinstance(cc, dict) else None
    return ec.get("sector_etf") if isinstance(ec, dict) else None


def lines_on(d: date, ticker: str, etf, at: datetime, calendar: list[dict], tol: int) -> list[dict]:
    """The calendar rows created at or before `at` that hold day d for this
    verdict (module docstring), as {kind, event_type, date, days_off,
    confidence, superseded}."""
    out = []
    for e in calendar:
        created = _aware(e.get("created_at"))
        if created is None or created > at:
            continue
        ev = _as_date(e.get("event_date"))
        lo = _as_date(e.get("window_start")) or ev
        hi = _as_date(e.get("window_end")) or ev
        if not (lo and hi):
            continue
        off = 0 if lo <= d <= hi else (lo - d).days if d < lo else (hi - d).days
        if abs(off) > tol:
            continue
        scope, kind = e.get("scope"), None
        if scope == "ticker" and str(e.get("ticker") or "").upper() == ticker:
            kind = "own_earnings_line" if e.get("event_type") == "earnings" else "own_other_line"
        elif scope == "sector" and etf and e.get("ticker") == etf:
            kind = "sector_window"
        elif scope == "macro":
            kind = "macro_line"
        if kind:
            out.append({"kind": kind, "event_type": e.get("event_type"),
                        "date": ev.isoformat() if ev else None, "days_off": off,
                        "confidence": e.get("confidence"), "superseded": bool(e.get("superseded_by"))})
    return out


def classify(r: dict, calendar: list[dict], tol: int = 0) -> dict:
    """One verdict's migration record (module docstring)."""
    cat = r.get("catalyst") if isinstance(r.get("catalyst"), dict) else {}
    cc = r.get("catalyst_check")
    ticker = str(r.get("ticker") or "?").upper()
    at = _aware(r.get("ts"))
    vday = _et_day(at)
    raw = cat.get("date")
    d, why = _day(raw)
    names = _names_earnings(cc, cat.get("event"))
    rec = {"id": r.get("id"), "ticker": ticker, "verdict_day": vday.isoformat() if vday else None,
           "catalyst_date": str(raw) if raw not in (None, "") else None,
           "catalyst_event": (cat.get("event") or "")[:120] if isinstance(cat.get("event"), str) else None,
           "names_earnings": names, "category": None, "why_no_day": why, "lines": [],
           "own_earnings_rows": None}
    if d is None or at is None:
        rec["category"] = "no_day_date"
        return rec
    lines = lines_on(d, ticker, _sector_etf(cc), at, calendar, tol)
    rec["lines"] = lines
    rec["own_earnings_rows"] = sum(
        1 for e in calendar
        if e.get("scope") == "ticker" and str(e.get("ticker") or "").upper() == ticker
        and e.get("event_type") == "earnings" and not e.get("superseded_by")
        and (_aware(e.get("created_at")) or at) <= at)
    kinds = {x["kind"] for x in lines}
    if d < vday:
        rec["category"] = "past_date"
    else:
        rec["category"] = next((k for k in CATEGORIES[2:6] if k in kinds), None) or (
            "invented_earnings" if names else "unmatched_date")
    return rec


def migration(rows: list[dict], calendar: list[dict], tol: int = 0) -> dict:
    recs = [classify(r, calendar, tol) for r in rows]
    by_date = defaultdict(lambda: {"verdicts": 0, "category": Counter(), "tickers": Counter(),
                                   "line_kinds": set()})
    by_cat = {c: {"verdicts": 0, "names_earnings": 0, "tickers": Counter()} for c in CATEGORIES}
    for x in recs:
        b = by_date[x["catalyst_date"] or "none"]
        b["verdicts"] += 1
        b["category"][x["category"]] += 1
        b["tickers"][x["ticker"]] += 1
        b["line_kinds"].update(f"{ln['kind']}:{ln['event_type']}" for ln in x["lines"])
        c = by_cat[x["category"]]
        c["verdicts"] += 1
        c["names_earnings"] += bool(x["names_earnings"])
        c["tickers"][x["ticker"]] += 1

    def counts(c):
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"verdicts": len(recs), "tol_days": tol,
            "by_category": {k: {**v, "tickers": counts(v["tickers"])} for k, v in by_cat.items()},
            "by_catalyst_date": {k: {"verdicts": v["verdicts"], "category": counts(v["category"]),
                                     "tickers": counts(v["tickers"]),
                                     "line_kinds": sorted(v["line_kinds"])}
                                 for k, v in sorted(by_date.items())},
            "rows": recs}


def _v(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k} {v[k]}" for k in v) or "none"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) or "none"
    return "n/a" if v is None else str(v)


def render(res: dict) -> str:
    L = [f"# Catalyst-claim readout ({res['env']})", ""]
    if "claims" in res:
        c, (s, u) = res["claims"], res["claim_window"]
        L += [f"## Claims, {s} to {u} ET (stored decision.catalyst_check)", "",
              "| ticker | verdicts | earnings_claim_unmatched | matches_sector_cluster_date "
              "| both true | catalyst dates |", "|---|---|---|---|---|---|"]
        for t, a in list(c["tickers"].items()) + [("all", c["total"])]:
            L.append(f"| {t} | {a['verdicts']} | {_v(a['earnings_claim_unmatched'])} | "
                     f"{_v(a['matches_sector_cluster_date'])} | {a['both_true']} | {_v(a['dates'])} |")
        L.append("")
    if "migration" in res:
        m, (s, u) = res["migration"], res["migration_window"]
        L += [f"## Where the claim goes, {s} to {u} ET (catalyst_calendar.db, tol {m['tol_days']} d)",
              "", f"Verdicts: {m['verdicts']}", "",
              "| category | verdicts | names earnings | tickers |", "|---|---|---|---|"]
        for k, v in m["by_category"].items():
            L.append(f"| {k} | {v['verdicts']} | {v['names_earnings']} | {_v(v['tickers'])} |")
        L += ["", "| catalyst date | verdicts | category | calendar rows on the date | tickers |",
              "|---|---|---|---|---|"]
        for k, v in m["by_catalyst_date"].items():
            L.append(f"| {k} | {v['verdicts']} | {_v(v['category'])} | {_v(v['line_kinds'])} | "
                     f"{_v(v['tickers'])} |")
        L.append("")
    return "\n".join(L)


def fetch_verdicts(conn, env: str, since: str, until: str, non_analyst) -> list[dict]:
    """Full analyst verdicts in ET days since..until, in one read-only
    transaction, rolled back."""
    lo, hi = _bounds(since, until)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(VERDICTS_SQL, {"lo": lo, "hi": hi, "env": env, "non": list(non_analyst)})
            return [{"id": i, "ticker": t, "signal": s, "ts": ts, "catalyst_check": cc, "catalyst": cat}
                    for i, t, s, ts, cc, cat in cur.fetchall()]
    finally:
        conn.rollback()


def read_calendar(path) -> list[dict]:
    """Every catalyst_events row, from the store opened read-only (never
    created)."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    try:
        cols = ("scope", "ticker", "event_type", "event_date", "window_start", "window_end",
                "confidence", "created_at", "superseded_by")
        return [dict(zip(cols, r)) for r in conn.execute(CALENDAR_SQL).fetchall()]
    finally:
        conn.close()


def run(conn, env: str, section: str, claim_window, migration_window, tol: int,
        calendar_path, non_analyst) -> dict:
    res = {"env": env}
    windows = []
    if section in ("claims", "both"):
        windows.append(claim_window)
    if section in ("migration", "both"):
        windows.append(migration_window)
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    rows = fetch_verdicts(conn, env, lo, hi, non_analyst)
    if section in ("claims", "both"):
        res["claim_window"] = list(claim_window)
        res["claims"] = claims([r for r in rows if _in_window(r, *claim_window)])
    if section in ("migration", "both"):
        res["migration_window"] = list(migration_window)
        res["migration"] = migration([r for r in rows if _in_window(r, *migration_window)],
                                     read_calendar(calendar_path), tol)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="catalyst-claim window readout")
    ap.add_argument("--env", help="trading env (default config.TRADING_ENV)")
    ap.add_argument("--section", choices=("claims", "migration", "both"), default="both")
    ap.add_argument("--claim-since", default=CLAIM_WINDOW[0])
    ap.add_argument("--claim-until", default=CLAIM_WINDOW[1])
    ap.add_argument("--migration-since", default=MIGRATION_WINDOW[0])
    ap.add_argument("--migration-until", default=MIGRATION_WINDOW[1])
    ap.add_argument("--tol-days", type=int, default=0)
    ap.add_argument("--calendar", help="catalyst_calendar.db (default catalyst_service.storage.DB_PATH)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    for v in (args.claim_since, args.claim_until, args.migration_since, args.migration_until):
        date.fromisoformat(v)
    path = args.calendar
    if path is None:
        from catalyst_service import storage
        path = storage.DB_PATH
    import db
    if args.env:
        env = args.env
    else:
        import config
        env = config.TRADING_ENV
    conn = db._get_conn()
    try:
        res = run(conn, env, args.section, (args.claim_since, args.claim_until),
                  (args.migration_since, args.migration_until), args.tol_days, path,
                  getattr(db, "_NON_ANALYST_SIGNALS", ()))
    finally:
        db._put_conn(conn)
    text = json.dumps(res, default=str, indent=1) + "\n" if args.json else render(res)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
