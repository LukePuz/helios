"""Catalyst topic readout: how a ticker's plan catalyst moves across its
trade_plans revision chains (audit 2026-11-26 22:53 #1). Read-only
engineering readout; counts only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/catalyst_kind_readout.py [--days 30] [--env live] [--ticker NVDA] \\
        [--out FILE] [--export FILE.json]

Plans: trade_plans rows created in the last --days days, in --env or with a
NULL env. A chain is a plan and its revisions (replaces_id), walked back to
the oldest row read; a chain whose first row replaces a plan outside the
window is marked "starts before window". Rows in a chain are ordered by
revision, then created_at, then id.

Per plan, as stored: catalyst_event (text), catalyst_date, horizon_end,
created_at, revision, source. Derived:
  - topic_kind: the first of TOPIC_ORDER whose TOPIC_PATTERNS group matches
    catalyst_event (TOPIC_VERSION, TOPIC_SHA); "unlabelled" when none does;
    topic_kinds lists every group that matched. A reporting aid only: the
    word groups and their order are not a study definition (Luke's). Not
    earnings_claim.CATALYST_KINDS, which labels a verdict's catalyst date.
  - born_past: catalyst_date before the plan's created date (New York).
  - horizon_basis: "catalyst" when catalyst_date is on or after the created
    date, else "bucket_default" (a horizon fallback), the rule
    agents/plans.py _horizon_record states; horizon_matches says whether
    horizon_end is the catalyst date plus CATALYST_GRACE_DAYS ("catalyst")
    or not ("bucket_default", whose length depends on the bucket).

Per chain and per ticker: plans, topic-kind changes between consecutive rows
in revision order, distinct topic kinds, distinct catalyst texts (exact, and
case- and space-folded), distinct catalyst dates (and rows without one),
plans born past-dated, and horizon fallbacks. Counts only; no rule or
threshold is chosen. The markdown prints no catalyst text, price or level;
--export writes the rows with catalyst text, dollar figures masked as $X,
and no price or level column.

SUPABASE_PORT must be 6543 (the transaction pooler); not run 04:50-05:12
ET; one read-only transaction with a statement timeout, rolled back. db is
imported from the working directory when it holds a db.py (the live clone,
with its .env), else from beside this script; with no SUPABASE_HOST after
the import the readout stops before connecting. Imports: db and the
standard library only. Not loaded by weekly_readouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 30000
NY = ZoneInfo("America/New_York")
QUIET = (time(4, 50), time(5, 12))        # no DB reads in this New York window
CATALYST_GRACE_DAYS = 5                   # agents/plans.py _CATALYST_GRACE_DAYS
OPEN_LABEL = "a reporting aid, not a study definition (the word groups and their order are Luke's)"

TOPIC_VERSION = 1
TOPIC_ORDER = ("earnings", "governance", "regulatory", "product", "macro", "sector")
UNLABELLED = "unlabelled"
TOPIC_PATTERNS = (
    ("earnings", r"\b(?:earnings|quarterly|results|q[1-4]|fy\s?\d{2,4}|fiscal|guidance|eps|"
                 r"revenue)\b"),
    ("governance", r"\b(?:ceo|cfo|chief executive|board|leadership|management|succession|"
                   r"founder|activist|proxy|governance|executive|chair(?:man|woman)?)\b"),
    ("regulatory", r"\b(?:regulat\w*|export\w*|curbs?|bans?|antitrust|ftc|doj|sec|fda|"
                   r"licen[cs]\w*|tariffs?|sanctions?|probes?|investigation|lawsuits?|court|"
                   r"ruling|government|policy|subsid\w*|chips act|subpoenas?|legal)\b"),
    ("product", r"\b(?:launch\w*|products?|chips?|gpus?|platforms?|unveil\w*|ship\w*|roadmap|"
                r"models?|architecture|blackwell|rubin|contracts?|deals?|partnerships?|orders?|"
                r"customers?|design wins?|gtc|keynote|conference)\b"),
    ("macro", r"\b(?:fed|fomc|rates?|inflation|cpi|ppi|jobs report|payrolls?|gdp|recession|"
              r"yields?|treasur\w*|election|macro\w*|econom\w*)\b"),
    ("sector", r"\b(?:sector|industry|peers?|hyperscalers?|capex|ai spending|"
               r"semiconductor cycle|xlk|sox|smh|etfs?|index|rebalanc\w*|cluster)\b"),
)
_TOPIC = tuple((k, re.compile(p, re.I)) for k, p in TOPIC_PATTERNS)
TOPIC_SHA = hashlib.sha256(json.dumps([list(t) for t in TOPIC_PATTERNS],
                                      separators=(",", ":")).encode()).hexdigest()[:12]
_DOLLAR = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:[BMKT]\b|bn\b|mn\b|billion\b|million\b|"
                     r"trillion\b|thousand\b))?", re.I)

_SQL = """
SELECT id, ticker, status, source, revision, replaces_id, created_at, closed_at,
       catalyst_event, catalyst_date, horizon_end
FROM trade_plans
WHERE (trading_env = %(env)s OR trading_env IS NULL) AND created_at >= %(since)s
ORDER BY ticker, created_at, id
"""
_COLS = ("id", "ticker", "status", "source", "revision", "replaces_id", "created_at",
         "closed_at", "catalyst_event", "catalyst_date", "horizon_end")


def db_root(cwd: str | None = None) -> str:
    """The directory db is imported from: cwd when it holds a db.py, else
    this script's checkout."""
    cwd = os.getcwd() if cwd is None else cwd
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET[0] <= t < QUIET[1]


def topic_kinds(text: str | None) -> list[str]:
    """Every TOPIC_ORDER group whose pattern matches the text, in that order."""
    return [k for k, rx in _TOPIC if rx.search(text or "")]


def topic_kind(text: str | None) -> str:
    kinds = topic_kinds(text)
    return kinds[0] if kinds else UNLABELLED


def mask_dollars(text: str | None) -> str | None:
    return _DOLLAR.sub("$X", text) if text else text


def _fold(text: str | None) -> str | None:
    return " ".join(text.lower().split()) if text and text.strip() else None


def _day(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _created_day(t) -> date | None:
    if isinstance(t, datetime):
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(NY).date()
    return _day(t)


def plan_record(row: dict) -> dict:
    """The derived fields of one plan row. Pure; no price or level is read."""
    cdate, created, end = _day(row.get("catalyst_date")), _created_day(row.get("created_at")), \
        _day(row.get("horizon_end"))
    basis = "catalyst" if cdate and created and cdate >= created else "bucket_default"
    keyed = cdate + timedelta(days=CATALYST_GRACE_DAYS) if cdate else None
    if end is None:
        matches = None
    elif basis == "catalyst":
        matches = end == keyed
    else:
        matches = end != keyed      # a fallback horizon is not the catalyst's
    text = row.get("catalyst_event")
    return {"plan_id": row.get("id"), "ticker": row.get("ticker"), "status": row.get("status"),
            "source": row.get("source"), "revision": row.get("revision"),
            "replaces_id": row.get("replaces_id"),
            "created_et": (row["created_at"].astimezone(NY).strftime("%Y-%m-%d %H:%M")
                           if isinstance(row.get("created_at"), datetime) else None),
            "created_day": created.isoformat() if created else None,
            "catalyst_date": cdate.isoformat() if cdate else None,
            "horizon_end": end.isoformat() if end else None,
            "topic_kind": topic_kind(text), "topic_kinds": topic_kinds(text),
            "has_text": bool(text and text.strip()),
            "born_past": bool(cdate and created and cdate < created),
            "horizon_basis": basis, "horizon_fallback": basis == "bucket_default",
            "horizon_matches": matches,
            "_text": text}


def chains(recs: list[dict]) -> list[dict]:
    """Group records into replaces_id chains, each ordered by revision,
    created, id. Pure."""
    by_id = {r["plan_id"]: r for r in recs}

    def root(r):
        seen = set()
        while r.get("replaces_id") in by_id and r["plan_id"] not in seen:
            seen.add(r["plan_id"])
            r = by_id[r["replaces_id"]]
        return r

    groups: dict = {}
    for r in recs:
        groups.setdefault(root(r)["plan_id"], []).append(r)
    out = []
    for rid, rows in groups.items():
        rows.sort(key=lambda r: (r.get("revision") or 0, r.get("created_et") or "", r["plan_id"]))
        first = rows[0]
        out.append({"root_id": rid, "ticker": first["ticker"], "rows": rows,
                    "starts_before_window": first.get("replaces_id") is not None
                    and first.get("replaces_id") not in by_id})
    out.sort(key=lambda c: (c["ticker"] or "", c["rows"][0].get("created_et") or "", c["root_id"]))
    return out


def summarize(rows: list[dict]) -> dict:
    """Counts over rows in the order given (a chain in revision order, or a
    ticker's chains one after another). Pure."""
    kinds = [r["topic_kind"] for r in rows]
    texts = [r["_text"] for r in rows if r["has_text"]]
    dates = [r["catalyst_date"] for r in rows if r["catalyst_date"]]
    return {"plans": len(rows),
            "kind_changes": sum(a != b for a, b in zip(kinds, kinds[1:])),
            "distinct_kinds": len(set(kinds)),
            "distinct_texts": len(set(texts)),
            "distinct_texts_folded": len({_fold(t) for t in texts}),
            "without_text": len(rows) - len(texts),
            "distinct_dates": len(set(dates)), "without_date": len(rows) - len(dates),
            "born_past": sum(r["born_past"] for r in rows),
            "horizon_fallbacks": sum(r["horizon_fallback"] for r in rows),
            "horizon_mismatch": sum(r["horizon_matches"] is False for r in rows)}


def build(raw: list[tuple]) -> dict:
    recs = [plan_record(dict(zip(_COLS, r))) for r in raw]
    chs = chains(recs)
    tickers: dict = {}
    for c in chs:
        tickers.setdefault(c["ticker"], []).append(c)
    per_ticker = {}
    for t, cs in tickers.items():
        rows = [r for c in cs for r in c["rows"]]
        s = summarize(rows)
        # kind changes are counted within chains only, never across two chains
        s["kind_changes"] = sum(summarize(c["rows"])["kind_changes"] for c in cs)
        s["chains"] = len(cs)
        per_ticker[t] = s
    return {"records": recs, "chains": chs, "per_ticker": per_ticker}


_SUM_COLS = ("plans", "kind_changes", "distinct_kinds", "distinct_texts", "distinct_texts_folded",
             "without_text", "distinct_dates", "without_date", "born_past", "horizon_fallbacks",
             "horizon_mismatch")
_SUM_HEAD = ("plans", "kind changes", "distinct kinds", "distinct texts", "distinct texts "
             "(folded)", "no text", "distinct dates", "no date", "born past-dated",
             "horizon fallbacks", "horizon mismatch")


def render(out: dict, env: str, days: int, since: datetime, head: str) -> str:
    recs = out["records"]
    lines = ["# Catalyst topic readout", "",
             f"Env {env}; trade_plans created since {since.astimezone(NY):%Y-%m-%d %H:%M} ET "
             f"({days} days); code {head}. Counts only; no rule or threshold is chosen. "
             f"topic_kind: regex v{TOPIC_VERSION} (sha {TOPIC_SHA}), {OPEN_LABEL}. It is not "
             "earnings_claim's catalyst_kind (a verdict's catalyst-date label). No catalyst text, "
             "price or level is printed.", "",
             f"Plans: {len(recs)}; chains: {len(out['chains'])}; tickers: {len(out['per_ticker'])}.",
             "", "## Per ticker", "",
             "| ticker | chains | " + " | ".join(_SUM_HEAD) + " |",
             "|---|" + "---:|" * (len(_SUM_HEAD) + 1)]
    for t, s in sorted(out["per_ticker"].items(), key=lambda kv: (-kv[1]["plans"], kv[0] or "")):
        lines.append(f"| {t} | {s['chains']} | " + " | ".join(str(s[k]) for k in _SUM_COLS) + " |")
    lines += ["", "Kind changes count consecutive rows of one chain in revision order; a "
              "ticker's figure adds its chains'.", "", "## topic_kind by ticker", ""]
    kc = Counter((r["ticker"], r["topic_kind"]) for r in recs)
    cols = [k for k in (*TOPIC_ORDER, UNLABELLED) if any(c == k for _, c in kc)]
    lines += ["| ticker | " + " | ".join(cols) + " | all |", "|---|" + "---:|" * (len(cols) + 1)]
    for t in sorted({t for t, _ in kc}, key=lambda x: x or ""):
        cells = [kc.get((t, c), 0) for c in cols]
        lines.append(f"| {t} | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    multi = sum(len(r["topic_kinds"]) > 1 for r in recs)
    lines += ["", f"Rows matching more than one group (the first in {', '.join(TOPIC_ORDER)} is "
              f"the label): {multi}.", "", "## Chains", "",
              "| ticker | chain (first plan) | starts before window | first created (ET) | "
              "last created (ET) | " + " | ".join(_SUM_HEAD) + " |",
              "|---|---|---|---|---|" + "---:|" * len(_SUM_HEAD)]
    for c in out["chains"]:
        s = summarize(c["rows"])
        lines.append(f"| {c['ticker']} | {c['root_id']} | {'yes' if c['starts_before_window'] else 'no'} "
                     f"| {c['rows'][0]['created_et']} | {c['rows'][-1]['created_et']} | "
                     + " | ".join(str(s[k]) for k in _SUM_COLS) + " |")
    lines += ["", "## Plan rows (revision order within each chain)", "",
              "| ticker | chain | plan | rev | source | status | created (ET) | topic_kind | "
              "catalyst date | born past-dated | horizon basis | horizon end |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in out["chains"]:
        for r in c["rows"]:
            lines.append(f"| {r['ticker']} | {c['root_id']} | {r['plan_id']} | {r['revision']} | "
                         f"{r['source'] or '-'} | {r['status']} | {r['created_et']} | "
                         f"{r['topic_kind']} | {r['catalyst_date'] or '-'} | "
                         f"{'yes' if r['born_past'] else 'no'} | {r['horizon_basis']} | "
                         f"{r['horizon_end'] or '-'} |")
    return "\n".join(lines).rstrip() + "\n"


def export_rows(recs: list[dict]) -> list[dict]:
    """The rows for --export: catalyst text with dollar figures masked; no
    price or level column exists to leave out. Pure."""
    return [{**{k: v for k, v in r.items() if k != "_text"},
             "catalyst_event": mask_dollars(r["_text"])} for r in recs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--env", default="live")
    ap.add_argument("--ticker", help="only this ticker's plans")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--export", help="write the rows (JSON, dollar figures masked) here")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    if in_quiet_window(now):
        print("04:50-05:12 ET: no DB reads in this window; not reading.", file=sys.stderr)
        return 2
    since = now - timedelta(days=args.days)

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
            cur.execute(_SQL, {"env": args.env, "since": since})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    if args.ticker:
        raw = [r for r in raw if str(r[1]).upper() == args.ticker.upper()]
    out = build(raw)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(out, args.env, args.days, since, head or "unknown")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    if args.export:
        with open(args.export, "w") as f:
            json.dump({"topic_version": TOPIC_VERSION, "topic_sha": TOPIC_SHA,
                       "topic_order": list(TOPIC_ORDER), "label_status": OPEN_LABEL,
                       "days": args.days, "env": args.env, "per_ticker": out["per_ticker"],
                       "chains": [{"root_id": c["root_id"], "ticker": c["ticker"],
                                   "starts_before_window": c["starts_before_window"],
                                   "plan_ids": [r["plan_id"] for r in c["rows"]],
                                   **summarize(c["rows"])} for c in out["chains"]],
                       "rows": export_rows(out["records"])}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
