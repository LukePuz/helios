"""Position terms readout: how often the analyst's verdicts on a name mention
the position-management terms (the stop, trimming or exiting, the entry or
its P&L, the plan, invalidation, the horizon), per ticker, day and held
flag, with whether a held verdict's prompt carried POSITION CONTEXT (audit
2026-11-27 00:53 #1). Read-only engineering readout; counts and percent
only, no text.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/position_terms_readout.py [--since 2026-11-12] [--until 2026-11-28] \
        [--env live] [--out FILE] [--json FILE]

Verdicts: signals rows with decision.model_output (the analyst's verdicts),
timestamp in [--since, --until) as New York dates, in --env or with a NULL
env. Each verdict is counted once; its day is the New York date of
signals.timestamp. held is provenance.plan.held as stored ("held", "not
held", or "unknown" when it is missing).

Text read: model_output's reasoning, thesis_summary, key_risks and
expectation_gap. A list or object field is read as all of its strings. The
text is matched and dropped; none of it is printed or exported.

TERM_GROUPS (version TERM_GROUPS_VERSION, sha TERM_GROUPS_SHA): named regex
groups matched case-insensitively on each field: stop, trim_exit (trim /
exit / take profit), entry_pnl (entry / cost basis / P&L), plan,
invalidation (invalidation / thesis broken) and horizon (horizon / holding
period); "any" is a hit on at least one group. The groups are a reporting
aid, not a study definition: which words count as a mention, and whether
any of this is a measure, is Luke's. Other lists can be run on the same
verdicts by changing TERM_GROUPS, whose sha the output prints.

POSITION CONTEXT, per verdict: the SQL returns only flags, never prompt text:
"prompt NULL", "this ticker" (the prompt holds "=== POSITION CONTEXT —
<verdict ticker> (held) ===", context_engine._position_context_section's
header), "another ticker" (a POSITION CONTEXT header, not for this ticker)
or "none".

Tables: verdicts by held; per ticker and held; per day and held; held
verdicts by field; held verdicts by POSITION CONTEXT. Each gives n and, per
group, the count with its percent of n. No rate is chosen and no threshold.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db and the standard library. Not loaded by
weekly_readouts.
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
STATEMENT_TIMEOUT_MS = 60000
NY = ZoneInfo("America/New_York")
QUIET_START, QUIET_END = time(4, 50), time(5, 12)   # no DB reads at the stack's start
DEFAULT_SINCE = "2026-11-12"
POSITION_HEADER = "=== POSITION CONTEXT — "
FIELDS = ("reasoning", "thesis_summary", "key_risks", "expectation_gap")
HELD_LABELS = ("held", "not held", "unknown")
CONTEXT_LABELS = ("this ticker", "another ticker", "none", "prompt NULL")
AID_NOTE = ("The term groups are a reporting aid, not a study definition (which words count "
            "as a mention, and whether this is a measure, is Luke's).")

TERM_GROUPS_VERSION = 1
# (group, pattern), matched case-insensitively on each field's text.
TERM_GROUPS = (
    ("stop", r"\bstop(?:s|ped)?\b|\bstoploss(?:es)?\b"),
    ("trim_exit", r"\btrim(?:s|med|ming)?\b|\bexit(?:s|ed|ing)?\b|\btake[- ]profits?\b|"
                  r"\btaking\s+profits?\b|\bprofit[- ]taking\b|"
                  r"\b(?:reduce|reducing|cut|cutting)\s+(?:the\s+)?(?:position|exposure|stake)\b"),
    ("entry_pnl", r"\bentry\b|\bentered\b|\bcost\s+basis\b|\baverage\s+cost\b|\bavg\.?\s+cost\b|"
                  r"\bunreali[sz]ed\b|\bp&l\b|\bpnl\b|\bp/l\b|\bsince\s+entry\b"),
    ("plan", r"\b(?:trade|position|exit)\s+plan\b|\bthe\s+plan(?:'s)?\b|"
             r"\bplan(?:'s)?\s+(?:target|stop|horizon|thesis)\b|"
             r"\bplanned\s+(?:exit|target|stop|horizon)\b"),
    ("invalidation", r"\binvalidat(?:e|es|ed|ing|ion|ions)\b|"
                     r"\bthesis\s+(?:is\s+|was\s+|would\s+be\s+)?(?:broken|breaks?|violated|"
                     r"fails?|failed|void)\b|\bbroken\s+thesis\b|\bthesis[- ]break(?:er|ing)?\b"),
    ("horizon", r"\bhorizon\b|\bholding\s+period\b|\bhold(?:ing)?\s+(?:window|period)\b|"
                r"\btime[- ]stop\b|\btime[- ]based\s+exit\b"),
)
GROUPS = tuple(g for g, _ in TERM_GROUPS)
COLUMNS = GROUPS + ("any",)
_RX = tuple((g, re.compile(p, re.I)) for g, p in TERM_GROUPS)
TERM_GROUPS_SHA = hashlib.sha256(
    json.dumps([list(t) for t in TERM_GROUPS], separators=(",", ":")).encode()).hexdigest()[:12]

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol,
       s.provenance->'plan'->>'held',
       jsonb_typeof(s.decision->'model_output'),
       s.decision->'model_output'->'reasoning',
       s.decision->'model_output'->'thesis_summary',
       s.decision->'model_output'->'key_risks',
       s.decision->'model_output'->'expectation_gap',
       s.prompt IS NULL,
       strpos(s.prompt, %(pc_hdr)s || s.symbol || ' (held) ===') > 0,
       strpos(s.prompt, %(pc_hdr)s) > 0
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


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def field_text(v) -> str:
    """Every string in a field (a string, or a list / object of them, at any
    depth), joined by newlines; "" for anything else. Pure."""
    v = _json(v)
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, list):
        return "\n".join(t for t in (field_text(x) for x in v) if t)
    return ""


def groups_hit(text: str | None) -> list[str]:
    """The TERM_GROUPS groups the text matches, in list order. Pure."""
    return [g for g, rx in _RX if rx.search(text or "")]


def held_label(v) -> str:
    return {"true": "held", "false": "not held"}.get(str(v).lower() if v is not None else "",
                                                     "unknown")


def context_label(prompt_null, this_ticker, any_header) -> str:
    """The POSITION CONTEXT flag of a verdict's prompt (module docstring). Pure."""
    if prompt_null:
        return "prompt NULL"
    if this_ticker:
        return "this ticker"
    return "another ticker" if any_header else "none"


def classify(row) -> dict:
    """One verdict row (as _VERDICTS_SQL returns it) as a record of flags;
    the text is matched here and not kept. Pure."""
    (sid, ts, sym, held, mo_type, reasoning, thesis, risks, gap,
     prompt_null, pc_this, pc_any) = row
    by_field = {f: groups_hit(field_text(v))
                for f, v in zip(FIELDS, (reasoning, thesis, risks, gap))}
    hit = [g for g in GROUPS if any(g in by_field[f] for f in FIELDS)]
    return {"signal_id": sid, "day": ts.astimezone(NY).date().isoformat() if ts else None,
            "ticker": sym or "none", "held": held_label(held),
            "model_output_object": mo_type == "object",
            "position_context": context_label(prompt_null, pc_this, pc_any),
            "fields": by_field, "groups": hit + (["any"] if hit else [])}


def _pct(k: int, n: int) -> str:
    return f"{k} ({100.0 * k / n:.1f}%)" if n else "0"


def _group_table(title: str, keyed: dict, keys: list, key_cols: list[str]) -> list[str]:
    """Rows of n and per-group count (percent of n), one per key."""
    out = [f"### {title}", "", "| " + " | ".join(key_cols) + " | n | " + " | ".join(COLUMNS)
           + " |", "|" + "---|" * len(key_cols) + "---:|" * (1 + len(COLUMNS))]
    for k in keys:
        recs = keyed[k]
        n = len(recs)
        cells = [_pct(sum(c in r["groups"] for r in recs), n) for c in COLUMNS]
        kk = k if isinstance(k, tuple) else (k,)
        out.append("| " + " | ".join(str(x) for x in kk) + f" | {n} | " + " | ".join(cells) + " |")
    return out + [""]


def _by(rows: list[dict], key) -> tuple[dict, list]:
    keyed: dict = {}
    for r in rows:
        keyed.setdefault(key(r), []).append(r)
    return keyed, sorted(keyed)


def render(rows: list[dict], env: str, since: date, until: date, head: str) -> str:
    held = [r for r in rows if r["held"] == "held"]
    lines = ["# Position terms readout", "",
             f"Env {env}; analyst verdicts {since.isoformat()} to {until.isoformat()} "
             f"(New York dates, end exclusive); code {head}. Term groups "
             f"v{TERM_GROUPS_VERSION} (sha {TERM_GROUPS_SHA}) on {', '.join(FIELDS)}. "
             "Counts and percent of n only; no text is printed.", "", AID_NOTE, "",
             f"Verdicts: {len(rows)} (" + ", ".join(
                 f"{h} {sum(r['held'] == h for r in rows)}" for h in HELD_LABELS) + "). "
             f"model_output not an object: {sum(not r['model_output_object'] for r in rows)}.", ""]
    hk = {h: [r for r in rows if r["held"] == h] for h in HELD_LABELS}
    lines += _group_table("Verdicts by held", hk, [h for h in HELD_LABELS if hk[h]], ["held"])
    k, keys = _by(rows, lambda r: (r["ticker"], r["held"]))
    lines += _group_table("Per ticker and held", k, keys, ["ticker", "held"])
    k, keys = _by(rows, lambda r: (r["day"] or "none", r["held"]))
    lines += _group_table("Per day and held", k, keys, ["day", "held"])
    n = len(held)
    lines += ["### Held verdicts by field (a verdict can hit a group in several fields)", "",
              "| field | n | " + " | ".join(COLUMNS) + " |", "|---|" + "---:|" * (1 + len(COLUMNS))]
    for f in FIELDS:
        cells = []
        for c in COLUMNS:
            hit = (lambda r: bool(r["fields"][f])) if c == "any" else (
                lambda r, c=c: c in r["fields"][f])
            cells.append(_pct(sum(hit(r) for r in held), n))
        lines.append(f"| {f} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    ck = {c: [r for r in held if r["position_context"] == c] for c in CONTEXT_LABELS}
    lines += _group_table("Held verdicts by POSITION CONTEXT in the prompt (header check only)",
                          ck, [c for c in CONTEXT_LABELS if ck[c]], ["POSITION CONTEXT"])
    tc = Counter((r["ticker"], r["position_context"]) for r in held)
    lines += ["### Held verdicts: ticker by POSITION CONTEXT", "",
              "| ticker | " + " | ".join(CONTEXT_LABELS) + " | all |",
              "|---|" + "---:|" * (len(CONTEXT_LABELS) + 1)]
    for t in sorted({t for t, _ in tc}):
        cells = [tc.get((t, c), 0) for c in CONTEXT_LABELS]
        lines.append(f"| {t} | " + " | ".join(str(x) for x in cells) + f" | {sum(cells)} |")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-verdict flags (no text) here")
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
            cur.execute(_VERDICTS_SQL, {"env": args.env, "start": start, "end": end,
                                        "pc_hdr": POSITION_HEADER})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = [classify(r) for r in raw]
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(rows, args.env, since, until, head or "unknown")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head or "unknown",
                       "env": args.env, "since": since.isoformat(), "until": until.isoformat(),
                       "term_groups_version": TERM_GROUPS_VERSION,
                       "term_groups_sha": TERM_GROUPS_SHA, "note": AID_NOTE,
                       "verdicts": rows}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
