"""Prompt injection readout: counts of instruction-like phrases, JSON action
fragments and "=== X ===" header lines inside the untrusted spans of each
stored analyst prompt (audit 2026-11-27 00:53 #4). Read-only engineering
readout. It prints counts and signal ids only, never the matched text.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/prompt_injection_readout.py [--since 2026-10-28] [--until 2026-11-28] \
        [--env live] [--out FILE] [--json FILE]

Rows: signals with a stored prompt, or with decision.model_output (an
analyst verdict) and no prompt, counted as "prompt NULL". Timestamp in
[--since, --until) as New York dates, in --env or with a NULL env. Each row is
counted once. Its day is the New York date of signals.timestamp. Its path is
provenance.trigger.path ("none" when absent).

The SQL returns each prompt from its first "=== SITUATION BRIEF ===" line to
the end (the brief and the task; the philosophy and thesis excerpts before it
are not read). A prompt without that line is counted as "no brief".

Untrusted spans (context_engine.assemble_brief / build_scan_brief), found
from the real section headers, searched in this order:
  - title: after the first "\\nEvent Title: " up to the next
    "\\nCurrent Price: " (the event's title).
  - event: after the first "\\n=== EVENT CONTENT ===\\n" (after the title)
    up to the next "\\n=== CURRENT PORTFOLIO ===\\n", the only section that
    follows it (up to EVENT_TEXT_CAP, 3000, characters of event text and the
    truncation line).
  - memory: after the first "\\n=== RELEVANT HISTORICAL CONTEXT (from
    memory) ===\\n" or "\\n=== RELEVANT HISTORICAL CONTEXT ===\\n" (after the
    event span) up to the next "\\n=== RELATED SYMBOLS IN WATCHLIST ===\\n" or
    "\\n=== YOUR TASK ===\\n", whichever comes first, the only two sections
    that follow it (the memory items, up to 400 characters each, and our own
    "[SOURCE | score | date]" lines).
A span whose opening line is there but whose closing line isn't is counted
as "unbounded" and not scanned. A copied closing header inside a span would
end it early, so the readout also counts prompts where a bounding header
line (CURRENT PORTFOLIO, EVENT CONTENT, RELEVANT HISTORICAL CONTEXT, RELATED
SYMBOLS IN WATCHLIST, YOUR TASK) appears more than once ("bounding header
repeated"), with their ids.

The 7B screener's prompt (agents/screener.py: the title and the first 500
characters of the event text) is not stored, only its shas, so it is not
read here. For an event the screener passed, those characters are the start
of the title and event spans.

Pattern list v1 (PATTERNS_V1, re.IGNORECASE | re.MULTILINE; its sha is
printed): instruction-like phrases (I1-I9) and JSON action fragments in the
analyst's output keys (J1-J4), over all three spans; and header lines
"^=== .* ===$" (H1, re.MULTILINE) inside a span. A header line is labelled by
the real section it copies (REAL_HEADERS, our own names), or "other"; the
line itself is not printed.

The list is a reporting aid: a hit is a string to look at, not a finding of
injection, and no hit is not proof of none. Any fencing or escaping of
untrusted text, or a rule in the prompt, is Luke's (audit 00:53 #4 options
a-c); this readout changes nothing.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db and the standard library.
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
DEFAULT_SINCE = "2026-10-28"

PATTERN_VERSION = "v1"
_ACTIONS = "OPEN|ADD|HOLD|TRIM|EXIT|WATCH|PASS"
# (id, kind, regex). Changing any entry changes the printed sha; bump the
# version with it.
PATTERNS_V1 = (
    ("I1", "instruction", r"\bignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|preceding)\s+(?:instructions?|prompts?|rules|directions|messages)\b"),
    ("I2", "instruction", r"\bdisregard\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous\s+|prior\s+|above\s+|earlier\s+|preceding\s+)?(?:instructions?|prompts?|rules|directions|messages)\b"),
    ("I3", "instruction", r"\bforget\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous\s+|prior\s+|above\s+|earlier\s+)?(?:instructions?|prompts?|rules)\b"),
    ("I4", "instruction", r"\byou\s+are\s+now\b"),
    ("I5", "instruction", r"\bsystem\s+prompt\b"),
    ("I6", "instruction", r"\bnew\s+instructions?\s*:"),
    ("I7", "instruction", r"\boverride\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous\s+|prior\s+)?(?:instructions?|rules|guidelines)\b"),
    ("I8", "instruction", r"^\s*(?:system|assistant|user)\s*:"),
    ("I9", "instruction", r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[/?INST\]|<<SYS>>|<\|system\|>"),
    ("J1", "json_action", r'"action"\s*:\s*"(?:' + _ACTIONS + r')"'),
    ("J2", "json_action", r'"conviction"\s*:\s*\d'),
    ("J3", "json_action", r'"pass_to_risk"\s*:\s*(?:true|false)\b'),
    ("J4", "json_action", r'"(?:upside_downside_ratio|thesis_bucket|expected_return_pct)"\s*:'),
)
PATTERN_FLAGS = re.IGNORECASE | re.MULTILINE
HEADER_LINE = r"^=== .* ===$"
KINDS = ("instruction", "json_action", "header")
SPANS = ("title", "event", "memory")
_COMPILED = tuple((pid, kind, re.compile(rx, PATTERN_FLAGS)) for pid, kind, rx in PATTERNS_V1)
_HEADER = re.compile(HEADER_LINE, re.MULTILINE)

BRIEF_HEADER = "=== SITUATION BRIEF ==="
TITLE_START, TITLE_END = "\nEvent Title: ", "\nCurrent Price: "
EVENT_START, EVENT_END = "\n=== EVENT CONTENT ===\n", "\n=== CURRENT PORTFOLIO ===\n"
MEMORY_STARTS = ("\n=== RELEVANT HISTORICAL CONTEXT (from memory) ===\n",
                 "\n=== RELEVANT HISTORICAL CONTEXT ===\n")
MEMORY_ENDS = ("\n=== RELATED SYMBOLS IN WATCHLIST ===\n", "\n=== YOUR TASK ===\n")
_T = r"[A-Z0-9.\-]+"
# The bounding header lines; each appears at most once in a real prompt.
BOUNDING = (("CURRENT PORTFOLIO", r"^=== CURRENT PORTFOLIO ===$"),
            ("EVENT CONTENT", r"^=== EVENT CONTENT ===$"),
            ("RELEVANT HISTORICAL CONTEXT", r"^=== RELEVANT HISTORICAL CONTEXT(?: \(from memory\))? ===$"),
            ("RELATED SYMBOLS IN WATCHLIST", r"^=== RELATED SYMBOLS IN WATCHLIST ===$"),
            ("YOUR TASK", r"^=== YOUR TASK ===$"))
_BOUNDING = tuple((name, re.compile(rx, re.MULTILINE)) for name, rx in BOUNDING)
# The real section headers of the analyst prompt and its brief (context_engine,
# agents/analyst.py, ingestion/macro.py, ingestion/polygon_stream.py), as the
# label a copied header line is reported under.
REAL_HEADERS = (
    ("ACCOUNT STATE", r"ACCOUNT STATE"),
    ("ADDITIONAL REQUIRED FIELD: *", r"ADDITIONAL REQUIRED FIELD: .+"),
    ("CANDIDATE DETAILS", r"CANDIDATE DETAILS"),
    ("CURRENT MACRO ENVIRONMENT", r"CURRENT MACRO ENVIRONMENT"),
    ("CURRENT PORTFOLIO", r"CURRENT PORTFOLIO"),
    ("EVENT CONTENT", r"EVENT CONTENT"),
    ("FUNDAMENTALS — T", r"FUNDAMENTALS — " + _T),
    ("MARKET REGIME", r"MARKET REGIME"),
    ("OPTIONS FLOW — T", r"OPTIONS FLOW — " + _T),
    ("PORTFOLIO PHILOSOPHY (excerpt)", r"PORTFOLIO PHILOSOPHY \(excerpt\)"),
    ("POSITION CONTEXT — T (...)", r"POSITION CONTEXT — " + _T + r" \(.+\)"),
    ("PRICE HISTORY — T (...)", r"PRICE HISTORY — " + _T + r" \(.+\)"),
    ("PRICE STRUCTURE — T", r"PRICE STRUCTURE — " + _T),
    ("RELATED SYMBOLS IN WATCHLIST", r"RELATED SYMBOLS IN WATCHLIST"),
    ("RELEVANT HISTORICAL CONTEXT", r"RELEVANT HISTORICAL CONTEXT(?: \(from memory\))?"),
    ("SECTOR THESIS", r"SECTOR THESIS"),
    ("SITUATION BRIEF", r"SITUATION BRIEF(?: \(daily scan\))?"),
    ("T TECHNICALS", _T + r" TECHNICALS"),
    ("UPCOMING CATALYST CALENDAR (...)", r"UPCOMING CATALYST CALENDAR \(.+\)"),
    ("YOUR TASK", r"YOUR TASK"),
    ("YOUR TRACK RECORD", r"YOUR TRACK RECORD"),
)
_REAL = tuple((label, re.compile(r"=== " + rx + r" ===")) for label, rx in REAL_HEADERS)
REASONS = ("prompt NULL", "no brief")

_ROWS_SQL = """
SELECT s.id, s.timestamp,
       s.provenance->'trigger'->>'path',
       s.prompt IS NULL,
       CASE WHEN strpos(s.prompt, %(brief_hdr)s) > 0
            THEN substr(s.prompt, strpos(s.prompt, %(brief_hdr)s)) END
FROM signals s
WHERE (s.prompt IS NOT NULL OR s.decision ? 'model_output')
  AND s.timestamp >= %(start)s AND s.timestamp < %(end)s
  AND (s.trading_env = %(env)s OR s.trading_env IS NULL)
ORDER BY s.timestamp, s.id
"""


def pattern_sha() -> str:
    """sha256 (12 hex) of the pattern list, its flags and the header regex."""
    blob = json.dumps({"version": PATTERN_VERSION, "flags": "IGNORECASE|MULTILINE",
                       "patterns": [list(p) for p in PATTERNS_V1],
                       "header": HEADER_LINE, "header_flags": "MULTILINE"},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def db_root(cwd: str | None = None) -> str:
    """The directory db is imported from: cwd when it holds a db.py, else
    this script's checkout."""
    cwd = os.getcwd() if cwd is None else cwd
    return cwd if os.path.isfile(os.path.join(cwd, "db.py")) else ROOT


def in_quiet_window(now: datetime) -> bool:
    """True between 04:50 and 05:12 ET, when the readout doesn't read."""
    t = now.astimezone(NY).time()
    return QUIET_START <= t < QUIET_END


def _between(text: str, start: str, ends: tuple[str, ...], frm: int):
    """(span_start, span_end) of the text after the first `start` at or
    after frm up to the nearest of `ends`; None when `start` isn't there;
    "unbounded" when no end follows it."""
    i = text.find(start, frm)
    if i < 0:
        return None
    a = i + len(start)
    found = [j for j in (text.find(e, a) for e in ends) if j >= 0]
    return (a, min(found)) if found else "unbounded"


def find_spans(brief: str) -> dict:
    """{span: (start, end) | None | "unbounded"} over the prompt slice that
    starts at the SITUATION BRIEF line. Pure."""
    title = _between(brief, TITLE_START, (TITLE_END,), 0)
    frm = title[1] if isinstance(title, tuple) else 0
    event = _between(brief, EVENT_START, (EVENT_END,), frm)
    frm = event[1] if isinstance(event, tuple) else frm
    memory = None
    for start in MEMORY_STARTS:
        memory = _between(brief, start, MEMORY_ENDS, frm)
        if memory is not None:
            break
    return {"title": title, "event": event, "memory": memory}


def real_header_label(line: str) -> str:
    for label, rx in _REAL:
        if rx.fullmatch(line):
            return label
    return "other"


def scan(brief: str) -> dict:
    """Counts for one prompt slice: per span whether it was found, matches
    per (kind, span) and per pattern id, the real-section labels of header
    lines inside spans, and the bounding headers seen more than once. No
    matched text leaves this function."""
    spans = find_spans(brief)
    out = {"spans": {}, "unbounded": [], "matches": {}, "patterns": {},
           "header_labels": {}, "bounding_repeated": []}
    for name in SPANS:
        b = spans[name]
        out["spans"][name] = isinstance(b, tuple)
        if b == "unbounded":
            out["unbounded"].append(name)
        if not isinstance(b, tuple):
            continue
        text = brief[b[0]:b[1]]
        for pid, kind, rx in _COMPILED:
            n = len(rx.findall(text))
            if n:
                out["matches"][f"{kind}/{name}"] = out["matches"].get(f"{kind}/{name}", 0) + n
                out["patterns"][f"{pid}/{name}"] = n
        lines = _HEADER.findall(text)
        if lines:
            out["matches"][f"header/{name}"] = len(lines)
            out["patterns"][f"H1/{name}"] = len(lines)
            labels = Counter(real_header_label(x) for x in lines)
            out["header_labels"][name] = dict(sorted(labels.items()))
    out["bounding_repeated"] = [n for n, rx in _BOUNDING if len(rx.findall(brief)) > 1]
    return out


def classify(row) -> dict:
    """One row (as _ROWS_SQL returns it) as counts and labels."""
    sid, ts, path, prompt_null, brief = row
    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    out = {"id": sid, "day": ts.astimezone(NY).date().isoformat(),
           "at": ts.astimezone(NY).strftime("%Y-%m-%d %H:%M"), "path": path or "none",
           "reason": None}
    if prompt_null:
        out["reason"] = "prompt NULL"
    elif brief is None:
        out["reason"] = "no brief"
    if out["reason"]:
        out.update(spans={s: False for s in SPANS}, unbounded=[], matches={}, patterns={},
                   header_labels={}, bounding_repeated=[])
        return out
    out.update(scan(brief))
    return out


def tally(rows: list[dict]) -> dict:
    """Per span: prompts with the span, unbounded, and per kind the prompts
    with a match and the match count."""
    t = {"rows": len(rows), "scanned": sum(1 for r in rows if not r["reason"]),
         "reasons": {k: sum(1 for r in rows if r["reason"] == k) for k in REASONS},
         "bounding_repeated": sum(1 for r in rows if r["bounding_repeated"]),
         "any_hit": sum(1 for r in rows if r["matches"]), "spans": {}}
    for s in SPANS:
        cell = {"present": sum(1 for r in rows if r["spans"][s]),
                "unbounded": sum(1 for r in rows if s in r["unbounded"])}
        for k in KINDS:
            key = f"{k}/{s}"
            cell[k] = {"prompts": sum(1 for r in rows if r["matches"].get(key)),
                       "matches": sum(r["matches"].get(key, 0) for r in rows)}
        t["spans"][s] = cell
    return t


def build(rows: list[dict]) -> dict:
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["day"], r["path"]), []).append(r)
    hits: dict = {}
    for r in rows:
        for key in r["patterns"]:
            hits.setdefault(key, []).append(r["id"])
    labels: dict = {}
    for r in rows:
        for span, c in r["header_labels"].items():
            for label in c:
                labels.setdefault(f"{span} / {label}", []).append(r["id"])
    return {"total": tally(rows),
            "by_day_path": {f"{d} / {p}": tally(v) for (d, p), v in sorted(cells.items())},
            "hit_ids": dict(sorted(hits.items())),
            "header_label_ids": dict(sorted(labels.items())),
            "unbounded_ids": {s: [r["id"] for r in rows if s in r["unbounded"]] for s in SPANS},
            "bounding_repeated_ids": {n: [r["id"] for r in rows if n in r["bounding_repeated"]]
                                      for n, _ in BOUNDING}}


def _pm(c: dict) -> str:
    return f"{c['prompts']} / {c['matches']}"


def render(out: dict, env: str, since: date, until: date, head: str, now: datetime) -> str:
    t = out["total"]
    lines = [f"# Prompt injection readout ({env}, prompts {since.isoformat()} to before "
             f"{until.isoformat()} ET)", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             f"Pattern list {PATTERN_VERSION}, sha {pattern_sha()}. Engineering readout; counts "
             "and signal ids only, no matched text. Spans are the untrusted text of the brief "
             "(title: the event title; event: EVENT CONTENT; memory: RELEVANT HISTORICAL CONTEXT), "
             "bounded by the real section headers. The pattern list is a reporting aid: a hit is a "
             "string to look at, not a finding of injection, and no hit is not proof of none. Any "
             "fencing or escaping of untrusted text, or a prompt rule, is Luke's (audit 2026-11-27 "
             "00:53 #4).", "",
             f"- rows: {t['rows']}; scanned: {t['scanned']}; not scanned: "
             + ", ".join(f"{k} {t['reasons'][k]}" for k in REASONS),
             f"- prompts with any match: {t['any_hit']}; with a bounding header repeated: "
             f"{t['bounding_repeated']}"]
    for s in SPANS:
        c = t["spans"][s]
        lines.append(f"- {s}: present {c['present']}, unbounded {c['unbounded']}; "
                     + "; ".join(f"{k} {_pm(c[k])}" for k in KINDS) + " (prompts / matches)")
    lines += ["", f"## Patterns ({PATTERN_VERSION}; IGNORECASE, MULTILINE)", ""]
    lines += [f"- {pid} ({kind}): `{rx}`" for pid, kind, rx in PATTERNS_V1]
    lines.append(f"- H1 (header, MULTILINE, inside a span): `{HEADER_LINE}`")
    lines += ["", "## By day, path and span (prompts / matches)", "",
              "| day / path | span | prompts | span present | unbounded | instruction | "
              "json_action | header |", "|---|" + "---|" * 7]
    for k, c in out["by_day_path"].items():
        for s in SPANS:
            sc = c["spans"][s]
            lines.append(f"| {k} | {s} | {c['rows']} | {sc['present']} | {sc['unbounded']} | "
                         + " | ".join(_pm(sc[kind]) for kind in KINDS) + " |")
    lines += ["", "## Hits by pattern and span (signal ids)", ""]
    lines += ([f"- {k} ({len(v)}): " + ", ".join(f"#{i}" for i in v)
               for k, v in out["hit_ids"].items()] or ["- none"])
    lines += ["", "## Header lines inside a span, by the real section they copy (signal ids)", ""]
    lines += ([f"- {k} ({len(v)}): " + ", ".join(f"#{i}" for i in v)
               for k, v in out["header_label_ids"].items()] or ["- none"])
    lines += ["", "## Unbounded spans and repeated bounding headers (signal ids)", ""]
    rest = [f"- unbounded {s} ({len(v)}): " + ", ".join(f"#{i}" for i in v)
            for s, v in out["unbounded_ids"].items() if v]
    rest += [f"- {n} repeated ({len(v)}): " + ", ".join(f"#{i}" for i in v)
             for n, v in out["bounding_repeated_ids"].items() if v]
    lines += rest or ["- none"]
    return "\n".join(lines) + "\n"


def code_version() -> str:
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    return head or "unknown"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-row counts (no text) here")
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
            cur.execute(_ROWS_SQL, {"env": args.env, "start": start, "end": end,
                                    "brief_hdr": BRIEF_HEADER})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = [classify(r) for r in raw]
    out = build(rows)
    head = code_version()
    text = render(out, args.env, since, until, head, now)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head, "env": args.env,
                       "since": since.isoformat(), "until": until.isoformat(),
                       "pattern_version": PATTERN_VERSION, "pattern_sha": pattern_sha(),
                       "rows": rows}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
