"""Trigger engagement readout: does an analyst verdict's text use any content
word of the item that triggered it? (audit 2026-11-26 14:53 #1). Read-only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/trigger_engagement_readout.py [--days 14] [--env live] \
        [--out FILE] [--json FILE] [--watchlist knowledge/watchlist.md]

Rows: every signals row in the window with decision.model_output (analyst
verdicts; the shadow rows carry none) in --env, or with a NULL env (rows
before 10-28 have none). Scheduled reviews are left out: their title is
generic ("Periodic position review"). Their text still counts toward the
word frequencies below, as in the audit.

Per verdict, the trigger title's content words: lower-case words matching
[a-z][a-z0-9&-]{2,}, minus the audit's stopword list (STOPWORDS, which
holds a few company names), the ticker, and the words of the ticker's
company name in knowledge/watchlist.md. The verdict text is the string
values of model_output's reasoning, thesis_summary, key_risks and
expectation_gap (not their JSON keys), lower-cased.

Two bounds, as the audit defined them:
  - loose: a title word occurs in the text as a substring, or a title word
    longer than PREFIX_LEN characters has its first PREFIX_LEN characters
    in the text.
  - strict: only the title words that are rare in this ticker's own
    verdicts count, rare meaning they occur (as a substring) in less than
    RARE_DF of the ticker's verdict texts in the window. Then a rare word
    must occur in the text as a substring. A title with no rare word is
    reported as "no rare word", not as a miss.

Per row, each title word's flags (company word, substring, prefix) and its
frequency are kept (--json), so any other stopword set, cutoff or matching
rule can be applied to the same rows. Which bound, cutoff or definition is
the measure is not this script's call (Luke's); the counts are description
only. No prices or account data; titles are not printed.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction, rolled back. Imports: db and the standard library only. Not loaded by
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
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
PREFIX_LEN = 5        # the audit's prefix rule (words longer than 5 characters)
RARE_DF = 0.25        # the audit's strict cutoff
EXCLUDED_PATHS = ("scheduled_review",)
TEXT_FIELDS = ("reasoning", "thesis_summary", "key_risks", "expectation_gap")

# The audit's stopword list (audit 2026-11-26 14:53, a1456/p2.py), unchanged
# so the counts compare with its 18% / 54%.
STOPWORDS = frozenset("""a an the and or of to in on for as at by with from is are was were be
been its it this that these those after before over into amid why what how will should could
would can may might stock stocks shares share inc corp company nasdaq nyse report reports says
said new up down vs than more most just now here there their his her our your you who has have
had not but if so about still yahoo finance cnbc barron s reuters bloomberg marketwatch investor
business daily motley fool zacks benzinga seeking alpha wsj live updates today week year month
first last top buy sell hold price prices market markets nvidia arista networks palo alto
alphabet google teradyne ionq microsoft apple amazon meta broadcom tesla amd ai""".split())
STOPWORDS_SHA = hashlib.sha256(" ".join(sorted(STOPWORDS)).encode()).hexdigest()[:12]
_WORD = re.compile(r"[a-z][a-z0-9&\-]{2,}")
_WATCH_ROW = re.compile(r"^\|\s*([A-Z]{1,6})\s*\|\s*([^|]+?)\s*\|")

_SQL = (
    "SELECT id, timestamp, symbol, signal, provenance->'trigger'->>'path', "
    "provenance->'trigger'->>'source', provenance->'trigger'->>'new_fact', "
    "provenance->'trigger'->>'title', decision->'model_output' FROM signals "
    "WHERE timestamp > %(since)s AND decision ? 'model_output' "
    "AND (trading_env IS NULL OR trading_env = %(env)s) ORDER BY id")


def words(text: str | None) -> list[str]:
    """Lower-case words of text, in order, duplicates kept. Pure."""
    return _WORD.findall((text or "").lower())


def company_names(path: str | None = None) -> dict[str, frozenset]:
    """ticker -> the words of its company name, from every ticker row of
    knowledge/watchlist.md (any role column). {} when the file can't be
    read."""
    try:
        with open(path or os.path.join(ROOT, "knowledge", "watchlist.md")) as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        m = _WATCH_ROW.match(line)
        if m and m.group(1) not in ("TICKER",):
            out.setdefault(m.group(1), frozenset(words(m.group(2))))
    return out


def _strings(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, list):
        return [s for x in v for s in _strings(x)]
    return []


def verdict_text(model_output: dict | None) -> str:
    """The lower-cased string values of the four text fields. Pure."""
    mo = model_output if isinstance(model_output, dict) else {}
    return " ".join(s for k in TEXT_FIELDS for s in _strings(mo.get(k))).lower()


def title_words(title: str | None, ticker: str | None, company: frozenset = frozenset(),
                stopwords: frozenset = STOPWORDS) -> list[dict]:
    """The title's content words, first occurrence order: [{w, company}].
    Stopwords and the ticker are dropped; company-name words are kept and
    flagged (company=True), so a rule without them can be reapplied. Pure."""
    seen, out = set(), []
    tick = (ticker or "").lower()
    for w in words(title):
        if w in stopwords or w == tick or w in seen:
            continue
        seen.add(w)
        out.append({"w": w, "company": w in company})
    return out


def match_flags(w: str, text: str) -> dict:
    """{sub, prefix}: w occurs in text as a substring; w is longer than
    PREFIX_LEN and its first PREFIX_LEN characters occur in text. Pure."""
    return {"sub": w in text,
            "prefix": len(w) > PREFIX_LEN and w[:PREFIX_LEN] in text}


def loose_hit(tw: list[dict]) -> bool | None:
    """The loose bound over the non-company words: True when any word has
    sub or prefix; None when there is no such word. Pure."""
    ws = [x for x in tw if not x["company"]]
    if not ws:
        return None
    return any(x["sub"] or x["prefix"] for x in ws)


def strict_hit(tw: list[dict], rare_df: float = RARE_DF) -> bool | None:
    """The strict bound over the non-company words with df < rare_df: True
    when any of them has sub; None when there is none. Pure."""
    ws = [x for x in tw if not x["company"] and x["df"] is not None and x["df"] < rare_df]
    if not ws:
        return None
    return any(x["sub"] for x in ws)


def build_rows(raw, names: dict[str, frozenset]) -> list[dict]:
    """Rows from the SELECT tuples: per-word flags, df over the ticker's
    verdict texts (all paths), and both bounds. Excluded paths are dropped
    after df is computed. Pure."""
    texts = defaultdict(list)
    recs = []
    for sid, ts, sym, sig, path, src, nf, title, mo in raw:
        text = verdict_text(mo if isinstance(mo, dict) else _json(mo))
        texts[sym].append(text)
        recs.append((sid, ts, sym, sig, path, src, nf, title, text))
    out = []
    for sid, ts, sym, sig, path, src, nf, title, text in recs:
        if path in EXCLUDED_PATHS:
            continue
        corpus = texts[sym]
        tw = title_words(title, sym, names.get(sym, frozenset()))
        for x in tw:
            x.update(match_flags(x["w"], text))
            x["df"] = round(sum(1 for t in corpus if x["w"] in t) / len(corpus), 4)
        out.append({"id": sid, "ts": ts.isoformat() if hasattr(ts, "isoformat") else ts,
                    "ticker": sym, "action": sig, "path": path, "source": src,
                    "new_fact": nf, "company_known": sym in names,
                    "n_ticker_verdicts": len(corpus), "title_words": tw,
                    "loose": loose_hit(tw), "strict": strict_hit(tw)})
    return out


def _json(v):
    try:
        return json.loads(v) if isinstance(v, str) else {}
    except ValueError:
        return {}


def tally(rows, key) -> dict:
    """key(row) -> Counter of n / loose_n / loose_miss / strict_n /
    strict_miss / no_words / no_rare. Pure."""
    out = defaultdict(Counter)
    for r in rows:
        c = out[key(r)]
        c["n"] += 1
        if r["loose"] is None:
            c["no_words"] += 1
        else:
            c["loose_n"] += 1
            c["loose_miss"] += r["loose"] is False
        if r["strict"] is None:
            c["no_rare"] += 1
        else:
            c["strict_n"] += 1
            c["strict_miss"] += r["strict"] is False
    return dict(out)


def _pct(a: int, b: int) -> str:
    return f"{a} ({100 * a / b:.0f}%)" if b else "0"


def _table(title: str, groups: dict, label: str) -> list[str]:
    out = [f"## {title}", "",
           f"| {label} | verdicts | no title word | loose: measured | loose: no word in text "
           "| no rare word | strict: measured | strict: no rare word in text |",
           "|---|---|---|---|---|---|---|---|"]
    for k in sorted(groups, key=lambda k: (-groups[k]["n"], str(k))):
        c = groups[k]
        out.append(f"| {k} | {c['n']} | {c['no_words']} | {c['loose_n']} | "
                   f"{_pct(c['loose_miss'], c['loose_n'])} | {c['no_rare']} | {c['strict_n']} | "
                   f"{_pct(c['strict_miss'], c['strict_n'])} |")
    return out + [""]


def render(rows, env: str, days: int, since: datetime, head: str, excluded: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    unknown = sorted({r["ticker"] for r in rows if not r["company_known"]})
    lines = [
        "# Trigger engagement readout", "",
        f"Verdicts with model_output since {since:%Y-%m-%d %H:%M} UTC ({days} days), env {env} "
        f"or NULL; git {head}; generated {now}.", "",
        "Record only. Two word-overlap bounds from audit 2026-11-26 14:53 #1: loose (any title "
        f"content word, or its first {PREFIX_LEN} characters, in the verdict text) and strict (only "
        f"title words found in < {RARE_DF:g} of the ticker's verdict texts). Stopwords sha "
        f"{STOPWORDS_SHA}; company-name words from knowledge/watchlist.md are dropped. "
        "Which bound, cutoff or word rule is the measure is a research choice (Luke).", "",
        f"Rows: {len(rows)} verdicts; {excluded} scheduled reviews left out. Tickers with no "
        f"watchlist name (company words not dropped): {', '.join(unknown) or 'none'}.", "",
    ]
    lines += _table("By path", tally(rows, lambda r: r["path"]), "path")
    ev = [r for r in rows if r["path"] == "event"]
    lines += _table("Event path, by source", tally(ev, lambda r: r["source"]), "source")
    lines += _table("Event path, by new_fact", tally(ev, lambda r: r["new_fact"]), "new_fact")
    lines += _table("Event path, by action", tally(ev, lambda r: r["action"]), "action")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="write the per-row records (ids, words, flags) here")
    ap.add_argument("--watchlist", help="watchlist.md for company names "
                    "(default: knowledge/watchlist.md beside this script)")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
            cur.execute(_SQL, {"since": since, "env": args.env})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = build_rows(raw, company_names(args.watchlist))
    excluded = sum(1 for r in raw if r[4] in EXCLUDED_PATHS)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(rows, args.env, args.days, since, head or "unknown", excluded)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"stopwords_sha": STOPWORDS_SHA, "prefix_len": PREFIX_LEN,
                       "rare_df": RARE_DF, "rows": rows}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
