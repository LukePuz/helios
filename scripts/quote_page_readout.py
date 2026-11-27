"""Quote page readout: how many screened news items are option-chain quote
pages ("<TICKER> <Mon> <YYYY> <strike> call|put (<OCC symbol>) ..."), per
day and ticker, with their share of screens and of screen time (audit
2026-11-27 06:53 #5). Read-only engineering readout from the screen
worker's log; counts, percent and seconds only, no title text.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/quote_page_readout.py \
        [--log logs/celery_screen_worker.log ...] [--since 2026-11-19] \
        [--until 2026-11-28] [--out FILE] [--json FILE]

Source: the log only (default logs/celery_screen_worker.log under the
working directory; --log can be given more than once, e.g. with the rotated
.1). No database, no runtime module, no vector store is read. The file is
read as bytes and decoded with errors replaced, as grep -a reads it. A day
is the date of the log line's timestamp (the host's clock, New York time).

Per screen (crew.screen_event's "[TICKER] Screening event: <source> —
<title>" line, whose title is the event title's first 60 characters):
  - ticker: the [TICKER] tag of that line (the only part printed);
  - quote page: OCC_RULES (version OCC_RULES_VERSION, sha OCC_RULES_SHA)
    on the logged title: "occ", a full OCC option symbol (root, YYMMDD,
    C or P, 8-digit strike); else "prefix", the quote page's leading
    "<root> <Mon> <YYYY> <strike> call|put" (the 60-character cut can end
    the title before the symbol); else not a quote page;
  - root mismatch: an "occ" match whose root is not the [TICKER] tag;
  - outcome: relevant=True / False from the next "Screened [source] T:"
    line on the same worker process, else unknown;
  - seconds: the next "Task tasks.screen_event[...] succeeded in Xs" line on
    the same worker process. A screen whose worker logs another screen (or
    a failure) first has no duration and is counted as unpaired.

Tables: per day, per ticker, and per day and ticker (quote-page rows only):
screens, quote pages and their percent of screens, rule counts, outcomes,
and screen seconds (all screens, quote pages, and the quote pages' percent
of the seconds). OCC_RULES is a reporting aid: any drop or skip rule for
these pages (before the vector store or the screen) is Luke's, and the
readout chooses none. Every screened item, quote page or not, is stored in
vector memory before its screen (crew.screen_event); that store is not read
here. Imports: the standard library only. Not loaded by weekly_readouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG = os.path.join("logs", "celery_screen_worker.log")
RULE_NOTE = ("OCC_RULES is a reporting aid; any drop or skip rule for these pages (before the "
             "vector store or the screen) is Luke's, and none is chosen here.")

OCC_RULES_VERSION = 1
# (rule, pattern), tried in order on the logged title; case-sensitive.
OCC_RULES = (
    ("occ", r"\b([A-Z]{1,6})(\d{6})([CP])(\d{8})\b"),
    ("prefix", r"^[A-Z][A-Z.\-]{0,6} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
               r"\d{4} \d+(?:\.\d+)? (?:call|put)\b"),
)
_RULES = tuple((r, re.compile(p)) for r, p in OCC_RULES)
OCC_RULES_SHA = hashlib.sha256(
    json.dumps([list(t) for t in OCC_RULES], separators=(",", ":")).encode()).hexdigest()[:12]

_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d{3}: \w+/([^\]]+)\] (.*)$")
_START = re.compile(r"^\[([^\]]*)\] Screening event: \S+ — (.*)$")
_SCREENED = re.compile(r"^Screened \[[^\]]*\] \S+: relevant=(True|False)\b")
_DONE = re.compile(r"^Task tasks\.screen_event\[[^\]]*\] succeeded in ([0-9.]+)s")
_FAIL = re.compile(r"^Task tasks\.screen_event\[[^\]]*\] (?:raised|failed)")


def quote_rule(title: str | None) -> tuple[str | None, str | None]:
    """(rule, OCC root) for a logged title: ("occ", root), ("prefix", None)
    or (None, None). Pure."""
    for rule, rx in _RULES:
        m = rx.search(title or "")
        if m:
            return rule, (m.group(1) if rule == "occ" else None)
    return None, None


def parse(lines, since: date | None = None, until: date | None = None) -> tuple[list[dict], int]:
    """One record per screen, in log order, and the number of log lines
    read. A record keeps flags, never the title. since / until bound the
    day ([since, until)). Pure over the lines."""
    open_by_worker: dict[str, dict] = {}
    out, n = [], 0
    for raw in lines:
        n += 1
        m = _LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        day, worker, msg = m.groups()
        s = _START.match(msg)
        if s:
            d = date.fromisoformat(day)
            if (since and d < since) or (until and d >= until):
                open_by_worker.pop(worker, None)
                continue
            rule, root = quote_rule(s.group(2))
            rec = {"day": day, "ticker": s.group(1) or "none", "rule": rule,
                   "root_mismatch": bool(root and root != s.group(1)),
                   "relevant": None, "seconds": None}
            out.append(rec)
            open_by_worker[worker] = rec
            continue
        rec = open_by_worker.get(worker)
        if rec is None:
            continue
        r = _SCREENED.match(msg)
        if r:
            rec["relevant"] = r.group(1) == "True"
            continue
        dn = _DONE.match(msg)
        if dn:
            try:
                rec["seconds"] = float(dn.group(1))
            except ValueError:
                pass
            open_by_worker.pop(worker, None)
        elif _FAIL.match(msg):
            open_by_worker.pop(worker, None)
    return out, n


def _pct(k, n) -> str:
    return f"{100.0 * k / n:.1f}%" if n else "-"


def summarise(recs: list[dict]) -> dict:
    """Counts and seconds for a group of screens. Pure."""
    q = [r for r in recs if r["rule"]]
    sec_all = sum(r["seconds"] for r in recs if r["seconds"] is not None)
    sec_q = sum(r["seconds"] for r in q if r["seconds"] is not None)
    return {"screens": len(recs), "quote_pages": len(q),
            "occ": sum(r["rule"] == "occ" for r in q),
            "prefix": sum(r["rule"] == "prefix" for r in q),
            "root_mismatch": sum(r["root_mismatch"] for r in q),
            "relevant_true": sum(r["relevant"] is True for r in q),
            "relevant_false": sum(r["relevant"] is False for r in q),
            "relevant_unknown": sum(r["relevant"] is None for r in q),
            "unpaired": sum(r["seconds"] is None for r in recs),
            "seconds_all": round(sec_all, 1), "seconds_quote": round(sec_q, 1)}


_COLS = ("screens | quote pages | % of screens | occ | prefix | root mismatch | "
         "relevant T/F/? | s all | s quote | % of s | unpaired")


def _cells(s: dict) -> str:
    return (f"{s['screens']} | {s['quote_pages']} | {_pct(s['quote_pages'], s['screens'])} | "
            f"{s['occ']} | {s['prefix']} | {s['root_mismatch']} | "
            f"{s['relevant_true']}/{s['relevant_false']}/{s['relevant_unknown']} | "
            f"{s['seconds_all']:.1f} | {s['seconds_quote']:.1f} | "
            f"{_pct(s['seconds_quote'], s['seconds_all'])} | {s['unpaired']}")


def _table(title: str, key_cols: list[str], groups: dict, keys) -> list[str]:
    out = [f"### {title}", "", "| " + " | ".join(key_cols) + " | " + _COLS + " |",
           "|" + "---|" * len(key_cols) + "---:|" * 11]
    for k in keys:
        kk = k if isinstance(k, tuple) else (k,)
        out.append("| " + " | ".join(kk) + " | " + _cells(summarise(groups[k])) + " |")
    return out + [""]


def _group(recs, key) -> dict:
    g = defaultdict(list)
    for r in recs:
        g[key(r)].append(r)
    return g


def render(recs: list[dict], logs: list[str], n_lines: int, head: str) -> str:
    tot = summarise(recs)
    lines = ["# Quote page readout", "",
             f"Log {', '.join(os.path.basename(p) for p in logs)} ({n_lines} lines); code {head}. "
             f"OCC rules v{OCC_RULES_VERSION} (sha {OCC_RULES_SHA}) on the logged title (its first "
             "60 characters). Ticker from the [TICKER] tag; no title text is printed. Days are "
             "the log's local (New York) dates.", "", RULE_NOTE, "",
             f"Screens: {tot['screens']}; quote pages {tot['quote_pages']} "
             f"({_pct(tot['quote_pages'], tot['screens'])}); screen seconds {tot['seconds_all']:.1f}, "
             f"on quote pages {tot['seconds_quote']:.1f} "
             f"({_pct(tot['seconds_quote'], tot['seconds_all'])}); unpaired {tot['unpaired']}.", ""]
    g = _group(recs, lambda r: r["day"])
    lines += _table("Per day", ["day"], g, sorted(g))
    g = _group(recs, lambda r: r["ticker"])
    lines += _table("Per ticker (all days)", ["ticker"], g, sorted(g))
    g = _group(recs, lambda r: (r["day"], r["ticker"]))
    keys = sorted(k for k, v in g.items() if any(r["rule"] for r in v))
    lines += _table("Per day and ticker (rows with a quote page)", ["day", "ticker"], g, keys)
    return "\n".join(lines).rstrip() + "\n"


def _read(path: str):
    with open(path, "rb") as f:
        for b in f:
            yield b.decode("utf-8", errors="replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append",
                    help=f"screen worker log (repeatable; default {DEFAULT_LOG} under the cwd)")
    ap.add_argument("--since", help="YYYY-MM-DD, first log day")
    ap.add_argument("--until", help="YYYY-MM-DD, exclusive")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-day and per-ticker counts here")
    args = ap.parse_args(argv)
    try:
        since = date.fromisoformat(args.since) if args.since else None
        until = date.fromisoformat(args.until) if args.until else None
    except ValueError:
        print("--since / --until must be YYYY-MM-DD.", file=sys.stderr)
        return 2
    if since and until and until <= since:
        print("--until must be after --since.", file=sys.stderr)
        return 2
    logs = args.log or [DEFAULT_LOG]
    missing = [p for p in logs if not os.path.isfile(p)]
    if missing:
        print(f"No log file: {', '.join(missing)}.", file=sys.stderr)
        return 2
    recs, n = [], 0
    for p in logs:
        r, k = parse(_read(p), since, until)
        recs += r
        n += k
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(recs, logs, n, head or "unknown")
    if args.json:
        by_day = _group(recs, lambda r: r["day"])
        by_tk = _group(recs, lambda r: r["ticker"])
        with open(args.json, "w") as f:
            json.dump({"code_version": head or "unknown", "occ_rules_version": OCC_RULES_VERSION,
                       "occ_rules_sha": OCC_RULES_SHA, "note": RULE_NOTE,
                       "logs": [os.path.basename(p) for p in logs], "lines": n,
                       "total": summarise(recs),
                       "per_day": {k: summarise(v) for k, v in sorted(by_day.items())},
                       "per_ticker": {k: summarise(v) for k, v in sorted(by_tk.items())}},
                      f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
