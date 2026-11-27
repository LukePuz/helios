"""Memory relevance readout: what the brief's memory section held for each
analyst verdict, read from the prompt stored on the signals row (audit
2026-11-26 22:53 #7). Read-only engineering readout; counts, scores and
collection names only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/memory_relevance_readout.py [--since 2026-11-20] [--until 2026-11-27] \
        [--env live] [--watchlist FILE] [--out FILE] [--json FILE]

Verdicts: signals rows with decision.model_output (the analyst's verdicts),
timestamp in [--since, --until) as New York dates, in --env or with a NULL
env. Each verdict is counted once; its day is the New York date of
signals.timestamp.

The memory section is the "=== RELEVANT HISTORICAL CONTEXT (from memory) ==="
block that context_engine.assemble_brief and build_scan_brief write from
vector_store.search_all_collections (the top n_results*2 items across the
collections, with no score floor and no ticker filter). Each item is written
as "[COLLECTION | score=S | DATE]" followed by up to 400 characters of the
item's text. The readout parses the first such block in the stored prompt:
it ends at the next "=== NAME ===" line that is not an item's first text
line (an item's text can itself start with a section header: see stale
macro blocks below). The SQL returns only that block (MEMORY_SLICE
characters from its header) and the live macro section (MACRO_SLICE
characters from the first "=== CURRENT MACRO ENVIRONMENT ===", which comes
before the memory block), never the rest of the prompt.
A verdict with no block is counted by why: "prompt NULL", "empty memory"
(the "No prior history found" block) or "no memory section" (the search
failed or the brief has no block).

Per item:
  - score: the similarity as printed (2 decimals; 1 - Chroma distance).
    "below 0" is a printed score with a minus sign, so "-0.00" counts.
  - collection: the header's label (NEWS, FILINGS, OPTIONS, INSIDER, MACRO,
    ARXIV, THESIS, ...). FILINGS items are also counted by the form named in
    the first 60 characters of the item text (8-K, 10-Q, ...; "other").
  - on company: the item text names the verdict's ticker or company by the
    rule ingestion.news.mentions_company applies to headlines (the company
    name or an alias as a whole word, $TICKER, (TICKER), "NYSE: TICKER",
    "TICKER stock/shares", or the bare uppercase ticker unless it is word-
    like). Its constants (aliases, overrides, word-like names and tickers,
    the name suffixes) are read from ingestion/news.py's source with ast, so
    nothing in ingestion is imported. Company names come from the ticker
    rows of knowledge/watchlist.md; a ticker with no row is matched on its
    ticker forms only. The rule looks for the name, not for what the item
    is about: an item that names the company in passing (an arXiv abstract
    naming NVIDIA hardware) counts as on company.

Per verdict: n items, its scores, n below 0, n on company and the collection
counts (the --json file holds these per verdict, with no text). Per cell
(day, held, path, ticker): verdicts, verdicts with items, items, the score
distribution over the items (n, min, p25, median, p75, max; p25/p75 by
statistics.quantiles, inclusive), items below 0 and on company (n and % of
items), the median over verdicts of each verdict's below-0 and on-company
fractions, and the collection counts. The collection tables give, per
collection, items, below 0, on company and the score distribution, for all
verdicts and per held label.

Held: provenance.plan.held as plans.observe recorded it (a stop row
existed): held, not held, or unknown when the verdict has no plan record.

Stale macro blocks (audit 2026-11-27 00:53 #3): tasks.poll_macro stores the
brief's macro text, its "=== CURRENT MACRO ENVIRONMENT ===" header
included, in the macro collection, and an older copy can come back as a
memory item. Per verdict: stale_macro_blocks, the MACRO items whose text
starts with that header; their ages in days, the verdict's New York date
minus the item's printed date (stale_macro_max_age_days, the largest; the
printed date of a macro item is the UTC date vector_store.store stamped
when poll_macro stored it); and,
per block whose text and the live macro section both print a "VIX
Volatility Index: N" value, the absolute difference in points (no level is
printed). Tables by held, day and path: verdicts, verdicts with a block,
blocks, the per-verdict max age and the per-block age and VIX-difference
distributions. Record only, no label.

No score floor, no ticker filter and no "irrelevant" label: those are
Luke's (audit 2026-11-26 22:53 #7). No prompt text and no price is printed.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the stack's start). db is imported from the working
directory when it holds a db.py (the live clone, with its .env), else from
beside this script; with no SUPABASE_HOST after the import the readout stops
before connecting. Imports: db and the standard library.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
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
DEFAULT_SINCE = "2026-11-20"
MEMORY_HEADER = "=== RELEVANT HISTORICAL CONTEXT (from memory) ==="
EMPTY_HEADER = "=== RELEVANT HISTORICAL CONTEXT ===\nNo prior history found"
MEMORY_SLICE = 12000    # characters from the header; 8 items of <= 400 chars fit with room
MACRO_HEADER = "=== CURRENT MACRO ENVIRONMENT ==="
MACRO_SLICE = 2000      # the live macro section, from its header
NO_SECTION = ("prompt NULL", "empty memory", "no memory section")
_SECTION = re.compile(r"^=== .+ ===$", re.M)
_ITEM = re.compile(r"^\[([A-Z0-9_]+) \| score=(-?\d+(?:\.\d+)?) \| ([^\]\n]*)\]$", re.M)
_VIX = re.compile(r"^VIX Volatility Index: (-?\d+(?:\.\d+)?)", re.M)
_FORM = re.compile(r"\b(8-K|10-Q|10-K|6-K|20-F|S-1|424B\d|DEF 14A|SC 13[DG]|Form 4)\b")
_WATCH_ROW = re.compile(r"^\|\s*([A-Z]{1,6})\s*\|\s*([^|]+?)\s*\|")
_NEWS_CONSTANTS = ("_NAME_ALIASES", "_NAME_OVERRIDES", "_WORDLIKE_NAMES", "_WORDLIKE_TICKERS")

_VERDICTS_SQL = """
SELECT s.id, s.timestamp, s.symbol,
       s.provenance->'trigger'->>'path',
       s.provenance->'plan'->>'held',
       s.prompt IS NULL,
       CASE WHEN strpos(s.prompt, %(hdr)s) > 0
            THEN substr(s.prompt, strpos(s.prompt, %(hdr)s), %(slice)s) END,
       strpos(s.prompt, %(empty)s) > 0,
       CASE WHEN strpos(s.prompt, %(mac)s) > 0
            THEN substr(s.prompt, strpos(s.prompt, %(mac)s), %(mac_slice)s) END
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


def news_constants(path: str | None = None) -> dict:
    """mentions_company's constants from ingestion/news.py's source, by ast
    (no import): the four literal tables and the _NAME_SUFFIXES pattern."""
    with open(path or os.path.join(ROOT, "ingestion", "news.py")) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if not (isinstance(node, (ast.Assign, ast.AnnAssign))):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            if t.id in _NEWS_CONSTANTS:
                out[t.id] = ast.literal_eval(node.value)
            elif t.id == "_NAME_SUFFIXES" and isinstance(node.value, ast.Call):
                out[t.id] = re.compile(ast.literal_eval(node.value.args[0]), re.I)
    missing = [k for k in _NEWS_CONSTANTS + ("_NAME_SUFFIXES",) if k not in out]
    if missing:
        raise ValueError(f"ingestion/news.py has no {', '.join(missing)}")
    return out


def _company_core(company: str, c: dict) -> str:
    core = (company or "").strip()
    while True:
        new = c["_NAME_SUFFIXES"].sub("", core).strip()
        if new == core:
            return core
        core = new


def _company_names(ticker: str, company: str, c: dict) -> list[str]:
    if ticker in c["_NAME_OVERRIDES"]:
        return c["_NAME_OVERRIDES"][ticker]
    core = _company_core(company, c)
    return ([core] if core else []) + c["_NAME_ALIASES"].get(ticker, [])


def on_company(ticker: str, company: str, text: str, c: dict) -> bool:
    """ingestion.news.mentions_company's rule with the constants in c. Pure."""
    text = text or ""
    for name in _company_names(ticker, company, c):
        flags = 0 if name in c["_WORDLIKE_NAMES"] else re.I
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text, flags):
            return True
    t = re.escape(ticker)
    if re.search(rf"(\${t}\b|\({t}\)|\b(?:NYSE|NASDAQ|Nasdaq):\s*{t}\b)", text):
        return True
    wordlike = c["_WORDLIKE_TICKERS"]
    if re.search(rf"\b{t}\s+(?:[Ss]tock|[Ss]hares)\b", text) and (
            ticker not in wordlike
            or (re.search(rf"\b{t}\s+Stock\b", text) and text != text.upper())):
        return True
    return ticker not in wordlike and bool(re.search(rf"\b{t}\b", text))


def company_names(path: str | None = None) -> dict[str, str]:
    """ticker -> company name, from the first ticker row of each ticker in
    knowledge/watchlist.md. {} when the file can't be read."""
    try:
        with open(path or os.path.join(ROOT, "knowledge", "watchlist.md")) as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        m = _WATCH_ROW.match(line)
        if m and m.group(1) != "TICKER":
            out.setdefault(m.group(1), m.group(2))
    return out


def parse_memory(block: str | None) -> list[dict]:
    """The items of a memory block (it starts at MEMORY_HEADER): [{collection,
    score, below0, date, text}], in order. The block ends at the next
    "=== NAME ===" line that does not directly follow an item header (that
    one is the item's own text). Pure."""
    if not block or not block.startswith(MEMORY_HEADER):
        return []
    body = block[len(MEMORY_HEADER):]
    for nxt in _SECTION.finditer(body):
        # The line right before it (no blank line between): an item header
        # means this header is that item's first text line.
        prev = body[:nxt.start()].removesuffix("\n").rsplit("\n", 1)[-1]
        if not _ITEM.fullmatch(prev):
            body = body[:nxt.start()]
            break
    heads = list(_ITEM.finditer(body))
    items = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        items.append({"collection": m.group(1), "score": float(m.group(2)),
                      "below0": m.group(2).startswith("-"), "date": m.group(3),
                      "text": body[m.end():end].strip()})
    return items


def section_vix(block: str | None):
    """The VIX value printed in a macro section (it starts at MACRO_HEADER
    and ends at the next "=== NAME ===" line), else None. Pure."""
    if not block or not block.startswith(MACRO_HEADER):
        return None
    body = block[len(MACRO_HEADER):]
    nxt = _SECTION.search(body)
    m = _VIX.search(body[:nxt.start()] if nxt else body)
    return float(m.group(1)) if m else None


def _age_days(day: str, printed: str):
    try:
        return (date.fromisoformat(day) - date.fromisoformat(printed.strip()[:10])).days
    except ValueError:
        return None


def stale_macro(items: list[dict], day: str, live_vix) -> dict:
    """The stale macro block fields for one verdict's items. Pure."""
    stale = [it for it in items if it["collection"] == "MACRO" and it["text"].startswith(MACRO_HEADER)]
    ages = [_age_days(day, it["date"]) for it in stale]
    diffs = []
    for it in stale:
        v = section_vix(it["text"])
        if v is not None and live_vix is not None:
            diffs.append(round(abs(v - live_vix), 2))
    known = [a for a in ages if a is not None]
    return {"stale_macro_blocks": len(stale), "stale_macro_ages_days": ages,
            "stale_macro_max_age_days": max(known) if known else None,
            "live_vix_seen": live_vix is not None, "stale_vix_abs_diff_pts": diffs}


def held_label(v) -> str:
    return {"true": "held", "false": "not held"}.get(str(v).lower() if v is not None else "", "unknown")


def classify(row, names: dict, consts: dict) -> dict:
    """One verdict row (as _VERDICTS_SQL returns it) as a dict of counts,
    with no item text."""
    sid, ts, ticker, path, held, prompt_null, block, empty = row[:8]
    macro = row[8] if len(row) > 8 else None
    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    ticker = ticker or ""
    company = names.get(ticker, "")
    items = parse_memory(block)
    if items:
        missing = None
    elif prompt_null:
        missing = "prompt NULL"
    elif empty:
        missing = "empty memory"
    else:
        missing = "no memory section"
    forms = Counter()
    for it in items:
        if it["collection"] == "FILINGS":
            f = _FORM.search(it["text"][:60])
            forms[f.group(1) if f else "other"] += 1
    day = ts.astimezone(NY).date().isoformat()
    return {"id": sid, "day": day, "ticker": ticker,
            "path": path or "none", "held": held_label(held), "has_company_name": bool(company),
            "missing": missing, "n": len(items),
            "scores": [it["score"] for it in items],
            "below0": [it["below0"] for it in items],
            "on_company": [on_company(ticker, company, it["text"], consts) for it in items],
            "collections": [it["collection"] for it in items],
            "filing_forms": dict(forms), **stale_macro(items, day, section_vix(macro))}


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) >= 2 else [v[0], v[0], v[0]]
    return {"n": len(v), "min": v[0], "p25": round(q[0], 2), "median": round(statistics.median(v), 2),
            "p75": round(q[2], 2), "max": v[-1]}


def _pct(n: int, d: int):
    return round(100 * n / d, 1) if d else None


def _median(values: list[float]):
    return round(statistics.median(values), 3) if values else None


def tally(rows: list[dict]) -> dict:
    """The column set for one cell."""
    with_items = [r for r in rows if r["n"]]
    scores = [s for r in rows for s in r["scores"]]
    below = sum(sum(r["below0"]) for r in rows)
    onco = sum(sum(r["on_company"]) for r in rows)
    coll = Counter(c for r in rows for c in r["collections"])
    forms = Counter()
    for r in rows:
        forms.update(r["filing_forms"])
    missing = Counter(r["missing"] for r in rows if r["missing"])
    return {"verdicts": len(rows), "with_items": len(with_items), "items": len(scores),
            "missing": {k: missing.get(k, 0) for k in NO_SECTION},
            "no_company_name": sum(1 for r in rows if not r["has_company_name"]),
            "score": _dist(scores),
            "below0": below, "below0_pct": _pct(below, len(scores)),
            "on_company": onco, "on_company_pct": _pct(onco, len(scores)),
            "verdict_below0_frac_median": _median([sum(r["below0"]) / r["n"] for r in with_items]),
            "verdict_on_company_frac_median": _median([sum(r["on_company"]) / r["n"]
                                                       for r in with_items]),
            "verdict_min_score_median": _median([min(r["scores"]) for r in with_items]),
            "collections": dict(coll.most_common()), "filing_forms": dict(forms.most_common())}


def collection_tally(rows: list[dict]) -> dict:
    """Per collection: items, below 0, on company and the score distribution."""
    by: dict[str, dict] = {}
    for r in rows:
        for c, s, b, o in zip(r["collections"], r["scores"], r["below0"], r["on_company"]):
            cell = by.setdefault(c, {"scores": [], "below0": 0, "on_company": 0})
            cell["scores"].append(s)
            cell["below0"] += b
            cell["on_company"] += o
    out = {}
    for c, cell in sorted(by.items(), key=lambda kv: (-len(kv[1]["scores"]), kv[0])):
        n = len(cell["scores"])
        out[c] = {"items": n, "below0": cell["below0"], "below0_pct": _pct(cell["below0"], n),
                  "on_company": cell["on_company"], "on_company_pct": _pct(cell["on_company"], n),
                  "off_company": n - cell["on_company"], "score": _dist(cell["scores"])}
    return out


def stale_tally(rows: list[dict]) -> dict:
    """The stale macro block columns for one cell."""
    with_any = [r for r in rows if r.get("stale_macro_blocks")]
    ages = [a for r in rows for a in r.get("stale_macro_ages_days", []) if a is not None]
    blocks = sum(r.get("stale_macro_blocks", 0) for r in rows)
    diffs = [d for r in rows for d in r.get("stale_vix_abs_diff_pts", [])]
    return {"verdicts": len(rows), "with_blocks": len(with_any),
            "with_blocks_pct": _pct(len(with_any), len(rows)), "blocks": blocks,
            "blocks_per_verdict_median": _median([r["stale_macro_blocks"] for r in with_any]),
            "max_age": _dist([r["stale_macro_max_age_days"] for r in with_any
                              if r["stale_macro_max_age_days"] is not None]),
            "age": _dist(ages), "vix_diff": _dist(diffs), "blocks_without_vix_diff": blocks - len(diffs)}


def build(rows: list[dict]) -> dict:
    def group(key):
        out = {}
        for r in rows:
            out.setdefault(key(r), []).append(r)
        return {k: tally(v) for k, v in sorted(out.items())}

    def sgroup(key):
        out = {}
        for r in rows:
            out.setdefault(key(r), []).append(r)
        return {k: stale_tally(v) for k, v in sorted(out.items())}
    held = {}
    for r in rows:
        held.setdefault(r["held"], []).append(r)
    return {"total": tally(rows), "by_day": group(lambda r: r["day"]),
            "by_held": group(lambda r: r["held"]),
            "by_held_day": group(lambda r: f"{r['held']} / {r['day']}"),
            "by_path": group(lambda r: r["path"]), "by_ticker": group(lambda r: r["ticker"]),
            "collections": collection_tally(rows),
            "collections_by_held": {k: collection_tally(v) for k, v in sorted(held.items())},
            "stale_total": stale_tally(rows),
            "stale_by_held": sgroup(lambda r: r["held"]),
            "stale_by_day": sgroup(lambda r: r["day"]),
            "stale_by_path": sgroup(lambda r: r["path"])}


def _c(v) -> str:
    return "-" if v is None else str(v)


def _d(d: dict) -> str:
    return f"{d['median']} [{d['p25']}, {d['p75']}], {d['min']}..{d['max']}" if d["n"] else "-"


def _counts(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "-"


def _table(title: str, label: str, cells: dict) -> list[str]:
    lines = ["", f"## {title}", "",
             f"| {label} | verdicts | with items | items | score: median [p25, p75], min..max | "
             "below 0 | below 0 % of items | on company | on company % of items | "
             "per-verdict below-0 frac, median | per-verdict on-company frac, median | "
             "per-verdict min score, median | collections | FILINGS forms |",
             "|---|" + "---|" * 13]
    for k, t in cells.items():
        lines.append(f"| {k} | {t['verdicts']} | {t['with_items']} | {t['items']} | {_d(t['score'])} | "
                     f"{t['below0']} | {_c(t['below0_pct'])} | {t['on_company']} | "
                     f"{_c(t['on_company_pct'])} | {_c(t['verdict_below0_frac_median'])} | "
                     f"{_c(t['verdict_on_company_frac_median'])} | {_c(t['verdict_min_score_median'])} | "
                     f"{_counts(t['collections'])} | {_counts(t['filing_forms'])} |")
    return lines


def _coll_table(title: str, cells: dict) -> list[str]:
    lines = ["", f"## {title}", "",
             "| collection | items | below 0 | below 0 % | on company | on company % | off company | "
             "score: median [p25, p75], min..max |",
             "|---|" + "---|" * 7]
    for c, t in cells.items():
        lines.append(f"| {c} | {t['items']} | {t['below0']} | {_c(t['below0_pct'])} | {t['on_company']} | "
                     f"{_c(t['on_company_pct'])} | {t['off_company']} | {_d(t['score'])} |")
    return lines


def _stale_table(title: str, label: str, cells: dict) -> list[str]:
    lines = ["", f"### {title}", "",
             f"| {label} | verdicts | with stale blocks | % of verdicts | stale blocks | "
             "blocks per verdict with any, median | max age days per verdict: median [p25, p75], "
             "min..max | block age days: median [p25, p75], min..max | VIX abs diff, points: "
             "median [p25, p75], min..max | blocks without a VIX diff |",
             "|---|" + "---|" * 9]
    for k, t in cells.items():
        lines.append(f"| {k} | {t['verdicts']} | {t['with_blocks']} | {_c(t['with_blocks_pct'])} | "
                     f"{t['blocks']} | {_c(t['blocks_per_verdict_median'])} | {_d(t['max_age'])} | "
                     f"{_d(t['age'])} | {_d(t['vix_diff'])} | {t['blocks_without_vix_diff']} |")
    return lines


def render_stale(out: dict) -> str:
    """The stale macro block section, appended after render()'s tables."""
    t = out["stale_total"]
    lines = ["", "## Stale macro blocks", "",
             f'MACRO memory items whose text starts with "{MACRO_HEADER}" (tasks.poll_macro stores '
             "the brief's macro text in the macro collection). Age = the verdict's New York date "
             "minus the item's printed date, in days. VIX abs diff = |the block's VIX - the live "
             "macro section's VIX|, in points; no level is printed. Record only.", "",
             f"- verdicts with a stale block: {t['with_blocks']} of {t['verdicts']}"
             + (f" ({t['with_blocks_pct']}%)" if t["with_blocks_pct"] is not None else "")
             + f"; stale blocks: {t['blocks']}; block age days: {_d(t['age'])}; "
             f"VIX abs diff, points: {_d(t['vix_diff'])}; blocks without a VIX diff: "
             f"{t['blocks_without_vix_diff']}"]
    lines += _stale_table("By held", "held", out["stale_by_held"])
    lines += _stale_table("By day", "day", out["stale_by_day"])
    lines += _stale_table("By trigger path", "path", out["stale_by_path"])
    return "\n".join(lines) + "\n"


def render(out: dict, env: str, since: date, until: date, head: str, now: datetime) -> str:
    t = out["total"]
    lines = [f"# Memory relevance readout ({env}, verdicts {since.isoformat()} to before "
             f"{until.isoformat()} ET)", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET. Code version: {head}. "
             "Engineering readout of the stored prompts' memory sections; counts, scores and "
             "collection names only. On company: the ingestion.news.mentions_company rule on the "
             "item text. No score floor, ticker filter or relevance label is applied (Luke's).", "",
             f"- verdicts: {t['verdicts']}; with memory items: {t['with_items']}; items: {t['items']}",
             "- without items: " + ", ".join(f"{k} {t['missing'][k]}" for k in NO_SECTION),
             f"- below 0: {t['below0']} ({_c(t['below0_pct'])}% of items); on company: "
             f"{t['on_company']} ({_c(t['on_company_pct'])}% of items)",
             f"- verdicts whose ticker has no watchlist name (ticker forms only): {t['no_company_name']}"]
    lines += _table("By held (provenance.plan.held)", "held", out["by_held"])
    lines += _table("By day (New York date of the verdict)", "day", out["by_day"])
    lines += _table("Held x day", "held / day", out["by_held_day"])
    lines += _coll_table("Collections, all verdicts", out["collections"])
    for h, cells in out["collections_by_held"].items():
        lines += _coll_table(f"Collections, {h} verdicts", cells)
    lines += _table("By trigger path", "path", out["by_path"])
    lines += _table("By ticker", "ticker", out["by_ticker"])
    return "\n".join(lines) + "\n"


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD, New York date")
    ap.add_argument("--until", help="YYYY-MM-DD, New York date, exclusive; default tomorrow")
    ap.add_argument("--env", default="live")
    ap.add_argument("--watchlist", help="watchlist.md for company names; default this checkout's")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--json", help="also write the per-verdict records (no text) here")
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
    consts = news_constants()
    names = company_names(args.watchlist)

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
                                        "hdr": MEMORY_HEADER, "empty": EMPTY_HEADER,
                                        "slice": MEMORY_SLICE, "mac": MACRO_HEADER,
                                        "mac_slice": MACRO_SLICE})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = [classify(r, names, consts) for r in raw]
    out = build(rows)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(out, args.env, since, until, head or "unknown", now) + render_stale(out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"generated": now.isoformat(), "code_version": head or "unknown",
                       "env": args.env, "since": since.isoformat(), "until": until.isoformat(),
                       "verdicts": rows}, f, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
