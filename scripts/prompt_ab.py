"""Old vs new analyst output schema on the same stored prompts.

31d52f8 moved the verdict fields (action, conviction, scores) to the end of
the JSON schema so the model reasons first. Since the 10-28 05:00 start the
conviction distribution shifted (17 verdicts: 0 at c7 vs 34% before), but the
brief cap (b961c38) went live at the same time. This replays stored prompts
(signals.prompt, recorded from 10-28 16:05) twice: as stored (new schema) and
with the schema block swapped for the pre-31d52f8 one taken from git, k times
each, with the live Ollama options. Only the schema block differs, so this
isolates the reorder. Also gives test-retest spread (limitations audit 09:00
#3, professor C11).

    env -C ~/Desktop/trading_bot venv/bin/python scripts/prompt_ab.py [--n 24] [--k 3] [--out FILE]

--n is the number of stored prompts: the newest per ticker, then the
second newest, and so on round-robin (15 tickers had stored prompts on
10-30). Results pair by prompt. The two variants alternate run by run, and
which goes first flips by prompt. Each run also records the logprob
expected conviction. The summary applies the conviction-gate rule in the
research log of 10-30 07:02 (old c7+ share at least 15 points higher AND
paired mean expected conviction at least 0.5 higher). That rule was set
after the 10-28 6-prompt pilot (prompt-ab-pilot-2026-10-28.md), so the
pilot's prompts are excluded by default (--include-pilot keeps them).
Runs with no parseable conviction, a regex-salvaged reply or no logprobs
are counted per arm and left out of the rule.

The new arm tests 31d52f8 as shipped: the reorder plus its 3-line "fill the
fields in the order shown" instruction, not the reorder alone.

Refuses to run while helios-local is active (the 72B is shared with the
live analyst) unless --force. Prints and, with --out, writes row by row
(flushed) prompt_id, signal_id (signals.id of the stored prompt), ticker,
variant, run, action, conviction, salvaged, expected, p_ge7, plus
OUT.provenance.json (commits, schema-block hashes, model, sample, times) and
OUT.old_schema.txt (the old block, so a rerun works without git history:
--old-schema FILE). No portfolio values: prompts are read from the DB and
never written out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
_START = "Then output ONLY valid JSON"
_END = "\nRules:"
OLD_COMMIT = "31d52f8^"
RULE_SHARE_PTS = 15.0     # research log 10-30 07:02 (set after the 10-28 pilot)
RULE_EXPECTED = 0.5
# The 10-28 pilot's prompts (TER, META, MSFT x2, KLAC, ORCL; checked against
# the DB 10-30): the rule was set after seeing them.
PILOT_SIGNAL_IDS = frozenset({79199, 79189, 79173, 79172, 79150, 79137})


def schema_block(source: str) -> str:
    """The schema text from _START up to the Rules section, as rendered
    (format() turns {{ }} into { })."""
    i = source.index(_START)
    j = source.index(_END, i)
    return source[i:j].replace("{{", "{").replace("}}", "}")


def old_variant(prompt: str, new_block: str, old_block: str) -> str:
    if new_block not in prompt:
        raise ValueError("stored prompt does not contain the current schema block")
    return prompt.replace(new_block, old_block, 1)


def summarize(runs: list[dict]) -> dict:
    """Per-variant c7+ share and mean expected conviction, the paired
    (by prompt) difference old - new, and the pre-set rule's verdict."""
    by = defaultdict(lambda: defaultdict(list))   # variant -> prompt -> rows
    for r in runs:
        by[r["variant"]][r.get("prompt_id", r["ticker"])].append(r)
    out = {}
    for v in ("new", "old"):
        rows = [r for t in by[v].values() for r in t]
        convs = [r["conviction"] for r in rows if r["conviction"] is not None and not r.get("salvaged")]
        out[v] = {"runs": len(convs), "counts": dict(sorted(Counter(convs).items())),
                  "no_conviction": sum(r["conviction"] is None for r in rows),
                  "salvaged": sum(bool(r.get("salvaged")) for r in rows),
                  "no_logprobs": sum(r["expected"] is None for r in rows),
                  "c7_share_pct": round(100 * sum(c >= 7 for c in convs) / len(convs), 1) if convs else None}
    diffs = []
    for t in set(by["new"]) & set(by["old"]):
        e = {v: [r["expected"] for r in by[v][t] if r["expected"] is not None and not r.get("salvaged")]
             for v in ("new", "old")}
        if e["new"] and e["old"]:
            diffs.append(sum(e["old"]) / len(e["old"]) - sum(e["new"]) / len(e["new"]))
    out["paired_prompts"] = len(diffs)
    out["paired_mean_expected_diff"] = round(sum(diffs) / len(diffs), 3) if diffs else None
    share_diff = (out["old"]["c7_share_pct"] - out["new"]["c7_share_pct"]
                  if out["old"]["c7_share_pct"] is not None and out["new"]["c7_share_pct"] is not None else None)
    out["c7_share_diff_pts"] = round(share_diff, 1) if share_diff is not None else None
    out["order_effect"] = (share_diff is not None and out["paired_mean_expected_diff"] is not None
                           and share_diff >= RULE_SHARE_PTS
                           and out["paired_mean_expected_diff"] >= RULE_EXPECTED)
    return out


def _round_robin(rows: list[tuple], n: int) -> list[tuple]:
    """rows: (symbol, prompt, timestamp, ...). Newest per ticker first (tickers
    by their newest prompt), then each ticker's next newest, until n. Returns
    (symbol, prompt, *rest) with the timestamp dropped."""
    per = defaultdict(list)
    for r in sorted(rows, key=lambda r: r[2], reverse=True):
        per[r[0]].append(r)
    tickers = list(per)          # dicts keep insertion order: newest first
    out, depth = [], 0
    while len(out) < n and any(len(per[t]) > depth for t in tickers):
        out += [(per[t][depth][0], per[t][depth][1], *per[t][depth][3:])
                for t in tickers if len(per[t]) > depth]
        depth += 1
    return out[:n]


def _conviction(v) -> int | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return int(v) if math.isfinite(v) else None


def _stored_prompts(n: int, exclude: frozenset = frozenset()) -> list[tuple]:
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            # Same environment only (professor C13, 2026-10-28).
            cur.execute("SELECT symbol, prompt, timestamp, id FROM signals "
                        "WHERE prompt IS NOT NULL AND trading_env=%s", (db.TRADING_ENV,))
            rows = [r for r in cur.fetchall() if r[3] not in exclude]
    finally:
        db._put_conn(conn)
    return _round_robin(rows, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-pilot", action="store_true")
    ap.add_argument("--old-schema", default=None, help="file with the old schema block (instead of git)")
    args = ap.parse_args()

    live = subprocess.run(["systemctl", "is-active", "helios-local"], capture_output=True, text=True).stdout.strip()
    if live == "active" and not args.force:
        sys.exit("helios-local is running; the A/B shares its 72B. Run it while the bot is stopped (or --force).")

    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    from agents import analyst

    if args.old_schema:
        with open(args.old_schema) as f:
            old_block, old_commit = f.read(), f"file {args.old_schema}"
    else:
        old_src = subprocess.run(["git", "-C", REPO, "show", f"{OLD_COMMIT}:agents/analyst.py"],
                                 capture_output=True, text=True, check=True).stdout
        old_block = schema_block(old_src)
        old_commit = subprocess.run(["git", "-C", REPO, "rev-parse", OLD_COMMIT],
                                    capture_output=True, text=True).stdout.strip()
    with open(os.path.join(REPO, "agents", "analyst.py")) as f:
        new_block = schema_block(f.read())

    exclude = frozenset() if args.include_pilot else PILOT_SIGNAL_IDS
    rows = _stored_prompts(args.n, exclude)
    if not rows:
        sys.exit("no stored prompts yet (signals.prompt is recorded from 10-28 16:05)")

    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()[:12]
    meta = {"started": datetime.now(timezone.utc).isoformat(), "n": args.n, "k": args.k,
            "head": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
            "old_schema_from": old_commit, "old_block_sha": sha(old_block), "new_block_sha": sha(new_block),
            "model": analyst.ANALYST_MODEL, "excluded_signal_ids": sorted(exclude),
            "signal_ids": [r[2] for r in rows],
            "rule": {"share_pts": RULE_SHARE_PTS, "expected": RULE_EXPECTED}}
    out = None
    if args.out:
        with open(args.out + ".old_schema.txt", "w") as f:
            f.write(old_block)
        out = open(args.out, "w")

    header = "prompt_id,signal_id,ticker,variant,run,action,conviction,salvaged,expected,p_ge7"
    lines, runs = [header], []
    print(header, flush=True)
    if out:
        out.write(header + "\n")
        out.flush()
    for i, (ticker, prompt, signal_id) in enumerate(rows):
        try:
            variants = {"new": prompt, "old": old_variant(prompt, new_block, old_block)}
        except ValueError as e:
            print(f"{ticker}: skipped ({e})")
            continue
        order = ("new", "old") if i % 2 == 0 else ("old", "new")
        for run in range(args.k):
            for name in order:
                raw = analyst._call_ollama(variants[name])
                parsed = analyst._extract_json(raw) if raw else None
                dist = analyst._LAST_CALL_STATS.get("conviction_logprobs") or {}
                action = str((parsed or {}).get("action", "ERR")).upper()
                c = _conviction((parsed or {}).get("conviction"))
                salvaged = bool((parsed or {}).get("_salvaged"))
                runs.append({"ticker": ticker, "prompt_id": i, "variant": name, "conviction": c,
                             "salvaged": salvaged, "expected": dist.get("expected")})
                lines.append(f"{i},{signal_id},{ticker},{name},{run},{action},{c},{int(salvaged)},"
                             f"{dist.get('expected')},{dist.get('p_ge7')}")
                print(lines[-1], flush=True)
                if out:
                    out.write(lines[-1] + "\n")
                    out.flush()
    summary = summarize(runs)
    print("summary:", summary)
    if out:
        out.write(f"# summary: {json.dumps(summary)}\n")
        out.close()
        meta["finished"] = datetime.now(timezone.utc).isoformat()
        meta["summary"] = summary
        with open(args.out + ".provenance.json", "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
