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

    env -C ~/Desktop/trading_bot venv/bin/python scripts/prompt_ab.py [--n 12] [--k 3] [--out FILE]

Prints and optionally writes ticker, variant, run, action, conviction. No
portfolio values: prompts are read from the DB and never written out.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
_START = "Then output ONLY valid JSON"
_END = "\nRules:"
OLD_COMMIT = "31d52f8^"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    import db
    from agents import analyst

    old_src = subprocess.run(["git", "-C", REPO, "show", f"{OLD_COMMIT}:agents/analyst.py"],
                             capture_output=True, text=True, check=True).stdout
    with open(os.path.join(REPO, "agents", "analyst.py")) as f:
        new_block = schema_block(f.read())
    old_block = schema_block(old_src)

    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, prompt FROM signals WHERE prompt IS NOT NULL "
                        "ORDER BY timestamp DESC LIMIT %s", (args.n,))
            rows = cur.fetchall()
    finally:
        db._put_conn(conn)
    if not rows:
        sys.exit("no stored prompts yet (signals.prompt is recorded from 10-28 16:05)")

    lines = ["ticker,variant,run,action,conviction"]
    conv = defaultdict(list)
    for ticker, prompt in rows:
        try:
            variants = {"new": prompt, "old": old_variant(prompt, new_block, old_block)}
        except ValueError as e:
            print(f"{ticker}: skipped ({e})")
            continue
        for name, p in variants.items():
            for run in range(args.k):
                raw = analyst._call_ollama(p)
                parsed = analyst._extract_json(raw) if raw else None
                action = str((parsed or {}).get("action", "ERR")).upper()
                c = (parsed or {}).get("conviction")
                lines.append(f"{ticker},{name},{run},{action},{c}")
                if isinstance(c, (int, float)):
                    conv[name].append(int(c))
                print(lines[-1], flush=True)
    summary = {name: dict(sorted(Counter(v).items())) for name, v in conv.items()}
    print("conviction counts:", summary)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
            f.write(f"# conviction counts: {summary}\n")


if __name__ == "__main__":
    main()
