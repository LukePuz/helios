"""Check that every run of the prompt-order A/B used the same configuration.

The runner appends a {"kind": "run"} line (config only, no arm data) each time
it starts, and a resume on a later night appends another. The report compares
them, but under the blinding rule the report runs only after the last night,
so an Ollama upgrade or model re-pull between nights would surface only then.

This reads the run lines only (pair and skip lines are never parsed or
printed) and compares each with the first, on the keys the report compares
(prompt_order_ab_report.run_config). With --preflight it also builds the
config a run started now would record (live call settings, model digest,
Ollama version, schema hashes, tool_sha) and compares that with the first run
line; settings that come from the command line are taken from the first run
line and not checked. Run it from the checkout the runner will use.

    venv/bin/python scripts/g33_resume_check.py <out.jsonl> [--preflight]

Prints only the names of keys that differ and the config values involved.
Exit 0: all match. 2: a mismatch. 1: no run line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompt_order_ab_report import run_config  # noqa: E402

# Set on the runner's command line, so a preflight can't know them.
CLI_KEYS = ("seed", "timeout_s", "arms", "arm_a_num_predict", "repeat_arm", "repeat_subsample")


def run_lines(path: str) -> list[dict]:
    """The {"kind": "run"} records, in file order. Other lines are skipped
    without being decoded further than their kind."""
    runs = []
    with open(path) as f:
        for line in f:
            if '"kind": "run"' not in line and '"kind":"run"' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "run":
                runs.append(rec)
    return runs


def diff(first: dict, other: dict) -> dict:
    """{key: (first value, other value)} for each run_config key that differs."""
    a, b = run_config(first), run_config(other)
    return {k: (a.get(k), b.get(k)) for k in a.keys() | b.keys()
            if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True)}


def current_run(first: dict) -> dict:
    """The config a run started now would record. Command-line keys are
    copied from the first run line."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    import prompt_order_ab as ab
    from agents import analyst
    live = ab.live_payload()
    ident = analyst._ollama_identity()
    blocks = ab.load_blocks()
    run = {k: first.get(k) for k in CLI_KEYS}
    run.update({"model": live["payload"]["model"], "options": live["payload"]["options"],
                "live_timeout_s": live["timeout"], "model_digest": ident.get("model_digest"),
                "ollama_version": ident.get("ollama_version"),
                "schema_sha_A": ab.sha12(blocks[0]), "schema_sha_B": ab.sha12(blocks[1]),
                "schema_sha_C": ab.sha12(ab.block_c(*blocks)) if "C" in (first.get("arms") or "") else None,
                "tool_sha": ab.tool_sha()})
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preflight", action="store_true",
                    help="also compare the config a run started now would record")
    a = ap.parse_args(argv)
    runs = run_lines(a.out)
    if not runs:
        print("no run line")
        return 1
    bad = False
    for i, r in enumerate(runs[1:], start=2):
        d = diff(runs[0], r)
        if d:
            bad = True
            print(f"run line {i} ({r.get('started_at')}) differs from run line 1: " +
                  "; ".join(f"{k}: {v[0]!r} -> {v[1]!r}" for k, v in sorted(d.items())))
    if a.preflight:
        d = diff(runs[0], current_run(runs[0]))
        if d:
            bad = True
            print("a run started now would differ from run line 1: " +
                  "; ".join(f"{k}: {v[0]!r} -> {v[1]!r}" for k, v in sorted(d.items())))
    print(f"{len(runs)} run line(s); " + ("MISMATCH" if bad else "config matches run line 1"))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
