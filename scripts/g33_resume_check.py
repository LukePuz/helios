"""Check that every run of the prompt-order A/B used the same configuration.

The runner appends a {"kind": "run"} line (config only, no arm data) each time
it starts, and a resume on a later night appends another. The report compares
them, but under the blinding rule the report runs only after the last night,
so an Ollama upgrade or model re-pull between nights would surface only then.

This reads the run lines and compares each with the first, on the keys the
report compares (prompt_order_ab_report.run_config) plus the sample (signal
ids, repeat ids, sample frame and population size), logprob settings and the
full live options (professor M291: the runner draws the sample again every
night). From pair lines it reads only signal_id and sha_stored, never the
replies. With --preflight it also builds the config a run started now would
record (live call settings, model digest, Ollama version, schema hashes,
tool_sha), redraws the sample read-only from the first run line's frame and
seed, and hashes the stored prompts of the sample against the sha_stored the
earlier nights recorded. Settings that come from the command line are taken
from the first run line and not checked. Run it from the checkout the runner
will use; the repo path and HEAD are printed.

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


# Recorded on the run line but not in run_config.
EXTRA_KEYS = ("signal_ids", "repeat_ids", "sample", "logprobs", "top_logprobs", "options")


def _config(run: dict) -> dict:
    cfg = run_config(run)
    for k in EXTRA_KEYS:
        v = run.get(k)
        cfg[k] = sorted(v) if k == "repeat_ids" and isinstance(v, list) else v
    return cfg


def stored_shas(path: str) -> dict:
    """{signal_id: sha_stored} from pair lines. Only those two keys are kept."""
    out = {}
    with open(path) as f:
        for line in f:
            if '"kind": "pair"' not in line and '"kind":"pair"' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "pair" and rec.get("sha_stored"):
                out[rec.get("signal_id")] = rec["sha_stored"]
    return out


def diff(first: dict, other: dict) -> dict:
    """{key: (first value, other value)} for each compared key that differs."""
    a, b = _config(first), _config(other)
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
    frame = first.get("sample") or {}
    if frame.get("mode") == "range":
        pop = ab.population_ids(ab._et(frame["from"]), ab._et(frame["to"]), frame["trading_env"])
        ids = ab.draw_sample(pop, frame["n"], first.get("seed"))
        run["signal_ids"] = ids
        run["sample"] = {"mode": "range", "from": frame["from"], "to": frame["to"],
                         "trading_env": frame["trading_env"], "population": len(pop), "n": frame["n"]}
        run["repeat_ids"] = sorted(ab.repeat_ids_for(ids, first.get("repeat_arm"),
                                                     first.get("repeat_subsample"), first.get("seed")))
    else:   # an ids file: the ids are what the first run recorded
        run.update({k: first.get(k) for k in ("signal_ids", "sample", "repeat_ids")})
    run.update({"model": live["payload"]["model"], "options": live["payload"]["options"],
                "logprobs": live["payload"].get("logprobs"),
                "top_logprobs": live["payload"].get("top_logprobs"),
                "live_timeout_s": live["timeout"], "model_digest": ident.get("model_digest"),
                "ollama_version": ident.get("ollama_version"),
                "schema_sha_A": ab.sha12(blocks[0]), "schema_sha_B": ab.sha12(blocks[1]),
                "schema_sha_C": ab.sha12(ab.block_c(*blocks)) if "C" in (first.get("arms") or "") else None,
                "tool_sha": ab.tool_sha()})
    return run


def prompt_changes(path: str, ids: list) -> list:
    """Signal ids whose stored prompt now hashes differently from the
    sha_stored an earlier night recorded (redact_stored_prompts.py can
    rewrite them). Ids not yet run are not checked."""
    import prompt_order_ab as ab
    seen = stored_shas(path)
    todo = [i for i in ids if i in seen]
    if not todo:
        return []
    rows = ab.fetch_signals(todo)
    return [i for i in todo if ab.sha12((rows.get(i) or {}).get("prompt") or "") != seen[i]]


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
        import subprocess
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
        print(f"checkout {REPO} HEAD {head}")
        now_run = current_run(runs[0])
        d = diff(runs[0], now_run)
        if d:
            bad = True
            print("a run started now would differ from run line 1: " +
                  "; ".join(f"{k}: {v[0]!r} -> {v[1]!r}" for k, v in sorted(d.items())))
        changed = prompt_changes(a.out, now_run.get("signal_ids") or [])
        if changed:
            bad = True
            print(f"stored prompt changed since it was run for signal ids {sorted(changed)}")
    print(f"{len(runs)} run line(s); " + ("MISMATCH" if bad else "config matches run line 1"))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
