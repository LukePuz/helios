"""Map commits that carry the same change under different hashes.

A change cherry-picked onto a deploy branch runs under a new hash while the
research log may cite the original (professor m490). `git patch-id --stable`
gives the same id to the same diff, so this lists every patch id that has more
than one commit, one row per commit, and marks whether that commit is an
ancestor of the live checkout's HEAD at the moment the map is generated
(professor m496). That is not a record of which hash ran at any earlier time.
The commit column holds full 40-character hashes. The generation time and the
full live HEAD go in a sidecar file next to the CSV (<out stem>.meta.txt,
e.g. patch-id-map.meta.txt), so the CSV stays a plain header-plus-rows file.

    python scripts/patch_id_map.py [--since 2026-10-15] [--out research-log/patch-id-map.csv]

Read-only on both repositories.
"""
import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIVE = Path.home() / "Desktop" / "trading_bot"


def _git(repo: Path, *args: str, input_text: str | None = None, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], input=input_text, capture_output=True,
                          text=True, check=check).stdout


def patch_ids(since: str) -> list[tuple[str, str]]:
    """(patch_id, commit) for every non-merge commit on any ref since `since`."""
    log = _git(REPO, "log", "--all", "--no-merges", f"--since={since}", "-p", "--format=commit %H")
    out = _git(REPO, "patch-id", "--stable", input_text=log)
    return [tuple(line.split()) for line in out.splitlines() if line.strip()]


def in_live(commit: str) -> bool:
    """True when the commit is an ancestor of the live checkout's HEAD."""
    if not (LIVE / ".git").exists():
        return False
    r = subprocess.run(["git", "-C", str(LIVE), "merge-base", "--is-ancestor", commit, "HEAD"],
                       capture_output=True)
    return r.returncode == 0


def live_head() -> str | None:
    """Full hash of the live checkout's HEAD, or None if there is no live checkout."""
    if not (LIVE / ".git").exists():
        return None
    r = subprocess.run(["git", "-C", str(LIVE), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-15")
    ap.add_argument("--out", default=str(REPO / "research-log" / "patch-id-map.csv"))
    a = ap.parse_args(argv)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = live_head()
    groups: dict[str, list[str]] = defaultdict(list)
    for pid, commit in patch_ids(a.since):
        if commit not in groups[pid]:
            groups[pid].append(commit)
    rows = []
    for pid, commits in groups.items():
        if len(commits) < 2:
            continue
        for c in commits:
            date, subject = _git(REPO, "log", "-1", "--format=%cI%x09%s", c).rstrip("\n").split("\t", 1)
            rows.append({"patch_id": pid[:12], "commit": c, "committed": date,
                         "in_live": "yes" if in_live(c) else "no", "subject": subject[:120]})
    rows.sort(key=lambda r: (r["patch_id"], r["committed"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch_id", "commit", "committed", "in_live", "subject"])
        w.writeheader()
        w.writerows(rows)
    meta = Path(a.out).with_suffix(".meta.txt")
    meta.write_text(
        f"file: {Path(a.out).name}\n"
        f"generated_utc: {generated}\n"
        f"live_checkout: {LIVE}\n"
        f"live_head: {head or 'none (no live checkout)'}\n"
        f"since: {a.since}\n"
        "in_live: yes = the commit is an ancestor of live_head above, at generated_utc.\n"
        "It does not say which hash was running at any earlier time.\n")
    print(f"{len({r['patch_id'] for r in rows})} changes under more than one hash, {len(rows)} rows -> {a.out} (+ {meta.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
