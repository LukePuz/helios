#!/usr/bin/env python3
"""AI prompt log capture for ISEF / Regeneron STS (Form 2A "AI Prompt Log").

Called by the Claude Code hooks in .claude/settings.json. Standard library only
and cross-platform, so it runs the same on this Linux box and on the Windows PC.

  prompt   (UserPromptSubmit) append {ts, session_id, prompt} verbatim to
           research-log/raw-prompts.jsonl
  stop     (Stop)             refresh a working copy of the session transcript
                              in the archive (no manifest row: one per reply
                              would rewrite the committed manifest constantly)
  end      (SessionEnd)       freeze an immutable snapshot of the transcript
                              and append its SHA-256 to the committed
                              research-log/transcripts-manifest.csv
  sweep    (SessionStart, and by hand for the backfill) snapshot + manifest
           every transcript for this project that has changed since it was
           last recorded. Catches sessions that never got a SessionEnd
           (killed, crashed) and archives pre-existing history.

Transcripts never enter git: they can contain keys or account data. Only their
hashes do, which proves the archived files existed unchanged at that time.

Snapshots are content-addressed (<name>.<sha12>.jsonl) and never overwritten.
A resumed session keeps appending to the same transcript, so overwriting one
archive copy would orphan every earlier manifest hash.

A hook must never break the session it runs in: every mode swallows its own
errors, logs them to the archive's errors.log, prints nothing, and exits 0.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(os.environ.get("HELIOS_AI_LOG_DIR") or Path.home() / "helios-ai-log")
MANIFEST_FIELDS = ["recorded_utc", "event", "session_id", "transcript",
                   "bytes", "sha256", "archive_file"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_dir(event: dict) -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or os.getcwd())


def _transcripts_dir(event: dict, project: Path) -> Path:
    """~/.claude/projects/<key>/ for this repo. Taken from the hook's own
    transcript_path when there is one; otherwise derived the way Claude Code
    names it (every non-alphanumeric character of the path becomes '-')."""
    tp = event.get("transcript_path")
    if tp:
        return Path(tp).expanduser().parent
    key = re.sub(r"[^A-Za-z0-9]", "-", str(project.resolve()))
    return Path.home() / ".claude" / "projects" / key


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(transcript: Path, root: Path) -> str:
    try:
        return transcript.relative_to(root).as_posix()
    except ValueError:
        return transcript.name


def _snapshot(transcript: Path, root: Path) -> tuple[str, int, str]:
    """Copy to an immutable, content-addressed archive file.
    Returns (sha256, bytes, archive path relative to the archive root)."""
    digest = _sha256(transcript)
    rel = Path(_rel(transcript, root))
    dest = ARCHIVE / "transcripts" / rel.parent / f"{rel.stem}.{digest[:12]}{rel.suffix}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(transcript, tmp)
        if _sha256(tmp) != digest:      # transcript grew mid-copy; hash what we kept
            digest = _sha256(tmp)
            dest = dest.with_name(f"{rel.stem}.{digest[:12]}{rel.suffix}")
        os.replace(tmp, dest)
    return digest, dest.stat().st_size, dest.relative_to(ARCHIVE).as_posix()


def _manifest_path(project: Path) -> Path:
    return project / "research-log" / "transcripts-manifest.csv"


def _recorded_hashes(manifest: Path) -> set[tuple[str, str]]:
    if not manifest.exists():
        return set()
    with open(manifest, newline="", encoding="utf-8") as f:
        return {(r["transcript"], r["sha256"]) for r in csv.DictReader(f)}


def _record(project: Path, root: Path, transcript: Path, event_name: str, session_id: str) -> bool:
    """Snapshot + manifest row, skipped when this exact content is already recorded."""
    manifest = _manifest_path(project)
    digest, size, archived = _snapshot(transcript, root)
    rel = _rel(transcript, root)
    if (rel, digest) in _recorded_hashes(manifest):
        return False
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({"recorded_utc": _now(), "event": event_name, "session_id": session_id,
                    "transcript": rel, "bytes": size, "sha256": digest, "archive_file": archived})
    return True


def cmd_prompt(event: dict) -> None:
    out = _project_dir(event) / "research-log" / "raw-prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": _now(), "session_id": event.get("session_id", ""), "prompt": event.get("prompt", "")}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_stop(event: dict) -> None:
    tp = event.get("transcript_path")
    if not tp or not Path(tp).expanduser().is_file():
        return
    transcript = Path(tp).expanduser()
    root = _transcripts_dir(event, _project_dir(event))
    dest = ARCHIVE / "working" / f"{Path(_rel(transcript, root)).stem}.latest.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcript, dest)


def cmd_end(event: dict) -> None:
    tp = event.get("transcript_path")
    if not tp or not Path(tp).expanduser().is_file():
        return
    project = _project_dir(event)
    _record(project, _transcripts_dir(event, project), Path(tp).expanduser(),
            "session_end", event.get("session_id", ""))


def cmd_sweep(event: dict, event_name: str = "sweep") -> int:
    project = _project_dir(event)
    root = _transcripts_dir(event, project)
    if not root.is_dir():
        return 0
    added = 0
    for transcript in sorted(root.rglob("*.jsonl")):
        if "memory" in transcript.relative_to(root).parts:
            continue
        sid = transcript.relative_to(root).parts[0].removesuffix(".jsonl")
        added += _record(project, root, transcript, event_name, sid)
    return added


def _log_error(mode: str, err: BaseException) -> None:
    try:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        with open(ARCHIVE / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"{_now()} {mode}: {type(err).__name__}: {err}\n")
    except Exception:
        pass


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        event = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except Exception:
        event = {}
    try:
        # Owner-only: archived transcripts can contain keys or account data.
        ARCHIVE.mkdir(mode=0o700, parents=True, exist_ok=True)
        if mode == "prompt":
            cmd_prompt(event)
        elif mode == "stop":
            cmd_stop(event)
        elif mode == "end":
            cmd_end(event)
        elif mode in ("sweep", "backfill"):
            n = cmd_sweep(event, "backfill" if mode == "backfill" else "session_start")
            if mode == "backfill":          # interactive use only; hooks must stay silent
                print(f"recorded {n} transcript snapshot(s) in {_manifest_path(_project_dir(event))}")
    except Exception as e:  # noqa: BLE001 -- see module docstring
        _log_error(mode, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
