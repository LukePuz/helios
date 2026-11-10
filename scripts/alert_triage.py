#!/usr/bin/env python3
"""
Standing check on the Telegram alert outbox, for a 30-minute cron.

Reads the rows logs/telegram_outbox.jsonl gained since the last run (alerts.py,
deploy/watchdog.py and deploy/notify_failure.py append one per alert, sent or
held), groups repeats, and lists alert types never seen before. Read-only
apart from its cursor file, logs/.alert_triage_cursor.json, which holds the
read offset and the set of known types.

    venv/bin/python scripts/alert_triage.py              # new rows since last run
    venv/bin/python scripts/alert_triage.py --no-advance # look without moving the cursor
    venv/bin/python scripts/alert_triage.py --from-start # reread the whole outbox
    venv/bin/python scripts/alert_triage.py --baseline   # mark what is there as known, exit 0

Exit status is 0, or 2 when a new critical type appeared, so a cron wrapper
can escalate on it.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

DEFAULT_LOGS = Path(__file__).resolve().parent.parent / "logs"
OUTBOX_NAME = "telegram_outbox.jsonl"
CURSOR_NAME = ".alert_triage_cursor.json"
MAX_KNOWN = 5000


def type_key(row: dict) -> str:
    """Low and report alerts are keyed by their policy type, critical ones
    by fingerprint (numbers collapsed)."""
    return f"{row.get('severity', '?')}:{row.get('type') or row.get('fingerprint') or row.get('title', '')}"


def load_cursor(path: Path) -> dict:
    try:
        cur = json.loads(path.read_text())
        if isinstance(cur, dict):
            cur.setdefault("known", {})
            return cur
    except Exception:
        pass
    return {"inode": None, "offset": 0, "known": {}}


def save_cursor(path: Path, cursor: dict) -> None:
    known = cursor.get("known", {})
    if len(known) > MAX_KNOWN:
        keep = sorted(known.items(), key=lambda kv: kv[1])[-MAX_KNOWN:]
        cursor["known"] = dict(keep)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cursor))
    os.replace(tmp, path)


def _head(path: Path) -> str | None:
    """Fingerprint of the file's first complete line (its first 4 KB when
    the line is longer), or None while that row is still being written. Tells a recreated file from the old one when the filesystem hands
    the freed inode straight back."""
    try:
        with open(path, "rb") as f:
            first = f.readline(4096)
    except OSError:
        return None
    if not first.endswith(b"\n") and len(first) < 4096:
        return None                                  # first row still being written
    return hashlib.sha1(first).hexdigest()


def read_new_rows(outbox: Path, cursor: dict,
                  from_start: bool = False) -> tuple[list[dict], int, int | None, str | None]:
    """(rows, new_offset, inode, head), all describing the same file. Starts over when the file was rotated
    (other inode, or the same inode with a different first line) or truncated
    (shorter than the saved offset). Stops at the last complete line so a row
    being written is read next time."""
    try:
        st = outbox.stat()
    except FileNotFoundError:
        return [], 0, None, None
    offset = 0 if from_start else int(cursor.get("offset") or 0)
    rows: list[dict] = []
    saved_head = cursor.get("head")
    head = _head(outbox)
    if saved_head and head and head != saved_head and cursor.get("inode") == st.st_ino:
        offset = 0                                   # recreated on a reused inode
    elif cursor.get("inode") not in (None, st.st_ino):
        # Rotated (alert_policy keeps one old file as .1): finish the old
        # file from the saved offset, then read the new one from the start.
        old = outbox.with_name(outbox.name + ".1")
        try:
            if not from_start and old.stat().st_ino == cursor.get("inode"):
                rows += _parse(_read_from(old, offset)[0])
        except FileNotFoundError:
            pass
        offset = 0
    elif st.st_size < offset:
        offset = 0
    data, end = _read_from(outbox, offset)
    rows += _parse(data)
    return rows, offset + end, st.st_ino, head


def _read_from(path: Path, offset: int) -> tuple[bytes, int]:
    """Bytes up to the last complete line from offset, and their length."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    return data[:end], end


def _parse(data: bytes) -> list[dict]:
    rows = []
    for line in data.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            continue
    return rows


def group(rows: list[dict]) -> "OrderedDict[str, dict]":
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        k = type_key(r)
        g = groups.setdefault(k, {"severity": r.get("severity", "?"), "title": r.get("title", ""),
                                  "sources": set(), "count": 0, "first": r.get("ts"),
                                  "last": r.get("ts"), "status": {}})
        g["count"] += 1
        g["last"] = r.get("ts") or g["last"]
        g["sources"].add(r.get("source", "?"))
        st = r.get("status", "?")
        g["status"][st] = g["status"].get(st, 0) + 1
    return groups


def report(groups, known: dict) -> tuple[list[str], list[str]]:
    """(lines to print, new type keys)."""
    new = [k for k in groups if k not in known]
    out = []
    order = {"critical": 0, "report": 1, "low": 2}
    for k, g in sorted(groups.items(), key=lambda kv: (order.get(kv[1]["severity"], 9), -kv[1]["count"])):
        status = ", ".join(f"{s} {n}" for s, n in sorted(g["status"].items()))
        flag = "NEW " if k in new else ""
        out.append(f"{flag}[{g['severity']}] {g['title']}  ×{g['count']} ({status})  "
                   f"{g['first']} .. {g['last']}  from {', '.join(sorted(g['sources']))}")
    return out, new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--no-advance", action="store_true", help="don't move the cursor")
    ap.add_argument("--from-start", action="store_true", help="reread the whole outbox")
    ap.add_argument("--baseline", action="store_true",
                    help="mark every type in the rows read as known and exit 0 (first install)")
    args = ap.parse_args(argv)
    if args.baseline and args.no_advance:
        ap.error("--baseline needs the cursor to advance; drop --no-advance")

    outbox = args.logs_dir / OUTBOX_NAME
    cursor_path = args.logs_dir / CURSOR_NAME
    cursor = load_cursor(cursor_path)
    rows, offset, inode, head = read_new_rows(outbox, cursor, args.from_start)
    groups = group(rows)
    lines, new = report(groups, cursor["known"])

    if not rows:
        print("alert_triage: no new alerts since the last run")
    else:
        held = sum(g["status"].get("held", 0) for g in groups.values())
        failed = sum(g["status"].get("failed", 0) for g in groups.values())
        print(f"alert_triage: {len(rows)} new alert row(s), {len(groups)} type(s), "
              f"{held} held for digest, {failed} failed to send, {len(new)} new type(s)")
        for ln in lines:
            print("  " + ln)
        if new:
            print("New alert types (not seen before):")
            for k in new:
                print(f"  {k}")

    if not args.no_advance:
        for k in groups:
            cursor["known"].setdefault(k, groups[k]["first"] or "")
        cursor["offset"] = offset
        cursor["inode"] = inode
        cursor["head"] = head
        try:
            save_cursor(cursor_path, cursor)
        except OSError as e:
            print(f"alert_triage: could not save cursor: {e}", file=sys.stderr)
    if args.baseline:
        return 0
    return 2 if any(groups[k]["severity"] == "critical" for k in new) else 0


if __name__ == "__main__":
    sys.exit(main())
