"""72B panel-call timing from existing logs and signals rows (read-only).

No model is called. Two sources, each optional:

* logs: the llm worker log (``celery_llm_worker.log`` and its rotation
  ``.log.1``). One analyst call is the span from ``[T] Analyzing with <model>...``
  to the next ``[T] -> ACTION | conviction=`` line on the same worker. The llm
  worker runs ``-c 1``, so pairing by worker name is unambiguous. Log times are
  the box's local time, America/New_York.
* db: ``signals.provenance->'llm_call'->>'total_duration'`` (Ollama's own
  figure, nanoseconds), written on every verdict that reached the model. Routing
  exit rows copy the verdict's provenance, llm_call included, and carry
  ``decision.route_exit`` (``db._VERDICT_COPY_KEY``), so they are left out: one
  call, one row. Only
  SELECTs run, inside a READ ONLY transaction that is rolled back.

Output: median and p90 seconds per call, then a capacity table: how many calls,
and how many 6/7/8-call panels, fit serially in the pre-open window
(05:00-09:30 ET) and in one hour, plus the same per start hour of the session.
Capacity assumes the single 72B slot does nothing else (Ollama
NUM_PARALLEL=1): event analyses, position reviews and the GPU health probe
compete for it in practice.

Usage:
    python -m helios3d.panel.timing [--source logs|db|both] [--start YYYY-MM-DD]
        [--end YYYY-MM-DD] [--logs-dir DIR] [--model SUBSTR] [--repo DIR]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Iterator

# The checkout this file is in, unless HELIOS3D_REPO / HELIOS3D_LOGS_DIR say
# otherwise (no home-directory paths in the repo).
_HERE_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REPO = os.getenv("HELIOS3D_REPO") or _HERE_REPO
DEFAULT_LOGS_DIR = os.getenv("HELIOS3D_LOGS_DIR") or os.path.join(DEFAULT_REPO, "logs")
LOG_FILES = ("celery_llm_worker.log.1", "celery_llm_worker.log")
# c99519f (10-25) moved the 7B and 3B models off the GPU; calls before it
# ran about twice as slow, so the default window starts after it.
DEFAULT_START = "2026-10-26"
DEFAULT_MODEL = "72b"
PANEL_SIZES = (6, 7, 8)
PREOPEN_START_H, PREOPEN_END_H = 5.0, 9.5      # 05:00-09:30 ET
PREOPEN_SECONDS = int((PREOPEN_END_H - PREOPEN_START_H) * 3600)   # 16200
HOUR_SECONDS = 3600

_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}): \w+/([\w-]+)\] (.*)$")
_START = re.compile(r"^\[([^\]]+)\] Analyzing with (\S+?)\.\.\.")
_DONE = re.compile(r"^\[([^\]]+)\] → (\w+) \| conviction=")


@dataclass(frozen=True)
class Call:
    start: datetime      # naive, America/New_York
    seconds: float
    model: str
    source: str          # "logs" or "db"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log_lines(lines: Iterable[str]) -> Iterator[Call]:
    """Pair start and verdict lines per worker. A start with no verdict
    (timeout, stub, parse failure) is dropped when the worker's next start
    arrives; a verdict for a different ticker than the open start is ignored."""
    open_: dict[str, tuple[str, datetime, str]] = {}
    for raw in lines:
        m = _LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
            microsecond=int(m.group(2)) * 1000)
        worker, msg = m.group(3), m.group(4)
        s = _START.match(msg)
        if s:
            open_[worker] = (s.group(1), ts, s.group(2))
            continue
        d = _DONE.match(msg)
        if d and worker in open_:
            ticker, t0, model = open_.pop(worker)
            if ticker == d.group(1):
                yield Call(t0, (ts - t0).total_seconds(), model, "logs")


def read_log_calls(logs_dir: str, files: Iterable[str] = LOG_FILES) -> list[Call]:
    """Read the worker logs in rotation order (oldest first). Missing files
    are skipped. Opened read-only."""
    def _lines():
        for name in files:
            path = os.path.join(logs_dir, name)
            try:
                with open(path, "r", errors="replace") as fh:
                    yield from fh
            except FileNotFoundError:
                continue
    return list(parse_log_lines(_lines()))


DB_SQL = """
SELECT (timestamp AT TIME ZONE 'America/New_York') AS ts_et,
       provenance->>'model' AS model,
       (provenance->'llm_call'->>'total_duration')::numeric AS total_ns
  FROM signals
 WHERE provenance ? 'llm_call'
   AND NOT (COALESCE(decision, '{}'::jsonb) ? %s)
   AND provenance->'llm_call'->>'total_duration' IS NOT NULL
   AND timestamp >= %s::date AND timestamp < (%s::date + 1)
 ORDER BY timestamp
"""


def calls_from_db_rows(rows: Iterable[tuple]) -> list[Call]:
    """(ts_et, model, total_ns) rows -> calls. The row time is when the verdict
    was written, so the call started total_duration earlier."""
    from datetime import timedelta
    out = []
    for ts_et, model, total_ns in rows:
        if total_ns is None or ts_et is None:
            continue
        secs = float(total_ns) / 1e9
        if secs <= 0:
            continue
        ts = ts_et.replace(tzinfo=None) if getattr(ts_et, "tzinfo", None) else ts_et
        out.append(Call(ts - timedelta(seconds=secs), secs, model or "?", "db"))
    return out


def read_db_calls(start: str, end: str, repo: str = DEFAULT_REPO) -> list[Call]:
    """SELECT-only read through the repo's own db module (its pool and
    connection settings), inside a READ ONLY transaction, always rolled back."""
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    import db  # noqa: WPS433 (runtime import: the CLI path only)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(DB_SQL, (db._VERDICT_COPY_KEY, start, end))
            rows = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    return calls_from_db_rows(rows)


# ---------------------------------------------------------------------------
# Statistics and capacity
# ---------------------------------------------------------------------------

def percentile(xs: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(secs: list[float]) -> dict:
    if not secs:
        return {"n": 0, "median": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"n": len(secs), "median": statistics.median(secs), "p90": percentile(secs, 0.9),
            "min": min(secs), "max": max(secs)}


def capacity(sec_per_call: float, window_s: int, panel_sizes=PANEL_SIZES) -> dict:
    """Serial calls in the window, then whole panels of each size."""
    if not sec_per_call or sec_per_call != sec_per_call or sec_per_call <= 0:
        return {"calls": 0, **{k: 0 for k in panel_sizes}}
    calls = int(window_s // sec_per_call)
    return {"calls": calls, **{k: calls // k for k in panel_sizes}}


def filter_calls(calls: Iterable[Call], start: str, end: str, model_substr: str) -> list[Call]:
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    return [c for c in calls
            if lo <= c.start.date() <= hi and model_substr.lower() in (c.model or "").lower()]


def by_start_hour(calls: Iterable[Call]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for c in calls:
        out.setdefault(c.start.hour, []).append(c.seconds)
    return dict(sorted(out.items()))


def in_preopen(c: Call) -> bool:
    h = c.start.hour + c.start.minute / 60
    return PREOPEN_START_H <= h < PREOPEN_END_H


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return "-" if x != x else f"{x:.1f}"


def _panels(cap: dict, sizes=PANEL_SIZES) -> str:
    return " / ".join(str(cap[k]) for k in sizes)


def report(calls: list[Call], label: str, sizes=PANEL_SIZES) -> str:
    lines = []
    secs = [c.seconds for c in calls]
    s = summarize(secs)
    lines.append(f"== {label}")
    lines.append(f"{'slice':<22}{'n':>6}{'median s':>10}{'p90 s':>9}{'min':>8}{'max':>8}")
    rows = [("all", s), ("pre-open 05:00-09:30", summarize([c.seconds for c in calls if in_preopen(c)]))]
    for name, st in rows:
        lines.append(f"{name:<22}{st['n']:>6}{_fmt(st['median']):>10}{_fmt(st['p90']):>9}"
                     f"{_fmt(st['min']):>8}{_fmt(st['max']):>8}")
    hdr = "/".join(str(k) for k in sizes)
    lines.append("")
    lines.append(f"Capacity, serial on one 72B slot (panels of {hdr} calls)")
    lines.append(f"{'basis':<10}{'s/call':>8}{'pre-open calls':>16}{'pre-open panels':>18}"
                 f"{'calls/h':>9}{'panels/h':>12}")
    for basis in ("median", "p90"):
        spc = s[basis]
        pre, hr = capacity(spc, PREOPEN_SECONDS, sizes), capacity(spc, HOUR_SECONDS, sizes)
        lines.append(f"{basis:<10}{_fmt(spc):>8}{pre['calls']:>16}{_panels(pre, sizes):>18}"
                     f"{hr['calls']:>9}{_panels(hr, sizes):>12}")
    lines.append("")
    lines.append(f"Per start hour (ET); capacity of that hour at its own median / p90 (panels {hdr})")
    lines.append(f"{'hour':<7}{'n':>5}{'median s':>10}{'p90 s':>9}{'calls/h med':>13}"
                 f"{'panels med':>13}{'calls/h p90':>13}{'panels p90':>13}")
    for h, xs in by_start_hour(calls).items():
        st = summarize(xs)
        cm, cp = capacity(st["median"], HOUR_SECONDS, sizes), capacity(st["p90"], HOUR_SECONDS, sizes)
        lines.append(f"{h:02d}:00{'':<2}{st['n']:>5}{_fmt(st['median']):>10}{_fmt(st['p90']):>9}"
                     f"{cm['calls']:>13}{_panels(cm, sizes):>13}{cp['calls']:>13}{_panels(cp, sizes):>13}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=("logs", "db", "both"), default="both")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="substring of the model name")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="checkout whose db module is used")
    args = ap.parse_args(argv)

    out = [f"72B call timing, {args.start} to {args.end}, model contains '{args.model}'. "
           "Read-only; no model calls."]
    if args.source in ("logs", "both"):
        calls = filter_calls(read_log_calls(args.logs_dir), args.start, args.end, args.model)
        out.append(report(calls, "logs: 'Analyzing with' -> verdict line, llm worker"))
    if args.source in ("db", "both"):
        try:
            calls = filter_calls(read_db_calls(args.start, args.end, args.repo),
                                 args.start, args.end, args.model)
            out.append(report(calls, "db: signals.provenance.llm_call.total_duration (Ollama)"))
        except Exception as e:  # the log report still stands on its own
            out.append(f"== db: unavailable ({type(e).__name__}: {str(e)[:200]})")
    print("\n\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
