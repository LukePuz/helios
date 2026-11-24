"""Weekly forward readouts (engineering diagnostics, not study analyses).

tasks.weekly_readouts runs this on Fridays after the nightly price refresh. It
writes ONE markdown file, logs/readouts/weekly-YYYY-MM-DD.md (the ET run
date), with these sections, each over the last READOUT_WEEKS weeks:
  - gate:        scripts/forward_readout.py --by gate (ENTRY_GATE_BLOCKED by
                 the gate that fired, OPEN, ADD);
  - conviction:  scripts/forward_readout.py --by conviction (OPEN, ADD);
  - shadows:     forward_readout on STOPOUT_REENTRY_SHADOW,
                 ENTRY_QUALITY_SHADOW and TRIM_EXIT_RULE, the last split by
                 decision.held_below_both_sma (true / false / unknown);
  - vol_shadow:  scripts/vol_shadow_readout.py --env all, buy fills in the
                 window (the position ledger still reads all trades);
  - fill_lag:    scripts/fill_lag_readout.py;
  - plan_levels: scripts/plan_level_readout.py (compiled plan level
                 crossings, per plan and in force at the open, and the
                 forward return after a crossing);
  - entry_funnel: scripts/entry_funnel.py over the last 5 sessions (not the
                 8 weeks): OPEN/ADD verdicts -> stops by gate -> approvals
                 -> orders -> fills, per session;
  - shadow_outcomes: scripts/shadow_outcomes.py (held-exit fires per rule,
                 story de-dup per trigger source, book heat at each buy,
                 recheck / add-to-loser would_block, the above-market stop
                 guard by path, setup-gate qualifiers, the drop watch by
                 would_review), with forward_readout's windows.
The scripts' own functions run in-process (no subprocess). Each section has
its own try: a failing one writes "section failed: <class>" and the rest
still run. Every read is the scripts' own: one transaction opened READ ONLY,
then rolled back. Their output is percentages, ratios, times and counts only.

Nothing is sent (no Telegram, no e-mail). It writes the file, keeps the newest
KEEP_FILES weekly files, and writes one activity row (stage weekly_readouts:
the path and each section's status). Which number decides anything is a
separate choice, not this module's. Never raises.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "logs", "readouts")
NY = ZoneInfo("America/New_York")
READOUT_WEEKS = 8
KEEP_FILES = 12
SHADOW_KINDS = ("STOPOUT_REENTRY_SHADOW", "ENTRY_QUALITY_SHADOW", "TRIM_EXIT_RULE")
_FILE_RE = re.compile(r"^weekly-\d{4}-\d{2}-\d{2}\.md$")

_MODULES: dict = {}


def _load(name: str):
    """scripts/<name>.py as a module, loaded once per process (scripts/ is
    not a package)."""
    mod = _MODULES.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(f"helios_readout_{name}",
                                                      os.path.join(ROOT, "scripts", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return mod


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def shadow_label(kind: str, decision) -> str:
    """TRIM_EXIT_RULE by held_below_both_sma; the other kinds as is."""
    if kind != "TRIM_EXIT_RULE":
        return kind
    v = decision.get("held_below_both_sma") if isinstance(decision, dict) else None
    return f"TRIM_EXIT_RULE (held_below_both_sma {'true' if v is True else 'false' if v is False else 'unknown'})"


# ---------------------------------------------------------------- sections

def _gate(since: date, env: str, head: str) -> str:
    fr = _load("forward_readout")
    rows, bars = fr._fetch(list(fr.DEFAULT_KINDS), since, env)
    return fr.readout(fr.collect(rows, bars), since, env, head)


def _conviction(since: date, env: str, head: str) -> str:
    fr = _load("forward_readout")
    rows, bars = fr._fetch(list(fr.CONVICTION_KINDS), since, env, by_conviction=True)
    return fr.conviction_readout(rows, bars, since, env, head)


def _shadows(since: date, env: str, head: str) -> str:
    fr = _load("forward_readout")
    rows, bars = fr._fetch(list(SHADOW_KINDS), since, env)
    return fr.readout(fr.collect(rows, bars, label_fn=shadow_label), since, env, head)


def _vol_shadow(since: date, env: str, head: str) -> str:
    vs = _load("vol_shadow_readout")
    fills, bars, snapshots, approvals = vs._fetch("all")
    start = datetime.combine(since, datetime.min.time(), tzinfo=NY)
    fills = [f for f in fills if f["ts"] is not None
             and (f["ts"] if f["ts"].tzinfo else f["ts"].replace(tzinfo=timezone.utc)) >= start]
    return (f"Buy fills since {since}.\n\n"
            + vs.readout(vs.fill_rows(fills, bars, snapshots, approvals), "all", head))


def _fill_lag(since: date, env: str, head: str) -> str:
    fl = _load("fill_lag_readout")
    outcomes, trades, seen, signals, approvals = fl._fetch(since, env)
    return fl.readout(fl.attach(fl.exits(outcomes), trades, seen, signals, approvals),
                      since, env, head)


def _plan_levels(since: date, env: str, head: str) -> str:
    pl = _load("plan_level_readout")
    plans, outcomes, bars = pl._fetch(since, env)
    return pl.build(plans, outcomes, bars, since, env, head, datetime.now(timezone.utc))


def _entry_funnel(since: date, env: str, head: str) -> str:
    ef = _load("entry_funnel")
    sessions = ef.sessions_until(ef.default_until(datetime.now(timezone.utc)),
                                 ef.DEFAULT_SESSIONS)
    return ef.readout(ef.build_funnel(sessions, *ef._fetch(sessions, env)), sessions, env, head)


def _shadow_outcomes(since: date, env: str, head: str) -> str:
    so = _load("shadow_outcomes")
    return so.build(so._fetch(since, env), since, env, head)


SECTIONS = (
    ("gate", "Forward returns by entry gate", _gate),
    ("conviction", "Forward returns by conviction", _conviction),
    ("shadows", "Forward returns of the shadow kinds", _shadows),
    ("vol_shadow", "Volatility shadow on past buy fills", _vol_shadow),
    ("fill_lag", "Fill lag on broker exits", _fill_lag),
    ("plan_levels", "Compiled plan levels", _plan_levels),
    ("entry_funnel", "Entry funnel, last 5 sessions", _entry_funnel),
    ("shadow_outcomes", "Shadow outcomes", _shadow_outcomes),
)


def _demote(md: str) -> str:
    """A script's markdown under this file's section heading: '#' -> '###'."""
    return "\n".join("##" + line if line.startswith("#") else line for line in md.splitlines())


def build(now: datetime, env: str, head: str, sections=SECTIONS) -> tuple[str, dict]:
    """The file's markdown and {section key: "ok" | "failed: <class>"}."""
    since = now.astimezone(NY).date() - timedelta(weeks=READOUT_WEEKS)
    statuses, parts = {}, []
    for key, title, fn in sections:
        try:
            body = _demote(fn(since, env, head))
            statuses[key] = "ok"
        except Exception as e:
            log.warning(f"weekly readouts: section {key} failed: {type(e).__name__}: {e}")
            body = f"section failed: {type(e).__name__}"
            statuses[key] = f"failed: {type(e).__name__}"
        parts.append(f"## {title}\n\n{body.rstrip()}\n")
    header = (f"# Weekly readouts ({env}), {now.astimezone(NY):%Y-%m-%d}\n\n"
              f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET at code {head}. Rows since "
              f"{since} ({READOUT_WEEKS} weeks). Engineering diagnostics, not study results: "
              "percentages, ratios, times and counts only. Section status: "
              + ", ".join(f"{k} {v}" for k, v in statuses.items()) + ".\n")
    return header + "\n" + "\n".join(parts), statuses


def prune(out_dir: str, keep: int = KEEP_FILES) -> list[str]:
    """Delete all but the newest `keep` weekly-YYYY-MM-DD.md files (by the
    date in the name). Other files are left alone. Returns the removed names."""
    names = sorted(n for n in os.listdir(out_dir) if _FILE_RE.match(n))
    removed = []
    for n in names[:-keep] if keep > 0 else names:
        try:
            os.remove(os.path.join(out_dir, n))
            removed.append(n)
        except OSError as e:
            log.warning(f"weekly readouts: could not remove {n}: {e}")
    return removed


def run(now: datetime | None = None, out_dir: str | None = None) -> dict | None:
    """Write the file, prune, log one activity row. Returns {path, sections,
    removed}, or None when the file could not be written. Never raises."""
    try:
        from config import TRADING_ENV
        import db
        now = now or datetime.now(timezone.utc)
        out_dir = out_dir or OUT_DIR
        text, statuses = build(now, TRADING_ENV, _git_head())
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"weekly-{now.astimezone(NY):%Y-%m-%d}.md")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        removed = prune(out_dir)
        failed = [k for k, v in statuses.items() if v != "ok"]
        try:
            db.log_activity("weekly_readouts", "",
                            f"weekly readouts written to {os.path.relpath(path, ROOT)}: "
                            f"{len(statuses) - len(failed)} of {len(statuses)} sections ok"
                            + (f" (failed: {', '.join(failed)})" if failed else ""),
                            "alert" if failed else "neutral",
                            extra={"path": os.path.relpath(path, ROOT), "sections": statuses,
                                   "env": TRADING_ENV, "weeks": READOUT_WEEKS,
                                   "removed": removed})
        except Exception as e:
            log.warning(f"weekly readouts: activity row not written: {type(e).__name__}: {e}")
        return {"path": path, "sections": statuses, "removed": removed}
    except Exception as e:
        log.error(f"weekly readouts failed: {type(e).__name__}: {e}")
        return None
