"""Pre-cut date-bomb check: run the test suite with the wall clock shifted to
the cut's load day + N days (default N = 7 and 30) and report only the
failures that are new against an unshifted baseline.

  python scripts/clock_shift_suite.py --tree <worktree> --load-date 2026-11-27 \
      [--plus 7,30] [--modes default,shipped] [--files tests/...] [--out report.md]

Each mode runs a baseline (the plugin installed with a shift of 0, so plugin
artifacts cancel out) and then one run per N, with the shift
load_date + N - today in whole days. scripts/clock_shift_plugin.py does the
shifting (stdlib only; nothing is installed). Runs are niced, use
REDIS_URL=redis://127.0.0.1:1/0, and go one at a time; the script refuses to
start while another `python -m pytest` runs unless --force. Test-only: no DB,
broker or network access of its own, and no runtime module imports it.

Each run uses --tb=long with full assertion diffs (verbosity_assertions=2),
and the report ends with the full failure text of every new failure, so a
one-off can be read without a rerun (1130 clock-shift, 2026-11-26).

Exit status: 0 when no shifted run has a new failure, 1 when one does, 2 on a
usage or busy refusal."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
_BUSY = re.compile(r"^[^ ]*python[^ ]* -m pytest")


def shift_days(load: date, plus: int, today: date) -> int:
    return (load - today).days + plus


def busy_count(ps_lines) -> int:
    return sum(1 for ln in ps_lines if _BUSY.match(ln))


def _ps_lines() -> list[str]:
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10)
        return out.stdout.splitlines()
    except Exception:
        return []


def parse_junit(text: str) -> dict:
    """{test id: first line of its failure or error message} from a junit
    XML report, plus the pass/fail/error/skip counts under "_counts" and
    {test id: its full traceback text (the message when there is none)}
    under "_text"."""
    root = ET.fromstring(text)
    out, counts, full = {}, {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}, {}
    for case in root.iter("testcase"):
        counts["tests"] += 1
        cls, name = case.get("classname", ""), case.get("name", "")
        parts = cls.split(".")
        mod = [p for p in parts if p.startswith("test_")]
        if mod:
            i = parts.index(mod[0])
            path = "/".join(parts[:i + 1]) + ".py"
            tid = "::".join([path, *parts[i + 1:], name])
        else:
            tid = f"{cls}::{name}"
        for kind in ("failure", "error"):
            el = case.find(kind)
            if el is not None:
                counts["failures" if kind == "failure" else "errors"] += 1
                msg = (el.get("message") or el.text or "").strip().splitlines()
                out[tid] = f"{kind}: {msg[0][:200] if msg else ''}"
                # the traceback holds the message's E lines; the message alone
                # only when there is none
                full[tid] = (el.text or "").strip() or (el.get("message") or "").strip()
                break
        if case.find("skipped") is not None:
            counts["skipped"] += 1
    out["_counts"] = counts
    out["_text"] = full
    return out


def new_failures(base: dict, shifted: dict) -> dict:
    return {k: v for k, v in shifted.items() if not k.startswith("_") and k not in base}


def run_once(tree: str, files: list[str], days: int, shipped: bool, python: str,
             shift_time: bool = False) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["REDIS_URL"] = "redis://127.0.0.1:1/0"
    env["HELIOS_TEST_CLOCK_SHIFT_DAYS"] = str(days)
    env["HELIOS_TEST_CLOCK_SHIFT_ROOT"] = os.path.abspath(tree)
    if shift_time:
        env["HELIOS_TEST_CLOCK_SHIFT_TIME"] = "1"
    else:
        env.pop("HELIOS_TEST_CLOCK_SHIFT_TIME", None)
    if shipped:
        env["HELIOS_TEST_SHIPPED_DEFAULTS"] = "1"
    else:
        env.pop("HELIOS_TEST_SHIPPED_DEFAULTS", None)
    with tempfile.TemporaryDirectory() as td:
        xml = os.path.join(td, "junit.xml")
        cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "clock_shift_plugin",
               "--tb=long", "-o", "verbosity_assertions=2", f"--junitxml={xml}", *files]
        subprocess.run(cmd, cwd=tree, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       preexec_fn=lambda: os.nice(19))
        if not os.path.exists(xml):
            return {"_counts": None}
        with open(xml, encoding="utf-8") as f:
            return parse_junit(f.read())


# Clocks the plugin can't shift. A new failure in a test file that reads one
# may be the plugin's, not a date bomb: check it by hand.
_UNSHIFTED = re.compile(r"Timestamp\.now|Timestamp\.today|st_mtime|getmtime|os\.utime|datetime64\(")


def unshifted_clock_note(tree: str, tid: str) -> str:
    path = os.path.join(tree, tid.split("::", 1)[0])
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            hits = sorted(set(_UNSHIFTED.findall(f.read())))
    except OSError:
        return ""
    return f" (the test file reads an unshifted clock: {', '.join(hits)}; check by hand)" if hits else ""


def report(tree_label: str, load: date, today: date, results: dict, tree: str | None = None) -> str:
    """results: {mode: {"baseline": parsed, plus: parsed, ...}}."""
    lines = [f"# Clock-shift check: {tree_label}", "",
             f"Real date {today.isoformat()}; load date {load.isoformat()}. Baseline = the plugin "
             "installed with a 0-day shift. Listed: failures new against the baseline.", ""]
    any_new, details = False, []
    for mode, runs in results.items():
        base = runs["baseline"]
        bc = base.get("_counts")
        lines.append(f"## {mode} mode")
        lines.append(f"- baseline: {_fmt(bc)}")
        for plus, parsed in runs.items():
            if plus == "baseline":
                continue
            target = load.toordinal() + plus
            new = new_failures(base, parsed)
            any_new = any_new or bool(new)
            lines.append(f"- load + {plus} ({date.fromordinal(target).isoformat()}): "
                         f"{_fmt(parsed.get('_counts'))}; new failures: {len(new)}")
            for tid in sorted(new):
                note = unshifted_clock_note(tree, tid) if tree else ""
                lines.append(f"  - `{tid}`: {new[tid]}{note}")
                body = (parsed.get("_text") or {}).get(tid) or new[tid]
                details += [f"### {mode}, load + {plus}: `{tid}`", ""]
                details += ["    " + ln for ln in body.splitlines()] + [""]
        lines.append("")
    if details:
        lines += ["## Full failure text (new failures)", ""] + details
    lines.append("Result: " + ("NEW FAILURES (date bombs to fix with relative dates)" if any_new
                               else "no new failures"))
    return "\n".join(lines) + "\n"


def _fmt(c) -> str:
    if not c:
        return "no report (pytest did not finish)"
    bad = c["failures"] + c["errors"]
    return f"{c['tests']} tests, {bad} failed or errored, {c['skipped']} skipped"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", default=os.path.dirname(HERE))
    ap.add_argument("--load-date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--plus", default="7,30")
    ap.add_argument("--modes", default="default,shipped")
    ap.add_argument("--files", nargs="*", default=["tests"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--force", action="store_true", help="run even if another pytest is running")
    ap.add_argument("--shift-time", action="store_true",
                    help="also shift time.time (off by default: file-mtime ages then fail falsely)")
    a = ap.parse_args(argv)
    today = date.today()
    try:
        load = date.fromisoformat(a.load_date) if a.load_date else today
        pluses = [int(x) for x in a.plus.split(",") if x.strip()]
        modes = [m.strip() for m in a.modes.split(",") if m.strip()]
        assert set(modes) <= {"default", "shipped"} and pluses
    except Exception:
        print("bad --load-date, --plus or --modes", file=sys.stderr)
        return 2
    if not a.force and busy_count(_ps_lines()):
        print("another python -m pytest is running; wait for it (or --force)", file=sys.stderr)
        return 2
    results = {}
    for mode in modes:
        shipped = mode == "shipped"
        runs = {"baseline": run_once(a.tree, a.files, 0, shipped, a.python, a.shift_time)}
        for plus in pluses:
            runs[plus] = run_once(a.tree, a.files, shift_days(load, plus, today), shipped, a.python,
                                  a.shift_time)
        results[mode] = runs
    label = a.tree
    try:
        head = subprocess.run(["git", "-C", a.tree, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        label = f"{a.tree} @ {head}" if head else a.tree
    except Exception:
        pass
    text = report(label, load, today, results, tree=a.tree)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text, end="")
    return 1 if "NEW FAILURES" in text.splitlines()[-1] else 0


if __name__ == "__main__":
    sys.exit(main())
