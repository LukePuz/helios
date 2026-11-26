"""The pre-cut clock-shift check (scripts/clock_shift_suite.py and its plugin,
scripts/clock_shift_plugin.py): the fakes shift every wall-clock read by whole
days and keep isinstance true for real objects; the driver reports only
failures new against its 0-day baseline. Test-only tooling."""
import datetime as dt
import importlib.util
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pytest

# These check the tool against the real clock, so they don't run inside a
# shifted run (where the test's own clock is the fake one).
pytestmark = pytest.mark.skipif(os.environ.get("HELIOS_TEST_CLOCK_SHIFT_DAYS") not in (None, ""),
                                reason="inside a clock-shift run")

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.delenv("HELIOS_TEST_CLOCK_SHIFT_DAYS", raising=False)
    return _load("clock_shift_plugin")


suite = _load("clock_shift_suite")


def test_plugin_is_inert_without_the_env_var(plugin):
    assert dt.datetime is plugin.REAL_DATETIME and dt.date is plugin.REAL_DATE
    assert plugin._STATE == {}


def test_fakes_shift_every_clock_read_once(plugin):
    fdt, fd, ft = plugin.make_fakes(3)
    real_today = datetime.now().date()
    assert fd.today() == real_today + timedelta(days=3)
    assert fdt.now().date() == real_today + timedelta(days=3)
    assert fdt.today().date() == real_today + timedelta(days=3)
    assert (fdt.now(timezone.utc) - datetime.now(timezone.utc)).days in (2, 3)
    assert fdt.now(timezone.utc).tzinfo is timezone.utc
    assert abs(ft() - time.time() - 3 * 86400) < 5
    assert (fdt.utcnow() - datetime.utcnow()) > timedelta(days=2, hours=23)


def test_fakes_keep_isinstance_and_arithmetic(plugin):
    fdt, fd, _ = plugin.make_fakes(7)
    real = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
    assert isinstance(real, fdt) and isinstance(real.date(), fd) and isinstance(real, fd)
    assert not isinstance(date(2026, 11, 26), fdt)
    assert issubclass(datetime, fdt) and issubclass(date, fd)
    made = fdt(2026, 12, 1, 9, 30, tzinfo=timezone.utc)
    assert made == datetime(2026, 12, 1, 9, 30, tzinfo=timezone.utc)
    assert made - timedelta(days=1) == datetime(2026, 11, 30, 9, 30, tzinfo=timezone.utc)
    assert fd(2026, 12, 1) > date(2026, 11, 30)
    assert fdt.fromisoformat("2026-12-01T09:30:00+00:00") == made
    assert fdt.__name__ == "datetime" and fd.__name__ == "date"


def test_zero_shift_fakes_read_the_real_clock(plugin):
    fdt, fd, ft = plugin.make_fakes(0)
    assert fd.today() == datetime.now().date()
    assert abs(ft() - time.time()) < 5


def test_shift_days():
    assert suite.shift_days(date(2026, 11, 27), 7, date(2026, 11, 26)) == 8
    assert suite.shift_days(date(2026, 11, 26), 30, date(2026, 11, 26)) == 30
    assert suite.shift_days(date(2026, 11, 20), 0, date(2026, 11, 26)) == -6


_XML = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_a" name="test_ok" time="0.1"/>
<testcase classname="tests.test_a" name="test_bad" time="0.1"><failure message="assert 2 == 1&#10;more">tb</failure></testcase>
<testcase classname="tests.test_b.TestK" name="test_m[x]" time="0.1"><error message="boom">tb</error></testcase>
<testcase classname="tests.test_b" name="test_skip" time="0"><skipped message="s"/></testcase>
</testsuite></testsuites>"""


def test_parse_junit_ids_and_counts():
    p = suite.parse_junit(_XML)
    assert p["tests/test_a.py::test_bad"] == "failure: assert 2 == 1"
    assert p["tests/test_b.py::TestK::test_m[x]"] == "error: boom"
    assert "tests/test_a.py::test_ok" not in p
    assert p["_counts"] == {"tests": 4, "failures": 1, "errors": 1, "skipped": 1}


def test_only_new_failures_are_reported():
    base = {"a": "failure: x", "_counts": {}}
    shifted = {"a": "failure: x", "b": "failure: y", "_counts": {}}
    assert suite.new_failures(base, shifted) == {"b": "failure: y"}
    c = {"tests": 2, "failures": 1, "errors": 0, "skipped": 0}
    text = suite.report("t", date(2026, 11, 27), date(2026, 11, 26),
                        {"default": {"baseline": {**base, "_counts": c},
                                     7: {**shifted, "_counts": c}}})
    assert "load + 7 (2026-12-04)" in text and "`b`: failure: y" in text
    assert "`a`" not in text
    assert text.rstrip().endswith("NEW FAILURES (date bombs to fix with relative dates)")
    same = suite.report("t", date(2026, 11, 27), date(2026, 11, 26),
                        {"default": {"baseline": {**base, "_counts": c}, 7: {**base, "_counts": c}}})
    assert same.rstrip().endswith("no new failures")


def test_busy_check_counts_pytest_processes_only():
    lines = ["/x/venv/bin/python -m pytest -q tests", "bash -c '[p]ython -m pytest'",
             "python3 -m pytest tests/test_a.py", "/usr/bin/python3 app.py"]
    assert suite.busy_count(lines) == 2


def test_main_refuses_while_another_suite_runs(monkeypatch):
    monkeypatch.setattr(suite, "_ps_lines", lambda: ["python -m pytest -q"])
    ran = []
    monkeypatch.setattr(suite, "run_once", lambda *a, **k: ran.append(a) or {"_counts": None})
    assert suite.main(["--load-date", "2026-11-27"]) == 2
    assert ran == []
    assert suite.main(["--load-date", "bad"]) == 2


def test_main_runs_baseline_then_each_shift(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "_ps_lines", lambda: [])
    calls = []
    c = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}

    def fake_run(tree, files, days, shipped, python, shift_time=False):
        calls.append((days, shipped))
        return {"_counts": c}

    monkeypatch.setattr(suite, "run_once", fake_run)
    out = tmp_path / "r.md"
    load = date.today() + timedelta(days=1)
    assert suite.main(["--load-date", load.isoformat(), "--plus", "7,30", "--out", str(out)]) == 0
    assert calls == [(0, False), (8, False), (31, False), (0, True), (8, True), (31, True)]
    assert out.read_text().rstrip().endswith("no new failures")


def test_not_imported_by_runtime_code():
    root = os.path.dirname(_SCRIPTS)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("tests", "venv", ".git", "scripts", "__pycache__", "logs")]
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(dirpath, f), encoding="utf-8", errors="ignore") as fh:
                    assert "clock_shift" not in fh.read(), f


def test_unshifted_clock_note_names_what_the_file_reads(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("now = pd.Timestamp.now(tz='UTC')\nage = p.stat().st_mtime\n")
    (tmp_path / "tests" / "test_y.py").write_text("from datetime import date\n")
    note = suite.unshifted_clock_note(str(tmp_path), "tests/test_x.py::test_a")
    assert "Timestamp.now" in note and "st_mtime" in note and "check by hand" in note
    assert suite.unshifted_clock_note(str(tmp_path), "tests/test_y.py::test_b") == ""
    assert suite.unshifted_clock_note(str(tmp_path), "tests/missing.py::t") == ""


def test_zero_day_baseline_keeps_the_system_arrow_pool():
    import subprocess
    root = os.path.dirname(_SCRIPTS)
    env = {k: v for k, v in os.environ.items()
           if k != "ARROW_DEFAULT_MEMORY_POOL"}
    env.update(HELIOS_TEST_CLOCK_SHIFT_DAYS="0", HELIOS_TEST_CLOCK_SHIFT_ROOT=root,
               PYTHONPATH=_SCRIPTS, REDIS_URL="redis://127.0.0.1:1/0")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          "-p", "clock_shift_plugin",
                          "tests/test_arrow_pool.py::test_test_session_runs_the_system_pool"],
                         cwd=root, capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0 and "1 passed" in out.stdout, out.stdout[-2000:]


def test_time_time_is_shifted_only_on_request():
    import subprocess
    code = ("import time, datetime as d; r = time.time(); import clock_shift_plugin as p; "
            "print(round((time.time() - r) / 86400), d.date.today() - d.datetime.now().date())")
    env = {**os.environ, "HELIOS_TEST_CLOCK_SHIFT_DAYS": "5", "PYTHONPATH": _SCRIPTS}
    for flag, want_days in (("0", "0"), ("1", "5")):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**env, "HELIOS_TEST_CLOCK_SHIFT_TIME": flag}, timeout=120)
        assert out.stdout.split()[0] == want_days, out.stderr
