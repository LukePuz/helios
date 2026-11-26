"""Test-only pytest plugin: run the suite with the wall clock shifted by whole
days, to find tests whose fixed dates go stale ("date bombs"). Loaded only by
scripts/clock_shift_suite.py (`-p clock_shift_plugin`, with scripts/ on
PYTHONPATH); inert unless HELIOS_TEST_CLOCK_SHIFT_DAYS is set. Not imported by
any runtime module.

Stdlib only, so nothing is installed in the live venv: datetime.datetime and
datetime.date are replaced on the datetime module by subclasses whose now(),
utcnow(), today() and date.today() add the shift. time.time is shifted only with
HELIOS_TEST_CLOCK_SHIFT_TIME=1: the code's time.time() reads are durations and
ages against file mtimes, which a shift can't move, so shifting it makes false
failures (deploy/watchdog.py's marker and ping-file ages). The metaclass keeps isinstance()/issubclass() true for real datetime
and date objects. Project modules (files under the repo root) that bound the
real classes before the patch are rebound; third-party modules that the plugin
imports first keep the real classes. time.monotonic and perf_counter are not
shifted (durations stay real).

A shift of 0 still installs the fakes: the driver's baseline run, so plugin
artifacts cancel out of the report."""
from __future__ import annotations

import datetime as _dt
import os
import sys
import time as _time

# The real classes even if a shift is already active in this process (a fake
# carries its real class as _real, and the fake time.time as _real_time).
REAL_DATETIME = getattr(_dt.datetime, "_real", _dt.datetime)
REAL_DATE = getattr(_dt.date, "_real", _dt.date)
REAL_TIME = getattr(_time.time, "_real_time", _time.time)
ENV = "HELIOS_TEST_CLOCK_SHIFT_DAYS"

# Imported before the patch so their own module-level bindings stay real.
_PREIMPORT = ("pandas", "numpy", "psycopg2", "psycopg2.extras", "celery", "kombu", "redis",
              "requests", "urllib3", "dateutil", "dateutil.parser", "pytz", "zoneinfo",
              "yfinance", "_pytest.junitxml", "_pytest.terminal")


def make_fakes(days: int):
    """(FakeDatetime, FakeDate, fake_time) for a shift of `days` whole days.
    Pure: nothing is patched."""
    delta = _dt.timedelta(days=days)
    secs = days * 86400.0

    class _Meta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, cls._real)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, cls._real)

    class FakeDate(REAL_DATE, metaclass=_Meta):
        _real = REAL_DATE

        @classmethod
        def today(cls):
            # From datetime.now, not date.today: CPython's date.today reads
            # time.time, which is shifted too (that would apply the shift twice).
            d = REAL_DATETIME.now().date() + delta
            return cls(d.year, d.month, d.day)

    class FakeDatetime(REAL_DATETIME, metaclass=_Meta):
        _real = REAL_DATETIME

        @classmethod
        def now(cls, tz=None):
            t = REAL_DATETIME.now(tz) + delta
            return cls(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond,
                       t.tzinfo, fold=t.fold)

        @classmethod
        def utcnow(cls):
            t = REAL_DATETIME.utcnow() + delta
            return cls(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)

        @classmethod
        def today(cls):
            return cls.now()

    def fake_time():
        return REAL_TIME() + secs

    fake_time._real_time = REAL_TIME

    FakeDate.__name__ = FakeDate.__qualname__ = "date"
    FakeDatetime.__name__ = FakeDatetime.__qualname__ = "datetime"
    FakeDate.__module__ = FakeDatetime.__module__ = "datetime"
    return FakeDatetime, FakeDate, fake_time


def _rebind(root: str, fdt, fd) -> int:
    """Point project modules' module-level names bound to the real classes at
    the fakes. Returns how many names were rebound."""
    n = 0
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None) or ""
        if not f.startswith(root.rstrip(os.sep) + os.sep) or mod is sys.modules.get(__name__):
            continue
        try:
            items = list(vars(mod).items())
        except Exception:
            continue
        for name, val in items:
            if val is REAL_DATETIME:
                setattr(mod, name, fdt)
                n += 1
            elif val is REAL_DATE:
                setattr(mod, name, fd)
                n += 1
    return n


_STATE: dict = {}


def activate(days: int, root: str) -> None:
    """Install the shift for this process (idempotent)."""
    if _STATE:
        return
    import importlib
    # Pre-importing pandas loads pyarrow, which reads its memory pool from the
    # environment once. conftest sets this before the suite imports pandas; set
    # it here first too, or test_arrow_pool fails in every run, baseline included.
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
    for m in _PREIMPORT:
        try:
            importlib.import_module(m)
        except Exception:
            pass
    fdt, fd, ft = make_fakes(days)
    _dt.datetime, _dt.date = fdt, fd
    if os.environ.get("HELIOS_TEST_CLOCK_SHIFT_TIME") == "1":
        _time.time = ft
    _STATE.update(days=days, root=root, fakes=(fdt, fd), rebound=_rebind(root, fdt, fd))


def _root() -> str:
    """The tree under test (the driver sets HELIOS_TEST_CLOCK_SHIFT_ROOT when it
    runs another worktree); else this file's repo."""
    return os.path.abspath(os.environ.get("HELIOS_TEST_CLOCK_SHIFT_ROOT")
                           or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if os.environ.get(ENV) not in (None, ""):
    activate(int(os.environ[ENV]), _root())


def pytest_configure(config):
    if _STATE:
        _STATE["rebound"] += _rebind(_STATE["root"], *_STATE["fakes"])


def pytest_collection_finish(session):
    if _STATE:
        _STATE["rebound"] += _rebind(_STATE["root"], *_STATE["fakes"])


def pytest_report_header(config):
    if _STATE:
        return (f"clock shift: {_STATE['days']:+d} days (date.today() = "
                f"{_dt.date.today().isoformat()}); module names rebound so far: "
                f"{_STATE['rebound']}")
    return None
