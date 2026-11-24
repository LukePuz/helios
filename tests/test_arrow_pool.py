"""pyarrow's system allocator by environment: the default (mimalloc)
segfaulted on threaded pandas allocations in this stack (2026-11-08), and
config must not import pyarrow (BLAS threads before Celery's fork)."""
import subprocess
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent


def test_system_pool_in_force():
    """The bot's import order in a fresh interpreter: config, then pandas
    (which loads pyarrow). In-process the pool is whatever the first pyarrow
    import of the test session picked, so this runs in a subprocess."""
    assert config.ARROW_MEMORY_POOL == "system"
    code = ("import os; os.environ.pop('ARROW_DEFAULT_MEMORY_POOL', None); import config; "
            "import pandas; import pyarrow as pa; print(pa.default_memory_pool().backend_name)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=60).stdout.split()
    assert out[-1:] == ["system"]


def test_without_config_first_the_pool_is_not_system():
    """The case conftest guards: pandas imported before config picks
    pyarrow's default pool, which is why the test session sets the variable
    before any test module is imported."""
    code = ("import os; os.environ.pop('ARROW_DEFAULT_MEMORY_POOL', None); import pandas; "
            "import pyarrow as pa; print(pa.default_memory_pool().backend_name)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=60).stdout.split()
    assert out and out[-1] != "system"


def test_test_session_runs_the_system_pool():
    import pyarrow as pa
    assert pa.default_memory_pool().backend_name == "system"   # conftest sets it first


def test_config_imports_no_numpy_or_pyarrow():
    code = ("import sys, os; os.environ.pop('ARROW_DEFAULT_MEMORY_POOL', None); import config; "
            "print('pyarrow' in sys.modules, 'numpy' in sys.modules, "
            "os.environ['ARROW_DEFAULT_MEMORY_POOL'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=60).stdout.split()[-3:]
    assert out == ["False", "False", "system"]


def test_workers_get_the_native_settings():
    src = (ROOT / "process_manager.py").read_text()
    for k in ("ARROW_DEFAULT_MEMORY_POOL", "OPENBLAS_NUM_THREADS", "PYTHONFAULTHANDLER"):
        assert k in src
