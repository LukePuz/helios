"""pyarrow's system allocator by environment: the default (mimalloc)
segfaulted on threaded pandas allocations in this stack (2026-11-08), and
config must not import pyarrow (BLAS threads before Celery's fork)."""
import subprocess
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent


def test_system_pool_in_force():
    import pyarrow as pa
    assert config.ARROW_MEMORY_POOL == "system"
    assert pa.default_memory_pool().backend_name == "system"


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
