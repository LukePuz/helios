"""
Process lifecycle manager for the trading system.
Stores subprocess PIDs in .processes.json so the dashboard can
start/stop/monitor each service across Streamlit reruns.
"""
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PID_FILE = BASE_DIR / ".processes.json"

if sys.platform == "win32":
    _BIN    = BASE_DIR / "venv" / "Scripts"
    PYTHON  = str(_BIN / "python.exe")
    CELERY  = str(_BIN / "celery.exe")
    UVICORN = str(_BIN / "uvicorn.exe")
else:
    _BIN    = BASE_DIR / "venv" / "bin"
    PYTHON  = str(_BIN / "python")
    CELERY  = str(_BIN / "celery")
    UVICORN = str(_BIN / "uvicorn")

_OPS_POOL_ARGS    = ["--pool=solo"] if sys.platform == "win32" else ["--pool=prefork", "-c", "3"]
_LLM_POOL_ARGS    = ["--pool=solo"] if sys.platform == "win32" else ["--pool=prefork", "-c", "1"]
# Screening is CPU-forced 7B inference (see agents/screener.py num_gpu=0) —
# unlike the 72B stage, concurrent CPU calls don't serialize on GPU memory,
# so a few can genuinely run in parallel. 3 on a 24-core host leaves room
# for the 72B's own CPU-offloaded portion (it doesn't fully fit in 48GB
# combined VRAM) and the ops worker.
_SCREEN_POOL_ARGS = ["--pool=solo"] if sys.platform == "win32" else ["--pool=prefork", "-c", "3"]

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

SERVICES: dict[str, dict] = {
    # Ops worker — fast I/O tasks only (pollers, snapshots, stop checks).
    # LLM analysis lives on its own worker below so a multi-minute 72B call
    # can never block stop monitoring or the drawdown circuit breaker.
    "celery_worker": {
        "label": "Celery Worker (ops)",
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info",
                  "-Q", "ops", "-n", "ops@%h"] + _OPS_POOL_ARGS,
        "log":   LOG_DIR / "celery_worker.log",
    },
    # LLM worker — drains llm_p0 (events on held tickers) strictly before
    # llm_p1 (new-idea analyses); queue_order_strategy=sorted in tasks.py
    # makes the name order the priority order. Concurrency 1: the GPU
    # serializes 72B inference anyway, parallel calls just slow each other.
    # Only receives events that already passed screening (see below) —
    # added 2026-07-16 after a busy news day backed this queue up past
    # 700 items with screen+analyze combined in one task.
    "celery_llm_worker": {
        "label": "Celery Worker (LLM)",
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info",
                  "-Q", "llm_p0,llm_p1", "-n", "llm@%h",
                  "--prefetch-multiplier=1"] + _LLM_POOL_ARGS,
        "log":   LOG_DIR / "celery_llm_worker.log",
    },
    # Screening worker — fast 7B relevance filter, CPU-only (see
    # agents/screener.py), its own queue so a screen-reject never waits
    # behind the 72B stage. Only genuine survivors get forwarded to the
    # LLM worker above.
    "celery_screen_worker": {
        "label": "Celery Worker (screen)",
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info",
                  "-Q", "screen", "-n", "screen@%h"] + _SCREEN_POOL_ARGS,
        "log":   LOG_DIR / "celery_screen_worker.log",
    },
    "celery_beat": {
        "label": "Celery Beat",
        "cmd":   [CELERY, "-A", "tasks", "beat", "--loglevel=info"],
        "log":   LOG_DIR / "celery_beat.log",
    },
    "catalyst_service": {
        "label": "Catalyst Service",
        "cmd":   [UVICORN, "catalyst_service.main:app", "--host", "127.0.0.1", "--port", "8001"],
        "log":   LOG_DIR / "catalyst_service.log",
    },
}


_LOG_ROTATE_MAX_BYTES = 20_000_000


def _rotate_log_if_large(log_path: Path) -> None:
    """Subprocess logs are plain redirected stdout/stderr (opened with
    `open(path, "a")`, not a Python logging handler), so RotatingFileHandler
    doesn't apply — these grow unbounded otherwise. Rename to .1 (dropping
    any previous .1) right before a fresh start, since that's the one
    moment nothing is actively writing to the file."""
    try:
        if log_path.exists() and log_path.stat().st_size > _LOG_ROTATE_MAX_BYTES:
            backup = log_path.with_suffix(log_path.suffix + ".1")
            backup.unlink(missing_ok=True)
            log_path.rename(backup)
    except Exception:
        pass  # rotation is best-effort — never block a service start over it


def get_log(name: str, lines: int = 50) -> str:
    """Returns the last N lines of a service's log file."""
    if name not in SERVICES:
        return "Unknown service."
    log_path = SERVICES[name]["log"]
    if not log_path.exists():
        return "No log yet — start the service to begin logging."
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        return "\n".join(all_lines[-lines:])
    except Exception as e:
        return f"Could not read log: {e}"


def _load() -> dict[str, dict]:
    """Returns {service_name: {"pid": int, "created": float | None}}.
    Transparently upgrades the old {"name": pid_int} format on read."""
    if not PID_FILE.exists():
        return {}
    try:
        raw = json.loads(PID_FILE.read_text())
    except Exception:
        return {}
    entries: dict[str, dict] = {}
    for name, val in raw.items():
        if isinstance(val, dict):
            entries[name] = val
        else:
            entries[name] = {"pid": val, "created": None}
    return entries


def _save(pids: dict[str, dict]) -> None:
    PID_FILE.write_text(json.dumps(pids))


def _alive(entry: dict) -> bool:
    """A PID alone isn't proof of identity — after a crash, the OS can
    reuse that PID for an unrelated process, and a plain psutil.Process(pid)
    check would then report the service as running (and start() would skip
    launching a real one) when it's actually a stranger process. Compare
    the process's own create_time against what we recorded when we started
    it; a mismatch means the PID was recycled."""
    pid = entry.get("pid")
    if pid is None:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        if p.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return False
        created = entry.get("created")
        if created is not None and abs(p.create_time() - created) > 1.0:
            return False
        return True
    except Exception:
        return False


def ollama_status() -> tuple[bool, str]:
    """Returns (is_running, info_string) for Ollama."""
    try:
        import requests as _req
        resp = _req.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, "Models loaded: " + (", ".join(models) if models else "none")
        return False, "Ollama not responding"
    except Exception:
        return False, "Ollama not running"


def redis_ok() -> bool:
    try:
        import socket
        with socket.create_connection(("127.0.0.1", 6379), timeout=0.25):
            return True
    except Exception:
        return False


def get_status() -> dict[str, bool]:
    """Returns {service_name: is_running} for all services plus redis."""
    pids = _load()
    status: dict[str, bool] = {
        name: (_alive(pids[name]) if name in pids else False)
        for name in SERVICES
    }
    status["redis"] = redis_ok()
    return status


def start(name: str) -> str:
    """Start a service. Returns 'ok', 'already_running', or an error string."""
    if name not in SERVICES:
        return f"Unknown service: {name}"
    pids = _load()
    if name in pids and _alive(pids[name]):
        return "already_running"
    svc = SERVICES[name]
    try:
        import psutil
        flags    = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        extra    = {"PYTHONPATH": str(BASE_DIR)}
        # Each celery worker needs a distinct IBKR clientId base so it
        # doesn't conflict with main (clientId=1) or dashboard (clientId=2).
        # The base itself isn't enough for celery_worker: --pool=prefork -c 3
        # forks 3 children that would all inherit this one value — ibkr.py's
        # _resolve_client_id() adds each fork's billiard worker index (0,1,2)
        # on top of the base, so use bases spaced further apart than any
        # worker's concurrency (10 -> 10,11,12; 20 -> 20,21,22).
        if name == "celery_worker":
            extra["IB_CLIENT_ID"] = "10"
            # Each of the 3 prefork forks gets its own DB connection pool —
            # a shared default of 5/process would let this worker alone
            # claim up to 15 Supabase connections. Capped tighter since
            # ops tasks are quick, low-concurrency-per-fork I/O.
            extra["DB_MAXCONN"] = "2"
        elif name == "celery_llm_worker":
            extra["IB_CLIENT_ID"] = "20"
            extra["DB_MAXCONN"] = "2"   # single-concurrency worker, needs little
        elif name == "celery_screen_worker":
            # No IB_CLIENT_ID — screening never touches IBKR (just Ollama,
            # vector store, and db.log_signal), so it doesn't need a
            # broker connection at all.
            extra["DB_MAXCONN"] = "2"
        env      = {**__import__("os").environ, **extra}
        _rotate_log_if_large(svc["log"])
        log_file = open(svc["log"], "a", encoding="utf-8")
        proc = subprocess.Popen(
            svc["cmd"],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=log_file,
            creationflags=flags,
            env=env,
        )
        try:
            created = psutil.Process(proc.pid).create_time()
        except Exception:
            created = None
        pids[name] = {"pid": proc.pid, "created": created}
        _save(pids)
        return "ok"
    except Exception as e:
        return str(e)


def stop(name: str) -> str:
    """Stop a service. Returns 'ok' or an error string."""
    pids = _load()
    if name not in pids:
        return "ok"
    pid = pids[name].get("pid")
    result = "ok"
    try:
        import psutil
        if pid is not None and psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
    except Exception as e:
        # Still drop the PID entry below — a service that failed to
        # terminate cleanly shouldn't leave a stale "running" record that
        # makes a later start() think it's already up.
        result = str(e)
    finally:
        pids.pop(name, None)
        _save(pids)
    return result
