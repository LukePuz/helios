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
# Single concurrency is load-bearing, not a performance choice — see
# celery_orders_worker's comment below.
_ORDERS_POOL_ARGS = ["--pool=solo"] if sys.platform == "win32" else ["--pool=prefork", "-c", "1"]
# Screening is CPU-forced 7B inference (see agents/screener.py num_gpu=0) —
# unlike the 72B stage, concurrent CPU calls don't serialize on GPU memory,
# so a few can genuinely run in parallel. 3 on a 24-core host leaves room
# for the 72B's own CPU-offloaded portion (it doesn't fully fit in 48GB
# combined VRAM) and the ops worker.
_SCREEN_POOL_ARGS = ["--pool=solo"] if sys.platform == "win32" else ["--pool=prefork", "-c", "3"]
# Ingestion pollers are pure I/O-wait (external HTTP calls) — higher
# concurrency than the CPU-bound workers above is cheap and lets several
# slow/degraded sources (EDGAR, options flow, etc.) run in parallel instead
# of queueing behind each other on top of already being isolated from ops.
_INGESTION_POOL_ARGS = ["--pool=solo"] if sys.platform == "win32" else ["--pool=prefork", "-c", "4"]

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
    # Ingestion worker — external-data pollers only (EDGAR, options flow,
    # macro, arXiv, congressional, insider, analyst ratings, news,
    # fundamentals, price history). Added 2026-08-18 after poll_edgar was
    # found taking 28-31 minutes per run against a degraded SEC endpoint
    # while sharing ops's 3 workers with check_portfolio_drawdown,
    # take_portfolio_snapshot, and reconcile_stop_orders — see
    # tasks.INGESTION_QUEUE's comment for the full incident (752 unrelated
    # tasks silently discarded as expired in one day, surfacing as an
    # hourly repeat Telegram alert that never resolved). An unreliable
    # external data source must never be able to starve safety-critical or
    # fast local tasks just by sharing a worker pool with them.
    "celery_ingestion_worker": {
        "label": "Celery Worker (ingestion)",
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info",
                  "-Q", "ingestion", "-n", "ingestion@%h"] + _INGESTION_POOL_ARGS,
        "log":   LOG_DIR / "celery_ingestion_worker.log",
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
        # llm_p2 (chat-requested "analyze TICKER" research, see
        # tasks.LLM_QUEUE_RESEARCH) sorts and drains last, behind held-ticker
        # events and organic news-driven ideas — same worker/concurrency,
        # just always at the back of the line.
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info",
                  "-Q", "llm_p0,llm_p1,llm_p2", "-n", "llm@%h",
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
    # Orders worker — the ONLY process in the system allowed to place or
    # cancel a stop order at the broker. Added 2026-08-11 after tracing
    # MRVL/SNDK/VRT/ORCL/DDOG's stuck-order incidents to a single root
    # cause: IBKR silently ignores cancelOrder()/modify calls from a
    # clientId that didn't place the order (confirmed repeatedly, including
    # live-testing that clientId=0's documented "master client" status does
    # NOT actually grant cancel rights over another client's order in this
    # account/TWS config — see tasks.py's manage_stop_* docstrings). Before
    # this, a stop got PLACED under whichever process handled the entry
    # (celery_llm_worker, clientId=20, or main.py, clientId=1) but every
    # LATER touch — trailing-stop conversion, orphan cleanup, a manual
    # sell's pre-cancel — ran from celery_worker's ops pool (clientId
    # 10/11/12, whichever of its 3 forks happened to pick up the task) —
    # a guaranteed mismatch, not an intermittent one, for every stop this
    # system has ever placed.
    #
    # Fix: exactly one process, one clientId, one order of operations for
    # a stop's entire lifecycle. Every other process (LLM worker, ops
    # worker, main.py, dashboard) now dispatches here via
    # tasks.manage_stop_place/manage_cancel_stop_orders_for/
    # manage_cancel_order instead of calling ibkr.py's placement/cancel
    # functions directly. Concurrency MUST stay at 1 — the whole point is
    # a single, unfragmented clientId; forking this the way celery_worker
    # forks would just reintroduce the same bug one level down.
    "celery_orders_worker": {
        "label": "Celery Worker (orders)",
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info",
                  "-Q", "orders", "-n", "orders@%h"] + _ORDERS_POOL_ARGS,
        "log":   LOG_DIR / "celery_orders_worker.log",
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
        elif name == "celery_ingestion_worker":
            # No IB_CLIENT_ID — pollers only hit external HTTP APIs (EDGAR,
            # yfinance, etc.) and the DB, never IBKR. -c 4 forks, so cap
            # tight like the other multi-fork workers above.
            extra["DB_MAXCONN"] = "2"
        elif name == "celery_orders_worker":
            # Base 30 -- clear of every other worker's base+fork-index range
            # (10-12, 20). -c 1 (see _ORDERS_POOL_ARGS) means this base is
            # never offset by a fork index, so IB_CLIENT_ID=30 IS this
            # worker's clientId, always -- other code checks os.environ for
            # exactly this value to decide whether it's already running
            # inside the one true orders process (see tasks.py's
            # manage_stop_* functions).
            extra["IB_CLIENT_ID"] = "30"
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


# Warm shutdown for Celery workers (limitations audit 18:53, #7). terminate()
# on every pool child killed whatever task was running (21 WorkerLostError on
# 10-28, all at restarts). SIGTERM to the worker's main process alone makes
# Celery stop taking tasks and wait for the running ones; after the grace
# period the rest is killed. The LLM worker gets the 72B timeout (400 s) plus
# margin; the others' tasks are short, except the ingestion worker's ~2.5 h
# S&P refresh, which still gets cut off (it needs to be resumable instead).
_CELERY_GRACE_S = {"celery_llm_worker": 420}
_DEFAULT_CELERY_GRACE_S = 60
_OTHER_GRACE_S = 5


def _is_celery_worker(name: str) -> bool:
    cmd = SERVICES.get(name, {}).get("cmd", [])
    return "worker" in cmd and "-A" in cmd


def _grace(name: str) -> int:
    if _is_celery_worker(name):
        return _CELERY_GRACE_S.get(name, _DEFAULT_CELERY_GRACE_S)
    return _OTHER_GRACE_S


def _request_stop(name: str):
    """Ask one service to stop without waiting. Returns its psutil.Process
    (or None if it isn't running)."""
    import psutil
    pid = (_load().get(name) or {}).get("pid")
    if pid is None or not psutil.pid_exists(pid):
        return None
    proc = psutil.Process(pid)
    if not _is_celery_worker(name):
        for child in proc.children(recursive=True):
            child.terminate()
    proc.terminate()
    return proc


def _finish_stop(name: str, proc, deadline: float) -> str:
    """Wait for a requested stop until deadline, then kill what's left."""
    import time
    import psutil
    result = "ok"
    try:
        if proc is not None:
            children = proc.children(recursive=True)
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except psutil.TimeoutExpired:
                result = f"killed after {_grace(name)}s grace"
                proc.kill()
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
    except Exception as e:
        result = str(e)
    return result


def stop(name: str) -> str:
    """Stop a service. Returns 'ok' or an error string."""
    import time
    pids = _load()
    if name not in pids:
        return "ok"
    try:
        proc = _request_stop(name)
        result = _finish_stop(name, proc, time.monotonic() + _grace(name))
    except Exception as e:
        # Still drop the PID entry below — a service that failed to
        # terminate cleanly shouldn't leave a stale "running" record that
        # makes a later start() think it's already up.
        result = str(e)
    finally:
        pids = _load()
        pids.pop(name, None)
        _save(pids)
    return result


def stop_all() -> dict[str, str]:
    """Stop every service: beat first (no new tasks), then everything else
    asked at once and waited on in parallel, so the whole stop takes about
    the longest grace period rather than the sum."""
    import time
    results = {}
    if "celery_beat" in _load():
        results["celery_beat"] = stop("celery_beat")
    pending = {}
    for name in SERVICES:
        # The orders worker goes last: the other workers' running tasks
        # dispatch stop orders to it during their grace (professor M57).
        if name in results or name not in _load() or name == "celery_orders_worker":
            continue
        try:
            pending[name] = (_request_stop(name), time.monotonic() + _grace(name))
        except Exception as e:
            results[name] = str(e)
    for name, (proc, deadline) in pending.items():
        results[name] = _finish_stop(name, proc, deadline)
    if "celery_orders_worker" in _load():
        results["celery_orders_worker"] = stop("celery_orders_worker")
    pids = _load()
    for name in results:
        pids.pop(name, None)
    _save(pids)
    return results
