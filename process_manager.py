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
PYTHON   = str(BASE_DIR / "venv" / "Scripts" / "python.exe")
CELERY   = str(BASE_DIR / "venv" / "Scripts" / "celery.exe")

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

SERVICES: dict[str, dict] = {
    "celery_worker": {
        "label": "Celery Worker",
        "cmd":   [CELERY, "-A", "tasks", "worker", "--loglevel=info", "--pool=solo"],
        "log":   LOG_DIR / "celery_worker.log",
    },
    "celery_beat": {
        "label": "Celery Beat",
        "cmd":   [CELERY, "-A", "tasks", "beat", "--loglevel=info"],
        "log":   LOG_DIR / "celery_beat.log",
    },
}


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


def _load() -> dict[str, int]:
    if PID_FILE.exists():
        try:
            return json.loads(PID_FILE.read_text())
        except Exception:
            pass
    return {}


def _save(pids: dict[str, int]) -> None:
    PID_FILE.write_text(json.dumps(pids))


def _alive(pid: int) -> bool:
    try:
        import psutil
        p = psutil.Process(pid)
        return p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
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
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_timeout=1)
        return bool(r.ping())
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
        flags    = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        env      = {**__import__("os").environ, "PYTHONPATH": str(BASE_DIR)}
        log_file = open(svc["log"], "a", encoding="utf-8")
        proc = subprocess.Popen(
            svc["cmd"],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=log_file,
            creationflags=flags,
            env=env,
        )
        pids[name] = proc.pid
        _save(pids)
        return "ok"
    except Exception as e:
        return str(e)


def stop(name: str) -> str:
    """Stop a service. Returns 'ok' or an error string."""
    pids = _load()
    if name not in pids:
        return "ok"
    pid = pids[name]
    try:
        import psutil
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
    except Exception as e:
        return str(e)
    del pids[name]
    _save(pids)
    return "ok"
