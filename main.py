"""
Main entry point.

Starts everything in order:
  1. DB init (creates tables if missing)
  2. Redis (via Docker, if not already running)
  3. Market regime assessment (daily baseline)
  4. Telegram bot polling thread (approval callbacks)
  5. Streamlit dashboard
  6. Celery Worker + Beat + Catalyst Service (auto-started after 6s)

Press Ctrl+C or close the window to stop everything.
"""

import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _setup_logging() -> None:
    log_path = Path(__file__).parent / "trading_bot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to {log_path}")


def _try_start_redis() -> None:
    """Start the Redis Docker container, creating it if needed."""
    try:
        # Try to start an existing container named 'redis'
        result = subprocess.run(
            ["docker", "start", "redis"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            logging.info("Redis container started.")
            return
        # Container doesn't exist yet — create it
        result2 = subprocess.run(
            ["docker", "run", "-d", "--name", "redis", "-p", "6379:6379", "redis:alpine"],
            capture_output=True, text=True, timeout=30,
        )
        if result2.returncode == 0:
            logging.info("Redis container created and started.")
        else:
            logging.warning(
                f"Could not start Redis via Docker: {result2.stderr.strip()}\n"
                "Make sure Docker Desktop is running, then retry."
            )
    except FileNotFoundError:
        logging.warning(
            "Docker not found in PATH. Start Redis manually: "
            "docker run -d --name redis -p 6379:6379 redis:alpine"
        )
    except Exception as e:
        logging.warning(f"Redis auto-start failed: {e}")


def _check_ollama() -> None:
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            logging.info(f"Ollama running. Models: {', '.join(models) or 'none loaded'}")
        else:
            logging.warning("Ollama is reachable but returned an unexpected status.")
    except Exception:
        logging.warning(
            "Ollama not running. AI reasoning will use keyword fallback. "
            "Start Ollama from the system tray, then run: ollama pull qwen2.5:7b"
        )


def _init_regime() -> None:
    try:
        from agents.market_regime import assess_regime
        logging.info("Running initial market regime assessment...")
        regime = assess_regime()
        logging.info(f"Initial regime: {regime['regime']} (score={regime['score']})")
    except Exception as e:
        logging.warning(f"Initial regime assessment failed: {e}")


def _start_telegram_bot() -> threading.Thread:
    from telegram_bot import start_polling
    t = start_polling()
    logging.info("Telegram bot polling thread started.")
    return t


def _start_dashboard() -> subprocess.Popen:
    import os as _os
    dashboard = str(Path(__file__).parent / "dashboard.py")
    # Give the dashboard clientId=2 so clientId=1 stays reserved for the
    # main process (Telegram bot → order execution). TWS allows up to 32
    # simultaneous connections with unique IDs.
    env = {**_os.environ, "IB_CLIENT_ID": "2"}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", dashboard,
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logging.info("Streamlit dashboard started at http://localhost:8501")
    return proc


def _auto_start_services() -> None:
    """Start Celery Worker, Celery Beat, and Catalyst Service after a short delay."""
    def _start():
        time.sleep(6)  # Give the dashboard time to come up first
        import process_manager as pm
        for name in pm.SERVICES:
            result = pm.start(name)
            if result == "ok":
                logging.info(f"Auto-started {pm.SERVICES[name]['label']}.")
            elif result == "already_running":
                logging.info(f"{pm.SERVICES[name]['label']} already running.")
            else:
                logging.warning(f"Could not start {pm.SERVICES[name]['label']}: {result}")

    t = threading.Thread(target=_start, daemon=True, name="service-autostart")
    t.start()


def main() -> None:
    _setup_logging()

    print("\n" + "=" * 55)
    print("  Trading Bot — Long-Duration Compounding System")
    print("=" * 55)

    # 1. Redis
    print("  [1/6] Starting Redis...")
    _try_start_redis()

    # 2. Database
    print("  [2/6] Initializing database...")
    import db
    db.init_db()
    logging.info("Database ready.")

    # 3. Ollama check
    print("  [3/6] Checking Ollama...")
    _check_ollama()

    # 4. Regime assessment
    print("  [4/6] Running regime assessment...")
    _init_regime()

    # 5. Telegram bot
    print("  [5/6] Starting Telegram bot...")
    _start_telegram_bot()

    # 6. Dashboard
    print("  [6/6] Starting dashboard + services...")
    dashboard_proc = _start_dashboard()

    # Auto-start Celery Worker, Beat, Catalyst in background
    _auto_start_services()

    def _shutdown(signum, frame):
        logging.info("Shutdown signal received — stopping all services.")
        try:
            import process_manager as pm
            for name in pm.SERVICES:
                pm.stop(name)
        except Exception:
            pass
        try:
            from telegram_bot import stop_polling
            stop_polling()
        except Exception:
            pass
        dashboard_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (OSError, AttributeError):
        pass

    print()
    print("  Dashboard  -> http://localhost:8501")
    print("  Logs       -> trading_bot.log")
    print()
    print("  All services starting automatically...")
    print("  Press Ctrl+C to stop everything.")
    print("=" * 55 + "\n")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
