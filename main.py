"""
Main runner: starts the trading agent in a background thread and launches
the Streamlit dashboard. Press Ctrl+C to shut everything down cleanly.
"""

import logging
import signal
import subprocess
import sys
import threading
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


def _start_agent() -> threading.Thread:
    import trading_agent  # imported here so logging is set up first
    t = threading.Thread(target=trading_agent.run, name="TradingAgent", daemon=True)
    t.start()
    logging.info("Trading agent started in background thread.")
    return t


def _start_dashboard() -> subprocess.Popen:
    dashboard = str(Path(__file__).parent / "dashboard.py")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", dashboard,
         "--server.port", "8501",
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logging.info("Streamlit dashboard started at http://localhost:8501")
    return proc


def main() -> None:
    _setup_logging()

    # Initialize database tables before anything else
    import db
    db.init_db()

    agent_thread = _start_agent()
    dashboard_proc = _start_dashboard()

    def _shutdown(signum, frame):
        logging.info("Shutdown signal received — stopping.")
        dashboard_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (OSError, AttributeError):
        pass  # SIGTERM unavailable on some Windows configurations

    print("\n" + "=" * 50)
    print("  MANGOS Paper Trading Bot")
    print("  Dashboard → http://localhost:8501")
    print("  Logs     → trading_bot.log")
    print("  Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    agent_thread.join()


if __name__ == "__main__":
    main()
