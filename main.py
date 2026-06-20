"""
Main entry point.

Starts:
  1. DB init (creates tables)
  2. Market regime assessment (daily baseline)
  3. Telegram bot polling thread (for approval callbacks)
  4. Streamlit dashboard

Celery worker + beat (task scheduler) are started separately via the dashboard
service controls or CLI.

Press Ctrl+C to stop.
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


def _init_regime() -> None:
    """Run an initial regime assessment so the dashboard has data immediately."""
    try:
        from agents.market_regime import assess_regime
        logging.info("Running initial market regime assessment...")
        regime = assess_regime()
        logging.info(f"Initial regime: {regime['regime']} (score={regime['score']})")
    except Exception as e:
        logging.warning(f"Initial regime assessment failed: {e}")


def _start_telegram_bot() -> threading.Thread:
    """Start the Telegram bot polling thread for approval callbacks."""
    from telegram_bot import start_polling
    t = start_polling()
    logging.info("Telegram bot polling thread started.")
    return t


def _start_dashboard() -> subprocess.Popen:
    dashboard = str(Path(__file__).parent / "dashboard.py")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", dashboard,
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logging.info("Streamlit dashboard started at http://localhost:8501")
    return proc


def main() -> None:
    _setup_logging()

    # Initialize all DB tables (original + new)
    import db
    db.init_db()
    logging.info("Database ready.")

    # Initial regime snapshot
    _init_regime()

    # Start Telegram bot for approval callbacks
    _start_telegram_bot()

    # Launch dashboard
    dashboard_proc = _start_dashboard()

    def _shutdown(signum, frame):
        logging.info("Shutdown signal received — stopping.")
        from telegram_bot import stop_polling
        stop_polling()
        dashboard_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (OSError, AttributeError):
        pass

    print("\n" + "=" * 55)
    print("  Trading Bot — Long-Duration Compounding System")
    print("  Dashboard  -> http://localhost:8501")
    print("  Logs       -> trading_bot.log")
    print()
    print("  Start Celery worker:  celery -A tasks worker --loglevel=info --pool=solo")
    print("  Start Celery beat:    celery -A tasks beat --loglevel=info")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 55 + "\n")

    # Keep main thread alive
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
