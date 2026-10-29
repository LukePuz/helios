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

import log_redaction  # noqa: E402

log_redaction.install()  # bot token / account id out of this process's logs


def _setup_logging() -> None:
    from logging.handlers import RotatingFileHandler

    log_path = Path(__file__).parent / "helios.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            # A plain FileHandler grows unbounded — on a machine that runs
            # every weekday indefinitely this eventually fills the disk,
            # at which point DB writes and everything else start failing
            # too. 10MB x 5 backups keeps ~50MB of history on disk.
            RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to {log_path}")


def _try_start_redis() -> None:
    """Try to start Redis via: remote URL passthrough → systemctl → redis-server → Docker."""
    import shutil
    import os as _os

    redis_url = _os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # If REDIS_URL points somewhere non-local, skip local startup entirely
    if "localhost" not in redis_url and "127.0.0.1" not in redis_url:
        logging.info(f"Using remote Redis ({redis_url}) — skipping local startup.")
        return

    # Strategy 1: systemctl (Linux systemd service)
    if shutil.which("systemctl"):
        r = subprocess.run(["systemctl", "is-active", "--quiet", "redis"],
                           capture_output=True)
        if r.returncode == 0:
            logging.info("Redis: already running via systemd.")
            return
        r2 = subprocess.run(["systemctl", "is-active", "--quiet", "redis-server"],
                             capture_output=True)
        if r2.returncode == 0:
            logging.info("Redis: already running via systemd (redis-server).")
            return

    # Strategy 2: native redis-server binary (apt/brew installed)
    if shutil.which("redis-server"):
        try:
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logging.info("Redis: started via redis-server.")
            return
        except Exception as e:
            logging.warning(f"redis-server start failed: {e}")

    # Strategy 3: Docker
    if shutil.which("docker"):
        try:
            r = subprocess.run(
                ["docker", "start", "redis"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                logging.info("Redis: started via Docker (existing container).")
                return
            r2 = subprocess.run(
                ["docker", "run", "-d", "--name", "redis", "-p", "6379:6379", "redis:alpine"],
                capture_output=True, text=True, timeout=30,
            )
            if r2.returncode == 0:
                logging.info("Redis: container created and started via Docker.")
                return
            logging.warning(f"Docker Redis failed: {r2.stderr.strip()}")
        except Exception as e:
            logging.warning(f"Docker Redis error: {e}")

    logging.warning(
        "Could not start Redis automatically. "
        "If Celery fails to connect, run:  sudo systemctl start redis  "
        "OR  docker start redis"
    )


def _adopt_position(client, rm, symbol, qty, manage_stop_place, resting_sell_stops):
    """Adopt one unrecorded IBKR position under its ticker lock. Re-checks
    first: another path may have written the record meanwhile, and a stop
    already resting covers its shares (audit 02:53, A0253-5)."""
    import db
    if db.get_position_stop(symbol):
        return None
    resting = sum(float(o["qty"]) for o in resting_sell_stops(symbol))
    need = round(abs(qty) - resting, 4)
    pos = client.get_open_position(symbol)
    entry_price = float(pos.avg_entry_price)
    stop_price, stop_method = rm.calculate_stop(symbol, entry_price, "Accelerator")
    db.upsert_position_stop(
        ticker=symbol, entry_price=entry_price, stop_price=stop_price,
        stop_method=f"AUTO-ADOPTED ({stop_method}) — verify bucket/stop manually",
        bucket="Accelerator", conviction=5, trail_distance_pct=None,
    )
    if need <= 0:
        return f"{symbol} ({qty} shares, record only: existing stops cover {resting:g})"
    try:
        stop_result = manage_stop_place(symbol, need, stop_price)
        db.set_stop_broker_order_id(symbol, stop_result["order_id"])
        return f"{symbol} ({need:g} shares, stop placed @ ${stop_price:.2f})"
    except Exception as e:
        return (f"{symbol} ({qty} shares, DB stop ${stop_price:.2f} — "
                f"BROKER STOP PLACEMENT FAILED: {e})")


def _reconcile_positions() -> None:
    """
    Compare DB-recorded positions against live IBKR holdings and ACT on
    discrepancies instead of only logging them:
      - An IBKR position with no DB stop record is invisible to the trailing-
        stop monitor (which reads db.get_all_position_stops(), not the
        broker directly). This adopts it: computes a conservative stop,
        places it at the broker, records it in the DB, and alerts that the
        bucket/conviction/stop were guessed and need human review.
      - A DB stop record with no matching IBKR position is left for the
        periodic `reconcile_stop_orders` Celery task (every 15 min) to clean
        up — this startup pass only reports it.
    """
    try:
        from ibkr import get_client
        import db

        client   = get_client()
        ibkr_pos = {p.symbol: float(p.qty) for p in client.get_all_positions()}
        db_pos   = db.get_open_position_symbols()

        in_ibkr_not_db = {s: q for s, q in ibkr_pos.items() if s not in db_pos}
        in_db_not_ibkr = {s for s in db_pos if s not in ibkr_pos}

        if not in_ibkr_not_db and not in_db_not_ibkr:
            logging.info("Position reconciliation: DB and IBKR agree.")
            return

        logging.warning("=== POSITION RECONCILIATION DISCREPANCY ===")
        if in_db_not_ibkr:
            logging.warning(
                f"  In DB stops table but NOT in IBKR: {in_db_not_ibkr} "
                "(periodic reconcile_stop_orders task will clean these up)"
            )

        if in_ibkr_not_db:
            from risk_manager import RiskManager
            from tasks import manage_stop_place
            rm = RiskManager(client)
            adopted: list[str] = []
            import stop_lock
            from tasks import _resting_sell_stops
            for symbol, qty in in_ibkr_not_db.items():
                try:
                    with stop_lock.ticker_lock(symbol):
                        msg = _adopt_position(client, rm, symbol, qty, manage_stop_place,
                                              _resting_sell_stops)
                    if msg:
                        adopted.append(msg)
                except Exception as e:
                    logging.error(f"Could not adopt orphaned position {symbol}: {e}")

            if adopted:
                logging.warning(f"  Adopted into monitoring (bucket/stop GUESSED — verify manually): {adopted}")
                try:
                    import alerts
                    alerts._send(
                        "🔧 <b>Position Reconciliation — Auto-Adopted</b>\n"
                        "These IBKR positions had no stop record. A conservative stop has been "
                        "computed and placed automatically — verify the bucket, conviction, and "
                        "stop level are appropriate:\n" + "\n".join(f"• {a}" for a in adopted)
                    )
                except Exception:
                    pass
    except Exception as e:
        logging.warning(f"Position reconciliation skipped: {e}")


def _reconcile_open_orders() -> None:
    """
    Cancels stale open MARKET BUY orders left over from a previous session
    -- a startup safety net for the incident found 2026-07-26: a DAY market
    BUY submitted while the market was closed gets queued by IBKR for the
    next session instead of rejected, and (before this and the
    place_market_order/execute_approved fixes landed) our own code declared
    the trade FAILED after a 30s wait without ever cancelling the
    now-orphaned order. Any such order still open at startup is disconnected
    from any active approval/trade record and would otherwise silently
    attempt to fill at today's open with no stop-loss and no bookkeeping.

    Scoped to plain "MKT" orders only -- bracket/limit entries (see
    config.USE_BRACKET_ORDERS) can legitimately sit open longer and aren't
    what this incident was about.
    """
    try:
        from ibkr import get_client, get_open_orders
        from tasks import manage_cancel_order
        get_client()  # ensure connected before querying
        stale = [
            o for o in get_open_orders()
            if o["action"] == "BUY" and o["order_type"] == "MKT"
        ]
        if not stale:
            logging.info("Open-order reconciliation: no stale market BUY orders.")
            return

        cancelled = []
        for o in stale:
            if manage_cancel_order(o["order_id"]):
                cancelled.append(f"{o['ticker']} ({o['qty']} shares, orderId={o['order_id']})")
            else:
                cancelled.append(f"{o['ticker']} — COULD NOT CANCEL orderId={o['order_id']}, check TWS manually")

        logging.warning(f"Open-order reconciliation: cancelled stale market BUY orders: {cancelled}")
        try:
            import alerts
            alerts._send(
                "🔧 <b>Startup Order Reconciliation</b>\n"
                "Cancelled stale market BUY order(s) left over from a previous session "
                "(orphaned by a market-closed submission, not tied to any active approval):\n"
                + "\n".join(f"• {c}" for c in cancelled)
            )
        except Exception:
            pass
    except Exception as e:
        logging.warning(f"Open-order reconciliation skipped: {e}")


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
            "Start it with: ollama serve   (or: sudo systemctl start ollama), "
            "then run: ollama pull qwen2.5:7b && ollama pull qwen2.5:72b"
        )


def _init_regime() -> None:
    try:
        from agents.market_regime import assess_regime
        logging.info("Running initial market regime assessment...")
        regime = assess_regime()
        logging.info(f"Initial regime: {regime['regime']} (score={regime['score']})")
    except Exception as e:
        logging.warning(f"Initial regime assessment failed: {e}")


def _start_telegram_bot() -> list[threading.Thread]:
    from telegram_bot import start_polling
    threads = start_polling()
    logging.info("Telegram bot polling + worker threads started.")
    return threads


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


_ORDERS_WORKER_WAIT_S = 180


def _orders_worker_ready(wait_s: float = _ORDERS_WORKER_WAIT_S) -> bool:
    """Poll until the orders worker answers a Celery ping."""
    import socket
    from tasks import app
    dest = [f"orders@{socket.gethostname()}"]
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            if app.control.ping(destination=dest, timeout=2):
                return True
        except Exception as e:
            logging.debug(f"orders worker ping failed: {e}")
        time.sleep(3)
    return False


def _startup_reconcile() -> None:
    """Startup position/order reconciliation. Both place or cancel orders
    through the orders queue, so they wait for the orders worker: run before
    it existed, an orphan market BUY couldn't be cancelled and would fill at
    the open with no stop (audit 00:53, A0053-3)."""
    if not _orders_worker_ready():
        logging.error("Startup reconciliation skipped: orders worker did not answer "
                      f"within {_ORDERS_WORKER_WAIT_S}s")
        try:
            import alerts
            alerts._send(
                "⚠️ <b>Startup reconciliation skipped</b>\n"
                "The orders worker didn't come up, so stale market BUYs were not "
                "cancelled and unstopped positions were not adopted. Check TWS."
            )
        except Exception:
            pass
        return
    _reconcile_positions()
    _reconcile_open_orders()


def _auto_start_services() -> None:
    """Start Celery Worker, Celery Beat, and Catalyst Service after a short delay."""
    def _start():
        time.sleep(6)  # Give the dashboard time to come up first
        import process_manager as pm

        def _start_one(name):
            result = pm.start(name)
            if result == "ok":
                logging.info(f"Auto-started {pm.SERVICES[name]['label']}.")
            elif result == "already_running":
                logging.info(f"{pm.SERVICES[name]['label']} already running.")
            else:
                logging.warning(f"Could not start {pm.SERVICES[name]['label']}: {result}")

        # Orders worker first, then the startup reconcile, then everything
        # else: with beat and the other workers already up, overdue
        # reconcile tasks raced the adoption and a fresh market BUY could be
        # cancelled as "stale" (audit 02:53, A0253-5).
        if "celery_orders_worker" in pm.SERVICES:
            _start_one("celery_orders_worker")
        _startup_reconcile()
        for name in pm.SERVICES:
            if name != "celery_orders_worker":
                _start_one(name)

    t = threading.Thread(target=_start, daemon=True, name="service-autostart")
    t.start()


def main() -> None:
    _setup_logging()

    print("\n" + "=" * 55)
    print("  Helios Hedgefund")
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

    # 5b/5c. Position and open-order reconciliation run once the orders
    # worker is up (see _startup_reconcile).

    # 6. Dashboard
    print("  [6/6] Starting dashboard + services...")
    dashboard_proc = _start_dashboard()

    # Auto-start Celery Worker, Beat, Catalyst in background
    _auto_start_services()

    def _shutdown(signum, frame):
        logging.info("Shutdown signal received — stopping all services.")
        try:
            import process_manager as pm
            for name, result in pm.stop_all().items():
                if result != "ok":
                    logging.warning(f"Shutdown: {name}: {result}")
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
    print("  Logs       -> helios.log")
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
