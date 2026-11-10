"""Telegram stops taking updates before the services stop, a running
approval gets until a deadline to finish, and the stop fits TimeoutStopSec
(audit 2026-11-08 15:08 #5)."""
import re
import threading
import time
from pathlib import Path
from unittest.mock import patch

import main
import process_manager as pm

ROOT = Path(__file__).resolve().parent.parent


def test_stop_budget_fits_timeout_stop_sec():
    unit = (ROOT / "deploy/systemd/helios-local.service").read_text()
    timeout = int(re.search(r"^TimeoutStopSec=(\d+)", unit, re.M).group(1))
    # the Telegram join runs inside stop_all before the orders worker, and its
    # deadline counts from the stop's start, so it overlaps the other graces
    worst = (pm._OTHER_GRACE_S + max(max(pm._CELERY_GRACE_S.values()), main._TELEGRAM_JOIN_S)
             + pm._DEFAULT_CELERY_GRACE_S)
    assert worst + 15 <= timeout


def test_join_covers_fill_wait_and_the_stop_call():
    import tasks
    import inspect
    stop_timeout = inspect.signature(tasks.manage_stop_place).parameters["timeout"].default
    assert main._TELEGRAM_JOIN_S >= 30 + stop_timeout + tasks._ORDERS_EXTRA_WAIT_S


def test_join_waits_for_the_worker_only_until_the_deadline():
    ev = threading.Event()
    worker = threading.Thread(target=ev.wait, name="telegram-worker", daemon=True)
    poller = threading.Thread(target=ev.wait, name="telegram-poller", daemon=True)
    worker.start(); poller.start()
    t0 = time.monotonic()
    main._join_telegram_worker([poller, worker], t0 + 0.3)
    took = time.monotonic() - t0
    ev.set()
    assert 0.25 < took < 1.0       # the poller is not joined


def test_join_returns_when_the_handler_finishes():
    worker = threading.Thread(target=lambda: time.sleep(0.1), name="telegram-worker")
    worker.start()
    t0 = time.monotonic()
    main._join_telegram_worker([worker], t0 + 30)
    assert time.monotonic() - t0 < 5 and not worker.is_alive()


def test_never_raises():
    with patch("telegram_bot.stop_polling", side_effect=RuntimeError("x")):
        main._stop_telegram_polling()
    main._join_telegram_worker(None, 0.0)


def test_shutdown_order():
    src = (ROOT / "main.py").read_text()
    body = src[src.index("def _shutdown("):]
    assert body.index("_stop_telegram_polling()") < body.index("pm.stop_all(") < body.index("sys.exit(0)")
    call = body[body.index("pm.stop_all("):body.index("for name, result")]
    assert "before_orders=" in call and "_join_telegram_worker(" in call
