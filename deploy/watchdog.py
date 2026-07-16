"""
External liveness watchdog — runs as its own systemd timer, independent of
main.py and Celery, so it can still notice and report a failure that takes
those processes down with it.

check_ingestion_health (tasks.py) is the only other health check in the
system, and it runs AS a Celery task — if the worker process dies, the
thing that would have alerted about it dies too. This script is the
external check: it verifies the managed processes are actually alive
(PID-reuse-safe, via process_manager's own liveness check) and that the
ops worker is still producing fresh portfolio snapshots, then posts
straight to the Telegram API — no dependency on anything else in the app
being healthy.

Intended to run every 10 minutes, 08:50-19:00 ET on weekdays (the awake
window) via helios-watchdog.timer.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))
load_dotenv(dotenv_path=_BASE_DIR / ".env")

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Services whose absence means the trading pipeline itself is down —
# catalyst_service is informational, not load-bearing, so it's excluded.
_CRITICAL_SERVICES = ["celery_worker", "celery_llm_worker", "celery_beat"]

# take_portfolio_snapshot runs every 5 minutes during market hours; anything
# much older than that means the ops worker has stopped processing tasks.
_SNAPSHOT_STALE_MINUTES = 15


def _send_telegram(text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        print(f"[watchdog] No Telegram credentials configured — would have sent: {text}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[watchdog] Telegram send failed: {e}")


def _check_processes() -> list[str]:
    import process_manager as pm

    pids = pm._load()
    problems = []
    for name in _CRITICAL_SERVICES:
        entry = pids.get(name)
        if entry is None or not pm._alive(entry):
            problems.append(f"{pm.SERVICES.get(name, {}).get('label', name)} is not running")
    return problems


def _check_snapshot_freshness() -> list[str]:
    try:
        import db
        conn = db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT timestamp FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1")
                row = cur.fetchone()
        finally:
            db._put_conn(conn)
    except Exception as e:
        return [f"Could not query portfolio_snapshots: {e}"]

    if row is None:
        return ["No portfolio snapshots exist yet"]

    age_minutes = (datetime.now(timezone.utc) - row[0]).total_seconds() / 60
    if age_minutes > _SNAPSHOT_STALE_MINUTES:
        return [f"Last portfolio snapshot is {age_minutes:.0f} min old "
                f"(expected every 5 min) — ops worker may be stuck or dead"]
    return []


def main() -> None:
    problems = _check_processes() + _check_snapshot_freshness()
    if problems:
        msg = "🚨 <b>Helios watchdog alert</b>\n" + "\n".join(f"• {p}" for p in problems)
        print(f"[watchdog] {msg}")
        _send_telegram(msg)
    else:
        print("[watchdog] OK")


if __name__ == "__main__":
    main()
