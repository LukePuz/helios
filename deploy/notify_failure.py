"""
Sends a Telegram alert when a systemd unit fails, via OnFailure=. Deliberately
minimal and self-contained (no imports from the app) — it exists specifically
for the case where the main stack itself failed to start, so it can't depend
on anything the main stack provides.

Usage: notify_failure.py <failed-unit-name>
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=_BASE_DIR / ".env")

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def main() -> None:
    unit = sys.argv[1] if len(sys.argv) > 1 else "unknown unit"
    text = (
        f"🚨 <b>systemd unit failed: {unit}</b>\n"
        f"Likely causes: IBKR/TWS 2FA hung waiting for a factor no script can "
        f"provide, TWS not running, or a startup exception.\n"
        f"Check: <code>systemctl status {unit}</code> and "
        f"<code>journalctl -u {unit} -n 50</code>."
    )
    if not _TOKEN or not _CHAT_ID:
        print(f"[notify_failure] No Telegram credentials — would have sent: {text}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[notify_failure] Telegram send failed: {e}")


if __name__ == "__main__":
    main()
