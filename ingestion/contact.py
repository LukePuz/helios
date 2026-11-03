"""Contact address for outgoing User-Agent headers, from CONTACT_EMAIL in .env.

SEC EDGAR requires a contact address in the User-Agent and throttles or
blocks requests without one; Wikipedia asks for one too. It lives in .env so
the address isn't in the source.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip()
if not CONTACT_EMAIL:
    logging.getLogger(__name__).warning(
        "CONTACT_EMAIL is not set in .env; SEC EDGAR may reject requests without a contact address")

SEC_USER_AGENT = f"trading-bot {CONTACT_EMAIL}".strip()
BROWSER_USER_AGENT = (f"Mozilla/5.0 (trading-bot; +{CONTACT_EMAIL})" if CONTACT_EMAIL
                      else "Mozilla/5.0 (trading-bot)")

_startup_alert_sent = False


def alert_if_missing() -> bool:
    """Send one Telegram alert when CONTACT_EMAIL is unset. Called from
    main.main() at startup, not on import, so worker processes don't each
    send one. Sends at most once per process and never raises. Returns True
    if an alert was attempted."""
    global _startup_alert_sent
    if CONTACT_EMAIL or _startup_alert_sent:
        return False
    _startup_alert_sent = True
    try:
        import alerts
        alerts._send("CONTACT_EMAIL is not set in .env. SEC EDGAR requests go out with the "
                     "bare User-Agent 'trading-bot' and may be throttled or blocked.")
    except Exception as e:
        logging.getLogger(__name__).warning(f"CONTACT_EMAIL startup alert failed: {e}")
    return True
