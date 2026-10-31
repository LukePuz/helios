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
