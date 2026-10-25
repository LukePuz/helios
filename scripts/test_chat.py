"""
Quick test for the group chat assistant.
Run with: python scripts/test_chat.py

Tests:
  1. Trade intent parsing (no external deps)
  2. chat_assistant.respond() via Ollama (IBKR offline is fine)
  3. Sends a real message to Telegram so you can confirm the bot replies
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# 1. Trade intent parsing
# ---------------------------------------------------------------------------
print("=== Trade intent parsing ===")
from chat_assistant import parse_trade_intent

cases = [
    ("buy 5 NVDA",                {"side": "BUY",  "qty": 5.0,  "ticker": "NVDA"}),
    ("sell TSLA 10 shares",       {"side": "SELL", "qty": 10.0, "ticker": "TSLA"}),
    ("buy AAPL !urgent",          {"side": "BUY",  "qty": None, "ticker": "AAPL"}),
    ("sell 2.5 shares of AMD",    {"side": "SELL", "qty": 2.5,  "ticker": "AMD"}),
    ("what is the latest on NVDA", None),  # not a trade
]

all_ok = True
for text, expected in cases:
    result = parse_trade_intent(text)
    ok = result == expected
    status = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{status}] '{text}' -> {result}  (expected {expected})")

print()

# ---------------------------------------------------------------------------
# 2. chat_assistant.respond via Ollama
# ---------------------------------------------------------------------------
print("=== Ollama response (IBKR offline is fine) ===")
from chat_assistant import respond

try:
    reply = respond("how many shares of NVDA do we have?", "testuser")
    print(f"  Response: {reply[:300]}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# ---------------------------------------------------------------------------
# 3. Send a real Telegram message to confirm bot is reachable
# ---------------------------------------------------------------------------
print("=== Telegram send test ===")
import requests

TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

if not TOKEN or not CHAT_ID:
    print("  SKIP — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
else:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": "🧪 Chat assistant test — bot is reachable.", "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"  OK — message sent (id={resp.json()['result']['message_id']})")
    except Exception as e:
        print(f"  ERROR: {e}")

print()
print("Done." if all_ok else "Done (some parsing tests FAILED — check output above).")
