"""
Telegram alert sender. All functions are no-ops if credentials are missing.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_ET = ZoneInfo("America/New_York")


def _send(text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        logging.debug("Telegram not configured — alert skipped.")
        return
    try:
        url = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
        requests.post(
            url,
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        logging.warning(f"Telegram alert failed: {e}")


def send_trade_alert(symbol: str, side: str, qty: int, price: float, reason: str) -> None:
    icon = "🟢" if side == "BUY" else "🔴"
    now = datetime.now(_ET).strftime("%I:%M %p ET")
    _send(
        f"{icon} <b>TRADE EXECUTED</b>\n"
        f"Symbol:  <b>{symbol}</b>\n"
        f"Side:    {side}  ×{qty} shares\n"
        f"Price:   ${price:.2f}\n"
        f"Reason:  {reason}\n"
        f"Time:    {now}"
    )


def send_daily_summary(portfolio_value: float, daily_pl: float, start_value: float = 100_000.0) -> None:
    sign = "+" if daily_pl >= 0 else ""
    pct = daily_pl / start_value
    date = datetime.now(_ET).strftime("%b %d, %Y")
    _send(
        f"📊 <b>Daily Summary — {date}</b>\n"
        f"Portfolio: ${portfolio_value:,.2f}\n"
        f"P&amp;L vs Start: {sign}${daily_pl:,.2f} ({sign}{pct:.2%})"
    )


def send_halt_alert(current_value: float, start_value: float) -> None:
    drawdown = (start_value - current_value) / start_value
    _send(
        f"🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b>\n"
        f"Drawdown: -{drawdown:.1%}\n"
        f"Current:  ${current_value:,.2f}\n"
        f"Start:    ${start_value:,.2f}\n"
        f"All trading is paused. Restart manually to resume."
    )
