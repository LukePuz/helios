"""
Telegram alert sender. All functions are no-ops if credentials are missing.
Includes approval-request messages with inline buttons for human approval gate.
"""

import logging
import os
import time as _time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")
_ET      = ZoneInfo("America/New_York")
_BASE    = f"https://api.telegram.org/bot{_TOKEN}"


# ---------------------------------------------------------------------------
# Core send helpers
# ---------------------------------------------------------------------------

def _send(text: str, _max_attempts: int = 3) -> int | None:
    """Send a plain text message. Returns message_id or None. Retries up to 3× on failure."""
    if not _TOKEN or not _CHAT_ID:
        log.debug("Telegram not configured — alert skipped.")
        return None
    for attempt in range(1, _max_attempts + 1):
        try:
            resp = requests.post(
                f"{_BASE}/sendMessage",
                json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("message_id")
        except Exception as e:
            log.warning(f"Telegram send attempt {attempt}/{_max_attempts} failed: {e}")
            if attempt < _max_attempts:
                _time.sleep(2 ** attempt)  # 2s, 4s
    log.error(f"Telegram send failed after {_max_attempts} attempts — message lost.")
    return None


def _send_with_buttons(text: str, buttons: list[list[dict]], _max_attempts: int = 3) -> int | None:
    """
    Send a message with an InlineKeyboardMarkup.
    buttons is a list of rows, each row is a list of button dicts:
        {"text": "Label", "callback_data": "action:id"}
    Returns message_id or None. Retries up to 3× on failure.
    """
    if not _TOKEN or not _CHAT_ID:
        log.debug("Telegram not configured — approval request skipped.")
        return None
    for attempt in range(1, _max_attempts + 1):
        try:
            resp = requests.post(
                f"{_BASE}/sendMessage",
                json={
                    "chat_id":      _CHAT_ID,
                    "text":         text,
                    "parse_mode":   "HTML",
                    "reply_markup": {"inline_keyboard": buttons},
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("message_id")
        except Exception as e:
            log.warning(f"Telegram button send attempt {attempt}/{_max_attempts} failed: {e}")
            if attempt < _max_attempts:
                _time.sleep(2 ** attempt)  # 2s, 4s
    log.error(f"Telegram button send failed after {_max_attempts} attempts — approval message lost.")
    return None


def _edit_message(message_id: int, new_text: str) -> None:
    """Edit an existing message (used to update approval status)."""
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        requests.post(
            f"{_BASE}/editMessageText",
            json={
                "chat_id":    _CHAT_ID,
                "message_id": message_id,
                "text":       new_text,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception as e:
        log.debug(f"Telegram edit failed: {e}")


def _answer_callback(callback_query_id: str, text: str = "") -> None:
    """Acknowledge a button press (removes the loading spinner)."""
    if not _TOKEN:
        return
    try:
        requests.post(
            f"{_BASE}/answerCallbackQuery",
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=5,
        )
    except Exception as e:
        log.debug(f"answerCallbackQuery failed: {e}")


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

def send_approval_request(
    approval_id: int,
    ticker: str,
    action: str,
    conviction: int,
    bucket: str,
    size_pct: float,
    stop_price: float | None,
    entry_strategy: str,
    reasoning: str,
) -> int | None:
    """
    Send a Telegram message with Approve / Reject inline buttons.
    Returns the Telegram message_id so it can be stored in the DB.
    """
    action_icon = {
        "OPEN":  "🟢", "ADD":  "🔵", "TRIM": "🟡",
        "EXIT":  "🔴", "WATCH":"👁️",
    }.get(action, "⚪")

    bucket_icon = {"Compounder": "🏛️", "Accelerator": "🚀", "Moonshot": "🌙"}.get(bucket, "")

    stop_line = f"Stop:       ${stop_price:.2f}\n" if stop_price else ""
    text = (
        f"{action_icon} <b>RECOMMENDATION — {action} {ticker}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Bucket:     {bucket_icon} {bucket}\n"
        f"Conviction: {conviction}/10\n"
        f"Size:       {size_pct:.1%} of portfolio\n"
        f"{stop_line}"
        f"Entry:      {entry_strategy}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>{reasoning[:400]}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"ID: #{approval_id}"
    )

    buttons = [[
        {"text": "✅ Approve",  "callback_data": f"approve:{approval_id}"},
        {"text": "❌ Reject",   "callback_data": f"reject:{approval_id}"},
    ]]

    return _send_with_buttons(text, buttons)


def update_approval_message(
    message_id: int,
    ticker: str,
    action: str,
    decision: str,
    decided_at: datetime | None = None,
) -> None:
    """Edit the approval message to show the decision made."""
    icon = "✅ APPROVED" if decision == "APPROVED" else "❌ REJECTED"
    ts   = (decided_at or datetime.now(_ET)).strftime("%b %d %I:%M %p ET")
    _edit_message(
        message_id,
        f"{icon}\n<b>{action} {ticker}</b>\nDecided: {ts}",
    )


def ack_callback(callback_query_id: str, decision: str) -> None:
    _answer_callback(callback_query_id, f"{decision} recorded.")


# ---------------------------------------------------------------------------
# Existing alert functions (unchanged)
# ---------------------------------------------------------------------------

def send_trade_alert(symbol: str, side: str, qty: int, price: float, reason: str) -> None:
    icon = "🟢" if side == "BUY" else "🔴"
    now  = datetime.now(_ET).strftime("%I:%M %p ET")
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
    pct  = daily_pl / start_value
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


def send_stop_triggered(ticker: str, current_price: float, stop_price: float, reason: str) -> None:
    _send(
        f"🛑 <b>STOP TRIGGERED — {ticker}</b>\n"
        f"Current: ${current_price:.2f}\n"
        f"Stop:    ${stop_price:.2f}\n"
        f"Reason:  {reason}"
    )


def send_catalyst_conflict(
    ticker: str,
    event_type: str,
    window_start: str | None,
    window_end: str | None,
    note: str | None = None,
) -> None:
    window = f"{window_start} to {window_end}" if window_start and window_end else "date unresolved"
    _send(
        f"<b>CATALYST CONFLICT - {ticker}</b>\n"
        f"Type:   {event_type}\n"
        f"Window: {window}\n"
        f"{note or 'Resolve before any catalyst-based trade gating relies on it.'}"
    )
