"""
Telegram Bot — polls for callback queries (inline button presses) and
executes approved recommendations.

Runs as a daemon thread started from main.py.
When a user clicks Approve on a pending recommendation:
  1. Fetches the approval record from DB
  2. Submits the order via Alpaca
  3. Logs the trade and stop level
  4. Sends a trade alert
  5. Updates the approval status

When a user clicks Reject:
  1. Updates approval status to REJECTED
  2. Edits the Telegram message to reflect the decision
"""
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID",  "")
_BASE_URL = f"https://api.telegram.org/bot{_TOKEN}"

_POLL_INTERVAL = 2   # seconds between getUpdates calls
_STOP_EVENT    = threading.Event()


# ---------------------------------------------------------------------------
# Telegram polling helpers
# ---------------------------------------------------------------------------

def _get_updates(offset: int) -> list[dict]:
    if not _TOKEN:
        return []
    try:
        resp = requests.get(
            f"{_BASE_URL}/getUpdates",
            params={"offset": offset, "timeout": 30, "allowed_updates": ["callback_query"]},
            timeout=35,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        log.debug(f"getUpdates failed: {e}")
        return []


def _answer_callback(callback_query_id: str, text: str = "") -> None:
    if not _TOKEN:
        return
    try:
        requests.post(
            f"{_BASE_URL}/answerCallbackQuery",
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=5,
        )
    except Exception as e:
        log.debug(f"answerCallbackQuery failed: {e}")


def _edit_message(message_id: int, text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        requests.post(
            f"{_BASE_URL}/editMessageText",
            json={
                "chat_id":    _CHAT_ID,
                "message_id": message_id,
                "text":       text,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception as e:
        log.debug(f"editMessageText failed: {e}")


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def _execute_approved_order(approval: dict) -> bool:
    """
    Submit the approved order to Alpaca, log the trade and stop.
    Returns True on success.
    """
    ticker   = approval["ticker"]
    action   = approval["action"]
    risk_json= approval.get("risk_json") or {}

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        import db
        import alerts as alert_module

        api_key    = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        client = TradingClient(api_key, api_secret, paper=True)

        # Determine shares and side
        shares = risk_json.get("shares_initial", 0)
        if shares <= 0:
            log.error(f"[{ticker}] No shares_initial in risk_json — cannot execute")
            return False

        if action in ("OPEN", "ADD"):
            side = OrderSide.BUY
        elif action in ("TRIM", "EXIT"):
            side = OrderSide.SELL
            # For EXIT: sell all; for TRIM: use shares from risk calc
            if action == "EXIT":
                try:
                    pos    = client.get_open_position(ticker)
                    shares = int(pos.qty)
                except Exception:
                    log.error(f"[{ticker}] EXIT requested but no position found")
                    return False
        else:
            log.info(f"[{ticker}] Action {action} does not require order submission")
            return True

        # Submit order
        order_req = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_req)
        log.info(f"[{ticker}] Order submitted: {side.value} {shares} shares (id={order.id})")

        # Get fill price (use last known or estimate)
        try:
            current_price = float(client.get_latest_trade(ticker).price)
        except Exception:
            current_price = risk_json.get("stop_loss_price", 0) * 1.3 or 0  # rough estimate

        # Log trade
        db.log_trade(
            symbol=ticker,
            side=side.value,
            quantity=shares,
            price=current_price,
            reason=f"{action} — approved via Telegram (conviction={approval['conviction']}/10)",
        )

        # Log stop level
        stop_price = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")
        if stop_price and side == OrderSide.BUY:
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=current_price,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
            )

        # Log outcome record for feedback loop
        db.log_trade_outcome(
            approval_id=approval["id"],
            ticker=ticker,
            action=action,
            conviction=approval.get("conviction", 5),
            bucket=approval.get("thesis_bucket", ""),
            entry_price=current_price,
        )

        # Send trade alert
        alert_module.send_trade_alert(
            symbol=ticker,
            side=side.value,
            qty=shares,
            price=current_price,
            reason=f"{action} approved",
        )

        return True

    except Exception as e:
        log.error(f"[{ticker}] Order execution failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------

def _handle_callback(update: dict) -> None:
    cq      = update.get("callback_query", {})
    cq_id   = cq.get("id", "")
    data    = cq.get("data", "")
    msg_id  = cq.get("message", {}).get("message_id")
    user    = cq.get("from", {}).get("username", "unknown")

    if not data or ":" not in data:
        return

    decision, approval_id_str = data.split(":", 1)
    try:
        approval_id = int(approval_id_str)
    except ValueError:
        log.warning(f"Invalid approval_id in callback: {approval_id_str}")
        return

    try:
        import db
        import alerts as alert_module

        approval = db.get_approval_by_id(approval_id)
        if not approval:
            _answer_callback(cq_id, "Approval not found.")
            return

        if approval["status"] != "PENDING":
            _answer_callback(cq_id, f"Already {approval['status']}.")
            return

        # Check expiry before executing
        expires_at = approval.get("expires_at")
        if expires_at and datetime.now(timezone.utc) > expires_at:
            db.update_approval_status(approval_id, "EXPIRED")
            _answer_callback(cq_id, "This recommendation has expired (>6 hours old).")
            if msg_id:
                ticker_exp = approval.get("ticker", "")
                action_exp = approval.get("action", "")
                _edit_message(
                    msg_id,
                    f"⏰ EXPIRED\n<b>{action_exp} {ticker_exp}</b>\nRecommendation expired before decision.",
                )
            return

        ticker = approval["ticker"]
        action = approval["action"]

        if decision == "approve":
            log.info(f"[{ticker}] Approval #{approval_id} approved by @{user}")
            _answer_callback(cq_id, "Processing order...")

            success = _execute_approved_order(approval)
            status  = "APPROVED" if success else "FAILED"
            db.update_approval_status(approval_id, status)

            now = datetime.now(timezone.utc)
            if msg_id:
                icon = "✅ APPROVED" if success else "⚠️ APPROVAL FAILED"
                _edit_message(
                    msg_id,
                    f"{icon}\n<b>{action} {ticker}</b>\n"
                    f"Approved by @{user}\n"
                    f"{now.strftime('%b %d %I:%M %p UTC')}",
                )

        elif decision == "reject":
            log.info(f"[{ticker}] Approval #{approval_id} rejected by @{user}")
            _answer_callback(cq_id, "Recommendation rejected.")
            db.update_approval_status(approval_id, "REJECTED")

            now = datetime.now(timezone.utc)
            if msg_id:
                _edit_message(
                    msg_id,
                    f"❌ REJECTED\n<b>{action} {ticker}</b>\n"
                    f"Rejected by @{user}\n"
                    f"{now.strftime('%b %d %I:%M %p UTC')}",
                )

    except Exception as e:
        log.error(f"Callback handler error: {e}")
        _answer_callback(cq_id, "Error processing decision.")


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------

def _poll_loop() -> None:
    log.info("Telegram bot polling started.")
    offset = 0

    while not _STOP_EVENT.is_set():
        updates = _get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            if "callback_query" in update:
                try:
                    _handle_callback(update)
                except Exception as e:
                    log.error(f"Unhandled error in callback handler: {e}")

        if not updates:
            time.sleep(_POLL_INTERVAL)

    log.info("Telegram bot polling stopped.")


def start_polling() -> threading.Thread:
    """Start the Telegram bot polling loop as a daemon thread."""
    if not _TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN not set — approval gate disabled.")
        return threading.Thread(target=lambda: None, daemon=True)

    t = threading.Thread(target=_poll_loop, name="telegram-bot", daemon=True)
    t.start()
    return t


def stop_polling() -> None:
    """Signal the polling loop to stop."""
    _STOP_EVENT.set()
