"""
Telegram Bot — polls for callback queries (inline button presses) and
executes approved recommendations. Also handles @mention group chat messages
with a priority queue so approvals always take precedence over chat.

Priority queue levels:
  0 — !urgent messages (trade commands or time-sensitive questions)
  1 — callback_query (Approve/Reject button presses)
  2 — regular @mention chat messages

Runs as two daemon threads: _poll_loop (enqueues) and _worker_loop (processes).
"""
import itertools
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import log_redaction  # noqa: E402

log_redaction.install()  # bot token / account id out of this process's logs

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID",  "")
_BASE_URL = f"https://api.telegram.org/bot{_TOKEN}"

_POLL_INTERVAL = 2   # seconds between getUpdates calls
_STOP_EVENT    = threading.Event()
# Entries are (priority, seq, update). The monotonic seq breaks priority
# ties — without it heapq falls through to comparing the update dicts and
# raises TypeError, which killed the poller thread silently.
_UPDATE_QUEUE: queue.PriorityQueue = queue.PriorityQueue()
_QUEUE_SEQ = itertools.count()

# Populated at startup via _fetch_bot_username()
_BOT_USERNAME: str = os.getenv("TELEGRAM_BOT_USERNAME", "")

# Comma-separated Telegram numeric user IDs allowed to @mention the bot.
# Find your ID by messaging @userinfobot in Telegram.
_ALLOWED_USER_IDS: set[int] = {
    int(u.strip())
    for u in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").split(",")
    if u.strip().lstrip("-").isdigit()
}


# ---------------------------------------------------------------------------
# Telegram polling helpers
# ---------------------------------------------------------------------------

def _fetch_bot_username() -> None:
    """Populate _BOT_USERNAME from the Telegram API if not set via env."""
    global _BOT_USERNAME
    if _BOT_USERNAME or not _TOKEN:
        return
    try:
        resp = requests.get(f"{_BASE_URL}/getMe", timeout=10)
        resp.raise_for_status()
        _BOT_USERNAME = resp.json().get("result", {}).get("username", "")
        log.info(f"Bot username resolved: @{_BOT_USERNAME}")
    except Exception as e:
        log.warning(f"Could not resolve bot username: {e}")


def _send_chat_action(chat_id: str | int, action: str = "typing") -> None:
    try:
        requests.post(
            f"{_BASE_URL}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
            timeout=5,
        )
    except Exception:
        pass


def _reply(chat_id: str | int, text: str, reply_to: int | None = None) -> None:
    payload: dict = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        requests.post(f"{_BASE_URL}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        log.debug(f"_reply failed: {e}")


def _get_updates(offset: int) -> list[dict]:
    if not _TOKEN:
        return []
    try:
        resp = requests.get(
            f"{_BASE_URL}/getUpdates",
            params={"offset": offset, "timeout": 30,
                    "allowed_updates": ["callback_query", "message"]},
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
# Moved to execution.py — shared by the auto-execute pipeline (crew.py),
# this module's own callback handler below, !urgent chat commands, and the
# dashboard's manual trade form, so a fix in one path fixes all four. This
# name is kept as a thin alias since it's referenced from here and from
# agents/crew.py.
from execution import execute_approved as _execute_approved_order


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------

def _handle_callback(update: dict) -> None:
    cq      = update.get("callback_query", {})
    cq_id   = cq.get("id", "")
    data    = cq.get("data", "")
    msg_id  = cq.get("message", {}).get("message_id")
    from_   = cq.get("from", {})
    user    = from_.get("username") or from_.get("first_name") or str(from_.get("id", "unknown"))

    # Authorization — these buttons execute orders. Mentions are gated on
    # _ALLOWED_USER_IDS in _handle_mention; callbacks must be too, and they
    # fail CLOSED when no allowlist is configured (anyone in the chat could
    # otherwise fire trades).
    user_id = from_.get("id")
    if not _ALLOWED_USER_IDS:
        log.error("Callback rejected — TELEGRAM_ALLOWED_USER_IDS is not set; "
                  "approval buttons are disabled until it is.")
        _answer_callback(cq_id, "Approvals disabled: no allowlist configured.")
        return
    if user_id not in _ALLOWED_USER_IDS:
        log.warning(f"Blocked callback from unauthorized user_id={user_id} ({user})")
        _answer_callback(cq_id, "Not authorized.")
        return
    cb_chat_id = cq.get("message", {}).get("chat", {}).get("id")
    try:
        if cb_chat_id is not None and _CHAT_ID and int(cb_chat_id) != int(_CHAT_ID):
            log.warning(f"Blocked callback from unexpected chat_id={cb_chat_id}")
            _answer_callback(cq_id, "Not authorized.")
            return
    except (TypeError, ValueError):
        pass

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

        if approval["status"] not in ("PENDING", "PROCESSING"):
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
            # Atomic claim — only one callback wins the race
            if not db.claim_approval_for_execution(approval_id):
                _answer_callback(cq_id, "Already being processed.")
                return
            log.info(f"[{ticker}] Approval #{approval_id} approved by @{user}")
            try:
                from feed import emit as _fe
                _fe("telegram", ticker, f"Approved by @{user} — executing {action} {ticker}", "bullish")
            except Exception:
                pass
            _answer_callback(cq_id, "Processing order...")

            success = _execute_approved_order(approval)
            status  = "APPROVED" if success else "FAILED"
            db.update_approval_status(approval_id, status)
            try:
                from feed import emit as _fe
                if success:
                    _fe("execution", ticker, f"Order executed: {action} {ticker} — filled", "bullish")
                else:
                    _fe("execution", ticker, f"Order FAILED: {action} {ticker} — check logs", "alert")
            except Exception:
                pass

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
            try:
                from feed import emit as _fe
                _fe("telegram", ticker, f"Rejected by @{user} — {action} {ticker} skipped", "bearish")
            except Exception:
                pass

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

# ---------------------------------------------------------------------------
# !urgent trade execution (chat-triggered, no approval queue)
# ---------------------------------------------------------------------------

def _execute_chat_order(
    ticker: str,
    side: str,
    qty: float | None,
    chat_id: int,
    reply_to: int | None,
) -> None:
    """
    Execute a trade from a !urgent chat command via the shared execution
    module (execution.execute_manual) — same phantom-fill-safety,
    owned-qty-clamping, and stop-lifecycle handling as every other trade
    path. If qty is None, sizes via RiskManager at conviction 7.
    """
    try:
        import ibkr
        from risk_manager import RiskManager
        from agents.market_regime import get_cached_regime
        from execution import execute_manual

        rm = RiskManager(ibkr.get_client())

        if side == "BUY" and rm.is_halted():
            _reply(chat_id, "🛑 <b>Blocked</b> — portfolio drawdown halt is active. No new buys.", reply_to)
            return

        bucket = "Accelerator"
        if qty is None:
            price = ibkr.get_latest_price(ticker)
            if not price or price <= 0:
                _reply(chat_id, f"⚠️ Could not get a valid price for <b>{ticker}</b> — order aborted.", reply_to)
                return
            try:
                from knowledge_loader import get_watchlist_entry
                wl = get_watchlist_entry(ticker) or {}
                bucket = wl.get("role", "Accelerator")
            except Exception:
                bucket = "Accelerator"
            regime = (get_cached_regime() or {}).get("regime", "Neutral")
            sizing = rm.size_position(ticker, conviction=7, bucket=bucket,
                                      regime=regime, price=price)
            qty = sizing.get("shares_initial", 0)
            if qty <= 0:
                flags = sizing.get("flags", [])
                _reply(chat_id, f"⚠️ Risk manager returned 0 shares for <b>{ticker}</b>: {'; '.join(flags)}", reply_to)
                return

        log.info(f"[CHAT !urgent] {side} {qty} {ticker}")
        result = execute_manual(ticker, side, qty, reason="!urgent chat command",
                                bucket=bucket, conviction=7)

        if not result["success"]:
            _reply(chat_id, f"⚠️ <b>{ticker}</b>: {result['message']}", reply_to)
            return

        icon = "🟢" if side == "BUY" else "🔴"
        _reply(chat_id, f"{icon} <b>{result['message']}</b> executed", reply_to)

    except Exception as e:
        log.error(f"[CHAT !urgent] Order failed for {ticker}: {e}")
        _reply(chat_id, f"⚠️ Order failed for <b>{ticker}</b>: {e}", reply_to)


# ---------------------------------------------------------------------------
# Mention handler (group chat @mentions)
# ---------------------------------------------------------------------------

def _handle_mention(update: dict, is_urgent: bool) -> None:
    msg      = update.get("message", {})
    chat_id  = msg.get("chat", {}).get("id")
    msg_id   = msg.get("message_id")
    from_    = msg.get("from", {})
    user_id  = from_.get("id")
    display  = from_.get("first_name") or from_.get("username") or str(user_id)
    text     = msg.get("text", "")

    if not chat_id or not text:
        return

    # Allowlist check — uses numeric user IDs (find yours via @userinfobot)
    if _ALLOWED_USER_IDS and user_id not in _ALLOWED_USER_IDS:
        log.warning(f"[chat] Blocked @mention from user_id={user_id} ({display}) — not in TELEGRAM_ALLOWED_USER_IDS")
        return

    # Strip @BotUsername and !urgent from the text
    clean = text
    if _BOT_USERNAME:
        clean = re.sub(rf"@{re.escape(_BOT_USERNAME)}", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"!urgent", "", clean, flags=re.IGNORECASE).strip()

    if not clean:
        return

    # Halt/resume commands — persisted circuit breaker control
    cmd = clean.lower().strip()
    if cmd in ("/halt", "halt trading"):
        try:
            import db
            from config import TRADING_ENV
            db.set_state(f"trading_halted_{TRADING_ENV}", "true")
            db.set_state(f"halt_reason_{TRADING_ENV}", f"Manual halt by {display}")
            _reply(chat_id, f"🛑 <b>Trading HALTED</b> ({TRADING_ENV}). No new BUY orders will "
                            "execute. Send /resume to lift the halt.", msg_id)
        except Exception as e:
            _reply(chat_id, f"⚠️ Could not set halt flag: {e}", msg_id)
        return
    if cmd in ("/resume", "resume trading"):
        try:
            import db
            from config import TRADING_ENV
            db.set_state(f"trading_halted_{TRADING_ENV}", "false")
            db.set_state(f"halt_reason_{TRADING_ENV}", "")
            _reply(chat_id, f"🟢 <b>Trading RESUMED</b> ({TRADING_ENV}). Halt flag cleared.", msg_id)
        except Exception as e:
            _reply(chat_id, f"⚠️ Could not clear halt flag: {e}", msg_id)
        return

    _send_chat_action(chat_id)

    if is_urgent:
        try:
            from chat_assistant import parse_trade_intent
            intent = parse_trade_intent(clean)
        except Exception as e:
            # Was a bare `except: intent = None` -- a real parse/import
            # failure here silently fell through to the general chat
            # assistant with zero trace anywhere. Found 2026-08-17: a
            # correctly-formatted "!urgent buy 2 DRAM" from the group chat
            # produced no error, no trade, and no clue why -- only a
            # get_portfolio tool call from the general-assistant fallback,
            # discovered by re-testing parse_trade_intent standalone (which
            # worked fine) and having no log line to explain the mismatch.
            log.error(f"[chat] parse_trade_intent failed for {clean!r}: {e}")
            intent = None

        if intent:
            _execute_chat_order(
                ticker=intent["ticker"],
                side=intent["side"],
                qty=intent.get("qty"),
                chat_id=chat_id,
                reply_to=msg_id,
            )
            return

    # Regular (or urgent non-trade) → Ollama
    try:
        import chat_assistant
        response = chat_assistant.respond(clean, display, user_id=user_id)
        _reply(chat_id, response, msg_id)
    except Exception as e:
        log.error(f"chat_assistant.respond failed: {e}")
        _reply(chat_id, "Sorry, I hit an error trying to answer that.", msg_id)


# ---------------------------------------------------------------------------
# Priority queue polling + worker
# ---------------------------------------------------------------------------

def _poll_loop() -> None:
    log.info("Telegram bot polling started.")
    _fetch_bot_username()
    offset = 0

    while not _STOP_EVENT.is_set():
        try:
            updates = _get_updates(offset)
        except Exception as e:
            log.error(f"_get_updates failed: {e}")
            time.sleep(_POLL_INTERVAL)
            continue

        for update in updates:
            try:
                offset = update["update_id"] + 1

                if "callback_query" in update:
                    _UPDATE_QUEUE.put((1, next(_QUEUE_SEQ), update))

                elif "message" in update:
                    msg  = update.get("message", {})
                    text = msg.get("text", "")
                    uid  = msg.get("from", {}).get("id")
                    bot_mention = f"@{_BOT_USERNAME}".lower() if _BOT_USERNAME else None
                    is_mention  = bot_mention and bot_mention in text.lower()
                    log.info(f"[chat] msg from user_id={uid} | mention={is_mention} | text={text[:80]!r}")
                    if is_mention:
                        is_urgent = "!urgent" in text.lower()
                        priority  = 0 if is_urgent else 2
                        _UPDATE_QUEUE.put((priority, next(_QUEUE_SEQ), update))
            except Exception as e:
                # Never let one malformed update kill the poller thread —
                # that silently stops the entire approval pipeline until a
                # manual restart.
                log.error(f"Failed to process update {update.get('update_id')}: {e}")

        if not updates:
            time.sleep(_POLL_INTERVAL)

    log.info("Telegram bot polling stopped.")


def _worker_loop() -> None:
    log.info("Telegram bot worker started.")
    while not _STOP_EVENT.is_set():
        try:
            priority, _seq, update = _UPDATE_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        try:
            if "callback_query" in update:
                _handle_callback(update)
            elif "message" in update:
                is_urgent = "!urgent" in (update.get("message", {}).get("text", "")).lower()
                _handle_mention(update, is_urgent)
        except Exception as e:
            log.error(f"Worker error (priority={priority}): {e}")
        finally:
            _UPDATE_QUEUE.task_done()
    log.info("Telegram bot worker stopped.")


def start_polling() -> list[threading.Thread]:
    """Start the Telegram bot polling and worker threads as daemons."""
    if not _TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN not set — approval gate disabled.")
        return []

    poller = threading.Thread(target=_poll_loop,  name="telegram-poller", daemon=True)
    worker = threading.Thread(target=_worker_loop, name="telegram-worker", daemon=True)
    poller.start()
    worker.start()
    return [poller, worker]


def stop_polling() -> None:
    """Signal the polling and worker loops to stop."""
    _STOP_EVENT.set()
