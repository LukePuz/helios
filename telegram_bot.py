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
import logging
import os
import queue
import re
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
_UPDATE_QUEUE: queue.PriorityQueue = queue.PriorityQueue()

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

def _execute_approved_order(approval: dict) -> bool:
    """
    Submit the approved order to IBKR via TWS, log the trade and stop.
    Returns True on success.
    """
    ticker    = approval["ticker"]
    action    = approval["action"]
    risk_json = approval.get("risk_json") or {}

    # Pre-flight: verify IBKR is reachable before touching the market
    try:
        import ibkr as _ibkr_check
        _ibkr_check.get_ib()  # retries 3× internally
    except Exception as conn_err:
        log.error(f"[{ticker}] IBKR not connected — cannot execute order: {conn_err}")
        try:
            import alerts as _al
            _al._send(
                f"⚠️ IBKR CONNECTION FAILED\n"
                f"Could not execute approved order for <b>{ticker}</b>.\n"
                f"Error: {conn_err}\n"
                f"Manual execution required."
            )
        except Exception:
            pass
        return False

    try:
        import ibkr
        import db
        import alerts as alert_module

        shares = risk_json.get("shares_initial", 0)
        if shares <= 0:
            log.error(f"[{ticker}] No shares_initial in risk_json — cannot execute")
            return False

        # Drawdown halt — block new BUY entries if portfolio is in drawdown
        if action in ("OPEN", "ADD"):
            try:
                from risk_manager import RiskManager
                rm = RiskManager(ibkr.get_client())
                if rm.is_halted():
                    log.error(f"[{ticker}] Order blocked — portfolio drawdown halt active")
                    alert_module._send(
                        f"🛑 ORDER BLOCKED — Drawdown Halt\n"
                        f"<b>{ticker}</b> {action} order was approved but cannot execute.\n"
                        f"Portfolio drawdown limit is active. No new BUY entries allowed.\n"
                        f"Review and reset the halt flag when drawdown is resolved."
                    )
                    return False
            except Exception as halt_err:
                log.warning(f"[{ticker}] Drawdown halt check failed: {halt_err}")

        if action in ("OPEN", "ADD"):
            side = "BUY"
        elif action in ("TRIM", "EXIT"):
            side = "SELL"
            if action == "EXIT":
                qty = ibkr.get_client().get_open_position(ticker).qty
                shares = round(abs(float(qty)), 4)
                if shares <= 0:
                    log.error(f"[{ticker}] EXIT requested but no position found")
                    return False
            # Cancel the existing broker stop BEFORE selling so the fill can't
            # race the stop and leave an order that would short the account.
            try:
                cancelled = ibkr.cancel_stop_orders_for(ticker)
                if cancelled:
                    log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before {action}")
            except Exception as e:
                log.warning(f"[{ticker}] Could not cancel existing stops before {action}: {e}")
        else:
            log.info(f"[{ticker}] Action {action} does not require order submission")
            return True

        # Price freshness check — reject if market moved >8% since sizing.
        # Uses the staleness-aware quote so a last-RTH price hours after the
        # close can't masquerade as "no drift".
        price_at_sizing = risk_json.get("price_at_sizing")
        current_price_pre = None
        try:
            from market_data import get_priced_quote
            pq = get_priced_quote(ticker)
            if pq:
                current_price_pre = pq["price"]
                if pq["is_stale"]:
                    log.warning(
                        f"[{ticker}] Price is STALE (age={pq['age_minutes']}min, "
                        f"market_open={pq['market_open']}) — drift check unreliable"
                    )
                    try:
                        alert_module._send(
                            f"⚠️ STALE PRICE — {ticker}\n"
                            f"Latest price ${current_price_pre:.2f} is from a stale bar "
                            f"(market open: {pq['market_open']}). "
                            f"Order will proceed but verify the price in TWS."
                        )
                    except Exception:
                        pass
        except Exception:
            current_price_pre = ibkr.get_latest_price(ticker)
        if price_at_sizing and current_price_pre:
            drift = abs(current_price_pre - price_at_sizing) / price_at_sizing
            if drift > 0.08:
                log.error(
                    f"[{ticker}] Price drift {drift:.1%} since sizing "
                    f"(was ${price_at_sizing:.2f}, now ${current_price_pre:.2f}) — aborting order"
                )
                try:
                    alert_module._send(
                        f"⚠️ ORDER ABORTED — Price Drift\n"
                        f"<b>{ticker}</b> moved {drift:.1%} since recommendation was sized.\n"
                        f"Sized at: <b>${price_at_sizing:.2f}</b> | Now: <b>${current_price_pre:.2f}</b>\n"
                        f"Re-run analysis to get fresh sizing before approving."
                    )
                except Exception:
                    pass
                return False

        stop_price_for_entry = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")

        try:
            from config import USE_BRACKET_ORDERS
        except Exception:
            USE_BRACKET_ORDERS = False

        bracket_used = False
        if side == "BUY" and USE_BRACKET_ORDERS and stop_price_for_entry:
            # Native bracket: entry + stop linked by parentId — TWS enforces
            # the pairing atomically, so this can't produce an orphaned stop.
            try:
                bracket_result = ibkr.place_bracket_order(
                    ticker, side, shares, stop_loss_price=float(stop_price_for_entry),
                )
                bracket_used = True
                order_result = {"order_id": bracket_result["parent_order_id"], "fill_price": None, "filled_qty": None}
                log.info(f"[{ticker}] Bracket order submitted: {side} {shares} shares "
                        f"(parent={bracket_result['parent_order_id']}, stop={bracket_result['stop_order_id']})")
            except Exception as e:
                log.warning(f"[{ticker}] Bracket order failed, falling back to market+separate stop: {e}")
                bracket_used = False

        if not bracket_used:
            # Submit market order — result includes the confirmed fill price
            order_result = ibkr.place_market_order(ticker, side, shares)
            log.info(f"[{ticker}] Order submitted: {side} {shares} shares (id={order_result['order_id']})")

        # Real fill price; fall back to latest quote only if fill unconfirmed
        current_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
        filled_qty    = order_result.get("filled_qty") or shares

        # Log trade
        db.log_trade(
            symbol=ticker,
            side=side,
            quantity=filled_qty,
            price=current_price,
            reason=f"{action} approved via Telegram (conviction={approval['conviction']}/10)",
        )

        # Order-lifecycle + slippage logging (decision price = price used at sizing time)
        try:
            local_order_id = db.log_order(
                ticker=ticker, side=side,
                order_type="BRACKET" if bracket_used else "MARKET",
                qty=shares, broker_order_id=order_result.get("order_id"),
                decision_price=price_at_sizing, approval_id=approval["id"],
                status="submitted",
            )
            if local_order_id > 0 and order_result.get("fill_price"):
                db.update_order_fill(local_order_id, order_result["fill_price"],
                                     order_result.get("filled_qty") or shares, "filled")
        except Exception as e:
            log.debug(f"[{ticker}] Order/slippage logging failed (non-critical): {e}")

        # Stop lifecycle on SELL: EXIT removes the stop record entirely;
        # TRIM re-places the stop for the remaining shares.
        if side == "SELL":
            if action == "EXIT":
                db.delete_position_stop(ticker)
                log.info(f"[{ticker}] Position closed — stop record deleted")
                # Close the feedback-loop outcome record with the real fill price
                try:
                    db.close_trade_outcome(ticker, exit_price=current_price)
                except Exception as e:
                    log.warning(f"[{ticker}] close_trade_outcome failed: {e}")
            else:  # TRIM — re-place stop at same price for remaining quantity
                try:
                    remaining = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
                except Exception:
                    remaining = 0
                stop_rec = db.get_position_stop(ticker)
                if remaining > 0 and stop_rec:
                    try:
                        new_stop = ibkr.place_stop_order(ticker, remaining, stop_rec["stop_price"])
                        db.set_stop_broker_order_id(ticker, new_stop["order_id"])
                        log.info(f"[{ticker}] Stop re-placed for remaining {remaining} shares")
                    except Exception as e:
                        log.error(f"[{ticker}] Failed to re-place stop after TRIM: {e}")
                        alert_module._send(
                            f"🚨 UNPROTECTED POSITION — {ticker}\n"
                            f"TRIM filled but stop re-placement failed. "
                            f"Place stop @ ${stop_rec['stop_price']:.2f} for {remaining} shares manually."
                        )
                elif remaining <= 0:
                    db.delete_position_stop(ticker)

        # Log stop level and place stop order with IBKR.
        # Skipped when a bracket order already placed the stop atomically —
        # re-placing it here would create a second, redundant stop.
        stop_price = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")
        if stop_price and side == "BUY" and bracket_used:
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=current_price,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
            )
            db.set_stop_broker_order_id(ticker, bracket_result["stop_order_id"])
            log.info(f"[{ticker}] Bracket stop recorded (orderId={bracket_result['stop_order_id']})")
        elif stop_price and side == "BUY":
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=current_price,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
            )
            stop_placed = False
            for stop_attempt in range(1, 4):
                try:
                    stop_result = ibkr.place_stop_order(ticker, filled_qty, float(stop_price))
                    db.set_stop_broker_order_id(ticker, stop_result["order_id"])
                    stop_placed = True
                    break
                except Exception as e:
                    log.warning(f"[{ticker}] Stop order attempt {stop_attempt}/3 failed: {e}")
                    if stop_attempt < 3:
                        import time as _time
                        _time.sleep(2 * stop_attempt)
            if not stop_placed:
                log.error(f"[{ticker}] CRITICAL — stop order failed after 3 attempts. Position is UNPROTECTED.")
                try:
                    alert_module._send(
                        f"🚨 UNPROTECTED POSITION — ACTION REQUIRED\n"
                        f"<b>{ticker}</b>: entry order was filled but stop order failed after 3 attempts.\n"
                        f"Stop target: <b>${float(stop_price):.2f}</b> for {filled_qty} shares.\n"
                        f"Place stop manually in TWS immediately."
                    )
                except Exception:
                    pass

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
            side=side,
            qty=filled_qty,
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
    from_   = cq.get("from", {})
    user    = from_.get("username") or from_.get("first_name") or str(from_.get("id", "unknown"))

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
    Execute a trade from a !urgent chat command.
    If qty is None, sizes via RiskManager at conviction 7.
    Places ATR-based stop for BUY orders. Logs to DB.
    """
    try:
        import ibkr
        import db
        import alerts as alert_module
        from risk_manager import RiskManager
        from agents.market_regime import get_cached_regime

        rm = RiskManager(ibkr.get_client())

        if side == "BUY" and rm.is_halted():
            _reply(chat_id, f"🛑 <b>Blocked</b> — portfolio drawdown halt is active. No new buys.", reply_to)
            return

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

        if side == "SELL":
            # Cancel existing stops first so the sell can't leave an orphaned
            # stop order that would short the account later.
            try:
                cancelled = ibkr.cancel_stop_orders_for(ticker)
                if cancelled:
                    log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before !urgent SELL")
            except Exception as e:
                log.warning(f"[{ticker}] Could not cancel stops before !urgent SELL: {e}")

        order_result = ibkr.place_market_order(ticker, side, qty)
        log.info(f"[CHAT !urgent] {side} {qty} {ticker} (id={order_result['order_id']})")

        fill_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
        db.log_trade(
            symbol=ticker, side=side, quantity=order_result.get("filled_qty") or qty,
            price=fill_price,
            reason="!urgent chat command",
        )

        if side == "SELL":
            # Keep DB stop records in sync with what remains at the broker
            try:
                remaining = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
            except Exception:
                remaining = 0
            if remaining <= 0:
                db.delete_position_stop(ticker)
                try:
                    db.close_trade_outcome(ticker, exit_price=fill_price)
                except Exception as e:
                    log.debug(f"[{ticker}] close_trade_outcome (no open outcome row is expected "
                             f"for positions not opened via approval): {e}")
            else:
                stop_rec = db.get_position_stop(ticker)
                if stop_rec:
                    try:
                        new_stop = ibkr.place_stop_order(ticker, remaining, stop_rec["stop_price"])
                        db.set_stop_broker_order_id(ticker, new_stop["order_id"])
                    except Exception as e:
                        log.error(f"[{ticker}] Failed to re-place stop after !urgent trim: {e}")

        # ATR-based stop for buys
        stop_info = ""
        if side == "BUY":
            stop_price, _ = rm.calculate_stop(ticker, fill_price, bucket if qty is None else "Accelerator")
            if stop_price:
                db.upsert_position_stop(
                    ticker=ticker, entry_price=fill_price,
                    stop_price=float(stop_price), stop_method="!urgent ATR stop",
                    bucket="Accelerator", conviction=7, trail_distance_pct=None,
                )
                stop_placed = False
                for attempt in range(1, 4):
                    try:
                        stop_result = ibkr.place_stop_order(ticker, qty, float(stop_price))
                        db.set_stop_broker_order_id(ticker, stop_result["order_id"])
                        stop_placed = True
                        break
                    except Exception as e:
                        log.warning(f"[{ticker}] Stop attempt {attempt}/3: {e}")
                        if attempt < 3:
                            time.sleep(2 * attempt)
                if stop_placed:
                    stop_info = f"\nStop placed @ <b>${stop_price:.2f}</b>"
                else:
                    stop_info = "\n⚠️ <b>Stop order FAILED — place manually in TWS!</b>"
                    try:
                        alert_module._send(
                            f"🚨 UNPROTECTED POSITION — {ticker}\n"
                            f"!urgent order filled but stop failed. Place stop @ ${stop_price:.2f} manually."
                        )
                    except Exception:
                        pass

        icon = "🟢" if side == "BUY" else "🔴"
        _reply(
            chat_id,
            f"{icon} <b>{side} {qty} {ticker}</b> executed\n"
            f"Fill price: ~${fill_price:.2f}{stop_info}",
            reply_to,
        )

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
            db.set_state("trading_halted", "true")
            db.set_state("halt_reason", f"Manual halt by {display}")
            _reply(chat_id, "🛑 <b>Trading HALTED.</b> No new BUY orders will execute. "
                            "Send /resume to lift the halt.", msg_id)
        except Exception as e:
            _reply(chat_id, f"⚠️ Could not set halt flag: {e}", msg_id)
        return
    if cmd in ("/resume", "resume trading"):
        try:
            import db
            db.set_state("trading_halted", "false")
            db.set_state("halt_reason", "")
            _reply(chat_id, "🟢 <b>Trading RESUMED.</b> Halt flag cleared.", msg_id)
        except Exception as e:
            _reply(chat_id, f"⚠️ Could not clear halt flag: {e}", msg_id)
        return

    _send_chat_action(chat_id)

    if is_urgent:
        try:
            from chat_assistant import parse_trade_intent
            intent = parse_trade_intent(clean)
        except Exception:
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
        response = chat_assistant.respond(clean, display)
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
        updates = _get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1

            if "callback_query" in update:
                _UPDATE_QUEUE.put((1, update))

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
                    _UPDATE_QUEUE.put((priority, update))

        if not updates:
            time.sleep(_POLL_INTERVAL)

    log.info("Telegram bot polling stopped.")


def _worker_loop() -> None:
    log.info("Telegram bot worker started.")
    while not _STOP_EVENT.is_set():
        try:
            priority, update = _UPDATE_QUEUE.get(timeout=1)
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
