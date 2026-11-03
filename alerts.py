"""
Telegram alert sender. All functions are no-ops if credentials are missing.
Includes approval-request messages with inline buttons for human approval gate.
"""

import logging
import os
import re
import time as _time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

import log_redaction  # noqa: E402
log_redaction.install()
import alert_policy  # noqa: E402

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")
_ET      = ZoneInfo("America/New_York")
_BASE    = f"https://api.telegram.org/bot{_TOKEN}"


# ---------------------------------------------------------------------------
# Core send helpers
# ---------------------------------------------------------------------------

_TELEGRAM_MAX_CHARS = 4000   # Telegram's limit is 4096; leave room for HTML


def _plain(text: str) -> str:
    """HTML tags removed and entities unescaped, for a resend without
    parse_mode."""
    import html
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def _chunks(text: str, limit: int = _TELEGRAM_MAX_CHARS) -> list[str]:
    """Split on line boundaries into pieces Telegram accepts."""
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    if len(out) > _MAX_PARTS:
        out = out[:_MAX_PARTS]
        out[-1] = out[-1][:limit - len(_TRUNCATED)] + _TRUNCATED
    return out


_MAX_PARTS = 3
_TRUNCATED = "\n…(truncated)"


def _truncate(text: str, limit: int = _TELEGRAM_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - len(_TRUNCATED)] + _TRUNCATED


def _post_message(payload: dict, _max_attempts: int, what: str) -> int | None:
    """POST sendMessage with retries. A 400 is a formatting rejection
    (unescaped '<' in model or exception text, or HTML Telegram won't
    parse): retrying the same body can't help, so it is resent once as
    plain text instead. On 2026-10-26 the weekly self-review was lost to
    three identical 400s."""
    for attempt in range(1, _max_attempts + 1):
        try:
            resp = requests.post(f"{_BASE}/sendMessage", json=payload, timeout=10)
            if getattr(resp, "status_code", None) == 400 and payload.get("parse_mode"):
                log.warning(f"Telegram {what} rejected as HTML (400); resending as plain text")
                payload = {k: v for k, v in payload.items() if k != "parse_mode"}
                payload["text"] = _plain(payload["text"])
                resp = requests.post(f"{_BASE}/sendMessage", json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json().get("result", {}).get("message_id")
        except Exception as e:
            log.warning(f"Telegram {what} attempt {attempt}/{_max_attempts} failed: {e}")
            if attempt < _max_attempts:
                _time.sleep(2 ** attempt)  # 2s, 4s
    return None


def _caller() -> str:
    """file:function of the first frame outside this module."""
    import sys
    f = sys._getframe(1)
    here = os.path.abspath(__file__)
    while f is not None and os.path.abspath(f.f_code.co_filename) == here:
        f = f.f_back
    if f is None:
        return "unknown"
    try:
        path = os.path.relpath(f.f_code.co_filename, os.path.dirname(here))
    except ValueError:
        path = f.f_code.co_filename
    return f"{path}:{f.f_code.co_name}"


def _send(text: str, _max_attempts: int = 3, severity: str | None = None) -> int | None:
    """Send a plain text message. Returns message_id or None. Retries up to 3× on failure.

    Routed by alert_policy: critical alerts (halts, stops, unprotected or
    short positions, order blocks, broker/DB failures, and any text it does
    not recognise) always go out at once. Only the known low-severity types
    are deduplicated and rate limited; their repeats arrive in the digest
    (flush_digest). Every alert is recorded in logs/telegram_outbox.jsonl."""
    if not _TOKEN or not _CHAT_ID:
        log.debug("Telegram not configured — alert skipped.")
        return None
    # Alert text often embeds a raw exception, which bypasses log redaction
    # (review of 23673b7).
    text = log_redaction.redact(str(text))
    try:
        source = _caller()
    except Exception:
        source = "unknown"
    try:
        sev, atype = alert_policy.classify(text)
        if severity in (alert_policy.CRITICAL, alert_policy.REPORT, alert_policy.LOW):
            sev = severity
    except Exception:
        sev, atype = alert_policy.CRITICAL, ""
    if sev == alert_policy.LOW and not alert_policy.admit_low(atype, text):
        log.info(f"Telegram alert held for digest ({atype}): {alert_policy.title(text)}")
        alert_policy.record(source, text, sev, atype, "held")
        return None
    message_id = None
    try:
        parts = _chunks(text)
    except Exception:
        parts = [_truncate(text)]
    for i, part in enumerate(parts):
        mid = _post_message({"chat_id": _CHAT_ID, "text": part, "parse_mode": "HTML"},
                            _max_attempts, "send")
        if i == 0:
            message_id = mid
        if mid is None:
            # Telegram is down or refusing: don't spend another ~36 s of
            # retries per remaining part (the caller may be an order path).
            break
    if message_id is None:
        log.error(f"Telegram send failed after {_max_attempts} attempts — message lost.")
    alert_policy.record(source, text, sev, atype, "sent" if message_id is not None else "failed")
    return message_id


def flush_digest() -> int | None:
    """Send one message listing the low-severity alerts held since the last
    digest. No-op when nothing was held."""
    text, snapshot = alert_policy.digest_text()
    if not text:
        return None
    message_id = _send(text, severity=alert_policy.REPORT)
    if message_id is not None:
        # Cleared only once Telegram took it; a failed digest is retried
        # with the next flush.
        alert_policy.clear_digest(snapshot)
    return message_id


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
    text = log_redaction.redact(str(text))
    message_id = _post_message(
        {
            "chat_id":      _CHAT_ID,
            "text":         _truncate(text),
            "parse_mode":   "HTML",
            "reply_markup": {"inline_keyboard": buttons},
        },
        _max_attempts, "button send")
    if message_id is None:
        log.error(f"Telegram button send failed after {_max_attempts} attempts — approval message lost.")
    try:
        alert_policy.record(_caller(), text, alert_policy.CRITICAL, "",
                            "sent" if message_id is not None else "failed")
    except Exception:
        pass
    return message_id


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
                "text":       log_redaction.redact(str(new_text)),
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

def send_trade_alert(
    symbol: str,
    side: str,
    qty: int,
    price: float,
    reason: str,
    conviction: int | None = None,
    thesis_bucket: str | None = None,
    reasoning: str | None = None,
    thesis_strength: int | None = None,
    catalyst_quality: int | None = None,
    valuation_risk_reward: int | None = None,
    data_completeness: int | None = None,
    chase_risk: int | None = None,
    price_action: int | None = None,
) -> None:
    """
    conviction/thesis_bucket/reasoning/sub-scores are optional so this still
    works for any caller without analyst context — but every analyst-driven
    auto-execution has them, and the user explicitly wants to see the full
    case (rationale + all conviction scores + bucket) on every trade, not
    just the bare fill details this used to show.
    """
    icon = "🟢" if side == "BUY" else "🔴"
    now  = datetime.now(_ET).strftime("%I:%M %p ET")
    lines = [
        f"{icon} <b>TRADE EXECUTED</b>",
        f"Symbol:  <b>{symbol}</b>",
        f"Side:    {side}  ×{qty} shares",
        f"Price:   ${price:.2f}",
    ]
    if thesis_bucket:
        lines.append(f"Bucket:  {thesis_bucket}")
    if conviction is not None:
        sub_scores = [
            ("Thesis", thesis_strength), ("Catalyst", catalyst_quality),
            ("Valuation", valuation_risk_reward), ("Data", data_completeness),
            ("Chase Risk", chase_risk), ("Price Action", price_action),
        ]
        sub_str = " · ".join(f"{label} {v}/5" for label, v in sub_scores if v is not None)
        conv_line = f"Conviction: {conviction}/10"
        if sub_str:
            conv_line += f" ({sub_str})"
        lines.append(conv_line)
    lines.append(f"Reason:  {reason}")
    if reasoning:
        lines.append(f"Rationale: {reasoning}")
    lines.append(f"Time:    {now}")
    _send("\n".join(lines))


def send_daily_summary(portfolio_value: float, daily_pl: float, start_value: float = 100_000.0) -> None:
    sign = "+" if daily_pl >= 0 else ""
    pct  = daily_pl / start_value
    date = datetime.now(_ET).strftime("%b %d, %Y")
    _send(
        f"📊 <b>Daily Summary — {date}</b>\n"
        f"Portfolio: ${portfolio_value:,.2f}\n"
        f"P&amp;L vs Start: {sign}${daily_pl:,.2f} ({sign}{pct:.2%})"
    )


def send_daily_digest(
    portfolio_value: float,
    day_pl: float,
    day_pl_pct: float,
    spy_pl_pct: float | None,
    positions_text: list[str],
    catalysts_text: list[str],
    pending_count: int,
    halted: bool,
    ingestion_issues: list[str],
    atr_fallback_tickers: list[str] | None = None,
) -> None:
    """
    Richer post-close digest — day P&L vs SPY, open positions with stop
    distances, tomorrow's catalysts, pending approvals, ingestion health,
    and halt status. send_daily_summary() (above) existed but nothing ever
    scheduled it; this replaces it on the beat schedule.
    """
    date = datetime.now(_ET).strftime("%b %d, %Y")
    sign = "+" if day_pl >= 0 else ""

    lines = [f"📊 <b>Daily Digest — {date}</b>", "━━━━━━━━━━━━━━━━━━━━━━━━━"]
    lines.append(f"Portfolio: ${portfolio_value:,.2f}")
    lines.append(f"Today's P&amp;L: {sign}${day_pl:,.2f} ({sign}{day_pl_pct:.2%})")
    if spy_pl_pct is not None:
        alpha = day_pl_pct - spy_pl_pct
        alpha_sign = "+" if alpha >= 0 else ""
        lines.append(f"SPY today: {spy_pl_pct:+.2%}  (alpha: {alpha_sign}{alpha:.2%})")

    if halted:
        lines.append("\n🛑 <b>TRADING HALTED</b> — send /resume to lift.")

    if positions_text:
        lines.append("\n<b>Open Positions:</b>")
        lines.extend(f"  {p}" for p in positions_text[:15])
    else:
        lines.append("\nNo open positions.")

    if catalysts_text:
        lines.append("\n<b>Tomorrow's Catalysts:</b>")
        lines.extend(f"  {c}" for c in catalysts_text[:10])

    lines.append(f"\nPending approvals: {pending_count}")

    if ingestion_issues:
        lines.append("\n⚠️ <b>Ingestion Health:</b>")
        lines.extend(f"  {i}" for i in ingestion_issues[:8])

    if atr_fallback_tickers:
        n = len(atr_fallback_tickers)
        lines.append(f"\nATR fallback (sizing or stop) today: {n} ticker{'s' if n != 1 else ''} "
                     f"({', '.join(atr_fallback_tickers)})")

    _send("\n".join(lines))


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
