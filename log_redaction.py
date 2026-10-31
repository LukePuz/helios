"""Redacts secrets from every log message in the process.

Found 2026-10-28 (monitor): failed Telegram sends logged the request URL,
which embeds the bot token, and ib_async's INFO order dumps include the IBKR
account id. Researchers' scripts parse these logs into ~/helios-research.
install() wraps the LogRecord factory, so it covers every logger and handler,
including third-party ones. Idempotent.
"""
import logging
import re

_PATTERNS = [
    (re.compile(r"bot\d{6,}:[A-Za-z0-9_-]{20,}"), "bot[REDACTED]"),
    (re.compile(r"\b[UD]U?\d{7,8}\b"), "[ACCOUNT]"),
    # Credentials in a query string: FRED's api_key rode along in every
    # requests error message (audit A1031-0253-5).
    # Stops at "<" too, so a closing HTML tag in a Telegram alert survives.
    (re.compile(r"(?i)\b(api_?key|access_token|token)=[^&\s'\"<]+"), r"\1=[REDACTED]"),
]


def redact(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


class _RedactingRecord(logging.LogRecord):
    def getMessage(self) -> str:
        return redact(super().getMessage())


_installed = False


def install() -> None:
    """Also redacts formatted tracebacks and stack info, which the Formatter
    builds outside getMessage() (professor M34: exc_info tracebacks from
    requests errors carry the bot URL)."""
    global _installed
    if _installed:
        return
    logging.setLogRecordFactory(_RedactingRecord)
    _format_exception = logging.Formatter.formatException
    _format_stack = logging.Formatter.formatStack
    logging.Formatter.formatException = lambda self, ei: redact(_format_exception(self, ei))
    logging.Formatter.formatStack = lambda self, si: redact(_format_stack(self, si))
    _installed = True
