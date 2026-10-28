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
    global _installed
    if _installed:
        return
    logging.setLogRecordFactory(_RedactingRecord)
    _installed = True
