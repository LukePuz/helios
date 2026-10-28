"""Bot tokens and IBKR account ids never reach a log line (monitor 2026-10-28)."""
import logging

import log_redaction


def test_redacts_token_and_accounts():
    text = ("url /bot1234567890:AAexampleFAKEtoken0000000000000000/sendMessage "
            "account='U12345678' paper DU1234567")
    out = log_redaction.redact(text)
    assert "AAexample" not in out and "12345678" not in out and "1234567" not in out
    assert "bot[REDACTED]" in out and out.count("[ACCOUNT]") == 2


def test_installed_factory_redacts_formatted_args(caplog):
    log_redaction.install()
    with caplog.at_level(logging.INFO):
        logging.getLogger("ib_async.wrapper").info("openOrder %s", "account='U12345678'")
    assert "U12345678" not in caplog.text and "[ACCOUNT]" in caplog.text


def test_ordinary_text_untouched():
    assert log_redaction.redact("NVDA UDR 2.5 at 05:00, order 14") == "NVDA UDR 2.5 at 05:00, order 14"
