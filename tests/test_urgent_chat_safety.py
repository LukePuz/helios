"""!urgent chat orders: strict grammar, SELL needs a quantity, allowlist fails
closed (limitations audit 2026-10-28 16:53, #5)."""
from unittest.mock import patch

import pytest

from chat_assistant import parse_trade_intent


@pytest.mark.parametrize("text,expected", [
    ("!urgent buy 5 NVDA", {"side": "BUY", "qty": 5.0, "ticker": "NVDA"}),
    ("sell TSLA 10 shares", {"side": "SELL", "qty": 10.0, "ticker": "TSLA"}),
    ("sell 2.5 shares of AMD", {"side": "SELL", "qty": 2.5, "ticker": "AMD"}),
    ("buy AAPL.", {"side": "BUY", "qty": None, "ticker": "AAPL"}),
])
def test_commands_still_parse(text, expected):
    assert parse_trade_intent(text) == expected


@pytest.mark.parametrize("text", [
    "!urgent dont sell PLTR yet", "why did we sell NVDA", "should I sell AMD?",
    "sell NVDA if it drops", "I want to buy 3 MSFT",
])
def test_sentences_are_not_orders(text):
    assert parse_trade_intent(text) is None


def test_sell_without_quantity_is_refused():
    import telegram_bot
    with patch("ibkr.get_client"), patch("risk_manager.RiskManager"), \
         patch("execution.execute_manual") as execute, \
         patch.object(telegram_bot, "_reply") as reply:
        telegram_bot._execute_chat_order("PLTR", "SELL", None, chat_id=1, reply_to=None)
    execute.assert_not_called()
    assert "Nothing was sold" in reply.call_args.args[1]


def test_empty_allowlist_blocks_mentions(monkeypatch):
    import telegram_bot
    monkeypatch.setattr(telegram_bot, "_ALLOWED_USER_IDS", set())
    update = {"message": {"chat": {"id": 1}, "message_id": 2, "from": {"id": 99},
                          "text": "!urgent buy 1 NVDA"}}
    with patch.object(telegram_bot, "_execute_chat_order") as order, \
         patch.object(telegram_bot, "_reply"), patch.object(telegram_bot, "_send_chat_action"):
        telegram_bot._handle_mention(update, is_urgent=True)
    order.assert_not_called()
