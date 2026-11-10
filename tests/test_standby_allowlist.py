"""The standby bot fails closed when no allowlist is set, like telegram_bot's
mention and callback paths (audit 11-06 19:37 #7c): any DM used to wake the
full stack."""
from unittest.mock import patch

import standby_bot


def _dm(user_id):
    return {"message": {"chat": {"id": 1, "type": "private"}, "message_id": 2,
                        "from": {"id": user_id, "first_name": "x"}, "text": "status?"}}


def test_empty_allowlist_wakes_nothing():
    with patch.object(standby_bot.tb, "_ALLOWED_USER_IDS", set()), \
         patch.object(standby_bot, "_wake_system") as wake, \
         patch.object(standby_bot.tb, "_reply") as reply:
        standby_bot._handle_message(_dm(42))
    wake.assert_not_called()
    reply.assert_not_called()


def test_unlisted_user_is_blocked_and_listed_user_wakes_it():
    with patch.object(standby_bot.tb, "_ALLOWED_USER_IDS", {7}), \
         patch.object(standby_bot, "_wake_system") as wake, \
         patch.object(standby_bot, "_system_awake", return_value=False), \
         patch.object(standby_bot.tb, "_reply"), patch.object(standby_bot.tb, "_send_chat_action"), \
         patch("chat_assistant.respond", return_value="ok"):
        standby_bot._handle_message(_dm(42))
        wake.assert_not_called()
        standby_bot._handle_message(_dm(7))
        wake.assert_called_once()
