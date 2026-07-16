"""
Tests for telegram_bot.py's queue safety and callback authorization —
no live Telegram/DB/IBKR required.
"""
import queue
from unittest.mock import MagicMock, patch

import pytest

import telegram_bot


class TestPriorityQueueTiebreak:
    """Two equal-priority items used to raise TypeError (heapq falling
    through to comparing the update dicts), silently killing the poller
    thread and stopping the entire approval pipeline until a manual
    restart. The (priority, seq, update) tuple form fixes this."""

    def test_equal_priority_updates_do_not_raise(self):
        q: queue.PriorityQueue = queue.PriorityQueue()
        seq = telegram_bot._QUEUE_SEQ

        update_a = {"callback_query": {"id": "a"}}
        update_b = {"callback_query": {"id": "b"}}

        # Same priority (1) for both — this is exactly what killed the
        # poller: heapq compares the second tuple element on a tie, and two
        # plain dicts don't support '<'.
        q.put((1, next(seq), update_a))
        q.put((1, next(seq), update_b))

        priority1, seq1, first = q.get()
        priority2, seq2, second = q.get()

        assert priority1 == priority2 == 1
        assert seq1 < seq2  # FIFO within the same priority
        assert first == update_a
        assert second == update_b

    def test_seq_breaks_ties_across_priorities_too(self):
        q: queue.PriorityQueue = queue.PriorityQueue()
        seq = telegram_bot._QUEUE_SEQ
        q.put((2, next(seq), {"message": "chat1"}))
        q.put((0, next(seq), {"message": "urgent"}))
        q.put((1, next(seq), {"callback_query": "approve"}))

        order = [q.get()[2] for _ in range(3)]
        assert order == [{"message": "urgent"}, {"callback_query": "approve"},
                         {"message": "chat1"}]


class TestCallbackAuthorization:
    """_handle_callback executes real orders on Approve — it must reject
    unauthorized users and fail closed when no allowlist is configured,
    matching the mention handler's existing _ALLOWED_USER_IDS gate."""

    def _make_callback_update(self, user_id: int, chat_id: int = -5066821070):
        return {
            "callback_query": {
                "id": "cbq1",
                "data": "approve:42",
                "from": {"id": user_id, "username": "someone"},
                "message": {"message_id": 99, "chat": {"id": chat_id}},
            }
        }

    def test_unauthorized_user_is_rejected(self):
        update = self._make_callback_update(user_id=999999)
        with patch.object(telegram_bot, "_ALLOWED_USER_IDS", {12345}), \
             patch.object(telegram_bot, "_CHAT_ID", "-5066821070"), \
             patch("telegram_bot._answer_callback") as mock_answer, \
             patch("db.claim_approval_for_execution") as mock_claim:
            telegram_bot._handle_callback(update)

        mock_claim.assert_not_called()
        mock_answer.assert_called_once()
        assert "not authorized" in mock_answer.call_args[0][1].lower()

    def test_empty_allowlist_fails_closed(self):
        """No allowlist configured must reject, not fail open — these
        buttons execute real orders, unlike the informational mention path."""
        update = self._make_callback_update(user_id=12345)
        with patch.object(telegram_bot, "_ALLOWED_USER_IDS", set()), \
             patch("telegram_bot._answer_callback") as mock_answer, \
             patch("db.claim_approval_for_execution") as mock_claim:
            telegram_bot._handle_callback(update)

        mock_claim.assert_not_called()
        mock_answer.assert_called_once()

    def test_authorized_user_proceeds_to_lookup(self):
        """A valid user should pass the auth gate and reach the approval
        lookup (proven by get_approval_by_id being called) — we stop the
        test there rather than executing a full mocked order."""
        update = self._make_callback_update(user_id=12345)
        with patch.object(telegram_bot, "_ALLOWED_USER_IDS", {12345}), \
             patch.object(telegram_bot, "_CHAT_ID", "-5066821070"), \
             patch("db.get_approval_by_id", return_value=None) as mock_get, \
             patch("telegram_bot._answer_callback"):
            telegram_bot._handle_callback(update)

        mock_get.assert_called_once_with(42)
