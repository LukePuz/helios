"""Tests must never publish to the live Celery broker: the running worker
would execute the task against the live database (approval #1, IONQ, was
retried and re-analysed this way)."""
import pytest

import tasks


def test_unpatched_apply_async_is_blocked_and_recorded(_no_live_celery_publish):
    with pytest.raises(RuntimeError, match="live broker"):
        tasks.retry_held_sell.apply_async(args=[1], countdown=60, queue="ops")
    assert _no_live_celery_publish == ["tasks.retry_held_sell"]
    _no_live_celery_publish.clear()   # handled here; the teardown check stays quiet


def test_swallowed_publish_is_still_recorded(_no_live_celery_publish):
    try:
        tasks.retry_deferred_entry.apply_async(args=[1])
    except Exception:
        pass   # what code under test with a broad except would do
    assert _no_live_celery_publish == ["tasks.retry_deferred_entry"]
    _no_live_celery_publish.clear()


def test_the_test_session_broker_is_a_dead_port():
    import tasks
    assert "127.0.0.1:1/" in tasks.REDIS_URL
    assert "127.0.0.1:1/" in str(tasks.app.conf.broker_url)


def test_the_test_session_has_no_telegram_credentials():
    import alerts
    assert alerts._TOKEN == "" and alerts._CHAT_ID == ""


def test_a_real_telegram_request_fails_the_test():
    import pytest
    import requests
    with pytest.raises(RuntimeError, match="real Telegram API"):
        requests.post("https://api.telegram.org/botX/sendMessage", json={}, timeout=1)
    import conftest  # this test triggers the guard on purpose; clear its record
    conftest._BLOCKED_HTTP.clear()


def test_a_live_ollama_request_fails_the_test():
    import pytest
    import requests
    with pytest.raises(RuntimeError, match="live Ollama"):
        requests.post("http://localhost:11434/api/generate", json={}, timeout=1)
    import conftest
    conftest._BLOCKED_HTTP.clear()


def test_the_test_session_points_ollama_at_a_dead_port():
    from agents import analyst, screener
    assert analyst.OLLAMA_HOST == "http://127.0.0.1:1"
    assert screener.OLLAMA_HOST == "http://127.0.0.1:1" and screener.OLLAMA_CPU_HOST == "http://127.0.0.1:1"

