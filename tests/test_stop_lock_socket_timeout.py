"""The stop-lock Redis client has socket timeouts (audit 2026-11-08 15:08 #8)."""
from unittest.mock import patch

import stop_lock

_REAL_REDIS = stop_lock._redis      # captured before conftest's autouse stub


def test_client_gets_socket_timeouts(monkeypatch):
    monkeypatch.setattr(stop_lock, "_client", None)
    with patch("redis.Redis.from_url") as from_url:
        _REAL_REDIS()
    kw = from_url.call_args.kwargs
    assert kw["socket_timeout"] == stop_lock._REDIS_SOCKET_TIMEOUT_S == 5.0
    assert kw["socket_connect_timeout"] == 5.0
