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


def test_query_string_credentials_are_redacted():
    """Audit A1031-0253-5: FRED errors logged the full URL with api_key."""
    import log_redaction
    url = ("502 Server Error for url: https://api.stlouisfed.org/fred/series/observations"
           "?series_id=DEXUSEU&api_key=abc123DEF456&file_type=json")
    out = log_redaction.redact(url)
    assert "abc123DEF456" not in out and "api_key=[REDACTED]&file_type=json" in out
    assert log_redaction.redact("x?apikey=Q1&access_token=Z9 y") == "x?apikey=[REDACTED]&access_token=[REDACTED] y"


def test_fred_failure_logs_no_url(caplog):
    import requests
    from unittest.mock import MagicMock, patch
    from ingestion import macro
    err = requests.HTTPError("502 Server Error for url: https://x/?api_key=SECRETKEY1")
    err.response = MagicMock(status_code=502)
    with patch.object(macro, "FRED_API_KEY", "SECRETKEY1"), \
         patch("requests.get", side_effect=err), caplog.at_level("WARNING"):
        assert macro._fetch_series("DEXUSEU") == []
    assert "SECRETKEY1" not in caplog.text and "HTTP 502" in caplog.text


def test_polygon_apikey_and_bare_token_are_redacted():
    import log_redaction
    assert log_redaction.redact("https://api.polygon.io/v2/x?apiKey=P0LY9&limit=1") == \
        "https://api.polygon.io/v2/x?apiKey=[REDACTED]&limit=1"
    assert log_redaction.redact("ws://h/?token=T0K3N") == "ws://h/?token=[REDACTED]"


def test_polygon_failure_logs_no_url(caplog):
    import requests
    from unittest.mock import MagicMock, patch
    from ingestion import polygon_stream
    err = requests.HTTPError("500 Server Error for url: https://x/?apiKey=SECRETKEY2")
    err.response = MagicMock(status_code=500, text="boom")
    with patch.object(polygon_stream, "API_KEY", "SECRETKEY2"), \
         patch("requests.get", side_effect=err), caplog.at_level("WARNING"):
        assert polygon_stream._get("/v2/snapshot/NVDA") == {}
    assert "SECRETKEY2" not in caplog.text and "HTTP 500" in caplog.text


def test_heartbeat_fail_stores_the_error_redacted():
    """Professor m139: ingestion_health.last_error kept the raw request URL."""
    from unittest.mock import patch
    import tasks
    with patch("db.record_ingestion_error") as rec:
        tasks._heartbeat_fail("fred", "HTTPError for url: https://x/series?api_key=abc123&file_type=json")
    stored = rec.call_args[0][1]
    assert "abc123" not in stored and "api_key=[REDACTED]" in stored


def test_telegram_alert_text_is_redacted_before_it_is_sent():
    """Alert text embeds raw exceptions, which skip log redaction."""
    from unittest.mock import patch, MagicMock
    import alerts
    resp = MagicMock()
    resp.json.return_value = {"result": {"message_id": 1}}
    with patch.object(alerts, "_TOKEN", "t"), patch.object(alerts, "_CHAT_ID", "c"), \
         patch("alerts.requests.post", return_value=resp) as post:
        alerts._send("failed: https://x/v2?apiKey=abc123&limit=1")
        alerts._send_with_buttons("failed: https://x/v2?apiKey=abc123", [])
    for call in post.call_args_list:
        sent = call.kwargs["json"]["text"]
        assert "abc123" not in sent and "apiKey=[REDACTED]" in sent


def test_credential_match_stops_at_a_closing_tag():
    assert log_redaction.redact("<code>x?token=abc</code> ok") == "<code>x?token=[REDACTED]</code> ok"
