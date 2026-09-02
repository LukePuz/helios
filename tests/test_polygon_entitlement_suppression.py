"""
Tests for the Polygon entitlement-error noise suppression (2026-09-02,
found during an ops audit): the current Polygon plan doesn't include
real-time snapshot data (403 NOT_AUTHORIZED, a per-ENDPOINT plan
restriction, not per-ticker/transient) -- every get_snapshot() call across
the whole day logged an identical WARNING. First hit per endpoint template
still warns (so it's not silent); repeats downgrade to DEBUG until either
the process restarts or a real 200 clears the flag (plan upgraded).
"""
import logging
from unittest.mock import MagicMock, patch

import requests

import ingestion.polygon_stream as polygon_stream
from ingestion.polygon_stream import _get


def _entitlement_error() -> requests.exceptions.HTTPError:
    resp = MagicMock()
    resp.status_code = 403
    resp.text = '{"status":"NOT_AUTHORIZED","message":"You are not entitled to this data."}'
    return requests.exceptions.HTTPError(response=resp)


class TestEntitlementSuppression:
    def setup_method(self):
        polygon_stream._ENTITLEMENT_BLOCKED.clear()

    def test_first_403_logs_a_warning(self, caplog):
        with patch("ingestion.polygon_stream.API_KEY", "some-key"), \
             patch("ingestion.polygon_stream.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = _entitlement_error()
            with caplog.at_level(logging.WARNING, logger="ingestion.polygon_stream"):
                result = _get("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
        assert result == {}
        assert any("doesn't include" in r.message for r in caplog.records)

    def test_second_403_same_endpoint_does_not_repeat_warning(self, caplog):
        with patch("ingestion.polygon_stream.API_KEY", "some-key"), \
             patch("ingestion.polygon_stream.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = _entitlement_error()
            with caplog.at_level(logging.WARNING, logger="ingestion.polygon_stream"):
                _get("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
                caplog.clear()
                _get("/v2/snapshot/locale/us/markets/stocks/tickers/MSFT")
        # Second call (different ticker, SAME endpoint template) must not
        # log another WARNING -- this is the whole point of the fix.
        assert not any(r.levelno == logging.WARNING for r in caplog.records)

    def test_endpoint_template_shared_across_tickers(self):
        with patch("ingestion.polygon_stream.API_KEY", "some-key"), \
             patch("ingestion.polygon_stream.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = _entitlement_error()
            _get("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
        assert "/v2/snapshot/locale/us/markets/stocks/tickers" in polygon_stream._ENTITLEMENT_BLOCKED

    def test_non_entitlement_403_still_warns_every_time(self, caplog):
        resp = MagicMock()
        resp.status_code = 403
        resp.text = '{"status":"FORBIDDEN","message":"something else"}'
        err = requests.exceptions.HTTPError(response=resp)
        with patch("ingestion.polygon_stream.API_KEY", "some-key"), \
             patch("ingestion.polygon_stream.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = err
            with caplog.at_level(logging.WARNING, logger="ingestion.polygon_stream"):
                _get("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
                caplog.clear()
                _get("/v2/snapshot/locale/us/markets/stocks/tickers/MSFT")
        # A different kind of 403 (not the plan-entitlement one) is not
        # suppressed -- must not be silently swallowed as if it were.
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_successful_response_clears_the_suppression_flag(self):
        polygon_stream._ENTITLEMENT_BLOCKED.add(
            "/v2/snapshot/locale/us/markets/stocks/tickers"
        )
        with patch("ingestion.polygon_stream.API_KEY", "some-key"), \
             patch("ingestion.polygon_stream.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.return_value = None
            mock_get.return_value.json.return_value = {"ok": True}
            result = _get("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
        assert result == {"ok": True}
        assert "/v2/snapshot/locale/us/markets/stocks/tickers" not in polygon_stream._ENTITLEMENT_BLOCKED
