"""
Tests for the Quiver Quant 401 short-circuit (2026-09-02, found during an
ops audit): a dead API key used to make poll_once() loop all ~75 SYMBOLS
every run, each one hitting the same guaranteed 401 and logging an
identical warning. QuiverAuthError lets poll_once abort the rest of the
batch on the first 401 instead.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingestion.congressional import (
    QuiverAuthError, _get_congressional_trades, poll_once,
)


def _http_error(status_code: int, text: str = "") -> requests.exceptions.HTTPError:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    err = requests.exceptions.HTTPError(response=resp)
    return err


class TestGetCongressionalTrades:
    def test_401_raises_quiver_auth_error(self):
        with patch("ingestion.congressional.API_KEY", "dead-key"), \
             patch("ingestion.congressional.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = _http_error(401)
            with pytest.raises(QuiverAuthError):
                _get_congressional_trades("AAPL")

    def test_other_http_error_returns_empty_list_not_raise(self):
        with patch("ingestion.congressional.API_KEY", "some-key"), \
             patch("ingestion.congressional.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = _http_error(500)
            result = _get_congressional_trades("AAPL")
        assert result == []

    def test_timeout_returns_empty_list_not_raise(self):
        with patch("ingestion.congressional.API_KEY", "some-key"), \
             patch("ingestion.congressional.requests.get",
                   side_effect=requests.exceptions.Timeout("timed out")):
            result = _get_congressional_trades("AAPL")
        assert result == []

    def test_no_api_key_returns_empty_list(self):
        with patch("ingestion.congressional.API_KEY", ""):
            result = _get_congressional_trades("AAPL")
        assert result == []


class TestPollOnceShortCircuit:
    def test_401_on_first_ticker_aborts_remaining_batch(self):
        with patch("ingestion.congressional.API_KEY", "dead-key"), \
             patch("ingestion.congressional.SYMBOLS", ["AAPL", "MSFT", "NVDA"]), \
             patch("ingestion.congressional._get_congressional_trades") as mock_fetch:
            mock_fetch.side_effect = QuiverAuthError("401 Unauthorized")
            result = poll_once()
        assert result == []
        # Only the FIRST ticker's request should have been attempted --
        # this is the entire point of the short-circuit.
        assert mock_fetch.call_count == 1
        mock_fetch.assert_called_once_with("AAPL")

    def test_401_partway_through_stops_further_tickers(self):
        def side_effect(ticker):
            if ticker == "MSFT":
                raise QuiverAuthError("401 Unauthorized")
            return []
        with patch("ingestion.congressional.API_KEY", "dead-key"), \
             patch("ingestion.congressional.SYMBOLS", ["AAPL", "MSFT", "NVDA"]), \
             patch("ingestion.congressional._get_congressional_trades", side_effect=side_effect):
            with patch("ingestion.congressional._get_congressional_trades") as mock_fetch:
                mock_fetch.side_effect = side_effect
                result = poll_once()
        assert result == []
        # AAPL then MSFT attempted, NVDA never reached.
        assert mock_fetch.call_count == 2

    def test_non_auth_failures_do_not_abort_the_batch(self):
        with patch("ingestion.congressional.API_KEY", "some-key"), \
             patch("ingestion.congressional.SYMBOLS", ["AAPL", "MSFT"]), \
             patch("ingestion.congressional._get_congressional_trades", return_value=[]) as mock_fetch:
            poll_once()
        # Both tickers attempted -- a transient/empty result must not
        # trigger the auth short-circuit path.
        assert mock_fetch.call_count == 2
