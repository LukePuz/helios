"""Stub reasons say timeout vs unreachable vs error (audit 2026-10-28 12:53, #8)."""
from unittest.mock import patch

import requests

from agents import analyst


def _kind(exc):
    with patch("agents.analyst.requests.post", side_effect=exc):
        assert analyst._call_ollama("p") is None
    return analyst._LAST_CALL_FAILURE["kind"]


def test_timeout_is_labelled_timeout():
    assert _kind(requests.exceptions.ReadTimeout("t")) == "timeout"


def test_connection_error_is_unreachable():
    assert _kind(requests.exceptions.ConnectionError("c")) == "unreachable"


def test_other_error():
    assert _kind(ValueError("v")) == "error"
