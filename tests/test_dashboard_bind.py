"""The dashboard binds to loopback unless HELIOS_DASHBOARD_ADDRESS says
otherwise; Streamlit's default was every interface (audit 2026-11-07 03:08 #4)."""
import importlib
from unittest.mock import patch

import config


def test_default_address_is_loopback(monkeypatch):
    monkeypatch.delenv("HELIOS_DASHBOARD_ADDRESS", raising=False)
    try:
        assert importlib.reload(config).DASHBOARD_ADDRESS == "127.0.0.1"
        monkeypatch.setenv("HELIOS_DASHBOARD_ADDRESS", "  ")
        assert importlib.reload(config).DASHBOARD_ADDRESS == "127.0.0.1"
        monkeypatch.setenv("HELIOS_DASHBOARD_ADDRESS", "100.64.0.1")
        assert importlib.reload(config).DASHBOARD_ADDRESS == "100.64.0.1"
    finally:
        monkeypatch.delenv("HELIOS_DASHBOARD_ADDRESS", raising=False)
        importlib.reload(config)


def test_streamlit_is_started_with_the_bind_address():
    import main
    with patch("main.subprocess.Popen") as popen, patch("config.DASHBOARD_ADDRESS", "127.0.0.1"):
        main._start_dashboard()
    argv = popen.call_args.args[0]
    i = argv.index("--server.address")
    assert argv[i + 1] == "127.0.0.1" and "--server.port" in argv
