"""The watchdog logs one line when the analyst model sits partly on the CPU
(audit 2026-11-07 09:07 #8). Log-only: no Telegram, no heal."""
from unittest.mock import MagicMock, patch

import requests

from deploy import watchdog


def _ps(models):
    r = MagicMock()
    r.json.return_value = {"models": models}
    return r


def _note(models, monkeypatch, effect=None):
    monkeypatch.setenv("ANALYST_MODEL", "qwen2.5:72b-instruct-q3_K_M")
    with patch("deploy.watchdog.requests.get", return_value=_ps(models), side_effect=effect) as get:
        out = watchdog._analyst_gpu_split_note()
    return out, get


def test_low_share_gives_a_line(monkeypatch):
    out, get = _note([{"name": "qwen2.5:72b-instruct-q3_K_M", "size": 100, "size_vram": 79,
                       "context_length": 32768}], monkeypatch)
    assert get.call_args.args[0].endswith("/api/ps")
    assert "79% of its bytes in VRAM" in out and "context 32768" in out


def test_healthy_share_unloaded_or_unreadable_give_none(monkeypatch):
    m = {"name": "qwen2.5:72b-instruct-q3_K_M", "size": 100, "size_vram": 97}
    assert _note([m], monkeypatch)[0] is None
    assert _note([{"name": "qwen2.5:7b", "size": 100, "size_vram": 0}], monkeypatch)[0] is None
    assert _note([{**m, "size": 0}], monkeypatch)[0] is None
    assert _note([], monkeypatch, effect=requests.exceptions.ConnectionError())[0] is None


def test_main_logs_the_line_without_alerting(monkeypatch):
    monkeypatch.setattr(watchdog.sys, "argv", ["watchdog.py"])
    with patch.object(watchdog, "_service_state", return_value="active"), \
         patch.object(watchdog, "_planned_restart_in_progress", return_value=False), \
         patch.object(watchdog, "_check_processes", return_value=[]), \
         patch.object(watchdog, "_check_snapshot_freshness", return_value=[]), \
         patch.object(watchdog, "_check_orders_worker_responds", return_value=[]), \
         patch.object(watchdog, "_check_orders_task_not_stuck", return_value=[]), \
         patch.object(watchdog, "_analyst_gpu_split_note", return_value="low split"), \
         patch.object(watchdog, "_emit") as emit, \
         patch.object(watchdog, "_send_telegram") as tg, \
         patch.object(watchdog, "_say") as say:
        watchdog.main()
    emit.assert_called_once_with([], "watchdog_alert_at")
    tg.assert_not_called()
    say.assert_any_call("low split")


def test_a_tagless_model_name_matches_its_latest_form(monkeypatch):
    monkeypatch.setenv("ANALYST_MODEL", "helios-analyst")
    m = {"name": "helios-analyst:latest", "size": 100, "size_vram": 50}
    with patch("deploy.watchdog.requests.get", return_value=_ps([m])):
        assert "50% of its bytes" in watchdog._analyst_gpu_split_note()
