"""Dated halt/resume/rebase rows (professor C33)."""
from unittest.mock import MagicMock, patch

import db
import tasks


def test_log_halt_event_writes_a_ratio_only_row():
    with patch("db.log_activity") as act:
        db.log_halt_event("rebase_resume", "telegram", hwm_change_ratio=0.0019)
    stage, ticker, msg, sentiment, extra = act.call_args.args
    assert stage == "halt" and sentiment == "neutral"
    assert extra["event"] == "rebase_resume" and extra["hwm_change_ratio"] == 0.0019 and extra["at"]
    with patch("db.log_activity") as act:
        db.log_halt_event("manual_halt", "telegram")
    assert act.call_args.args[3] == "alert"
    with patch("db.log_activity", side_effect=RuntimeError("db down")):
        db.log_halt_event("resume", "telegram")              # never raises


def _command(text, state, monkeypatch):
    import telegram_bot
    monkeypatch.setattr(telegram_bot, "_ALLOWED_USER_IDS", {99})
    update = {"message": {"chat": {"id": 1}, "message_id": 2, "from": {"id": 99}, "text": text}}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("db.log_halt_event") as ev, patch("config.TRADING_ENV", "live"), \
         patch.object(telegram_bot, "_reply"), patch.object(telegram_bot, "_send_chat_action"):
        telegram_bot._handle_mention(update, is_urgent=False)
    return ev


def test_commands_log_events(monkeypatch):
    assert _command("/halt", {}, monkeypatch).call_args.args[:2] == ("manual_halt", "telegram")
    assert _command("/resume", {}, monkeypatch).call_args.args[:2] == ("resume", "telegram")
    ev = _command("/rebase", {"hwm_rebase_candidate_live": "1900.0", "high_water_mark_live": "1000000"},
                  monkeypatch)
    assert ev.call_args.args[0] == "rebase_resume" and ev.call_args.kwargs["hwm_change_ratio"] == 0.0019
    _command("/rebase", {}, monkeypatch).assert_not_called()  # nothing pending


def test_drawdown_task_logs_a_new_halt_once():
    state = {"high_water_mark_live": "1000000"}
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1900.0)
    calls = []
    for _ in range(4):
        with patch("tasks._get_client", return_value=client), \
             patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
             patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
             patch("db.log_halt_event", side_effect=lambda *a, **k: calls.append((a, k))), \
             patch("alerts._send"), patch("config.TRADING_ENV", "live"):
            tasks.check_portfolio_drawdown.run()
    assert len(calls) == 1 and calls[0][0][:2] == ("auto_halt", "drawdown_task")
    assert calls[0][1]["implausible_reads"] == 3
