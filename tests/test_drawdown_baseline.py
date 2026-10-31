"""A stale high-water mark must not disable the drawdown halt forever
(limitations audit 2026-10-28: live HWM was the $1M paper default)."""
from unittest.mock import MagicMock, patch

import tasks


def _run(state):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1900.0)
    sets = {}

    def set_state(k, v):
        state[k] = v
        sets[k] = v

    with patch("tasks._get_client", return_value=client), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=set_state), \
         patch("alerts._send") as send, \
         patch("config.TRADING_ENV", "live"):
        tasks.check_portfolio_drawdown.run()
    return sets, send


def test_three_implausible_reads_halt_and_never_rebase():
    """Audit A1031-0853-4: rebasing would re-arm the breaker below a real collapse."""
    state = {"high_water_mark_live": "1000000"}
    for _ in range(2):
        sets, send = _run(state)
        assert "trading_halted_live" not in sets
        send.assert_not_called()
    sets, send = _run(state)
    assert state["high_water_mark_live"] == "1000000"          # baseline untouched
    assert state["trading_halted_live"] == "true" and state["hwm_rebase_candidate_live"] == "1900.0"
    send.assert_called_once()
    assert "/rebase" in send.call_args[0][0]
    sets, send = _run(state)                                     # already halted: no repeat alert
    send.assert_not_called()


def test_a_plausible_read_resets_the_counter():
    state = {"high_water_mark_live": "2000", "drawdown_implausible_count_live": "2"}
    state["hwm_rebase_candidate_live"] = "900"
    sets, _ = _run(state)
    assert state["drawdown_implausible_count_live"] == "0"
    assert state["hwm_rebase_candidate_live"] == ""


def _rebase(state, monkeypatch):
    import telegram_bot
    monkeypatch.setattr(telegram_bot, "_ALLOWED_USER_IDS", {99})
    update = {"message": {"chat": {"id": 1}, "message_id": 2, "from": {"id": 99}, "text": "/rebase"}}
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("config.TRADING_ENV", "live"), patch.object(telegram_bot, "_reply") as reply, \
         patch.object(telegram_bot, "_send_chat_action"):
        telegram_bot._handle_mention(update, is_urgent=False)
    return reply.call_args.args[1]


def test_rebase_command_applies_the_candidate_and_resumes(monkeypatch):
    state = {"hwm_rebase_candidate_live": "1900.0", "high_water_mark_live": "1000000",
             "trading_halted_live": "true", "drawdown_implausible_count_live": "5"}
    text = _rebase(state, monkeypatch)
    assert state["high_water_mark_live"] == "1900.0" and state["trading_halted_live"] == "false"
    assert state["hwm_rebase_candidate_live"] == "" and state["drawdown_implausible_count_live"] == "0"
    assert "Baseline reset" in text


def test_rebase_without_a_candidate_changes_nothing(monkeypatch):
    state = {"high_water_mark_live": "2000", "trading_halted_live": "true"}
    text = _rebase(state, monkeypatch)
    assert state == {"high_water_mark_live": "2000", "trading_halted_live": "true"}
    assert "No baseline reset is pending" in text
