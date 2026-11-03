"""Watchdog re-alert backoff, send status and the systemd failure notifier."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import watchdog            # noqa: E402
import notify_failure      # noqa: E402


@pytest.fixture
def state():
    store = {}
    with patch("db.get_state", side_effect=lambda k, d=None: store.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: store.__setitem__(k, v)):
        yield store


def _resp(status=200):
    r = MagicMock()
    r.status_code = status
    r.ok = status < 400
    return r


def _age(store, key, minutes):
    rec = json.loads(store[key])
    rec["at"] = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    store[key] = json.dumps(rec)


def test_backoff_intervals():
    assert [watchdog._repeat_interval_minutes(n) for n in range(6)] == [30, 60, 120, 240, 240, 240]


def test_unchanged_problem_backs_off_and_a_change_alerts_at_once(state):
    fp = watchdog._problems_fingerprint(["Celery Beat is not running"])
    assert watchdog._alert_due("k", fp)
    assert not watchdog._alert_due("k", fp)
    _age(state, "k", 31)
    assert watchdog._alert_due("k", fp)          # 1st repeat after 30 min
    _age(state, "k", 45)
    assert not watchdog._alert_due("k", fp)      # 2nd waits 60
    _age(state, "k", 61)
    assert watchdog._alert_due("k", fp)
    other = watchdog._problems_fingerprint(["Celery Beat is not running", "IBKR down"])
    assert watchdog._alert_due("k", other)       # changed set: immediate


def test_changing_numbers_are_the_same_problem():
    assert (watchdog._problems_fingerprint(["snapshot is 845 min old"])
            == watchdog._problems_fingerprint(["snapshot is 905 min old"]))


def test_legacy_iso_value_reads_as_the_same_problem(state):
    state["k"] = datetime.now(timezone.utc).isoformat()
    assert not watchdog._alert_due("k", "fp")


def test_clean_run_clears_so_a_recurrence_alerts_at_once(state):
    with patch.object(watchdog, "_send_telegram", return_value=True) as send:
        watchdog._emit(["IBKR gateway down"], "k")
        watchdog._emit(["IBKR gateway down"], "k")
        assert send.call_count == 1
        watchdog._emit([], "k")
        watchdog._emit(["IBKR gateway down"], "k")
        assert send.call_count == 2


def test_failed_send_gives_the_throttle_back(state):
    with patch.object(watchdog, "_send_telegram", return_value=False) as send:
        watchdog._emit(["IBKR gateway down"], "k")
        watchdog._emit(["IBKR gateway down"], "k")
    assert send.call_count == 2


def test_send_reports_rejection_and_resends_400_as_plain(monkeypatch):
    monkeypatch.setattr(watchdog, "_TOKEN", "t")
    monkeypatch.setattr(watchdog, "_CHAT_ID", "c")
    with patch("watchdog.requests.post", side_effect=[_resp(400), _resp(200)]) as post:
        assert watchdog._send_telegram("🚨 <b>x</b> a&amp;b") is True
    assert post.call_args_list[1].kwargs["json"] == {"chat_id": "c", "text": "🚨 x a&b"}
    with patch("watchdog.requests.post", return_value=_resp(500)):
        assert watchdog._send_telegram("x") is False
    with patch("watchdog.requests.post", side_effect=OSError("dns")):
        assert watchdog._send_telegram("x") is False


def test_say_lines_are_timestamped(capsys):
    watchdog._say("OK")
    out = capsys.readouterr().out
    assert out.startswith("[watchdog] 20") and out.rstrip().endswith(" OK")


# ── notify_failure ────────────────────────────────────────────────────────


@pytest.fixture
def nf(tmp_path, monkeypatch):
    monkeypatch.setattr(notify_failure, "STATE_PATH", tmp_path / "nf.json")
    monkeypatch.setattr(notify_failure, "_TOKEN", "t")
    monkeypatch.setattr(notify_failure, "_CHAT_ID", "c")
    monkeypatch.setattr(notify_failure.sys, "argv", ["notify_failure.py", "helios-local.service"])
    monkeypatch.setattr(notify_failure, "_recovered_since", lambda unit, since, now: False)
    return notify_failure


def test_first_unit_failure_alerts_and_repeats_are_counted(nf):
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        nf.main(now=1000)
        nf.main(now=1100)
        assert post.call_count == 1
        nf.main(now=1000 + nf.REPEAT_WINDOW_S + 1)
        assert post.call_count == 2
        assert "1 more failure(s)" in post.call_args.kwargs["json"]["text"]


def test_failure_after_two_held_is_sent_anyway(nf):
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        for i in range(6):                     # the 10-05 morning: 6 in 8 min
            nf.main(now=1000 + i * 100)
    assert post.call_count == 2                # sent, held, held, sent (+2), held, held
    assert "2 more failure(s)" in post.call_args_list[1].kwargs["json"]["text"]


def test_failure_after_a_recovery_is_a_new_incident(nf, monkeypatch):
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        nf.main(now=1000)
        monkeypatch.setattr(nf, "_recovered_since", lambda unit, since, now: True)
        nf.main(now=1100)
    assert post.call_count == 2


def test_recovered_since_needs_the_unit_up_for_a_while(monkeypatch):
    props = {}
    monkeypatch.setattr(notify_failure, "_unix_prop", lambda unit, p: props.get(p))
    props.update(ActiveEnterTimestamp=2000.0, InactiveEnterTimestamp=2010.0)
    assert not notify_failure._recovered_since("u", 1000, 2010)       # restart loop
    props.update(InactiveEnterTimestamp=2000.0 + notify_failure.RECOVERED_S)
    assert notify_failure._recovered_since("u", 1000, 9999)
    props.update(ActiveEnterTimestamp=900.0)
    assert not notify_failure._recovered_since("u", 1000, 9999)       # up before the alert
    props.clear()
    assert notify_failure._recovered_since("u", 1000, 9999)           # systemctl unavailable: send


def test_future_last_sent_is_not_a_hold(nf):
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        nf.main(now=5000)
        nf.main(now=1000)                       # clock stepped back
    assert post.call_count == 2


def test_other_units_are_not_held_by_one_units_window(nf, monkeypatch):
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        nf.main(now=1000)
        monkeypatch.setattr(nf.sys, "argv", ["notify_failure.py", "ibc-live.service"])
        nf.main(now=1001)
    assert post.call_count == 2


def test_a_rejected_send_does_not_start_the_window(nf):
    with patch("notify_failure.requests.post", return_value=_resp(500)):
        nf.main(now=1000)
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        nf.main(now=1001)
    assert post.call_count == 1


def test_html_rejection_is_resent_plain(nf):
    with patch("notify_failure.requests.post", side_effect=[_resp(400), _resp()]) as post:
        nf.main(now=1000)
    assert "parse_mode" not in post.call_args_list[1].kwargs["json"]


def test_unreadable_state_still_alerts(nf, tmp_path, monkeypatch):
    blocker = tmp_path / "f"
    blocker.write_text("")
    monkeypatch.setattr(nf, "STATE_PATH", blocker / "nf.json")
    with patch("notify_failure.requests.post", return_value=_resp()) as post:
        nf.main(now=1000)
        nf.main(now=1001)
    assert post.call_count == 2


def test_negative_elapsed_is_due(state):
    fp = "x"
    assert watchdog._alert_due("k", fp)
    rec = json.loads(state["k"])
    rec["at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    state["k"] = json.dumps(rec)
    assert watchdog._alert_due("k", fp)
