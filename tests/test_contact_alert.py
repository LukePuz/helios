"""A missing CONTACT_EMAIL sends one Telegram alert from startup, never raises."""
import alerts
from ingestion import contact


def _reset(monkeypatch, email):
    monkeypatch.setattr(contact, "CONTACT_EMAIL", email)
    monkeypatch.setattr(contact, "_startup_alert_sent", False)


def test_missing_email_alerts_once(monkeypatch):
    _reset(monkeypatch, "")
    sent = []
    monkeypatch.setattr(alerts, "_send", lambda text: sent.append(text))
    assert contact.alert_if_missing() is True
    assert contact.alert_if_missing() is False
    assert len(sent) == 1 and "CONTACT_EMAIL" in sent[0]


def test_set_email_sends_nothing(monkeypatch):
    _reset(monkeypatch, "someone@example.com")
    sent = []
    monkeypatch.setattr(alerts, "_send", lambda text: sent.append(text))
    assert contact.alert_if_missing() is False
    assert sent == []


def test_send_failure_does_not_raise(monkeypatch):
    _reset(monkeypatch, "")

    def boom(text):
        raise RuntimeError("telegram down")
    monkeypatch.setattr(alerts, "_send", boom)
    assert contact.alert_if_missing() is True


def test_main_calls_it_at_startup():
    import inspect
    import main
    assert "alert_if_missing()" in inspect.getsource(main.main)
