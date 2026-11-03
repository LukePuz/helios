"""Severity routing, dedup, digest and outbox for Telegram alerts."""
import json
from unittest.mock import MagicMock, patch

import pytest

import alert_policy as ap

# Real alert texts from the call sites, one per critical family.
CRITICAL_TEXTS = [
    "🚨 <b>TRADING HALTED — DRAWDOWN LIMIT</b>\nDrawdown: -12.0%",
    "🛑 <b>STOP TRIGGERED — AAPL</b>\nCurrent: $1.00",
    "🚨 UNPROTECTED POSITION — AAPL\nA sell failed after its stop was cancelled",
    "⚠️ STOP STATUS UNKNOWN — AAPL\nA stop for 3 shares got no reply",
    "⚠️ STOP REFUSED — AAPL\nerr",
    "🚨 SHORT POSITION — <b>AAPL</b>\n3 shares at the broker.",
    "🛑 ORDER BLOCKED — Drawdown Halt\n<b>AAPL</b> BUY",
    "⚠️ IBKR CONNECTION FAILED\nCould not execute approved order",
    "⚠️ DB WRITE FAILED — trades\nconnection refused",
    "⚠️ MONITOR BLIND — check_trailing_stops failed 3 checks in a row.",
    "STOP RECONCILE — 🔧 AAPL: placed stop",
    "🔁 STILL OPEN — <b>AAPL</b>\nAsked 60 min ago",
    "⚠️ FILL UNCONFIRMED — <b>AAPL</b>\nBUY 3",
    "🟢 <b>TRADE EXECUTED</b>\nSymbol: <b>AAPL</b>",
    "🚨 <b>Helios watchdog alert</b>\n• Celery Beat is not running",
    "🚨 <b>systemd unit failed: helios-local.service</b>",
    "⚠️ <b>Ollama auto-recovery FAILED</b>\nprobe, and the automatic restart failed",
    "something nobody has classified yet",
    "⏰ 2 recommendation(s) expired without a decision.",
    "⏸️ WAITING FOR FRESH PRICE — AAPL\nOnly a stale price",
    "⏸️ WAITING FOR CONFIRMATION — AAPL\nGapped +4.0%",
    # Throttled per stale-source set at its call site, so not batched here.
    "⚠️ <b>Ingestion Health Check</b>\n• news: last success 45min ago",
]

LOW_TEXTS = {
    "analyst_stub": "⚠️ <b>Analyst falling back to stub</b> — AAPL: timeout",
    "ollama_recovered": "🔧 <b>Ollama auto-recovery</b>\nunresponsive — restarted ollama.service",
    "vector_store_recovered": "🔧 <b>Vector store auto-recovery</b>\nCorrupted collection(s) rebuilt: news",
    "monitor_recovered": "✅ check_trailing_stops monitor recovered after 4 failed checks.",
    "catalyst_conflict": "<b>CATALYST CONFLICT - AAPL</b>\nType: earnings",
}


@pytest.mark.parametrize("text", CRITICAL_TEXTS)
def test_real_risk_and_unknown_alerts_are_critical(text):
    assert ap.classify(text)[0] == ap.CRITICAL


@pytest.mark.parametrize("name,text", sorted(LOW_TEXTS.items()))
def test_known_informational_types_are_low(name, text):
    assert ap.classify(text) == (ap.LOW, name)


def test_a_critical_word_in_the_first_line_overrides_a_low_type():
    assert ap.classify("⚠️ <b>Analyst falling back to stub</b> — IBKR down\n• x")[0] == ap.CRITICAL


def test_reports_are_their_own_severity():
    assert ap.classify("📊 <b>Daily Digest — Nov 02, 2026</b>\n🛑 <b>TRADING HALTED</b>")[0] == ap.REPORT
    assert ap.classify("🗓️ <b>Weekly Self-Review</b>\n\nx")[0] == ap.REPORT


def test_title_masks_money_long_numbers_and_tokens():
    t = ap.title("<b>Portfolio $123,456.78</b> acct 12345678 bot123456789:" + "A" * 35 + "\nbody $9")
    assert "123" not in t and "$[amt]" in t and "[n]" in t and "body" not in t
    assert "<b>" not in t


def test_fingerprint_ignores_changing_numbers():
    assert ap.fingerprint("snapshot is 845 min old") == ap.fingerprint("snapshot is 905 min old")


def test_first_low_alert_is_sent_and_repeats_are_held(tmp_path):
    s = tmp_path / "state.json"
    assert ap.admit_low("ingestion_health", "a", now=1000, state_path=s)
    assert not ap.admit_low("ingestion_health", "a", now=1100, state_path=s)
    assert not ap.admit_low("ingestion_health", "a", now=1200, state_path=s)
    assert ap.admit_low("ingestion_health", "a", now=1000 + ap.DEDUP_WINDOW_S, state_path=s)


def test_low_alerts_are_rate_limited_per_hour(tmp_path):
    s = tmp_path / "state.json"
    results = [ap.admit_low(f"t{i}", "x", now=1000 + i, state_path=s)
               for i in range(ap.LOW_RATE_PER_HOUR + 2)]
    assert results.count(True) == ap.LOW_RATE_PER_HOUR
    assert ap.admit_low("later", "x", now=1000 + 3601, state_path=s)


def test_digest_lists_held_alerts_once_and_clears(tmp_path):
    s = tmp_path / "state.json"
    ap.admit_low("analyst_stub", "⚠️ <b>Analyst falling back to stub</b> — AAPL", now=1000, state_path=s)
    for t in (1100, 1200, 1300):
        ap.admit_low("analyst_stub", "⚠️ <b>Analyst falling back to stub</b> — MSFT", now=t, state_path=s)
    text = ap.take_digest(now=2000, state_path=s)
    assert "Alert digest" in text and "MSFT ×2" in text   # first MSFT is a distinct alert, sent
    assert ap.take_digest(now=2001, state_path=s) is None
    assert ap.classify(text)[0] == ap.REPORT


def test_corrupt_state_admits_the_alert(tmp_path):
    s = tmp_path / "state.json"
    s.write_text("{not json")
    assert ap.admit_low("ingestion_health", "a", now=1, state_path=s)


def test_unwritable_state_admits_the_alert(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    assert ap.admit_low("ingestion_health", "a", state_path=blocker / "state.json")


def test_record_writes_masked_jsonl_and_never_raises(tmp_path):
    out = tmp_path / "outbox.jsonl"
    ap.record("tasks.py:x", "🚨 <b>UNPROTECTED POSITION — AAPL</b>\nPortfolio $50,000", ap.CRITICAL,
              "", "sent", now=0, outbox_path=out)
    row = json.loads(out.read_text())
    assert row["title"] == "🚨 UNPROTECTED POSITION — AAPL" and row["status"] == "sent"
    assert "50,000" not in out.read_text()
    blocker = tmp_path / "f"
    blocker.write_text("")
    ap.record("s", "t", ap.LOW, "x", "held", outbox_path=blocker / "o.jsonl")   # no raise


# ── alerts._send routing ──────────────────────────────────────────────────


def _resp(status=200, mid=7):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {"result": {"message_id": mid}}
    if status >= 400:
        import requests
        r.raise_for_status.side_effect = requests.HTTPError(f"{status} Client Error")
    return r


@pytest.fixture
def tg():
    import alerts
    with patch.object(alerts, "_TOKEN", "t"), patch.object(alerts, "_CHAT_ID", "c"), \
         patch("alerts._time.sleep"), patch("alerts.requests.post", return_value=_resp()) as post:
        yield alerts, post


def _outbox():
    return [json.loads(l) for l in ap.OUTBOX_PATH.read_text().splitlines()]


def test_critical_alert_repeats_are_all_sent_immediately(tg):
    alerts, post = tg
    for _ in range(10):
        assert alerts._send("🚨 UNPROTECTED POSITION — AAPL\nplace a stop") == 7
    assert post.call_count == 10
    rows = _outbox()
    assert len(rows) == 10 and {r["status"] for r in rows} == {"sent"}
    assert rows[0]["source"].endswith(":test_critical_alert_repeats_are_all_sent_immediately")


def test_low_alert_repeats_are_held_then_digested(tg):
    alerts, post = tg
    text = "⚠️ <b>Analyst falling back to stub</b> — AAPL: timeout after 300s"
    for _ in range(5):
        alerts._send(text)
    assert post.call_count == 1
    assert [r["status"] for r in _outbox()] == ["sent"] + ["held"] * 4
    alerts.flush_digest()
    assert post.call_count == 2
    assert "×4" in post.call_args.kwargs["json"]["text"]
    assert alerts.flush_digest() is None and post.call_count == 2


def test_explicit_severity_overrides_classification(tg):
    alerts, post = tg
    text = "⚠️ <b>Analyst falling back to stub</b> — AAPL: x"
    alerts._send(text)
    alerts._send(text, severity="critical")
    assert post.call_count == 2


def test_html_rejection_is_resent_as_plain_text(tg):
    alerts, post = tg
    post.side_effect = [_resp(400), _resp(200, 9)]
    assert alerts._send("🗓️ <b>Weekly Self-Review</b>\nwin rate <50% &amp; ok") == 9
    plain = post.call_args_list[1].kwargs["json"]
    assert "parse_mode" not in plain and plain["text"] == "🗓️ Weekly Self-Review\nwin rate <50% & ok"


def test_button_message_html_rejection_is_resent_as_plain_text(tg):
    alerts, post = tg
    post.side_effect = [_resp(400), _resp(200, 11)]
    assert alerts._send_with_buttons("<b>REC</b> x<y", [[{"text": "ok", "callback_data": "a:1"}]]) == 11
    plain = post.call_args_list[1].kwargs["json"]
    assert "parse_mode" not in plain and plain["reply_markup"]


def test_long_message_is_split_under_the_telegram_limit(tg):
    alerts, post = tg
    text = "🗓️ <b>Weekly Self-Review</b>\n" + "\n".join("line %d " % i + "x" * 90 for i in range(120))
    alerts._send(text)
    sent = [c.kwargs["json"]["text"] for c in post.call_args_list]
    assert len(sent) >= 3 and all(len(s) <= alerts._TELEGRAM_MAX_CHARS for s in sent)
    assert "\n".join(sent) == text


def test_failed_send_is_recorded_as_failed(tg):
    alerts, post = tg
    post.side_effect = Exception("network down")
    assert alerts._send("🚨 SHORT POSITION — AAPL") is None
    assert _outbox()[-1]["status"] == "failed"


def test_policy_error_falls_back_to_sending(tg):
    alerts, post = tg
    with patch("alert_policy.classify", side_effect=RuntimeError("boom")):
        assert alerts._send("⚠️ <b>Analyst falling back to stub</b> — AAPL") == 7


def test_very_long_message_is_capped_with_a_truncated_marker(tg):
    alerts, post = tg
    text = "🗓️ <b>Weekly Self-Review</b>\n" + ("y" * 100 + "\n") * 400
    alerts._send(text)
    sent = [c.kwargs["json"]["text"] for c in post.call_args_list]
    assert len(sent) == alerts._MAX_PARTS
    assert sent[-1].endswith("(truncated)") and all(len(s) <= alerts._TELEGRAM_MAX_CHARS for s in sent)


def test_button_message_is_truncated_with_marker(tg):
    alerts, post = tg
    alerts._send_with_buttons("x" * 5000, [])
    sent = post.call_args.kwargs["json"]["text"]
    assert len(sent) <= alerts._TELEGRAM_MAX_CHARS and sent.endswith("(truncated)")


def test_send_never_raises_when_everything_fails(tg):
    alerts, post = tg
    post.side_effect = RuntimeError("x")
    with patch("alert_policy.record", side_effect=None), patch("alerts._caller", side_effect=RuntimeError):
        assert alerts._send("🚨 SHORT POSITION — AAPL") is None


# Fake tokens only: digits:35 chars, the shape of a bot token.
_FAKE = "1234567890:" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345678"


@pytest.mark.parametrize("line", [
    f"send failed: https://api.telegram.org/bot{_FAKE}/sendMessage",
    f"token is {_FAKE} ok",
    f"<b>{_FAKE}</b>",
])
def test_title_never_keeps_any_part_of_a_bot_token(line):
    t = ap.title(line)
    assert "AbCdEfGhIjKlMnOp" not in t and "1234567890" not in t and "[REDACTED]" in t


def test_bare_token_is_redacted_in_logs_too():
    import log_redaction
    out = log_redaction.redact(f"x {_FAKE} y")
    assert "AbCdEfGh" not in out and "1234567890" not in out


@pytest.mark.parametrize("line,leak", [
    ("Portfolio 123,456.78 today", "123,456"),
    ("value 1,234 shares", "1,234"),
    ("filled at 187.25 each", "187.25"),
])
def test_title_masks_amounts_without_a_dollar_sign(line, leak):
    t = ap.title(line)
    assert leak not in t and "[amt]" in t


def test_a_failed_part_stops_the_remaining_parts(tg):
    """Worst-case blocking stays one message's retries (~36 s), not one per part."""
    alerts, post = tg
    post.side_effect = Exception("down")
    text = "🗓️ <b>Weekly Self-Review</b>\n" + ("z" * 100 + "\n") * 120
    assert len(alerts._chunks(text)) >= 3
    alerts._send(text)
    assert post.call_count == 3          # one part's three attempts


def test_a_later_part_failing_stops_there(tg):
    alerts, post = tg
    post.side_effect = [_resp(200, 5)] + [Exception("down")] * 3 + [_resp(200, 6)] * 3
    text = "🗓️ <b>Weekly Self-Review</b>\n" + ("z" * 100 + "\n") * 120
    assert alerts._send(text) == 5
    assert post.call_count == 4


def test_dedup_keeps_distinct_issues_of_one_type_apart(tmp_path):
    s = tmp_path / "state.json"
    a = "⚠️ <b>Analyst falling back to stub</b> — AAPL: timeout after 300s"
    b = "⚠️ <b>Analyst falling back to stub</b> — MSFT: timeout after 300s"
    assert ap.admit_low("analyst_stub", a, now=1000, state_path=s)
    assert ap.admit_low("analyst_stub", b, now=1001, state_path=s)       # other ticker
    assert not ap.admit_low("analyst_stub", a.replace("300", "301"), now=1002, state_path=s)


def test_dedup_key_keeps_words_and_drops_numbers():
    k = ap.dedup_key
    assert k("x", "news 45 min") == k("x", "news 46 min") != k("x", "edgar 45 min")
    assert k("x", "a") != k("y", "a")


def test_digest_lists_each_distinct_alert_of_a_type(tmp_path):
    s = tmp_path / "state.json"
    for t in ("AAPL", "MSFT"):
        text = f"⚠️ <b>Analyst falling back to stub</b> — {t}: timeout"
        ap.admit_low("analyst_stub", text, now=1000, state_path=s)
        ap.admit_low("analyst_stub", text, now=1100, state_path=s)
    digest = ap.take_digest(now=2000, state_path=s)
    assert "AAPL ×1" in digest.replace(": timeout", "") and "MSFT ×1" in digest.replace(": timeout", "")
    assert digest.count("<b>analyst_stub</b>") == 1


def test_monitor_recovered_is_keyed_per_monitor(tmp_path):
    s = tmp_path / "state.json"
    t = "✅ {} monitor recovered after {} failed checks."
    assert ap.dedup_key("monitor_recovered", t.format("check_trailing_stops", 4)) == \
        ap.dedup_key("monitor_recovered", t.format("check_trailing_stops", 9))
    assert ap.admit_low("monitor_recovered", t.format("check_trailing_stops", 4), now=1, state_path=s)
    assert ap.admit_low("monitor_recovered", t.format("check_profit_targets", 4), now=2, state_path=s)
    assert not ap.admit_low("monitor_recovered", t.format("check_trailing_stops", 7), now=3, state_path=s)


def test_failed_digest_keeps_its_counts(tg):
    alerts, post = tg
    text = "⚠️ <b>Analyst falling back to stub</b> — AAPL: x"
    alerts._send(text)
    alerts._send(text)
    post.side_effect = Exception("down")
    assert alerts.flush_digest() is None
    post.side_effect = None
    post.return_value = _resp(200, 8)
    assert alerts.flush_digest() == 8
    assert "×1" in post.call_args.kwargs["json"]["text"]
    assert alerts.flush_digest() is None


def test_alerts_held_after_the_snapshot_survive_the_clear(tmp_path):
    s = tmp_path / "state.json"
    t = "⚠️ <b>Analyst falling back to stub</b> — AAPL"
    ap.admit_low("analyst_stub", t, now=1, state_path=s)
    ap.admit_low("analyst_stub", t, now=2, state_path=s)
    text, snap = ap.digest_text(now=3, state_path=s)
    ap.admit_low("analyst_stub", t, now=4, state_path=s)       # held while sending
    ap.clear_digest(snap, state_path=s)
    assert "×1" in ap.take_digest(now=5, state_path=s)


def test_button_send_survives_a_record_failure(tg):
    alerts, post = tg
    with patch("alert_policy.record", side_effect=RuntimeError("x")), \
         patch("alerts._caller", side_effect=RuntimeError("y")):
        assert alerts._send_with_buttons("<b>REC</b>", []) == 7


def test_outbox_rotates_and_keeps_one_old_file(tmp_path, monkeypatch):
    out = tmp_path / "outbox.jsonl"
    monkeypatch.setattr(ap, "OUTBOX_ROTATE_BYTES", 300)
    for i in range(20):
        ap.record("s", f"🚨 alert {i}", ap.CRITICAL, "", "sent", now=i, outbox_path=out)
    old = tmp_path / "outbox.jsonl.1"
    assert old.exists() and out.stat().st_size <= 300 + 200
    assert not (tmp_path / "outbox.jsonl.2").exists()
    rows = [json.loads(l) for l in (old.read_text() + out.read_text()).splitlines()]
    assert rows[-1]["title"] == "🚨 alert N".replace("N", "19")
