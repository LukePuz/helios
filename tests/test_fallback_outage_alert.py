"""A keyword-fallback screen has new_fact None, which the new-fact gate
counts as false, so a 7B outage blocks every news-path buy. Two throttled
Telegram alerts say so (alert only; routing is unchanged): the 3rd fallback
within 15 minutes, and a buy blocked for new_fact missing after a fallback."""
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import config
import test_new_fact_gate as nfg
from agents import crew, screener

T0 = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
_EVENT = {"ticker": "NVDA", "source": "news", "title": "NVIDIA wins DoD contract", "text": "x"}


def _state(state):
    return (patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)),
            patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)))


# ── (a) the screener on the keyword fallback ─────────────────────────────────

def test_third_fallback_in_15_min_alerts_once_per_hour():
    state = {}
    get, put = _state(state)
    with patch("alerts._send", return_value=7) as send, get, put:
        n = screener._note_keyword_fallback
        assert n("ollama_unavailable", now=T0) is False
        assert n("ollama_unavailable", now=T0 + timedelta(minutes=5)) is False
        assert n("no_json", now=T0 + timedelta(minutes=10)) is True
        assert n("no_json", now=T0 + timedelta(minutes=11)) is False      # throttled
        assert n("no_json", now=T0 + timedelta(minutes=69)) is False      # one in the window
        assert n("no_json", now=T0 + timedelta(minutes=70)) is False
        assert n("no_json", now=T0 + timedelta(minutes=71)) is True       # 3rd, an hour later
    assert send.call_count == 2
    assert "keyword fallback" in send.call_args_list[0].args[0]


def test_spread_out_fallbacks_do_not_alert():
    state = {}
    get, put = _state(state)
    with patch("alerts._send", return_value=7) as send, get, put:
        for i in range(5):
            assert screener._note_keyword_fallback("invalid_json",
                                                   now=T0 + timedelta(minutes=10 * i)) is False
    send.assert_not_called()


def test_failed_send_does_not_start_the_throttle():
    state = {}
    get, put = _state(state)
    with patch("alerts._send", return_value=None), get, put:
        for i in range(3):
            assert screener._note_keyword_fallback("x", now=T0 + timedelta(minutes=i)) is False
    assert f"screener_fallback_alert:{config.TRADING_ENV}" not in state


def test_fallback_alert_never_raises():
    with patch("db.get_state", side_effect=RuntimeError("db down")), patch("alerts._send") as send:
        assert screener._note_keyword_fallback("x") is False
    send.assert_not_called()


def _screen(reply):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value=None), \
         patch("agents.screener._note_keyword_fallback") as note:
        r = screener.screen(dict(_EVENT))
    return r, note


def test_screen_counts_each_fallback_and_only_fallbacks():
    for reply, reason in ((None, "ollama_unavailable"), ("no json here", "no_json"),
                          ('{"relevant": tru}', "invalid_json")):
        r, note = _screen(reply)
        note.assert_called_once_with(reason)
        assert r["new_fact"] is None
    r, note = _screen('{"relevant": true, "urgency": "eod", "reason": "x", "new_fact": true}')
    note.assert_not_called()
    assert r["new_fact"] is True


def test_ollama_unavailable_logs_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="agents.screener"):
        _screen(None)
    assert any("keyword fallback" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


# ── (b) a buy blocked for a fallback's missing new_fact ──────────────────────

_MISSING = {"passed": False, "item_new_fact_missing": True}


def test_block_after_fallback_alerts_once_per_hour():
    state = {}
    get, put = _state(state)
    trig = {"path": "event", "new_fact": None, "screener_fallback_reason": "ollama_unavailable"}
    with patch("alerts._send", return_value=3) as send, get, put:
        a = crew._alert_new_fact_fallback_block
        assert a("NVDA", _MISSING, trig, now=T0) is True
        assert a("AMD", _MISSING, trig, now=T0 + timedelta(minutes=30)) is False
        assert a("AMD", _MISSING, trig, now=T0 + timedelta(minutes=61)) is True
    assert send.call_count == 2
    assert "keyword fallback (ollama_unavailable)" in send.call_args_list[0].args[0]


def test_real_false_or_7b_missing_does_not_alert():
    with patch("db.get_state", return_value=None), patch("db.set_state"), \
         patch("alerts._send", return_value=3) as send:
        a = crew._alert_new_fact_fallback_block
        # a real false: the gate does not mark it missing
        assert a("NVDA", {"passed": False}, {"path": "event", "new_fact": False,
                                             "screener_fallback_reason": None}) is False
        # missing, but the 7B judged it (no fallback)
        assert a("NVDA", _MISSING, {"path": "event", "new_fact": None,
                                    "screener_fallback_reason": None}) is False
        assert a("NVDA", _MISSING, None) is False
    send.assert_not_called()


def test_block_alert_never_raises():
    trig = {"screener_fallback_reason": "no_json"}
    with patch("db.get_state", return_value=None), \
         patch("alerts._send", side_effect=RuntimeError("telegram down")):
        assert crew._alert_new_fact_fallback_block("NVDA", _MISSING, trig) is False


def test_route_block_after_fallback_sends_the_alert_and_still_blocks():
    trig = {"path": "event", "new_fact": None, "screener_fallback_reason": "no_json"}
    with patch("agents.crew._alert_new_fact_fallback_block") as alert:
        rm_cls, lpa, ls = nfg._route(nfg._result("OPEN", trig))
    lpa.assert_not_called()
    rm_cls.assert_not_called()
    assert ls.call_args.args[1] == "ROUTE_NO_NEW_FACT"
    alert.assert_called_once()
    assert alert.call_args.args[0] == "NVDA"
    assert alert.call_args.args[1]["item_new_fact_missing"] is True
    assert alert.call_args.args[2] == trig


def test_trigger_carries_the_screener_fallback_reason():
    seen = {}

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        seen["trigger"] = trigger
        return {"action": "HOLD", "ticker": "NVDA", "conviction": 5, "pass_to_risk": False}
    for sprov, want in (({"fallback_reason": "ollama_unavailable"}, "ollama_unavailable"),
                        ({"model_digest": "d"}, None)):
        with patch("ibkr.IBKRClient", side_effect=RuntimeError("no broker")), \
             patch("agents.analyst.analyze", fake_analyze), \
             patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
             patch("knowledge_loader.get_watchlist_entry", return_value={}), \
             patch("agents.crew._c6_bar_unrouted"):
            crew.deep_analyze(dict(_EVENT), {"relevant": True, "new_fact": None, "provenance": sprov})
        assert seen["trigger"]["screener_fallback_reason"] == want
