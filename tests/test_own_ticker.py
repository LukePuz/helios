"""Record-only own_ticker_rejected flag on SCREENED_OUT provenance
(agents/own_ticker.py, audit 2026-11-22 06:56 #2): the screener's prompt and
decisions are unchanged."""
import hashlib
import importlib.util
import json
import os
from unittest.mock import patch

import pytest

from agents import crew, own_ticker, screener

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The screener's prompt_template_sha at d7cdac3, before this change.
SCREENER_TEMPLATE_SHA = "ee4bf82515e1"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Verbatim screener reasons from SCREENED_OUT rows, 11-05..11-20.
OWN_TICKER_REASONS = [
    "earnings beat and outlook with price target change, but no direct impact on other watchlist stocks",
    "earnings beat and guidance raise, but no new fact for watchlist stocks.",
    "Earnings result but no specific new fact affecting other watchlist stocks",
    "Q3 revenue growth is specific to NET and does not have a direct impact on other tech stocks in the watchlist.",
    "Earnings and revenue surpassing estimates is not directly relevant to the watchlist without additional "
    "specific information affecting related stocks.",
    "Dividend raise announcement for PWR; no direct impact on watchlist.",
    "BMO upgrade and revenue growth are not directly relevant to other watchlist stocks.",
    "Quarterly profit report; no new facts affecting other watchlist stocks.",
    "Third quarter results announcement for a speculative stock, no direct impact on the watchlist.",
]
OTHER_REASONS = [
    "Price target cut to $340 from $380.",
    "Quarterly results and guidance beat expectations",
    "earnings result and guidance change for AppLovin (APP)",
    "Outlook and Teams outage is a specific negative development for MSFT.",
    "Analyst price-target change with a stated reason but no position held.",
    "earnings news for AMAT, but we do not hold the stock.",
    "Micron's stock price reaction is driven by a specific event and not a general memory market recovery.",
    "Relevant to other watchlist stocks through the supply chain",     # cites them, no negation
    None, "", 42,
]


@pytest.mark.parametrize("reason", OWN_TICKER_REASONS)
def test_reasons_that_judge_by_other_names_match(reason):
    assert own_ticker.reason_match(reason)
    rec = own_ticker.own_ticker_record(True, True, reason)
    assert rec["own_ticker_rejected"] is True and rec["own_ticker_patterns_sha"] == own_ticker.PATTERNS_SHA


@pytest.mark.parametrize("reason", OTHER_REASONS)
def test_other_reasons_do_not_match(reason):
    assert own_ticker.reason_match(reason) is None
    assert own_ticker.own_ticker_record(True, True, reason)["own_ticker_rejected"] is False


@pytest.mark.parametrize("names, new_fact, want", [
    (True, True, True), (False, True, False), (True, False, False), (False, False, False),
    (None, True, None), (True, None, None), (None, None, None), (False, None, False)])
def test_flag_truth_table(names, new_fact, want):
    rec = own_ticker.own_ticker_record(names, new_fact, OWN_TICKER_REASONS[0])
    assert rec["own_ticker_rejected"] is want
    assert rec["own_ticker_reason_match"] == "other watchlist stocks"


def test_match_text_is_short_and_lowercased():
    m = own_ticker.reason_match("No impact on OTHER  WATCHLIST STOCKS at all")
    assert m == "other watchlist stocks" and len(m) <= 60


def test_patterns_sha_is_the_hash_of_the_patterns():
    assert own_ticker.PATTERNS_SHA == hashlib.sha256(
        "\n".join((own_ticker.OTHER_PATTERN, own_ticker.NEGATION_PATTERN)).encode()).hexdigest()[:12]


def test_record_never_raises():
    with patch.object(own_ticker, "reason_match", side_effect=RuntimeError("x")):
        rec = own_ticker.own_ticker_record(True, True, "r")
    assert rec["own_ticker_rejected"] is None and rec["own_ticker_error"] == "RuntimeError"


# --- the call site: crew.screen_event_only's SCREENED_OUT row ------------------

def _screened_out(screening):
    with patch("vector_store.store"), patch("agents.screener.screen", return_value=screening), \
         patch("db.log_signal") as log_signal:
        crew.screen_event_only({"ticker": "NET", "source": "news", "title": "Cloudflare beats", "text": "x"})
    return log_signal


def test_screened_out_row_carries_the_flag():
    prov = {"code_sha": "abc", "model": "qwen2.5:7b", "title_names_ticker": True}
    log_signal = _screened_out({"relevant": False, "new_fact": True, "provenance": prov,
                                "reason": OWN_TICKER_REASONS[1]})
    args, kw = log_signal.call_args
    assert args[1] == "SCREENED_OUT" and args[3] == OWN_TICKER_REASONS[1]     # reason unchanged
    p = kw["provenance"]
    assert p["own_ticker_rejected"] is True and p["own_ticker_reason_match"] == "watchlist"
    assert p["own_ticker_patterns_sha"] == own_ticker.PATTERNS_SHA
    assert {k: v for k, v in p.items()
            if not k.startswith(("own_ticker_", "material_event_"))} == prov
    assert "own_ticker_rejected" not in prov            # the screener's dict is not mutated
    json.dumps(p)


def test_screened_out_row_without_provenance_stays_without():
    log_signal = _screened_out({"relevant": False, "new_fact": True, "reason": OWN_TICKER_REASONS[0]})
    assert log_signal.call_args.kwargs["provenance"] is None


def test_a_failing_record_keeps_the_row():
    prov = {"model": "qwen2.5:7b", "title_names_ticker": True}
    with patch("agents.own_ticker.own_ticker_record", side_effect=RuntimeError("x")):
        log_signal = _screened_out({"relevant": False, "provenance": prov, "reason": "r"})
    assert {k: v for k, v in log_signal.call_args.kwargs["provenance"].items()
            if not k.startswith("material_event_")} == prov


def test_screened_in_rows_are_untouched():
    prov = {"model": "qwen2.5:7b", "title_names_ticker": True}
    with patch("vector_store.store"), \
         patch("agents.screener.screen", return_value={"relevant": True, "urgency": "eod", "new_fact": True,
                                                       "provenance": prov, "reason": OWN_TICKER_REASONS[0]}), \
         patch("db.log_signal") as log_signal:
        crew.screen_event_only({"ticker": "NET", "source": "news", "title": "t", "text": "x"})
    for c in log_signal.call_args_list:
        assert "own_ticker_rejected" not in (c.kwargs.get("provenance") or {})


def test_screener_prompt_and_decision_are_unchanged():
    """The template sha is the pre-change one, and screen() itself writes no own_ticker key."""
    assert hashlib.sha256(screener._SCREENER_PROMPT.encode()).hexdigest()[:12] == SCREENER_TEMPLATE_SHA
    reply = json.dumps({"relevant": False, "urgency": "eod", "reason": OWN_TICKER_REASONS[1], "new_fact": True})
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"):
        r = screener.screen({"ticker": "NET", "source": "news", "title": "Cloudflare beats estimates", "text": "x"})
    assert r["relevant"] is False and r["provenance"]["prompt_template_sha"] == SCREENER_TEMPLATE_SHA
    assert not any(k.startswith("own_ticker_") for k in r["provenance"])


# --- batch_verify own_ticker_rejected and the registry stage --------------------

def _so(i, **kw):
    return {"id": i, "has_provenance": True, "has_key": True, "value": False,
            "sha": own_ticker.PATTERNS_SHA, "error": None, **kw}


def test_check_ok_and_tallies():
    bv = _load("batch_verify_ot", "scripts/batch_verify.py")
    rows = [_so(1, value=True), _so(2), _so(3, value=None), {"id": 4, "has_provenance": False}]
    c = bv.check_own_ticker_rejected(rows, own_ticker.PATTERNS_SHA)
    assert c["status"] == "OK" and c["triggered"] == 3
    assert dict(c["reads"]["own_ticker_rejected"]) == {"true": 1, "false": 1, "none": 1}


def test_check_fails_on_missing_errors_and_another_sha():
    bv = _load("batch_verify_ot", "scripts/batch_verify.py")
    c = bv.check_own_ticker_rejected([_so(1), _so(2, has_key=False, sha=None)], own_ticker.PATTERNS_SHA)
    assert c["status"] == "FAIL" and c["missing"] == [2]
    c = bv.check_own_ticker_rejected([_so(1), _so(2, error="TypeError")], own_ticker.PATTERNS_SHA)
    assert c["status"] == "FAIL" and "error: TypeError" in c["reads"]["record"]
    c = bv.check_own_ticker_rejected([_so(1, sha="000000000000")], own_ticker.PATTERNS_SHA)
    assert c["status"] == "FAIL" and "000000000000" in c["why"]
    assert bv.check_own_ticker_rejected([], own_ticker.PATTERNS_SHA)["status"] == "WARN"
    assert bv.own_ticker_patterns_sha() == own_ticker.PATTERNS_SHA


def test_fetch_screened_out_is_read_only_and_selects_no_text():
    import inspect
    bv = _load("batch_verify_ot", "scripts/batch_verify.py")
    src = inspect.getsource(bv._fetch_screened_out)
    assert "SET TRANSACTION READ ONLY" in src and "rollback" in src
    assert "reasoning" not in src and "title" not in src


def test_registry_stage():
    reg = _load("shadow_registry_ot", "scripts/shadow_registry_check.py")
    st = next(s for s in reg.STAGES if s["stage"] == "own_ticker_rejected")
    assert st["switch"] is None and st["expect_rows"] == "conditional"
    assert st["trigger"] == (reg._SIG, "signal = 'SCREENED_OUT' AND provenance IS NOT NULL")
    assert st["sources"] == ((reg._SIG, "provenance ? 'own_ticker_rejected'"),)
    assert reg.writer_present(ROOT, st["writer"])
