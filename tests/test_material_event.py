"""Record-only material-event terms on SCREENED_OUT provenance and the
per-ticker deal record (agents/material_event.py, audit 2026-11-23 22:56
N1-N3): the screener's prompt, its decisions and the analyst's prompt are
unchanged. Titles are verbatim from screened rows, 11-09..11-23."""
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agents import analyst, crew, material_event as me, screener

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_NOTE = me.deal_state_note        # conftest stubs it per test; restored below
SCREENER_TEMPLATE_SHA = "ee4bf82515e1"
WL = {"CRWD": {"company": "CrowdStrike"}, "ZS": {"company": "Zscaler"},
      "MRVL": {"company": "Marvell Technology"}, "LITE": {"company": "Lumentum"},
      "AMZN": {"company": "Amazon"}, "OKLO": {"company": "Oklo"}, "ORCL": {"company": "Oracle"},
      "RKLB": {"company": "Rocket Lab"}, "SNOW": {"company": "Snowflake"},
      "BWXT": {"company": "BWX Technologies"}}


@pytest.fixture
def real_note(monkeypatch):
    monkeypatch.setattr(me, "deal_state_note", _REAL_NOTE)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── the title terms ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("title, want", [
    ("CrowdStrike to buy Zscaler in ~$43.9B all-stock deal (ZS:NASDAQ) - Seeking Alpha",
     ["to buy", "all-stock"]),
    ("Revenue rose 20%, but Zscaler withdrew its financial guidance after agreeing to be acquired.",
     ["acquire", "withdraws guidance"]),
    ("CrowdStrike explores acquisition of Zscaler – report", ["acquire"]),
    ("CrowdStrike-Zscaler Merger Could Make Cybersecurity Rival ‘More Formidable’", ["merger"]),
    ("Zscaler Soars 23% on $43.9 Billion CrowdStrike Buyout", ["buyout"]),
    ("CrowdStrike Holdings (CRWD) Agreed $44 Billion All Stock Cybersecurity Deal", ["all-stock"]),
    ("Acme enters definitive merger agreement with Beta", ["merger", "definitive agreement"]),
    ("Acme launches tender offer for Beta shares", ["tender offer"]),
    ("Applied Optoelectronics (AAOI) Could Be 63% Undervalued After $550 Million Stock Offering",
     ["offering"]),
    ("Acme withdraws full-year guidance", ["withdraws guidance"]),
    ("425  - Prospectuses and communications, business combinations", []),
    ("CrowdStrike Reports Third Quarter Fiscal Year 2027 Financial Results", []),
])
def test_terms(title, want):
    assert me.material_event_terms(title) == want


@pytest.mark.parametrize("title", [None, "", "   ", 42])
def test_terms_without_a_title_are_none(title):
    assert me.material_event_terms(title) is None


def test_terms_sha_is_the_hash_of_the_terms():
    assert me.TERMS_SHA == hashlib.sha256(
        "\n".join("\t".join(t) for t in me.TERMS).encode()).hexdigest()[:12]
    assert [t[0] for t in me.TERMS] == ["acquire", "to buy", "merger", "buyout", "all-stock",
                                        "definitive agreement", "tender offer", "offering",
                                        "withdraws guidance"]


@pytest.mark.parametrize("form, want", [
    ("425", "deal"), ("SC TO-T", "deal"), ("DEFM14A", "deal"), ("SC 14D9", "deal"),
    ("424B5", "offering"), ("S-3", "offering"), ("8-K", None), ("4", None), (None, None), ("", None)])
def test_form_kind(form, want):
    assert me.form_kind(form) == want


def test_record_keys():
    rec = me.material_event_record("CrowdStrike to buy Zscaler in all-stock deal; withdraws guidance",
                                   "8-K")
    assert rec == {"material_event_terms": ["to buy", "all-stock", "withdraws guidance"],
                   "material_event_groups": ["deal", "guidance"],
                   "material_event_form_kind": None, "material_event_terms_sha": me.TERMS_SHA}
    rec = me.material_event_record("425  - Prospectuses and communications, business combinations",
                                   "425")
    assert rec["material_event_terms"] == [] and rec["material_event_form_kind"] == "deal"
    json.dumps(rec)


def test_record_never_raises():
    with patch.object(me, "material_event_terms", side_effect=RuntimeError("x")):
        rec = me.material_event_record("t")
    assert rec["material_event_terms"] is None and rec["material_event_error"] == "RuntimeError"


# ── role, consideration, deal items ─────────────────────────────────────────

@pytest.mark.parametrize("ticker, title, want", [
    ("CRWD", "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal", "acquirer"),
    ("ZS", "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal", "target"),
    ("CRWD", "CrowdStrike (CRWD) in Talks to Buy Zscaler (ZS), Sending Shares in Opposite Directions",
     "acquirer"),
    ("ZS", "Crowdstrike is spending $43.9 billion to buy ZScaler in a huge cybersecurity deal", "target"),
    ("CRWD", "CrowdStrike Announces $43.9 Billion Zscaler Acquisition and Raises Fiscal 2027 Guidance",
     "acquirer"),
    ("ZS", "CrowdStrike Announces $43.9 Billion Zscaler Acquisition and Raises Fiscal 2027 Guidance",
     "target"),
    ("CRWD", "CrowdStrike's Zscaler Acquisition Could Broaden Security Platform, Says JPMorgan", "acquirer"),
    ("ZS", "CrowdStrike's Zscaler Acquisition Could Broaden Security Platform, Says JPMorgan", "target"),
    ("ZS", "Revenue rose 20%, but Zscaler withdrew its financial guidance after agreeing to be acquired.",
     "target"),
    ("CRWD", "Zscaler Jumps 21% as CrowdStrike Moves to Acquire It", "acquirer"),
    ("CRWD", "Zscaler Stock Surged on CrowdStrike's $43.9 Billion Buyout.", "acquirer"),
    ("LITE", "Marvell Acquires Lumentum in Blockbuster $80 Billion Deal", "target"),
    ("MRVL", "Marvell Acquires Lumentum in Blockbuster $80 Billion Deal", "acquirer"),
    ("AMZN", "Amazon agrees to buy Decart AI technology and team in reported $7.5 billion deal", "acquirer"),
    # not a role: "<name> buyout" reads either way; listicles; a rating
    ("ZS", "Zscaler Soars 23% on $43.9 Billion CrowdStrike Buyout", None),
    ("CRWD", "Zscaler Soars 23% on $43.9 Billion CrowdStrike Buyout", None),
    ("ORCL", "Down 40% in 2026, Is it Time to Buy Oracle Stock on the Dip?", None),
    ("RKLB", "There Are 8 Billion Reasons to Buy Rocket Lab At $50", None),
    ("OKLO", "Guggenheim Upgrades Oklo to Buy From Hold, Price Target is $38", None),
    ("CRWD", "CrowdStrike Holdings (CRWD) Agreed $44 Billion All Stock Cybersecurity Deal", None),
    ("CRWD", None, None),
])
def test_role(ticker, title, want):
    assert me.deal_role(ticker, WL[ticker]["company"], title, WL) == want


def test_role_both_ways_is_mixed():
    assert me.deal_role("CRWD", "CrowdStrike",
                        "CrowdStrike to acquire Acme; Zscaler to buy CrowdStrike", WL) == "mixed"


@pytest.mark.parametrize("title, want", [
    ("CrowdStrike to buy Zscaler in ~$43.9B all-stock deal", "stock"),
    ("CrowdStrike to Buy Zscaler in $44 Billion Stock Swap", "stock"),
    ("Acme to buy Beta in all-cash deal", "cash"),
    ("Acme to buy Beta in cash-and-stock deal", "cash_and_stock"),
    ("CrowdStrike to acquire Zscaler for $44 billion", None), (None, None)])
def test_consideration(title, want):
    assert me.deal_consideration(title) == want


def _item(ticker, title, form=None, names=True):
    return me.deal_item(ticker, WL[ticker]["company"], title, form, me.material_event_terms(title),
                        names, WL)


def test_deal_items():
    it = _item("ZS", "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal")
    assert it == {"via": "title", "role": "target", "consideration": "stock",
                  "terms": ["to buy", "all-stock"]}
    it = _item("ZS", "425  - Prospectuses and communications, business combinations", "425", False)
    assert it == {"via": "form", "role": None, "consideration": None, "terms": []}
    it = _item("CRWD", "CrowdStrike to acquire Zscaler", "425")
    assert it["via"] == "form+title" and it["role"] == "acquirer"


@pytest.mark.parametrize("ticker, title, names", [
    ("ORCL", "Down 40% in 2026, Is it Time to Buy Oracle Stock on the Dip?", True),   # bare "to buy"
    ("OKLO", "Guggenheim Upgrades Oklo to Buy From Hold, Price Target is $38", True),
    ("SNOW", "Snowflake Inc. $SNOW Shares Acquired by First Trust Advisors LP", True),  # holdings
    ("BWXT", "BWX Technologies (BWXT) HR chief acquires 124 shares at $119.52 each.", True),
    ("ZS", "CrowdStrike to buy Zscaler", False),         # title doesn't name the ticker
    ("ZS", "Zscaler Reports Results", True),             # no deal term
    ("ZS", None, None),
])
def test_not_deal_items(ticker, title, names):
    assert _item(ticker, title, None, names) is None


# ── the deal record ──────────────────────────────────────────────────────────

def _apply(book, ticker, title, item_id, signal="SCREENED_OUT", form=None, at="2026-11-23T16:24:00+00:00"):
    item = _item(ticker, title, form, form is None or ticker in (title or ""))
    return me.deal_state_apply(book, ticker, item, item_id=item_id, title=title, signal=signal,
                               actual_form=form, source="news" if form is None else "sec_form4",
                               at_iso=at, et_date=at[:10])


def test_deal_record_accumulates():
    book = _apply({}, "ZS", "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal", "u1")
    book = _apply(book, "ZS", "425  - Prospectuses and communications, business combinations", "u2",
                  form="425", at="2026-11-23T17:42:00+00:00")
    book = _apply(book, "ZS", "Revenue rose 20%, but Zscaler withdrew its financial guidance after "
                              "agreeing to be acquired.", "u3", signal="SCREENED_IN",
                  at="2026-11-23T18:00:00+00:00")
    r = book["ZS"]
    assert r["first_seen_at"] == "2026-11-23T16:24:00+00:00" and r["announce_date"] == "2026-11-23"
    assert r["first_item_id"] == "u1" and r["last_item_id"] == "u3"
    assert r["items"] == 3 and r["screened_out"] == 2 and r["screened_in"] == 1
    assert r["via"] == {"title": 2, "form": 1}
    assert r["role"] == "target" and r["role_votes"] == {"target": 2}
    assert r["consideration"] == "stock" and r["forms_seen"] == ["425"]
    assert r["terms_seen"] == ["acquire", "all-stock", "to buy"]
    assert r["sources_seen"] == ["news", "sec_form4"]
    assert r["terms_sha"] == me.TERMS_SHA and r["deal_sha"] == me.DEAL_SHA
    json.dumps(book)


def test_deal_record_counts_an_item_once():
    book = _apply({}, "ZS", "CrowdStrike to buy Zscaler", "u1")
    assert _apply(book, "ZS", "CrowdStrike to buy Zscaler", "u1") is None


def test_deal_record_mixed_role_and_cap(monkeypatch):
    book = _apply({}, "CRWD", "CrowdStrike to acquire Zscaler", "u1")
    book = _apply(book, "CRWD", "Zscaler to buy CrowdStrike", "u2")
    assert book["CRWD"]["role"] == "mixed"
    monkeypatch.setattr(me, "_DEAL_STATE_MAX", 2)
    book = _apply(book, "ZS", "CrowdStrike to buy Zscaler", "u3", at="2026-11-24T00:00:00+00:00")
    book = _apply(book, "LITE", "Marvell Acquires Lumentum", "u4", at="2026-11-25T00:00:00+00:00")
    assert set(book) == {"ZS", "LITE"}            # the least recently seen goes first


class _State:
    def __init__(self, raw="{}", fail=False):
        self.raw, self.fail, self.writes = raw, fail, []

    def get(self, key, default=""):
        if self.fail:
            raise RuntimeError("db down")
        return self.raw

    def set(self, key, value):
        self.writes.append((key, json.loads(value)))
        self.raw = value


def _note(state, ticker, title, form=None, prov=None, signal="SCREENED_OUT"):
    with patch("db.get_state_strict", side_effect=state.get), patch("db.set_state", side_effect=state.set), \
         patch("knowledge_loader.load_watchlist", return_value=WL):
        return me.deal_state_note(ticker, {"ticker": ticker, "title": title, "url": f"u:{title}",
                                           "source": "news", "actual_form": form}, signal, prov,
                                  now=datetime(2026, 11, 23, 16, 24, tzinfo=timezone.utc))


def test_note_writes_the_env_key(real_note):
    st = _State()
    assert _note(st, "ZS", "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal",
                 prov={"title_names_ticker": True}) is True
    key, book = st.writes[0]
    import config
    assert key == f"deal_state_{config.TRADING_ENV}" and book["ZS"]["role"] == "target"
    assert book["ZS"]["announce_date"] == "2026-11-23"


def test_note_skips_non_deal_items_without_io(real_note):
    st = _State(fail=True)
    assert _note(st, "ORCL", "Is it Time to Buy Oracle Stock on the Dip?",
                 prov={"title_names_ticker": True}) is False
    assert _note(st, "ZS", "Zscaler Reports Results", prov={"title_names_ticker": True}) is False


def test_note_reads_the_title_test_when_the_provenance_has_none(real_note):
    st = _State()
    assert _note(st, "ZS", "CrowdStrike to buy Zscaler", prov=None, signal="SCREENED_IN") is True
    assert st.writes[0][1]["ZS"]["screened_in"] == 1


def test_note_never_raises_and_a_failed_read_writes_nothing(real_note):
    st = _State(fail=True)
    assert _note(st, "ZS", "425  - x", form="425") is False and st.writes == []


def test_note_resets_a_bad_book(real_note, caplog):
    st = _State(raw="[1, 2]")
    assert _note(st, "ZS", "425  - x", form="425") is True
    assert set(st.writes[0][1]) == {"ZS"} and "reset to {}" in caplog.text


def test_note_skips_on_a_busy_lock(real_note, monkeypatch):
    class Busy:
        def acquire(self):
            return False

        def release(self):
            pass
    import stop_lock
    monkeypatch.setattr(type(stop_lock._redis()), "lock", lambda *a, **k: Busy())
    st = _State()
    assert _note(st, "ZS", "425  - x", form="425") is False and st.writes == []


# ── the call sites ───────────────────────────────────────────────────────────

def _screen(screening, title, form=None):
    with patch("vector_store.store"), patch("agents.screener.screen", return_value=screening), \
         patch("db.log_signal") as log_signal, patch("agents.material_event.deal_state_note") as note:
        crew.screen_event_only({"ticker": "ZS", "source": "sec_form4" if form else "news",
                                "title": title, "text": "x", "actual_form": form})
    return log_signal, note


def test_screened_out_row_carries_the_terms_and_notes_the_deal():
    prov = {"model": "qwen2.5:7b", "title_names_ticker": True, "actual_form": None}
    title = "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal"
    log_signal, note = _screen({"relevant": False, "provenance": prov, "reason": "Recap piece"}, title)
    args, kw = log_signal.call_args
    assert args[1] == "SCREENED_OUT" and args[3] == "Recap piece"       # reason unchanged
    p = kw["provenance"]
    assert p["material_event_terms"] == ["to buy", "all-stock"]
    assert p["material_event_groups"] == ["deal"] and p["material_event_terms_sha"] == me.TERMS_SHA
    assert "material_event_terms" not in prov            # the screener's dict is not mutated
    assert note.call_args.args[:3] == ("ZS", note.call_args.args[1], "SCREENED_OUT")
    assert note.call_args.args[3]["material_event_terms"] == ["to buy", "all-stock"]


def test_a_425_is_recorded_by_its_form():
    prov = {"model": "qwen2.5:7b", "title_names_ticker": False, "actual_form": "425"}
    log_signal, _ = _screen({"relevant": False, "provenance": prov, "reason": None},
                            "425  - Prospectuses and communications, business combinations", "425")
    p = log_signal.call_args.kwargs["provenance"]
    assert p["material_event_terms"] == [] and p["material_event_form_kind"] == "deal"
    assert p["actual_form"] == "425"


def test_screened_in_rows_get_no_terms_but_note_the_deal():
    prov = {"model": "qwen2.5:7b", "title_names_ticker": True}
    log_signal, note = _screen({"relevant": True, "urgency": "eod", "provenance": prov, "reason": "r"},
                               "Zscaler withdrew its financial guidance after agreeing to be acquired")
    for c in log_signal.call_args_list:
        assert "material_event_terms" not in (c.kwargs.get("provenance") or {})
    assert note.call_args.args[2] == "SCREENED_IN" and note.call_args.args[3] is prov


def test_a_failing_record_keeps_the_row():
    prov = {"model": "qwen2.5:7b", "title_names_ticker": True}
    with patch("agents.material_event.material_event_record", side_effect=RuntimeError("x")):
        log_signal, _ = _screen({"relevant": False, "provenance": prov, "reason": "r"}, "t")
    assert not any(k.startswith("material_event_") for k in log_signal.call_args.kwargs["provenance"])


def test_a_failing_note_keeps_the_screen():
    with patch("vector_store.store"), \
         patch("agents.screener.screen", return_value={"relevant": False, "reason": "r"}), \
         patch("db.log_signal"), patch("agents.material_event.deal_state_note",
                                       side_effect=RuntimeError("x")):
        out = crew.screen_event_only({"ticker": "ZS", "source": "news", "title": "t", "text": "x"})
    assert out["relevant"] is False


def test_screener_prompt_and_decision_are_unchanged():
    assert hashlib.sha256(screener._SCREENER_PROMPT.encode()).hexdigest()[:12] == SCREENER_TEMPLATE_SHA
    reply = json.dumps({"relevant": False, "urgency": "eod", "reason": "Recap piece", "new_fact": False})
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"):
        r = screener.screen({"ticker": "ZS", "source": "news", "text": "x",
                             "title": "CrowdStrike to buy Zscaler in ~$43.9B all-stock deal"})
    assert r["relevant"] is False
    assert not any(k.startswith("material_event_") for k in r["provenance"])


def test_trigger_record_carries_the_filing_s_own_form():
    rec = analyst.trigger_record("event", {"source": "sec_form4", "type": "Form 4", "actual_form": "425",
                                           "title": "425  - Prospectuses and communications"})
    assert rec["actual_form"] == "425" and rec["source"] == "sec_form4" and rec["type"] == "Form 4"
    assert analyst.trigger_record("event", {"source": "news", "title": "t"})["actual_form"] is None


def test_registry_stage():
    reg = _load("shadow_registry_me", "scripts/shadow_registry_check.py")
    st = next(s for s in reg.STAGES if s["stage"] == "material_event_terms")
    assert st["switch"] is None and st["expect_rows"] == "conditional"
    assert st["sources"] == ((reg._SIG, "provenance ? 'material_event_terms'"),)
    assert reg.writer_present(ROOT, st["writer"])
