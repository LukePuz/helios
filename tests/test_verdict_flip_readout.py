"""scripts/verdict_flip_readout.py on fixture signals rows: which rows are
normalised how, what counts as a flip (same ticker, same New York day,
consecutive, signal differs), the gap / conviction / UDR / same-event
fields, the per-day totals and the read-only fetch."""
import importlib.util
import inspect
import os
import re
from datetime import date, datetime, timezone

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "verdict_flip_readout.py")
_spec = importlib.util.spec_from_file_location("verdict_flip_readout", _PATH)
vfr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vfr)

NY = vfr.NY
NOW = datetime(2026, 11, 25, 15, tzinfo=timezone.utc)
D24, D25 = date(2026, 11, 24), date(2026, 11, 25)


def _et(d, h, m=0, s=0):
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=NY)


def _row(i, at, signal, conv=7, udr=2.0, ticker="NVDA", story="s1", path="event", **kw):
    trig = {"path": path}
    if story:
        trig["story_dedup"] = {"story_key": story}
    row = {"id": i, "symbol": ticker, "timestamp": at.astimezone(timezone.utc),
           "signal": signal, "confidence": conv / 10 if conv is not None else None,
           "decision": {"final_conviction": conv, "model_action": signal,
                        "udr_check": {"model": udr} if udr is not None else {}},
           "trigger": trig}
    row.update(kw)
    return row


def _v(*rows):
    return [vfr.verdict(r) for r in rows]


def test_verdict_normalises_fields_and_ny_day():
    v = vfr.verdict(_row(1, _et(D24, 23, 30), "ADD", conv=7, udr=3.17))
    assert v["day"] == D24          # 04:30 UTC on 11-25 is still 11-24 in New York
    assert (v["signal"], v["conviction"], v["udr"]) == ("ADD", 7.0, 3.17)
    assert (v["event_key"], v["event_basis"], v["path"]) == ("s1", "story_key", "event")


def test_conviction_falls_back_to_confidence():
    r = _row(1, _et(D24, 10), "HOLD")
    r["decision"].pop("final_conviction")
    r["confidence"] = 0.6
    assert vfr.verdict(r)["conviction"] == 6.0


def test_event_key_falls_back_to_url_then_title():
    r = _row(1, _et(D24, 10), "HOLD", story=None)
    r["trigger"].update(url="https://x/a", title="T")
    assert vfr.verdict(r)["event_basis"] == "url"
    r["trigger"].pop("url")
    assert vfr.verdict(r)["event_basis"] == "title"
    r["trigger"].pop("title")
    assert vfr.verdict(r)["event_key"] is None


def test_missing_decision_and_trigger_are_tolerated():
    r = _row(1, _et(D24, 10), "HOLD")
    r["decision"], r["trigger"] = None, None
    v = vfr.verdict(r)
    assert v["udr"] is None and v["event_key"] is None and v["conviction"] == 7.0


def test_flip_fields_reproduce_the_audit_example():
    vs = _v(_row(103808, _et(D25, 7, 17, 15), "ADD", conv=7, udr=3.17, story="a"),
            _row(103812, _et(D25, 7, 18, 56), "HOLD", conv=6, udr=1.67, story="b"))
    (f,) = vfr.flips(vfr.group(vs)[(D25, "NVDA")])
    assert f["gap_min"] == 1.7
    assert (f["conv_changed"], f["udr_changed"], f["same_event"]) == (True, True, False)


def test_same_signal_is_not_a_flip_and_pairs_are_consecutive():
    vs = _v(_row(1, _et(D24, 9), "HOLD"), _row(2, _et(D24, 9, 10), "HOLD"),
            _row(3, _et(D24, 9, 20), "ADD"), _row(4, _et(D24, 9, 40), "HOLD"))
    fl = vfr.flips(vfr.group(vs)[(D24, "NVDA")])
    assert [(f["a"]["id"], f["b"]["id"]) for f in fl] == [(2, 3), (3, 4)]
    assert [f["gap_min"] for f in fl] == [10.0, 20.0]


def test_rows_are_ordered_by_time_then_id():
    vs = _v(_row(5, _et(D24, 9, 30), "ADD"), _row(2, _et(D24, 9), "HOLD"),
            _row(4, _et(D24, 9, 30), "HOLD"))
    rows = vfr.group(vs)[(D24, "NVDA")]
    assert [v["id"] for v in rows] == [2, 4, 5]
    assert [(f["a"]["id"], f["b"]["id"]) for f in vfr.flips(rows)] == [(4, 5)]


def test_pairs_across_days_and_tickers_are_not_compared():
    vs = _v(_row(1, _et(D24, 15), "HOLD"), _row(2, _et(D25, 9), "ADD"),
            _row(3, _et(D25, 9, 1), "WATCH", ticker="AMD"))
    g = vfr.group(vs)
    assert sum(len(vfr.flips(rows)) for rows in g.values()) == 0


def test_unchanged_and_unknown_fields():
    vs = _v(_row(1, _et(D24, 9), "HOLD", conv=7, udr=2.0, story="s"),
            _row(2, _et(D24, 9, 3), "ADD", conv=7, udr=None, story="s"),
            _row(3, _et(D24, 9, 6), "HOLD", conv=7, udr=1.0, story=None, path="scheduled_review"))
    f1, f2 = vfr.flips(vfr.group(vs)[(D24, "NVDA")])
    assert (f1["conv_changed"], f1["udr_changed"], f1["same_event"]) == (False, None, True)
    assert (f2["conv_changed"], f2["udr_changed"], f2["same_event"]) == (False, None, None)


def test_same_event_needs_the_same_basis():
    a = _row(1, _et(D24, 9), "HOLD", story=None)
    a["trigger"]["url"] = "u"
    b = _row(2, _et(D24, 9, 1), "ADD", story="u")
    (f,) = vfr.flips(vfr.group(_v(a, b))[(D24, "NVDA")])
    assert f["same_event"] is None


def _fixture():
    return _v(_row(1, _et(D24, 9), "HOLD", udr=2.0, story="a"),
              _row(2, _et(D24, 9, 4), "ADD", udr=3.0, story="a"),      # 4 min, same event
              _row(3, _et(D24, 9, 30), "HOLD", conv=6, udr=1.5, story="b"),  # 26 min
              _row(4, _et(D24, 11), "ADD", udr=2.5, story="c"),         # 90 min
              _row(5, _et(D24, 10), "WATCH", ticker="MSFT", story="m"),
              _row(6, _et(D25, 7, 17, 15), "ADD", udr=3.17, story="x"),
              _row(7, _et(D25, 7, 18, 56), "HOLD", conv=6, udr=1.67, story="y"))


def test_readout_per_day_totals():
    text = vfr.readout(_fixture(), D24, D25, "live", "abc1234", NOW)
    assert ("| 2026-11-24 | 5 | 2 | 3 | 1 | 1 | 2 | 1 of 3 | 0 of 3 | 1 of 3 |") in text
    assert ("| 2026-11-25 | 2 | 1 | 1 | 1 | 1 | 1 | 0 of 1 | 0 of 1 | 0 of 1 |") in text
    assert "Total: 7 verdicts, 4 flips, 2 within 5 min, 3 within 30 min." in text


def test_readout_per_ticker_and_detail_rows():
    text = vfr.readout(_fixture(), D24, D25, "live", "abc1234", NOW)
    assert "| 2026-11-24 | NVDA | 4 | 3 | 1 | 2 | ADD->HOLD 1, HOLD->ADD 2 |" in text
    assert "| 2026-11-24 | MSFT |" not in text.split("## 3.")[0].split("## 2.")[1]
    assert ("| 2026-11-25 | NVDA | ADD 07:17:15 | HOLD 07:18:56 | 1.7 | 7 -> 6 | 3.17 -> 1.67 "
            "| yes | yes | no | event -> event | 6, 7 |") in text


def test_readout_no_detail_and_empty():
    text = vfr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, detail=False)
    assert "## 3." not in text
    empty = vfr.readout([], D24, D25, "live", "abc1234", NOW, ticker="NVDA")
    assert "ticker NVDA" in empty
    assert "Total: 0 verdicts, 0 flips" in empty


def test_readout_has_no_prices_thresholds_or_account_words():
    text = vfr.readout(_fixture(), D24, D25, "live", "abc1234", NOW)
    assert "$" not in text
    assert not re.search(r"shares|account|recommend|should", text, re.I)


def test_fetch_is_read_only_and_verdict_rows_only():
    src = inspect.getsource(vfr._fetch)
    assert "SET TRANSACTION READ ONLY" in src
    assert "rollback()" in src
    assert "decision ? 'final_action'" in src
    for word in ("INSERT", "UPDATE", "DELETE", "commit("):
        assert word not in src


def _path_fixture():
    return _v(_row(1, _et(D24, 9), "HOLD", udr=1.5, path="event", story="a"),
              _row(2, _et(D24, 9, 3), "ADD", udr=3.0, path="event", story="a"),   # event->event, 3 min
              _row(3, _et(D24, 9, 40), "HOLD", udr=3.0, path="scheduled_review",
                   story=None),                                                   # event->sr, 37 min
              _row(4, _et(D24, 10), "WATCH", udr=2.0, ticker="AMD", path="daily_scan", story=None),
              _row(5, _et(D24, 10, 20), "OPEN", udr=2.6, ticker="AMD", path="event",
                   story="m"),                                                    # ds->event, 20 min
              _row(6, _et(D25, 9), "HOLD", udr=1.2, path=None, story=None),
              _row(7, _et(D25, 9, 2), "ADD", udr=1.2, path="event", story="z"))  # none->event, 2 min


def test_by_path_section_absent_by_default():
    assert "## 4." not in vfr.readout(_path_fixture(), D24, D25, "live", "abc1234", NOW)


def test_by_path_verdict_counts_per_day_and_path():
    text = vfr.readout(_path_fixture(), D24, D25, "live", "abc1234", NOW, by_path_section=True)
    sec = text.split("## 4. By trigger path")[1]
    for row in ("| 2026-11-24 | event | 3 |", "| 2026-11-24 | scheduled_review | 1 |",
                "| 2026-11-24 | daily_scan | 1 |", "| 2026-11-25 | event | 1 |",
                "| 2026-11-25 | none | 1 |"):
        assert row in sec


def test_by_path_flip_pairs_with_tallies():
    text = vfr.readout(_path_fixture(), D24, D25, "live", "abc1234", NOW, by_path_section=True)
    sec = text.split("## 4. By trigger path")[1]
    assert "| 2026-11-24 | event | event | 1 | 1 | 1 | 1 of 1 | 0 of 1 | 1 of 1 |" in sec
    assert "| 2026-11-24 | event | scheduled_review | 1 | 0 | 0 | 1 of 1 | 1 of 1 | 0 of 0 |" in sec
    assert "| 2026-11-24 | daily_scan | event | 1 | 0 | 1 | 1 of 1 | 0 of 1 | 0 of 0 |" in sec
    assert "| 2026-11-25 | none | event | 1 | 1 | 1 | 1 of 1 | 1 of 1 | 0 of 0 |" in sec
    assert "| all | event | event | 1 |" in sec
    # every flip lands in exactly one per-day pair row
    day_rows = [l for l in sec.splitlines() if l.startswith("| 2026-11-2") and l.count("|") == 10]
    assert sum(int(l.split("|")[4]) for l in day_rows) == 4


def test_by_path_tallies_match_section_one():
    vs = _fixture()
    text = vfr.readout(vs, D24, D25, "live", "abc1234", NOW, by_path_section=True)
    sec = text.split("## 4. By trigger path")[1]
    # _fixture is all event-path: the per-day pair row repeats section 1's tallies
    assert "| 2026-11-24 | event | event | 3 | 1 | 2 | 1 of 3 | 0 of 3 | 1 of 3 |" in sec
    assert "| all | event | event | 4 | 2 | 3 | 1 of 4 | 0 of 4 | 1 of 4 |" in sec


def test_by_path_empty():
    text = vfr.readout([], D24, D25, "live", "abc1234", NOW, by_path_section=True)
    sec = text.split("## 4. By trigger path")[1]
    assert "| - | - | 0 |" in sec and "| - | - | - | 0 | 0 | 0 | 0 | 0 | 0 |" in sec


def test_missing_path_shows_none_in_detail():
    text = vfr.readout(_path_fixture(), D24, D25, "live", "abc1234", NOW)
    assert "| none -> event | 6, 7 |" in text


def test_docstring_notes_null_trading_env():
    assert "trading_env is NULL" in vfr.__doc__
