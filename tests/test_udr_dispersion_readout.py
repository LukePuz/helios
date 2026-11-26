"""scripts/udr_dispersion_readout.py on fixture verdict rows: normalisation,
spreads, crossings of the bar passed in (below vs at or above), unknown
pairs, the model - structural difference, null reasons, the per-ticker
filter, the detail table and the read-only fetch."""
import importlib.util
import inspect
import os
import re
from datetime import date, datetime, timezone

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "udr_dispersion_readout.py")
_spec = importlib.util.spec_from_file_location("udr_dispersion_readout", _PATH)
udr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(udr)

NY = udr.NY
NOW = datetime(2026, 11, 25, 21, tzinfo=timezone.utc)
D24, D25 = date(2026, 11, 24), date(2026, 11, 25)
BAR = 2.0


def _row(i, day, h, m, model, structural, ticker="NVDA", t2=None, signal="HOLD", conv=7,
         record=True, why=None):
    at = datetime(day.year, day.month, day.day, h, m, tzinfo=NY).astimezone(timezone.utc)
    dec = {"final_conviction": conv, "udr_check": {"model": model}}
    if record:
        dec["udr_structural"] = {"udr_structural": structural, "udr_structural_t2": t2}
        if why:
            dec["udr_structural"]["why_null"] = why
    return {"id": i, "symbol": ticker, "timestamp": at, "signal": signal,
            "confidence": conv / 10, "decision": dec}


def _v(*a, **k):
    return udr.verdict(_row(*a, **k))


def test_verdict_normalises_fields():
    v = _v(1, D25, 14, 13, 2.01, 2.29, t2=2.27)
    assert (v["model"], v["structural"], v["structural_t2"], v["conviction"]) == (2.01, 2.29, 2.27, 7)
    assert v["day"] == D25 and v["has_structural_record"]
    old = udr.verdict({"id": 2, "symbol": "NVDA", "timestamp": "2026-11-01T15:00:00+00:00",
                       "signal": "HOLD", "confidence": 0.6, "decision": {"udr_check": {}}})
    assert old["model"] is None and old["structural"] is None and not old["has_structural_record"]
    assert old["conviction"] == 6


def test_verdict_ignores_bool_and_text():
    v = _v(1, D25, 9, 40, True, "n/a")
    assert v["model"] is None and v["structural"] is None


def test_spread():
    s = udr.spread([2.01, 4.24, 1.68, 2.84, None])
    assert (s["n"], s["min"], s["max"]) == (4, 1.68, 4.24)
    assert round(s["median"], 3) == 2.425
    assert round(s["p25"], 4) == 1.9275 and round(s["p75"], 2) == 3.19
    assert round(s["iqr"], 4) == round(3.19 - 1.9275, 4)
    assert udr.spread([None]) is None
    one = udr.spread([3.0])
    assert (one["p25"], one["median"], one["p75"], one["iqr"]) == (3.0, 3.0, 3.0, 0)


def test_crossings_count_side_changes_at_the_bar():
    # the 2026-11-25 NVDA afternoon run: 2.01 / 4.24 / 1.68 / 2.84
    rows = [_v(1, D25, 14, 13, 2.01, 2.29), _v(2, D25, 14, 15, 4.24, 2.27),
            _v(3, D25, 14, 28, 1.68, 2.26), _v(4, D25, 14, 57, 2.84, 2.26)]
    assert udr.crossings(rows, "model", BAR) == {"pairs": 3, "known": 3, "crossed": 2}
    assert udr.crossings(rows, "structural", BAR) == {"pairs": 3, "known": 3, "crossed": 0}
    assert udr.both_crossed(rows, BAR) == 0


def test_equal_to_bar_is_at_or_above():
    rows = [_v(1, D25, 10, 0, 2.0, 2.0), _v(2, D25, 10, 5, 2.5, 1.99)]
    assert udr.crossings(rows, "model", BAR)["crossed"] == 0
    assert udr.crossings(rows, "structural", BAR)["crossed"] == 1


def test_bar_is_a_parameter():
    rows = [_v(1, D25, 10, 0, 2.5, 2.5), _v(2, D25, 10, 5, 3.5, 2.6)]
    assert udr.crossings(rows, "model", BAR)["crossed"] == 0
    assert udr.crossings(rows, "model", 3.0)["crossed"] == 1


def test_unknown_pairs_not_compared():
    rows = [_v(1, D25, 10, 0, 1.5, None), _v(2, D25, 10, 5, 2.5, 2.1), _v(3, D25, 10, 9, None, 1.9)]
    assert udr.crossings(rows, "model", BAR) == {"pairs": 2, "known": 1, "crossed": 1}
    assert udr.crossings(rows, "structural", BAR) == {"pairs": 2, "known": 1, "crossed": 1}
    assert udr.both_crossed(rows, BAR) == 0


def test_both_cross():
    rows = [_v(1, D25, 10, 0, 1.5, 1.8), _v(2, D25, 10, 5, 2.5, 2.1)]
    assert udr.both_crossed(rows, BAR) == 1


def test_group_orders_and_splits_days_and_tickers():
    vs = [_v(3, D25, 11, 0, 2.0, 2.0), _v(1, D25, 10, 0, 1.0, 1.0), _v(2, D24, 15, 0, 3.0, 3.0),
          _v(4, D25, 10, 30, 1.0, 1.0, ticker="AMD")]
    g = udr.group(vs)
    assert sorted(g) == [(D24, "NVDA"), (D25, "AMD"), (D25, "NVDA")]
    assert [v["id"] for v in g[(D25, "NVDA")]] == [1, 3]


def test_pairs_across_days_not_compared():
    vs = [_v(1, D24, 15, 0, 1.0, 1.0), _v(2, D25, 9, 40, 3.0, 3.0)]
    text = udr.readout(vs, D24, D25, "live", "abc1234", NOW, BAR)
    sec = text.split("## 1.")[1].split("## 2.")[0]
    assert "| 2026-11-24 | 1 | 1 |" in sec and "| 0 | 0 of 0 | 0 of 0 | 0 |" in sec


def test_diffs():
    rows = [_v(1, D25, 10, 0, 2.5, 2.0), _v(2, D25, 10, 5, 1.0, 1.5), _v(3, D25, 10, 9, 2.0, None)]
    assert udr.diffs(rows) == [0.5, -0.5]


def _fixture():
    return [_v(1, D25, 14, 13, 2.01, 2.29, t2=2.27), _v(2, D25, 14, 15, 4.24, 2.27, t2=3.14),
            _v(3, D25, 14, 28, 1.68, 2.26, t2=1.9), _v(4, D25, 14, 57, 2.84, 2.26, t2=3.21),
            _v(5, D25, 10, 0, 1.2, None, ticker="AMD", why="no t1"),
            _v(6, D24, 12, 0, 1.5, 1.4, ticker="AMD", record=True),
            _v(7, D24, 12, 30, 2.5, None, ticker="AMD", record=False)]


def test_readout_per_day_and_difference():
    text = udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR)
    sec1 = text.split("## 1.")[1].split("## 2.")[0]
    assert ("| 2026-11-25 | 5 | 2 | 5 | 1.2 | 2.01 | 4.24 | 1.16 | 4 | 2.26 | 2.26 | 2.29 "
            "| 0.02 | 3 | 2 of 3 | 0 of 3 | 0 |") in sec1
    assert "| 2026-11-24 | 2 | 1 | 2 | 1.5 | 2 | 2.5 | 0.5 | 1 | 1.4 | 1.4 | 1.4 | 0 | 1 | 1 of 1 | 0 of 0 | 0 |" in sec1
    sec2 = text.split("## 2.")[1].split("## 3.")[0]
    assert "| 2026-11-25 | 4 | -0.58 | -0.35 | 0.15 | 0.93 | 1.97 | 2 | 2 | 0 |" in sec2


def test_readout_null_reasons_and_t2():
    text = udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR)
    sec3 = text.split("## 3.")[1].split("## 4.")[0]
    assert "| 2026-11-25 | 5 | 0 | no t1 | 1 |" in sec3
    assert "| 2026-11-24 | 2 | 1 | - | 0 |" in sec3
    sec4 = text.split("## 4.")[1].split("## 5.")[0]
    assert "| 2026-11-25 | 4 | 1.9 | 2.71 | 3.21 | 0.98 |" in sec4
    assert sec4.strip().splitlines()[-1].endswith("| 2 of 3 |")


def test_readout_per_ticker_min_calls():
    text = udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR)
    sec5 = text.split("## 5.")[1]
    assert "| 2026-11-25 | NVDA | 4 |" in sec5 and "| 2026-11-24 | AMD | 2 |" in sec5
    assert "| 2026-11-25 | AMD |" not in sec5
    nvda = [ln for ln in sec5.splitlines() if "| 2026-11-25 | NVDA |" in ln][0]
    assert nvda.endswith("| 0.15 | -0.58 | 1.97 | 2 of 3 | 0 of 3 | 0 |")
    one = udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR, min_calls=1)
    assert "| 2026-11-25 | AMD | 1 |" in one.split("## 5.")[1]


def test_readout_detail_marks_crossings():
    text = udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR, detail=True)
    sec6 = text.split("## 6. Each call")[1]
    assert "| 2026-11-25 | NVDA | 14:15:00 | HOLD | 7 | 4.24 | at/above | 2.27 | at/above | 3.14 | 1.97 | 2 |" in sec6
    assert "| 14:28:00 | HOLD | 7 | 1.68 | below x | 2.26 | at/above |" in sec6
    assert "| 14:57:00 | HOLD | 7 | 2.84 | at/above x |" in sec6
    assert "## 6." not in udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR)


def test_readout_states_bar_source_and_empty():
    text = udr.readout([], D24, D25, "live", "abc1234", NOW, 2.0, detail=True)
    assert "config.MIN_UPSIDE_DOWNSIDE_RATIO" in text and "bar 2 " in text
    for s in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert s in text


def test_readout_has_no_prices_or_proposals():
    text = udr.readout(_fixture(), D24, D25, "live", "abc1234", NOW, BAR, detail=True)
    assert "$" not in text
    assert not re.search(r"shares|account|recommend|should|suggest", text, re.I)


def test_fetch_is_read_only():
    src = inspect.getsource(udr._fetch)
    assert "SET TRANSACTION READ ONLY" in src
    assert "rollback()" in src
    for word in ("INSERT", "UPDATE", "DELETE", "commit("):
        assert word not in src


def test_main_reads_bar_from_config():
    src = inspect.getsource(udr.main)
    assert "MIN_UPSIDE_DOWNSIDE_RATIO" in src
