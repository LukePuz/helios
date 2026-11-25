"""scripts/scan_score_readout.py on fixture scanner rows and funnel rows:
scan grouping (day, re-run gap), quantiles, the fixed-bin histogram, the
threshold read from the row or the funnel (never computed), the floor
check, the near-cut counts, per-tier counts vs the funnel, the components,
and the read-only fetch."""
import importlib.util
import inspect
import os
import re
from datetime import date, datetime, timezone

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "scan_score_readout.py")
_spec = importlib.util.spec_from_file_location("scan_score_readout", _PATH)
ssr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ssr)

NY = ssr.NY
NOW = datetime(2026, 11, 25, 16, tzinfo=timezone.utc)
D24, D25 = date(2026, 11, 24), date(2026, 11, 25)


def _et(d, h, m=0, s=0):
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=NY)


def _row(i, at, ticker, score, thr=80, regime="Risk-Off", comps=(0, 0, 0, None), priority=None):
    tech, mom, cat, thesis = comps
    if priority is None:
        priority = "PASS" if score < 25 else ("LOW" if score < (thr or 80) else "MEDIUM")
    return {"id": i, "timestamp": at.astimezone(timezone.utc), "ticker": ticker,
            "composite_score": score, "technical_score": tech, "momentum_score": mom,
            "catalyst_score": cat, "thesis_alignment_score": score if thesis is None else thesis,
            "priority": priority, "regime_at_scan": regime,
            "regime_detail": {"threshold": thr} if thr is not None else None}


TIERS = {"A": "Core", "B": "Core", "C": "Watch", "D": "Watch", "E": "Speculative"}


def _scan(day, scores, thr=80, start_id=1, h=9, m=35):
    return [_row(start_id + i, _et(day, h, m, i), t, s, thr=thr)
            for i, (t, s) in enumerate(zip("ABCDE", scores))]


def test_scans_split_by_day_and_rerun_gap():
    rows = (_scan(D24, [10, 20, 30, 40, 50]) + _scan(D25, [10, 20, 30, 40, 50], start_id=10)
            + _scan(D25, [11, 21, 31, 41, 51], start_id=20, h=10, m=10))
    sc = ssr.scans(rows)
    assert [s["label"] for s in sc] == ["2026-11-24", "2026-11-25 #1", "2026-11-25 #2"]
    assert [len(s["rows"]) for s in sc] == [5, 5, 5]


def test_rows_within_the_gap_stay_one_scan():
    rows = _scan(D25, [10, 20, 30]) + [_row(9, _et(D25, 9, 49), "D", 40)]
    assert len(ssr.scans(rows)) == 1


def test_quantiles_inclusive():
    q = ssr.quantiles([10, 20, 30, 40, 50])
    assert (q["min"], q["p25"], q["p50"], q["p75"], q["max"], q["n"]) == (10, 20, 30, 40, 50, 5)
    assert q["p10"] == 14 and q["p90"] == 46
    assert ssr.quantiles([]) is None
    assert ssr.quantiles([7])["p50"] == 7


def test_histogram_fixed_bins():
    h = ssr.histogram([0, 9, 10, 19, 25, 89, 90, 100])
    assert h == [2, 2, 1, 0, 0, 0, 0, 0, 1, 2]
    assert len(ssr.BINS) == 10 and ssr.BINS[-1] == (90, 100)


def test_threshold_from_row_and_funnel():
    sc = ssr.scans(_scan(D25, [10, 20, 30, 40, 50], thr=80))[0]
    assert ssr.thresholds(sc, {})["value"] == 80
    assert ssr.thresholds(sc, {})["source"] == "row"
    assert ssr.thresholds(sc, {D25: {"threshold": 80}})["source"] == "row = funnel"
    th = ssr.thresholds(sc, {D25: {"threshold": 65}})
    assert (th["value"], th["source"]) == (80, "row (funnel differs)")


def test_threshold_from_funnel_only_or_not_recorded():
    sc = ssr.scans(_scan(D25, [10, 20, 30, 40, 50], thr=None))[0]
    assert ssr.thresholds(sc, {D25: {"threshold": 65}})["value"] == 65
    th = ssr.thresholds(sc, {})
    assert (th["value"], th["source"]) == (None, "not recorded")


def test_threshold_mixed_rows():
    rows = _scan(D25, [10, 20, 30, 40, 50], thr=80)
    rows[0]["regime_detail"] = {"threshold": 65}
    th = ssr.thresholds(ssr.scans(rows)[0], {})
    assert th["value"] is None and th["source"] == "rows mixed"


def test_floor_check():
    rows = _scan(D25, [10, 24, 25, 40, 50])
    assert ssr.floor_check(rows) == "ok"
    rows[2]["priority"] = "PASS"
    assert ssr.floor_check(rows) == "1 rows disagree"


def test_near_and_cut_counts():
    scores = [20, 23, 24, 25, 70, 75, 78, 79, 80]
    assert ssr.near(scores, 25, 2) == 2          # 23, 24
    assert ssr.near(scores, 25, 5) == 3          # 20, 23, 24
    assert ssr.near(scores, 80, 2) == 2          # 78, 79
    assert ssr.near(scores, 80, 10) == 4         # 70, 75, 78, 79
    assert ssr.near(scores, None, 5) is None
    cc = ssr.cut_counts(scores, 80)
    assert cc == {"below_floor": 3, "below_thr": 8, "floor_to_thr": 5, "at_or_above": 1}


def _fixture():
    rows = _scan(D24, [15, 30, 22, 78, 10], thr=80) + _scan(D25, [24, 33, 18, 65, 12], thr=80,
                                                            start_id=10)
    funnel = {D25: {"threshold": 80, "by_tier": {"Core": {"watchlist_in": 2},
                                                 "Watch": {"watchlist_in": 2},
                                                 "Speculative": {"watchlist_in": 2}}}}
    return rows, funnel


def test_readout_per_scan_row():
    rows, funnel = _fixture()
    text = ssr.readout(rows, funnel, TIERS, D24, D25, "abc1234", NOW)
    assert ("| 2026-11-25 | 09:35 | Risk-Off | 80 (row = funnel) | ok | 5 | 12 | 14.4 | 18 "
            "| 24 | 33 | 52.2 | 65 | 3 | 2 | 0 | LOW 2, PASS 3 |") in text
    assert "| 2026-11-24 | 09:35 | Risk-Off | 80 (row) | ok | 5 |" in text


def test_readout_histogram_and_near_rows():
    rows, funnel = _fixture()
    text = ssr.readout(rows, funnel, TIERS, D24, D25, "abc1234", NOW)
    assert "| 2026-11-24 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |" in text
    # floor -2/-5/-10, threshold -2/-5/-10, max, points from max
    assert "| 2026-11-24 | 80 | 0 | 1 | 2 | 1 | 1 | 1 | 78 | 2 |" in text
    assert "| 2026-11-25 | 80 | 1 | 1 | 2 | 0 | 0 | 0 | 65 | 15 |" in text


def test_readout_per_tier_with_funnel_check():
    rows, funnel = _fixture()
    text = ssr.readout(rows, funnel, TIERS, D24, D25, "abc1234", NOW)
    sec = text.split("## 4. Per tier")[1].split("## 5.")[0]
    assert "| 2026-11-25 | Core | 2 | 2 |" in sec
    assert "| 2026-11-25 | Speculative | 2 (differs) | 1 |" in sec
    assert "| 2026-11-24 | Core | - | 2 |" in sec


def test_unknown_tier():
    rows, funnel = _fixture()
    text = ssr.readout(rows, funnel, {"A": "Core"}, D24, D25, "abc1234", NOW)
    assert "| 2026-11-24 | unknown | - | 4 |" in text


def test_readout_components():
    rows = [_row(1, _et(D25, 9, 35), "A", 40, comps=(5, 10, 0, 25)),
            _row(2, _et(D25, 9, 36), "B", 20, comps=(0, 0, 10, 10))]
    text = ssr.readout(rows, {}, TIERS, D25, D25, "abc1234", NOW)
    assert "| 2026-11-25 | technical | 2 | 0 | 0.5 | 1.2 | 2.5 | 3.8 | 4.5 | 5 | 2.5 |" in text
    assert "| 2026-11-25 | thesis_alignment | 2 | 10 |" in text


def test_readout_custom_within_and_empty():
    rows, funnel = _fixture()
    text = ssr.readout(rows, funnel, TIERS, D24, D25, "abc1234", NOW, within=(3,))
    assert "| floor -3 | threshold -3 |" in text
    empty = ssr.readout([], {}, TIERS, D24, D25, "abc1234", NOW)
    assert "## 1. Per scan" in empty and "## 5. Components" in empty


def test_readout_has_no_tickers_prices_or_proposals():
    rows, funnel = _fixture()
    text = ssr.readout(rows, funnel, {"NVDA": "Core"}, D24, D25, "abc1234", NOW)
    assert "$" not in text
    for t in ("| A |", "| B |", "NVDA"):
        assert t not in text
    assert not re.search(r"shares|account|recommend|should|suggest", text, re.I)


def test_fetch_is_read_only():
    src = inspect.getsource(ssr._fetch)
    assert "SET TRANSACTION READ ONLY" in src
    assert "rollback()" in src
    for word in ("INSERT", "UPDATE", "DELETE", "commit("):
        assert word not in src
