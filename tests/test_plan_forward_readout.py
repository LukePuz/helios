"""scripts/plan_forward_readout.py on fixture plans and bars: the horizon
window, which level a plan reached first under each hit rule, the
both/same-bar tie, the unscored cases, and the markdown (insufficient under
n=5, no prices, read-only fetch)."""
import importlib.util
import inspect
import os
import re
from datetime import date, datetime, timedelta, timezone

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "plan_forward_readout.py")
_spec = importlib.util.spec_from_file_location("plan_forward_readout", _PATH)
pfr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfr)

NY = pfr.NY


def _days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _days(date(2026, 11, 2), 30)
NOW = datetime(2026, 12, 20, 12, tzinfo=timezone.utc)


def _et(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=NY)


def _bar(d, close, hi=None, lo=None):
    return {"date": d, "high": hi or close * 1.01, "low": lo or close * 0.99, "close": close}


def _plan(i=1, ticker="NVDA", created=None, horizon=None, **kw):
    p = {"id": i, "ticker": ticker, "status": "active", "source": "analyst",
         "created_at": created or _et(DAYS[0], 11), "closed_at": None, "close_reason": None,
         "entry_price": 100.0, "invalidation_price": 90.0, "target_price": 120.0,
         "horizon_end": horizon or DAYS[10], "last_relation": None, "revision": 0,
         "structural_stop": 88.0, "t1": 110.0, "levels_source": "ladder"}
    p.update(kw)
    return p


# --- horizon_sessions ---------------------------------------------------------

def test_window_skips_creation_day_and_stops_at_horizon_end():
    bars = [_bar(d, 100.0) for d in DAYS[:8]]
    w = pfr.horizon_sessions(bars, _et(DAYS[0], 11), DAYS[4])
    assert [b["date"] for b in w] == DAYS[1:5]


def test_window_includes_creation_day_before_the_open():
    bars = [_bar(d, 100.0) for d in DAYS[:3]]
    w = pfr.horizon_sessions(bars, _et(DAYS[0], 8), DAYS[2])
    assert [b["date"] for b in w] == DAYS[0:3]


# --- score_pair ----------------------------------------------------------------

def _score(sessions, rule="intraday", down=90.0, up=120.0, entry=100.0, end=DAYS[10], last=DAYS[20]):
    return pfr.score_pair(sessions, entry, down, up, rule, end, last)


def test_down_first():
    s = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 95.0, lo=89.5), _bar(DAYS[3], 125.0)]
    r = _score(s)
    assert r["outcome"] == "down first"
    assert r["index"] == 1
    assert r["date"] == DAYS[2]


def test_up_first():
    s = [_bar(DAYS[1], 110.0, hi=121.0), _bar(DAYS[2], 85.0)]
    assert _score(s)["outcome"] == "up first"


def test_both_in_one_bar_is_never_ordered():
    s = [_bar(DAYS[1], 100.0, hi=121.0, lo=89.0)]
    r = _score(s)
    assert r["outcome"] == "both/same bar"
    assert r["index"] == 0


def test_close_rule_ignores_wicks():
    s = [_bar(DAYS[1], 100.0, hi=121.0, lo=89.0), _bar(DAYS[2], 119.0), _bar(DAYS[3], 121.0)]
    assert _score(s, rule="intraday")["outcome"] == "both/same bar"
    r = _score(s, rule="close")
    assert r["outcome"] == "up first"
    assert r["index"] == 2


def test_level_exactly_touched_counts():
    assert _score([_bar(DAYS[1], 95.0, lo=90.0)])["outcome"] == "down first"
    assert _score([_bar(DAYS[1], 115.0, hi=120.0)])["outcome"] == "up first"


def test_neither_horizon_over_vs_open():
    s = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 101.0)]
    assert _score(s, end=DAYS[2], last=DAYS[2])["outcome"] == "neither, horizon over"
    assert _score(s, end=DAYS[10], last=DAYS[2])["outcome"] == "neither yet, horizon open"


def test_no_sessions_yet_vs_no_full_session_in_horizon():
    assert _score([], end=DAYS[10], last=DAYS[0])["outcome"] == "no sessions yet"
    assert _score([], end=DAYS[0], last=DAYS[3])["outcome"] == "no full session in horizon"


def test_level_past_entry_or_missing_is_not_scored():
    s = [_bar(DAYS[1], 100.0, lo=50.0)]
    assert _score(s, down=101.0)["outcome"] == "level past entry at creation"
    assert _score(s, up=99.0)["outcome"] == "level past entry at creation"
    assert _score(s, up=None)["outcome"] == "level missing"
    assert _score(s, entry=None)["outcome"] == "level missing"


# --- analyse_plan ----------------------------------------------------------------

def test_analyse_plan_scores_both_pairs_and_percentages():
    bars = [_bar(DAYS[0], 100.0, hi=130.0, lo=70.0),        # creation day: ignored
            _bar(DAYS[1], 105.0, hi=111.0),                 # compiled t1 first
            _bar(DAYS[2], 95.0, lo=89.0)]                   # analyst invalidation first
    r = pfr.analyse_plan(_plan(), bars)
    assert r["sessions"] == 2
    assert r["pct"]["invalidation_price"] == -10.0
    assert r["pct"]["t1"] == 10.0
    assert r["pairs"]["analyst"]["intraday"]["outcome"] == "down first"
    assert r["pairs"]["compiled"]["intraday"]["outcome"] == "up first"
    assert r["pairs"]["compiled"]["intraday"]["index"] == 0


def test_window_runs_past_supersession_and_in_force_flag():
    bars = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 100.0), _bar(DAYS[3], 95.0, lo=89.0)]
    later = pfr.analyse_plan(_plan(status="superseded", closed_at=_et(DAYS[1], 12)), bars)
    hit = later["pairs"]["analyst"]["intraday"]
    assert hit["outcome"] == "down first"
    assert hit["in_force"] is False
    same_day = pfr.analyse_plan(_plan(status="superseded", closed_at=_et(DAYS[3], 12)), bars)
    assert same_day["pairs"]["analyst"]["intraday"]["in_force"] is True
    open_plan = pfr.analyse_plan(_plan(), bars)
    assert open_plan["pairs"]["analyst"]["intraday"]["in_force"] is True


def test_in_force_none_without_a_hit():
    r = pfr.analyse_plan(_plan(horizon=DAYS[2]), [_bar(DAYS[1], 100.0), _bar(DAYS[2], 100.0)])
    assert r["pairs"]["analyst"]["intraday"]["outcome"] == "neither, horizon over"
    assert r["pairs"]["analyst"]["intraday"]["in_force"] is None


def test_plan_without_bars():
    r = pfr.analyse_plan(_plan(), [])
    assert r["sessions"] == 0
    assert r["pairs"]["analyst"]["intraday"]["outcome"] == "no sessions yet"


# --- describe / count ----------------------------------------------------------------

def test_describe_and_count_insufficient_under_five():
    assert pfr.describe([1.0, 2.0, 3.0, 4.0]) == "insufficient (n=4)"
    assert pfr.count(1, 4) == "1 of 4 (insufficient)"
    assert pfr.count(2, 5) == "2 of 5 (40%)"
    assert pfr.describe([1.0, 2.0, 3.0, 4.0, 5.0]).startswith("n=5, min +1.00")


# --- readout ----------------------------------------------------------------------

def _fixture(n, ticker_of=lambda i: "NVDA"):
    plans, bars = [], {}
    for i in range(n):
        t = ticker_of(i)
        plans.append(_plan(i + 1, ticker=t, entry_price=100.0, invalidation_price=91.37,
                           target_price=118.25))
        bars[t] = [_bar(DAYS[1], 100.0), _bar(DAYS[2], 95.0, lo=90.5), _bar(DAYS[3], 119.0)]
    return plans, bars


def test_readout_counts_and_insufficient():
    plans, bars = _fixture(3)
    text = pfr.build(plans, bars, date(2026, 11, 1), "live", "abc1234", NOW)
    assert "Plans: 3 on 1 names; 3 with compiled levels." in text
    assert "| down first | 3 of 3 (insufficient) |" in text
    assert "not independent" in text
    assert '"both/same bar", never ordered' in text
    assert "price_history daily bars (latest " + str(DAYS[3]) in text


def test_readout_percentages_with_five_and_per_name_table():
    plans, bars = _fixture(6, ticker_of=lambda i: "NVDA" if i < 4 else "AMD")
    text = pfr.build(plans, bars, date(2026, 11, 1), "live", "abc1234", NOW)
    assert "| down first | 6 of 6 (100%) |" in text
    assert "| NVDA | 4 | 4 | 0 | 0 | 0 | 0 |" in text
    assert "| AMD | 2 | 2 | 0 | 0 | 0 | 0 |" in text
    assert "Sessions to down hit (0 = first full session): n=6, min +1.00" in text


def test_plan_without_compiled_levels_shows_dash_and_is_left_out_of_compiled():
    plans, bars = _fixture(1)
    plans.append(_plan(9, levels_source=None, structural_stop=None, t1=None))
    text = pfr.build(plans, bars, date(2026, 11, 1), "live", "abc1234", NOW)
    assert "Plans: 2 on 1 names; 1 with compiled levels." in text
    row = next(l for l in text.splitlines() if l.startswith("| 9 |"))
    assert row.endswith("| - | - | - |")


def test_readout_has_no_prices_or_dollars():
    plans, bars = _fixture(6)
    text = pfr.build(plans, bars, date(2026, 11, 1), "live", "abc1234", NOW)
    assert "$" not in text
    for raw in ("91.37", "118.25", "90.5", "119.0", "100.0"):
        assert raw not in text
    assert not re.search(r"shares|account", text, re.I)


def test_fetch_is_read_only():
    src = inspect.getsource(pfr._fetch)
    assert "SET TRANSACTION READ ONLY" in src
    assert "rollback()" in src
    for word in ("INSERT", "UPDATE", "DELETE", "commit("):
        assert word not in src


# --- level-distance bins ------------------------------------------------------------

def test_bin_label_edges_low_inclusive_high_exclusive():
    assert pfr._bin_label(0.5, pfr.RATIO_BINS) == "<1"
    assert pfr._bin_label(1.0, pfr.RATIO_BINS) == "1-2"
    assert pfr._bin_label(2.999, pfr.RATIO_BINS) == "2-3"
    assert pfr._bin_label(3.0, pfr.RATIO_BINS) == ">=3"
    assert pfr._bin_label(None, pfr.RATIO_BINS) is None
    assert pfr._bin_label(5.0, pfr.DOWN_BINS) == "5-10%"
    assert pfr._bin_label(50.0, pfr.UP_BINS) == ">=50%"


def test_distances_and_ratio():
    r = {"pct": {"invalidation_price": -10.0, "target_price": 25.0}}
    d = pfr._distances(r, "invalidation_price", "target_price")
    assert d == {"down": 10.0, "up": 25.0, "ratio": 2.5}
    bad = {"pct": {"invalidation_price": None, "target_price": 25.0}}
    assert pfr._distances(bad, "invalidation_price", "target_price")["ratio"] is None
    wrong_side = {"pct": {"invalidation_price": 3.0, "target_price": -2.0}}
    w = pfr._distances(wrong_side, "invalidation_price", "target_price")
    assert w == {"down": None, "up": None, "ratio": None}


def _rec(down_pct, up_pct, outcome):
    return {"pct": {"invalidation_price": down_pct, "target_price": up_pct},
            "pairs": {"analyst": {"intraday": {"outcome": outcome}}}}


def test_bin_table_counts_scored_only_and_marks_small_bins():
    recs = ([_rec(-10.0, 35.0, "down first")] * 5             # ratio 3.5
            + [_rec(-10.0, 15.0, "up first")] * 2             # ratio 1.5
            + [_rec(-10.0, 15.0, "neither yet, horizon open")])   # not scored
    lines = pfr._bin_table(recs, "analyst", "invalidation_price", "target_price",
                           "intraday", "ratio", pfr.RATIO_BINS)
    by = {l.split(" | ")[0].lstrip("| "): l for l in lines[2:]}
    assert by[">=3"] == "| >=3 | 5 | 5 of 5 (100%) | 0 of 5 (0%) | 0 of 5 (0%) | 0 of 5 (0%) |"
    assert by["1-2"].startswith("| 1-2 | 2 | 0 of 2 (insufficient) | 2 of 2 (insufficient)")
    assert by["<1"].startswith("| <1 | 0 |")


def test_bin_table_by_each_level_distance():
    recs = [_rec(-4.0, 60.0, "down first"), _rec(-25.0, 8.0, "up first")]
    down = pfr._bin_table(recs, "analyst", "invalidation_price", "target_price",
                          "intraday", "down", pfr.DOWN_BINS)
    up = pfr._bin_table(recs, "analyst", "invalidation_price", "target_price",
                        "intraday", "up", pfr.UP_BINS)
    assert "| <5% | 1 | 1 of 1 (insufficient)" in "\n".join(down)
    assert "| >=20% | 1 | 0 of 1 (insufficient) | 1 of 1 (insufficient)" in "\n".join(down)
    assert "| >=50% | 1 | 1 of 1 (insufficient)" in "\n".join(up)
    assert "| <10% | 1 | 0 of 1 (insufficient) | 1 of 1 (insufficient)" in "\n".join(up)


def test_readout_has_distance_section_for_both_pairs_and_rules():
    plans, bars = _fixture(6)
    text = pfr.build(plans, bars, date(2026, 11, 1), "live", "abc1234", NOW)
    assert "## Outcomes by level distance (fixed display bins)" in text
    assert "not thresholds" in text
    for pair in ("analyst", "compiled"):
        for rule in ("intraday", "close"):
            assert f"### {pair}, {rule} rule: by up / down distance ratio" in text
    assert "### analyst, intraday rule: by invalidation_price distance below entry" in text
    assert "### compiled, close rule: by t1 distance above entry" in text
    # fixture: inval -8.63%, target +18.25% -> ratio about 2.1; all 6 down first (intraday)
    assert "| 2-3 | 6 | 6 of 6 (100%) | 0 of 6 (0%) | 0 of 6 (0%) | 0 of 6 (0%) |" in text
    # compiled: stop -12%, t1 +10% -> ratio 0.83, all 6 up first
    assert "| <1 | 6 | 0 of 6 (0%) | 6 of 6 (100%) | 0 of 6 (0%) | 0 of 6 (0%) |" in text
