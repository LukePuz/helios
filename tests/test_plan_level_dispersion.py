"""plan observations carry the model's 24 h level spread, log-only (audit
2026-11-08 19:19 #2)."""
from unittest.mock import patch

from agents import plans


def _rows(*rets):
    return [{"expected_return_pct": r, "expected_downside_pct": -10, "catalyst_date": d}
            for r, d in rets]


def test_spread_and_noise_flag():
    parsed = {"expected_return_pct": 80.6, "expected_downside_pct": -15.0,
              "catalyst": {"date": "2026-11-17"}}
    with patch("db.get_recent_verdict_levels", return_value=_rows((20.0, "2026-11-17"), (40, "2026-11"))):
        d = plans._level_dispersion("NVDA", parsed)
    assert d["n"] == 3 and d["n_return"] == 3 and d["catalyst_dates_distinct"] == 2
    assert d["return_range_pp"] == 60.6 and d["return_median"] == 40.0
    assert d["return_pairwise_median_pp"] == 40.6
    assert d["target_noise_frac"] == round(40.6 / 140, 4) and d["revise_within_noise"] is True
    assert d["downside_range_pp"] == 5.0 and d["status"] == "ok"


def test_tight_spread_is_not_noise():
    parsed = {"expected_return_pct": 20.0}
    with patch("db.get_recent_verdict_levels", return_value=_rows((21.0, None), (22.0, None))):
        d = plans._level_dispersion("MSFT", parsed)
    assert d["revise_within_noise"] is False


def test_single_verdict_and_read_error():
    with patch("db.get_recent_verdict_levels", side_effect=RuntimeError("db")):
        d = plans._level_dispersion("X", {"expected_return_pct": 10})
    assert d["status"] == "read_error:RuntimeError" and d["n"] == 1
    assert d["revise_within_noise"] is None and "return_range_pp" not in d


def test_unparseable_values_are_skipped():
    with patch("db.get_recent_verdict_levels",
               return_value=[{"expected_return_pct": "n/a"}, {"expected_return_pct": float("nan")}]):
        d = plans._level_dispersion("X", {"expected_return_pct": 10})
    assert d["n"] == 3 and d["n_return"] == 1


def test_observe_adds_it_to_every_result():
    with patch.object(plans, "_observe", return_value={"relation": "NO_PLAN"}), \
         patch.object(plans, "_level_dispersion", return_value={"n": 1}):
        assert plans.observe("X", {}, "HOLD", 5, "")["level_dispersion"] == {"n": 1}
    with patch.object(plans, "_observe", return_value={"relation": "ERROR"}), \
         patch.object(plans, "_level_dispersion", side_effect=ValueError()):
        assert plans.observe("X", {}, "HOLD", 5, "")["level_dispersion"] == {"status": "error:ValueError"}
