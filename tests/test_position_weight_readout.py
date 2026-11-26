"""scripts/position_weight_readout.py: each held name's weight per NYSE session
from the stored portfolio snapshots, and the sessions above 15% and above the
bucket's tier-resolved at-entry cap. Record only; percentages and counts only.
Synthetic snapshots: no DB."""
import importlib.util
import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import config

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "scripts", "position_weight_readout.py")
_spec = importlib.util.spec_from_file_location("position_weight_readout_t", _P)
pwr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pwr)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
MON, TUE, WED = date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18)
PV = 100_000.0          # a standard-tier book: caps 12 / 7 / 2
ROLES = {"NVDA": "Compounder", "AMD": "Accelerator", "RGTI": "Moonshot"}


def _et(d, hh, mm=0):
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY).astimezone(UTC)


def _pos(**w):
    """positions_json with each name at w percent of PV (qty 1)."""
    return {t: {"qty": 1, "market_value": PV * x / 100, "avg_entry_price": 1.0}
            for t, x in w.items()}


def _snap(d, hh, mm=0, pv=PV, **w):
    return (_et(d, hh, mm), pv, _pos(**w))


# ── weights and caps ─────────────────────────────────────────────────────────

def test_weights_are_long_priced_names_over_total_value():
    pos = {**_pos(NVDA=14, AMD=6), "TSLA": {"qty": -2, "market_value": -500.0},
           "MU": {"qty": 3, "market_value": None}, "OLD": {"qty": 0, "market_value": 0.0},
           "BAD": "x"}
    w, counts = pwr.weights(pos, PV)
    assert w == {"NVDA": pytest.approx(0.14), "AMD": pytest.approx(0.06)}
    assert counts == {"n_short": 1, "n_unpriced": 1}
    assert pwr.weights(pos, 0) == ({}, {"n_short": 0, "n_unpriced": 0})
    assert pwr.weights(pos, None)[0] == {}
    assert pwr.weights(None, PV)[0] == {}


def test_weights_match_the_book_deployment_rule():
    import tasks
    pos = {**_pos(NVDA=17.25, AMD=6, RGTI=1.5), "TSLA": {"qty": -1, "market_value": -10.0}}
    bd = tasks._book_deployment(pos, PV, 0.0, None)
    w, counts = pwr.weights(pos, PV)
    top = max(w, key=w.get)
    assert (top, round(w[top] * 100, 3)) == (bd["largest_name"], bd["largest_name_weight_pct"])
    assert round(sum(w.values()) * 100, 3) == bd["invested_pct"]
    assert (len(w), counts["n_short"]) == (bd["n_positions"], bd["n_short"])


@pytest.mark.parametrize("pv", [100_000.0, 10_000.0, 1_000.0])
def test_caps_are_the_tier_resolved_at_entry_caps_sizing_reads(pv):
    lim = config.get_active_limits(pv)
    c = pwr.caps(pv)
    assert c == {"tier": lim["tier"], "Compounder": lim["max_position_pct_compounder"],
                 "Accelerator": lim["max_position_pct_accelerator"],
                 "Moonshot": lim["max_position_pct_moonshot"]}
    assert pwr.caps(0) == {"tier": None, "Compounder": None, "Accelerator": None, "Moonshot": None}


# ── which snapshots count toward a session ───────────────────────────────────

def test_only_in_session_snapshots_count():
    snaps = [_snap(MON, 9, 29, NVDA=20), _snap(MON, 9, 30, NVDA=5), _snap(MON, 16, 0, NVDA=6),
             _snap(MON, 16, 1, NVDA=30), _snap(TUE, 3, 0, NVDA=30)]
    by = pwr.in_session(snaps, [MON, TUE])
    assert [r[0] for r in by[MON]] == [_et(MON, 9, 30), _et(MON, 16, 0)]
    assert by[TUE] == []


def test_an_early_close_ends_at_13_00_and_naive_times_are_utc():
    fri = date(2026, 11, 27)                        # NYSE early close
    naive = datetime(2026, 11, 27, 17, 30)           # 12:30 ET as naive UTC
    snaps = [(naive, PV, _pos(NVDA=5)), _snap(fri, 13, 0, NVDA=6), _snap(fri, 13, 5, NVDA=40)]
    rows = pwr.in_session(snaps, [fri])[fri]
    assert len(rows) == 2 and rows[-1][0] == _et(fri, 13, 0)


# ── per-session and per-name counts ──────────────────────────────────────────

def _tally(snaps, sessions=(MON, TUE, WED), roles=ROLES):
    return pwr.tally(list(sessions), snaps, roles)


def test_close_and_any_snapshot_readings_are_kept_apart():
    snaps = [_snap(MON, 10, NVDA=16.0, AMD=7.5), _snap(MON, 15, 55, NVDA=14.0, AMD=6.0),
             _snap(TUE, 15, 55, NVDA=15.5, AMD=8.0, RGTI=1.0)]
    t = _tally(snaps)
    mon, tue, wed = (t["sessions"][d] for d in (MON, TUE, WED))
    assert mon["n_snapshots"] == 2 and mon["over15_close"] == {}
    assert mon["over15_any"] == {"NVDA": pytest.approx(0.16)}
    assert mon["overcap_close"] == {"NVDA": pytest.approx(0.14)}           # 14% > 12%
    assert set(mon["overcap_any"]) == {"NVDA", "AMD"}
    assert mon["overcap_any"]["NVDA"] == pytest.approx(0.16)               # the session's max
    assert set(tue["over15_close"]) == {"NVDA"} and set(tue["overcap_close"]) == {"NVDA", "AMD"}
    assert wed == {"n_snapshots": 0}
    n = t["names"]
    assert n["NVDA"] == {"bucket": "Compounder", "sessions_held": 2,
                         "max_close_pct": pytest.approx(15.5),
                         "s_over15_close": 1, "s_over15_any": 2,
                         "s_overcap_close": 2, "s_overcap_any": 2}
    assert n["AMD"]["s_overcap_close"] == 1 and n["AMD"]["s_overcap_any"] == 2
    assert n["RGTI"]["s_overcap_any"] == 0 and n["RGTI"]["sessions_held"] == 1


def test_exactly_at_a_threshold_is_not_above_it():
    t = _tally([_snap(MON, 15, NVDA=15.0, AMD=7.0, RGTI=2.0)], sessions=(MON,))
    s = t["sessions"][MON]
    assert s["over15_any"] == {} and s["overcap_any"] == {"NVDA": pytest.approx(0.15)}


def test_a_name_off_the_watchlist_has_no_bucket_and_is_never_above_a_cap():
    t = _tally([_snap(MON, 15, XYZ=30.0)], sessions=(MON,), roles={})
    s = t["sessions"][MON]
    assert set(s["over15_close"]) == {"XYZ"} and s["overcap_close"] == {}
    assert t["names"]["XYZ"]["bucket"] == pwr.NO_BUCKET
    assert pwr.tally([MON], [_snap(MON, 15, XYZ=30.0)], {"XYZ": ""})["names"]["XYZ"]["bucket"] \
        == pwr.NO_BUCKET


def test_the_cap_follows_each_snapshots_tier():
    # 30% Compounder: above the standard 12% cap; under the micro tier's 40%.
    small = config.ACCOUNT_TIER_REDUCED_MIN / 2
    snaps = [(_et(MON, 15), PV, _pos(NVDA=30.0)),
             (_et(TUE, 15), small, {"NVDA": {"qty": 1, "market_value": small * 0.30}})]
    t = _tally(snaps, sessions=(MON, TUE))
    assert set(t["sessions"][MON]["overcap_close"]) == {"NVDA"}
    assert t["sessions"][TUE]["overcap_close"] == {} and t["sessions"][TUE]["tier"] == "micro"
    assert set(t["sessions"][TUE]["over15_close"]) == {"NVDA"}


# ── output ───────────────────────────────────────────────────────────────────

def test_readout_prints_percentages_and_counts_only():
    snaps = [_snap(MON, 10, NVDA=16.0, AMD=7.5), _snap(MON, 15, 55, NVDA=14.0, AMD=6.0),
             (_et(TUE, 15), PV, {**_pos(NVDA=15.5), "TSLA": {"qty": -7, "market_value": -98765.0}})]
    text = pwr.readout([MON, TUE, WED], _tally(snaps), "fedcba0",
                       datetime(2026, 11, 23, 22, 0, tzinfo=UTC))
    assert "| close (last in-session snapshot) | 1 of 2 | 2 of 2 |" in text
    assert "| any in-session snapshot | 2 of 2 | 2 of 2 |" in text
    assert "| 2026-11-16 | 2 | standard | 12/7/2% | 2 | NVDA 14.0% | none | NVDA 14.0% | " \
           "NVDA 16.0% | NVDA 16.0%, AMD 7.5% |" in text
    assert "| 2026-11-17 | 1 | standard | 12/7/2% | 1 (+1 short) | NVDA 15.5% |" in text
    assert "| 2026-11-18 | 0 | no snapshot |" in text
    assert "| NVDA | Compounder | 2 | 15.5% | 1 / 2 | 2 / 2 |" in text
    assert "research choice (Luke)" in text and "enforced nowhere" in text
    assert "$" not in text and "100000" not in text and "98765" not in text
    assert "qty" not in text and "share" not in text.lower().replace("shares", "")


def test_readout_with_no_names():
    text = pwr.readout([WED], _tally([], sessions=(WED,)), "h", datetime(2026, 11, 23, tzinfo=UTC))
    assert "| 0 of 0 | 0 of 0 |" in text and "| none | | | | | |" in text


# ── main ─────────────────────────────────────────────────────────────────────

def test_main_refuses_without_the_pooler_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    with patch.object(pwr, "_fetch") as f:
        assert pwr.main([]) == 2
    f.assert_not_called()
    assert "6543" in capsys.readouterr().err


def test_main_without_snapshots_or_sessions(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    with patch.object(pwr, "_fetch", return_value=(None, [])):
        assert pwr.main(["--until", "2026-11-18"]) == 2
    with patch.object(pwr, "_fetch", return_value=(MON, [])):
        assert pwr.main(["--since", "2026-11-21", "--until", "2026-11-22"]) == 2   # a weekend


def test_main_runs_from_the_first_snapshot(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SUPABASE_PORT", "6543")
    out = tmp_path / "r.md"
    ef = pwr._ef()
    with patch.object(pwr, "_fetch", return_value=(MON, [_snap(TUE, 15, NVDA=16.0)])) as f, \
         patch.object(pwr, "_roles", return_value=ROLES), \
         patch.object(ef, "_git_head", return_value="abc1234"):
        assert pwr.main(["--until", "2026-11-18", "--out", str(out)]) == 0
    f.assert_called_once_with(None, WED)
    text = capsys.readouterr().out
    assert "Sessions 2026-11-16 to 2026-11-18 (3 NYSE sessions, 1 with an in-session snapshot)" in text
    assert "git abc1234" in text and out.read_text() == text


def test_no_db_import_and_no_writes():
    import inspect
    src = inspect.getsource(pwr)
    assert "db" not in pwr.__dict__
    assert "READ ONLY" in src and "rollback()" in src
    for verb in ("INSERT", "UPDATE ", "DELETE", "set_state", "log_activity"):
        assert verb not in src
    assert pwr.ENV == "live" and pwr.READ_PORT == 6543 and pwr.VIA_GAINS_PCT == 15.0
