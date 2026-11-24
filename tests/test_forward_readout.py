"""Forward-return readout: session-0 rule, k-session returns, gate split, no prices."""
import importlib.util
import os
from datetime import date, datetime, timezone

_spec = importlib.util.spec_from_file_location(
    "forward_readout", os.path.join(os.path.dirname(__file__), "..", "scripts", "forward_readout.py"))
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

SESSIONS = [date(2026, 11, d) for d in (16, 17, 18, 19, 20, 23, 24, 25, 27, 30)]


def test_session_zero_before_and_after_the_close():
    # 15:59 ET (20:59 UTC) on a session -> that session; 16:00 ET -> the next one
    assert fr.session_zero(datetime(2026, 11, 18, 20, 59, tzinfo=timezone.utc), SESSIONS) == date(2026, 11, 18)
    assert fr.session_zero(datetime(2026, 11, 18, 21, 0, tzinfo=timezone.utc), SESSIONS) == date(2026, 11, 19)
    # Saturday -> Monday; past the last cached bar -> None
    assert fr.session_zero(datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc), SESSIONS) == date(2026, 11, 23)
    assert fr.session_zero(datetime(2026, 12, 1, 15, 0, tzinfo=timezone.utc), SESSIONS) is None
    # naive timestamps read as UTC
    assert fr.session_zero(datetime(2026, 11, 18, 14, 0), SESSIONS) == date(2026, 11, 18)


def test_forward_return_counts_the_names_own_bars_and_pending():
    closes = {d: 100.0 + i for i, d in enumerate(SESSIONS)}
    dates = list(SESSIONS)
    assert abs(fr.forward_return(closes, dates, SESSIONS[0], 5) - 0.05) < 1e-12
    assert fr.forward_return(closes, dates, SESSIONS[6], 5) is None     # 5th bar not cached yet
    assert fr.forward_return(closes, dates, date(2026, 11, 21), 1) is None  # not a bar of the name


def test_gate_label_splits_by_which_gate_fired():
    both = {"entry_gate": {"trend": {"blocked": True}, "stopout": {"blocked": True}}}
    trend = {"entry_gate": {"trend": {"blocked": True}, "stopout": {"blocked": False}}}
    assert fr.gate_label("ENTRY_GATE_BLOCKED", both) == "ENTRY_GATE_BLOCKED (trend+stopout)"
    assert fr.gate_label("ENTRY_GATE_BLOCKED", trend) == "ENTRY_GATE_BLOCKED (trend)"
    assert fr.gate_label("ENTRY_GATE_BLOCKED", None) == "ENTRY_GATE_BLOCKED (gate unknown)"
    assert fr.gate_label("OPEN", both) == "OPEN"
    loser = {"entry_gate": {"add_loser": {"blocked": True}}}
    assert fr.gate_label("ENTRY_GATE_BLOCKED", loser) == "ENTRY_GATE_BLOCKED (add_loser)"
    loser_trend = {"entry_gate": {"trend": {"blocked": True}, "add_loser": {"blocked": True}}}
    assert fr.gate_label("ENTRY_GATE_BLOCKED", loser_trend) == "ENTRY_GATE_BLOCKED (trend+add_loser)"


def test_collect_dedups_per_name_per_session_and_subtracts_spy():
    spy = [(d, 100.0) for d in SESSIONS]
    spy[5] = (SESSIONS[5], 102.0)                    # SPY +2% at session 5
    nv = [(d, 50.0) for d in SESSIONS]
    nv[5] = (SESSIONS[5], 55.0)                      # name +10% at session 5
    bars = {"SPY": spy, "NVDA": nv}
    t0 = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)   # same name, same session: dropped
    rows = [("NVDA", "OPEN", t1, None), ("NVDA", "OPEN", t0, None),
            ("NVDA", "ENTRY_GATE_BLOCKED", t0, {"entry_gate": {"trend": {"blocked": True}}})]
    g = fr.collect(rows, bars)
    assert len(g["OPEN"]) == 1 and len(g["ENTRY_GATE_BLOCKED (trend)"]) == 1
    rec = g["OPEN"][0]
    assert abs(rec["f5"] - 0.10) < 1e-12 and abs(rec["x5"] - 0.08) < 1e-12
    assert rec["f10"] is None and rec["x10"] is None


def test_readout_reports_percentages_and_pending_never_prices():
    spy = [(d, 412.37) for d in SESSIONS]
    bars = {"SPY": spy, "MSFT": [(d, 487.65) for d in SESSIONS]}
    rows = [("MSFT", "ADD", datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc), None)]
    text = fr.readout(fr.collect(rows, bars), date(2026, 11, 16), "live", "abc123")
    assert "| ADD | 5 | 1 | 0 | 1 | +0.0 | +0.0 | 0% | +0.0 |" in text
    assert "| ADD | 10 | 1 | 1 | 1 | - | - | - | - |" in text
    assert "412.37" not in text and "487.65" not in text and "not a study result" in text


def test_readout_with_no_rows():
    text = fr.readout({}, date(2026, 11, 23), "live", "h")
    assert "(no rows)" in text


def test_excess_uses_spy_over_the_names_own_dates():
    # The name skips SESSIONS[2]; its 5th bar after SESSIONS[0] is SESSIONS[6].
    name_days = [d for i, d in enumerate(SESSIONS) if i != 2]
    name = [(d, 100.0 * (1.01 ** i)) for i, d in enumerate(name_days)]
    spy = [(d, 100.0 + i) for i, d in enumerate(SESSIONS)]
    rows = [("AAA", "OPEN", datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc), {})]
    rec = fr.collect(rows, {"AAA": name, "SPY": spy})["OPEN"][0]
    raw = name[5][1] / name[0][1] - 1
    spy_same_dates = spy[6][1] / spy[0][1] - 1
    assert abs(rec["f5"] - raw) < 1e-12
    assert abs(rec["x5"] - (raw - spy_same_dates)) < 1e-12


# --------------------------------------------------------------------------
# --by conviction
# --------------------------------------------------------------------------

def test_conviction_of_prefers_final_conviction_then_confidence():
    assert fr.conviction_of({"final_conviction": 8, "confidence": 0.7}) == 8
    assert fr.conviction_of({"final_conviction": "7", "confidence": 0.8}) == 7
    assert fr.conviction_of({"final_conviction": None, "confidence": 0.6}) == 6
    for bad in ({"final_conviction": 0}, {"confidence": 0.0}, {"final_conviction": True},
                {"confidence": float("nan")}, {}, None, "junk", {"final_conviction": 11}):
        assert fr.conviction_of(bad) is None


def test_logprob_bands_are_fixed_and_lower_bound_inclusive():
    assert fr.LOGPROB_BANDS == ((None, 6.5, "< 6.5"), (6.5, 7.5, "6.5 to < 7.5"), (7.5, None, ">= 7.5"))
    assert fr.logprob_band(6.4999) == "< 6.5"
    assert fr.logprob_band(6.5) == "6.5 to < 7.5"
    assert fr.logprob_band(7.4999) == "6.5 to < 7.5"
    assert fr.logprob_band(7.5) == ">= 7.5"
    assert fr.logprob_band(None) == "no logprobs"
    assert fr.logprob_expected_of({"logprobs_expected": 6.9}) == 6.9
    assert fr.logprob_expected_of({"logprobs_expected": 12}) is None
    assert fr.logprob_expected_of({"logprobs_expected": "x"}) is None


def test_spearman_matches_known_values_with_ties():
    assert abs(fr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(fr.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    # ties: ranks x = [1.5, 1.5, 3, 4], y = [1, 2, 3, 4] -> rho = 4.5 / sqrt(4.5 * 5)
    assert abs(fr.spearman([7, 7, 8, 9], [1, 2, 3, 4]) - 4.5 / (4.5 * 5) ** 0.5) < 1e-12
    assert fr.spearman([7, 7, 7], [1, 2, 3]) is None


def _conv_rows_and_bars(n_days=12):
    """Per day: one c8 name that rises, two c7 names flat; SPY flat. 25+ sessions."""
    import datetime as _dt
    sessions = []
    d = date(2026, 9, 1)
    while len(sessions) < n_days + 25:
        if d.weekday() < 5:
            sessions.append(d)
        d += _dt.timedelta(days=1)
    bars = {"SPY": [(s, 100.0) for s in sessions],
            "UP": [(s, 100.0 * 1.01 ** i) for i, s in enumerate(sessions)],
            "FLAT1": [(s, 50.0) for s in sessions], "FLAT2": [(s, 70.0) for s in sessions]}
    rows = []
    for s in sessions[:n_days]:
        ts = _dt.datetime(s.year, s.month, s.day, 15, 0, tzinfo=timezone.utc)
        rows.append(("UP", "OPEN", ts, {"final_conviction": 8, "confidence": 0.8, "logprobs_expected": 7.8}))
        rows.append(("FLAT1", "ADD", ts, {"final_conviction": None, "confidence": 0.7,
                                          "logprobs_expected": 6.9}))
        rows.append(("FLAT2", "OPEN", ts, {"final_conviction": 7, "confidence": 0.7,
                                           "logprobs_expected": None}))
        # a second verdict on UP the same session: dropped (first per name per session)
        rows.append(("UP", "ADD", ts.replace(hour=17), {"final_conviction": 6, "confidence": 0.6}))
    return rows, bars


def test_conviction_groups_dedup_and_label_both_views():
    rows, bars = _conv_rows_and_bars()
    by_int, by_band = fr.conviction_groups(rows, bars)
    assert set(by_int) == {"c8", "c7"} and len(by_int["c8"]) == 12 and len(by_int["c7"]) == 24
    assert set(by_band) == {"logprob >= 7.5", "logprob 6.5 to < 7.5", "logprob no logprobs"}
    rec = by_int["c8"][0]
    assert rec["conviction"] == 8 and rec["logprob_expected"] == 7.8
    assert abs(rec["x5"] - (1.01 ** 5 - 1)) < 1e-12


def test_day_block_rho_reports_positive_rank_and_needs_enough_rows():
    rows, bars = _conv_rows_and_bars()
    by_int, _ = fr.conviction_groups(rows, bars)
    recs = [r for v in by_int.values() for r in v]
    out = fr.day_block_rho(recs, "conviction", 5, 5)
    rho = float(out.split()[0])
    assert rho > 0.5 and "to" in out
    assert fr.day_block_rho(recs[:6], "conviction", 5, 5) == "too few"       # under 10 dates
    assert fr.day_block_rho([r for r in recs if r["conviction"] == 7], "conviction", 5, 5) == "too few"
    assert fr.day_block_rho(recs, "conviction", 5, 5) == out                   # seeded: repeatable


def test_conviction_readout_prints_the_fixed_bands_and_no_prices():
    rows, bars = _conv_rows_and_bars()
    text = fr.conviction_readout(rows, bars, date(2026, 9, 1), "live", "abc123")
    assert text.startswith("# Forward-return readout by conviction (live")
    assert "fixed before any outcome was read: < 6.5; 6.5 to < 7.5; >= 7.5" in text
    assert "| c8 | 5 | 12 |" in text and "| logprob >= 7.5 | 5 | 12 |" in text
    assert "## Rank correlation with the excess return" in text
    for price in ("100.0", "50.0", "70.0"):
        assert price not in text
    assert "not a study result" in text


def test_default_mode_is_unchanged_by_the_new_hooks():
    rows = [("AAA", "OPEN", datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc), None)]
    bars = {"SPY": [(d, 100.0) for d in SESSIONS], "AAA": [(d, 50.0) for d in SESSIONS]}
    g = fr.collect(rows, bars)
    assert set(g) == {"OPEN"} and set(g["OPEN"][0]) == {"symbol", "d0", "f5", "x5", "f10", "x10",
                                                       "f20", "x20"}
