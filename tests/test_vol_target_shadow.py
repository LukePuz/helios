"""Record-only volatility shadows on size_position (vol_target_shadow, stop_vol).

The shadows must never change shares, stop or order fields, never raise into
sizing, and never write a non-finite number into risk_json.
"""
import json
import math
import os
import re
import statistics
from unittest.mock import MagicMock, patch

import pytest

import config
import risk_manager

SHADOW_KEYS = ("vol_target_shadow", "stop_vol")


def _closes(n=25, start=100.0, pattern=(0.02, -0.015, 0.01, -0.005)):
    out, c = [], start
    for i in range(n):
        out.append(c)
        c *= 1 + pattern[i % len(pattern)]
    return out


def _bars(closes):
    return [{"date": f"2026-10-{i + 1:02d}", "close": c} for i, c in enumerate(closes)]


def _client(pv=100_000.0, positions=None):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=pv, cash=pv)
    client.get_all_positions.return_value = positions or []
    return client


def _state(k, d=None):
    return {"trading_halted_paper": "false", "high_water_mark_paper": "100000"}.get(k, d)


def _size(monkeypatch, on: bool, bars=None, stop=(120.0, "mocked"), bucket="Compounder",
          price=130.0, atr=None, bars_raise=None):
    monkeypatch.setattr(config, "VOL_TARGET_SIZING_SHADOW_ENABLED", on)
    if bars_raise is not None:
        def boom(ticker, window):
            raise bars_raise
        monkeypatch.setattr(risk_manager, "_vol_shadow_bars", boom)
    elif bars is not None:
        monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda ticker, window: bars)
    patches = [patch("risk_manager.TRADING_ENV", "paper"),
               patch("db.get_state", side_effect=_state), patch("db.set_state")]
    if stop is not None:
        patches.append(patch.object(risk_manager.RiskManager, "calculate_stop", return_value=stop))
    if atr is not None:
        patches.append(patch.object(risk_manager.RiskManager, "get_atr", return_value=atr))
    for p in patches:
        p.start()
    try:
        rm = risk_manager.RiskManager(_client())
        rm.quiet_fallback_alert = True
        return rm.size_position("NVDA", conviction=8, bucket=bucket, regime="Risk-On", price=price)
    finally:
        for p in reversed(patches):
            p.stop()


def _without_shadow(sizing):
    return {k: v for k, v in sizing.items() if k not in SHADOW_KEYS}


# --------------------------------------------------------------------------
# _realised_vol
# --------------------------------------------------------------------------

def test_realised_vol_matches_sample_stdev_of_log_returns():
    closes = _closes(30)
    last = closes[-21:]
    rets = [math.log(b / a) for a, b in zip(last, last[1:])]
    want = statistics.stdev(rets) * math.sqrt(252)
    assert abs(risk_manager._realised_vol(closes, 20) - want) < 1e-12


@pytest.mark.parametrize("closes", [
    _closes(20),                                    # one short of window + 1
    _closes(24)[:-1] + [0.0],                       # non-positive close
    _closes(24)[:-1] + [float("nan")],              # non-finite close
    _closes(24)[:-1] + [float("inf")],
    ["x"] * 25,                                     # not numbers
])
def test_realised_vol_none_on_unusable_closes(closes):
    assert risk_manager._realised_vol(closes, 20) is None


# --------------------------------------------------------------------------
# _vol_shadow_record
# --------------------------------------------------------------------------

def test_record_values(monkeypatch):
    closes = _closes(25)
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: _bars(closes))
    rec = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, 4.0, 2.5, 10_000.0)
    vt, sv = rec["vol_target_shadow"], rec["stop_vol"]
    vol = risk_manager._realised_vol(closes, 20)
    assert vt["read"] == "ok" and vt["bars"] == 25 and vt["last_bar_date"] == "2026-10-25"
    assert vt["target_contribution"] == config.VOL_TARGET_CONTRIBUTION
    assert vt["order_weight"] == 0.1 and vt["position_weight_after"] == 0.1
    assert vt["rvol20"] == round(vol, 4)
    vw = config.VOL_TARGET_CONTRIBUTION / vol
    assert vt["vol_weight"] == round(vw, 4)
    assert vt["weight_ratio"] == round(vw / 0.1, 3)
    assert vt["vol_weight_shares"] == int(vw * 10_000.0 / 100.0)
    assert vt["would_cut"] is (vt["vol_weight_shares"] < 10)
    assert sv["stop_pct"] == 0.1 and sv["atr"] == 4.0 and sv["atr_multiple"] == 2.5
    assert sv["stop_distance_atr"] == 2.5
    sigma20 = vol * math.sqrt(20 / 252)
    assert sv["sigma20_pct"] == round(sigma20, 4)
    assert sv["stop_sigma20"] == round(0.1 / sigma20, 3)


def test_record_counts_held_value_toward_the_vol_weight(monkeypatch):
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: _bars(_closes(25)))
    fresh = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, 4.0, 2.5, 10_000.0)
    add = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 500.0, 90.0, 4.0, 2.5, 10_000.0)
    f, a = fresh["vol_target_shadow"], add["vol_target_shadow"]
    assert a["position_weight_after"] == 0.15 and a["order_weight"] == 0.1
    vw = config.VOL_TARGET_CONTRIBUTION / risk_manager._realised_vol(_closes(25), 20)
    assert f["vol_weight_shares"] == int(vw * 10_000.0 / 100.0)
    assert a["vol_weight_shares"] == int(max(vw * 10_000.0 - 500.0, 0.0) / 100.0)
    huge_held = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 9_000.0, 90.0, 4.0, 2.5, 10_000.0)
    assert huge_held["vol_target_shadow"]["vol_weight_shares"] == 0
    assert huge_held["vol_target_shadow"]["would_cut"] is True


@pytest.mark.parametrize("bars,read", [([], "no_cached_bars"),
                                       (_bars(_closes(10)), "insufficient_bars"),
                                       (_bars([100.0] * 25), "zero_vol")])
def test_record_without_usable_bars(monkeypatch, bars, read):
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: bars)
    rec = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, 4.0, 2.5, 10_000.0)
    assert rec["vol_target_shadow"]["read"] == read
    assert rec["vol_target_shadow"]["vol_weight"] is None
    assert rec["stop_vol"]["stop_sigma20"] is None
    assert rec["stop_vol"]["stop_pct"] == 0.1        # what needs no bars is still recorded


def test_record_never_raises(monkeypatch):
    def boom(t, w):
        raise RuntimeError("db down")
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", boom)
    rec = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, 4.0, 2.5, 10_000.0)
    assert rec["vol_target_shadow"]["read"] == "error: RuntimeError"
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: _bars(_closes(25)))
    for args in [("X", None, None, None, None, None, None, None),
                 ("X", "abc", 1, 0, 1, 1, 1, 1),
                 ("X", 100.0, 10, 0.0, 150.0, True, True, 0.0),        # stop above price, bools
                 ("X", 100.0, 10, 0.0, 90.0, float("nan"), float("inf"), 10_000.0),
                 ("X", float("inf"), 10, 0.0, 90.0, 4.0, 2.5, 10_000.0)]:
        rec = risk_manager._vol_shadow_record(*args)
        json.dumps(rec, allow_nan=False)                           # finite or None only


def test_record_is_json_safe_with_bad_bars(monkeypatch):
    bars = _bars(_closes(24)) + [{"date": "2026-10-25", "close": float("inf")}]
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: bars)
    rec = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, 4.0, 2.5, 10_000.0)
    json.dumps(rec, allow_nan=False)
    assert rec["vol_target_shadow"]["read"] == "insufficient_bars"


def test_record_holds_no_dollar_values(monkeypatch):
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: _bars(_closes(25)))
    rec = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, 4.0, 2.5, 12_345.0)
    assert "12345" not in json.dumps(rec)
    keys = set(rec["vol_target_shadow"]) | set(rec["stop_vol"])
    assert not any(w in k for k in keys for w in ("dollar", "equity", "cash", "portfolio", "account"))


# --------------------------------------------------------------------------
# size_position parity: switch on vs off
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stop", [(120.0, "mocked"), (126.0, "mocked"), (None, "no stop")])
def test_sizing_identical_with_switch_on_and_off(monkeypatch, stop):
    off = _size(monkeypatch, False, bars=_bars(_closes(25)), stop=stop)
    on = _size(monkeypatch, True, bars=_bars(_closes(25)), stop=stop)
    assert not any(k in off for k in SHADOW_KEYS)
    assert all(k in on for k in SHADOW_KEYS)
    assert _without_shadow(on) == off
    assert on["shares_initial"] == off["shares_initial"] > 0
    assert on["stop_loss_price"] == off["stop_loss_price"]
    assert on["vol_target_shadow"]["read"] == "ok"
    json.dumps(on, allow_nan=False, default=str)


def test_sizing_identical_when_the_bar_read_fails(monkeypatch):
    off = _size(monkeypatch, False)
    on = _size(monkeypatch, True, bars_raise=RuntimeError("db down"))
    assert _without_shadow(on) == off
    assert on["vol_target_shadow"]["read"] == "error: RuntimeError"


def test_sizing_survives_the_record_itself_raising(monkeypatch):
    off = _size(monkeypatch, False)

    def boom(*a, **k):
        raise ValueError("bug")
    monkeypatch.setattr(risk_manager, "_vol_shadow_record", boom)
    on = _size(monkeypatch, True)
    assert on == off                                   # no shadow keys, same sizing


def test_real_atr_stop_is_unchanged_and_records_the_multiple(monkeypatch):
    off = _size(monkeypatch, False, stop=None, atr=4.0, bars=_bars(_closes(25)))
    on = _size(monkeypatch, True, stop=None, atr=4.0, bars=_bars(_closes(25)))
    lo, hi = config.ATR_STOP_MULTIPLIER["Compounder"]
    mult = (lo + hi) / 2
    assert off["stop_loss_price"] == on["stop_loss_price"] == round(130.0 - mult * 4.0, 2)
    assert _without_shadow(on) == off
    assert on["stop_vol"]["atr_multiple"] == mult
    assert on["stop_vol"]["stop_distance_atr"] == round(mult * 4.0 / 4.0, 2)
    assert on["stop_vol"]["stop_sigma20"] is not None


def test_moonshot_and_mocked_stops_record_no_atr_multiple(monkeypatch):
    moon = _size(monkeypatch, True, stop=None, bucket="Moonshot", price=20.0, bars=_bars(_closes(25)))
    if moon.get("shares_initial"):
        assert moon["stop_vol"]["atr_multiple"] is None
    mocked = _size(monkeypatch, True, bars=_bars(_closes(25)))   # calculate_stop not run
    assert mocked["stop_vol"]["atr_multiple"] is None and mocked["stop_vol"]["atr"] is None


def test_zero_share_returns_carry_no_shadow(monkeypatch):
    # a stop too wide for even one share within the per-trade risk cap
    on = _size(monkeypatch, True, stop=(1.0, "mocked"), price=5_000.0, bars=_bars(_closes(25)))
    off = _size(monkeypatch, False, stop=(1.0, "mocked"), price=5_000.0)
    assert on["shares_initial"] == 0 and on == off


def test_nothing_outside_risk_manager_reads_the_shadow_keys():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pat = re.compile(r"vol_target_shadow|stop_vol\b|stop_sigma20|VOL_TARGET_")
    readers = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in {".git", "venv", "tests", "__pycache__", ".claude",
                                                 "node_modules", ".chromadb"}]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(d, f)
            rel = os.path.relpath(path, root)
            if rel in ("risk_manager.py", "config.py") or rel.startswith("scripts" + os.sep):
                continue
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            if rel == os.path.join("agents", "crew.py"):
                # crew's only uses: restating the record at the final share count,
                # and two record-only functions reading it: the book heat
                # (_book_heat_record) and a blocked buy's would-be sizing
                # (_would_be_sizing_record)
                for fn in ("def _book_heat_record(", "def _would_be_sizing_record("):
                    a = text.find(fn)
                    b = text.find("\ndef ", a + 1) if a >= 0 else -1   # that one function only
                    if 0 <= a < b:
                        text = text[:a] + text[b:]
                text = text.replace('sizing.get("vol_target_shadow")', "").replace(
                    'sizing["vol_target_shadow"] = _vol_shadow_reweigh(sizing["vol_target_shadow"],', "")
            if pat.search(text):
                readers.append(rel)
    assert readers == []


# --------------------------------------------------------------------------
# review fixes: final-order restatement, skips, stale sessions, ATR guard
# --------------------------------------------------------------------------

def _ok_record(monkeypatch, shares=10, held=0.0):
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: _bars(_closes(25)))
    return risk_manager._vol_shadow_record("NVDA", 100.0, shares, held, 90.0, 4.0, 2.5, 10_000.0)


def test_reweigh_restates_weights_at_the_final_share_count(monkeypatch):
    vt = _ok_record(monkeypatch, shares=10, held=500.0)["vol_target_shadow"]
    assert vt["weights_at"] == "size_position" and vt["shares_at_sizing"] == 10
    assert vt["held_weight"] == 0.05
    out = risk_manager._vol_shadow_reweigh(vt, 4)
    assert out["weights_at"] == "final_order" and out["shares_final"] == 4
    assert out["order_weight_at_sizing"] == 0.1 and abs(out["order_weight"] - 0.04) < 1e-9
    assert abs(out["position_weight_after"] - 0.09) < 1e-9
    assert out["weight_ratio"] == round(vt["vol_weight"] / out["position_weight_after"], 3)
    assert out["vol_weight_shares"] == vt["vol_weight_shares"]           # does not depend on the order
    assert out["would_cut"] is (vt["vol_weight_shares"] < 4)
    assert vt["weights_at"] == "size_position"                           # input not mutated
    json.dumps(out, allow_nan=False)


def test_reweigh_never_raises_and_leaves_unrestatable_records(monkeypatch):
    vt = _ok_record(monkeypatch)["vol_target_shadow"]
    for shares in (None, "x", 0, 3):
        risk_manager._vol_shadow_reweigh(vt, shares)
    assert risk_manager._vol_shadow_reweigh(vt, 0)["order_weight"] == 0.0
    assert risk_manager._vol_shadow_reweigh(None, 3) is None
    assert risk_manager._vol_shadow_reweigh("junk", 3) == "junk"
    no_pv = dict(vt, shares_at_sizing=10, order_weight=None)
    assert risk_manager._vol_shadow_reweigh(no_pv, 3) is no_pv            # nothing to scale from
    # a non-ok read still restates the weights, with no ratio
    bad = dict(vt, read="no_cached_bars", vol_weight=None, vol_weight_shares=None)
    out = risk_manager._vol_shadow_reweigh(bad, 5)
    assert out["weights_at"] == "final_order" and out["weight_ratio"] is None and out["would_cut"] is None


def test_crew_restates_the_record_after_a_concentration_cut(monkeypatch):
    """The concentration cap halves the order (4 -> 2 shares); risk_json's
    vol_target_shadow follows the final share count."""
    vt = {"read": "ok", "vol_weight": 0.08, "vol_weight_shares": 3, "would_cut": False,
          "shares_at_sizing": 4, "order_weight": 0.2, "held_weight": 0.0,
          "position_weight_after": 0.2, "weight_ratio": 0.4, "weights_at": "size_position"}
    lpa = _route_with_vol(vt, {"passed": False,
                                    "flags": ["ANET position 25.0% exceeds Compounder cap 12%"],
                                    "ai_infra_size_cap_pct": 0.10})
    risk = lpa.call_args.kwargs["risk_json"]
    assert risk["shares_initial"] == 2
    rec = risk["vol_target_shadow"]
    assert rec["weights_at"] == "final_order" and rec["shares_final"] == 2
    assert rec["order_weight_at_sizing"] == 0.2 and abs(rec["order_weight"] - 0.1) < 1e-9
    assert abs(rec["position_weight_after"] - 0.1) < 1e-9 and rec["weight_ratio"] == 0.8
    assert rec["would_cut"] is False                                      # 3 >= 2 after the cut


def test_crew_without_a_cut_still_marks_final_order(monkeypatch):
    vt = {"read": "ok", "vol_weight": 0.08, "vol_weight_shares": 3, "would_cut": True,
          "shares_at_sizing": 4, "order_weight": 0.2, "held_weight": 0.0,
          "position_weight_after": 0.2, "weight_ratio": 0.4, "weights_at": "size_position"}
    lpa = _route_with_vol(vt, {"passed": True, "flags": []})
    rec = lpa.call_args.kwargs["risk_json"]["vol_target_shadow"]
    assert rec["weights_at"] == "final_order" and rec["shares_final"] == 4 and rec["would_cut"] is True


def _route_with_vol(vt, conc, shares=4, price=100.0):
    """test_concentration_record's route harness, with vol_target_shadow on the sizing dict."""
    from contextlib import ExitStack
    from agents import crew
    PV = 2_000.0
    sizing = {"shares_initial": shares, "initial_dollar": shares * price, "position_size_pct": 0.25,
              "order_pct": round(shares * price / PV, 4), "initial_entry_pct": 0.55,
              "flags": [], "stop_loss_price": price * 0.9, "vol_target_shadow": dict(vt)}
    result = {"ticker": "ANET", "action": "OPEN", "conviction": 7,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    with ExitStack() as st:
        st.enter_context(patch("config.MICRO_TIER_NAME_CAP_ENABLED", False))
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=price))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        st.enter_context(patch("db.log_signal"))
        st.enter_context(patch("execution.execute_manual"))
        rm = rm_cls.return_value
        rm.size_position.return_value = sizing
        rm.check_concentration.return_value = conc
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        crew._route_to_approval(result, {"regime": "Neutral"}, rm_cls.return_value)
    return lpa


def test_simulations_and_dry_runs_skip_the_read(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "VOL_TARGET_SIZING_SHADOW_ENABLED", True)
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: calls.append(t) or _bars(_closes(25)))
    with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(120.0, "mocked")), \
         patch("risk_manager.TRADING_ENV", "paper"), \
         patch("db.get_state", side_effect=_state), patch("db.set_state"):
        sim = risk_manager.RiskManager(_client(), simulation=True)
        s1 = sim.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On", price=130.0)
        live = risk_manager.RiskManager(_client())
        s2 = live.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On",
                                price=130.0, log_fallback=False)
        s3 = live.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On", price=130.0)
    assert not any(k in s1 for k in SHADOW_KEYS) and not any(k in s2 for k in SHADOW_KEYS)
    assert all(k in s3 for k in SHADOW_KEYS) and calls == ["NVDA"]
    assert _without_shadow(s3) == s2


def test_stale_sessions_counts_weekdays_to_the_last_completed_session():
    from datetime import datetime as dt, timezone as tz
    # Fri 2026-11-20 bar; Mon 11-23 10:00 ET (15:00 UTC): last completed = Fri -> 0
    assert risk_manager._stale_sessions("2026-11-20", dt(2026, 11, 23, 15, 0, tzinfo=tz.utc)) == 0
    # Mon 11-23 17:00 ET: Monday completed -> 1
    assert risk_manager._stale_sessions("2026-11-20", dt(2026, 11, 23, 22, 0, tzinfo=tz.utc)) == 1
    # Wed 11-18 bar seen Mon 11-23 17:00 ET: Thu, Fri, Mon -> 3
    assert risk_manager._stale_sessions("2026-11-18", dt(2026, 11, 23, 22, 0, tzinfo=tz.utc)) == 3
    assert risk_manager._stale_sessions("not a date") is None


def test_record_carries_stale_sessions(monkeypatch):
    rec = _ok_record(monkeypatch)["vol_target_shadow"]
    assert isinstance(rec["stale_sessions"], int) and rec["stale_sessions"] >= 0


@pytest.mark.parametrize("atr", [float("inf"), float("nan"), -1.0, 0.0])
def test_non_finite_or_non_positive_atr_is_not_recorded(monkeypatch, atr):
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda t, w: _bars(_closes(25)))
    sv = risk_manager._vol_shadow_record("NVDA", 100.0, 10, 0.0, 90.0, atr, 2.5, 10_000.0)["stop_vol"]
    assert sv["atr"] is None and sv["stop_distance_atr"] is None
    assert sv["stop_sigma20"] is not None


def test_reweigh_clears_weights_when_order_weight_unknown():
    from risk_manager import _vol_shadow_reweigh
    vt = {"shares_at_sizing": 4, "order_weight": float("nan"), "held_weight": 0.1,
          "position_weight_after": 0.3, "vol_weight": 0.2, "weight_ratio": 0.667,
          "vol_weight_shares": 3}
    out = _vol_shadow_reweigh(vt, 2)
    assert out["weights_at"] == "final_order"
    assert out["order_weight"] is None and out["position_weight_after"] is None
    assert out["weight_ratio"] is None


def test_stale_sessions_accepts_a_datetime():
    from datetime import datetime as _dt, timezone as _tz
    from risk_manager import _stale_sessions
    now = _dt(2026, 11, 23, 15, 0, tzinfo=_tz.utc)            # Monday 10:00 ET
    assert _stale_sessions(_dt(2026, 11, 20, 21, 0, tzinfo=_tz.utc), now) == \
        _stale_sessions("2026-11-20", now) == 0
