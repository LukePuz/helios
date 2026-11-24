"""Record-only book heat at each buy (agents/crew.py:_book_heat_record).

Fractions only, never raises, no broker call, and the approval is identical
apart from the new key with the switch on or off.
"""
import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest

import config
from agents import crew

PV = 20_000.0


def _inputs(age=3.0, pv=PV, positions=None, stops=None):
    positions = positions if positions is not None else {
        "NVDA": {"qty": 10, "market_value": 1_800.0},        # price 180, stop 162 -> 180 / pv
        "ANET": {"qty": 20, "market_value": 2_400.0},        # price 120, stop 125 -> stop above: 0
        "IONQ": {"qty": 30, "market_value": 900.0},          # no stop record
        "CASHX": {"qty": 0, "market_value": 0.0},            # flat: ignored
    }
    stops = stops if stops is not None else [
        {"ticker": "NVDA", "stop_price": 162.0}, {"ticker": "ANET", "stop_price": 125.0},
        {"ticker": "MSFT", "stop_price": 400.0},             # stop row without a position: ignored
    ]
    return {"positions": positions, "pv": pv, "age_min": age, "stops": stops}


def _sizing(shares=5, price=100.0, stop=90.0, vol=True):
    s = {"shares_initial": shares, "price_at_sizing": price, "stop_loss_price": stop}
    if vol:
        s["vol_target_shadow"] = {"order_weight": shares * price / PV}
        s["stop_vol"] = {"stop_sigma20": 0.8, "sigma20_pct": 0.125}
    return s


def test_record_values(monkeypatch):
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: _inputs())
    rec = crew._book_heat_record("AMD", _sizing())
    assert rec["read"] == "ok" and rec["snapshot_age_min"] == 3.0
    assert rec["n_positions"] == 3 and rec["n_with_stop"] == 2 and rec["n_no_stop"] == 1
    assert rec["n_stop_at_or_above_price"] == 1
    assert rec["sum_open_risk_pct"] == round(10 * 18.0 / PV, 4)
    assert rec["max_single_risk_pct"] == round(180.0 / PV, 4) and rec["max_single_ticker"] == "NVDA"
    assert rec["unprotected_value_pct"] == round(900.0 / PV, 4)
    by = {p["ticker"]: p for p in rec["positions"]}
    assert by["ANET"]["risk_pct"] == 0.0 and by["IONQ"]["risk_pct"] is None and by["IONQ"]["stop"] == "none"
    assert "CASHX" not in by and "MSFT" not in by
    assert rec["order_risk_pct"] == round(5 * 10.0 / PV, 4)
    assert rec["sum_open_risk_after_pct"] == round((180.0 + 50.0) / PV, 4)
    assert rec["order_risk_sigma20"] == 0.8
    assert rec["order_sigma20_loss_pct"] == round(500.0 / PV * 0.125, 4)
    assert rec["held_sigma20"] is None and rec["weights_at"] == "final_order"


def test_record_is_fractions_only(monkeypatch):
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: _inputs(pv=12_345.67))
    text = json.dumps(crew._book_heat_record("AMD", _sizing()), allow_nan=False)
    for bad in ("12345", "1800", "2400", "162.0", '"qty"', "market_value", "dollar", "equity"):
        assert bad not in text


@pytest.mark.parametrize("inputs,read", [
    (None, "no_snapshot"),
    (_inputs(age=16.0), "stale_snapshot"),
    (_inputs(pv=0.0), "no_portfolio_value"),
    (_inputs(pv=None), "no_portfolio_value"),
])
def test_read_states(monkeypatch, inputs, read):
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: inputs)
    rec = crew._book_heat_record("AMD", _sizing())
    assert rec["read"] == read and rec["sum_open_risk_pct"] is None


def test_failed_read_is_recorded_not_raised(monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(crew, "_book_heat_inputs", boom)
    assert crew._book_heat_record("AMD", _sizing())["read"] == "error: RuntimeError"


@pytest.mark.parametrize("sizing", [{}, {"shares_initial": "x"}, {"shares_initial": True, "price_at_sizing": 1,
                                                                  "stop_loss_price": 0.5},
                                    {"shares_initial": 3, "price_at_sizing": float("nan"), "stop_loss_price": 1},
                                    {"stop_vol": "junk", "vol_target_shadow": 5}])
def test_odd_sizing_never_raises(monkeypatch, sizing):
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: _inputs())
    rec = crew._book_heat_record("AMD", sizing)
    assert rec["read"] == "ok" and rec["order_risk_pct"] is None
    json.dumps(rec, allow_nan=False)


def test_odd_snapshot_rows_never_raise(monkeypatch):
    weird = {"A": "junk", "B": {"qty": "x"}, "C": {"qty": 5, "market_value": float("inf")},
             "D": {"qty": -5, "market_value": -500.0}, "E": {"qty": 2, "market_value": 200.0}}
    stops = [{"ticker": None}, {"ticker": "E", "stop_price": "bad"}, {"ticker": "e", "stop_price": 90.0}]
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: _inputs(positions=weird, stops=stops))
    rec = crew._book_heat_record("AMD", _sizing())
    assert rec["read"] == "ok" and rec["n_positions"] == 1           # only E is open long
    assert rec["sum_open_risk_pct"] == round(2 * 10.0 / PV, 4)       # lower-case stop ticker matched
    json.dumps(rec, allow_nan=False)


def test_empty_book(monkeypatch):
    monkeypatch.setattr(crew, "_book_heat_inputs", lambda: _inputs(positions={}, stops=[]))
    rec = crew._book_heat_record("AMD", _sizing())
    assert rec["n_positions"] == 0 and rec["sum_open_risk_pct"] == 0.0
    assert rec["max_single_risk_pct"] is None and rec["sum_open_risk_after_pct"] == rec["order_risk_pct"]


# --------------------------------------------------------------------------
# route: identical approval apart from book_heat, final share count
# --------------------------------------------------------------------------

def _route(conc, on, inputs, shares=4, price=100.0):
    """tests/test_concentration_record.py's route harness."""
    pv = 2_000.0
    sizing = {"shares_initial": shares, "initial_dollar": shares * price, "position_size_pct": 0.25,
              "order_pct": round(shares * price / pv, 4), "initial_entry_pct": 0.55,
              "flags": [], "stop_loss_price": price * 0.9}
    result = {"ticker": "ANET", "action": "OPEN", "conviction": 7,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    with ExitStack() as st:
        st.enter_context(patch.object(config, "BOOK_HEAT_SHADOW_ENABLED", on))
        st.enter_context(patch.object(crew, "_book_heat_inputs", lambda: inputs))
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
    return lpa.call_args.kwargs


def _strip(kw):
    """The approval call without book_heat and the wall-clock arrival.sizing_at
    (an existing shadow stamps the time of each call)."""
    risk = {k: v for k, v in kw["risk_json"].items() if k != "book_heat"}
    if isinstance(risk.get("arrival"), dict):
        risk["arrival"] = {k: v for k, v in risk["arrival"].items() if k != "sizing_at"}
    return {**kw, "risk_json": risk}


CUT = {"passed": False, "flags": ["ANET position 25.0% exceeds Compounder cap 12%"],
       "ai_infra_size_cap_pct": 0.10}


@pytest.mark.parametrize("conc", [{"passed": True, "flags": []}, CUT])
def test_approval_identical_with_switch_on_and_off(conc):
    inp = _inputs(pv=2_000.0)
    off = _route(conc, False, inp)
    on = _route(conc, True, inp)
    assert "book_heat" not in off["risk_json"] and on["risk_json"]["book_heat"]["read"] == "ok"
    assert _strip(on) == _strip(off)


def test_order_risk_follows_the_concentration_cut():
    on = _route(CUT, True, _inputs(pv=2_000.0))
    risk = on["risk_json"]
    assert risk["shares_initial"] == 2
    assert risk["book_heat"]["order_risk_pct"] == round(2 * 10.0 / 2_000.0, 4)


def test_route_survives_the_record_raising():
    def boom(*a, **k):
        raise ValueError("bug")
    with patch.object(crew, "_book_heat_record", boom):
        on = _route({"passed": True, "flags": []}, True, _inputs(pv=2_000.0))
    off = _route({"passed": True, "flags": []}, False, _inputs(pv=2_000.0))
    assert "book_heat" not in on["risk_json"] and _strip(on) == _strip(off)


def test_no_broker_call_in_the_inputs():
    import inspect
    src = inspect.getsource(crew._book_heat_inputs) + inspect.getsource(crew._book_heat_record)
    for call in ("get_all_positions", "get_open_position", "get_ib", "trading_client", "_client"):
        assert call not in src



def test_short_and_unpriced_positions_are_counted(monkeypatch):
    positions = {"AAA": {"qty": 2, "market_value": 200.0},
                 "BBB": {"qty": -1, "market_value": -50.0},
                 "CCC": {"qty": 3, "market_value": 0.0},
                 "DDD": {"qty": 1, "market_value": float("nan")}}
    monkeypatch.setattr(crew, "_book_heat_inputs",
                        lambda: _inputs(positions=positions, stops=[{"ticker": "AAA", "stop_price": 90.0}]))
    rec = crew._book_heat_record("NVDA", {"shares_initial": 1, "price_at_sizing": 100.0,
                                          "stop_loss_price": 95.0})
    assert rec["read"] == "ok" and rec["n_positions"] == 1
    assert rec["n_short"] == 1 and rec["n_unpriced"] == 2


def test_real_inputs_read_one_transaction_with_a_timeout(monkeypatch):
    """The real _book_heat_inputs on a fake connection: a statement timeout
    first, both SELECTs on one cursor, a rollback, and the connection back."""
    import db
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    executed, state = [], {"put": 0, "rolled": 0}
    snap = ({"AAA": {"qty": 1, "market_value": 100.0}}, Decimal("1000.50"),
            datetime.now(timezone.utc) - timedelta(minutes=3))

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None): executed.append(sql)
        def fetchone(self): return snap
        def fetchall(self): return [("AAA", Decimal("90.0"))]

    class Conn:
        def cursor(self): return Cur()
        def rollback(self): state["rolled"] += 1

    monkeypatch.setattr(db, "_get_conn", lambda: Conn())
    monkeypatch.setattr(db, "_put_conn", lambda c: state.__setitem__("put", state["put"] + 1))
    out = _REAL_BOOK_HEAT_INPUTS()          # the conftest autouse stub replaces crew's
    assert executed[0].startswith("SET LOCAL statement_timeout")
    assert "portfolio_snapshots" in executed[1] and "position_stops" in executed[2]
    assert state == {"put": 1, "rolled": 1}
    assert out["stops"] == [{"ticker": "AAA", "stop_price": Decimal("90.0")}]
    assert 2.5 < out["age_min"] < 3.5 and out["pv"] == Decimal("1000.50")


# Captured at import, before the conftest autouse stub replaces it per test.
_REAL_BOOK_HEAT_INPUTS = crew._book_heat_inputs
