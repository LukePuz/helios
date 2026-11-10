"""gap_risk_shadow on a sized OPEN/ADD: the stop-out cost if the stop fills
through a gap the size of max(stop distance, today's options implied move),
as a percent of equity, next to the per-trade risk cap. Percentages only,
no dollar amount; log-only, never changes shares or routing.
No live DB/IBKR/Redis."""
import json
from contextlib import ExitStack
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from agents import crew

TODAY = date(2026, 11, 6)
KEYS = {"implied_move_pct", "implied_move_dte", "implied_move_last_pct", "implied_move_last_date",
        "implied_move_last_dte",
        "gap_risk_pct_of_equity_last", "gap_risk_pct_of_equity_position", "stop_distance_pct", "gap_risk_pct_of_equity", "cap_pct",
        "would_exceed", "next_earnings_days", "status"}


def _book(move=8.0, et_date=TODAY.isoformat(), expiry="2026-11-13"):
    rec = {"implied_move_pct": move, "et_date": et_date, "underlying": 100.0, "straddle": 8.0}
    if expiry is not None:
        rec["expiry"] = expiry
    return json.dumps({"MSFT": rec})


def _shadow(shares=10, price=100.0, stop=95.0, equity=10_000.0, book=None, earnings=None, **kw):
    with patch("db.get_state", return_value=book), \
         patch("db.get_next_earnings", return_value=earnings):
        return crew._gap_risk_shadow("MSFT", shares, price, stop, equity, today=TODAY, **kw)


def test_known_case_uses_the_larger_implied_move():
    s = _shadow(book=_book(8.0), earnings=date(2026, 11, 20))
    assert set(s) == KEYS
    assert s["implied_move_pct"] == 8.0 and s["stop_distance_pct"] == 5.0
    assert s["gap_risk_pct_of_equity"] == pytest.approx(0.8)   # 10 x $100 x 8% / $10k
    assert s["cap_pct"] == 2.0 and s["would_exceed"] is False
    assert s["next_earnings_days"] == 14 and s["status"] == "ok"
    assert s["implied_move_dte"] == 7                          # expiry 2026-11-13
    big = _shadow(shares=30, book=_book(8.0))
    assert big["gap_risk_pct_of_equity"] == pytest.approx(2.4) and big["would_exceed"] is True


def test_the_stop_distance_is_used_when_it_is_wider():
    s = _shadow(stop=80.0, book=_book(3.0))
    assert s["gap_risk_pct_of_equity"] == pytest.approx(2.0) and s["would_exceed"] is False


def test_no_implied_move_falls_back_to_the_stop_distance():
    for book in (None, "{}", _book(8.0, et_date="2026-11-05"), _book(None), "not json"):
        s = _shadow(book=book)
        assert s["implied_move_pct"] is None and s["status"] == "no_implied_move"
        assert s["gap_risk_pct_of_equity"] == pytest.approx(0.5)
        assert s["next_earnings_days"] is None


def test_implied_move_dte_is_null_without_an_expiry():
    assert _shadow(book=_book(expiry=None))["implied_move_dte"] is None
    assert _shadow(book=_book(expiry="garbage"))["implied_move_dte"] is None
    assert _shadow(book=None)["implied_move_dte"] is None
    s = _shadow(book=_book(expiry="2026-11-06T16:00:00"))
    assert s["implied_move_dte"] == 0 and s["implied_move_pct"] == 8.0


def test_setup_gate_path_uses_its_category_cap():
    s = _shadow(book=_book(), setup_gate_path=True, bucket="Breakout", regime_multiplier=0.5)
    assert s["cap_pct"] == pytest.approx(0.75)   # 1.5% x 0.5


def test_no_dollar_amount_is_stored():
    shares, price, stop, equity = 7, 123.45, 111.0, 9_876.5
    s = _shadow(shares, price, stop, equity, book=_book(), earnings=date(2026, 11, 9))
    dollars = {equity, shares * price, price, stop, shares * stop, shares * (price - stop),
               round(shares * price, 2), round(shares * (price - stop), 2)}
    assert set(s) == KEYS
    assert not any(isinstance(v, float) and any(v == pytest.approx(d) for d in dollars)
                   for v in s.values())
    assert not any(w in k for k in s for w in ("dollar", "notional", "value", "equity_usd", "cost"))


def test_never_raises():
    assert _shadow(stop=None)["status"] == "no_stop"
    assert _shadow(stop=120.0)["status"] == "no_stop"
    assert _shadow(equity=None)["status"] == "no_equity"
    assert _shadow(price=float("nan"))["status"] == "no_stop"
    with patch("db.get_state", side_effect=RuntimeError("db down")), \
         patch("db.get_next_earnings", side_effect=RuntimeError("db down")):
        s = crew._gap_risk_shadow("MSFT", 10, 100.0, 95.0, 10_000.0, today=TODAY)
    assert s["status"] == "no_implied_move" and s["gap_risk_pct_of_equity"] == pytest.approx(0.5)
    with patch("config.RISK_PCT_PER_TRADE", "x"):
        assert _shadow(book=_book())["status"] == "error"


# ── crew routing ──────────────────────────────────────────────────────────────

PV = 100_000.0


def _sizing():
    return {"shares_initial": 40, "initial_dollar": 4_000.0, "position_size_pct": 0.05,
            "order_pct": 0.04, "initial_entry_pct": 0.55, "flags": [], "stop_loss_price": 95.0,
            "dollar_amount": 5_000.0, "max_loss_if_stopped": 200.0, "regime_multiplier": 1.0,
            "bucket": "Compounder"}


def _route(shadow_patch=None, action="OPEN", broker_held=(False, "snapshot", {})):
    result = {"ticker": "MSFT", "action": action, "conviction": 8,
              "thesis_bucket": "Compounder", "reasoning": "r"}
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(portfolio_value=PV, cash=PV, buying_power=PV)
    client.get_all_positions.return_value = []
    with ExitStack() as st:
        st.enter_context(patch("db.has_active_approval", return_value=False))
        st.enter_context(patch("db.get_last_executed_approval", return_value=None))
        st.enter_context(patch("db.is_near_earnings", return_value=False))
        et_today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        st.enter_context(patch("db.get_state", return_value=_book(8.0, et_date=et_today)))
        st.enter_context(patch("db.get_next_earnings", return_value=None))
        st.enter_context(patch("catalyst_service.gates.catalyst_gate_check",
                               return_value={"approved": True}))
        st.enter_context(patch("agents.crew._get_price", return_value=100.0))
        rm_cls = st.enter_context(patch("risk_manager.RiskManager"))
        lpa = st.enter_context(patch("db.log_pending_approval", return_value=-1))
        st.enter_context(patch("db.log_signal"))
        held = st.enter_context(patch("tasks._broker_held", return_value=broker_held))
        if shadow_patch is not None:
            st.enter_context(shadow_patch)
        rm = rm_cls.return_value
        rm.size_position.side_effect = lambda *a, **k: _sizing()
        rm.portfolio_value.return_value = PV
        rm.check_concentration.return_value = {"passed": True, "flags": []}
        rm.check_correlation.return_value = {"reduce_by": 0}
        rm.micro_name_cap.return_value = None
        with patch("agents.crew._apply_micro_name_cap", return_value=False):
            crew._route_to_approval(result, {"regime": "Neutral"}, client)
    return lpa


def test_route_records_the_shadow_without_dollars():
    lpa = _route()
    risk = lpa.call_args.kwargs["risk_json"]
    s = risk["gap_risk_shadow"]
    assert set(s) == KEYS and s["status"] == "ok"
    assert s["gap_risk_pct_of_equity"] == pytest.approx(0.32)   # 40 x $100 x 8% / $100k
    assert s["would_exceed"] is False and s["cap_pct"] == 2.0
    assert risk["shadow_status"]["gap_risk_shadow"] == "ok"
    assert not any(v in (PV, 4_000.0) for v in s.values() if isinstance(v, float))
    json.dumps(risk, allow_nan=False)


def test_sizing_and_routing_are_identical_with_and_without_the_shadow():
    def strip(call):
        kw = dict(call.kwargs)
        # arrival is log-only and carries the wall-clock sizing time.
        risk = {k: v for k, v in kw.pop("risk_json").items() if k not in ("gap_risk_shadow", "arrival")}
        risk["shadow_status"] = {k: v for k, v in (risk.get("shadow_status") or {}).items()
                                 if k != "gap_risk_shadow"}
        return kw, risk

    with_shadow = _route()
    without = _route(patch("agents.crew._attach_gap_risk_shadow"))
    broken = _route(patch("agents.crew._gap_risk_shadow", side_effect=RuntimeError("boom")))
    assert "gap_risk_shadow" not in without.call_args.kwargs["risk_json"]
    assert strip(with_shadow.call_args) == strip(without.call_args) == strip(broken.call_args)
    assert with_shadow.call_args.kwargs["risk_json"]["shares_initial"] == 40


def test_position_gap_risk_counts_held_plus_new_shares():
    s = _shadow(book=_book(8.0), held_shares=20)
    assert s["gap_risk_pct_of_equity"] == pytest.approx(0.8)            # 10 new shares
    assert s["gap_risk_pct_of_equity_position"] == pytest.approx(2.4)   # (20 + 10) x $100 x 8% / $10k
    for held in (None, 0, float("nan"), "x"):
        assert _shadow(book=_book(8.0), held_shares=held)["gap_risk_pct_of_equity_position"] is None


def test_route_add_records_the_whole_position_and_open_records_null():
    snap = (True, "snapshot", {"qty": 60.0, "market_value": 6_000.0})
    add = _route(action="ADD", broker_held=snap).call_args.kwargs["risk_json"]["gap_risk_shadow"]
    assert add["gap_risk_pct_of_equity"] == pytest.approx(0.32)            # 40 new shares
    assert add["gap_risk_pct_of_equity_position"] == pytest.approx(0.8)    # (60 + 40) x $100 x 8% / $100k
    assert 6_000.0 not in add.values()
    unknown = _route(action="ADD", broker_held=(None, "no_snapshot", {}))
    assert unknown.call_args.kwargs["risk_json"]["gap_risk_shadow"]["gap_risk_pct_of_equity_position"] is None
    opened = _route(action="OPEN", broker_held=snap).call_args.kwargs["risk_json"]["gap_risk_shadow"]
    assert "gap_risk_pct_of_equity_position" in opened                     # always present
    assert opened["gap_risk_pct_of_equity_position"] is None               # null on an OPEN



def test_pre_open_falls_back_to_the_last_sessions_move_with_its_date():
    """m604: Monday pre-open, the stored move is Friday's; today's fields
    stay empty and the _last fields carry it."""
    s = _shadow(book=_book(8.0, et_date="2026-11-05"))
    assert s["implied_move_pct"] is None and s["status"] == "no_implied_move"
    assert s["implied_move_last_pct"] == 8.0 and s["implied_move_last_date"] == "2026-11-05"
    assert s["implied_move_last_dte"] == 8                          # 11-05 to the 11-13 expiry
    assert s["gap_risk_pct_of_equity"] == pytest.approx(0.5)        # stop distance only
    assert s["gap_risk_pct_of_equity_last"] == pytest.approx(0.8)   # through Friday's move
    today = _shadow(book=_book(8.0))
    assert today["implied_move_last_pct"] == 8.0 and today["gap_risk_pct_of_equity_last"] is None
    future = _shadow(book=_book(8.0, et_date="2026-11-09"))
    assert future["implied_move_last_pct"] is None

    undated = _shadow(book=_book(8.0, et_date=None))
    assert undated["implied_move_last_pct"] is None and undated["implied_move_last_date"] is None
