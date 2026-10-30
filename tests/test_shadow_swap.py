"""Swap shadow (Luke 10-29): log what a ranking swap would do when a buy at
c7+ is blocked for room. Nothing is sold."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents import crew

NOW = datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc)


def _pos(sym, mv=100.0, pl=0.0):
    return SimpleNamespace(symbol=sym, qty=1, market_value=mv, unrealized_pl=pl)


def _rm(*positions):
    rm = MagicMock()
    rm._client.get_all_positions.return_value = list(positions)
    return rm


def _sigs(table):
    return lambda t: table.get(t)


def _run(rm, conviction, sizing, sigs):
    with patch("db.get_latest_signal_for", side_effect=_sigs(sigs)), \
         patch("db.log_signal") as log_signal:
        rec = crew._shadow_swap(rm, "NOW", conviction, sizing, {"action": "OPEN", "conviction": conviction}, now=NOW)
    return rec, log_signal


CAP = {"flags": ["At the micro-tier max of 5 concurrent positions (5 held) — skipping new NOW position"]}
FRESH = NOW - timedelta(hours=5)


def test_cap_block_picks_lowest_fresh_holding_including_compounders():
    sigs = {"MSFT": {"signal": "HOLD", "confidence": 0.5, "timestamp": FRESH},
            "NVDA": {"signal": "HOLD", "confidence": 0.7, "timestamp": FRESH}}
    rec, log_signal = _run(_rm(_pos("MSFT"), _pos("NVDA")), 8, CAP, sigs)
    assert rec["would_swap"] and rec["sell"] == "MSFT"
    args = log_signal.call_args
    assert args.args[1] == "SHADOW_SWAP" and args.kwargs["decision"]["shadow_swap"]["sell"] == "MSFT"


def test_gap_below_two_or_stale_verdicts_mean_no_swap():
    sigs = {"MSFT": {"signal": "HOLD", "confidence": 0.6, "timestamp": FRESH},
            "OLD": {"signal": "HOLD", "confidence": 0.3, "timestamp": NOW - timedelta(days=5)}}
    rec, _ = _run(_rm(_pos("MSFT"), _pos("OLD")), 7, CAP, sigs)
    assert rec["would_swap"] is False and rec["stale_holdings"] == ["OLD"]


def test_ties_prefer_the_weaker_performer():
    sigs = {"A": {"signal": "HOLD", "confidence": 0.5, "timestamp": FRESH},
            "B": {"signal": "HOLD", "confidence": 0.5, "timestamp": FRESH}}
    rec, _ = _run(_rm(_pos("A", 110, 10), _pos("B", 90, -10)), 7, {"cash_blocked": True, "flags": []}, sigs)
    assert rec["sell"] == "B"


def test_not_run_below_c7_or_when_not_room_blocked():
    rec, log_signal = _run(_rm(_pos("MSFT")), 6, CAP, {})
    assert rec is None and not log_signal.called
    rec, log_signal = _run(_rm(_pos("MSFT")), 8, {"flags": ["Conviction 6 below the micro-tier minimum (7)"]}, {})
    assert rec is None and not log_signal.called


def test_never_raises():
    rm = MagicMock()
    rm._client.get_all_positions.side_effect = RuntimeError("ib down")
    with patch("db.log_signal"):
        assert crew._shadow_swap(rm, "NOW", 8, CAP, {}, now=NOW) is None


def test_nan_pl_from_ibkr_does_not_break_the_row():
    sigs = {"A": {"signal": "HOLD", "confidence": 0.5, "timestamp": FRESH},
            "B": {"signal": "HOLD", "confidence": 0.5, "timestamp": FRESH}}
    rec, log_signal = _run(_rm(_pos("A", 100, float("nan")), _pos("B", 100, float("nan"))), 8, CAP, sigs)
    assert rec["would_swap"] and log_signal.called
    assert rec["lowest"]["unrealized_pl_pct"] is None    # NaN dropped, row still written


def test_routing_rows_do_not_copy_the_plan():
    fields = crew._route_row_fields({"provenance": {"model": "m", "plan": {"relation": "FOLLOW"}}}, "X")
    assert "plan" not in fields["provenance"] and fields["provenance"]["model"] == "m"


def test_r_shadow_uses_the_real_stop():
    """Audit A2114-2: reward-to-risk against the stop sizing will place."""
    from agents.crew import _r_shadow
    res = {"valuation": {"bull_case_target": "$130 (12 months)"}, "upside_downside_ratio": 2.4}
    r = _r_shadow(res, 100.0, 94.0)
    assert r["computed_r"] == 5.0 and r["model_udr"] == 2.4 and r["stop_pct"] == 6.0
    r = _r_shadow({"valuation": {"bull_case_target": "n/a"}, "expected_return_pct": 12.0}, 100.0, 96.0)
    assert r["computed_r"] == 3.0 and r["target_source"] == "expected_return_pct"
    assert _r_shadow({}, 100.0, None)["computed_r"] is None


def test_shadow_fields_are_always_json_safe():
    """A shadow value json can't encode would drop the buy at the approval insert."""
    import json
    from datetime import date
    from decimal import Decimal
    from agents import crew
    assert crew._json_safe({"a": Decimal("1.5"), "d": [Decimal("2")]}) == {"a": 1.5, "d": [2.0]}
    assert crew._json_safe({"d": date(2026, 1, 1)}) is crew._UNENCODABLE   # dropped, not raised
    sizing = {"stop_loss_price": 95.0, "heat_shadow": {"x": date(2026, 1, 1)}}
    crew._attach_shadows(sizing, "ZZZ", {"valuation": {"bull_case_target": "$130"}}, 100.0)
    json.dumps(sizing)
    assert sizing["shadow_status"]["heat_shadow"] == "unencodable" and sizing["heat_shadow"] is None
    assert sizing["shadow_status"]["r_shadow"] == "ok"
    sizing = {"heat_shadow": None}
    with patch("agents.crew._r_shadow", side_effect=None, return_value={"error": "boom"}):
        crew._attach_shadows(sizing, "ZZZ", {}, 100.0)
    assert sizing["shadow_status"]["r_shadow"] == "error" and sizing["shadow_status"]["heat_shadow"] == "n/a"


def test_nan_shadow_value_is_dropped_not_stored():
    """Postgres jsonb rejects NaN; it would drop the buy at the approval insert."""
    from agents import crew
    assert crew._json_safe({"r": float("nan")}) is crew._UNENCODABLE
