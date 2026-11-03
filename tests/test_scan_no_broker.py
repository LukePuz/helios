"""Scan path without a broker: verdicts still come back, no pass is routed,
each pass gets a ROUTE_NO_BROKER row, and the alert goes out once per ET day."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agents import crew, scan_guard

_CANDS = [{"ticker": "AAA", "priority": "HIGH"}, {"ticker": "BBB", "priority": "HIGH"}]
_RESULTS = [{"ticker": "AAA", "action": "OPEN", "conviction": 7, "pass_to_risk": True},
            {"ticker": "BBB", "action": "WATCH", "conviction": 4, "pass_to_risk": False}]


def _run(ibkr_patch, positions=None):
    batch = MagicMock(return_value=[dict(r) for r in _RESULTS])
    route = MagicMock()
    exits = MagicMock()
    alert = MagicMock(return_value=True)
    with patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("agents.analyst.analyze_batch", batch), ibkr_patch, \
         patch.object(crew, "_get_positions", MagicMock(return_value=positions)), \
         patch.object(crew, "_route_to_approval", route), \
         patch.object(crew, "_log_route_exit", exits), \
         patch.object(scan_guard, "alert_scan_without_broker", alert):
        out = crew.process_candidates(_CANDS)
    return out, batch, route, exits, alert


def test_client_failure_skips_routing_and_logs_route_no_broker():
    out, batch, route, exits, alert = _run(patch("ibkr.IBKRClient", side_effect=ConnectionError("gw down")))
    assert [r["ticker"] for r in out] == ["AAA", "BBB"]          # verdicts still returned
    assert batch.call_args.kwargs["positions"] == {} and batch.call_args.kwargs["trading_client"] is None
    route.assert_not_called()
    assert exits.call_count == 1
    ticker, kind, result, reason = exits.call_args.args
    assert (ticker, kind, result["ticker"]) == ("AAA", "ROUTE_NO_BROKER", "AAA")
    assert "ConnectionError: gw down" in reason
    alert.assert_called_once()
    assert alert.call_args.args[1] == 1


def test_unreadable_positions_count_as_no_broker_but_keep_the_client_for_the_brief():
    client = MagicMock()
    out, batch, route, exits, alert = _run(patch("ibkr.IBKRClient", return_value=client), positions=None)
    assert batch.call_args.kwargs["trading_client"] is client     # brief inputs unchanged
    assert batch.call_args.kwargs["positions"] == {}
    route.assert_not_called()
    assert exits.call_args.args[1] == "ROUTE_NO_BROKER"
    alert.assert_called_once()


def test_healthy_broker_routes_passes_as_before():
    client = MagicMock()
    out, batch, route, exits, alert = _run(patch("ibkr.IBKRClient", return_value=client),
                                           positions={"ZZZ": {}})
    route.assert_called_once()
    assert route.call_args.args[0]["ticker"] == "AAA" and route.call_args.args[2] is client
    exits.assert_not_called()
    alert.assert_not_called()


def test_alert_is_deduped_per_et_day():
    state = {}
    sent = MagicMock()
    with patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("alerts._send", sent):
        # 2026-11-03 03:00 UTC is 2026-11-02 22:00 ET
        t1 = datetime(2026, 11, 3, 3, 0, tzinfo=timezone.utc)
        assert scan_guard.alert_scan_without_broker("ConnectionError: x", 1, now=t1) is True
        assert scan_guard.alert_scan_without_broker("ConnectionError: x", 0, now=t1) is False
        assert state["scan_no_broker_alerted_on"] == "2026-11-02"
        t2 = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
        assert scan_guard.alert_scan_without_broker("ConnectionError: x", 0, now=t2) is True
    assert sent.call_count == 2


def test_alert_never_raises():
    with patch("db.get_state", side_effect=RuntimeError("db down")):
        assert scan_guard.alert_scan_without_broker("x", 0) is False


def test_route_no_broker_is_not_a_verdict():
    import db
    assert "ROUTE_NO_BROKER" in db._NON_ANALYST_SIGNALS
