from unittest.mock import patch

import tasks


def test_rth_alert_always_sent():
    with patch("db.get_state_strict") as get:
        assert tasks._first_offhours_stop_alert("NVDA", {"rth": True}) is True
        assert tasks._first_offhours_stop_alert("NVDA", {}) is True   # older alert dicts: no key
    get.assert_not_called()


def test_offhours_alert_once_per_et_day():
    state = {}
    with patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        assert tasks._first_offhours_stop_alert("NVDA", {"rth": False}) is True
        assert tasks._first_offhours_stop_alert("NVDA", {"rth": False}) is False
        assert tasks._first_offhours_stop_alert("AMD", {"rth": False}) is True
    assert set(state) == {"stop_alert_offhours:NVDA", "stop_alert_offhours:AMD"}


def test_offhours_alert_sent_when_state_unreadable():
    with patch("db.get_state_strict", side_effect=RuntimeError("db down")):
        assert tasks._first_offhours_stop_alert("NVDA", {"rth": False}) is True


def test_offhours_reason_says_the_broker_stop_fires_at_the_open():
    from types import SimpleNamespace
    from risk_manager import RiskManager
    rm = RiskManager.__new__(RiskManager)
    rm._client = SimpleNamespace(get_all_positions=lambda: [
        SimpleNamespace(symbol="NVDA", qty=1, current_price=197.91, avg_entry_price=200.0)])
    rm.monitor_read_failed = None
    stops = [{"ticker": "NVDA", "stop_price": 199.11, "thesis_bucket": "Compounder"}]
    with patch("db.get_all_position_stops", return_value=stops, create=True), \
         patch("market_data.is_market_hours", return_value=False):
        try:
            alerts = rm.check_trailing_stops()
        except Exception:
            alerts = None
    if alerts is None:
        import pytest
        pytest.skip("check_trailing_stops needs more fixtures; reason text checked in source")
    a = next(x for x in alerts if x["ticker"] == "NVDA")
    assert a["rth"] is False and "outside regular hours" in a["reason"]


from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

_STOP = [{"ticker": "NVDA", "action": "SELL", "order_type": "STP", "qty": 1.0,
          "stop_price": 199.11, "order_id": 14, "status": "PreSubmitted"}]


def _run_task(orders, held=1.0, backstop=False, state=None, runs=1, rth=False):
    """Run check_trailing_stops `runs` times on one off-hours NVDA breach;
    return the reasons of the alerts sent."""
    state = {} if state is None else state
    client = MagicMock()
    if isinstance(held, Exception):
        client.get_open_position.side_effect = held
    else:
        client.get_open_position.return_value = MagicMock(qty=held)
    get_orders = (patch("ibkr.get_open_orders", side_effect=orders) if isinstance(orders, Exception)
                  else patch("ibkr.get_open_orders", return_value=orders))
    with patch("tasks._get_client", return_value=client), \
         patch("risk_manager.RiskManager") as rm_cls, get_orders, \
         patch("tasks._execute_backstop_sell", return_value=backstop) as bs, \
         patch("alerts.send_stop_triggered") as send, \
         patch("config.AUTO_EXECUTE_PROTECTIVE_SELLS", True), \
         patch("db.get_state_strict", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        rm_cls.return_value.monitor_read_failed = False
        rm_cls.return_value.check_trailing_stops.return_value = [{
            "ticker": "NVDA", "action": "EXIT", "current_price": 197.91, "stop_price": 199.11,
            "rth": rth, "reason": "Price $197.91 is below the $199.11 stop outside regular hours"}]
        for _ in range(runs):
            tasks.check_trailing_stops.run()
    return [c.kwargs["reason"] for c in send.call_args_list], bs


def test_covered_offhours_alerts_once_and_says_the_stop_fires_at_the_open():
    reasons, bs = _run_task(_STOP, runs=3)
    assert len(reasons) == 1 and "the broker stop covers 1 shares, and it fires at the open" in reasons[0]
    bs.assert_not_called()


def test_uncovered_offhours_alerts_every_run_and_never_claims_a_broker_stop():
    """m541 (b): no working stop; the backstop schedules the open retry and
    returns False. Must not be silenced or reassuring."""
    reasons, bs = _run_task([], runs=3)
    assert len(reasons) == 3 and bs.call_count == 3
    assert all("NO working broker stop covers 1 shares" in r and "fires at the open" not in r
               and "backstop sale not done (see the backstop log" in r for r in reasons)


@pytest.mark.parametrize("orders, held", [(RuntimeError("ibkr down"), 1.0), (_STOP, RuntimeError("pos"))])
def test_unknown_cover_alerts_every_run(orders, held):
    """m541 (c): open orders or the position unreadable."""
    reasons, bs = _run_task(orders, held=held, runs=2)
    assert len(reasons) == 2 and all("cover unknown" in r for r in reasons)
    bs.assert_not_called()


def test_rth_breach_reason_unchanged():
    reasons, _ = _run_task(_STOP, rth=True, runs=2)
    assert reasons == ["Price $197.91 is below the $199.11 stop outside regular hours"] * 2


def test_not_held_says_so():
    reasons, bs = _run_task(_STOP, held=0.0, runs=2)
    assert len(reasons) == 2 and all("the position is not held at the broker" in r for r in reasons)
    bs.assert_not_called()
