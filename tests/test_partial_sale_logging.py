"""Profit-target partial sales are recorded in trades/orders, and the alert
shows the fill price, not the new stop (audit 2026-10-28 10:53 #3)."""
from unittest.mock import MagicMock, patch

import risk_manager
import tasks


def _rm():
    return object.__new__(risk_manager.RiskManager)


def test_partial_sale_logs_trade_and_order_with_fill():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=70)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order",
               return_value={"filled_qty": 30, "order_id": 7, "fill_price": 125.5}), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place", return_value={"order_id": 8}), \
         patch("alerts._send"), \
         patch("db.log_trade") as log_trade, \
         patch("db.log_order", return_value=11) as log_order, \
         patch("db.update_order_fill") as fill:
        _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0, decision_price=126.0)
    log_trade.assert_called_once()
    kw = log_trade.call_args.kwargs
    assert kw["side"] == "SELL" and kw["quantity"] == 30 and kw["price"] == 125.5
    assert log_order.call_args.kwargs["decision_price"] == 126.0
    fill.assert_called_once_with(11, 125.5, 30.0, "filled")


def test_logging_failure_does_not_block_stop_replacement():
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=70)
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("db.set_stop_broker_order_id"), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": 30, "order_id": 7}), \
         patch("ibkr.get_client", return_value=client), \
         patch("tasks.manage_stop_place", return_value={"order_id": 8}) as stop, \
         patch("alerts._send"), \
         patch("db.log_trade", side_effect=RuntimeError("db down")):
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0)
    assert result is not None
    stop.assert_called_once_with("NVDA", 70.0, 100.0)


def _run_task(results):
    with patch("tasks._get_client"), \
         patch("risk_manager.RiskManager") as rm_cls, \
         patch("alerts.send_trade_alert") as trade_alert, \
         patch("alerts._send") as send:
        rm_cls.return_value.check_profit_targets.return_value = results
        tasks.check_profit_targets.run()
    return trade_alert, send


def test_alert_uses_fill_price():
    trade_alert, _ = _run_task([{
        "ticker": "NVDA", "action": "PARTIAL_SELL", "shares_sold": 30, "new_stop": 100.0,
        "trade_style": "swing", "gain_pct": 0.25, "order_result": {"fill_price": 125.5}}])
    assert trade_alert.call_args.kwargs["price"] == 125.5


def test_breakeven_only_sends_no_trade_alert():
    trade_alert, send = _run_task([{
        "ticker": "PLTR", "action": "BREAKEVEN_STOP", "shares_sold": 0, "new_stop": 150.0,
        "trade_style": "swing", "gain_pct": 0.2}])
    trade_alert.assert_not_called()
    assert "BREAKEVEN STOP" in send.call_args.args[0]


def test_unconfirmed_partial_sale_is_logged_for_reconcile():
    """Professor C12: an unconfirmed sale must reach orders as 'unconfirmed' so
    reconcile_unconfirmed_orders can catch a late fill."""
    with patch("db.get_position_stop", return_value={"stop_price": 90.0}), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.place_market_order", return_value={"filled_qty": 0, "order_id": 7}), \
         patch("alerts._send"), \
         patch("db.log_order", return_value=12) as log_order:
        result = _rm()._partial_sell_and_resize_stop("NVDA", 100, 30, 100.0, decision_price=126.0)
    assert result is None
    kw = log_order.call_args.kwargs
    assert kw["status"] == "unconfirmed" and kw["broker_order_id"] == 7 and kw["side"] == "SELL"
