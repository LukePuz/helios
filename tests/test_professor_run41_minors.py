"""Professor review run 41 (2026-11-03) minors: m301 (the backstop sale's
trade row carries its orders.id) and m304 (the held-flat-stop alert on the
empty-fresh-read branch asks for a TWS check before any cancel). No live
DB, broker or Telegram."""
from unittest.mock import MagicMock, patch

import pytest

import tasks
import test_overstop as _ov   # module imports: no re-collection
import test_tasks_dedup as _dedup


def _backstop(log_order):
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(symbol="AMD", qty=10)]
    with patch("ibkr.get_client", return_value=client), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("tasks.manage_cancel_stop_orders_for"), \
         patch("ibkr.get_open_orders", return_value=[]), \
         patch("ibkr.place_market_order",
               return_value={"order_id": 77, "filled_qty": 10, "fill_price": 140.0}), \
         patch("db.log_trade", return_value=True) as log_trade, \
         patch("db.delete_position_stop"), \
         patch("db.log_order", **log_order), \
         patch("db.update_order_fill"), \
         patch("db.close_trade_outcome"), patch("alerts._send"):
        assert tasks._execute_backstop_sell("AMD", _ov.TestBackstopSellReads._alert) is True
    log_trade.assert_called_once()
    return log_trade.call_args.kwargs


class TestM301BackstopTradeRow:
    def test_trade_row_carries_the_orders_id(self):
        kw = _backstop({"return_value": 5})
        assert kw["order_row_id"] == 5
        assert kw["side"] == "SELL" and kw["quantity"] == 10 and kw["price"] == 140.0

    @pytest.mark.parametrize("log_order", [{"return_value": -1},
                                           {"side_effect": RuntimeError("db down")}])
    def test_failed_order_row_still_writes_the_trade(self, log_order):
        kw = _backstop(log_order)
        assert kw["order_row_id"] is None
        assert kw["reason"].startswith("Backstop stop-breach sell")


class TestM304EmptyReadWording:
    """The empty-fresh-read branch is treated as a probable reconnect gap; its
    held-flat-stop alert must not say both reads agree the ticker is flat."""
    NVDA_STOP = _dedup.TestReconcileConfirmsAFlatRead.NVDA_STOP
    STOP = _dedup.TestReconcileConfirmsAFlatRead.STOP

    def _run(self, fresh):
        state = {}
        with patch("tasks.open_manual_action") as opened, \
                patch("tasks._reconcile_cycle_alert_due", return_value=True):
            cancel, delete, alert = _dedup.TestReconcileConfirmsAFlatRead()._run(
                state, fresh, orders=[self.STOP, self.NVDA_STOP],
                cache=[MagicMock(symbol="NVDA", qty=5.0)])
        cancel.assert_not_called()
        delete.assert_not_called()
        held = [c.args[0] for c in alert.call_args_list if "Reconcile is on hold" in c.args[0]]
        actions = [c.args[1] for c in opened.call_args_list if c.args[0] == "AMD"]
        return held, actions

    def test_empty_fresh_read_asks_for_a_tws_check_first(self):
        held, actions = self._run({})
        assert len(held) == 1 and "AMD" in held[0]
        assert "reads flat in both reads" not in held[0]
        assert "came back empty" in held[0] and "Do not cancel it on this alert alone" in held[0]
        assert len(actions) == 1 and "reads flat in both reads" not in actions[0]
        assert "confirm in TWS" in actions[0]

    def test_incomplete_but_nonempty_read_keeps_the_old_wording(self):
        held, actions = self._run({"TSLA": 3.0})
        assert len(held) == 1 and "reads flat in both reads" in held[0]
        assert len(actions) == 1 and "reads flat in both reads" in actions[0]
