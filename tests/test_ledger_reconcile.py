"""Daily ledger_reconcile row: the trade ledger's net shares per ticker
against the broker, stop records and open outcomes (audit 2026-11-07 17:17
#3). Log-only."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import tasks


def _t(sym, side, q, day):
    return {"symbol": sym, "side": side, "quantity": q,
            "timestamp": datetime(2026, 10, day, 15, 0, tzinfo=timezone.utc)}


_TRADES = [_t("QBTS", "BUY", 3, 1), _t("QBTS", "SELL", 10, 13),
           _t("NVDA", "BUY", 1, 5), _t("PLTR", "BUY", 1, 2), _t("PLTR", "SELL", 1, 9)]


def test_by_ticker_flags_negative_ledgers_and_broker_mismatches():
    r = tasks._ledger_by_ticker(_TRADES, {"NVDA": 1.0, "AMD": 2.0}, {"NVDA", "AMD"}, {"NVDA": 1})
    assert r["QBTS"]["ledger_net"] == -7 and r["QBTS"]["broker_qty"] == 0
    assert r["QBTS"]["match"] is False and r["QBTS"]["first_negative_at"].startswith("2026-10-13")
    assert r["NVDA"] == {"ledger_buys": 1, "ledger_sells": 0, "ledger_net": 1, "broker_qty": 1.0,
                         "stop_record": True, "open_outcomes": 1, "match": True,
                         "first_negative_at": None, "other_side_rows": 0,
                         "last_trade_at": "2026-10-05T15:00:00+00:00"}
    assert r["PLTR"]["match"] is True and r["PLTR"]["first_negative_at"] is None
    # Held at the broker with no ledger rows at all.
    assert r["AMD"]["ledger_net"] == 0 and r["AMD"]["match"] is False and r["AMD"]["last_trade_at"] is None


def test_unknown_broker_gives_no_match_verdict():
    r = tasks._ledger_by_ticker(_TRADES, None, set(), {})
    assert r["NVDA"]["broker_qty"] is None and r["NVDA"]["match"] is None


def test_task_writes_one_row():
    client = SimpleNamespace(get_all_positions=lambda: [SimpleNamespace(symbol="NVDA", qty=1)])
    with patch("db.get_ledger_trades", return_value=_TRADES), \
         patch("tasks._get_client", return_value=client), \
         patch("db.get_all_position_stops", return_value=[{"ticker": "NVDA"}]), \
         patch("db.get_open_outcomes", return_value=[{"ticker": "NVDA"}]), \
         patch("db.log_activity_direct") as write:
        tasks.ledger_reconcile()
    stage, _, msg, _, extra = write.call_args.args
    assert stage == "ledger_reconcile" and extra["mismatched"] == ["QBTS"]
    assert extra["ledger_negative"] == ["QBTS"] and extra["broker_read"] is True
    assert "1 of 3" in msg


def test_task_skips_on_unreadable_trades_and_records_an_unread_broker():
    with patch("db.get_ledger_trades", return_value=None), patch("db.log_activity_direct") as write:
        tasks.ledger_reconcile()
        assert not write.called

    def boom():
        raise RuntimeError("ibkr down")
    with patch("db.get_ledger_trades", return_value=_TRADES), \
         patch("tasks._get_client", return_value=SimpleNamespace(get_all_positions=boom)), \
         patch("db.get_all_position_stops", return_value=[]), \
         patch("db.get_open_outcomes", return_value=[]), \
         patch("db.log_activity_direct") as write:
        tasks.ledger_reconcile()
    extra = write.call_args.args[4]
    assert extra["broker_read"] is False and extra["mismatched"] is None


def test_beat_entry():
    e = tasks.app.conf.beat_schedule["ledger-reconcile"]
    assert e["task"] == "tasks.ledger_reconcile" and e["options"]["queue"] == "ops"


def test_unknown_side_rows_are_counted():
    r = tasks._ledger_by_ticker([_t("AMD", "BUY", 2, 1), _t("AMD", "SHORT", 1, 2)], None, set(), {})
    assert r["AMD"]["ledger_net"] == 2 and r["AMD"]["other_side_rows"] == 1
