"""add_stale_record activity row (execution._record_add_stale): an ADD that
takes _whole_position_stop's stale-record branch (reads show no more than
the fill, as on 11-10 NVDA). Record only: the return value, the cancels and
the placed stop are unchanged."""
from unittest.mock import MagicMock, patch

import pytest

import execution


def _run(action="ADD", cache_qty=1, fresh=(1.0, 90.0), held_before=None, prior=None,
         queued_effect=None):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=cache_qty, avg_entry_price=90.0)
    status: dict = {}
    prior = prior if prior is not None else {"stop_price": 89.3, "opened_at": None,
                                             "stop_method": "atr"}
    with patch("db.get_position_stop", return_value=prior), \
         patch("ibkr.get_client", return_value=client), \
         patch("ibkr.get_fresh_stock_position", return_value=fresh), \
         patch("db.log_activity_direct"), patch("db.get_all_position_stops", return_value=[]), \
         patch("db.log_activity_queued", side_effect=queued_effect) as queued, \
         patch("tasks.manage_cancel_stop_orders_for") as cancel:
        out = execution._whole_position_stop("NVDA", 79.74, 1, status=status, action=action,
                                             held_before=held_before, market_price=90.9)
    rows = [c for c in queued.call_args_list if c.args[0] == "add_stale_record"]
    return out, status, rows, cancel


def test_lagging_add_writes_the_row_and_changes_nothing():
    out, status, rows, cancel = _run()
    assert out == (79.74, None) and status.get("stale_record") is True
    cancel.assert_not_called()
    assert len(rows) == 1
    stage, ticker, message, sentiment, extra = rows[0].args
    assert (stage, ticker, sentiment) == ("add_stale_record", "NVDA", "neutral")
    assert extra["record"] == "rewritten_at_new_stop"
    assert extra["new_stop_vs_prior_stop_pct"] == pytest.approx((79.74 / 89.3 - 1) * 100, abs=1e-3)
    assert extra["filled_qty"] == 1 and extra["cache_qty"] == 1 and extra["fresh_qty"] == 1.0
    assert extra["held_before"] is None and extra["reclassified_open"] is False
    assert extra["prior_stop_method"] == "atr"


def test_reclassified_open_over_a_leftover_record_writes_no_row():
    # held 2 before, reads agree on exactly the 1-share fill: the earlier shares are gone
    out, status, rows, _ = _run(action="OPEN", held_before=2.0)
    assert out == (79.74, None) and status.get("stale_record") is True
    assert rows == []


def test_open_over_a_leftover_record_writes_no_row():
    out, status, rows, _ = _run(action="OPEN")
    assert out == (79.74, None) and status.get("stale_record") is True
    assert rows == []


def test_normal_add_writes_no_row():
    with patch("tasks._resting_sell_stops", return_value=[]):
        out, status, rows, cancel = _run(cache_qty=2, fresh=(2.0, 95.0), held_before=1.0)
    assert out == (89.3, (90.0, 2.0)) and "stale_record" not in status
    cancel.assert_called_once_with("NVDA")
    assert rows == []


def test_ambiguous_add_writes_no_row():
    out, status, rows, cancel = _run(cache_qty=2, fresh=(1.0, 90.0))
    assert out == (79.74, None) and status.get("prior_unreadable") is True
    cancel.assert_not_called()
    assert rows == []


def test_a_failing_row_write_changes_nothing():
    ok = _run()
    bad = _run(queued_effect=RuntimeError("queue"))
    assert bad[0] == ok[0] and bad[1] == ok[1]


def test_an_unusable_prior_stop_still_returns_the_new_stop():
    out, status, rows, _ = _run(prior={"stop_price": "x", "opened_at": None})
    assert out == (79.74, None) and status.get("stale_record") is True
    assert rows == []                                       # the row failed inside, logged
