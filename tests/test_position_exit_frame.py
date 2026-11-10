"""Each held position's exit frame after the close (audit 2026-11-07 11:07
#1). Log-only."""
from unittest.mock import patch

import tasks


def _bars(*pairs):
    return [{"date": d, "close": c} for d, c in pairs]


_STOP = {"ticker": "NVDA", "entry_price": 100.0, "stop_price": 92.0, "thesis_bucket": "Compounder",
         "trailing_active": False, "profit_target_taken": False, "opened_at": "2026-10-01 14:00:00"}


def test_peak_giveback_and_distances():
    bars = _bars(("2026-09-30", 140.0), ("2026-10-01", 101.0), ("2026-10-15", 125.0),
                 ("2026-11-06", 110.0))
    f = tasks._exit_frame(_STOP, bars)
    assert f["close_date"] == "2026-11-06" and f["bars_since_entry"] == 3
    assert f["gain_pct"] == 10.0 and f["peak_gain_pct"] == 25.0     # 09-30's 140 is before entry
    assert f["giveback_from_peak_pct"] == -12.0
    assert f["trail_arm_pct"] == 30.0 and f["to_trail_arm_pp"] == 20.0
    assert f["above_stop_pct"] == round((110 / 92 - 1) * 100, 2) and f["stop_vs_entry_pct"] == -8.0


def test_trailing_or_no_arm_bucket_gives_no_arm_distance():
    bars = _bars(("2026-11-06", 110.0))
    assert tasks._exit_frame({**_STOP, "trailing_active": True}, bars)["to_trail_arm_pp"] is None
    f = tasks._exit_frame({**_STOP, "thesis_bucket": "Moonshot"}, bars)
    assert f["trail_arm_pct"] is None and f["to_trail_arm_pp"] is None


def test_missing_inputs_give_none_and_never_raise():
    f = tasks._exit_frame({"ticker": "X"}, [])
    assert f["gain_pct"] is None and f["peak_gain_pct"] is None and f["close_date"] is None
    f = tasks._exit_frame({**_STOP, "entry_price": "bad"}, _bars(("2026-11-06", "nan")))
    assert f["gain_pct"] is None


def _client(symbols=None, raises=False):
    from unittest.mock import MagicMock
    c = MagicMock()
    if raises:
        c.get_all_positions.side_effect = RuntimeError("ibkr")
    else:
        c.get_all_positions.return_value = [MagicMock(symbol=s, qty=1) for s in symbols or []]
    return c


def test_task_writes_one_row_per_position():
    with patch("db.get_all_position_stops", return_value=[_STOP, {**_STOP, "ticker": "AMD"}]), \
         patch("db.get_price_history", return_value=_bars(("2026-11-06", 110.0))), \
         patch("tasks._get_client", return_value=_client(["NVDA"])), \
         patch("db.log_activity_direct") as act:
        tasks.position_exit_frame()
    assert [c.args[0] for c in act.call_args_list] == ["position_exit_frame"] * 2
    assert [c.args[1] for c in act.call_args_list] == ["NVDA", "AMD"]
    nvda, amd = (c.args[4] for c in act.call_args_list)
    assert nvda["gain_pct"] == 10.0 and nvda["broker_held"] is True and amd["broker_held"] is False
    assert nvda["close_is_today"] is False                 # 11-06 is not today


def test_unreadable_broker_marks_held_unknown():
    with patch("db.get_all_position_stops", return_value=[_STOP]), \
         patch("db.get_price_history", return_value=[]), \
         patch("tasks._get_client", return_value=_client(raises=True)), \
         patch("db.log_activity_direct") as act:
        tasks.position_exit_frame()
    assert act.call_args.args[4]["broker_held"] is None
    assert act.call_args.args[4]["bars_read"] == 0


def test_unreadable_stops_write_nothing():
    with patch("db.get_all_position_stops", side_effect=RuntimeError("db")), \
         patch("db.log_activity_direct") as act:
        tasks.position_exit_frame()
    act.assert_not_called()


def test_scheduled_after_the_close():
    entry = tasks.app.conf.beat_schedule["position-exit-frame"]
    assert entry["task"] == "tasks.position_exit_frame" and entry["options"]["queue"] == "ops"
    refresh = tasks.app.conf.beat_schedule["refresh-price-history"]["schedule"]
    assert entry["schedule"].hour == {17} and entry["schedule"].minute == {30}
    assert refresh.hour == {16} and refresh.minute == {45}      # the cache it reads
