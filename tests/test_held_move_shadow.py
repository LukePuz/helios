from unittest.mock import patch

import db
import tasks


def _bars(*closes, last_day="2026-11-05"):
    days = ["2026-11-03", "2026-11-04", last_day][-len(closes):]
    return [{"time": d, "close": c} for d, c in zip(days, closes)]


def test_session_move_pct():
    assert tasks._session_move_pct(_bars(100.0, 105.0), "2026-11-05") == 5.0
    assert tasks._session_move_pct(_bars(100.0, 94.0), "2026-11-05") == -6.0
    assert tasks._session_move_pct(_bars(100.0, 105.0, last_day="2026-11-04"), "2026-11-05") is None
    assert tasks._session_move_pct(_bars(105.0), "2026-11-05") is None
    assert tasks._session_move_pct([], "2026-11-05") is None


def _run(stops, broker, bars, caplog):
    import logging
    from types import SimpleNamespace

    def get_bars(t, days=5):
        if isinstance(bars[t], Exception):
            raise bars[t]
        return bars[t]
    client = SimpleNamespace(get_all_positions=(lambda: [SimpleNamespace(symbol=t, qty=1) for t in broker])
                             if not isinstance(broker, Exception) else (lambda: (_ for _ in ()).throw(broker)))
    stop_read = (patch("db.get_all_position_stops", side_effect=stops) if isinstance(stops, Exception)
                 else patch("db.get_all_position_stops", return_value=[{"ticker": t} for t in stops]))
    with stop_read, patch("tasks._get_client", return_value=client), \
         patch("market_data.get_bars", side_effect=get_bars), \
         patch("db.log_signal") as log_signal, caplog.at_level(logging.INFO):
        tasks.held_move_shadow(today="2026-11-05")
    return log_signal, [r.getMessage() for r in caplog.records if "held names checked" in r.getMessage()]


def test_logs_only_names_at_or_over_5_pct_with_held_source(caplog):
    bars = {"NVDA": _bars(200.0, 189.0), "MSFT": _bars(500.0, 510.0), "ANET": [],
            "AMKR": _bars(40.0, 43.0), "OLD": RuntimeError("hmds")}
    log_signal, summary = _run(["NVDA", "MSFT", "OLD"], ["NVDA", "MSFT", "ANET", "AMKR"], bars, caplog)
    got = {c.args[0]: c.kwargs["provenance"] for c in log_signal.call_args_list}
    assert set(got) == {"NVDA", "AMKR"}
    assert got["NVDA"]["move_pct"] == -5.5 and got["NVDA"]["held_source"] == "both"
    assert got["AMKR"]["held_source"] == "broker"      # no stop row: the naked case (m542)
    assert got["NVDA"]["sources_read"] == {"broker": True, "stops": True}
    assert summary == ["held move shadow: 5 held names checked, 2 with no reading for 2026-11-05: ANET, OLD"]


def test_one_source_unreadable_still_checks_the_other(caplog):
    log_signal, _ = _run(RuntimeError("db"), ["NVDA"], {"NVDA": _bars(200.0, 189.0)}, caplog)
    assert log_signal.call_args.kwargs["provenance"]["held_source"] == "broker"
    assert log_signal.call_args.kwargs["provenance"]["sources_read"] == {"broker": True, "stops": False}
    log_signal, _ = _run(["NVDA"], RuntimeError("ibkr"), {"NVDA": _bars(200.0, 189.0)}, caplog)
    assert log_signal.call_args.kwargs["provenance"]["held_source"] == "stops"


def test_both_sources_unreadable_logs_nothing(caplog):
    log_signal, summary = _run(RuntimeError("db"), RuntimeError("ibkr"), {}, caplog)
    log_signal.assert_not_called()
    assert summary == ["held move shadow: 0 held names checked, 0 with no reading for 2026-11-05"]


def test_kind_is_not_a_verdict_and_is_scheduled_on_ops():
    assert "HELD_MOVE_SHADOW" in db._NON_ANALYST_SIGNALS and "HELD_MOVE_SHADOW" in db._RECORD_ONLY_SIGNALS
    entry = tasks.app.conf.beat_schedule["held-move-shadow"]
    assert entry["task"] == "tasks.held_move_shadow" and entry["options"]["queue"] == "ops"
