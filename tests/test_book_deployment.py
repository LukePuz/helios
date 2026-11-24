"""tasks.book_deployment_record: one book_deployment row per session after
the close (invested and cash as percent of net liquidation, the regime's
cash_target_pct and the gap, sector and largest-name weights), from one
bounded read, fail-open, percentages only; and its batch_verify check."""
import importlib.util
import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import tasks

_BV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "batch_verify.py")
_spec = importlib.util.spec_from_file_location("batch_verify_book_deployment", _BV_PATH)
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)

# Round, made-up inputs: the tests assert percentages only.
PV, CASH = 1000.0, 810.0
POSITIONS = {"NVDA": {"qty": 2, "market_value": 80.0}, "AMD": {"qty": 1, "market_value": 60.0},
             "ZZZZ": {"qty": 1, "market_value": 50.0}, "OLD": {"qty": 0, "market_value": 0.0}}
WATCHLIST = {"NVDA": {"sector": "AI Infrastructure — Compute / Accelerators"},
             "AMD": {"sector": "AI Infrastructure — Compute / Accelerators"}}
AT = datetime(2026, 11, 20, 22, 30, tzinfo=timezone.utc)
SNAP_AT = datetime(2026, 11, 20, 22, 25, tzinfo=timezone.utc)
REGIME_AT = datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)


def _inp(**over):
    base = {"positions": POSITIONS, "pv": PV, "cash": CASH, "ts": SNAP_AT, "regime": "Neutral",
            "cash_target_pct": 20, "regime_at": REGIME_AT}
    return {**base, **over}


def test_weights_cash_and_gap():
    rec = tasks.book_deployment_fields(_inp(), AT, WATCHLIST, {"NVDA", "AMD"})
    assert rec["read"] == "ok" and rec["reason"] is None
    assert rec["invested_pct"] == 19.0 and rec["cash_pct"] == 81.0 and rec["other_pct"] == 0.0
    assert rec["cash_target_pct"] == 20.0 and rec["gap_pct"] == 61.0
    assert rec["n_positions"] == 3 and rec["largest_name"] == "NVDA"
    assert rec["largest_name_weight_pct"] == 8.0
    assert rec["sector_weights_pct"] == {"AI Infrastructure — Compute / Accelerators": 14.0,
                                         "(off watchlist)": 5.0}
    assert rec["theme_prefix_weights_pct"] == {"AI Infrastructure": 14.0, "(off watchlist)": 5.0}
    assert rec["ai_infra_pct"] == 14.0
    assert rec["source"] == "portfolio_snapshots" and rec["source_as_of"] == SNAP_AT.isoformat()
    assert rec["source_age_min"] == 5.0 and rec["regime"] == "Neutral"
    assert rec["regime_as_of"] == REGIME_AT.isoformat()


def test_percent_only_no_dollar_or_share_values():
    rec = tasks.book_deployment_fields(_inp(), AT, WATCHLIST, {"NVDA"})
    text = json.dumps(rec)
    for v in ("1000", "810", "80.0,", "60.0", "50.0"):
        assert v not in text, v
    assert not set(rec) & set(bv.BOOK_DEPLOYMENT_DOLLAR_KEYS)


def test_missing_parts_are_none_with_a_reason():
    rec = tasks.book_deployment_fields(_inp(cash=None), AT, WATCHLIST)
    assert rec["cash_pct"] is None and rec["gap_pct"] is None and rec["invested_pct"] == 19.0
    assert rec["reason"] == "snapshot has no cash"
    rec = tasks.book_deployment_fields(_inp(cash_target_pct=None, regime=None), AT, WATCHLIST)
    assert rec["cash_target_pct"] is None and rec["gap_pct"] is None
    assert rec["reason"] == "no cash_target_pct on the regime row"
    rec = tasks.book_deployment_fields(_inp(pv=0), AT, WATCHLIST)
    assert rec["read"] == "no_portfolio_value" and rec["invested_pct"] is None
    rec = tasks.book_deployment_fields(_inp(), AT, None)
    assert rec["sector_weights_pct"] is None and rec["invested_pct"] == 19.0
    rec = tasks.book_deployment_fields(None, AT, WATCHLIST)
    assert rec["read"] == "no_snapshot" and rec["cash_pct"] is None
    rec = tasks.book_deployment_fields(_inp(positions={}, cash=PV), AT, WATCHLIST)
    assert rec["cash_pct"] == 100.0 and rec["n_positions"] == 0 and rec["largest_name"] is None


def test_shorts_and_unpriced_are_counted_apart():
    pos = {"NVDA": {"qty": 1, "market_value": 100.0}, "X": {"qty": -1, "market_value": -50.0},
           "Y": {"qty": 1, "market_value": None}}
    rec = tasks.book_deployment_fields(_inp(positions=pos, cash=950.0), AT, {})
    assert (rec["n_positions"], rec["n_short"], rec["n_unpriced"]) == (1, 1, 1)
    assert rec["invested_pct"] == 10.0 and rec["other_pct"] == -5.0


def test_an_error_inside_is_an_error_read_with_the_class_only():
    with patch("risk_manager._theme_prefix", side_effect=ValueError("123.45")):
        rec = tasks.book_deployment_fields(_inp(), AT, WATCHLIST)
    assert rec["read"] == "error: ValueError" and "123.45" not in json.dumps(rec)


def test_the_read_is_one_bounded_read_only_statement_rolled_back():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (POSITIONS, PV, CASH, SNAP_AT, "Neutral", 20, REGIME_AT)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put, \
         patch("config.REGIME_OVERALL_ONLY", True):
        inp = tasks._book_deployment_inputs(AT)
    assert cur.execute.call_count == 1
    sql, params = cur.execute.call_args.args
    assert sql.startswith("SET TRANSACTION READ ONLY; SET LOCAL statement_timeout = %(ms)s; SELECT")
    assert params["ms"] == tasks.BOOK_DEPLOYMENT_STATEMENT_TIMEOUT_MS == 3000
    assert params["at"] == AT and "regime_type = 'overall'" in sql
    assert inp["cash_target_pct"] == 20 and inp["regime"] == "Neutral"
    conn.rollback.assert_called_once()
    put.assert_called_once()
    cur.fetchone.return_value = (None, None, None, None, None, None, None)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert tasks._book_deployment_inputs(AT) is None
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("config.REGIME_OVERALL_ONLY", False):
        tasks._book_deployment_inputs(AT)
    assert "regime_type" not in cur.execute.call_args.args[0]


def _write(inp=None, side_effect=None, done=frozenset(), enabled=True):
    with patch("config.BOOK_DEPLOYMENT_RECORD_ENABLED", enabled), \
         patch("tasks._book_deployment_inputs", return_value=inp, side_effect=side_effect) as reads, \
         patch("knowledge_loader.load_watchlist", return_value=WATCHLIST), \
         patch("db.get_activity_shadow_keys", return_value=set(done)), \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("tasks.log") as lg:
        wrote = tasks.book_deployment_record("2026-11-20")
    return wrote, act, reads, lg


def test_writer_row_and_message_are_percent_only():
    wrote, act, reads, lg = _write(_inp())
    assert wrote is True and reads.call_count == 1
    stage, ticker, msg, sentiment, rec = act.call_args.args
    assert (stage, ticker, sentiment) == ("book_deployment", "", "neutral")
    assert rec["shadow_key"].endswith(":2026-11-20") and rec["session_date"] == "2026-11-20"
    assert rec["watchlist_read"] == "ok" and rec["definitions"] is tasks._BOOK_DEPLOYMENT_DEFINITIONS
    assert "cash 81.0%" in msg and "target 20.0%" in msg and "gap 61.0 pp" in msg
    for k, _getters in [bv._ACTIVITY_SPECS["book_deployment"]]:
        assert all(key in rec for key in k)
    text = msg + json.dumps(rec)
    assert "1000" not in text and "810" not in text


def test_writer_dedupes_disables_and_writes_an_error_row():
    from config import TRADING_ENV
    assert _write(_inp(), done={f"{TRADING_ENV}:2026-11-20"})[0] is False
    wrote, act, reads, _ = _write(_inp(), enabled=False)
    assert wrote is False and not act.called and not reads.called
    wrote, act, _, _ = _write(side_effect=TimeoutError("statement timeout"))
    rec = act.call_args.args[4]
    assert wrote is True and rec["read"] == "error: TimeoutError"
    assert rec["reason"] == "snapshot or regime row unreadable" and rec["cash_pct"] is None
    wrote, act, _, _ = _write(None)
    assert act.call_args.args[4]["read"] == "no_snapshot"


def test_writer_never_raises_and_logs_the_class_only():
    with patch("config.BOOK_DEPLOYMENT_RECORD_ENABLED", True), \
         patch("db.get_activity_shadow_keys", side_effect=RuntimeError("1000.0")), \
         patch("tasks.log") as lg:
        assert tasks.book_deployment_record("2026-11-20") is False
    assert "1000.0" not in str(lg.warning.call_args)


def test_an_unreadable_watchlist_still_writes_the_cash_fields():
    with patch("config.BOOK_DEPLOYMENT_RECORD_ENABLED", True), \
         patch("tasks._book_deployment_inputs", return_value=_inp()), \
         patch("knowledge_loader.load_watchlist", side_effect=OSError("gone")), \
         patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.log_activity_direct", return_value=True) as act:
        tasks.book_deployment_record("2026-11-20")
    rec = act.call_args.args[4]
    assert rec["watchlist_read"] == "error: OSError" and rec["sector_weights_pct"] is None
    assert rec["cash_pct"] == 81.0


def test_position_exit_frame_runs_it_after_gap_risk_on_both_paths():
    calls = []
    with patch("db.get_all_position_stops", return_value=[]), \
         patch("tasks._get_client") as client, \
         patch("tasks.book_gap_risk_record", side_effect=lambda d: calls.append("gap")), \
         patch("tasks.book_deployment_record", side_effect=lambda d: calls.append("deployment")):
        client.return_value.get_all_positions.return_value = []
        tasks.position_exit_frame()
    assert calls == ["gap", "deployment"]
    with patch("db.get_all_position_stops", side_effect=RuntimeError("x")), \
         patch("tasks.book_gap_risk_record", side_effect=lambda d: calls.append("gap")), \
         patch("tasks.book_deployment_record",
               side_effect=lambda d: calls.append("deployment") or (_ for _ in ()).throw(KeyError())):
        tasks.position_exit_frame()                      # no raise
    assert calls == ["gap", "deployment", "gap", "deployment"]


@pytest.mark.switches(BOOK_DEPLOYMENT_RECORD_ENABLED=False)
def test_switch_off_writes_nothing():
    with patch("db.log_activity_direct") as act, patch("tasks._book_deployment_inputs") as reads:
        assert tasks.book_deployment_record("2026-11-20") is False
    assert not act.called and not reads.called


# ── batch_verify ────────────────────────────────────────────────────────────

def _row(i, **over):
    with patch("config.BOOK_DEPLOYMENT_RECORD_ENABLED", True), \
         patch("tasks._book_deployment_inputs", return_value=_inp()), \
         patch("knowledge_loader.load_watchlist", return_value=WATCHLIST), \
         patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.log_activity_direct", return_value=True) as act:
        tasks.book_deployment_record("2026-11-20")
    extra = {**act.call_args.args[4], "env": "live", **over}
    return {"id": i, "stage": "book_deployment", "ticker": "", "extra": extra}


def _chk(rows):
    return {c["name"]: c for c in bv.check_activity(rows, "live", {})}["book_deployment"]


def test_batch_verify_ok_warn_and_fail():
    assert _chk([_row(1)])["status"] == "OK"
    assert _chk([])["status"] == "WARN"
    assert _chk([_row(1, read="error: TimeoutError")])["status"] == "FAIL"
    missing = _row(2)
    missing["extra"].pop("gap_pct")
    assert _chk([missing])["status"] == "FAIL"
    dollars = _row(3, cash=1.0)
    c = _chk([dollars])
    assert c["status"] == "FAIL" and any("dollar key" in f for f in c["extra_fail"])
