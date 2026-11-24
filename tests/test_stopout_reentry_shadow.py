"""Stop-out re-entry shadow (agents/crew.py:_stopout_reentry_shadow): one
record-only STOPOUT_REENTRY_SHADOW row for an OPEN/ADD that passed the rebuy
cooldown soon after an exit the bot did not place. Routing must not change."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import db
from agents import crew

# conftest stubs db.get_last_broker_exit for every test; the helper's own
# tests call the real one, taken at import (before fixtures apply).
_REAL_GET_LAST_BROKER_EXIT = db.get_last_broker_exit


def _result(action="OPEN", conviction=8):
    return {"ticker": "NVDA", "action": action, "conviction": conviction,
            "thesis_bucket": "Moonshot", "reasoning": "r", "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m"}}


def _exit(hours=17.5, outcomes=(-20.3, -12.0), convs=(7, 7)):
    return {"exit_at": "2026-11-19T20:56:32+00:00", "hours_since": hours, "lots": len(outcomes),
            "lot_outcome_pct": list(outcomes), "lot_alpha_pct": [-16.9, -8.0][:len(outcomes)],
            "lot_convictions": list(convs)}


def _rows(mock_log):
    return [c for c in mock_log.call_args_list if c[0][1] == "STOPOUT_REENTRY_SHADOW"]


class TestShadowRow:
    def test_no_exit_writes_nothing(self):
        with patch("db.get_last_broker_exit", return_value=None), patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result())
        ls.assert_not_called()

    @pytest.mark.parametrize("action", ["TRIM", "EXIT", "HOLD", "WATCH"])
    def test_non_buy_actions_are_not_read(self, action):
        with patch("db.get_last_broker_exit") as gx, patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result(action=action))
        gx.assert_not_called()
        ls.assert_not_called()

    def test_flag_off_skips_the_read(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", False)
        with patch("db.get_last_broker_exit") as gx, patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result())
        gx.assert_not_called()
        ls.assert_not_called()

    @pytest.mark.parametrize("hours,expect", [
        (17.5, {"1d": True, "3d": True, "5d": True, "10d": True}),
        (24.0, {"1d": False, "3d": True, "5d": True, "10d": True}),
        (125.6, {"1d": False, "3d": False, "5d": False, "10d": True}),
        (239.9, {"1d": False, "3d": False, "5d": False, "10d": True}),
    ])
    def test_would_block_per_window(self, hours, expect):
        with patch("db.get_last_broker_exit", return_value=_exit(hours=hours)), \
             patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result())
        (call,) = _rows(ls)
        args, kw = call
        assert args[0] == "NVDA" and args[2] == 0.0
        rec = kw["decision"]["stopout_reentry"]
        assert rec["would_block"] == expect
        assert rec["read"] == "ok"
        assert rec["worst_lot_outcome_pct"] == -20.3
        assert rec["hours_since"] == hours
        assert rec["lookback_days"] == 10
        assert kw["decision"]["shadow"] == "STOPOUT_REENTRY_SHADOW"
        # The provenance is a copy of the verdict's, so the copy marker stays.
        assert kw["decision"][db._VERDICT_COPY_KEY] == "STOPOUT_REENTRY_SHADOW"
        assert kw["decision"]["verdict_action"] == "OPEN"
        assert kw["model"] == "m"

    def test_lookback_matches_widest_window(self):
        with patch("db.get_last_broker_exit", return_value=None) as gx:
            crew._stopout_reentry_shadow("NVDA", _result())
        assert gx.call_args[0] == ("NVDA", 10)
        assert gx.call_args[1] == {"strict": True}

    @pytest.mark.parametrize("conv,above", [(8, True), (7, False), (6, False)])
    def test_conviction_against_the_stopped_lots(self, conv, above):
        with patch("db.get_last_broker_exit", return_value=_exit(convs=(7, 6))), \
             patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result(conviction=conv))
        assert _rows(ls)[0][1]["decision"]["stopout_reentry"]["conviction_above_lots"] is above

    def test_missing_lot_values_record_none(self):
        ex = _exit(outcomes=(None,), convs=(None,))
        with patch("db.get_last_broker_exit", return_value=ex), patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result())
        rec = _rows(ls)[0][1]["decision"]["stopout_reentry"]
        assert rec["worst_lot_outcome_pct"] is None
        assert rec["conviction_above_lots"] is None

    def test_read_error_records_the_error(self):
        with patch("db.get_last_broker_exit", side_effect=ConnectionError("down")), \
             patch("db.log_signal") as ls:
            crew._stopout_reentry_shadow("NVDA", _result())
        rec = _rows(ls)[0][1]["decision"]["stopout_reentry"]
        assert rec["read"] == "error: ConnectionError"
        assert rec["would_block"] is None

    def test_log_failure_never_raises(self):
        with patch("db.get_last_broker_exit", return_value=_exit()), \
             patch("db.log_signal", side_effect=RuntimeError("db down")):
            crew._stopout_reentry_shadow("NVDA", _result())   # no exception

    def test_kind_is_record_only_and_not_a_verdict(self):
        assert "STOPOUT_REENTRY_SHADOW" in db._RECORD_ONLY_SIGNALS
        assert "STOPOUT_REENTRY_SHADOW" in db._NON_ANALYST_SIGNALS


class TestRoutingUnchanged:
    """The buy reaches sizing the same way with the shadow writing a row, with
    its read failing, and with it off."""

    def _route(self, **patches):
        result = _result(action="OPEN", conviction=7)
        client = MagicMock()
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval", return_value=None), \
             patch("agents.crew._get_price", return_value=100.0), \
             patch("db.log_signal") as ls, \
             patch("risk_manager.RiskManager") as rm, \
             patch("db.get_last_broker_exit", **patches):
            rm.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
            crew._route_to_approval(result, {"regime": "Neutral"}, client)
        others = [c[0][1] for c in ls.call_args_list if c[0][1] != "STOPOUT_REENTRY_SHADOW"]
        return rm.return_value.size_position.call_count, others, len(_rows(ls))

    @pytest.mark.switches(STOPOUT_REENTRY_GATE_DAYS=0)   # the record-only shadow; the live gate is tested in test_entry_gate.py
    def test_same_route_with_row_error_and_none(self):
        base = self._route(return_value=None)
        with_row = self._route(return_value=_exit())
        with_err = self._route(side_effect=RuntimeError("x"))
        assert base[2] == 0 and with_row[2] == 1 and with_err[2] == 1
        assert base[:2] == with_row[:2] == with_err[:2]
        assert base[0] == 1

    def test_cooldown_skip_writes_no_shadow_row(self):
        result = _result(action="ADD", conviction=6)
        with patch("db.has_active_approval", return_value=False), \
             patch("db.get_last_executed_approval",
                   return_value={"conviction": 6, "decided_at": "2026-11-19T12:00:00Z"}), \
             patch("db.get_last_broker_exit", return_value=_exit()) as gx, \
             patch("db.log_signal") as ls, \
             patch("risk_manager.RiskManager"):
            crew._route_to_approval(result, {"regime": "Neutral"}, MagicMock())
        gx.assert_not_called()
        assert [c[0][1] for c in ls.call_args_list] == ["COOLDOWN_SKIPPED"]


class TestDbHelper:
    def _run(self, rows, since=(0,)):
        cur = MagicMock()
        cur.fetchall.return_value = rows
        cur.fetchone.return_value = since
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            return _REAL_GET_LAST_BROKER_EXIT("NVDA", 10, strict=True), cur

    def test_groups_lots_at_the_newest_exit(self):
        t1 = datetime(2026, 11, 19, 20, 56, 32, tzinfo=timezone.utc)
        t0 = datetime(2026, 11, 5, 14, 30, 3, tzinfo=timezone.utc)
        out, cur = self._run([(t1, 0.4, -0.203, -0.169, 7), (t1, 0.4, -0.12, None, 7),
                              (t0, 340.0, -0.093, -0.08, 6)], since=(1,))
        assert out == {"exit_at": t1.isoformat(), "hours_since": 0.4, "lots": 2,
                       "lot_outcome_pct": [-20.3, -12.0], "lot_alpha_pct": [-16.9, None],
                       "lot_convictions": [7, 7], "entries_since_exit": 1}
        (sql, params), (sql2, params2) = [c[0] for c in cur.execute.call_args_list]
        assert "exit_source='broker_exit'" in sql
        assert params[0] == "NVDA" and params[2] == 10
        # Buys counted from the newest exit, with the rebuy cooldown's statuses.
        assert "pending_approvals" in sql2 and "'APPROVED','FAILED'" in sql2
        assert params2[0] == "NVDA" and params2[2] == t1

    def test_no_rows_skips_the_entries_count(self):
        out, cur = self._run([])
        assert out is None and cur.execute.call_count == 1

    def test_strict_raises_and_lenient_returns_none(self):
        conn = MagicMock()
        conn.cursor.side_effect = RuntimeError("boom")
        with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
            with pytest.raises(RuntimeError):
                _REAL_GET_LAST_BROKER_EXIT("NVDA", 10, strict=True)
            assert _REAL_GET_LAST_BROKER_EXIT("NVDA", 10) is None
        assert put.call_count == 2
